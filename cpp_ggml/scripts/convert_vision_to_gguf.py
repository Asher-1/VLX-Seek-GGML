#!/usr/bin/env python3
"""Convert the VLX-Seek C-RADIOv4 auxiliary tower (+ HFRE projector) to GGUF.

Reads the HuggingFace VLX-Seek-1.5-10B checkpoint (safetensors) and emits the
vision-sub-chain GGUF consumed by radio-cli. Tensor discovery is fuzzy-keyed:
checkpoint prefixes differ between releases, so we match on the stable suffix
segments (patch_generator, blocks, object_vp_extractor, ...) and report
everything matched.

Usage:
  python3 convert_vision_to_gguf.py --model-path resources/VLX-Seek-1.5-10B \
      --dtype f16 --output cpp_ggml/models/gguf/vlx-radio-f16.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    # bf16 checkpoints need the torch framework; everything is cast to f32.
    # Only the aux tower / HFRE keys are materialized: the full 10B checkpoint
    # at f32 would be ~40 GB of host RAM.
    import torch
    from safetensors import safe_open

    tensors: dict[str, np.ndarray] = {}
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors under {path}")
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                if "vision_tower_aux" not in key and "object_vp_extractor" not in key \
                        and "cls_token" not in key and "patch_generator" not in key:
                    continue
                tensors[key] = sf.get_tensor(key).float().numpy()
    return tensors


def pick(tensors: dict[str, np.ndarray], *segments: str) -> tuple[str, np.ndarray]:
    """Find the unique tensor whose key contains every segment, in order."""
    hits = [
        (k, v)
        for k, v in tensors.items()
        if all(s in k for s in segments)
    ]
    if not hits:
        raise KeyError("no tensor matching " + " + ".join(segments))
    if len(hits) > 1:
        # Prefer the deepest (most specific) key; ties error out.
        hits.sort(key=lambda kv: kv[0].count("."))
        if hits[0][0].count(".") == hits[1][0].count("."):
            names = sorted(k for k, _ in hits)[:6]
            raise KeyError(f"ambiguous tensor for {' + '.join(segments)}: {names}")
    return hits[0]


def num_registers(tensors: dict[str, np.ndarray]) -> int:
    _, t = pick(tensors, "cls_token", "token")
    return int(t.shape[0]) - 1  # token = (num_cls_tokens + num_registers, C)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f16")
    ap.add_argument("--list-radio-keys", action="store_true",
                    help="print matched RADIO/HFRE tensor keys and exit")
    ap.add_argument("--num-heads", type=int, default=0,
                    help="override attention heads (0 = auto: 16 for dim1152, "
                         "else dim//72; synth models need this)")
    args = ap.parse_args()

    try:
        from gguf import GGUFWriter, GGMLQuantizationType
        from gguf.quants import quantize
    except ImportError:
        sys.exit("pip install gguf is required for conversion")

    model_path = Path(args.model_path)
    tensors = load_safetensors(model_path)

    # ---- discover the RADIO tensor family ----
    radio_keys = {
        k: v
        for k, v in tensors.items()
        if "cls_token" in k
        or "patch_generator" in k
        or ("vision_tower_aux" in k and ".blocks." in k)
    }
    if not radio_keys:
        sys.exit("no RADIO tensors found; is this a VLX-Seek checkpoint?")
    if args.list_radio_keys:
        for k in sorted(radio_keys):
            print(k, tuple(radio_keys[k].shape))
        return

    w_patch_name, w_patch = pick(tensors, "patch_generator", "embedder", "weight")
    pos_name, pos_embed = pick(tensors, "patch_generator", "pos_embed")
    cls_name, cls_tok = pick(tensors, "cls_token", "token")
    n_skip, embed_dim = cls_tok.shape
    patch_size = int(round((w_patch.shape[1] / 3) ** 0.5))
    pos_grid = int(round(pos_embed.shape[1] ** 0.5))

    # Config-derived facts (10B checkpoint layout).
    cfg_path = model_path / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    aux_args = (cfg.get("vision_aux_config") or {}).get("args") or {}
    num_cls = int(aux_args.get("num_cls_tokens", 1))
    reg_mult = aux_args.get("register_multiple")
    reg_explicit = aux_args.get("cpe_num_registers")
    if reg_explicit:
        num_reg = int(reg_explicit)
    elif reg_mult:
        num_reg = int(reg_mult) - (num_cls % int(reg_mult))
    else:
        num_reg = n_skip - num_cls
    if n_skip != num_cls + num_reg:
        print(f"WARNING: cls_token rows {n_skip} != {num_cls} cls + {num_reg} reg",
              file=sys.stderr)
        num_cls, num_reg = 1, n_skip - 1

    # Block tensors via the common block prefix.
    block_prefix = next(
        k[: k.rindex(".norm1.weight")]
        for k in radio_keys
        if ".blocks." in k and k.endswith(".norm1.weight")
    ).split(".blocks.")[0]  # strip the trailing ".blocks.{i}" segment
    blocks = sorted(
        {int(k.split(".blocks.")[1].split(".")[0]) for k in radio_keys if ".blocks." in k}
    )
    depth = len(blocks)
    mlp_hidden = int(pick(tensors, block_prefix, f".blocks.{blocks[0]}", "mlp.fc1.weight")[1].shape[0])
    num_heads = embed_dim // 72 if embed_dim % 72 == 0 else 16  # SO400M head_dim=72
    num_heads = 16 if embed_dim == 1152 else num_heads
    if args.num_heads:
        num_heads = args.num_heads

    print(
        f"radio: patch{patch_size} dim{embed_dim} depth{depth} "
        f"heads{num_heads} mlp{mlp_hidden} pos{pos_grid}x{pos_grid} skip{n_skip} "
        f"(from {w_patch_name})"
    )

    # ---- HFRE vt projector (Linear-GELU-Linear) ----
    vt_w0 = vt_b0 = vt_w1 = vt_b1 = None
    for k, v in tensors.items():
        if "object_vp_extractor" in k and "vision_tower_object_feature_projector" in k:
            if k.endswith("0.weight"):
                vt_w0 = (k, v)
            elif k.endswith("0.bias"):
                vt_b0 = (k, v)
            elif k.endswith("2.weight"):
                vt_w1 = (k, v)
            elif k.endswith("2.bias"):
                vt_b1 = (k, v)

    writer = GGUFWriter(args.output, "vlxvision")
    writer.add_string("general.name", "vlx-seek-radio-vision")
    writer.add_string("general.architecture", "vlxvision")

    writer.add_uint32("vlx.radio.patch_size", patch_size)
    writer.add_uint32("vlx.radio.embed_dim", embed_dim)
    writer.add_uint32("vlx.radio.depth", depth)
    writer.add_uint32("vlx.radio.num_heads", num_heads)
    writer.add_uint32("vlx.radio.mlp_hidden", mlp_hidden)
    writer.add_uint32("vlx.radio.num_cls_tokens", num_cls)
    writer.add_uint32("vlx.radio.num_registers", num_reg)
    writer.add_uint32("vlx.radio.pos_rows", pos_grid)
    writer.add_uint32("vlx.radio.pos_cols", pos_grid)

    def add(name: str, arr: np.ndarray, big: bool = False) -> None:
        # Big projections ride --dtype; small params stay f32. The checkpoint
        # is bf16, so f16 is lossless (bf16 values below 2^-14 land in the f16
        # subnormal range, abs error <= 2^-25) and f32 carries no extra bits.
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if big and arr.ndim == 2 and args.dtype != "f32":
            if args.dtype == "q8_0" and arr.shape[1] % 32 == 0:
                writer.add_tensor(name, quantize(arr, GGMLQuantizationType.Q8_0),
                                  raw_dtype=GGMLQuantizationType.Q8_0)
                return
            writer.add_tensor(name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            return
        writer.add_tensor(name, arr)

    # RADIO input conditioner: the official forward normalizes with these
    # checkpoint buffers (bf16 CLIP-like values), not the ImageNet constants.
    _, cond_mean = pick(tensors, "input_conditioner", "norm_mean")
    _, cond_std = pick(tensors, "input_conditioner", "norm_std")
    add("radio.input_conditioner.norm_mean", cond_mean.reshape(-1))
    add("radio.input_conditioner.norm_std", cond_std.reshape(-1))

    add("radio.patch_embed.weight", w_patch, big=True)
    add("radio.pos_embed", pos_embed.reshape(pos_grid * pos_grid, embed_dim))
    add("radio.cls_token", cls_tok)
    for i in blocks:
        p = f"{block_prefix}.blocks.{i}"
        o = f"radio.blocks.{i}"
        add(f"{o}.norm1.weight", tensors[f"{p}.norm1.weight"])
        add(f"{o}.norm1.bias", tensors[f"{p}.norm1.bias"])
        add(f"{o}.norm2.weight", tensors[f"{p}.norm2.weight"])
        add(f"{o}.norm2.bias", tensors[f"{p}.norm2.bias"])
        add(f"{o}.attn.qkv.weight", tensors[f"{p}.attn.qkv.weight"], big=True)
        add(f"{o}.attn.qkv.bias", tensors[f"{p}.attn.qkv.bias"])
        add(f"{o}.attn.proj.weight", tensors[f"{p}.attn.proj.weight"], big=True)
        add(f"{o}.attn.proj.bias", tensors[f"{p}.attn.proj.bias"])
        for ls in ("ls1", "ls2"):
            key = f"{p}.{ls}.gamma"
            if key in tensors:
                add(f"{o}.{ls}.gamma", tensors[key])
        add(f"{o}.mlp.fc1.weight", tensors[f"{p}.mlp.fc1.weight"], big=True)
        add(f"{o}.mlp.fc1.bias", tensors[f"{p}.mlp.fc1.bias"])
        add(f"{o}.mlp.fc2.weight", tensors[f"{p}.mlp.fc2.weight"], big=True)
        add(f"{o}.mlp.fc2.bias", tensors[f"{p}.mlp.fc2.bias"])

    if vt_w0:
        # 10B uses combination="concat" without a vt projector; the tensor only
        # exists in mean-fusion configs. Record the real HFRE facts from config.
        writer.add_uint32("vlx.hfre.roi_output_size", int(cfg.get("mm_roi_output_size", 7)))
        writer.add_uint32("vlx.hfre.roi_aligned", int(bool(cfg.get("mm_roi_algined", False))))
        writer.add_uint32("vlx.hfre.add_pos_embedding", int(bool(cfg.get("mm_add_pos_embed", True))))
        writer.add_uint32("vlx.hfre.pos_embedding_dim", int(cfg.get("mm_object_hidden_size", vt_w0[1].shape[0])))
        writer.add_string("vlx.hfre.combination", str(cfg.get("mm_object_feature_combination", "mean")))
        vcfg = cfg.get("vision_config") or {}
        vt_stride = int(vcfg.get("patch_size", 16)) * int(vcfg.get("spatial_merge_size", 2))
        writer.add_float32("vlx.hfre.aux_spatial_scale", 1.0 / patch_size)
        writer.add_float32("vlx.hfre.vt_spatial_scale", 1.0 / vt_stride)
        writer.add_uint32("vlx.hfre.use_simple_fpn_vt", int(bool(cfg.get("mm_use_simpleFPN_for_vt", False))))
        writer.add_uint32("vlx.hfre.simple_fpn_out_channels", int(cfg.get("mm_simpleFPN_out_channels_for_vt", 512)))
        add("hfre.vt_proj.0.weight", vt_w0[1], big=True)
        if vt_b0:
            add("hfre.vt_proj.0.bias", vt_b0[1])
        if vt_w1:
            add("hfre.vt_proj.2.weight", vt_w1[1], big=True)
        if vt_b1:
            add("hfre.vt_proj.2.bias", vt_b1[1])
        print(f"hfre vt projector: {tuple(vt_w0[1].shape)} -> {tuple(vt_w1[1].shape)}")
    else:
        # concat path (10B): still record HFRE facts for the C++ side.
        writer.add_uint32("vlx.hfre.roi_output_size", int(cfg.get("mm_roi_output_size", 7)))
        writer.add_uint32("vlx.hfre.roi_aligned", int(bool(cfg.get("mm_roi_algined", False))))
        writer.add_uint32("vlx.hfre.add_pos_embedding", int(bool(cfg.get("mm_add_pos_embed", True))))
        writer.add_uint32("vlx.hfre.pos_embedding_dim", int(cfg.get("mm_object_hidden_size", 6656)))
        writer.add_string("vlx.hfre.combination", str(cfg.get("mm_object_feature_combination", "concat")))
        vcfg = cfg.get("vision_config") or {}
        vt_stride = int(vcfg.get("patch_size", 16)) * int(vcfg.get("spatial_merge_size", 2))
        writer.add_float32("vlx.hfre.aux_spatial_scale", 1.0 / patch_size)
        writer.add_float32("vlx.hfre.vt_spatial_scale", 1.0 / vt_stride)
        writer.add_uint32("vlx.hfre.use_simple_fpn_vt", int(bool(cfg.get("mm_use_simpleFPN_for_vt", False))))
        writer.add_uint32("vlx.hfre.simple_fpn_out_channels", int(cfg.get("mm_simpleFPN_out_channels_for_vt", 512)))
        print("hfre: concat path (no vt projector tensor)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
