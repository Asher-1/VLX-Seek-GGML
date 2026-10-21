#!/usr/bin/env python3
"""Convert the VLX-Seek main vision tower (Qwen3.5-VL native) to GGUF.

The 10B checkpoint contains the main tower as `model.vision_tower.visual.*`
(Megatron-style names: mlp.linear_fc1/fc2) INCLUDING its native merger
(spatial-merge + out projection -> LLM hidden 4096). There is NO separate
CAbstractor in the released weights: the main tower emits 4096-d tokens
directly, so the P1 "mmproj segment" is a single ViT-family graph — same
per-block segment architecture as RADIO.

Weight inventory (verified from model.safetensors):
  patch_embed.proj.weight (1152, 3, 2, 16, 16)  # 5D: cin=3, kt=2, kh=kw=16
  patch_embed.proj.bias   (1152,)
  pos_embed.weight        (2304, 1152)          # 48x48 grid
  blocks.{i}.norm1/.norm2, attn.qkv/proj, mlp.linear_fc1/fc2   (27 blocks)
  merger.norm, merger.linear_fc1 (4608, 4608), merger.linear_fc2 (4096, 4608)

Usage:
  python3 convert_mmproj_to_gguf.py --model-path resources/VLX-Seek-1.5-10B \
      --dtype f32 --output cpp_ggml/models/gguf/vlx-mmproj-10b-f32.gguf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def load_tensors(path: Path, prefix: str) -> dict[str, np.ndarray]:
    # Main-tower keys only (the full checkpoint at f32 is ~40 GB).
    out: dict[str, np.ndarray] = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                if k.startswith(prefix):
                    out[k] = sf.get_tensor(k).float().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f16")
    ap.add_argument("--list-keys", action="store_true")
    args = ap.parse_args()

    try:
        from gguf import GGUFWriter, GGMLQuantizationType
        from gguf.quants import quantize
    except ImportError:
        sys.exit("pip install gguf is required for conversion")

    model_path = Path(args.model_path)
    prefix = "model.vision_tower.visual"
    tensors = load_tensors(model_path, prefix + ".")
    cfg_path = model_path / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    vcfg = cfg.get("vision_config") or {}

    tower_keys = [k for k in tensors if k.startswith(prefix + ".")]
    if not tower_keys:
        sys.exit("no main-tower tensors found")
    if args.list_keys:
        for k in sorted(tower_keys):
            print(k, tuple(tensors[k].shape))
        return

    blocks = sorted({int(k[len(prefix) + 8:].split(".")[0])
                     for k in tower_keys if f"{prefix}.blocks." in k})
    depth = len(blocks)
    patch_w = tensors[f"{prefix}.patch_embed.proj.weight"]
    kt, kh, kw = patch_w.shape[2], patch_w.shape[3], patch_w.shape[4]
    dim = patch_w.shape[0]
    pos_tokens = tensors[f"{prefix}.pos_embed.weight"].shape[0]
    pos_side = int(round(pos_tokens ** 0.5))
    hidden_out = int(vcfg.get("out_hidden_size", 4096))
    merge = int(vcfg.get("spatial_merge_size", 2))
    print(f"main tower: patch kt{kt} {kh}x{kw} dim{dim} depth{depth} "
          f"pos{pos_side}x{pos_side} out{hidden_out} merge{merge}")

    writer = GGUFWriter(args.output, "vlxmmproj")
    writer.add_string("general.name", "vlx-seek-mmproj-main")
    writer.add_string("general.architecture", "vlxmmproj")
    writer.add_uint32("vlx.mmproj.patch_size", kh)
    writer.add_uint32("vlx.mmproj.temporal", kt)
    writer.add_uint32("vlx.mmproj.embed_dim", dim)
    writer.add_uint32("vlx.mmproj.depth", depth)
    writer.add_uint32("vlx.mmproj.hidden_out", hidden_out)
    writer.add_uint32("vlx.mmproj.merge", merge)
    writer.add_uint32("vlx.mmproj.pos_side", pos_side)
    heads = int(vcfg.get("num_attention_heads", 16))
    writer.add_uint32("vlx.mmproj.num_heads", heads)
    mlp = int(tensors[f"{prefix}.blocks.{blocks[0]}.mlp.linear_fc1.weight"].shape[0])
    writer.add_uint32("vlx.mmproj.mlp_hidden", mlp)
    # The 10B checkpoint's vision_config has rope_parameters=None; HF's
    # Qwen3_5VisionConfig fills the generic default theta=10000 (verified via
    # config.rope_parameters on the loaded reference model) — NOT 1e6.
    theta = float((vcfg.get("rope_parameters") or {}).get("rope_theta", 10000.0))
    writer.add_float32("vlx.mmproj.rope_theta", theta)

    def add(name: str, arr: np.ndarray, big: bool = False) -> None:
        # bf16 checkpoint -> f16 is lossless; q8_0 needs ne[0] % 32 == 0.
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if big and arr.ndim == 2 and args.dtype != "f32":
            if args.dtype == "q8_0" and arr.shape[1] % 32 == 0:
                writer.add_tensor(name, quantize(arr, GGMLQuantizationType.Q8_0),
                                  raw_dtype=GGMLQuantizationType.Q8_0)
                return
            writer.add_tensor(name, arr.astype(np.float16),
                              raw_dtype=GGMLQuantizationType.F16)
            return
        writer.add_tensor(name, arr)

    # Patch conv flattened to GEMM form (M=dim, K=ic*kt*kh*kw): the C++ graph
    # consumes the patch sequence as (K, N) and does mul_mat directly; the
    # numpy C-order flatten matches the patch byte layout of dump_torch_mmproj
    # (k = ic*512 + it*256 + iy*16 + ix).
    add("mmproj.patch_embed.weight", patch_w.reshape(dim, -1), big=True)
    add("mmproj.patch_embed.bias", tensors[f"{prefix}.patch_embed.proj.bias"])
    add("mmproj.pos_embed", tensors[f"{prefix}.pos_embed.weight"])

    for i in blocks:
        p = f"{prefix}.blocks.{i}"
        o = f"mmproj.blocks.{i}"
        add(f"{o}.norm1.weight", tensors[f"{p}.norm1.weight"])
        add(f"{o}.norm1.bias", tensors[f"{p}.norm1.bias"])
        add(f"{o}.norm2.weight", tensors[f"{p}.norm2.weight"])
        add(f"{o}.norm2.bias", tensors[f"{p}.norm2.bias"])
        add(f"{o}.attn.qkv.weight", tensors[f"{p}.attn.qkv.weight"], big=True)
        add(f"{o}.attn.qkv.bias", tensors[f"{p}.attn.qkv.bias"])
        add(f"{o}.attn.proj.weight", tensors[f"{p}.attn.proj.weight"], big=True)
        add(f"{o}.attn.proj.bias", tensors[f"{p}.attn.proj.bias"])
        add(f"{o}.mlp.fc1.weight", tensors[f"{p}.mlp.linear_fc1.weight"], big=True)
        add(f"{o}.mlp.fc1.bias", tensors[f"{p}.mlp.linear_fc1.bias"])
        add(f"{o}.mlp.fc2.weight", tensors[f"{p}.mlp.linear_fc2.weight"], big=True)
        add(f"{o}.mlp.fc2.bias", tensors[f"{p}.mlp.linear_fc2.bias"])
        for extra in ("attn.q_norm.weight", "attn.q_norm.bias",
                      "attn.k_norm.weight", "attn.k_norm.bias"):
            if f"{p}.{extra}" in tensors:
                add(f"{o}.{extra}", tensors[f"{p}.{extra}"])

    add("mmproj.merger.norm.weight", tensors[f"{prefix}.merger.norm.weight"])
    add("mmproj.merger.norm.bias", tensors[f"{prefix}.merger.norm.bias"])
    add("mmproj.merger.fc1.weight", tensors[f"{prefix}.merger.linear_fc1.weight"], big=True)
    add("mmproj.merger.fc1.bias", tensors[f"{prefix}.merger.linear_fc1.bias"])
    add("mmproj.merger.fc2.weight", tensors[f"{prefix}.merger.linear_fc2.weight"], big=True)
    add("mmproj.merger.fc2.bias", tensors[f"{prefix}.merger.linear_fc2.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
