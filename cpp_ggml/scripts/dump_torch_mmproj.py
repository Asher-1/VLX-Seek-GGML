#!/usr/bin/env python3
"""Torch reference dump for the Qwen3.5-VL main tower (mmproj parity).

Authoritative reference = HF Qwen3_5VisionModel itself (eager attention, f32),
with weights injected from the VLX-Seek 10B checkpoint
(`model.vision_tower.visual.*` -> `visual.*`). The pixel patch sequence is
extracted by hand in spatial-merge-block order — verified against
Qwen2VLImageProcessor on the same image before dumping.

Outputs (into --out):
  input.bin            : f32 (N, 3*2*16*16) patch bytes, block-major, normalized
  torch_mm_blk{i}.bin  : f32 (N, 1152) block i output (token-major), i in --blocks
  torch_mm_final.bin   : f32 (N, 1152) blocks.26 output = merger input (HFRE tap)
  torch_mm_pooler.bin  : f32 (N/4, 4096) merged LLM tokens
  torch_mm_pos.bin     : f32 (N, 1152) interpolated pos embeds (host-side check)
  torch_mm_cossin.bin  : f32 (N, 144) [cos(72) | sin(72)] (host-side check)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_visual(model_path: Path):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from safetensors import safe_open

    cfg_dict = json.loads((model_path / "config.json").read_text())["vision_config"]
    config = Qwen3_5VisionConfig(**cfg_dict)
    model = Qwen3_5VisionModel._from_config(config, attn_implementation="eager")
    model = model.float()

    sd = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("model.vision_tower.visual."):
                sd[k[len("model.vision_tower.visual."):]] = sf.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if "inv_freq" not in k]
    print(f"loaded {len(sd)} tensors; missing={real_missing} unexpected={list(unexpected)}")
    if real_missing:
        raise RuntimeError(f"missing weights: {real_missing[:6]}")
    model.eval()
    return model, config


def extract_patches(img_f32: np.ndarray, side: int, patch: int = 16, merge: int = 2):
    """Block-major patch extraction: (block_row, block_col, in_row, in_col).

    Returns (N, 3, 2, 16, 16) with the temporal axis duplicated (static image).
    """
    H, W, _ = img_f32.shape
    assert H == W == side * patch
    ph, pw = side // merge, side // merge
    patches = []
    for br in range(ph):
        for bc in range(pw):
            for ir in range(merge):
                for ic in range(merge):
                    y = (br * merge + ir) * patch
                    x = (bc * merge + ic) * patch
                    p = img_f32[y:y + patch, x:x + patch, :]        # (16,16,3)
                    p = p.transpose(2, 0, 1)                        # (3,16,16)
                    patches.append(np.stack([p, p], axis=1))        # (3,2,16,16)
    return np.stack(patches).astype(np.float32)                     # (N,3,2,16,16)


def simplefpn_forward(pooled_nchw: torch.Tensor, sd: dict) -> list:
    """Hand-written SimpleFP forward (simple_fpn.py, norm="LN", use_bias=False
    for the 1x1/3x3 convs; nn.GELU() = exact-erf). Levels ordered as the
    python output (finest, stride 4, first)."""
    import torch.nn.functional as F

    def ln(x, w, b, eps=1e-6):  # channels-first LayerNorm over C
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return (x - u) / torch.sqrt(s + eps) * w[None, :, None, None] + b[None, :, None, None]

    def convt(x, p):
        return F.conv_transpose2d(x, sd[p + ".weight"], sd[p + ".bias"], stride=2)

    def conv(x, p):
        # both trailing convs: 1x1 (padding 0) and 3x3 (padding 1), stride 1
        pad = 1 if sd[p + ".weight"].shape[2] == 3 else 0
        return F.conv2d(x, sd[p + ".weight"], padding=pad)

    def stage(x, s, n_convt):
        # Sequential indices: ConvT at 3i, with LN(3i+1)+GELU(3i+2) only
        # BETWEEN ConvTs; the trailing 1x1/3x3 pair sits at 3n-2 / 3n-1.
        for i in range(n_convt):
            x = convt(x, f"simfp_{s}.{3 * i}")
            if i < n_convt - 1:
                x = ln(x, sd[f"simfp_{s}.{3 * i + 1}.weight"],
                       sd[f"simfp_{s}.{3 * i + 1}.bias"])
                x = F.gelu(x)
        j = 3 * n_convt - 2 if n_convt else 0
        x = conv(x, f"simfp_{s}.{j}")          # 1x1, no bias
        x = ln(x, sd[f"simfp_{s}.{j}.norm.weight"], sd[f"simfp_{s}.{j}.norm.bias"])
        x = conv(x, f"simfp_{s}.{j + 1}")      # 3x3 pad1, no bias
        x = ln(x, sd[f"simfp_{s}.{j + 1}.norm.weight"], sd[f"simfp_{s}.{j + 1}.norm.bias"])
        return x

    return [stage(pooled_nchw, s, n) for s, n in ((2, 3), (3, 2), (4, 1), (5, 0))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    ap.add_argument("--image", default="demo/demo_image.jpg")
    ap.add_argument("--out", default="/tmp/vlx-mmproj")
    ap.add_argument("--side", type=int, default=1024, help="resized square side (16-multiple)")
    ap.add_argument("--blocks", default="0,1,25,26", help="per-block dump indices")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blocks = [int(b) for b in args.blocks.split(",")]

    # Authoritative input = Qwen2VLImageProcessor output (the real pipeline's
    # bytes). smart_resize picks the side (factor 32); we record the grid.
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor(patch_size=16, merge_size=2, min_pixels=args.side * args.side, max_pixels=args.side * args.side)
    pil = Image.open(args.image).convert("RGB")
    enc = proc(images=pil, return_tensors="pt")
    pv = enc["pixel_values"].float()                       # (N, 3*2*16*16)
    t_n, gh, gw = enc["image_grid_thw"][0].tolist()
    N = t_n * gh * gw
    assert pv.shape[0] == N, (pv.shape, gthw if False else enc["image_grid_thw"])
    pv.numpy().astype(np.float32).tofile(out / "input.bin")
    print(f"patches: N={N} grid=({gh},{gw}) side={gh*16}")

    model, config = load_visual(Path(args.model_path))

    pixel_values = pv
    grid_thw = enc["image_grid_thw"]

    with torch.no_grad():
        blk0 = model.blocks[0]
        # Hook blocks for per-block taps.
        taps: dict[int, torch.Tensor] = {}
        model.patch_embed.register_forward_hook(
            lambda m, i, o: taps.__setitem__('patch', o.detach().clone()))
        for bi in blocks:
            model.blocks[bi].register_forward_hook(
                lambda m, i, o, bi=bi: taps.__setitem__(bi, o.detach().clone()))
        # block-0 interior references (diag bisection): attnbr/mlpbr/x
        from transformers.vision_utils import (get_vision_position_ids,
                                               get_vision_interpolation_indices_and_weights)
        from transformers.models.qwen3_5.modeling_qwen3_5 import rotate_half
        idx2, w2 = get_vision_interpolation_indices_and_weights(
            grid_thw, num_grid_per_side=model.num_grid_per_side, mode="bilinear",
            align_corners=True, spatial_merge_size=config.spatial_merge_size)
        pe = (model.pos_embed.weight[idx2] * w2[:, :, None]).sum(1)
        h = model.patch_embed(pixel_values).view(N, -1) + pe
        pids0 = get_vision_position_ids(grid_thw, config.spatial_merge_size)
        inv0 = model.rotary_pos_emb.inv_freq.float()
        fr = pids0[..., None].float() * inv0
        c0, s0 = fr.cos(), fr.sin()
        fh, fw = c0[:, 0], c0[:, 1]
        cosf = torch.cat([torch.cat([fh, fw], -1), torch.cat([fh, fw], -1)], -1)
        sh, sw = s0[:, 0], s0[:, 1]
        sinf = torch.cat([torch.cat([sh, sw], -1), torch.cat([sh, sw], -1)], -1)
        # n1 / qkv / q_rot references as well (single-source seg0 math)
        n1 = blk0.norm1(h)
        n1.contiguous().numpy().tofile(out / "torch_diag_n1.bin")
        qkv0 = blk0.attn.qkv(n1)
        qkv0.contiguous().numpy().tofile(out / "torch_diag_qkv.bin")
        q0 = qkv0.view(N, 3, config.num_heads, -1).permute(1, 0, 2, 3)[0]
        q_rot = (q0.float() * cosf.unsqueeze(-2)) + (rotate_half(q0.float()) * sinf.unsqueeze(-2))
        q_rot.contiguous().numpy().tofile(out / "torch_diag_q_rot.bin")

        attn_in = n1
        attn_out = blk0.attn(attn_in, cu_seqlens=torch.tensor([0, N]),
                             position_embeddings=(cosf, sinf))
        attn_out.contiguous().numpy().tofile(out / "torch_diag_attnbr.bin")
        h1 = h + attn_out
        mlp_out = blk0.mlp(blk0.norm2(h1))
        mlp_out.contiguous().numpy().tofile(out / "torch_diag_mlpbr.bin")
        (h1 + mlp_out).contiguous().numpy().tofile(out / "torch_diag_x.bin")

        outv = model(pixel_values, grid_thw=grid_thw, return_dict=True)

    taps['patch'].reshape(N, -1).contiguous().numpy().astype(np.float32).tofile(
        out / "torch_mm_patch.bin")
    for bi, t in taps.items():
        if bi == 'patch':
            continue
        t.reshape(N, -1).contiguous().numpy().astype(np.float32).tofile(out / f"torch_mm_blk{bi}.bin")
    outv.last_hidden_state.reshape(N, -1).numpy().astype(np.float32).tofile(out / "torch_mm_final.bin")
    outv.pooler_output.numpy().astype(np.float32).tofile(out / "torch_mm_pooler.bin")

    # Host-side-checkable intermediates: pos embeds + cos/sin.
    pos_ids = None
    from transformers.vision_utils import get_vision_position_ids, get_vision_interpolation_indices_and_weights
    idx, w = get_vision_interpolation_indices_and_weights(
        grid_thw, num_grid_per_side=model.num_grid_per_side, mode="bilinear",
        align_corners=True, spatial_merge_size=config.spatial_merge_size)
    pos = (model.pos_embed.weight[idx] * w[:, :, None]).sum(1).detach()
    pos.numpy().astype(np.float32).tofile(out / "torch_mm_pos.bin")

    pids = get_vision_position_ids(grid_thw, config.spatial_merge_size)   # (N, 2)
    inv = model.rotary_pos_emb.inv_freq.float()                            # (18,)
    freqs = pids[..., None].float() * inv                                  # (N,2,18)
    cos = freqs.cos()
    sin = freqs.sin()
    fh, fw = cos[:, 0], cos[:, 1]
    c = torch.cat([torch.cat([fh, fw], -1), torch.cat([fh, fw], -1)], -1)  # (N,72)
    sh, sw = sin[:, 0], sin[:, 1]
    s = torch.cat([torch.cat([sh, sw], -1), torch.cat([sh, sw], -1)], -1)
    torch.cat([c, s], -1).numpy().astype(np.float32).tofile(out / "torch_mm_cossin.bin")

    import json as _json
    (out / "meta.json").write_text(_json.dumps({"grid": [t_n, gh, gw]}))
    print(f"theta={config.rope_parameters['rope_theta']}")
    print(f"dumped: input(N,{3*2*16*16}) blocks={blocks} final pooler pos cossin")

    # ---- HFRE vt branch: SimpleFPN over the pooled (merged) feature map ----
    # Input = pooler_output tokens (N/4, 4096) reshaped to (1, 4096, gh/2, gw/2)
    # (block-major token order -> row-major merged grid, qwen3_5_vl_encoder
    # get_multi_level_features). The FPN weights come straight from the
    # checkpoint (object_vp_extractor.simple_fpn.*); no HF class involved.
    from safetensors import safe_open
    fpn_sd = {}
    with safe_open(str(Path(args.model_path) / "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("model.object_vp_extractor.simple_fpn.simfp_"):
                fpn_sd[k.split("simple_fpn.")[1]] = sf.get_tensor(k).float()
    H2, W2 = gh // 2, gw // 2
    pooled = outv.pooler_output.reshape(1, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    pooled.numpy().astype(np.float32).tofile(out / "torch_fpn_in.bin")
    levels = simplefpn_forward(pooled, fpn_sd)
    strides = [4, 8, 16, 32]
    for li, lv in enumerate(levels):
        lv.contiguous().numpy().astype(np.float32).tofile(out / f"torch_fpn_L{li}.bin")
        print(f"fpn L{li}: stride {strides[li]} {tuple(lv.shape)}")

    # vt pooled object features for a fixed box set (original-image pixels;
    # the same boxes.json the C++ CLI consumes).
    import torchvision
    W_img, H_img = pil.size
    boxes = torch.tensor([
        [0.0, 0.0, W_img, H_img],
        [W_img / 3, H_img / 3, 2 * W_img / 3, 2 * H_img / 3],
        [W_img / 2, H_img / 2, W_img - 1, H_img - 1],
    ], dtype=torch.float32)
    (out / "boxes.json").write_text(_json.dumps(boxes.tolist()))
    roi_sz, aligned = 14, True
    feats = []
    for li, lv in enumerate(levels):
        roi = torchvision.ops.roi_align(lv.float(), [boxes], output_size=roi_sz,
                                        spatial_scale=1.0 / strides[li],
                                        aligned=aligned)
        feats.append(roi.mean(dim=(2, 3)))
    vt = torch.cat(feats, dim=1)  # (N, 4*512 = 2048)
    vt.contiguous().numpy().astype(np.float32).tofile(out / "torch_fpn_vt.bin")
    print(f"fpn vt pooled: {tuple(vt.shape)} -> torch_fpn_vt.bin (boxes.json)")


if __name__ == "__main__":
    main()
