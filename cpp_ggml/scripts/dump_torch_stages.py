#!/usr/bin/env python3
"""Dump torch reference tensors for RADIO parity.

Self-contained: loads RADIO weights straight from the checkpoint safetensors
and runs a hand-written torch forward of c_radio_v4 (conditioner -> patch
embed -> CPE pos -> cls/registers -> blocks -> 4 intermediates). No
transformers model registration is needed, so this works on any torch build.

Outputs (into --out):
  input.bin          : f32 HWC [0,1] tensor, the exact C++ input
  torch_radio_L*.bin : f32 NCHW intermediate levels 0..3
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CLIP_MEAN = (0.485, 0.456, 0.406)
CLIP_STD = (0.229, 0.224, 0.225)


def load_tensors(path: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as sf:
            for k in sf.keys():
                out[k] = sf.get_tensor(k).float()
    return out


def pick(t: dict[str, torch.Tensor], *segs: str) -> torch.Tensor:
    hits = [v for k, v in t.items() if all(s in k for s in segs)]
    if not hits:
        raise KeyError(" + ".join(segs))
    hits.sort(key=lambda x: x.dim())
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--image", required=True, help="jpg/png input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--side", type=int, default=1024, help="aux input size (16-multiple)")
    ap.add_argument("--num-heads", type=int, default=16,
                    help="attention heads (SO400M: 16; synth model: 4)")
    ap.add_argument("--dump-blocks", action="store_true",
                    help="dump every block's spatial output for bisection")
    ap.add_argument("--dump-interior", action="store_true",
                    help="dump block-0 interior taps (n1/qkv/attnbr/n2/mlpbr)")
    args = ap.parse_args()

    t = load_tensors(Path(args.model_path))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- weights ----
    w_patch = pick(t, "patch_generator", "embedder", "weight")
    pos_embed = pick(t, "patch_generator", "pos_embed").squeeze(0)   # (Ntok, C)
    cls_tok = pick(t, "cls_token", "token")                          # (n_skip, C)
    n_skip, C = cls_tok.shape
    P = int(round((w_patch.shape[1] / 3) ** 0.5))
    pos_grid = int(round(pos_embed.shape[0] ** 0.5))
    # RADIO-only block family (the checkpoint also carries the main tower's
    # blocks, which share the ".blocks." infix).
    radio_prefix = next(k[: -len(".patch_generator.embedder.weight")]
                        for k in t if k.endswith(".patch_generator.embedder.weight"))
    blocks = sorted({int(k[len(radio_prefix) + 8:].split(".")[0])
                     for k in t if k.startswith(radio_prefix + ".blocks.")})
    depth = len(blocks)
    out_indices = [round((i + 1) * depth / 4) - 1 for i in range(4)]
    out_indices[-1] = depth - 1
    print(f"radio torch ref: P{P} C{C} depth{depth} idx{out_indices} skip{n_skip}")

    # ---- input: PIL resize to side x side, [0,1], HWC float ----
    from PIL import Image
    img = Image.open(args.image).convert("RGB").resize((args.side, args.side), Image.BILINEAR)
    x_img = torch.from_numpy(np.asarray(img)).float() / 255.0   # HWC
    x_img.numpy().tofile(out / "input.bin")

    # conditioner
    x = x_img.permute(2, 0, 1)[None]                             # 1CHW
    mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
    x = (x - mean) / std

    # patch embed: Im2Patches 'b c (py yy) (px xx) -> b (py px) (c yy xx)'
    B, Cn, Hs, Ws = x.shape
    py, px = Hs // P, Ws // P
    patches = x.reshape(B, Cn, py, P, px, P).permute(0, 2, 4, 1, 3, 5).reshape(B, py * px, -1)
    tok = patches @ w_patch.t().float()                          # (1, N, C)
    tok.contiguous().numpy().tofile(out / "torch_dbg_patchembed.bin")

    # CPE pos (eval path): CHW -> bilinear(max,max) -> crop -> flatten
    pe = pos_embed.reshape(pos_grid, pos_grid, C).permute(2, 0, 1)[None].float()
    max_dim = max(py, px)
    pe = F.interpolate(pe, size=(max_dim, max_dim), mode="bilinear", align_corners=False)
    pe = pe[:, :, :py, :px]                                      # top-left crop
    pos_enc = pe.flatten(2).permute(0, 2, 1)                     # (1, N, C)
    tok = tok + pos_enc
    tok.contiguous().numpy().tofile(out / "torch_dbg_postpos.bin")

    tok = torch.cat([cls_tok[None].float(), tok], dim=1)         # (1, skip+N, C)
    tok.contiguous().numpy().tofile(out / "torch_dbg_postcls.bin")

    # blocks
    def ln(y, w, b):
        return F.layer_norm(y, (y.shape[-1],), w, b, 1e-5)

    for bi in blocks:
        blk = f"{radio_prefix}.blocks.{bi}"
        n1 = ln(tok, t[f"{blk}.norm1.weight"], t[f"{blk}.norm1.bias"])
        if bi == 0 and args.dump_interior:
            n1.contiguous().numpy().tofile(out / "torch_dbg_blk0_n1.bin")
        qkv = n1 @ t[f"{blk}.attn.qkv.weight"].t().float() + t[f"{blk}.attn.qkv.bias"].float()
        if bi == 0 and args.dump_interior:
            qkv.contiguous().numpy().tofile(out / "torch_dbg_blk0_qkv.bin")
        q, k, v = qkv.chunk(3, dim=-1)
        nh = args.num_heads
        hd = C // nh
        q = q.view(1, -1, nh, hd).transpose(1, 2)
        k = k.view(1, -1, nh, hd).transpose(1, 2)
        v = v.view(1, -1, nh, hd).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        o = o.transpose(1, 2).reshape(1, -1, C)
        o = o @ t[f"{blk}.attn.proj.weight"].t().float() + t[f"{blk}.attn.proj.bias"].float()
        if f"{blk}.ls1.gamma" in t:
            o = o * t[f"{blk}.ls1.gamma"].float()
        if bi == 0 and args.dump_interior:
            o.contiguous().numpy().tofile(out / "torch_dbg_blk0_attnbr.bin")
        tok = tok + o
        n2 = ln(tok, t[f"{blk}.norm2.weight"], t[f"{blk}.norm2.bias"])
        h = n2 @ t[f"{blk}.mlp.fc1.weight"].t().float() + t[f"{blk}.mlp.fc1.bias"].float()
        h = F.gelu(h)
        h = h @ t[f"{blk}.mlp.fc2.weight"].t().float() + t[f"{blk}.mlp.fc2.bias"].float()
        if f"{blk}.ls2.gamma" in t:
            h = h * t[f"{blk}.ls2.gamma"].float()
        if bi == 0 and args.dump_interior:
            h.contiguous().numpy().tofile(out / "torch_dbg_blk0_mlpbr.bin")
        tok = tok + h
        if args.dump_blocks:
            (tok[:, n_skip:, :].reshape(1, -1, C).contiguous().numpy().astype(
                np.float32)).tofile(out / f"torch_blk{bi}.bin")
        if bi in out_indices:
            li = out_indices.index(bi)
            feats = tok[:, n_skip:, :]                            # spatial only
            gh, gw = Hs // P, Ws // P
            # NCHW — matches the C++ run() read-back convention
            feats.reshape(1, gh, gw, C).permute(0, 3, 1, 2).contiguous().numpy().astype(
                np.float32).tofile(out / f"torch_radio_L{li}.bin")
            print(f"L{li} @ block {bi}: spatial {gh}x{gw} c{C}")
    print(f"dumped to {out}")


if __name__ == "__main__":
    main()
