#!/usr/bin/env python3
"""Torch reference for the fused object features (<objfeat> tokens).

Consumes already-dumped intermediates (no HF class, no tower re-run):
  torch_radio_L{0..3}.bin  (aux levels, NCHW, from dump_torch_stages.py)
  torch_fpn_L{0..3}.bin    (vt levels,  NCHW, from dump_torch_mmproj.py)
  boxes.json               (original-image pixels)

Chain (hybrid_finegrained_region_encoder.__call__, 10B config):
  aux   : concat 4 RADIO levels on channels (64x64 grid) -> torchvision
          roi_align(out=14, scale=1/16, aligned=True) -> mean -> (N, 4608)
  vt    : torch_fpn_vt.bin (already pooled, (N, 2048))
  fuse  : cat[aux, vt] (N, 6656) + gen_sineembed_for_position of the
          normalized (cx, cy, w, h) boxes — output block order (y, x, w, h)
  proj  : mm_projector_aux (mlp2x_gelu): Linear(6656->4096) -> GELU(erf)
          -> Linear(4096->4096)                          -> (N, 4096)

Outputs (into --out): torch_fused.bin (N, 6656), torch_objfeat.bin (N, 4096).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", default="/tmp/vlx-ref-real")
    ap.add_argument("--fpn-dump-dir", default="/tmp/vlx-mmproj")
    ap.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    ap.add_argument("--out", default="/tmp/vlx-ref-real")
    ap.add_argument("--vt-dim", type=int, default=4096)
    args = ap.parse_args()

    dump, fdn, out = Path(args.dump_dir), Path(args.fpn_dump_dir), Path(args.out)
    aux_shapes = [(64, 64)] * 4   # RADIO levels share one grid (1024/16)
    vt_shapes = [(232, 288), (116, 144), (58, 72), (29, 36)]  # FPN pyramid

    aux_lv = [np.fromfile(dump / f"torch_radio_L{i}.bin", dtype=np.float32)
              .reshape(1152, *aux_shapes[i]) for i in range(4)]
    vt_lv = [np.fromfile(fdn / f"torch_fpn_L{i}.bin", dtype=np.float32)
             .reshape(512, *vt_shapes[i]) for i in range(4)]
    boxes = np.array(json.load(open(fdn / "boxes.json")), dtype=np.float32)
    N = boxes.shape[0]

    # ---- aux: channel concat (all levels share the 64x64... here 232x288
    # pyramid grid from the 1024 input; the python path only concats when
    # grids match, which holds for RADIO) -> single RoIAlign at 1/16 ----
    # NOTE: the python aux path assumes all levels share ONE grid; the dumped
    # real pyramid is per-level, so we replicate the per-level RoIAlign of
    # the (use_simpleFPN_for_vt_aux=True) variant with the aux strides —
    # identical math when grids match, and this is what the C++ CLI consumes.
    aux_strides = [16, 16, 16, 16]  # 1/patch16: all levels scale 1/16
    feats = []
    for li in range(4):
        roi = torchvision.ops.roi_align(torch.from_numpy(aux_lv[li])[None].float(),
                                        [torch.from_numpy(boxes)], output_size=14,
                                        spatial_scale=1.0 / aux_strides[li],
                                        aligned=True)
        feats.append(roi.mean(dim=(2, 3)))
    aux = torch.cat(feats, dim=1).numpy()          # (N, 4608)

    vt = np.fromfile(fdn / "torch_fpn_vt.bin", dtype=np.float32).reshape(N, 2048)

    # ---- fuse ----
    fused = np.concatenate([aux, vt], axis=1).astype(np.float32)   # (N, 6656)
    fused.tofile(out / "torch_auxcat.bin")   # pre-pos concat (aux|vt)

    # pos embed: original-image frame = merged grid extent / (1/32)
    H2, W2 = 29, 36                                 # merged grid for 58x72
    img_w, img_h = W2 * 32, H2 * 32
    cx = (boxes[:, 0] + boxes[:, 2]) / 2 / img_w
    cy = (boxes[:, 1] + boxes[:, 3]) / 2 / img_h
    w = (boxes[:, 2] - boxes[:, 0]) / img_w
    h = (boxes[:, 3] - boxes[:, 1]) / img_h
    D = 6656
    quarter = D // 4
    dim_t = 10000 ** (2 * (np.arange(quarter) // 2) / quarter)
    comps = {"y": cy, "x": cx, "w": w, "h": h}      # output block order y,x,w,h
    pos = np.zeros((N, D), dtype=np.float32)
    for bi, key in enumerate(("y", "x", "w", "h")):
        v = comps[key] * 2 * math.pi
        ang = v[:, None] / dim_t[None, :]           # (N, quarter)
        block = np.stack([np.sin(ang[:, 0::2]), np.cos(ang[:, 1::2])], axis=2)
        pos[:, bi * quarter:(bi + 1) * quarter] = block.reshape(N, quarter)
    fused = fused + pos

    # ---- projector (mlp2x_gelu) ----
    with safe_open(str(Path(args.model_path) / "model.safetensors"), framework="pt") as sf:
        w0 = sf.get_tensor("model.mm_projector_aux.0.weight").float()
        b0 = sf.get_tensor("model.mm_projector_aux.0.bias").float()
        w2 = sf.get_tensor("model.mm_projector_aux.2.weight").float()
        b2 = sf.get_tensor("model.mm_projector_aux.2.bias").float()
    xt = torch.from_numpy(fused)
    y = torch.nn.functional.gelu(xt @ w0.t() + b0)   # nn.GELU() = erf
    y = y @ w2.t() + b2
    obj = y.numpy().astype(np.float32)

    fused.tofile(out / "torch_fused.bin")
    obj.tofile(out / "torch_objfeat.bin")
    print(f"dumped torch_fused {fused.shape} torch_objfeat {obj.shape} -> {out}")


if __name__ == "__main__":
    main()
