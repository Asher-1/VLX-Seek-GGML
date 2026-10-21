#!/usr/bin/env python3
"""WeDetect gate: C++ `vlx-cli detect --dump-dir` vs dump_wedetect_ref.py.

Stages (abs band is the verdict, corr is auxiliary):
  letterbox_u8   byte-exact (PIL bilinear letterbox reproduced on the host)
  c1..c4/neck*   C++ dumps token-major (H*W, C); reference is NCHW -> transposed
  cls_logits     (8400, 256) pre-sigmoid
  bbox_preds     (8400, 4) distance * stride
  boxes          final proposal list: the downstream VLX prompt sorts boxes in
                 raster order, so the gate is set-equality under a pixel band
                 (Hungarian-free: greedy nearest match, each ref box used once)
                 plus the official order (score-descending) compared directly.

Bands: f32 CPU ref vs f32 ggml differ only by accumulation order; the 36-block
ConvNeXt amplifies ~1e-6 relative noise, so the per-stage band is relative to
the stage's max |ref| (|d|max / |ref|max <= rel). Box band is in pixels.

Tiers (--tier):
  f32  rel 2e-3, box 0.05 px: same fp32 math, only summation order differs.
  f16  rel 5e-3, box 1.0 px: f16 weights carry 2^-11 rounding. The band is
       ~2x the official pipeline's own CPU<->CUDA divergence (cls 1.2e-3,
       bbox 2.4e-3, box set 0.26 px measured with dump_wedetect_ref.py
       --device cuda), i.e. f16 must be as close to the official CPU run as
       official CUDA is. Backbone/neck taps get 1e-2: the 2^-11 weight
       rounding is largest relative to the shallow taps (measured c1 6.8e-3,
       neck0 8.9e-3 on CPU/CUDA/Vulkan alike) and is renormalised by the
       later LNs; the head outputs and boxes stay inside the official band.
       1e-2 still catches every structural bug (those show up as rel ~1).
  q8_0 has no tier: on this detector it changes the top-100 box set even on
  CPU (set max |d| > 300 px), so it is not an equivalent format.

The raster check sorts both lists by (x1, y1, idx) exactly like the VLX prompt
builder and compares element-wise: equal raster order => identical <objN>
numbering downstream.

Usage:
  python3 cpp_ggml/scripts/compare_wedetect_parity.py --cpp /tmp/wd-cpp \
      --ref /tmp/wedetect-ref [--tier f32|f16] [--rel R] [--box-px P]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TAPS = [("c1", 128, 160), ("c2", 256, 80), ("c3", 512, 40), ("c4", 1024, 20),
        ("neck0", 128, 80), ("neck1", 256, 40), ("neck2", 512, 20)]


def stat(name: str, a: np.ndarray, r: np.ndarray, rel: float) -> bool:
    d = np.abs(a.astype(np.float64) - r.astype(np.float64))
    scale = float(np.abs(r).max()) or 1.0
    corr = float(np.corrcoef(a.ravel(), r.ravel())[0, 1]) if a.size > 1 else 1.0
    ok = d.max() / scale <= rel
    print(f"  {name:12s} max|d|={d.max():.3e} mean|d|={d.mean():.3e} |ref|max={scale:.3e} "
          f"rel={d.max() / scale:.2e} corr={corr:.9f} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", default="/tmp/wd-cpp")
    ap.add_argument("--ref", default="/tmp/wedetect-ref")
    ap.add_argument("--official-boxes", default=None,
                    help="optional: boxes.json from the full official pipeline")
    ap.add_argument("--tier", choices=("f32", "f16"), default="f32")
    ap.add_argument("--rel", type=float, default=None, help="head (cls/bbox) band")
    ap.add_argument("--tap-rel", type=float, default=None, help="backbone/neck tap band")
    ap.add_argument("--box-px", type=float, default=None)
    ap.add_argument("--skip-taps", action="store_true", help="boxes-only (no dump-dir taps)")
    ap.add_argument("--cpp-nchw", action="store_true",
                    help="--cpp is also an official (NCHW) dump, e.g. official CUDA vs CPU")
    args = ap.parse_args()
    # (tap rel, head rel, box px) -- see the Tiers section of the docstring
    tier = {"f32": (2e-3, 2e-3, 0.05), "f16": (1e-2, 5e-3, 1.0)}[args.tier]
    args.tap_rel = tier[0] if args.tap_rel is None else args.tap_rel
    args.rel = tier[1] if args.rel is None else args.rel
    args.box_px = tier[2] if args.box_px is None else args.box_px
    c, r = Path(args.cpp), Path(args.ref)
    ok = True

    if not args.skip_taps:
        lb_c = np.fromfile(c / "letterbox_u8.bin", np.uint8)
        lb_r = np.fromfile(r / "letterbox_u8.bin", np.uint8)
        n_diff = int((lb_c != lb_r).sum())
        print(f"  letterbox_u8 {n_diff} differing bytes {'PASS' if n_diff == 0 else 'FAIL'}")
        ok &= n_diff == 0
        for name, ch, hw in TAPS:
            a = np.fromfile(c / f"{name}.bin", np.float32)
            a = a.reshape(ch, hw * hw).T if args.cpp_nchw else a.reshape(hw * hw, ch)
            b = np.fromfile(r / f"{name}.bin", np.float32).reshape(ch, hw * hw).T
            ok &= stat(name, a, b, args.tap_rel)
        for name, ch in (("cls_logits", 256), ("bbox_preds", 4)):
            a = np.fromfile(c / f"{name}.bin", np.float32).reshape(-1, ch)
            b = np.fromfile(r / f"{name}.bin", np.float32).reshape(-1, ch)
            ok &= stat(name, a, b, args.rel)

    bc = np.array(json.load(open(c / "boxes.json")), np.float64).reshape(-1, 4)
    for tag, path in (("ref", r / "boxes.json"),
                      ("official", Path(args.official_boxes) if args.official_boxes else None)):
        if path is None or not path.exists():
            continue
        br = np.array(json.load(open(path)), np.float64).reshape(-1, 4)
        if len(bc) != len(br):
            print(f"  boxes[{tag}] count {len(bc)} vs {len(br)} FAIL")
            ok = False
            continue
        d_order = np.abs(bc - br).max()
        # set equality: each C++ box matched to the nearest unused ref box
        used = np.zeros(len(br), bool)
        worst = 0.0
        for b in bc:
            dd = np.abs(br - b).max(1)
            dd[used] = np.inf
            j = int(dd.argmin())
            used[j] = True
            worst = max(worst, float(dd[j]))
        set_ok = worst <= args.box_px

        def raster(b: np.ndarray) -> np.ndarray:
            return b[np.lexsort((np.arange(len(b)), b[:, 1], b[:, 0]))]
        d_raster = float(np.abs(raster(bc) - raster(br)).max())
        raster_ok = d_raster <= args.box_px
        print(f"  boxes[{tag}] n={len(bc)} score-order max|d|={d_order:.3e}px "
              f"set max|d|={worst:.3e}px {'PASS' if set_ok else 'FAIL'} "
              f"raster-order max|d|={d_raster:.3e}px {'PASS' if raster_ok else 'FAIL'}")
        ok &= set_ok and raster_ok
    if (c / "scores.json").exists() and (r / "meta.json").exists():
        sc = np.array(json.load(open(c / "scores.json")), np.float64)
        sr = np.array(json.load(open(r / "meta.json"))["scores"], np.float64)
        if len(sc) == len(sr):
            print(f"  scores       max|d|={np.abs(np.sort(sc) - np.sort(sr)).max():.3e}")
    print("WEDETECT GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
