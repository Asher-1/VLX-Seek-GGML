#!/usr/bin/env python3
"""Official parity gate for the HFRE vt branch (SimpleFPN levels + pooled).

Usage:
  python3 compare_fpn_parity.py /tmp/vlx-mmproj /tmp/vlx-mmproj-rerun

Reads torch_fpn_L{0..3}.bin + torch_fpn_vt.bin from the torch dump dir and
ggml_fpn_L{0..3}.bin + ggml_vt.bin from the CLI dump dir
(radio-cli mmproj --fpn ... --hfre boxes.json --dump-fpn P --dump-vt out.bin).

Pass criterion: |cpp - torch| <= max_abs + max_rel*|torch| elementwise, applied
per level and to the pooled vt features. Defaults 0.5/0.05 (sibling f32 band).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("torch_dir")
    ap.add_argument("ggml_dir")
    ap.add_argument("--max-abs", type=float, default=0.5)
    ap.add_argument("--max-rel", type=float, default=0.05)
    args = ap.parse_args()

    td, gd = Path(args.torch_dir), Path(args.ggml_dir)
    ok = True
    pairs = [(f"torch_fpn_L{i}.bin", f"ggml_fpn_L{i}.bin", f"L{i}") for i in range(4)]
    pairs.append(("torch_fpn_vt.bin", "ggml_vt.bin", "vt"))
    for tn, gn, name in pairs:
        a_p, b_p = gd / gn, td / tn
        if not a_p.exists() or not b_p.exists():
            print(f"{name}: MISSING ({a_p.name} / {b_p.name})")
            ok = False
            continue
        a = np.fromfile(a_p, dtype=np.float32)
        b = np.fromfile(b_p, dtype=np.float32)
        if a.size != b.size:
            print(f"{name}: SIZE MISMATCH {a.size} vs {b.size}")
            ok = False
            continue
        d = np.abs(a - b)
        band = args.max_abs + args.max_rel * np.abs(b)
        viol = int((d > band).sum())
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"{name}: n={a.size} mean_abs={d.mean():.6g} max_abs={d.max():.6g} "
              f"violations={viol} corr={corr:.6f}")
        if viol:
            ok = False
    print("PARITY: PASS" if ok else "PARITY: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
