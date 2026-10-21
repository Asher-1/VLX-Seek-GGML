#!/usr/bin/env python3
"""Elementwise parity gate between the torch reference dump and radio-cli dump.

Usage:
  python3 compare_parity.py /tmp/vlx-ref [--max-abs 0.5] [--max-rel 0.05]

Reads <dir>/torch_radio_L*.bin and <dir>/ggml_radio_L*.bin, compares as float32,
and reports per-level mean/max abs error plus the rel-band verdict. Exit code 0
means every level passes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--max-abs", type=float, default=0.5)
    ap.add_argument("--max-rel", type=float, default=0.05)
    args = ap.parse_args()

    ref_dir = Path(args.refdir)
    ok = True
    for li in range(4):
        tr = ref_dir / f"torch_radio_L{li}.bin"
        gp = ref_dir / f"ggml_radio_L{li}.bin"
        if not tr.exists() or not gp.exists():
            print(f"L{li}: MISSING ({tr.name} / {gp.name})")
            ok = False
            continue
        a = np.fromfile(tr, dtype=np.float32)
        b = np.fromfile(gp, dtype=np.float32)
        if a.shape != b.shape:
            print(f"L{li}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            ok = False
            continue
        # token-major layout: (n_patch, C) — compare per-channel columns
        a2 = a.reshape(-1, a.size // max(1, 64 * 64) * 0 + a.size // 1)  # flat fallback
        diff = np.abs(a - b)
        band = args.max_abs + args.max_rel * np.abs(a)
        viol = diff > band
        print(
            f"L{li}: n={a.size} mean_abs={diff.mean():.6g} max_abs={diff.max():.6g} "
            f"violations={int(viol.sum())}"
        )
        if viol.any():
            ok = False
    print("PARITY: PASS" if ok else "PARITY: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
