#!/usr/bin/env python3
"""Official parity gate for the Qwen3.5-VL mmproj chain (final tap + pooler).

Usage:
  python3 compare_mmproj_parity.py /tmp/vlx-mmproj [--max-abs 0.5] [--max-rel 0.05]

Reads <dir>/torch_mm_final.bin vs <dir>/ggml_final.bin and
      <dir>/torch_mm_pooler.bin vs <dir>/ggml_pooler.bin
(byte-identical pipelines: scripts/dump_torch_mmproj.py --out <dir>, then
radio-cli mmproj --dump-final <dir>/ggml_final.bin --dump-llm <dir>/ggml_pooler.bin).

Pass criterion: |cpp - torch| <= max_abs + max_rel*|torch| elementwise.
Defaults 0.5/0.05 follow the sibling-repo f32 band: the residual f32 noise of
ggml FA/matmul cascades through 27 pre-norm blocks into ~1e-2 absolute on the
few 1e4-scale channels (ch 514 family), i.e. <=2.2% relative — the 5% relative
band covers it with zero violations on the demo image (58x72 grid).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

PAIRS = [
    ("final", "torch_mm_final.bin", "ggml_final.bin"),
    ("pooler", "torch_mm_pooler.bin", "ggml_pooler.bin"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("refdir")
    ap.add_argument("--max-abs", type=float, default=0.5)
    ap.add_argument("--max-rel", type=float, default=0.05)
    args = ap.parse_args()

    ref_dir = Path(args.refdir)
    ok = True
    for name, torch_name, ggml_name in PAIRS:
        tr = ref_dir / torch_name
        gp = ref_dir / ggml_name
        if not tr.exists() or not gp.exists():
            print(f"{name}: MISSING ({tr.name} / {gp.name})")
            ok = False
            continue
        a = np.fromfile(gp, dtype=np.float32)
        b = np.fromfile(tr, dtype=np.float32)
        if a.size != b.size:
            print(f"{name}: SIZE MISMATCH {a.size} vs {b.size}")
            ok = False
            continue
        diff = np.abs(a - b)
        band = args.max_abs + args.max_rel * np.abs(b)
        viol = int((diff > band).sum())
        print(
            f"{name}: n={a.size} mean_abs={diff.mean():.6g} max_abs={diff.max():.6g} "
            f"violations={viol}"
        )
        if viol:
            ok = False
    print("PARITY: PASS" if ok else "PARITY: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
