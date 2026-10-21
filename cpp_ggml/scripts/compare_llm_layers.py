#!/usr/bin/env python3
"""Per-layer LLM noise: C++ taps vs official bf16, both judged by official f32.

Inputs (same inputs_embeds everywhere, so only LLM numerics differ):
  --exact  dump_llm_layers.py --dtype float32  output dir
  --bf16   dump_llm_layers.py --dtype bfloat16 output dir
  --cpp    vlx-cli gen --dump-layers dir (+ optional --cpp-logits file)

For each layer: ||x - exact|| / ||exact|| over the rows both sides hold (the
C++ last layer keeps only the output row). ratio = cpp / bf16; the first layer
where the ratio jumps is where the C++ graph departs from the reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def rel(a: np.ndarray, e: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float64) - e) / (np.linalg.norm(e) or 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", default="/tmp/llm-f32")
    ap.add_argument("--bf16", default="/tmp/llm-bf16")
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--cpp-logits", default=None)
    ap.add_argument("--n-embd", type=int, default=4096)
    args = ap.parse_args()
    ex, bf, cp = Path(args.exact), Path(args.bf16), Path(args.cpp)
    D = args.n_embd
    il = 0
    print(f"{'layer':>5s} {'rows':>5s} {'|bf16-exact|':>13s} {'|cpp-exact|':>12s} {'ratio':>6s}")
    while (ex / f"l_out-{il}.f32").exists():
        e = np.fromfile(ex / f"l_out-{il}.f32", np.float32).reshape(-1, D).astype(np.float64)
        b = np.fromfile(bf / f"l_out-{il}.f32", np.float32).reshape(-1, D) if bf.exists() else None
        p = cp / f"l_out-{il}.f32"
        c = np.fromfile(p, np.float32).reshape(-1, D) if p.exists() else None
        rows = e.shape[0] if c is None else c.shape[0]
        e_r = e[-rows:]
        rb = rel(b[-rows:], e_r) if b is not None else float("nan")
        rc = rel(c, e_r) if c is not None else float("nan")
        print(f"{il:5d} {rows:5d} {rb:13.3e} {rc:12.3e} {rc / rb:6.2f}")
        il += 1
    if args.cpp_logits:
        e = np.fromfile(ex / "logits.f32", np.float32).astype(np.float64)
        c = np.fromfile(args.cpp_logits, np.float32)
        b = np.fromfile(bf / "logits.f32", np.float32)
        print(f"logits      {rel(b, e):13.3e} {rel(c, e):12.3e} {rel(c, e) / rel(b, e):6.2f}  "
              f"argmax exact/bf16/cpp {e.argmax()} {b.argmax()} {c.argmax()}")


if __name__ == "__main__":
    main()
