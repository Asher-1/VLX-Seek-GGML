#!/usr/bin/env python3
"""Per-layer LLM reference on FIXED inputs: localizes LLM-only numerics.

Feeds one inputs_embeds tensor (default: the official float32 run's, so the
vision chain contributes zero noise) through the official language model and
records every decoder layer's output (the residual stream, = llama.cpp
`l_out-<il>`) plus the last-position logits. Run once with --dtype float32
(the referee) and once with --dtype bfloat16 (the official release numerics);
compare_llm_layers.py then puts the C++ `vlx-cli gen --dump-layers` taps
between them.

  python3 cpp_ggml/scripts/dump_llm_layers.py --dtype float32 --out /tmp/llm-f32
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="/tmp/vlx-official-f32/ref.pt",
                    help="dump_official_ref.py output holding inputs_embeds/position_ids")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dt = getattr(torch, args.dtype)

    ref = torch.load(args.ref, weights_only=False)
    import vlx_seek_worker
    if dt == torch.float32:
        vlx_seek_worker.load_pretrained_model = functools.partial(
            vlx_seek_worker.load_pretrained_model, dtype=torch.float32)
    worker = vlx_seek_worker.VLXSeekWorker("resources/VLX-Seek-1.5-10B", device=args.device)
    if dt == torch.float32:
        worker.model.float()
    model = worker.model
    lm = model.model.language_model

    taps: dict[int, torch.Tensor] = {}

    def hook(il):
        def f(mod, a, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            taps[il] = h[0].float().cpu()
        return f
    for il, layer in enumerate(lm.layers):
        layer.register_forward_hook(hook(il))

    emb = ref["inputs_embeds"].to(args.device, dt)
    pid = ref["position_ids"].to(args.device)
    with torch.inference_mode():
        o = lm(inputs_embeds=emb, position_ids=pid, use_cache=False)
        logits = model.lm_head(o.last_hidden_state[:, -1]).float().cpu()[0]
    for il, h in taps.items():
        h.numpy().astype(np.float32).tofile(out / f"l_out-{il}.f32")
    logits.numpy().astype(np.float32).tofile(out / "logits.f32")
    print(f"[layers] {len(taps)} layers, hidden {tuple(taps[0].shape)} -> {out}")


if __name__ == "__main__":
    main()
