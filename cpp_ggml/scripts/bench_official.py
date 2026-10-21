#!/usr/bin/env python3
"""Official python latency baseline, split the way vlx-cli logs it.

Same request as the parity gate (demo image, fixed --bbox-list, greedy):
  TTFT    = generate(max_new_tokens=1)      vision + prefill + 1 sample
  full    = generate(max_new_tokens=2048)   until <|im_end|>
  decode  = (completion_tokens - 1) / (full - TTFT)
Warm-up once, then the median of --runs, torch.cuda.synchronize() around
every measurement (CUDA kernels are async; unsynced timers measure enqueue).

  .venv-cuda/bin/python cpp_ggml/scripts/bench_official.py --bbox-list "$(cat /tmp/vlx-official/boxes.json)"
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(ROOT / "resources/VLX-Seek-1.5-10B"))
    ap.add_argument("--image", default=str(ROOT / "demo/demo_image.jpg"))
    ap.add_argument("--task", default="detection")
    ap.add_argument("--text", default="orange; apple")
    ap.add_argument("--bbox-list", required=True, help="JSON [[x1,y1,x2,y2],...]")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    def sync() -> None:
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

    from vlx_seek_worker import VLXSeekWorker
    t = time.perf_counter()
    worker = VLXSeekWorker(args.model_path, device=args.device)
    sync()
    load_ms = (time.perf_counter() - t) * 1e3
    image = Image.open(args.image)
    boxes = json.loads(args.bbox_list)

    def run(n: int) -> tuple[float, dict]:
        sync()
        t0 = time.perf_counter()
        r = worker.run_task(image, args.task, args.text, bbox_list=boxes,
                            max_new_tokens=n, temperature=0.0)
        sync()
        return (time.perf_counter() - t0) * 1e3, r

    run(2048)  # warm-up: cuDNN autotune, allocator, lazy module init
    ttft = statistics.median(run(1)[0] for _ in range(args.runs))
    fulls = [run(2048) for _ in range(args.runs)]
    full = statistics.median(f for f, _ in fulls)
    res = fulls[0][1]
    n = res["completion_tokens"]
    dec = (n - 1) * 1e3 / (full - ttft) if n > 1 and full > ttft else float("nan")
    mem = torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else 0.0
    print(f"[official {args.device} bf16] load {load_ms:.0f} ms | TTFT(vision+prefill) {ttft:.1f} ms | "
          f"decode {n} tok {full - ttft:.1f} ms ({dec:.2f} tok/s) | e2e {full:.1f} ms | "
          f"peak {mem:.1f} GiB | prompt {res['prompt_tokens']}")
    print(json.dumps({"answer": res["answer"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
