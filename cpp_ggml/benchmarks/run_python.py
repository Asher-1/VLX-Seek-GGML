#!/usr/bin/env python3
"""Run the official python backend over all benchmark images with ONE model load.

Outputs the same JSON as inference.py plus the official visualization
(inference.py::visualize_result_boxes — imported, not reimplemented), so the
pngs are pixel-identical to what the official CLI would draw.

  PYTHONNOUSERSITE=1 .venv-cuda/bin/python cpp_ggml/benchmarks/run_python.py \
      --images cpp_ggml/benchmarks/images --out cpp_ggml/benchmarks/results/python
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_ggml/benchmarks"))

from cases import CASES  # single source of truth (shared with the cpp runner)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "cpp_ggml/benchmarks/images"))
    ap.add_argument("--out", default=str(ROOT / "cpp_ggml/benchmarks/results/python"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--proposals-dir", default=None,
                    help="per-image JSON with fixed boxes (alignment mode); "
                         "omit = closed-loop, each image runs the official "
                         "WeDetect path (load_wedetect_proposals)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    from inference import visualize_result_boxes
    from vlx_seek_worker import VLXSeekWorker

    worker = VLXSeekWorker(str(ROOT / "resources/VLX-Seek-1.5-10B"), device=args.device)
    summary = {}
    for image_name, task, text, lang in CASES:
        image = Image.open(Path(args.images) / image_name).convert("RGB")
        boxes = None
        if task != "vqa":
            if args.proposals_dir:
                boxes = json.loads((Path(args.proposals_dir) / f"{Path(image_name).stem}.json").read_text())
            else:
                from inference import load_wedetect_proposals
                boxes = load_wedetect_proposals(image, str(ROOT / "resources/wedetect_base_uni.pth"))
        t0 = time.perf_counter()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        if task == "vqa":
            result = worker.predict(image, text, max_new_tokens=2048, temperature=0.0)
        else:
            result = worker.run_task(image, task, text, lang=lang, bbox_list=boxes,
                                     max_new_tokens=2048, temperature=0.0)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1e3
        vis = out / f"{Path(image_name).stem}_{task}.png"
        if result.get("result_bbox_list"):
            visualize_result_boxes(image, result["result_bbox_list"], str(vis))
            result["visualization_path"] = str(vis)
        key = f"{Path(image_name).stem}_{task}"
        result["_wall_ms_first_run"] = round(wall_ms, 1)
        (out / f"{key}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        summary[key] = {"answer": result["answer"],
                        "prompt_tokens": result["prompt_tokens"],
                        "completion_tokens": result["completion_tokens"]}
        print(f"{key}: {result['completion_tokens']} tok, wall {wall_ms:.0f} ms | "
              f"{result['answer'][:60]!r}")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"python results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
