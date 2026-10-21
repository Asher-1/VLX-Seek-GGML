#!/usr/bin/env python3
"""Dump WeDetect reference intermediates for the C++ conv-graph gate.

The detector is loaded through convert_wedetect_to_gguf.load_official — the
VERBATIM inference.py::load_wedetect_proposals key remap — so the reference is
the module the official pipeline runs, not a re-derived variant.

Written to --out (raw little-endian, consumed by compare_wedetect_parity.py):
  letterbox_u8.bin   (640, 640, 3) uint8 HWC — the PIL letterbox canvas
  c{1..4}.bin        backbone stages, NCHW f32
  neck{0..2}.bin     neck outputs (stride 8/16/32), NCHW f32
  cls_logits.bin     (8400, 256) f32, pre-sigmoid, anchor-major (torch flatten)
  bbox_preds.bin     (8400, 4) f32, distance * stride (pre distance2bbox)
  boxes.json         the official forward()'s final xyxy list (inference.py)
  meta.json          ratio / offset / ori size

Usage:
  python3 cpp_ggml/scripts/dump_wedetect_ref.py --image demo/demo_image.jpg \
      --out /tmp/wedetect-ref
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(HERE.parents[1] / "demo/demo_image.jpg"))
    ap.add_argument("--checkpoint", default=str(HERE.parents[1] / "resources/wedetect_base_uni.pth"))
    ap.add_argument("--out", default="/tmp/wedetect-ref")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                    help="cuda = the official runtime as shipped (torch defaults: "
                         "cudnn conv TF32 on, matmul TF32 off)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "cpu":
        # the official forward hard-codes .cuda(); keep it on CPU in-process only
        torch.nn.Module.cuda = lambda self, *a, **k: self
        torch.Tensor.cuda = lambda self, *a, **k: self

    from convert_wedetect_to_gguf import load_official
    from generate_proposal import letterbox
    from PIL import Image

    det = load_official(args.checkpoint)
    if args.device == "cuda":
        det = det.cuda()
    torch.set_grad_enabled(False)
    img = Image.open(args.image).convert("RGB")
    lb, ratio, offset = letterbox(img, det.img_size)
    lb_u8 = np.array(lb)
    lb_u8.tofile(out / "letterbox_u8.bin")
    x = torch.tensor(lb_u8).permute(2, 0, 1)[None].to(det.embeddings.dtype) / 255.0
    x = x.to(args.device)

    feats = det.backbone(x)
    for i, f in enumerate(feats):
        f.float().cpu().numpy().tofile(out / f"c{i + 1}.bin")
    neck = det.neck(feats)
    for i, f in enumerate(neck):
        f.float().cpu().numpy().tofile(out / f"neck{i}.bin")
    cls_all, bp_all = [], []
    for li in range(3):
        _, bp, cl = det.head_module_forward_single(
            neck[li], det.bbox_head.cls_preds[li], det.bbox_head.reg_preds[li],
            det.bbox_head.cls_contrasts[li])
        cls_all.append(cl[0].permute(1, 2, 0).reshape(-1, cl.shape[1]))
        bp_all.append(bp[0].permute(1, 2, 0).reshape(-1, 4) * det.bbox_head.featmap_strides[li])
    torch.cat(cls_all).cpu().numpy().tofile(out / "cls_logits.bin")
    torch.cat(bp_all).cpu().numpy().tofile(out / "bbox_preds.bin")

    # the exact official entry point (inference.load_wedetect_proposals body)
    res = det([img])[0]
    if args.device == "cuda":   # official end-to-end latency (warm), CUDA events
        import time
        for _ in range(3):
            det([img])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 10
        for _ in range(n):
            det([img])
        torch.cuda.synchronize()
        print(f"official det([img]) warm latency: {(time.perf_counter() - t0) / n * 1e3:.1f} ms")
    boxes = res["bboxes"].float().cpu().tolist()
    json.dump(boxes, open(out / "boxes.json", "w"))
    json.dump({"ratio": ratio, "offset": list(offset), "size": list(img.size),
               "scores": res["scores"].float().cpu().tolist()},
              open(out / "meta.json", "w"))
    print(f"letterbox ratio={ratio} offset={offset}; {len(boxes)} boxes -> {out}")


if __name__ == "__main__":
    main()
