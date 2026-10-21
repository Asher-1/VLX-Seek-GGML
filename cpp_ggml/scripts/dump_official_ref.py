#!/usr/bin/env python3
"""End-to-end reference of the OFFICIAL python pipeline (inference.py path).

Runs exactly what `python inference.py --task <task> --text <text>` runs
(WeDetect-Uni proposals -> VLXSeekWorker.run_task -> greedy generate), but
with read-only hooks that record every boundary the C++ chain must match:

  wedetect_boxes      final 100 xyxy proposals (original pixels)
  primary_pixels      Qwen2VL processor pixel_values (N, 1536) + grid_thw
  aux_input           the tensor fed to RADIO after _resize_to_supported
  image_embeds        pooler embeddings scattered into <|image_pad|>
  object_feats        mm_projector_aux outputs scattered into <objfeat>
  input_ids           expanded prompt ids (what generate() sees)
  position_ids        (3, 1, L) M-RoPE positions of the prefill forward
  step_logits         last-position logits of every decode step (top-20)
  answer / result     the worker's final dict (identical to inference.py)

No upstream file is modified: CUDA-only calls in the WeDetect helper are
shimmed to the selected device locally in this process.

Usage:
  python3 cpp_ggml/scripts/dump_official_ref.py --text "orange; apple" \
      --device cpu --out /tmp/vlx-official
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "demo/demo_image.jpg"))
    ap.add_argument("--task", default="detection")
    ap.add_argument("--text", default="orange; apple")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--bbox-list", default=None, help="skip WeDetect, use these boxes")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="bfloat16 = what inference.py runs; float32 = noise referee "
                         "(same weights, no bf16 rounding anywhere)")
    ap.add_argument("--out", default="/tmp/vlx-official")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import inference

    if args.device == "cpu":
        # inference.load_wedetect_proposals / SimpleYOLOWorldDetector.forward
        # hard-code .cuda(); keep them on CPU without touching upstream code.
        torch.nn.Module.cuda = lambda self, *a, **k: self
        torch.Tensor.cuda = lambda self, *a, **k: self
        torch.cuda.empty_cache = lambda: None

    image = Image.open(args.image).convert("RGB")
    ref: dict = {"image_size": list(image.size)}
    t0 = time.time()
    if args.bbox_list is None:
        boxes = inference.load_wedetect_proposals(image, inference.DEFAULT_DETECTOR_CHECKPOINT)
    else:
        boxes = json.loads(args.bbox_list)
    ref["t_wedetect"] = time.time() - t0
    ref["wedetect_boxes"] = boxes
    json.dump(boxes, open(out / "boxes.json", "w"))
    print(f"[ref] {len(boxes)} proposals ({ref['t_wedetect']:.1f}s)", flush=True)

    from vlx_seek_worker import VLXSeekWorker
    if args.dtype == "float32":
        import functools
        import vlx_seek_worker
        vlx_seek_worker.load_pretrained_model = functools.partial(
            vlx_seek_worker.load_pretrained_model, dtype=torch.float32)
    t0 = time.time()
    worker = VLXSeekWorker("resources/VLX-Seek-1.5-10B", device=args.device)
    if args.dtype == "float32":
        worker.model.float()   # the aux tower forces bf16 inside load_model()
    ref["dtype"] = args.dtype
    ref["t_load"] = time.time() - t0
    model = worker.model
    inner = model.model

    # ---- read-only taps ----
    orig_prep = worker._prepare_image_inputs

    def prep(image, boxes):
        r = orig_prep(image, boxes)
        ref["primary_pixels"] = r[0][0].float().cpu()
        ref["grid_thw"] = r[1][0].cpu()
        ref["aux_pixels_raw"] = None if r[2] is None else r[2][0].float().cpu()
        return r
    worker._prepare_image_inputs = prep

    aux = inner.get_vision_tower_aux()
    orig_resize = aux._resize_to_supported

    def resize(img):
        r = orig_resize(img)
        ref["aux_input"] = r.float().cpu()
        return r
    aux._resize_to_supported = resize

    orig_enc_img = inner.encode_images

    def enc_img(*a, **k):
        r = orig_enc_img(*a, **k)
        ref["image_embeds"] = torch.cat(list(r[0]), 0).float().cpu()
        return r
    inner.encode_images = enc_img

    orig_enc_obj = inner.encode_objects

    def enc_obj(*a, **k):
        r = orig_enc_obj(*a, **k)
        ref["object_feats"] = r[0].float().cpu()
        ref["bbox_list_internal"] = a[1][0].float().cpu() if len(a) > 1 else None
        return r
    inner.encode_objects = enc_obj

    # HFRE output (roi_align + pos embed), i.e. the mm_projector_aux input.
    if getattr(inner, "mm_projector_aux", None) is not None:
        inner.mm_projector_aux.register_forward_pre_hook(
            lambda m, a: ref.__setitem__("hfre_fused", a[0][0].float().cpu()))

    first = {}

    def lm_pre(mod, a, kw):
        if "position_ids" not in first:
            first["position_ids"] = kw.get("position_ids")
            first["inputs_embeds"] = kw.get("inputs_embeds")
        return None
    inner.language_model.register_forward_pre_hook(lm_pre, with_kwargs=True)

    step_logits = []

    def lm_head_hook(mod, a, o):
        step_logits.append(o[0, -1].float().cpu())
    model.lm_head.register_forward_hook(lm_head_hook)

    orig_generate = model.generate

    def gen(**kw):
        ref["input_ids"] = kw["inputs"][0].cpu()
        return orig_generate(**kw)
    model.generate = gen

    t0 = time.time()
    result = worker.run_task(image, args.task, args.text, lang=args.lang,
                             bbox_list=boxes, max_new_tokens=args.max_new_tokens,
                             temperature=0.0, top_p=1.0)
    ref["t_generate"] = time.time() - t0
    ref["result"] = result
    pid = first["position_ids"]
    ref["position_ids"] = None if pid is None else pid.cpu()
    ref["inputs_embeds"] = first["inputs_embeds"].float().cpu()
    ref["step_argmax"] = [int(l.argmax()) for l in step_logits]
    ref["step_top20"] = [torch.topk(l, 20) for l in step_logits]
    ref["step0_logits"] = step_logits[0] if step_logits else None
    torch.save(ref, out / "ref.pt")
    json.dump(result, open(out / "result.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[ref] load {ref['t_load']:.1f}s generate {ref['t_generate']:.1f}s "
          f"-> {out / 'ref.pt'}", flush=True)


if __name__ == "__main__":
    main()
