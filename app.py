#!/usr/bin/env python3
"""Gradio side-by-side demo: official python backend vs the C++ GGML backend.

Both backends run through run_vlx.sh with identical arguments. Result boxes
are rendered with inference.py::visualize_result_boxes — the official
drawing function — so the two panels are directly comparable and match the
official demo style (red width-3 rectangles + label text).

  PYTHONNOUSERSITE=1 .venv-cuda/bin/python app.py
  # then open http://127.0.0.1:7860

Notes:
  * first python-side request loads the 19.5 GB checkpoint (~1 min); the C++
    side reloads per request (cold start ~8 s: GGUF reads + weight upload).
  * temperature is fixed at 0 (greedy): the only mode where the two backends
    are guaranteed token-identical, and the official recommendation for all
    detection tasks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "demo" / "_app_runs"
WORK.mkdir(parents=True, exist_ok=True)
for _stale in WORK.glob("*"):
    _stale.unlink()   # scratch dir: request inputs/renders are not kept across restarts

TASKS = ["detection", "grounding_single", "counting", "reasoning_detection",
         "brief_region_caption", "region_ocr", "vqa"]


def run_backend(backend: str, image_path: Path, task: str, lang: str, text: str,
                precision: str, target: str) -> tuple[dict | None, str]:
    args = ["bash", str(ROOT / "run_vlx.sh"), "--backend", backend,
            "--device", "cuda", "--no-visualize", "--image-path", str(image_path),
            "--task", task, "--lang", lang, "--text", text]
    if backend == "cpp" and precision != "f16":
        args += ["--precision", precision]
    if target.strip():
        args += ["--target-region-indexes", target.strip()]
    t0 = time.perf_counter()
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                          timeout=900)
    wall = time.perf_counter() - t0
    try:
        return json.loads(proc.stdout), (
            f"**{backend}** — wall {wall:.1f} s\n\n```\n{proc.stdout.strip()}\n```")
    except json.JSONDecodeError:
        return None, (f"**{backend}** FAILED (wall {wall:.1f} s)\n\n```\n"
                      f"{proc.stderr[-2000:]}\n```")


def official_draw(image: Image.Image, result: dict | None) -> Image.Image | None:
    """Render result boxes with the official inference.py drawing function."""
    if not result or not result.get("result_bbox_list"):
        return None
    sys.path.insert(0, str(ROOT))
    from inference import visualize_result_boxes

    out = WORK / f"vis_{time.time_ns()}.png"
    visualize_result_boxes(image, result["result_bbox_list"], str(out))
    return Image.open(out)


def compare(image: Image.Image | None, task: str, lang: str, text: str,
            precision: str, target: str):
    if image is None:
        raise gr.Error("upload an image first")
    if task in ("brief_region_caption", "region_ocr") and not target.strip():
        raise gr.Error("--target-region-indexes is required for that task, e.g. '[0]'")
    image = image.convert("RGB")
    stamp = time.time_ns()
    image_path = WORK / f"in_{stamp}.jpg"
    image.save(image_path)

    cpp = cpp_raw = py = py_raw = None
    if RUN_CPP:
        cpp, cpp_raw = run_backend("cpp", image_path, task, lang, text, precision, target)
    else:
        cpp_raw = f"**cpp** skipped (--backend {ARGS.backend})"
    if RUN_PY:
        py, py_raw = run_backend("python", image_path, task, lang, text, precision, target)
    else:
        py_raw = f"**python** skipped (--backend {ARGS.backend})"

    available = [(b, r) for b, r in (("cpp", cpp), ("python", py)) if r is not None]
    same = len(available) >= 2 and all(r["answer"] == available[0][1]["answer"]
                                       for _b, r in available)
    verdict = ("### ✅ answers are token-identical (greedy)"
               if same else "### ⚠️ answers differ — see the JSON panels")
    for b, r in available:
        verdict += (f"\n\n**{b}**: {r['prompt_tokens']} prompt / "
                    f"{r['completion_tokens']} completion tokens, "
                    f"{len(r.get('result_bbox_list') or [])} boxes")
    vis_cpp = official_draw(image, cpp)
    vis_py = official_draw(image, py)
    return (cpp, py, verdict, cpp_raw, py_raw, vis_cpp, vis_py)


_ap = argparse.ArgumentParser()
_ap.add_argument("--backend", choices=("both", "cpp", "python"), default="both",
                 help="which backends to run per request (cpp needs no python "
                      "deps at all; python needs the 19.5 GB checkpoint load)")
ARGS = _ap.parse_args()
RUN_CPP = ARGS.backend in ("both", "cpp")
RUN_PY = ARGS.backend in ("both", "python")

_TITLE = {
    "both": "VLX-Seek 1.5-10B — C++ (GGML) vs official python",
    "cpp": "VLX-Seek 1.5-10B — C++ (GGML) backend",
    "python": "VLX-Seek 1.5-10B — official python backend",
}
_BTN = {"both": "Run both backends", "cpp": "Run C++ backend",
        "python": "Run python backend"}

with gr.Blocks(title=_TITLE[ARGS.backend]) as demo:
    gr.Markdown(
        f"# {_TITLE[ARGS.backend]}\n"
        f"Backend mode: **{ARGS.backend}** (set with `python app.py --backend "
        f"cpp|python|both` at launch). One request, greedy decoding. Boxes are "
        f"drawn with the official `visualize_result_boxes` for a like-for-like view.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="input image")
            task = gr.Dropdown(TASKS, value="detection", label="task")
            lang = gr.Radio(["en", "zh"], value="en", label="lang")
            text = gr.Textbox("orange; apple", label="text (categories / question / description)")
            precision = gr.Dropdown(["f16", "q8_0"], value="f16",
                                    label="cpp precision (vision weights)",
                                    visible=RUN_CPP)
            target = gr.Textbox("", label="target-region-indexes (region tasks), e.g. [0]")
            btn = gr.Button(_BTN[ARGS.backend], variant="primary")
        with gr.Column(scale=2):
            verdict = gr.Markdown()
            with gr.Tab("visualization (official style)"):
                with gr.Row():
                    vis_c = gr.Image(label="C++ (GGML)", visible=RUN_CPP)
                    vis_p = gr.Image(label="python (official)", visible=RUN_PY)
            with gr.Tab("JSON"):
                with gr.Row():
                    js_c = gr.JSON(label="C++ result", visible=RUN_CPP)
                    js_p = gr.JSON(label="python result", visible=RUN_PY)
            with gr.Tab("raw CLI output"):
                with gr.Row():
                    raw_c = gr.Markdown(visible=RUN_CPP)
                    raw_p = gr.Markdown(visible=RUN_PY)
    btn.click(compare, [inp, task, lang, text, precision, target],
              [js_c, js_p, verdict, raw_c, raw_p, vis_c, vis_p])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
