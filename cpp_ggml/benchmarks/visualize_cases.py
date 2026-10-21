#!/usr/bin/env python3
"""Per-case alignment verdicts + official-style side-by-side visualizations.

The rendering calls inference.py::visualize_result_boxes directly (the
official function), so every panel is pixel-identical to what the official
CLI draws — red width-3 rectangles plus the model-generated label text.

Outputs:
  charts/case_<key>_cpp_py.png   left: C++ f16, right: python, one row per case
  results/alignment.json         per-case answer/token/box equality (fixed-boxes mode)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cpp_ggml/benchmarks"))

from cases import BACKENDS, CASES, case_key  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

RESULTS = ROOT / "cpp_ggml/benchmarks/results"
CHARTS = ROOT / "cpp_ggml/benchmarks/charts"
IMAGES = ROOT / "cpp_ggml/benchmarks/images"
MODE = "fixed"   # the parity criterion; closed-loop results stay in results/closed


def load_result(backend: str, key: str) -> dict:
    return json.loads((RESULTS / MODE / backend / f"{key}.json").read_text())


def official_visualize(image: Image.Image, result: dict, out: Path) -> Path:
    from inference import visualize_result_boxes
    boxes = result.get("result_bbox_list") or []
    if boxes:
        visualize_result_boxes(image, boxes, str(out))
    else:
        image.save(out)
    return out


def panel_caption(name: str, match: bool, n_boxes: int) -> Image.Image:
    img = Image.new("RGB", (360, 34), (24, 26, 32))
    d = ImageDraw.Draw(img)
    color = (140, 220, 140) if match else (240, 120, 120)
    d.text((8, 9), f"{name}  {'MATCH' if match else 'DIFFERS'}  boxes={n_boxes}",
           fill=color)
    return img


def _banner_font(height: int):
    """DejaVuSans at ~55% of the banner height; falls back to PIL's default."""
    from PIL import ImageFont
    size = max(12, int(height * 0.55))
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def prompt_banner(img: Image.Image, task: str, text: str) -> Image.Image:
    """Overlay the official-style prompt banner (pale yellow, red text) in the
    top-left corner, like the showcase images under assets/seek1.5-*.png —
    without it a side-by-side panel does not show what was asked."""
    img = img.copy()
    banner_h = max(26, int(img.height * 0.062))
    font = _banner_font(banner_h)
    label = f"{task}: {text}"
    probe = Image.new("RGB", (1, 1))
    text_w = ImageDraw.Draw(probe).textlength(label, font=font)
    banner_w = int(min(img.width * 0.92, text_w + 24))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, banner_w, banner_h), fill=(255, 248, 201))
    d.text((12, (banner_h - font.size) // 2), label, fill=(214, 39, 40), font=font)
    return img


def side_by_side(key: str, image_name: str, task: str, text: str,
                 rows: dict[str, dict], match: bool) -> Path:
    image = Image.open(IMAGES / image_name).convert("RGB")
    panels = []
    for backend in ("cpp_f16", "python"):
        out = RESULTS / MODE / backend / f"{key}.official.png"
        official_visualize(image, rows[backend], out)
        img = Image.open(out)
        img = prompt_banner(img, task, text)
        cap = panel_caption(backend, match, len(rows[backend].get("result_bbox_list") or []))
        w = min(img.width, 720)
        img = img.resize((w, int(img.height * w / img.width)))
        panel = Image.new("RGB", (w, img.height + 34), (24, 26, 32))
        panel.paste(cap, (0, 0))
        panel.paste(img, (0, 34))
        panels.append(panel)
    total = Image.new("RGB", (sum(p.width for p in panels) + 8,
                              max(p.height for p in panels)), (24, 26, 32))
    x = 0
    for p in panels:
        total.paste(p, (x, 0))
        x += p.width + 8
    out = CHARTS / f"case_{key}_cpp_py.png"
    total.save(out)
    return out


def main() -> int:
    CHARTS.mkdir(parents=True, exist_ok=True)
    # drop panels of cases that are no longer in CASES (renamed/retired inputs)
    live = {f"case_{case_key(i, t)}_cpp_py.png" for i, t, _x, _l in CASES}
    for stale in CHARTS.glob("case_*_cpp_py.png"):
        if stale.name not in live:
            stale.unlink()
            print(f"removed stale panel: {stale.name}")
    alignment = {}
    for image_name, task, text, lang in CASES:
        key = case_key(image_name, task)
        try:
            rows = {b: load_result(b, key) for b in BACKENDS}
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"{key}: results missing or truncated (run pending or GPU busy) — skipped")
            continue
        answers = {b: rows[b]["answer"] for b in BACKENDS}
        f16_match = answers["cpp_f16"] == answers["python"]
        q8_match = answers["cpp_q8_0"] == answers["python"]
        match = f16_match and q8_match
        tok_match = all(rows[b]["prompt_tokens"] == rows["python"]["prompt_tokens"]
                        and rows[b]["completion_tokens"] == rows["python"]["completion_tokens"]
                        for b in ("cpp_f16", "cpp_q8_0"))
        boxes = {b: len(rows[b].get("result_bbox_list") or []) for b in BACKENDS}
        box_match = boxes["cpp_f16"] == boxes["python"] and boxes["cpp_q8_0"] == boxes["python"]
        if box_match and boxes["python"]:
            maxd = 0.0
            for b in ("cpp_f16", "cpp_q8_0"):
                for a, c in zip(rows[b]["result_bbox_list"], rows["python"]["result_bbox_list"]):
                    maxd = max(maxd, max(abs(a[k] - c[k])
                                         for k in ("xmin", "ymin", "xmax", "ymax")))
        else:
            maxd = None
        alignment[key] = {
            "f16_vs_python_token_identical": f16_match,
            "q8_0_vs_python_token_identical": q8_match,
            "tokens_identical": tok_match,
            "n_boxes": boxes,
            "box_count_match": box_match,
            "max_box_coord_delta_px": None if maxd is None else round(maxd, 5),
            "answers": answers,
        }
        side_by_side(key, image_name, task, text, rows, match)
        print(f"{key}: f16 {'MATCH' if f16_match else 'DIFFERS'}, "
              f"q8_0 {'MATCH' if q8_match else 'DIFFERS'} vs python, boxes {boxes}"
              + (f", max delta {maxd:.4f} px" if maxd is not None else ""))

    (RESULTS / "alignment.json").write_text(json.dumps(alignment, ensure_ascii=False, indent=2))
    n_f16 = sum(1 for v in alignment.values() if v["f16_vs_python_token_identical"])
    n_q8 = sum(1 for v in alignment.values() if v["q8_0_vs_python_token_identical"])
    print(f"alignment: f16 {n_f16}/{len(alignment)}, q8_0 {n_q8}/{len(alignment)} "
          f"token-identical vs python -> results/alignment.json")
    return 0 if n_f16 == len(alignment) else 1


if __name__ == "__main__":
    sys.exit(main())
