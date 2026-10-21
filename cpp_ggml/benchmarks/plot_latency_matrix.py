#!/usr/bin/env python3
"""Latency matrix chart: {cuda, vulkan, cpu} x {f16, q8_0} + python reference.

Reads results/matrix/*.time.log (the C++ [time] lines and the python
reference line) and draws:
  charts/latency_matrix.png   three panels:
    left   : end-to-end TTFT (vision + prefill) for EVERY config including
             the python reference — the direct "is the C++ backend faster
             than pytorch?" answer
    middle : per-stage steady-state heatmap for the C++ configs
    right  : decode throughput
Also writes results/latency_matrix.json with the parsed numbers.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
CHARTS = BENCH / "charts"
STAGES = ("prep", "mmproj", "fpn", "radio", "hfre", "proj", "prefill")

VISION_RE = re.compile(
    r"vision warm median of \d+: prep ([\d.]+) \| mmproj ([\d.]+) \| fpn ([\d.]+) "
    r"\| radio ([\d.]+) \| hfre ([\d.]+) \| proj ([\d.]+) \| total ([\d.]+) ms")
LLM_RE = re.compile(
    r"prefill (\d+) tok ([\d.]+) ms \| decode (\d+) tok [\d.]+ ms \(([\d.]+) tok/s\)")
PY_RE = re.compile(r"TTFT median ([\d.]+) ms, decode median ([\d.]+) tok/s")


def parse_cpp(path: Path) -> dict | None:
    text = path.read_text()
    mv, ml = VISION_RE.search(text), LLM_RE.search(text)
    if not mv or not ml:
        return None
    d = dict(zip(STAGES[:6], map(float, mv.groups()[:6])))
    d["prefill"] = float(ml.group(2))
    d["decode_tps"] = float(ml.group(4))
    if mm := re.search(r"dev-mem ([\d.]+) GiB", text):
        d["dev_mem_gib"] = float(mm.group(1))
    return d


def main() -> int:
    CHARTS.mkdir(parents=True, exist_ok=True)
    matrix: dict[str, dict] = {}
    for log in sorted((RESULTS / "matrix").glob("*.time.log")):
        name = log.name.removesuffix(".time.log")
        if name.startswith("python"):
            if m := PY_RE.search(log.read_text()):
                matrix[name] = {"TTFT": float(m.group(1)),
                                "decode_tps": float(m.group(2))}
            continue
        if d := parse_cpp(log):
            d["TTFT"] = sum(d.get(s, 0.0) for s in STAGES)
            matrix[name] = d

    (RESULTS / "latency_matrix.json").write_text(json.dumps(matrix, indent=2))

    cpp_rows = [k for k in matrix if not k.startswith("python")]
    dev_rank = lambda k: 0 if k.startswith("cuda") else 1 if k.startswith("vulkan") else 2
    cpp_rows.sort(key=lambda k: (dev_rank(k), not k.endswith("f16")))
    labels = {"cuda": "CUDA", "vulkan": "Vulkan", "cpu": "CPU"}
    prec_of = lambda k: "f16" if k.endswith("f16") else "q8_0"
    row_label = lambda k: ("python bf16 (CUDA)" if k.startswith("python")
                           else f"{labels.get(k.rsplit('_', 1)[0], k)} {prec_of(k)}")
    py_key = next((k for k in matrix if k.startswith("python")), None)
    py_ttft = matrix.get(py_key, {}).get("TTFT", 0.0)
    py_tps = matrix.get(py_key, {}).get("decode_tps", 0.0)

    # ---------- panel 1: end-to-end TTFT, every config + python ----------
    ttft_rows = [(row_label(k), matrix[k]["TTFT"], "#4c72b0") for k in cpp_rows]
    if py_key:
        ttft_rows.append(("python bf16 (CUDA)", py_ttft, "#b0b0b0"))
    ttft_rows.sort(key=lambda r: r[1])
    fig, (ax, axm, ax2) = plt.subplots(
        1, 3, figsize=(17.5, 0.58 * len(cpp_rows) + 2.6),
        gridspec_kw={"width_ratios": [1.25, 1.6, 0.85]})
    ys = range(len(ttft_rows))
    bars = ax.barh(list(ys), [r[1] for r in ttft_rows],
                   color=[r[2] for r in ttft_rows])
    ax.set_yticks(list(ys), [r[0] for r in ttft_rows], fontsize=9)
    ax.invert_yaxis()
    for y, (_, v, _c) in zip(ys, ttft_rows):
        mult = f"  ({v / py_ttft:.2f}x py)" if py_ttft and _c != "#b0b0b0" else ""
        ax.text(v + max(v for _x, v, _c in ttft_rows) * 0.015, y,
                f"{v:.0f} ms{mult}", va="center", fontsize=8)
    ax.axvline(py_ttft, color="#777777", ls="--", lw=1.2)
    ax.set_xlabel("TTFT = vision + prefill (ms, steady state)")
    ax.set_title("end-to-end first-token latency\n(demo_image + 100 official boxes, greedy)")
    ax.set_xlim(0, max(v for _x, v, _c in ttft_rows) * 1.28)

    # ---------- panel 2: per-stage heatmap (cpp configs + python row) ----------
    # The python row comes from a live per-stage instrumented run
    # (results/matrix/python_stages.json: register_forward hooks around
    # vision_tower.visual / vision_tower_aux / object_vp_extractor.simple_fpn
    # / mm_projector_aux / language_model; hfre = ovp - simple_fpn; prefill
    # = LLM pre/post, by subtraction of the measured stages).
    stage_names = ["prep", "mmproj", "fpn", "radio", "hfre", "proj", "vision Σ", "prefill"]
    heat_keys = cpp_rows + ([py_key] if py_key else [])
    py_stages = {}
    py_log = RESULTS / "matrix" / "python_stages.json"
    if py_log.exists():
        py_stages = json.loads(py_log.read_text()).get("stages", {})
    data = np.full((len(heat_keys), len(stage_names)), np.nan)
    for i, k in enumerate(heat_keys):
        if k.startswith("python"):
            for j, s in enumerate(STAGES[:6]):
                data[i, j] = py_stages.get(s, np.nan)
            data[i, 6] = sum(v for v in data[i, :6] if not np.isnan(v))
            data[i, 7] = py_stages.get("prefill", np.nan)
            continue
        d = matrix[k]
        for j, s in enumerate(STAGES[:6]):
            data[i, j] = d.get(s, 0.0)
        data[i, 6] = sum(d.get(s, 0.0) for s in STAGES[:6])
        data[i, 7] = d.get("prefill", 0.0)
    im = axm.imshow(np.ma.masked_invalid(data), cmap="YlOrRd", aspect="auto")
    axm.set_facecolor("#e4e4e4")
    axm.set_xticks(range(data.shape[1]),
                   [*stage_names[:6], "vision Σ", "prefill"], rotation=30, ha="right")
    axm.set_yticks(range(len(heat_keys)),
                   [row_label(k) for k in heat_keys], fontsize=9)
    for i, k in enumerate(heat_keys):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                axm.text(j, i, "—", ha="center", va="center", fontsize=8,
                         color="#777777")
                continue
            axm.text(j, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                     ha="center", va="center", fontsize=8,
                     color="black" if v < np.nanmax(data) * 0.55 else "white")
    axm.set_title("stage breakdown (ms, warm median;\n"
                  "python row = live per-stage instrumented run)")
    fig.colorbar(im, ax=axm, shrink=0.8, label="ms")

    # ---------- panel 3: decode ----------
    dec_rows = [(row_label(k), matrix[k]["decode_tps"], "#4c72b0") for k in cpp_rows]
    if py_key:
        dec_rows.append(("python bf16 (CUDA)", py_tps, "#b0b0b0"))
    dec_rows.sort(key=lambda r: r[1])
    bars = ax2.barh(range(len(dec_rows)), [r[1] for r in dec_rows],
                    color=[r[2] for r in dec_rows])
    ax2.set_yticks(range(len(dec_rows)), [r[0] for r in dec_rows], fontsize=9)
    ax2.invert_yaxis()
    for y, (_, v, _c) in zip(range(len(dec_rows)), dec_rows):
        ax2.text(v + max(r[1] for r in dec_rows) * 0.02, y, f"{v:.1f}",
                 va="center", fontsize=8)
    ax2.set_xlabel("decode tok/s")
    ax2.set_title("decode")
    ax2.set_xlim(0, max(r[1] for r in dec_rows) * 1.2)

    fig.suptitle("VLX-Seek 1.5-10B latency matrix — C++ GGML backends × precision vs official python "
                 "(RTX 4090, greedy)", y=1.0)
    fig.tight_layout()
    fig.savefig(CHARTS / "latency_matrix.png", dpi=150)
    print(f"matrix -> results/latency_matrix.json, charts/latency_matrix.png "
          f"({len(cpp_rows)} cpp configs + python reference)")
    return 0


if __name__ == "__main__":
    main()
