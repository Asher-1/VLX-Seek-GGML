#!/usr/bin/env python3
"""Latency and decode charts from the C++ [time] logs + the python wall times.

  charts/latency_ttft_bar.png   stacked vision stages + prefill, warm medians,
                                with the python TTFT as the reference line
  charts/decode_tok_s.png       decode throughput
Numbers come from results/cpp_*/<case>.time.log (parsed fresh) and the
published d11 alignment table for the python reference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
CHARTS = BENCH / "charts"

STAGES = ("prep", "mmproj", "fpn", "radio", "hfre", "proj")
COLORS = {"prep": "#8ea9c9", "mmproj": "#4c72b0", "fpn": "#64b5cd",
          "radio": "#55a868", "hfre": "#c44e52", "proj": "#8172b2"}

# python reference (RTX 4090, detection, 100 fixed boxes, greedy — d11 table)
PY_TTFT_MS = 126.0
PY_DECODE_TPS = 35.7


def parse_time_log(path: Path) -> dict:
    vision_re = re.compile(
        r"vision warm median of \d+: prep ([\d.]+) \| mmproj ([\d.]+) \| fpn ([\d.]+) "
        r"\| radio ([\d.]+) \| hfre ([\d.]+) \| proj ([\d.]+) \| total ([\d.]+) ms")
    llm_re = re.compile(
        r"llm load [\d.]+ ms \| prefill (\d+) tok ([\d.]+) ms \| decode (\d+) tok "
        r"([\d.]+) ms \(([\d.]+) tok/s\).*dev-mem ([\d.]+) GiB")
    out = {}
    text = path.read_text()
    if m := vision_re.search(text):
        out["vision_stages"] = dict(zip(STAGES, map(float, m.groups()[:6])))
        out["vision_total"] = float(m.group(7))
    if m := llm_re.search(text):
        out["prefill_tok"] = int(m.group(1))
        out["prefill_ms"] = float(m.group(2))
        out["decode_tok"] = int(m.group(3))
        out["decode_ms"] = float(m.group(4))
        out["decode_tps"] = float(m.group(5))
        out["dev_mem_gib"] = float(m.group(6))
    return out


def main() -> int:
    CHARTS.mkdir(parents=True, exist_ok=True)
    logs = {}
    for backend in ("cpp_f16", "cpp_q8_0"):
        per_case = {}
        for log in sorted((RESULTS / "fixed" / backend).glob("*.time.log")):
            if d := parse_time_log(log):
                per_case[log.name.removesuffix(".time.log")] = d
        logs[backend] = per_case

    # ---- TTFT: stacked steady-state vision + prefill, median across cases ----
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    names = []
    for backend in ("cpp_f16", "cpp_q8_0"):
        cases = list(logs[backend].values())
        if not cases:
            continue
        med = {s: sorted(c["vision_stages"][s] for c in cases)[len(cases) // 2]
               for s in STAGES}
        prefill = sorted(c["prefill_ms"] for c in cases)[len(cases) // 2]
        bottom = 0.0
        for stage in STAGES:
            ax.bar(backend, med[stage], bottom=bottom, color=COLORS[stage],
                   label=stage if backend == "cpp_f16" else None)
            bottom += med[stage]
        ax.bar(backend, prefill, bottom=bottom, color="#dd8452",
               label="prefill" if backend == "cpp_f16" else None)
        names.append(f"{backend}\n(vision warm median + prefill)")
    ax.axhline(PY_TTFT_MS, color="#2f2f2f", ls="--", lw=1.4,
               label=f"python bf16 TTFT {PY_TTFT_MS:.0f} ms (incl. vision+prefill)")
    ax.set_ylabel("ms (steady state)")
    ax.set_title("VLX-Seek 1.5-10B first-token latency: C++ GGML vs official python\n"
                 "RTX 4090, cuda, greedy, median over benchmark cases")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHARTS / "latency_ttft_bar.png", dpi=150)

    # ---- decode throughput ----
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    labels, vals = [], []
    for backend in ("cpp_f16", "cpp_q8_0"):
        cases = [c["decode_tps"] for c in logs[backend].values() if "decode_tps" in c]
        if cases:
            labels.append(backend)
            vals.append(sorted(cases)[len(cases) // 2])
    labels.append("python bf16")
    vals.append(PY_DECODE_TPS)
    bars = ax.bar(labels, vals, color=["#4c72b0", "#64b5cd", "#b0b0b0"][:len(vals)])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}",
                ha="center", fontsize=10)
    ax.set_ylabel("tokens / s")
    ax.set_title("decode throughput (greedy, median over cases)")
    fig.tight_layout()
    fig.savefig(CHARTS / "decode_tok_s.png", dpi=150)

    print(f"charts -> {CHARTS} (latency_ttft_bar.png, decode_tok_s.png)")
    return 0


if __name__ == "__main__":
    main()
