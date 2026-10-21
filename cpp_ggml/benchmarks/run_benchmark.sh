#!/usr/bin/env bash
# One-click C++/python full-chain benchmark, TWO modes:
#
#   fixed  — the SAME official WeDetect-CUDA proposals are fed to both backends
#            (--bbox-list). This isolates detector implementation noise and is
#            the parity criterion: vision + LLM must be token-identical.
#   closed — no --bbox-list: each backend runs its own WeDetect path (the
#            official one is torch-CUDA, hard-coded .cuda()). The proposal SET
#            itself is device-dependent (torch CUDA vs torch CPU overlap is
#            only 32/100 on demo_image2), so closed-loop answers may differ in
#            <objN> numbering even though the vision+LLM chain is identical.
#            This mode shows the real user path, not a parity verdict.
#
#   bash cpp_ggml/benchmarks/run_benchmark.sh            # both modes, cuda
#   MODE=fixed bash cpp_ggml/benchmarks/run_benchmark.sh
#
# Outputs: results/<mode>/<backend>/<case>.json (+ .time.log), charts/*.png,
# results/proposals/*.json, results/alignment.json (fixed mode).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda}"
BUILD="${BUILD:-cpp_ggml/build-cuda}"
VLX_CLI="$BUILD/vlx-cli"
MODE="${MODE:-both}"   # fixed | closed | both
[ -x "$VLX_CLI" ] || { echo "missing $VLX_CLI (build it first)"; exit 1; }

PY=python3
[ -x .venv-cuda/bin/python ] && PY=.venv-cuda/bin/python

# ---- official proposals once (python CUDA wedetect, the shipped runtime) ----
mkdir -p cpp_ggml/benchmarks/results/proposals
if [ "$MODE" != "closed" ]; then
    PYTHONNOUSERSITE=1 "$PY" - <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "cpp_ggml/benchmarks")
from PIL import Image
from inference import load_wedetect_proposals
from cases import CASES
out = Path("cpp_ggml/benchmarks/results/proposals")
seen = set()
for image_name, task, _t, _l in CASES:
    stem = Path(image_name).stem
    if task == "vqa" or stem in seen:
        continue
    img = Image.open(f"cpp_ggml/benchmarks/images/{image_name}").convert("RGB")
    boxes = load_wedetect_proposals(img, "resources/wedetect_base_uni.pth")
    (out / f"{stem}.json").write_text(json.dumps(boxes))
    seen.add(stem)
    print(f"proposals {stem}: {len(boxes)} boxes")
EOF
fi

run_case () {  # mode backend prec image task text lang bbox_args...
    local mode=$1 backend=$2 prec=$3 image=$4 task=$5 text=$6 lang=$7; shift 7
    local key="${image%.*}_$task"
    mkdir -p "cpp_ggml/benchmarks/results/$mode/$backend"
    if [ "$backend" = python ]; then
        return 0   # python runs in one batch below
    fi
    "$VLX_CLI" run --device "$DEVICE" --precision "$prec" --warm-runs 3 \
        --no-visualize --image-path "cpp_ggml/benchmarks/images/$image" \
        --task "$task" --lang "$lang" --text "$text" "$@" \
        > "cpp_ggml/benchmarks/results/$mode/$backend/$key.json" \
        2> "cpp_ggml/benchmarks/results/$mode/$backend/$key.time.log"
    echo "[$mode/$backend] $key done"
}

CASES_TSV=$("$PY" - <<'EOF'
import sys
sys.path.insert(0, "cpp_ggml/benchmarks")
from cases import CASES
for image, task, text, lang in CASES:
    print(f"{image}|{task}|{text}|{lang}")
EOF
)

run_mode () {  # mode bbox_dir
    local mode=$1 bbox_dir=$2
    while IFS='|' read -r image task text lang; do
        local bbox_args=()
        if [ -n "$bbox_dir" ] && [ "$task" != vqa ]; then
            local stem="${image%.*}"
            bbox_args=(--bbox-list "$(cat "cpp_ggml/benchmarks/results/proposals/$stem.json")")
        fi
        run_case "$mode" cpp_f16 f16 "$image" "$task" "$text" "$lang" ${bbox_args+"${bbox_args[@]}"}
        run_case "$mode" cpp_q8_0 q8_0 "$image" "$task" "$text" "$lang" ${bbox_args+"${bbox_args[@]}"}
    done <<< "$CASES_TSV"
    MODE="$mode" PROPOSALS_DIR="$bbox_dir" PYTHONNOUSERSITE=1 "$PY" cpp_ggml/benchmarks/run_python.py \
        --out "cpp_ggml/benchmarks/results/$mode/python" ${bbox_dir:+--proposals-dir "$bbox_dir"}
}

if [ "$MODE" != "closed" ]; then
    echo "=== fixed-boxes mode (alignment criterion) ==="
    run_mode fixed cpp_ggml/benchmarks/results/proposals
fi
if [ "$MODE" != "fixed" ]; then
    echo "=== closed-loop mode (real user path) ==="
    run_mode closed ""
fi

echo "=== alignment matrix + official-style side-by-sides ==="
# a non-perfect alignment must not abort the charts below
PYTHONNOUSERSITE=1 "$PY" cpp_ggml/benchmarks/visualize_cases.py || true

echo "=== latency / decode charts ==="
PYTHONNOUSERSITE=1 "$PY" cpp_ggml/benchmarks/plot_benchmarks.py

echo "done: cpp_ggml/benchmarks/{charts,results}"
