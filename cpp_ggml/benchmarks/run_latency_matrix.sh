#!/usr/bin/env bash
# Multi-device × multi-precision latency matrix on the standard gate case
# (demo_image + the official fixed proposals, warm-runs 3):
#   {cuda, vulkan, cpu} × {f16, q8_0}  +  python (cuda bf16) as the reference.
#
#   bash cpp_ggml/benchmarks/run_latency_matrix.sh
#
# Outputs results/latency_matrix.json and charts/latency_matrix.png.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

PY=python3
[ -x .venv-cuda/bin/python ] && PY=.venv-cuda/bin/python

BOXES="$(cat cpp_ggml/benchmarks/results/proposals/demo_image.json)"
mkdir -p cpp_ggml/benchmarks/results/matrix

for dev in cuda vulkan cpu; do
    case "$dev" in
        cuda)   BUILD=cpp_ggml/build-cuda ;;
        vulkan) BUILD=cpp_ggml/build-vulkan ;;
        cpu)    BUILD=cpp_ggml/build-vlx ;;
    esac
    VLX_CLI="$BUILD/vlx-cli"
    [ -x "$VLX_CLI" ] || { echo "skip $dev: $VLX_CLI missing"; continue; }
    for prec in f16 q8_0; do
        echo "=== $dev / $prec ==="
        "$VLX_CLI" run --device "$dev" --precision "$prec" --warm-runs 3 \
            --no-visualize --image-path cpp_ggml/benchmarks/images/demo_image.jpg \
            --task detection --lang en --text "orange; apple" \
            --bbox-list "$BOXES" \
            > "cpp_ggml/benchmarks/results/matrix/${dev}_${prec}.json" \
            2> "cpp_ggml/benchmarks/results/matrix/${dev}_${prec}.time.log"
    done
done

echo "=== python reference (cuda bf16) ==="
PYTHONNOUSERSITE=1 "$PY" - <<'EOF'
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, ".")
from PIL import Image
from vlx_seek_worker import VLXSeekWorker
boxes = json.loads(Path("cpp_ggml/benchmarks/results/proposals/demo_image.json").read_text())
worker = VLXSeekWorker("resources/VLX-Seek-1.5-10B", device="cuda")
image = Image.open("cpp_ggml/benchmarks/images/demo_image.jpg").convert("RGB")
worker.run_task(image, "detection", "orange; apple", bbox_list=boxes,
                max_new_tokens=2048, temperature=0.0)   # warm-up
ttft, tps = [], []
for _ in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = worker.run_task(image, "detection", "orange; apple", bbox_list=boxes,
                        max_new_tokens=1, temperature=0.0)
    torch.cuda.synchronize()
    ttft.append((time.perf_counter() - t0) * 1e3)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = worker.run_task(image, "detection", "orange; apple", bbox_list=boxes,
                        max_new_tokens=2048, temperature=0.0)
    torch.cuda.synchronize()
    full = (time.perf_counter() - t0) * 1e3
    tps.append((r["completion_tokens"] - 1) * 1e3 / (full - ttft[-1]))
ttft.sort(); tps.sort()
Path("cpp_ggml/benchmarks/results/matrix/python_cuda.time.log").write_text(
    f"[time] python reference: TTFT median {ttft[1]:.1f} ms, "
    f"decode median {tps[1]:.2f} tok/s\n")
print(f"python: TTFT {ttft[1]:.1f} ms, decode {tps[1]:.2f} tok/s")
EOF

PYTHONNOUSERSITE=1 "$PY" cpp_ggml/benchmarks/plot_latency_matrix.py
echo "done: results/latency_matrix.json, charts/latency_matrix.png"
