#!/usr/bin/env bash
# =============================================================================
# VLX-Seek 1.5 launcher: one argument set, two equivalent backends.
#
#   ./run_vlx.sh [--backend cpp|python] <inference.py arguments...>
#
# Both backends take the official inference.py arguments verbatim
# (--image-path --task --text --lang --bbox-list --target-region-indexes
#  --device --max-new-tokens --temperature --top-p --visualize/--no-visualize
#  --visualization-output) and print the same JSON. Without --bbox-list both
# run WeDetect-Uni first (cpp: wedetect-base-uni-f32.gguf, the .pth dtype).
#
# cpp backend: `vlx-cli run` from the build dir matching --device
#   (cpu -> cpp_ggml/build-vlx, cuda -> build-cuda, vulkan -> build-vulkan);
#   extra cpp-only flags: --precision f32|f16|q8_0 [f16] --gguf-dir D -ngl N
#   -t N --dump-dir D, and --build-dir DIR to override the binary location.
# python backend: inference.py under $PYTHON (default: .venv-cuda/bin/python
#   for --device cuda when present, else python3).
#
# --device defaults to cuda, as in inference.py.
#
# Examples:
#   ./run_vlx.sh --task detection --text "orange; apple"
#   ./run_vlx.sh --device vulkan --precision q8_0 --task counting --text person
#   ./run_vlx.sh --backend python --device cpu --task detection --text "orange; apple"
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="cpp"
BUILD_DIR=""
DEVICE="cuda"
ARGS=()

die() { echo "run_vlx.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --backend)   [ $# -ge 2 ] || die "--backend needs a value"; BACKEND="$2"; shift 2 ;;
        --build-dir) [ $# -ge 2 ] || die "--build-dir needs a value"; BUILD_DIR="$2"; shift 2 ;;
        --device)    [ $# -ge 2 ] || die "--device needs a value"; DEVICE="$2"; shift 2 ;;
        *)           ARGS+=("$1"); shift ;;
    esac
done
ARGS+=("--device" "$DEVICE")
cd "$HERE"   # both backends resolve demo/, resources/ and cpp_ggml/models/ from the repo root

case "$BACKEND" in
cpp)
    case "$DEVICE" in
        cpu)    BUILD_DIR="${BUILD_DIR:-cpp_ggml/build-vlx}" ;;
        cuda)   BUILD_DIR="${BUILD_DIR:-cpp_ggml/build-cuda}" ;;
        vulkan) BUILD_DIR="${BUILD_DIR:-cpp_ggml/build-vulkan}" ;;
        *)      die "--device must be cpu, cuda or vulkan (got '$DEVICE')" ;;
    esac
    VLX_CLI="$BUILD_DIR/vlx-cli"
    [ -x "$VLX_CLI" ] || die "$VLX_CLI not built. See cpp_ggml/README.md (cmake -S cpp_ggml -B $BUILD_DIR ...)"
    exec "$VLX_CLI" run "${ARGS[@]}"
    ;;
python)
    [ "$DEVICE" != "vulkan" ] || die "the python backend has no vulkan device (cpu|cuda)"
    if [ -z "${PYTHON:-}" ]; then
        PYTHON=python3
        [ "$DEVICE" = "cuda" ] && [ -x .venv-cuda/bin/python ] && PYTHON=.venv-cuda/bin/python
    fi
    # A user-site package built against a different numpy ABI breaks the venv
    # import chain (observed: ~/.local pandas 1.5.3 vs venv numpy 2.x ->
    # "numpy.dtype size changed" through transformers' optional sklearn
    # import). The project venv is self-contained; PYTHONNOUSERSITE= (empty)
    # restores user-site visibility.
    PYTHONNOUSERSITE="${PYTHONNOUSERSITE-1}" exec "$PYTHON" inference.py "${ARGS[@]}"
    ;;
*)
    die "--backend must be cpp or python (got '$BACKEND')"
    ;;
esac
