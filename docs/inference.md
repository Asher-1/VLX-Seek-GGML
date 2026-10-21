# VLX-Seek 1.5 Inference Guide

English | [中文](inference_zh.md)

This guide covers command-line and Python inference for VLX-Seek 1.5-10B. For the model overview and installation instructions, see the [project README](../README.md).

## Prerequisites

The provided pipeline currently requires:

- inference dependencies, installed locally with `requirements.txt` or provided by the Docker image described below;
- the VLX-Seek 1.5-10B checkpoint from [omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B);
- the region detector checkpoint from [WeDetect](https://huggingface.co/fushh7/WeDetect)

You can load the VLX-Seek main model directly from Hugging Face:

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

When `--bbox-list` is omitted, `wedetect_base_uni.pth` is automatically downloaded to `resources/wedetect_base_uni.pth` if needed.

Alternatively, place local copies in the default directory:

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` accepts either a Hugging Face model identifier or a local checkpoint directory. The model architecture is provided by the local `vlx_seek` package in this repository.

Alternatively, set paths with `--model-path` and `--detector-checkpoint`.

### Optional: Faster Inference

The model uses Linear Attention layers. Without the corresponding acceleration kernels, you will see a warning like the following and fall back to a slower PyTorch implementation:

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

Installing [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) enables the fast path and speeds up inference.

## Docker Inference on NVIDIA x86 (e.g. A100)

Use the provided Docker environment on a Linux x86_64 host with an NVIDIA GPU such as A100. The host needs an NVIDIA driver, Docker, and NVIDIA Container Toolkit configured for GPU containers. The image includes the inference dependencies, including `flash-linear-attention` and `causal-conv1d`.

Host NVIDIA driver requirements:

| x86 GPU type | Minimum Driver Version |
| --- | --- |
| Data center GPUs such as A100 | 535 or later |
| Consumer GPUs such as RTX 4090 | 580 or later |

### Prepare the Image

Clone the repository if needed, then run all host-side commands from its root directory:

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
```

Choose one of the following:

- Build with [Dockerfile.nvidia_x86](../docker/Dockerfile.nvidia_x86):

  ```bash
  docker build -f docker/Dockerfile.nvidia_x86 -t vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 .
  ```

- Pull our prebuilt image:

  ```bash
  docker pull registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64
  ```

### Start and Enter the Container

Run the following command from the repository root. It selects host GPU `0` through `--gpus device=0`; change the index to use another GPU. For a locally built image, replace `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64` with `vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64`.

```bash
docker run -d --name seek-dev \
  --gpus device=0 \
  --shm-size=40g \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 \
  sleep infinity
docker exec -it seek-dev bash
```

The command mounts the repository at `/workspace` and sets it as the working directory, making your host-side source code, local weights, input images, and saved results available inside the container. The container stays running so you can enter it again with `docker exec -it seek-dev bash`.

### Run Inference Inside the Container

```bash
python inference.py \
  --model-path omlab/VLX-Seek-1.5-10B \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

The first run requires access to Hugging Face to download the main model and, when proposals are needed, `resources/wedetect_base_uni.pth`. To use a local main-model checkpoint, place it in the host repository's `resources/VLX-Seek-1.5-10B/` directory and use `--model-path resources/VLX-Seek-1.5-10B` instead. The Docker image does not bundle model weights.

Results are printed as JSON, and the example visualization is saved to `demo/demo_image_result.png` in the mounted host repository. You can run the other CLI examples and Python API below from `/workspace` in the same container.

## Docker Inference on T5000 and GB10 (aarch64)

Host platform requirements:

| Platform | Requirement |
| --- | --- |
| Jetson series (T5000) | JetPack 7.0 or later; JetPack 7.2 recommended |
| NVIDIA Spark GB10 | NVIDIA Driver Version 580 or later |

Use the prebuilt image matching your aarch64 platform:

| Platform | Image |
| --- | --- |
| T5000 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64` |
| GB10 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64` |

The host needs the NVIDIA driver, Docker, and NVIDIA Container Toolkit with the `nvidia` runtime configured. Clone the repository as shown above and run all host-side commands from its root directory.

### T5000 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

### GB10 Docker Run

Run from the repository root to start and enter the container:

```bash
docker run -d --name seek-dev \
  --runtime=nvidia \
  --restart=always \
  --shm-size=40g \
  -p 1233:8000 \
  -v "$PWD:/workspace" \
  -w /workspace \
  registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64 \
  sleep infinity
docker exec -it seek-dev bash
```

`docker run` pulls the selected image if it is not available locally. Both commands mount the repository at `/workspace`, publish container port `8000` on host port `1233`, and use the container name `seek-dev`. Stop and remove an existing container with that name before switching platforms.

Run the inference command in the NVIDIA x86 section from `/workspace` inside the container. Model downloads, local weight paths, and output paths work the same way as described there.

## Region Proposals

Most region-level tasks first require candidate bounding boxes. Due to company policy, we are unable to release the internally trained OPN referenced in our blog. The repository integrates the open-source, lightweight **WeDetect-Base-Uni** detector as an alternative proposal generator.

There are two ways to provide proposals:

1. Omit `--bbox-list`. The CLI loads WeDetect-Uni and generates 100 proposals.
2. Use any detector of your choice and pass its proposals through `--bbox-list`.

Custom boxes must use original-image pixel coordinates in `[x1, y1, x2, y2]` format:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple" \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

The model supports up to 100 object-region tokens, so keep the proposal list at or below 100 boxes.

## Command-Line Usage

### Open-Vocabulary Detection

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "orange; apple"
```

Multiple detection categories are separated by semicolons. Add `--lang zh` when using the built-in Chinese task templates.

### Referring Expression Grounding

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "the person in a white shirt"
```

### Object Counting

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "orange"
```

### Reasoning Detection

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "the woman wearing red clothes"
```

### Region Captioning

Region captioning requires both proposals and one or more zero-based proposal indexes:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]'
```

The indexes refer to the order of the boxes supplied through `--bbox-list`.

### Region OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]'
```

### General VQA

VQA operates on the full image and does not load a proposal model:

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "What fruits are in the image?"
```

## Supported Tasks

| Task | `--text` | Proposals | Additional requirement |
| --- | --- | --- | --- |
| `detection` | One or more `;`-separated categories/descriptions | Required | None |
| `grounding_single` | A referring expression | Required | None |
| `counting` | A target description | Required | None |
| `reasoning_detection` | A target description or category | Required | None |
| `brief_region_caption` | Generated from selected regions | Required | `--target-region-indexes` |
| `region_ocr` | Generated from selected regions | Required | `--target-region-indexes` |
| `vqa` | A free-form question | Not used | None |

For proposal-dependent tasks, omitting `--bbox-list` automatically invokes the proposal detector.

## Common Options

| Option | Default | Description |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | Local checkpoint directory or Hugging Face model identifier, e.g. `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | Input image |
| `--lang` | `en` | Built-in task-template language: `en` or `zh` |
| `--bbox-list` | None | JSON list of pixel-coordinate proposal boxes |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | WeDetect-Uni checkpoint used when proposals are omitted; auto downloaded if the default path is missing |
| `--target-region-indexes` | None | JSON list of selected proposal indexes for captioning or OCR |
| `--max-new-tokens` | `2048` | Maximum completion length |
| `--temperature` | `0.0` | Sampling temperature; values above zero enable sampling. Temperature `0.0` is recommended for all detection-related tasks |
| `--no-visualize` | Off | Disable result-box visualization |
| `--visualization-output` | `<image_stem>_result.png` | Custom visualization path |

Run `python inference.py --help` for the complete CLI reference.

## Output

The command prints a result dictionary as formatted JSON:

```json
{
  "answer": "<ground>orange</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "orange"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` is the raw model response.
- `result_bbox_list` resolves generated `<objN>` references to the input proposal coordinates.
- `prompt_tokens` and `completion_tokens` report token counts.
- `visualization_path` is included when selected boxes are visualized.

VQA, captioning, or OCR answers may be free-form text and can have an empty `result_bbox_list`.

## C++ (GGML) Backend

The repository also ships a pure C++17/GGML port of the full pipeline
(WeDetect-Uni proposals → Qwen3.5-VL vision tower → C-RADIOv4 aux tower +
HFRE → Qwen3.5 LLM via llama.cpp). It prints the same JSON as `inference.py`:
for the detection demo the generated tokens are identical and proposal box
coordinates agree within fp32 rounding. Build instructions and the parity-gate
protocol live in [cpp_ggml/README.md](../cpp_ggml/README.md).

### One-command launcher

`run_vlx.sh` dispatches to either backend with the same arguments:

```bash
./run_vlx.sh --backend cpp --device cuda --task detection --text "orange; apple"
./run_vlx.sh --backend python --device cuda --task detection --text "orange; apple"
```

`--backend cpp` expects the built binaries under `cpp_ggml/build-{vlx,cuda,vulkan}`
(one build directory per `--device cpu|cuda|vulkan`; the python backend has no
vulkan). Both backends read the same local model files: the C++ backend loads
converted GGUF files from `cpp_ggml/models/gguf/` with `--precision f16|q8_0`
(f16 by default; f32 is not shipped — see
[cpp_ggml/models/MODEL_CARD.md](../cpp_ggml/models/MODEL_CARD.md) for the
tier policy and how to rebuild a f32 file). Without `--bbox-list` both
backends run WeDetect-Uni first — in the C++ backend this is a native GGML
conv graph loaded from its f32 GGUF (the official `.pth` dtype), not a Python
call.

### Numerical criteria per precision

The C++ port is gated against the official python pipeline
(`cpp_ggml/scripts/compare_official_parity.py`). Two criteria apply:

- **Token identity (hard constraint, every precision).** Greedy generation must
  emit the exact same tokens as the official release, so the printed JSON
  (`answer`, `result_bbox_list`, token counts) matches verbatim. A precision
  that flips even one token fails, regardless of how small the float-level
  drift is.
- **Three-way noise ratio (f32 / f16 tiers).** The official release is bf16,
  i.e. itself a noisy estimate of the model. With an official f32 run as the
  referee, an f32/f16 C++ build passes when its stage noise stays within
  `ratio <= 1.0` of the official bf16 noise:
  `||cpp − f32|| / ||bf16 − f32|| <= 1.0`.

`q8_0` is a separate quantized tier: its quantization noise measurably exceeds
the bf16 representation error, so it is held to the token-identity hard
constraint plus a looser three-way band, `ratio <= 2.5`. In practice q8_0
stays token-identical on all backends (cpu/cuda/vulkan) with per-stage ratios
around 0.8–2.0.

WeDetect proposals always run from the f32 GGUF (the official `.pth` dtype):
q8_0 changes the proposal box set, and even f16 sub-pixel shifts can reorder
near-tied raster sort keys — the box set feeds the LLM, so it stays f32.

### Timing methodology and warm numbers

First-run latency has a large cold-start component (GGUF reads + parse, weight
upload, cuBLAS/cuDNN init). `vlx-cli run --warm-runs N` re-runs the vision
stages N times with weights and graphs resident and reports the median of
those runs, isolating steady-state compute;
`cpp_ggml/scripts/bench_official.py` warms the python backend the same way.

Measured on RTX 4090, detection task, 100 fixed proposals, greedy, 19-token
completion:

| Backend / precision | vision (warm median) | prefill | TTFT (vision+prefill) | decode | e2e (warm) | device mem |
| --- | --- | --- | --- | --- | --- | --- |
| python bf16 (official) | — | — | 107.3 ms | 44.0 tok/s | ~520 ms | 19.5 GiB |
| cpp f16 | 81.9 ms | 74.8 ms | 156.7 ms (1.46×) | 54.6 tok/s | ~487 ms | 15.5 GiB |
| cpp q8_0 | 67.0 ms | 44.2 ms | 111.2 ms (1.04×) | 93.2 tok/s | ~303 ms | 8.6 GiB |

The cpp q8_0 row now includes the fused CUDA RoIAlign consuming
GPU-resident levels (hfre 38 → ~19 ms) — decode is 2.1x the official
backend. e2e columns for f16/q8_0 are TTFT + 18/decode; python's is
TTFT + 18/decode at its measured rate.

Cold-start decomposition (cpp f16): GGUF reads + parse ~2.2 s, first-run weight
upload + kernel init ~0.5 s (the RADIO tower dominates), steady-state compute
~0.12 s. With `--warm-runs 3` the reported median excludes all of it.

## Python API

`VLXSeekWorker` keeps the model loaded across requests. Proposal generation is handled by the CLI, so Python API callers should prepare their own boxes for region-level tasks.

```python
from PIL import Image

from vlx_seek_worker import VLXSeekWorker

worker = VLXSeekWorker(
    model_path="omlab/VLX-Seek-1.5-10B",
    device="cuda",
)
image = Image.open("demo/demo_image.jpg").convert("RGB")
boxes = [
    [12, 20, 180, 210],
    [190, 25, 430, 250],
]

detection = worker.detect(
    image,
    boxes,
    ["orange", "apple"],
    lang="en",
)

vqa = worker.predict(
    image,
    "What fruits are in the image?",
)
```

Other task wrappers are `ground`, `count`, `reasoning_detect`, `describe_region`, and `read_region_text`. Boxes are clipped to the input image bounds and must retain positive area after clipping.
