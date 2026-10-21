# VLX-Seek 1.5 推理指南

[English](inference.md) | 中文

本文介绍 VLX-Seek 1.5-10B 的命令行与 Python 推理方式。模型简介和安装方法请参阅[项目 README](../README_zh.md)。

## 环境要求

当前提供的推理流程需要：

- 推理依赖，可通过 `requirements.txt` 在本地安装，或使用下文提供的 Docker 镜像
- VLX-Seek 1.5-10B 模型权重：[omlab/VLX-Seek-1.5-10B](https://huggingface.co/omlab/VLX-Seek-1.5-10B)
- WeDetect-Base-Uni 模型权重：[WeDetect](https://huggingface.co/fushh7/WeDetect)

可直接通过 Hugging Face 模型名加载 VLX-Seek 主模型：

```bash
python inference.py --model-path omlab/VLX-Seek-1.5-10B ...
```

未提供 `--bbox-list` 时，程序会在需要时自动把 `wedetect_base_uni.pth` 下载到 `resources/wedetect_base_uni.pth`。

也可以将权重下载到本地默认目录：

```text
resources/
├── VLX-Seek-1.5-10B/
└── wedetect_base_uni.pth
```

`--model-path` 既支持 Hugging Face 模型标识符，也支持本地 checkpoint 目录。模型结构由本仓库中的 `vlx_seek` 包提供。

也可以通过 `--model-path` 和 `--detector-checkpoint` 指定其他路径。

### 可选：加速推理

模型使用了 Linear Attention 层。如果缺少对应的加速内核，会看到类似如下警告，并回退到较慢的 PyTorch 实现：

```text
The fast path is not available because one of the required library is not installed. Falling back to torch implementation.
```

安装 [flash-linear-attention](https://github.com/fla-org/flash-linear-attention#installation) 和 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 可以启用快速路径，从而提升推理速度。

## NVIDIA x86（如 A100）Docker 推理

支持在搭载 A100 等 NVIDIA GPU 的 Linux x86_64 主机上使用 Docker 推理。宿主机需要安装 NVIDIA 驱动、Docker，并配置 NVIDIA Container Toolkit 以供容器访问 GPU。镜像已包含推理依赖，以及 `flash-linear-attention` 和 `causal-conv1d` 加速库。

宿主机 NVIDIA 驱动要求：

| x86 GPU 类型 | 最低 Driver Version |
| --- | --- |
| A100 等数据中心卡 | 535 及以上 |
| RTX 4090 等消费级卡 | 580 及以上 |

### 准备镜像

如果尚未克隆仓库，先执行以下命令。后续宿主机命令均在仓库根目录运行：

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git
cd VLX-Seek
```

任选一种方式准备镜像：

- 使用 [Dockerfile.nvidia_x86](../docker/Dockerfile.nvidia_x86) 构建：

  ```bash
  docker build -f docker/Dockerfile.nvidia_x86 -t vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64 .
  ```

- 拉取我们提供的预构建镜像：

  ```bash
  docker pull registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64
  ```

### 启动并进入容器

在仓库根目录执行以下命令，通过 `--gpus device=0` 选择宿主机 GPU `0`，可修改编号使用其他 GPU。如使用本地构建的镜像，将 `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64` 替换为 `vlx-seek:py3.12-ubuntu24.04-nvidia-x86_64`。

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

上述命令将仓库挂载到 `/workspace` 并将其设为工作目录，宿主机上的源码、本地权重、输入图片和保存的结果均可在容器内访问。容器会持续运行，后续可通过 `docker exec -it seek-dev bash` 再次进入。

### 在容器内执行推理

```bash
python inference.py \
  --model-path omlab/VLX-Seek-1.5-10B \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

首次运行需要访问 Hugging Face，以下载主模型，并在需要候选框时下载 `resources/wedetect_base_uni.pth`。如使用本地主模型权重，将其放在宿主机仓库的 `resources/VLX-Seek-1.5-10B/` 目录，再将参数改为 `--model-path resources/VLX-Seek-1.5-10B`。Docker 镜像不包含模型权重。

结果以 JSON 打印，示例可视化保存到宿主机挂载目录中的 `demo/demo_image_result.png`。下文其他命令行示例和 Python API 也可在该容器的 `/workspace` 目录下运行。

## T5000 和 GB10（aarch64）Docker 推理

宿主机平台要求：

| 平台 | 要求 |
| --- | --- |
| Jetson 系列（T5000） | JetPack 7.0 及以上，建议使用 JetPack 7.2 |
| NVIDIA Spark GB10 | NVIDIA Driver Version 580 及以上 |

根据 aarch64 平台选择对应的预构建镜像：

| 平台 | 镜像 |
| --- | --- |
| T5000 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-t5000-aarch64` |
| GB10 | `registry.linker.cc/vlx/vlx-seek:py3.12-ubuntu24.04-gb10-aarch64` |

宿主机需要安装 NVIDIA 驱动、Docker 及 NVIDIA Container Toolkit，并配置 `nvidia` runtime。按上文克隆仓库，以下宿主机命令均在仓库根目录运行。

### T5000 Docker Run

在仓库根目录启动并进入容器：

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

在仓库根目录启动并进入容器：

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

本地没有镜像时，`docker run` 会自动拉取对应镜像。两条命令均将仓库挂载到 `/workspace`，将容器端口 `8000` 映射到宿主机端口 `1233`，并使用容器名 `seek-dev`。切换平台前请先停止并移除同名容器。

进入容器后，在 `/workspace` 下执行 NVIDIA x86 小节中的推理命令。模型下载、本地权重路径及结果路径的使用方式与上文相同。

## 候选区域

大多数区域级任务需要先准备候选框。受公司政策限制，我们无法开源blog中使用的内部训练 OPN。本仓库集成开源、轻量的 **WeDetect-Base-Uni** 检测器，作为候选区域生成的替代方案。

有两种方式可以提供 proposals：

1. 不传 `--bbox-list`，由命令行程序加载 WeDetect-Uni 并生成 100 个候选框。
2. 使用任意检测器生成候选框，再通过 `--bbox-list` 传入。

自定义候选框应使用原图像素坐标，格式为 `[x1, y1, x2, y2]`：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]'
```

模型最多支持 100 个目标区域 token，因此 proposals 数量应不超过 100。

## 命令行用法

### 开放目标检测

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task detection \
  --text "橘子; 苹果" \
  --lang zh
```

多个检测类别使用分号分隔。使用内置中文任务模板时，请添加 `--lang zh`。

### 指代表达定位

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task grounding_single \
  --text "穿白色上衣的人" \
  --lang zh
```

### 目标计数

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task counting \
  --text "橘子" \
  --lang zh
```

### 推理检测

```bash
python inference.py \
  --image-path demo/demo_image2.jpg \
  --task reasoning_detection \
  --text "穿着红色衣服的女人" \
  --lang zh
```

### 区域描述

区域描述需要同时提供 proposals 和一个或多个从零开始的区域索引：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task brief_region_caption \
  --bbox-list '[[x1, y1, x2, y2], [x1, y1, x2, y2]]' \
  --target-region-indexes '[1]' \
  --lang zh
```

索引对应 `--bbox-list` 中的输入顺序。

### 区域 OCR

```bash
python inference.py \
  --image-path path/to/image.jpg \
  --task region_ocr \
  --bbox-list '[[40, 60, 360, 180]]' \
  --target-region-indexes '[0]' \
  --lang zh
```

### 通用 VQA

VQA 直接处理整幅图像，不会加载候选区域模型：

```bash
python inference.py \
  --image-path demo/demo_image.jpg \
  --task vqa \
  --text "图片中有哪些水果？"
```

## 支持的任务

| 任务 | `--text` 内容 | 是否需要 proposals | 额外要求 |
| --- | --- | --- | --- |
| `detection` | 一个或多个以 `;` 分隔的类别或描述 | 是 | 无 |
| `grounding_single` | 指代表达 | 是 | 无 |
| `counting` | 目标描述 | 是 | 无 |
| `reasoning_detection` | 目标描述或类别 | 是 | 无 |
| `brief_region_caption` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `region_ocr` | 根据选中区域自动生成 | 是 | `--target-region-indexes` |
| `vqa` | 自由形式问题 | 否 | 无 |

对于依赖 proposals 的任务，如果省略 `--bbox-list`，程序会自动调用候选框检测器生成。

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model-path` | `resources/VLX-Seek-1.5-10B` | 本地权重目录或 Hugging Face 模型标识符，例如 `omlab/VLX-Seek-1.5-10B` |
| `--image-path` | `demo/demo_image.jpg` | 输入图像 |
| `--lang` | `en` | 内置任务模板语言：`en` 或 `zh` |
| `--bbox-list` | 无 | 像素坐标 proposals 的 JSON 列表 |
| `--detector-checkpoint` | `resources/wedetect_base_uni.pth` | 未提供 proposals 时使用的 WeDetect-Uni 权重；默认路径缺失时会自动下载 |
| `--target-region-indexes` | 无 | 区域描述或 OCR 选中的 proposal 索引 JSON 列表 |
| `--max-new-tokens` | `2048` | 最大生成长度 |
| `--temperature` | `0.0` | 采样温度；大于零时启用采样。建议检测相关任务均使用温度 `0.0` |
| `--no-visualize` | 关闭 | 禁用检测结果可视化 |
| `--visualization-output` | `<image_stem>_result.png` | 自定义可视化输出路径 |

运行 `python inference.py --help` 可以查看完整命令行参数。

## 输出格式

命令最终会以格式化 JSON 打印结果字典：

```json
{
  "answer": "<ground>橘子</ground><objects><obj0><obj2></objects>",
  "result_bbox_list": [
    {
      "object_index": "<obj0>",
      "xmin": 12.0,
      "ymin": 20.0,
      "xmax": 180.0,
      "ymax": 210.0,
      "label": "橘子"
    }
  ],
  "prompt_tokens": 1234,
  "completion_tokens": 16,
  "visualization_path": "demo/demo_image_result.png"
}
```

- `answer` 是模型的原始回答。
- `result_bbox_list` 将生成的 `<objN>` 引用解析为输入 proposal 坐标。
- `prompt_tokens` 和 `completion_tokens` 是输入与输出 token 数量。
- 对选中目标进行可视化时，结果中会包含 `visualization_path`。

VQA、区域描述或 OCR 可以返回自由文本，其 `result_bbox_list` 可能为空。

## C++（GGML）后端

仓库同时提供全链路的纯 C++17/GGML 移植（WeDetect-Uni 候选框 → Qwen3.5-VL 视觉塔 → C-RADIOv4 辅助塔 + HFRE → 经 llama.cpp 的 Qwen3.5 LLM）。它打印与 `inference.py` 相同的 JSON：在 detection 示例上生成 token 逐字一致，候选框坐标仅相差 fp32 舍入级误差。构建方法与 parity gate 协议见 [cpp_ggml/README.md](../cpp_ggml/README.md)。

### 一键启动

`run_vlx.sh` 用相同的参数把请求分发到任一后端：

```bash
./run_vlx.sh --backend cpp --device cuda --task detection --text "orange; apple"
./run_vlx.sh --backend python --device cuda --task detection --text "orange; apple"
```

`--backend cpp` 要求二进制位于 `cpp_ggml/build-{vlx,cuda,vulkan}`（每个 `--device cpu|cuda|vulkan` 对应一个构建目录；python 后端没有 vulkan）。两个后端读取相同的本地模型文件：C++ 后端从 `cpp_ggml/models/gguf/` 加载转换后的 GGUF，通过 `--precision f16|q8_0` 选择精度档（默认 f16；f32 不随仓库分发，档位政策与重建方法见 [cpp_ggml/models/MODEL_CARD.md](../cpp_ggml/models/MODEL_CARD.md)）。不传 `--bbox-list` 时两个后端都会先跑 WeDetect-Uni——C++ 后端里这是原生的 GGML 卷积图（从其 f32 GGUF 加载，即官方 `.pth` 的精度），不是 Python 调用。

### 各精度档的数值判据

C++ 移植针对官方 python 流水线做门控（`cpp_ggml/scripts/compare_official_parity.py`），判据有两条：

- **token 一致（硬约束，所有精度档）。** 贪心解码必须与官方版本逐 token 相同，即打印的 JSON（`answer`、`result_bbox_list`、token 数）逐字一致。只要翻转一个 token 就判失败，无论浮点误差多小。
- **三方噪声比（f32 / f16 档）。** 官方发布本身是 bf16，即它本身就是带噪声的模型估计。以官方 f32 运行为裁判，f32/f16 C++ 构建在各阶段的噪声与官方 bf16 噪声之比满足 `ratio <= 1.0` 才通过：`||cpp − f32|| / ||bf16 − f32|| <= 1.0`。

`q8_0` 是独立的量化档：其量化噪声明显超过 bf16 表示误差，因此它受 token 一致硬约束加一个更宽松的三方判据带 `ratio <= 2.5` 约束。实践中 q8_0 在所有后端（cpu/cuda/vulkan）上均保持 token 一致，各阶段 ratio 在 0.8–2.0 之间。

WeDetect 候选框始终从 f32 GGUF 运行（与官方 `.pth` 精度一致）：q8_0 会改变候选框集合，f16 的亚像素偏移也可能重排接近并列的 raster 排序键——候选框集合直接喂给 LLM，因此保持 f32。

### 计时方法与 warm 数据

首次运行延迟有很大冷启动成分（GGUF 读盘 + 解析、权重上传、cuBLAS/cuDNN 初始化）。`vlx-cli run --warm-runs N` 会在权重与计算图常驻的情况下把视觉各阶段再跑 N 次，报告这些轮次的中位数，从而剥离冷启动得到稳态计算耗时；`cpp_ggml/scripts/bench_official.py` 用相同方式对 python 后端做 warm。

在 RTX 4090 上实测（detection 任务、100 个固定候选框、贪心解码、19 token 输出）：

| 后端 / 精度 | vision（warm 中位数） | prefill | TTFT（vision+prefill） | decode | e2e（warm） | 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| python bf16（官方） | — | — | 107.3 ms | 44.0 tok/s | ~520 ms | 19.5 GiB |
| cpp f16 | 81.9 ms | 74.8 ms | 156.7 ms（1.46×） | 54.6 tok/s | ~487 ms | 15.5 GiB |
| cpp q8_0 | 67.0 ms | 44.2 ms | 111.2 ms（1.04×） | 93.2 tok/s | ~303 ms | 8.6 GiB |

cpp q8_0 行已包含消费 GPU 常驻特征的融合 CUDA RoIAlign（hfre 38 → 约 19 ms）——decode 达官方后端的 2.1 倍。f16/q8_0 的 e2e 列为 TTFT + 18/decode 推算；python 列按其实测 decode 推算。

冷启动分解（cpp f16）：GGUF 读盘 + 解析约 2.2 s，首轮权重上传 + 内核初始化约 0.5 s（RADIO 塔占大头），稳态计算约 0.12 s。使用 `--warm-runs 3` 时报告的中位数不含以上任何一项。

## Python API

`VLXSeekWorker` 可以在多次请求之间保持模型常驻。候选区域生成由命令行入口负责，因此 Python API 调用方需要为区域级任务自行准备候选框。

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
    ["橘子", "苹果"],
    lang="zh",
)

vqa = worker.predict(
    image,
    "图片中有哪些水果？",
)
```

其他任务封装包括 `ground`、`count`、`reasoning_detect`、`describe_region` 和 `read_region_text`。候选框会被裁剪到输入图像边界，裁剪后必须保持有效的正面积。
