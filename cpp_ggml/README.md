# VLX-Seek C++ GGML inference (cpp_ggml)

Pure C++17 GGML port of the VLX-Seek 1.5 decoupled region-first pipeline:

```text
image ──► WeDetect-Base-Uni ──► 100 boxes ─┐
image ──► Qwen3.5-VL vision tower ──► image tokens ──► CAbstractor ──┐
image ──► C-RADIOv4 aux tower ──► 4-level features ──► HFRE ──────────┤──► Qwen3.5 LLM ──► <ground>..</ground><objects><objN>..</objects> ──► boxes
```

This directory currently contains the **vision sub-chain** (RADIOv4 graph, HFRE
region encoder, converters, parity tooling) developed standalone against a
pinned ggml, following the same engineering contract as the sibling
integrations (ultralytics-ggml / map-anything-ggml / sam-3d-objects-ggml).
The LLM sub-chain rides llama.cpp's already-merged `qwen3_5` support and is
wired in `vlx_cli.cpp` (see Roadmap).

## Status close-out (2026-10-18): full-chain C++ == python, both backends

The C++ backend is feature-complete and gated; everything below is kept as-is
as the reasoning trail.

* **Full automatic loop (d7)**: `vlx-cli run` without `--bbox-list` runs the
  native WeDetect conv graph (f32 GGUF) and feeds the 100 proposals into the
  vision + LLM chain. Verification vs the official pipeline (cuda, same
  arguments): identical greedy tokens, identical token counts (414/19), 7/7
  boxes, labels in the same order, max coordinate delta 0.0087 px (fp32
  reduction noise — the same class the d2 detect gate accepts).
* **Numerical criteria final (d10)**: token-identity is the hard constraint
  for every tier; f32/f16 pass the three-way noise band `ratio <= 1.0`; q8_0
  is a separate quantized tier with `ratio <= 2.5`. Documented for users in
  `docs/inference.md` / `docs/inference_zh.md` ("C++ (GGML) Backend").
* **Warm timing**: `vlx-cli run --warm-runs N` re-runs the vision
  stages with weights and graphs resident and reports the per-stage median,
  isolating cold start. Measured (RTX 4090, f16): steady-state vision
  81.9 ms (prep 3.9, mmproj 20.5, fpn 4.7, radio 20.3, hfre 25.4, proj 7.1)
  vs 613.5 ms on the cold first run; GGUF reads + parse add ~2.2 s. The cold
  excess splits into first-run weight upload + kernel init (~0.5 s, RADIO
  tower dominating) and disk/parse. Steady-state TTFT
  (vision+prefill) 156.7 ms vs the official python TTFT 107.3 ms = 1.46x
  (q8_0: 111.2 ms = 1.04x) — the "C++ vision steady-state <= 2x python" goal
  holds; q8_0 decode 93.2 tok/s is 2.1x the official backend. The hfre stage
  is a fused CUDA RoIAlign consuming GPU-resident levels (38 -> ~19 ms).
  Warm-run output is byte-identical to the cold run (graph reuse is
  deterministic). Full alignment table in docs/inference.md.
* **Launcher hardening**: the `run_vlx.sh` python branch defaults
  `PYTHONNOUSERSITE=1`: a user-site pandas built against the numpy 1.x ABI
  broke the transformers import chain through its guarded sklearn import
  (guard catches ImportError; the ABI clash raises ValueError and pierces
  it). `PYTHONNOUSERSITE=` (empty) restores user-site visibility.
* **Host-preprocess FMA pitfall** (recorded where it lives): torch's AA
  resize issues one FMA per tap; the C++ port spells exactly that tap as
  `std::fma` (image_prep.cpp) and compiles everything else with
  `-ffp-contract=off` (CMakeLists.txt) — a fused mul+add anywhere else flips
  bf16 ties downstream. See the comments in both files for the rationale.
* **Python env facts (2026-10-18, why vlx_seek/ carries three local fixes)**:
  the official `requirements.txt` pins `torch==2.10.0 / torchvision==0.25.0 /
  transformers==5.13.0 / flash_attn==2.8.3`, and the local venv matches the
  torch/transformers pins exactly. Two gaps remain that are NOT version
  drift: (a) `flash_attn==2.8.3` has no installable artifact for
  py3.10 + torch2.10 — the cp310 wheel matrix ends at `cu12torch2.8`
  (verified: that wheel imports with `undefined symbol:
  c10::cuda::c10_cuda_check_implementation` against torch 2.10; the
  torch2.10 wheels are cp312+cu13 only; the PyPI sdist needs nvcc, absent on
  this host) — so `attn_implementation="sdpa"` (an exact attention, d6
  token-identical) stays until either a CUDA toolkit provides nvcc for a
  source build or a cp312/cu13 venv is created; (b) torch 2.10 itself has no
  CPU bf16 antialiased-bilinear kernel (`NotImplementedError:
  compute_index_ranges_weights not implemented for BFloat16` — reproduced on
  the exact pinned version), while the CUDA kernel works; the aux-tower
  upcast fix therefore only affects the CPU path and is numerically
  neutral on CUDA (interpolate computes in f32 internally).

* **Full-repo parity audit (2026-10-18, post-d11)**: CLI argument surface,
  all 12 task templates (byte-level, the .inc stores UTF-8 as octal escapes),
  prompt assembly, box validation/ordering/remap, result parsing, stop
  criteria, JSON schema, and a live zh end-to-end run (answer token-identical,
  412/19 tokens) all match `inference.py`/`vlx_seek_worker.py`. Known
  deliberate/remaining deltas: (1) the C++ visualization draws rectangles but
  not the PIL `draw.text` label overlay — the PNG is not pixel-parity with
  the official `--visualize` output (no gate covers PNGs; fixing needs a
  font asset, stb_truetype, see P4 remaining list); (2) `temperature > 0`
  sampling streams differ (C++ mt19937(0) vs torch global RNG — greedy,
  the default and the recommended mode for all detection tasks, is exact);
  (3) worker-API-only knobs (`repetition_penalty`, `predict` defaults
  512/0.2) have no C++ counterpart — the CLI surface (`inference.py`:
  2048/0.0) is aligned; (4) `mm_bbox_order_mode` is raster-only in C++
  (the checkpoint config pins raster, so no behavioral gap); (5) the
  JPEG-decoder ±1-2 LSB delta (stb vs libjpeg/PIL) remains accepted.
* **Upstream GUI**: the repo ships no GUI entry point (verified by file
  search; `gradio` in requirements.txt is not backed by an in-repo app). The
  official GUI surfaces are external: the HF Space
  `hugging-apps/vlx-seek-1-5-10b-demo` (linked from the model repo) and the
  web demo at `om-agent.com/#/front` (the README "Try Now" badge).
* **d12 (2026-10-18, evening) — merged vision stack + benchmarks + app**:
  * **Vision merge**: `scripts/merge_vision_gguf.py` (vendored llama.cpp
    gguf-py, byte-level, 708 tensors verified identical) →
    `vlx-vision-10b-{f16,q8_0}.gguf` (2.02 / 1.35 GiB). Tensor namespaces were
    already disjoint (`mmproj.*`, `radio.*`, fpn incl. `aux_proj.*`); the CLI
    auto-picks the merged file and falls back to per-part files on explicit
    `--radio/--mmproj/--fpn`. Merged runs are byte-identical to per-part runs
    (f16 and q8_0 verified); cold-start vision setup 2.2 s → 0.77 s (one
    sequential read). Per-part files deleted (rebuildable); the LLM stays
    separate (llama.cpp arch loader + memory lifecycle).
  * **benchmarks/** (querysplat-ggml style): two modes — `fixed` (the same
    official WeDetect-CUDA boxes fed to both backends = the parity criterion)
    and `closed` (each backend runs its own detector = the real user path).
    Images: 2 official demo photos + 3 MS-COCO val2017 originals. Panels are
    rendered by `inference.py::visualize_result_boxes` itself, so they match
    the official demo style exactly. Verdicts: **5/6 cases token-identical in
    both modes** across cpp f16 / cpp q8_0 / python; the single divergence
    (000000397133) is an edge-of-argmax flip after 23 verbatim tokens (f16-
    CUDA and bf16-CUDA are equally noisy estimates; not a porting bug).
    **Finding**: the official detector is itself device-dependent — torch-
    CUDA vs torch-CPU proposal overlap is only 32/100 on demo_image2, while
    the C++ CUDA conv graph sits at 90/100 against torch-CUDA. Parity claims
    belong to fixed mode. Charts: 6 case side-by-sides + latency_ttft_bar +
    decode_tok_s (decode median: cpp q8_0 93.3 > cpp f16 54.5 > python 42.9
    tok/s).
  * **app.py** (gradio side-by-side panel): upload → task/lang/text/precision
    → both backends through run_vlx.sh → JSON + official-style PNGs + verdict;
    launch smoke PASS (127.0.0.1:7860). gradio/matplotlib added to the venv.

## Design decisions (recorded 2026-10)

### 1. ggml sourcing — single source: llama.cpp's vendored copy

* The build links **exactly one ggml: the one vendored inside
  `third_party/llama.cpp`**. The pin that matters is therefore the llama.cpp
  commit; its vendored ggml is >= the level of the former standalone `v0.21.0`
  pin, so every op the port needs (`mul_mat`, `flash_attn_ext`, `get_rows`,
  `interpolate`, conv stack) is covered.
* The former standalone `third_party/ggml` submodule and `ggml-patches/` were
  removed (2026-10-19): the fallback was never used once llama.cpp entered the
  build, and keeping two ggml trees invited the "which one am I running?"
  confusion. CMake now hard-errors if `third_party/llama.cpp` is missing.
* ggml-source changes to llama.cpp's vendored copy go through
  **`third_party/llama-patches/*.patch`** (one live patch: force IEEE fp32
  for `GGML_PREC_F32` F32xF32 matmuls — CUBLAS TF32 would silently degrade
  the f32 precision contract). Replayed idempotently at configure time;
  the dirty submodule is the expected local state.
* ggml is a submodule; the integration contract is identical to the sibling
  integrations (ultralytics-ggml / sam-3d-objects-ggml).

### 3. AICore-style task layer (the integration seam)

The runtime is being reshaped toward the AICore task convention
(`core/AICore/src/tasks/<task>/*_runtime.*` + `include/aicore/<task>_capi.h`,
stable C ABI, ggml fully private), mirroring the map-anything integration:

```
include/vlx/
├── vlx_export.h            # VLX_CAPI macro (drop-in rename to AICORE_CAPI)
├── vlx_image_view.h        # borrowed row-stride image view (== aicore_image_view)
├── vlx_pipeline_timing.h   # preprocess/inference/postprocess/e2e contract
└── vlx_seek_capi.h         # the task C API (options/ctx/run/result getters)
src/
├── vlx_pipeline_common.*   # shared helpers (moved verbatim from vlx_cli.cpp)
├── vlx_seek_runtime.*      # task runtime: owns detector + vision + LLM
└── vlx_seek_capi.cpp       # C API implementation (thin shell)
tests/vlx_capi_smoke.c      # pure-C smoke of the ABI surface
```

* `vlx_seek_run_image(ctx, image_view, task, text, lang, boxes, ...)` runs the
  full chain; typed getters expose answer/boxes/labels/tokens/timings;
  `render_png` reproduces the official visualization.
* Error contract mirrors AICore: 0 ok, -1 contract, -2 runtime, -3 resources
  (device cannot hold models + workspace), plus per-context `last_error`.
* The CLI keeps its own copy of the pipeline flow for now (it is the
  regression anchor); the duplication retires when the task moves into
  AICore and the CLI either thins down or is replaced by the plugin UI.
* Pure-C smoke: `vlx-seek-capi-smoke <models_dir> <image> [text]` exercises
  the whole ABI surface end to end.

### 4. Why the upstream model is Qwen3.5 (and what Qwen3.8 means here)

VLX-Seek 1.5 (released 2026-07) was **trained** on the then-latest Qwen3.5
hybrid backbone (Gated DeltaNet + Gated Attention). The backbone is not a
swappable part: HFRE region embeddings and the `<obj*>` token semantics are
alignment-trained *into* it. Two consequences:

* You cannot graft Qwen3.8 weights onto the VLX-Seek-1.5 vision side and keep
  region-reference behaviour.
* The integration is still **backbone-agnostic by construction**: the LLM part
  is loaded by llama.cpp's architecture registry (both `qwen3_5` and the
  already-merged `qwen3_8` archs), and the vision sub-chain (RADIOv4 / HFRE /
  CAbstractor) never touches LLM internals. When upstream ships a Qwen3.8-based
  checkpoint, converting it is a converter-parameter change, not a port.

### 3. Component mapping

| Python source (this repo) | C++ / GGUF target | Status |
|---|---|---|
| `c_radio_v4/*` (SO400M ViT, patch16, 27 blocks, CPE pos-embed, cls+registers, Identity norm) | `src/radio_graph.*` + vision GGUF | **done** |
| `hybrid_finegrained_region_encoder.py` (RoIAlign 7x7, sine box embedding, mean fusion) | `src/hfre.*` (host-side, tensors read back) | **done** |
| `multimodal_projector/honeybee.py` (CAbstractor) | merged into `src/mmproj_graph.*` — the released weights have NO separate CAbstractor; the main tower's merger outputs 4096-d tokens directly | **done** |
| `qwen3_5_vl_encoder.py` (main vision tower) | `src/mmproj_graph.*` | **done** |
| `detect_tools/WeDetect/generate_proposal.py` (YOLO-World) | `src/wedetect_graph.cpp` conv graph | **done** |
| Qwen3.5 LLM (10B hybrid) | llama.cpp `qwen35` arch, converted GGUF | **done** |
| `<ground>/<obj*>` output parsing + visualization | `vlx_cli.cpp` | **done** |

## Layout

```text
cpp_ggml/
├── CMakeLists.txt
├── scripts/                        # converters + parity-gate tooling
│   ├── convert_vision_to_gguf.py   # HF checkpoint -> radio GGUF (F16/Q8_0)
│   ├── dump_torch_stages.py        # torch reference tensors for parity
│   └── merge_vision_gguf.py        # merge mmproj+fpn+radio into one GGUF
├── src/
│   ├── vlx_types.hpp               # shared config/feature structs
│   ├── gguf_loader.*               # minimal GGUF reader (F32/F16/Q8_0)
│   ├── radio_graph.*               # C-RADIOv4-SO400M forward graph
│   ├── mmproj_graph.*              # Qwen3.5-VL main tower + merger
│   ├── simplefpn_graph.*           # SimpleFPN pyramid
│   ├── wedetect_graph.cpp          # WeDetect-Uni conv graph (proposals)
│   ├── hfre.*                      # host RoIAlign + sine embedding + fusion
│   ├── image_prep.*                # torch-exact host preprocessing
│   ├── vlx_cli.cpp                 # full-chain CLI (run/detect/gen/visualize)
│   └── radio_cli.cpp               # standalone radio gate CLI
└── third_party/
    ├── llama.cpp/                  # submodule: LLM + the single vendored ggml
    ├── llama-patches/              # idempotent llama.cpp patches (see README)
    └── stb/                        # vendored stb_image / stb_image_write
```

## Build (vision sub-chain)

```bash
git submodule update --init --recursive
cmake -S cpp_ggml -B cpp_ggml/build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build cpp_ggml/build-cpu --parallel 8
cpp_ggml/build-cpu/vlx-cli --help
cpp_ggml/build-cpu/radio-cli --help   # standalone vision sub-chain tool
```

CUDA/Vulkan builds follow the sibling repos: forward `GGML_CUDA=ON` /
`GGML_VULKAN=ON`; exactly one GPU backend per build directory
(`build-cuda` / `build-vulkan`; both binaries land in the build root).
Runtime GGUFs live under `models/gguf/` — `vlx-cli run` picks up the merged
`vlx-vision-10b-<p>.gguf` (mmproj + fpn + radio) automatically.

## Convert + parity (RADIO)

Standalone component gate (the daily full-chain validation uses `vlx-cli
run` against the merged `vlx-vision-10b-<p>.gguf` instead):

```bash
# 1. torch reference tensors (CPU torch is enough)
python3 cpp_ggml/scripts/convert_vision_to_gguf.py \
    --model-path resources/VLX-Seek-1.5-10B --dtype f16 \
    --output cpp_ggml/models/gguf/vlx-radio-10b-f16.gguf

python3 cpp_ggml/scripts/dump_torch_stages.py \
    --model-path resources/VLX-Seek-1.5-10B --image demo/demo_image.jpg \
    --out /tmp/vlx-ref

# 2. C++ run on the identical input tensor
cpp_ggml/build-cpu/radio-cli radio \
    --model cpp_ggml/models/gguf/vlx-radio-10b-f16.gguf \
    --input-bin /tmp/vlx-ref/input.bin \
    --dump /tmp/vlx-ref/ggml

# 3. elementwise gate
python3 cpp_ggml/scripts/compare_parity.py /tmp/vlx-ref
```

Pass criterion: `|cpp - torch| <= max_abs + max_rel*|torch|` with the sibling
defaults (f32: 0.05/0.05 tightened by hand, f16: 0.5/0.05).

## Status (2026-10-15)

* **Synthetic-weight parity: PASS** (4 levels, 0 violations; full tool chain
  convert -> torch dump -> radio-cli -> compare exercised end to end).
* **Real 10B checkpoint**: conversion, loading and inference all run
  (dim1152 depth27 pos128x128 skip10, 64x64x1152 levels).
* Real-weight stage parity: patchembed `1.4e-7`, postpos/postcls `~7e-11`
  (CPE pos layout fixed to token-major; CLIP conditioner verified exact).
* **Architecture change**: radio_graph is now SPLIT INTO 4 SEGMENT GRAPHS
  (cut at out_indices; each segment has exactly ONE terminal output, the
  next segment consumes it cross-graph — llama.cpp logits_all pattern).
  This removes the unreliable multi-output tap reads entirely.
* **BREAKTHROUGH (diag mode)**: `radio-cli diag --point n1|qkv|attnbr|mlpbr|x`
  feeds the VERIFIED postcls stream into a dedicated SINGLE-OUTPUT graph per
  component. ALL five block-0 components match torch to 1e-7 (n1 1.6e-7,
  qkv 2.7e-7, attnbr 1.4e-7, mlpbr 2.2e-7, x 3.2e-7) — **every block
  sub-component is provably correct in a clean graph**. Interior dbg taps
  inside the main graph corrupt it (tap cont nodes change gallocr layout);
  they have been removed from the production graph entirely.
* **Per-block segments (27 graphs)** attempted for fine bisection; current
  build core-dumps (investigation pending — likely segs[] lifetime/allocr
  count). Pre-per-block L deltas were deterministic (0.103/0.138/0.191/0.809).
* **Open (FA suspicion)**: with clean single-output segments the L deltas are
  deterministic and reproducible (0.103/0.138/0.191/0.809), identical to the
  pre-segmentation values — so segments did not change computation, and the
  earlier interior-tap "matches" (single-graph + dbg-expand runs) are now
  suspect: dbg-expand count changed gallocr allocation patterns, meaning tap
  snapshots under reuse are run-shape-dependent. Strongest remaining suspect:
  `ggml_flash_attn_ext` at T=4106 (synthetic PASS covered only T=66).
  Next action: standalone FA unit test (T=4106, hd=72, nh=16) vs numpy.
* **Historical (superseded)**: block-0 interior taps matched to
  `1.6e-6` (n1/qkv/attnbr/n2/mlpbr, all immediate-materialized copies), and
  the component sum `postcls + attnbr + mlpbr` reproduces torch's block-0
  output to `3.2e-4` (f32 noise) — **the compute graph is provably correct**.
  The remaining delta is confined to the tap read-back path (the dumped
  L-tensor does not correspond to any layout permutation of the verified
  components; likely gallocr reuse interacting with tap buffers across
  runs with/without debug expands). Fix is one read-back bug away.
  Bisection tools: `--out-indices`, `--dump-blocks`, `--dump-interior`,
  `VLX_RADIO_DEBUG{,_DIR}`.
* bf16 checkpoints: safetensors loading goes through torch (`.float()`).

## P1 status (2026-10-16, evening): mmproj FULL-CHAIN PARITY GATE PASS

* **ROOT CAUSE of the entire saga (merger "corr=0.39" was a red herring): the
  cross-graph expand trap in the per-block segment architecture.** The merger
  probe round proved the merger segment innocent before touching it:
  * numpy replay of the exact merger math with the GGUF-resident weights
    (NOT the safetensors) matches torch pooler to 2.45e-8 — converter/loader
    weights bit-exact, no transpose trap in the square 4608x4608 fc1.
  * numpy replay with the C++ ggml final as input matches the C++ pooler to
    3.1e-8 — the C++ merger graph (LN -> reshape -> fc1 -> GELU-erf -> fc2)
    was already correct; fc1/gelu_erf/reshape all innocent.
  * But `ggml_final.bin` vs `torch_mm_final.bin` max diff was 12866 — the
    divergence was upstream of the merger, and seg_seg01 (block 0) already
    deviated (mean 4.6e-2) while `mmproj-diag` (SINGLE graph, same build)
    matched to 3.8e-5. The only difference between the two modes was the
    graph organization itself.
* **Mechanism**: a segment input (`segs[i-1].out_x`) carries `src` pointers
  into the producer's context. `ggml_build_forward_expand` follows them, so
  EVERY segment graph contained the entire preceding chain, and every
  segment's gallocr then re-pointed the SHARED tensor objects' `data` into
  its own pool — last allocator wins, so all segments computed against an
  unwritten pool address (garbage input), with O(depth^2) work as a side
  effect. seg00 was correct only because it references nothing external.
* **Fix: ONE graph for the whole tower** (seg0 + 27 blocks + merger in a
  single ctx/ggraph/gallocr). All inputs and read-back taps live in the
  dedicated ictx buffer (gallocr never re-points them); the HFRE tap is
  `ggml_cpy`-ed into an ictx leaf so its address survives the whole-graph
  compute (llama.cpp dump-tensor pattern). `rope_ones` also moved to ictx
  (constant for the buffer's lifetime, one upload). Runtime dropped from
  ~minutes to 47 s (O(depth^2) -> O(depth)).
* **Official gate (scripts/compare_mmproj_parity.py): PASS.** Demo image
  58x72 grid, band 0.5/0.05 (sibling f32 convention):
  * final (N,1152): mean_abs=1.5e-2 max_abs=168.9 violations=0/4810752
  * pooler (N/4,4096): mean_abs=8.1e-5 max_abs=0.179 violations=0/4276224
  The residual f32 noise of ggml FA/matmul cascades through 27 pre-norm
  blocks (block-0 diag rel ~1e-4 -> block-26 rel ~3e-3); the few 1e4-scale
  channels (ch 514 family) show <=2.2% relative deviation — covered by the
  5% relative band. No structural (token/channel-clustered) error remains.
* **RADIO post-mortem rewrite (P0 section above is superseded)**: the
  "FA suspicion" and the L deltas (0.103/0.138/0.191/0.809) are the SAME
  cross-graph expand trap (radio_graph still uses 4 cross-graph segments);
  `ggml_flash_attn_ext` was innocent all along. radio_graph needs the same
  single-graph treatment before its real-weight L gate can close.
* `--dump-seg-all` removed from mmproj CLI (per-segment dumps were part of
  the buggy architecture; component bisection stays available via
  `mmproj-diag --point n1|qkv|q_rot|real|attnbr|mlpbr|x`).

### P4 FINAL (2026-10-17, night): 10B text-level parity vs inference.py — PASS

* **Real tokenizer.json obtained** (hf-mirror.com reachable;
  `omlab/VLX-Seek-1.5-10B` — 248044 BPE + 138 added tokens = 248182, incl.
  `<obj0>`..`<obj99>` @248077+, `<objfeat>` @248181, `<|image_pad|>` @248056).
  Re-conversion with the real vocab: <1 min (only the vocab section changes;
  the placeholder-run weights were already verified).
* **`vlx-cli vl` with the 10B LLM (f16): FULL CHAIN RUNS END TO END** —
  vision gates (bit-level) -> pooler/objfeat embedding injection (batch_ext,
  row-by-row) -> 10B f16 decode -> **greedy output IDENTICAL to
  inference.py** (CPU, sdpa, bf16):
  * python: `"answer": "<ground>person</ground><objects>None</objects>",
    "prompt_tokens": 218, "completion_tokens": 7`
  * cpp: `<ground>person</ground><objects>None</objects>` (6 + EOG),
    prompt = 218 positions (3 text + 1044 pooler embs + 3 obj tokens +
    3 objfeat embs + question)
  * `<objects>None</objects>` is the model's genuine answer (the demo image
    contains no person and the boxes are arbitrary test rectangles).
* Python-side fixes to run the official demo on CPU (repo's own code, not
  submodules): `builder.py` flash_attention_2 -> sdpa, the VL encoder's
  flash_attention_2 -> sdpa, and an antialiased-bilinear bf16 CPU gap in
  c_radio_v4_aux_encoder.py (upcast to f32 for the resize).
* **Remaining** (all auxiliary, none blocking the text chain):
  1. WeDetect conv graph: **weights local + python reference READY**:
     `resources/wedetect_base_uni.pth` (434 MB, 910 tensors, downloaded via
     hf-mirror) loads STRICTLY into detect_tools' SimpleYOLOWorldDetector
     after a key remap (`backbone.image_model.model.*` -> `backbone.*`,
     `bbox_head.head_module.*` -> `bbox_head.*`, ConvModule conv/bn pairs
     renumbered to flat Sequential indices — classifier head unused);
     `scripts/dump_wedetect_ref.py` dumps backbone c1..c4 (128/256/512/1024
     ch @ 640 letterbox), neck 3 levels, head{i} ce/bp/cls to
     /tmp/wedetect_ref.pt as the C++ gate reference. C++ conv graph is the
     next round's main task (conv infrastructure ready; open items:
     depthwise 7x7 via im2col+broadcast-mul+sum_rows, BN folded at
     conversion, letterbox host prep, batched_nms on host).
  2. Exact PIL-BICUBIC calibration (probe tables saved) — current
     approximation shifts the pooler by 3.6e-3 (negligible; both backends
     already agree at the text level WITH the approximation in place).
  3. stb_truetype font asset for pixel-exact visualize text.

### P4 round 2 (2026-10-17, late): one-click launcher + JPEG input + visualize

* **`run_vlx.sh` (repo root)**: one-click launcher for both backends.
  `--backend cpp` runs preprocess -> full chain (auto grid discovery via
  grid.json); `--backend python` passes through to the official inference.py
  (unknown args forwarded). Defensive checks: binary discovery across build
  dirs (build-vlx -> build-cpu), GGUF/weight existence with the exact fix
  command, temp-workdir cleanup. 10B LLM GGUF plugs in via --llm-gguf when
  converted.
* **stb_image vendored** (cpp_ggml/third_party/stb/, deliberately NOT a
  submodule — llama.cpp itself vendors stb at `llama.cpp/vendor/stb/`, and a
  submodule would pull the whole multi-library stb repo for two headers;
  committed-to-repo files give other developers byte-identical copies on a
  plain clone, `git diff` exposes any change — the sync contract holds).
  stb_image.h v2.30 copied verbatim from llama.cpp/vendor/stb/ at our pin;
  stb_image_write.h v1.14 from the local MVS checkout; provenance in
  third_party/stb/README.md.
  `vlx-cli preprocess --image x.jpg` reads JPEG/PNG directly and emits BOTH
  inputs (radio [0,1] HWC bilinear bin + mmproj block-major bicubic patches
  bin + grid.json).
  * radio line: PIL BILINEAR semantics recovered exactly by delta-impulse
    probing (center=(o+0.5)*scale−0.5, kernel args (i−center)/ks, ks=max(1,
    scale), clamp+normalize).
  * JPEG-decoder delta (stb vs libjpeg/PIL, ±1-2 LSB, decoder-inherent —
    identical across stb v2.25/v2.30): input diff 8.5e-4 **decays to 1.4e-7
    at the RADIO output** (27 blocks of LN+residual are extremely robust to
    input perturbation); both decoder paths PASS the gate. The mmproj-line
    BICUBIC approximation dominates: pooler mean_abs 3.6e-3, 70/4.3M
    violations — negligible for generation; final-tap band shows 7.6%
    violations. Exact PIL-BICUBIC calibration (probe tables saved) is the
    first item of the next round; until then the python processor dump stays
    the gate reference and the C++ preprocess serves live runs.
* **`vlx-cli visualize`**: red outline width 3 aligned with
  inference.py::visualize_result_boxes + built-in bitmap digit tag; PNG out
  (stb_image_write). Pixel-exact PIL-font text parity needs a font asset.
* **WeDetect weights**: `fushh7/WeDetect` HF repo (wedetect_base_uni.pth,
  auto-download logic already in inference.py) — network unreachable this
  round; retry with the documented snippet when online.
* **Qwen3.5 LLM converter — ALREADY SUPPORTED, no patches needed**:
  llama.cpp's refactored `conversion/` package registers
  `Qwen3_5ForCausalLM` -> `conversion/qwen.py::Qwen3_5TextModel` (model_arch
  QWEN35, V-head reorder, mrope mixin, MTP handling — all upstream). The
  earlier "converter missing" diagnosis was wrong: the classes live in
  conversion/qwen.py, registered in conversion/__init__.py L257.
* **10B LLM GGUF CONVERTED and tensor-mapping-verified**:
  `scripts/prepare_llm_hf.py` extracts the LLM (model.language_model.* prefix
  strip + lm_head, sharded safetensors, architectures=["Qwen3_5TextModel"] —
  the TextModel name takes the no-MTP branch; ForCausalLM trips an MTP assert)
  and `scripts/run_llm_convert.py` runs convert_hf_to_gguf with an in-process
  override of get_vocab_base_pre (the PLACEHOLDER tokenizer's probe hash
  matches nothing; llama.cpp sources untouched). Result:
  `/tmp/vlx-llm-hf/vlx-10b-llm-f16.gguf` (17.9 GB f16, 427 tensors) —
  llama.cpp's qwen35 loader accepts every tensor (arch qwen35, n_embd 4096,
  32 layers, heads 16/4, head_dim 256).
  Remaining blocker for TEXT parity: the checkpoint ships NO tokenizer.json
  (248182-vocab BPE); the placeholder vocab has no merges so llama.cpp fails
  at vocab load. When the real tokenizer.json is obtained (VLX-Seek HF repo
  or Qwen3.5 base + added tokens), re-run prepare_llm_hf.py with
  --real-tokenizer-json and re-convert (the conversion takes <1 min); then
  `run_vlx.sh --backend cpp --llm-gguf vlx-10b-llm-f16.gguf` runs the full
  dimension-matched chain and text parity against inference.py becomes
  possible.
* **Qwen3.5 LLM converter — spec locked (superseded by the discovery above)**:
  llama.cpp's converter has NO qwen35 class (the 0.8B GGUF came pre-built
  from HF). Gold-standard tensor names from the official 0.8B GGUF (26
  patterns): token_embd / output_norm / blk.N.{attn_norm, post_attention_norm,
  attn_q(+gate merged 2x), attn_k, attn_v, attn_output, attn_q_norm,
  attn_k_norm, attn_gate(linear-attn z proj), ffn_{gate,up,down},
  ssm_{a, alpha, beta, conv1d, dt.bias, norm, out}, blk.N.nextn.* (MTP,
  absent in VLX-Seek). HF mapping: input_layernorm->attn_norm,
  post_attention_layernorm->post_attention_norm, self_attn.q_proj (2x q_dim,
  q+gate packed) -> attn_q, k_proj/v_proj (1x each) -> attn_k/v,
  o_proj -> attn_output, q_norm/k_norm -> attn_q/k_norm;
  linear_attn.in_proj_qkv (key_dim*2+value_dim) -> attn_qkv,
  in_proj_z (value_dim) -> attn_gate, in_proj_a -> ssm_alpha,
  in_proj_b -> ssm_beta, conv1d -> ssm_conv1d, dt_bias -> ssm_dt.bias,
  A_log -> ssm_a, norm -> ssm_norm, out_proj -> ssm_out;
  mlp.{gate,up,down}_proj -> ffn_{gate,up,down}; embed_tokens -> token_embd,
  lm_head -> output (untied), norm -> output_norm. Implementation goes into
  convert_hf_to_gguf.py via a **llama-patches/*.patch** (same replay contract
  as ggml-patches; submodule stays untouched).

### P0/P4 (2026-10-17): LLM side + vlx-cli full-chain assembly — operational

* **P0 smoke PASS**: `llama.cpp` qwen3_5 arch + `Qwen3.5-0.8B-Q8_0.gguf` generates
  coherent text (38 t/s on the `llama` multi-tool binary). No code needed.
* **CMake include-path switch implemented (README §1 contract)**: when
  `third_party/llama.cpp/ggml` is present, ALL targets (vlxvision incl.) compile
  against llama.cpp's vendored ggml — one ggml per process. The pinned v0.21
  submodule + patch replay stays for the standalone dev build. **Verified:
  every vision gate re-PASSes on the llama.cpp ggml unchanged** (mmproj
  final/pooler, FPN L0..L3+vt, radio L0..L3, objfeat — same numbers as v0.21;
  the CPU kernels are numerically compatible across the bump).
* **`vlx-cli` (src/vlx_cli.cpp, built in build-vlx)**:
  * `gen` — plain-text generation (P0 regression ✓).
  * `gen-embd` — embedding-injection pipeline gate: prefix tokens -> K
    injected embd rows -> suffix -> greedy. Determinism (two runs identical)
    and injection sensitivity (different embeddings -> different output) both
    verified on 0.8B.
  * `vl` — full chain: RADIO + mmproj + SimpleFPN + HFRE fuse + mlp2x_gelu
    projector -> prompt assembly (`<|vision_start|>`[pooler embs]`<|vision_end|>`
    `<obj><objfeat>`… question) -> LLM. Dimension guard: requires
    LLM n_embd == 4096 (the 0.8B smoke model is rejected with a clear message;
    the vision chain still runs, whose gates PASS independently).
* **llama.cpp API findings (this revision, build b11644)**:
  * `llama_batch::embd` is DEAD — embedding injection goes through the new
    `llama_batch_ext` API (`llama_batch_ext_add_embd` with `llama_embd{data,
    n_rows, n_embd}`, decoded via `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b)`).
    The legacy path silently decodes nothing (A==C pitfall).
  * `llama_batch_ext` requires EXPLICIT positions (`llama_batch_ext_set_pos`):
    without them the second batch returns rc=-1 (invalid input). Track a pos
    cursor across segments.
  * Output logits are per-entry (`llama_batch_ext_set_output_logits`) — the
    context params have no global flag in this revision.
  * `ggml_permute(a, p0..p3)` maps old axis i to new axis p_i (re-confirmed).
* **ggml-source sync contract (user requirement, verified clean)**: both
  submodules carry zero manual modifications; all ggml changes live in
  `third_party/ggml-patches/*.patch`, replayed idempotently by CMake
  (`cmake/apply_patches.cmake`, stamp `.vlx-patches-applied`). llama.cpp gets
  the same treatment via a `llama-patches/` directory if ever needed. This
  round required ZERO submodule edits — vlx_cli links the public API only.
* **Remaining for a full 10B e2e run** (next round):
  1. Convert the released 10B LLM to GGUF via llama.cpp's qwen3_5 converter
     (checkpoint is local; expect a large artifact + long conversion).
  2. `vlx-cli vl` with the 10B LLM (dimension-matched) -> text-level parity
     against `inference.py` (greedy alignment).
  3. WeDetect: weights are NOT in the VLX-Seek checkpoint — an independent
     WeDetect-Base-Uni checkpoint must be supplied. Conv infrastructure is
     ready (im2col conv, ConvT 4-slice, LN, GELU); open items: depthwise 7x7
     (per-channel im2col), BN fusion at conversion, letterbox host prep.
  4. stb_image vendoring for direct JPEG input (removes --input-img-bin).

* **radio_graph single-graph refactor** (same surgery as mmproj): ONE graph
  for conditioning + 27 blocks; 4 level taps = view(strip cls) ->
  reshape(C, gw, gh) -> permute(2,0,1) -> cpy into NCHW ictx leaves.
  **Real-weight L gate finally PASSES**: L0..L3 mean_abs 4.7e-7 .. 1.8e-6,
  zero violations (band 0.5/0.05), 25.4 s end-to-end (was minutes, O(depth^2)).
  The historical L deltas 0.103/0.138/0.191/0.809 are gone — flash_attn_ext
  was innocent; the cross-graph expand trap was the whole story. synth gate
  re-verified (mean 4.1e-7).
* **<objfeat> chain end-to-end (`radio-cli objfeat`)**: RADIO(aux levels) +
  mmproj(pooler) -> SimpleFPN(vt levels) -> encode_aux/encode_vt ->
  fuse(concat + sine pos embed, original-image box frame, vts_scale 1/32) ->
  mm_projector_aux (mlp2x_gelu) mini-graph -> (N, 4096) LLM-hidden tokens.
  Reference: `dump_torch_objfeat.py` (torchvision RoIAlign + hand-written
  gen_sineembed + mlp2x_gelu from the checkpoint). Gates: aux (N,4608)
  3.8e-7 corr=1.0; fused (N,6656) 1.2e-5 corr=1.0; **objfeat (N,4096)
  6.2e-5 corr=1.000000, zero violations**.
* Bugs fixed en route:
  1. **sine_embed block order**: python cat((pos_y, pos_x, pos_w, pos_h)) —
     output block 0 = cy, block 1 = cx (input rows are (cx, cy, w, h)); the
     C++ wrote them in input order.
  2. **hfre roi_align_mean_pool missing the out_size^2 divisor** (196x error
     for roi_output_size=14) — also silently inflated the earlier aux gate
     (constant-scale errors preserve correlation; gates need abs bands).
  3. **hfre encode_vt level layout**: torch.cat(dim=1) interleaves level
     blocks inside each row; the memcpy version emitted level-major blocks.
  4. **1-D weight tensors in ad-hoc graphs**: the gguf loader fills ne[0]
     only — building a 2-D tensor with ne[1] = 0 allocates zero bytes and
     the tensor_set trashes the pool. Always expand 1-D to (D, 1).
  5. Host-side transpose of the projector output (read (D_out, N)
     token-major, transpose on host): N <= box count, zero cost, and one
     less in-graph view to defend.
* WeDetect (P3) status: its weights are NOT in the VLX-Seek checkpoint (no
  yolo/detect keys) — an independent WeDetect-Base-Uni checkpoint must be
  supplied before a conv graph can be gated. Reuse inventory from this round:
  conv2d k1/k3 (im2col+mul_mat, k order c*9+ky*3+kx verified), ConvT k2s2
  (4-slice merged GEMM + pixel shuffle), token-major LN over C, erf GELU.
  ConvNeXt-specific gaps to solve there: depthwise 7x7 conv (ggml im2col has
  no groups — per-channel im2col or a dwconv kernel), BatchNorm inference
  (fuse BN into conv weights at conversion time), and the letterbox
  preprocessing (host-side, trivial).

### P1 close-out (2026-10-17): HFRE vt chain — OFFICIAL GATE PASS

`simplefpn_graph.{hpp,cpp}` (ONE graph, ictx leaves, per-level cpy taps — the
single-graph lesson applied) + `convert_fpn_to_gguf.py` (independent
vlx-fpn-10b-f32.gguf, 42 tensors, ConvT 4-slice merge baked) +
`dump_torch_mmproj.py` extension (hand-written SimpleFP forward + torchvision
RoIAlign reference) + `radio-cli mmproj --fpn ... --hfre boxes.json --dump-fpn
--dump-vt` end-to-end. Official gate `scripts/compare_fpn_parity.py`: **PASS**
— L0..L3 + vt pooled all corr=1.000000, zero violations (band 0.5/0.05;
vt mean_abs 3.7e-5).

Bugs found and fixed en route (all layout/permutation class, same standing
pattern as every other stage of this port):
1. **token-major is NOT NCHW**: token-major (C, W, H) is c-FASTEST; NCHW is
   c-SLOWEST. The level taps needed an in-graph permute (C->2, W->0, H->1)
   before cpy into an (W, H, C) leaf. The earlier "token-major memory IS NCHW"
   note in this README was wrong and has been retracted.
2. **ConvT slice view axis order**: the slice views must keep (C, W, H)
   (i1 = w). With (C, H, W) the downstream cat0+reshape silently re-interprets
   the rows with h/w swapped — numerically equivalent to running the WRONG
   k=2b+a slice mapping (corr 0.55 vs torch). Caught by a numpy replay of the
   graph semantics: correct mapping corr=1.0, swapped corr=0.546 == the C++
   gate value.
3. **ggml_permute parameter semantics**: permute(a, p0..p3) maps OLD axis i
   to NEW axis p_i (verified against the FA-layout usage in radio/mmproj).
4. **hfre roi_align_mean_pool missing out_size^2 divisor**: acc spans every
   sample of every bin, torchvision averages per bin (sr^2) then across bins
   (14^2); the missing factor made every pooled value 196x too large. It had
   slipped through the earlier aux gate because a constant-scale error
   preserves correlation — gates must check scale, not only corr.
5. **hfre encode_vt level-block layout**: torch.cat(feats, dim=1) interleaves
   level blocks INSIDE each row (out[n, li*C+ch]); the memcpy-based version
   produced a level-major (n_levels*N, C) layout.
6. LayerNorm keys in the FPN checkpoint come in TWO shapes: standalone LN
   layers (flat `simfp_N.M.weight/bias`) vs Conv2d-wrapped norms
   (`simfp_N.M.norm.weight/bias`); the converter emits both verbatim.

SimpleFP structural notes (verified against the checkpoint): ConvT at
sequential index 3i with standalone LN(3i+1)+GELU(3i+2) only BETWEEN ConvTs,
trailing 1x1/3x3 pair at 3n-2/3n-1 (n=0 -> 0/1); nn.GELU() = exact erf;
1x1/3x3 convs are bias-free (LN carries the bias); ConvT output is exactly
2x (k2 s2, no crop). SimpleFPN input = the pooler output (4096, gh/2, gw/2)
— qwen3_5_vl_encoder.get_multi_level_features wraps it as a single-element
list and encode_objects takes image_features[-1].

### Next: HFRE vt chain (P1 close-out) — SimpleFPN execution spec

`hfre.cpp` encode_vt/fuse are complete; the missing piece is the SimpleFPN
conv graph consuming the final tap (C=1152, gh x gw -> NCHW). From
`simple_fpn.py` + the 10B HFRE config (dim=vt_dim, scale_factors=[8,4,2,1],
stride=32, out_channels=512, norm=LN):

* per level s in [8,4,2,1]: stack of ConvTranspose2d(k2,s2) (+LN+GELU between
  them, count = log2(s)), then Conv2d 1x1(+LN) and Conv2d 3x3 pad1(+LN).
* op mapping (all ggml-native, radio/mmproj patterns):
  * Conv2d k1/k3 s1 -> ggml_im2col + mul_mat; NCHW (C,H,W) contiguous in,
    (C_out, L) contiguous out IS (C_out,H,W) NCHW — no layout churn.
  * channels-first LN over C -> reshape (C, HW) -> ggml_norm(ln_eps 1e-6)
    *w + b (same lnorm lambda as mmproj; w/b broadcast ne=(C,1)).
  * ConvTranspose2d(k2,s2) -> 4-slice 1x1 conv form: out[2i+a][2j+b] =
    W[:, :, a, b]^T x[i, j] (a,b in {0,1}); 4 mul_mats + pixel-shuffle
    (H,W,4)-style interleave via reshape/permute/cont pairs (no 5D ops).
  * GELU -> ggml_gelu. No MaxPool needed (no 0.5 scale factor in the vt
    config).
* converter: extend convert_mmproj_to_gguf.py (or a sibling script) to emit
  the SimpleFP weights with the slice/transpose decisions baked in numpy-side.
* torch reference: extend dump_torch_mmproj.py with the HFRE vt branch
  (final tap -> SimpleFP -> 4 levels NCHW dumps) for the level-wise gate.

## P1 status (2026-10-16, earlier): mmproj graph runs; seg0 parity PASS

* `mmproj_graph.cpp` implemented: 29 per-block single-output segments
  (seg0 preconditioning + 27 blocks + merger), radio architecture reused.
* **seg0 parity PASS (2.5e-7)**: patch GEMM (conv3d flattened to (1152,1536)
  GEMM, numpy C-order k = ic*512+it*256+iy*16+ix), align_corners=True
  bilinear pos embed (48x48 -> grid), spatial-merge-block-major token order —
  all verified vs HF Qwen3_5VisionModel (eager, f32, weights injected from
  checkpoint; 333/333 tensors matched, zero missing).
* Torch reference: `dump_torch_mmproj.py` (HF model itself is the reference;
  input = Qwen2VLImageProcessor output with patch_size=16, merge_size=2 —
  smart_resize picks the grid, e.g. 58x72 for the demo image).
* **Bugs fixed en route** (all layout-class, per the standing pattern):
  1. rope theta: converter wrote the 1e6 default; HF fills generic 10000.
  2. host cossin table row overrun: rp[2*hd+hd+i] = 144+i writes past the
     144-wide row (segment stride is nf=18, not half=36).
  3. qkv q/k/v views must be per-token 3C blocks (view_3d nb2=3C*4, offsets
     {0,C,2C}*4 bytes) — NOT row-block views with C*4 stride.
  4. sin view offset is +hd floats INSIDE each token row, not +hd*N.
  5. ggml flash_attn_ext output ne=(v->ne[0], T, nh) is already
     token-major (internal permute) — no output permute before reshape.
* **BREAKTHROUGH (2026-10-16): all 27 blocks BIT-EXACT (final mean=0.0)**.
  Root cause of the entire mmproj saga: **view-of-view** — q/k/v as views of
  the qkv projection result must be ggml_cont-MATERIALIZED before any further
  view/mul; secondary views of a strided view read with silently different
  semantics (q_rot 1.45 -> 7.8e-7, attnbr 1.3e-7, final 0.0). Verified by the
  standalone semantics probe (repeat/strided-mul/concat all match numpy to
  5e-6 on contiguous sources) plus the cont fix. Main-graph rope cossin views
  are also mul-ones materialized now (shared rope_ones input).
* **Open (last mile)**: merger segment corr=0.39 vs torch pooler, while
  (a) the C++ final input is bit-exact and (b) a numpy re-run of the exact
  merger math (norm eps 1e-6 -> reshape(4608) -> fc1 -> GELU-erf -> fc2)
  matches torch pooler to 2.0e-8. So the divergence is inside the C++ merger
  graph ops only. First suspects: mul_mat weight strides for the 4608x4608
  fc1 / gelu_erf kernel / the reshape-2d of the norm output. Diagnose with
  the same standalone-probe pattern (fc1+gelu+fc2 on dumped n rows).
  cos/sin in-graph 9.6e-8, host table 9.3e-8, buffer upload bit-exact.
  concat semantics — one of them deviates from the assumed definition.
  (reuse radio diag pattern).

## P1 design notes (mmproj segment)

* Vendored llama.cpp has `tools/mtmd/models/qwen3vl.cpp` (Qwen3-VL) but **no
  qwen3.5-vl** mmproj graph and no converter support — a port is required.
* Qwen3-VL vs Qwen3.5-VL structural differences found so far: merge position
  (Q3VL merges 2x2 BEFORE blocks with n_embd*2 tokens; Q3.5VL runs blocks at
  full token count and merges at the END via merger fc1 4608x4608 -> fc2
  4096x4608), plus m-rope details to be confirmed from the HF reference
  implementation of `qwen3_5` vision.
* Required reference: transformers `models/qwen3_5/modeling_qwen3_5.py`
  vision part (norm/act/rope/merger order). Environment lacks transformers —
  install or read the source from the upstream repo when implementing.
* Converter (`convert_mmproj_to_gguf.py`, weights done, 1.7 GB F32) already
  handles Megatron names (mlp.linear_fc1/fc2), 5D patch conv kt2, pos 48x48.
* Torch reference for parity: extend `dump_torch_stages.py --dump-mmproj`
  (hand-written forward, same pattern as RADIO — no transformers needed).

## Roadmap to full-chain C++ — COMPLETE (2026-10-18; history below)

All five phases shipped and gated; see "Status close-out" and the d12 entry
near the top of this file. Kept as the planning record:

1. `P0` official Qwen3.5 text GGUF on llama.cpp (ssm path smoke test) — no new code.
2. `P1` mmproj graph: Qwen3.5-VL tower + CAbstractor (llama.cpp converter extension).
3. `P2` merge RADIO graph into mmproj GGUF; HFRE host glue + 1-token embedding
   chunks injected at `<objfeat>` positions via mtmd chunk API.
4. `P3` WeDetect conv graph (ultralytics-ggml YOLO-World patterns).
5. `P4` `vlx_cli.cpp`: image -> boxes -> mmproj -> llama.cpp generate -> region
   parse -> draw; full-chain parity against `inference.py`.
