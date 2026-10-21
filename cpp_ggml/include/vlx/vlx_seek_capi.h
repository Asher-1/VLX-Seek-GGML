// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// VLX-Seek C API: full-chain fine-grained VLM inference over the in-tree
// ggml port in cpp_ggml/src (upstream VLX-Seek-GGML cpp_ggml).
//
// Pipeline: WeDetect-Uni proposals -> Qwen3.5-VL main tower -> C-RADIOv4 aux
// tower + SimpleFPN -> HFRE region encoding -> Qwen3.5 LLM -> referring
// answer. One context owns the detector, the vision stack and the LLM.
//
// Thread-safety: a context is single-threaded. Use one context per consumer,
// or serialize access externally.
//
// Weights: GGUF files produced by cpp_ggml/scripts (wedetect-base-uni-f32,
// the merged vlx-vision-10b-<precision>, vlx-10b-llm-<precision>).
//
// The C API never exposes ggml, STL containers or exceptions across the
// boundary; errors are return codes plus a per-context last_error message,
// exactly like the AICore task APIs this header mirrors.
//
// This header is drop-in for include/aicore/vlx_seek_capi.h at integration
// time (rename the include of "vlx/vlx_export.h" and the VLX_/vlx_ prefixes
// to AICORE_/aicore_).

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vlx/vlx_export.h"
#include "vlx/vlx_image_view.h"
#include "vlx/vlx_pipeline_timing.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the ABI version of the VLX-Seek C API (bump on breaking ABI
 *  changes). */
VLX_CAPI int vlx_seek_abi_version(void);

typedef struct vlx_seek_ctx vlx_seek_ctx;
typedef struct vlx_seek_options vlx_seek_options;

/** Creates a default options struct (device "cuda", threads 0 = backend
 *  default, precision "f16", models_dir "cpp_ggml/models/gguf", max new
 *  tokens 2048). Release with vlx_seek_options_free. */
VLX_CAPI vlx_seek_options* vlx_seek_options_new(void);
/** Releases an options struct created by vlx_seek_options_new. */
VLX_CAPI void vlx_seek_options_free(vlx_seek_options* opts);
/** Selects the inference device: "cuda", "vulkan", "cpu". */
VLX_CAPI void vlx_seek_options_set_device(vlx_seek_options* opts,
                                          const char* device);
/** CPU thread count; <= 0 picks the backend default. */
VLX_CAPI void vlx_seek_options_set_threads(vlx_seek_options* opts,
                                           int n_threads);
/** Vision weight precision: "f16" (default) or "q8_0". The detector GGUF is
 *  always f32 (the official .pth dtype — quantization changes the proposal
 *  set); the LLM follows the vision precision unless overridden. */
VLX_CAPI void vlx_seek_options_set_precision(vlx_seek_options* opts,
                                             const char* precision);
/** Directory holding the GGUF files. Default: "cpp_ggml/models/gguf". */
VLX_CAPI void vlx_seek_options_set_models_dir(vlx_seek_options* opts,
                                              const char* dir);
/** Explicit per-model overrides; empty (default) derives the filename from
 *  models_dir + precision (wedetect is always f32). */
VLX_CAPI void vlx_seek_options_set_wedetect_gguf(vlx_seek_options* opts,
                                                 const char* path);
VLX_CAPI void vlx_seek_options_set_vision_gguf(vlx_seek_options* opts,
                                               const char* path);
VLX_CAPI void vlx_seek_options_set_llm_gguf(vlx_seek_options* opts,
                                            const char* path);
/** Maximum greedy generation length (default 2048). */
VLX_CAPI void vlx_seek_options_set_max_new_tokens(vlx_seek_options* opts,
                                                  int n_tokens);
/** Log level: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR (default 1). */
VLX_CAPI void vlx_seek_options_set_log_level(vlx_seek_options* opts,
                                             int log_level);
/** Debug: dump parity stage tensors as raw f32 files into this directory
 *  after every run call. The string is copied; empty (default) disables
 *  dumping. */
VLX_CAPI void vlx_seek_options_set_dump_dir(vlx_seek_options* opts,
                                            const char* dir);

/** Loads the detector, the vision stack and the LLM. Returns a context whose
 *  is_ready() may still be 0 (device admission failure); inspect
 *  vlx_seek_last_error(). NULL only on invalid input or out-of-memory. */
VLX_CAPI vlx_seek_ctx* vlx_seek_load_opts(const vlx_seek_options* opts);
/** Releases a context; safe on NULL. Contexts must not be destroyed while a
 *  worker thread is inside a run call (serialize externally). */
VLX_CAPI void vlx_seek_free(vlx_seek_ctx* ctx);
/** Returns 1 only when the context owns a successfully loaded model set. */
VLX_CAPI int vlx_seek_is_ready(const vlx_seek_ctx* ctx);
/** Returns the last error message of the context (NULL when none). */
VLX_CAPI const char* vlx_seek_last_error(const vlx_seek_ctx* ctx);
/** Releases any buffer returned by a vlx_seek_* function (JSON/RGB
 *  helpers). Safe on NULL. */
VLX_CAPI void vlx_seek_free_buffer(void* p);

/** Load an image file as tightly-packed RGB (HWC, 3 bytes/pixel).
 *  Compatibility helper for path-driven callers; the typed hot path is
 *  vlx_seek_run_image with a borrowed vlx_image_view. Caller frees \p out_rgb
 *  with vlx_seek_free_buffer. */
VLX_CAPI int vlx_seek_load_path_rgb(const char* image_path, uint8_t** out_rgb,
                                    int32_t* out_width, int32_t* out_height);

/** ---- typed inference ----
 *
 *  One request = [optional WeDetect-Uni proposals] + vision + region
 *  encoding + LLM greedy decode. \p img must be non-NULL with positive
 *  dimensions, a supported format (RGB8 native; RGBA8/BGRA8/GRAY8 are
 *  converted) and row_stride_bytes >= width * channels (else -1).
 *
 *  \p task: "detection" | "grounding_single" | "counting" |
 *  "reasoning_detection" | "brief_region_caption" | "region_ocr" | "vqa".
 *  \p text: categories / referring phrase / question (UTF-8).
 *  \p lang: "en" or "zh" (task template language; "vqa" ignores it).
 *  \p boxes: borrowed [x1, y1, x2, y2] float quadruples in original-image
 *  pixels; NULL runs WeDetect-Uni first (100 proposals). Region tasks
 *  require non-NULL boxes. \p n_boxes <= 100.
 *  \p target_indexes / \p n_targets: zero-based proposal indexes for
 *  brief_region_caption / region_ocr (required there, ignored otherwise).
 *
 *  Returns 0 on success, -1 on contract/validation errors, -2 on
 *  runtime/inference errors and -3 when the device cannot hold the models
 *  plus the graph workspace (see vlx_seek_last_error). On success the typed
 *  result getters below expose the answer until the next run call. */
VLX_CAPI int vlx_seek_run_image(
        vlx_seek_ctx* ctx, const vlx_image_view* img, const char* task,
        const char* text, const char* lang, const float* boxes,
        int32_t n_boxes, const int32_t* target_indexes, int32_t n_targets);

/** ---- typed result of the most recent run (valid until the next run) ---- */

/** Model answer with <objN> references remapped to caller ordering. */
VLX_CAPI const char* vlx_seek_result_answer(const vlx_seek_ctx* ctx);
/** Full result as the official inference.py JSON (answer, result_bbox_list,
 *  prompt_tokens, completion_tokens). Caller frees with
 *  vlx_seek_free_buffer. */
VLX_CAPI char* vlx_seek_result_json(const vlx_seek_ctx* ctx);
/** Number of resolved result boxes (0 when the answer references none). */
VLX_CAPI int32_t vlx_seek_result_box_count(const vlx_seek_ctx* ctx);
/** Box \p index in original-image pixels (out[4] = x1, y1, x2, y2).
 *  Returns 0 on success, -1 when out of range. */
VLX_CAPI int vlx_seek_result_box(const vlx_seek_ctx* ctx, int32_t index,
                                 float* out);
/** Model-generated label of box \p index (ASCII when encodable, else the
 *  <objN> index — the official fallback). NULL when out of range. */
VLX_CAPI const char* vlx_seek_result_box_label(const vlx_seek_ctx* ctx,
                                               int32_t index);
/** Token counts of the most recent run. */
VLX_CAPI int32_t vlx_seek_result_prompt_tokens(const vlx_seek_ctx* ctx);
VLX_CAPI int32_t vlx_seek_result_completion_tokens(const vlx_seek_ctx* ctx);
/** WeDetect proposal count of the most recent run (0 when the caller passed
 *  boxes). */
VLX_CAPI int32_t vlx_seek_result_proposal_count(const vlx_seek_ctx* ctx);

/** Renders the most recent result on \p img (borrowed view; the image is
 *  copied) in the official style: red width-3 rectangles plus the
 *  model-generated label at (xmin+2, ymin+2) — non-ASCII labels render as
 *  the <objN> index, exactly like the official visualize_result_boxes.
 *  Returns 0 on success, -1 on invalid arguments, -2 on write failure. */
VLX_CAPI int vlx_seek_result_render_png(const vlx_seek_ctx* ctx,
                                        const vlx_image_view* img,
                                        const char* out_png_path);

/** Per-stage wall-clock timings of the most recent run (preprocess includes
 *  the optional WeDetect pass; inference = vision graphs + LLM;
 *  postprocess = answer parsing; serialization = PNG render). Returns 0 on
 *  success, -1 when no inference has run yet. */
VLX_CAPI int vlx_seek_last_pipeline_timings(
        const vlx_seek_ctx* ctx, vlx_pipeline_timings* out_timings);

/** Model introspection. */
/** Backend-RESOLVED device name (surfaces silent CPU fallbacks). Owned by
 *  the context; copy before freeing. */
VLX_CAPI const char* vlx_seek_context_device(const vlx_seek_ctx* ctx);
/** Effective CPU thread count after the auto (<=0) resolution. */
VLX_CAPI int vlx_seek_context_threads(const vlx_seek_ctx* ctx);
/** Vision weight precision of the loaded context ("f16"/"q8_0"). */
VLX_CAPI const char* vlx_seek_context_precision(const vlx_seek_ctx* ctx);
/** Returns a JSON summary of the loaded models and the latest run. Caller
 *  frees with vlx_seek_free_buffer. */
VLX_CAPI char* vlx_seek_info_json(vlx_seek_ctx* ctx);

/** Warms up the backend for the context's device (compiles backend kernels
 *  so the first real request does not pay JIT costs). */
VLX_CAPI int vlx_seek_warmup_backend(vlx_seek_ctx* ctx);
/** Releases process-wide VLX-Seek runtime resources it owns (idempotent;
 *  never destroys live contexts). */
VLX_CAPI void vlx_seek_shutdown(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
