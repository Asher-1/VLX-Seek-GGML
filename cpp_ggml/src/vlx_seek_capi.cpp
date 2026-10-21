// C API implementation for the VLX-Seek task. Thin shell over
// vlx::VlxSeekRuntime: opaque handles, last_error, borrowed image views in,
// retained typed results out. No ggml, STL or exception types cross here.
#include "vlx/vlx_seek_capi.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vlx_seek_runtime.hpp"

#include "stb_image.h"

using vlx::VlxRunRequest;
using vlx::VlxRunResult;
using vlx::VlxSeekOptions;
using vlx::VlxSeekRuntime;

struct vlx_seek_options {
    VlxSeekOptions opts;
    std::string device = "cuda";
    std::string precision = "f16";
    std::string models_dir = "cpp_ggml/models/gguf";
    std::string wedetect_gguf, vision_gguf, llm_gguf;
    int log_level = 1;
};

struct vlx_seek_ctx {
    std::unique_ptr<VlxSeekRuntime> runtime;
    VlxRunResult last;        // retained typed result of the most recent run
    bool has_last = false;
};

int vlx_seek_abi_version(void) { return 1; }

vlx_seek_options* vlx_seek_options_new(void) {
    return new (std::nothrow) vlx_seek_options();
}

void vlx_seek_options_free(vlx_seek_options* opts) { delete opts; }

void vlx_seek_options_set_device(vlx_seek_options* opts, const char* device) {
    if (opts && device) opts->device = device;
}

void vlx_seek_options_set_threads(vlx_seek_options* opts, int n_threads) {
    if (opts) opts->opts.threads = n_threads;
}

void vlx_seek_options_set_precision(vlx_seek_options* opts, const char* precision) {
    if (opts && precision) opts->precision = precision;
}

void vlx_seek_options_set_models_dir(vlx_seek_options* opts, const char* dir) {
    if (opts && dir) opts->models_dir = dir;
}

void vlx_seek_options_set_wedetect_gguf(vlx_seek_options* opts, const char* path) {
    if (opts) opts->wedetect_gguf = path ? path : "";
}

void vlx_seek_options_set_vision_gguf(vlx_seek_options* opts, const char* path) {
    if (opts) opts->vision_gguf = path ? path : "";
}

void vlx_seek_options_set_llm_gguf(vlx_seek_options* opts, const char* path) {
    if (opts) opts->llm_gguf = path ? path : "";
}

void vlx_seek_options_set_max_new_tokens(vlx_seek_options* opts, int n_tokens) {
    if (opts) opts->opts.max_new_tokens = n_tokens;
}

void vlx_seek_options_set_log_level(vlx_seek_options* opts, int log_level) {
    if (opts) opts->log_level = log_level;
}

void vlx_seek_options_set_dump_dir(vlx_seek_options* opts, const char* dir) {
    if (opts) opts->opts.dump_dir = dir ? dir : "";
}

vlx_seek_ctx* vlx_seek_load_opts(const vlx_seek_options* opts) {
    if (!opts) return nullptr;
    vlx_seek_ctx* ctx = new (std::nothrow) vlx_seek_ctx();
    if (!ctx) return nullptr;
    VlxSeekOptions o = opts->opts;
    o.device = opts->device;
    o.precision = opts->precision;
    o.models_dir = opts->models_dir;
    o.wedetect_gguf = opts->wedetect_gguf;
    o.vision_gguf = opts->vision_gguf;
    o.llm_gguf = opts->llm_gguf;
    ctx->runtime = std::make_unique<VlxSeekRuntime>(o);
    if (!ctx->runtime->load()) {
        ctx->last.error = ctx->runtime->last_error();
        return ctx;  // is_ready()==0; caller inspects last_error
    }
    return ctx;
}

void vlx_seek_free(vlx_seek_ctx* ctx) { delete ctx; }

int vlx_seek_is_ready(const vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime && ctx->runtime->ready() ? 1 : 0;
}

const char* vlx_seek_last_error(const vlx_seek_ctx* ctx) {
    if (!ctx) return nullptr;
    if (!ctx->last.error.empty()) return ctx->last.error.c_str();
    return ctx->runtime && !ctx->runtime->last_error().empty()
               ? ctx->runtime->last_error().c_str()
               : nullptr;
}

void vlx_seek_free_buffer(void* p) { free(p); }

int vlx_seek_load_path_rgb(const char* image_path, uint8_t** out_rgb,
                           int32_t* out_width, int32_t* out_height) {
    if (!image_path || !out_rgb || !out_width || !out_height) return -1;
    *out_rgb = nullptr;
    // stb_image decodes to tightly-packed RGB8 when requested (n=3); the
    // implementation lives in image_prep.cpp — a private detail, only the
    // decoded buffer crosses here.
    int w = 0, h = 0, n = 0;
    uint8_t* p = stbi_load(image_path, &w, &h, &n, 3);
    if (!p) return -2;
    *out_rgb = p;
    *out_width = w;
    *out_height = h;
    return 0;
}

// Converts any supported view layout into tightly-packed RGB8.
static bool to_rgb8(const vlx_image_view* v, std::vector<uint8_t>* out) {
    if (!v || !v->data || v->width <= 0 || v->height <= 0) return false;
    const int w = v->width, h = v->height;
    out->assign(size_t(w) * h * 3, 0);
    auto px = [&](int x, int y, uint8_t* r, uint8_t* g, uint8_t* b) {
        const uint8_t* row = v->data + size_t(y) * v->row_stride_bytes;
        switch (v->format) {
            case VLX_IMAGE_RGB8: *r = row[size_t(x) * 3]; *g = row[size_t(x) * 3 + 1]; *b = row[size_t(x) * 3 + 2]; break;
            case VLX_IMAGE_RGBA8: *r = row[size_t(x) * 4]; *g = row[size_t(x) * 4 + 1]; *b = row[size_t(x) * 4 + 2]; break;
            case VLX_IMAGE_BGR8: *b = row[size_t(x) * 3]; *g = row[size_t(x) * 3 + 1]; *r = row[size_t(x) * 3 + 2]; break;
            case VLX_IMAGE_BGRA8: *b = row[size_t(x) * 4]; *g = row[size_t(x) * 4 + 1]; *r = row[size_t(x) * 4 + 2]; break;
            case VLX_IMAGE_GRAY8: *r = *g = *b = row[x]; break;
            default: return false;
        }
        return true;
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t r, g, b;
            if (!px(x, y, &r, &g, &b)) return false;
            uint8_t* dst = out->data() + (size_t(y) * w + x) * 3;
            dst[0] = r; dst[1] = g; dst[2] = b;
        }
    return true;
}

int vlx_seek_run_image(vlx_seek_ctx* ctx, const vlx_image_view* img,
                       const char* task, const char* text, const char* lang,
                       const float* boxes, int32_t n_boxes,
                       const int32_t* target_indexes, int32_t n_targets) {
    if (!ctx || !ctx->runtime || !img || !task || !text || !lang) return -1;
    std::vector<uint8_t> rgb;
    if (!to_rgb8(img, &rgb)) return -1;

    VlxRunRequest req;
    req.rgb = rgb.data();
    req.width = img->width;
    req.height = img->height;
    req.task = task;
    req.text = text;
    req.lang = lang ? lang : "en";
    req.boxes = boxes;
    req.n_boxes = boxes ? n_boxes : 0;
    req.target_indexes = target_indexes;
    req.n_targets = target_indexes ? n_targets : 0;
    VlxRunResult r = ctx->runtime->run(req);
    ctx->last = std::move(r);
    ctx->has_last = true;
    return ctx->last.code;
}

const char* vlx_seek_result_answer(const vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.answer.c_str() : nullptr;
}

char* vlx_seek_result_json(const vlx_seek_ctx* ctx) {
    if (!ctx || !ctx->has_last) return nullptr;
    const std::string& js = ctx->last.json;
    char* out = static_cast<char*>(malloc(js.size() + 1));
    if (out) std::memcpy(out, js.c_str(), js.size() + 1);
    return out;
}

int32_t vlx_seek_result_box_count(const vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? int32_t(ctx->last.boxes.size()) : 0;
}

int vlx_seek_result_box(const vlx_seek_ctx* ctx, int32_t index, float* out) {
    if (!ctx || !ctx->has_last || index < 0 ||
        size_t(index) >= ctx->last.boxes.size() || !out)
        return -1;
    const auto& b = ctx->last.boxes[size_t(index)].box;
    out[0] = float(b[0]); out[1] = float(b[1]);
    out[2] = float(b[2]); out[3] = float(b[3]);
    return 0;
}

const char* vlx_seek_result_box_label(const vlx_seek_ctx* ctx, int32_t index) {
    if (!ctx || !ctx->has_last || index < 0 ||
        size_t(index) >= ctx->last.boxes.size())
        return nullptr;
    return ctx->last.boxes[size_t(index)].label.c_str();
}

int32_t vlx_seek_result_prompt_tokens(const vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.prompt_tokens : 0;
}

int32_t vlx_seek_result_completion_tokens(const vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.completion_tokens : 0;
}

int32_t vlx_seek_result_proposal_count(const vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.proposal_count : 0;
}

int vlx_seek_result_render_png(const vlx_seek_ctx* ctx,
                               const vlx_image_view* img,
                               const char* out_png_path) {
    if (!ctx || !ctx->has_last || !img || !out_png_path) return -1;
    std::vector<uint8_t> rgb;
    if (!to_rgb8(img, &rgb)) return -1;
    return ctx->runtime->render_png(ctx->last, rgb.data(), img->width,
                                    img->height, out_png_path)
               ? 0
               : -2;
}

int vlx_seek_last_pipeline_timings(const vlx_seek_ctx* ctx,
                                   vlx_pipeline_timings* out) {
    if (!ctx || !ctx->has_last || !out) return -1;
    out->abi_version = VLX_PIPELINE_TIMINGS_ABI_VERSION;
    out->valid_fields = VLX_TIMING_PREPROCESS | VLX_TIMING_INFERENCE |
                        VLX_TIMING_POSTPROCESS | VLX_TIMING_E2E;
    if (ctx->last.serialization_ms > 0) out->valid_fields |= VLX_TIMING_SERIALIZATION;
    out->preprocess_ms = ctx->last.preprocess_ms;
    out->inference_ms = ctx->last.inference_ms;
    out->postprocess_ms = ctx->last.postprocess_ms;
    out->serialization_ms = ctx->last.serialization_ms;
    out->e2e_ms = ctx->last.e2e_ms;
    return 0;
}

const char* vlx_seek_context_device(const vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->device().c_str() : nullptr;
}

int vlx_seek_context_threads(const vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->threads() : 0;
}

const char* vlx_seek_context_precision(const vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->precision().c_str() : nullptr;
}

char* vlx_seek_info_json(vlx_seek_ctx* ctx) {
    if (!ctx || !ctx->runtime) return nullptr;
    const std::string js = ctx->runtime->info_json();
    char* out = static_cast<char*>(malloc(js.size() + 1));
    if (out) std::memcpy(out, js.c_str(), js.size() + 1);
    return out;
}

int vlx_seek_warmup_backend(vlx_seek_ctx* ctx) {
    // The weights are uploaded by load(); a run at the nominal resolution
    // would be the real warmup but depends on caller input, so this is a
    // device-admission no-op that reports readiness.
    return ctx && vlx_seek_is_ready(ctx) ? 0 : -2;
}

void vlx_seek_shutdown(void) {
    // Nothing process-wide is owned here yet; the AICore integration plugs
    // its runtime shutdown in at this seam.
}
