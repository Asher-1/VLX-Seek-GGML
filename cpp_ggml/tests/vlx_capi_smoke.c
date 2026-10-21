// Pure-C smoke test for the VLX-Seek C API: exercises the AICore-contract
// surface (options -> load -> typed run -> result getters -> render -> free)
// without touching any C++ types. Build with the vlx-seek-capi-smoke target.
//
//   ./vlx-seek-capi-smoke cpp_ggml/models/gguf demo/demo_image.jpg
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vlx/vlx_seek_capi.h"

int main(int argc, char** argv) {
    const char* models_dir = argc > 1 ? argv[1] : "cpp_ggml/models/gguf";
    const char* image = argc > 2 ? argv[2] : "demo/demo_image.jpg";
    const char* text = argc > 3 ? argv[3] : "orange; apple";

    if (vlx_seek_abi_version() != 1) {
        fprintf(stderr, "abi version mismatch\n");
        return 1;
    }

    vlx_seek_options* opts = vlx_seek_options_new();
    if (!opts) return 1;
    vlx_seek_options_set_device(opts, "cuda");
    vlx_seek_options_set_models_dir(opts, models_dir);
    vlx_seek_options_set_max_new_tokens(opts, 64);

    vlx_seek_ctx* ctx = vlx_seek_load_opts(opts);
    vlx_seek_options_free(opts);
    if (!ctx) {
        fprintf(stderr, "load returned NULL\n");
        return 1;
    }
    if (!vlx_seek_is_ready(ctx)) {
        fprintf(stderr, "load failed: %s\n",
                vlx_seek_last_error(ctx) ? vlx_seek_last_error(ctx) : "(none)");
        vlx_seek_free(ctx);
        return 1;
    }
    printf("device=%s threads=%d precision=%s\n", vlx_seek_context_device(ctx),
           vlx_seek_context_threads(ctx), vlx_seek_context_precision(ctx));

    uint8_t* rgb = NULL;
    int32_t w = 0, h = 0;
    if (vlx_seek_load_path_rgb(image, &rgb, &w, &h) != 0) {
        fprintf(stderr, "image decode failed: %s\n", image);
        vlx_seek_free(ctx);
        return 1;
    }
    vlx_image_view img = {rgb, w, h, (size_t)w * 3, VLX_IMAGE_RGB8};

    const int rc = vlx_seek_run_image(ctx, &img, "detection", text, "en", NULL, 0,
                                      NULL, 0);
    if (rc != 0) {
        fprintf(stderr, "run failed rc=%d: %s\n", rc,
                vlx_seek_last_error(ctx) ? vlx_seek_last_error(ctx) : "(none)");
        vlx_seek_free_buffer(rgb);
        vlx_seek_free(ctx);
        return 1;
    }

    printf("answer: %s\n", vlx_seek_result_answer(ctx));
    printf("prompt=%d completion=%d proposals=%d boxes=%d\n",
           vlx_seek_result_prompt_tokens(ctx),
           vlx_seek_result_completion_tokens(ctx),
           vlx_seek_result_proposal_count(ctx),
           vlx_seek_result_box_count(ctx));
    for (int32_t i = 0; i < vlx_seek_result_box_count(ctx); ++i) {
        float b[4];
        if (vlx_seek_result_box(ctx, i, b) == 0)
            printf("  box[%d] <%.0f,%.0f,%.0f,%.0f> %s\n", i, b[0], b[1], b[2],
                   b[3], vlx_seek_result_box_label(ctx, i));
    }
    vlx_pipeline_timings t;
    if (vlx_seek_last_pipeline_timings(ctx, &t) == 0)
        printf("timings: pre=%.1fms infer=%.1fms post=%.1fms e2e=%.1fms\n",
               t.preprocess_ms, t.inference_ms, t.postprocess_ms, t.e2e_ms);
    char* js = vlx_seek_result_json(ctx);
    if (js) {
        printf("json bytes: %zu\n", strlen(js));
        vlx_seek_free_buffer(js);
    }
    char* info = vlx_seek_info_json(ctx);
    if (info) {
        printf("info bytes: %zu\n", strlen(info));
        vlx_seek_free_buffer(info);
    }
    if (vlx_seek_result_render_png(ctx, &img, "/tmp/vlx_capi_render.png") != 0)
        fprintf(stderr, "render failed\n");
    else
        printf("render: /tmp/vlx_capi_render.png\n");

    vlx_seek_free_buffer(rgb);
    vlx_seek_free(ctx);
    vlx_seek_shutdown();
    printf("smoke OK\n");
    return 0;
}
