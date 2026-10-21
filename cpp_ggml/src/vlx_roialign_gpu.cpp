// GPU RoIAlign implementation — see vlx_roialign_gpu.hpp for the contract.
//
// Graph per pooled output (aux: one graph over the concatenated RADIO
// levels; vt: one graph per SimpleFPN level):
//   src  = concat(reshape_2d(level_l, C, H_l*W_l))   # (C, sum HW) free views
//   g    = get_rows(src, idx)                        # (C, NP) columns = taps
//   wm   = mul(g, w)                                 # w (1, NP) broadcast
//   wm4  = reshape_4d(wm, C, 196, cap4, N)           # column split [n][bin][s4]
//   p1   = permute(wm4, 2, 0, 1, 3)                  # s4 -> ne0
//   s1   = sum_rows(p1)                              # (1, C, 196, N)
//   p2   = permute(s1, 2, 0, 1, 3)                   # 196 -> ne0
//   s2   = sum_rows(p2) / (count * 196)              # (1, C, N, 1)
// The final tensor's memory layout is [N][C] row-major — read back directly
// into out_nxc. Zero-padded taps contribute zero to the numerator; the
// divisor never contains the padding capacity.
#include "vlx_roialign_gpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "hfre.hpp"

namespace vlx {

RoiGpuPlan plan_roi_gpu(const FeatureLevel& level_meta,
                        const std::vector<Box>& boxes_px, float spatial_scale,
                        int out_size, bool aligned, int32_t level_offset) {
    RoiGpuPlan plan;
    plan.n_boxes = int32_t(boxes_px.size());
    plan.out_size = out_size;
    plan.channels = level_meta.channels;

    const int C = level_meta.channels, H = level_meta.height, W = level_meta.width;
    int cap_taps = 0;
    std::vector<std::vector<PreCalc>> per_box(boxes_px.size());
    for (size_t bi = 0; bi < boxes_px.size(); ++bi) {
        const Box& b = boxes_px[bi];
        const float offset = aligned ? 0.5f : 0.f;
        const float rsw = b.x1 * spatial_scale - offset;
        const float rsh = b.y1 * spatial_scale - offset;
        float rw = (b.x2 * spatial_scale - offset) - rsw;
        float rh = (b.y2 * spatial_scale - offset) - rsh;
        if (!aligned) { rw = std::max(rw, 1.f); rh = std::max(rh, 1.f); }
        const float bin_h = rh / float(out_size), bin_w = rw / float(out_size);
        // Degenerate boxes (negative width/height can reach here when
        // aligned=true skips the max(rw,1) guard) clamp to zero taps: the
        // host kernel handles them the same way — per_bin <= 0 means the
        // bin contributes nothing, and the divisor uses count >= 1.
        const int grid_h = std::max(0, int(std::ceil(rh / out_size)));
        const int grid_w = std::max(0, int(std::ceil(rw / out_size)));
        pre_calc_bilinear(H, W, out_size, rsh, rsw, bin_h, bin_w, grid_h, grid_w,
                          &per_box[bi]);
        cap_taps = std::max(cap_taps, grid_h * grid_w);
        plan.count.push_back(float(std::max(grid_h * grid_w, 1)));
    }
    plan.cap_taps = cap_taps;
    // Row order [n][bin][tap x corner]; zero-fill to the uniform capacity.
    const int bins = out_size * out_size;
    plan.idx.reserve(size_t(boxes_px.size()) * bins * cap_taps * 4);
    plan.w.reserve(size_t(boxes_px.size()) * bins * cap_taps * 4);
    for (size_t bi = 0; bi < boxes_px.size(); ++bi) {
        const auto& pc = per_box[bi];
        size_t k = 0;
        for (int bin = 0; bin < bins; ++bin) {
            for (int t = 0; t < cap_taps; ++t) {
                if (k < pc.size()) {
                    const PreCalc& p = pc[k++];
                    plan.idx.push_back(int32_t(p.pos1 + level_offset));
                    plan.idx.push_back(int32_t(p.pos2 + level_offset));
                    plan.idx.push_back(int32_t(p.pos3 + level_offset));
                    plan.idx.push_back(int32_t(p.pos4 + level_offset));
                    plan.w.push_back(p.w1);
                    plan.w.push_back(p.w2);
                    plan.w.push_back(p.w3);
                    plan.w.push_back(p.w4);
                } else {
                    plan.idx.push_back(int32_t(level_offset));  // row 0, zero weight
                    plan.idx.push_back(int32_t(level_offset));
                    plan.idx.push_back(int32_t(level_offset));
                    plan.idx.push_back(int32_t(level_offset));
                    plan.w.push_back(0.f);
                    plan.w.push_back(0.f);
                    plan.w.push_back(0.f);
                    plan.w.push_back(0.f);
                }
            }
        }
    }
    return plan;
}

bool roi_align_pool_one(ggml_backend_t backend, const FeatureLevel& level,
                        const RoiGpuPlan& plan, std::vector<float>* out_nxc,
                        std::string* err) {
    const int C = level.channels, H = level.height, W = level.width;
    const int N = plan.n_boxes, bins = plan.out_size * plan.out_size;
    const int64_t NP4 = int64_t(N) * bins * plan.cap_taps * 4;
    if (plan.idx.size() != size_t(NP4) || plan.w.size() != size_t(NP4) ||
        plan.count.size() != size_t(N)) {
        if (err) *err = "plan shape mismatch";
        return false;
    }

    const int64_t bufsize = ggml_tensor_overhead() * 64 + ggml_graph_overhead();
    ggml_init_params ip = {size_t(bufsize), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) { if (err) *err = "ggml_init failed"; return false; }

    // create tensors first, allocate their buffer, then upload (ggml 0.26
    // requires an explicit backend buffer before tensor_set).
    ggml_tensor* level_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, W, H, C);
    ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NP4);
    ggml_tensor* w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, NP4);
    ggml_tensor* count_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, 1, N);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(level_t, level.data.data(), 0,
                            level.data.size() * sizeof(float));
    ggml_backend_tensor_set(idx_t, plan.idx.data(), 0, NP4 * sizeof(int32_t));
    ggml_backend_tensor_set(w_t, plan.w.data(), 0, NP4 * sizeof(float));
    ggml_backend_tensor_set(count_t, plan.count.data(), 0, N * sizeof(float));

    // gather source: get_rows copies ne0 rows and indexes ne1, so the
    // channel dim must be ne0. NCHW memory is channel-major, which is the
    // (H*W, C) row-major view — one real transpose (cont) materializes the
    // (C, H*W) layout the gather needs.
    ggml_tensor* src = ggml_cont(ctx, ggml_permute(
        ctx, ggml_reshape_2d(ctx, level_t, int64_t(H) * W, C), 1, 0, 2, 3));
    ggml_tensor* g = ggml_get_rows(ctx, src, idx_t);       // (C, NP)
    ggml_tensor* wm = ggml_mul(ctx, g, ggml_reshape_2d(ctx, w_t, 1, NP4));
    ggml_tensor* wm4 = ggml_reshape_4d(ctx, wm, C, bins, plan.cap_taps * 4, N);
    ggml_tensor* p1 = ggml_permute(ctx, wm4, 2, 0, 1, 3);  // taps -> ne0
    ggml_tensor* s1 = ggml_sum_rows(ctx, p1);              // (1, C, bins, N)
    ggml_tensor* p2 = ggml_permute(ctx, s1, 2, 0, 1, 3);   // bins -> ne0
    ggml_tensor* s2 = ggml_sum_rows(ctx, p2);              // (1, 1, C, N)
    ggml_tensor* out = ggml_div(ctx, s2, count_t);
    out = ggml_scale(ctx, out, 1.f / bins);

    ggml_cgraph* cgraph = ggml_new_graph(ctx);
    ggml_build_forward_expand(cgraph, out);
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(galloc, cgraph);
    ggml_backend_graph_compute(backend, cgraph);
    ggml_gallocr_free(galloc);

    out_nxc->resize(size_t(N) * C);
    ggml_backend_tensor_get(out, out_nxc->data(), 0, size_t(N) * C * sizeof(float));
    ggml_free(ctx);
    return true;
}

}  // namespace vlx
