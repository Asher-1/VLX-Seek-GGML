// EXPERIMENTAL GPU RoIAlign — NOT wired into the runtime.
//
// Status: the gather plan (pre_calc_bilinear semantics, [n][bin][tap]
// layout, zero-padded uniform capacity) is validated and reusable, but the
// pure-ggml-primitive graph materializes the (C, NP) gather output —
// at C=1152, N=100 that is ~23 GiB, which is not viable. The correct path
// is a fused gather+weight+pool CUDA kernel (a custom ggml op); keep this
// module for its validated precompute and as the reference for that work.
//
// Math: the official RoIAlign mean-pool collapses to
//   out[n, c] = sum(all taps w_t * feat[level(n), flat_pos_t, c])
//               / (count_n * out_size^2)
// because count (per-box sampling density) and denom (out_size^2) are
// per-box constants — the per-bin means merge into one final division.
// The gather rows are precomputed on the host with the exact torchvision
// CPU-kernel sampling geometry (pre_calc_bilinear), laid out
// [n][bin][tap x corner] and zero-padded to a uniform per-bin capacity.
// The float reduction order differs from the host kernel (f32 vs cascaded
// double across bins) — within the parity band, verified per level.
//
// The feature levels arrive on-device as (C, H, W) f32 tensors; the gather
// source is a FREE reshape_2d view (H*W, C) — NCHW memory IS the transposed
// row-major layout, no permute/copy needed.
#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "vlx_types.hpp"

#include <string>
#include <vector>

namespace vlx {

// One gather plan for a stack of same-channel levels (aux: 4 x 1152 with a
// shared grid; vt: one plan per SimpleFPN level). Rows are laid out
// [n][bin][tap x corner]; taps beyond a box's own grid density carry zero
// weight so the uniform capacity is mathematically neutral.
struct RoiGpuPlan {
    int n_boxes = 0;
    int out_size = 0;
    int channels = 0;
    int cap_taps = 0;               // per-bin tap capacity (incl. 4-corner expansion)
    std::vector<int32_t> idx;       // [NP4] row into (H*W, C) of the level's
                                    // concatenated (per-level offset added)
    std::vector<float> w;           // [NP4] bilinear weights (zero padding)
    std::vector<float> count;       // [n] per-box sampling density
};

// Builds the gather plan for `level` over `boxes` (torchvision semantics:
// roi_aligned flag, ceil-grid sampling density, edge-snapped taps).
RoiGpuPlan plan_roi_gpu(const FeatureLevel& level_meta, const std::vector<Box>& boxes_px,
                        float spatial_scale, int out_size, bool aligned,
                        int32_t level_offset = 0);

// Executes the pooling graph for ONE level: uploads its (C,H,W) f32 NCHW
// host data, builds the gather + weighted-sum graph, runs it on `backend`
// and reads back out_nxc (N, C) row-major. For the aux branch call it once
// with the channel-concatenated level; for the vt branch once per level
// (the caller interleaves the per-level blocks afterwards, like
// HfreEncoder::encode_vt does).
bool roi_align_pool_one(ggml_backend_t backend, const FeatureLevel& level,
                        const RoiGpuPlan& plan, std::vector<float>* out_nxc,
                        std::string* err);

}  // namespace vlx
