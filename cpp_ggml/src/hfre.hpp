// HFRE: Hybrid Fine-Grained Region Encoder, host-side C++ port.
//
// Port of hybrid_finegrained_region_encoder.py for the released 10B config:
//   mm_use_vision_tower_object_feature=True, mm_use_simpleFPN_for_vt=True,
//   combination="concat", roi_output_size=14, pos dim = 6656 (= 4608 + 2048).
//
//   aux path : 4 RADIO levels (same 64x64 grid) -> channel concat (4608)
//              -> RoIAlign 14x14 (scale 1/16) -> mean pool        (N, 4608)
//   vt path  : main-tower merged feature -> SimpleFPN 4 levels x 512
//              -> per-level RoIAlign -> mean -> concat            (N, 2048)
//   fusion   : concat[aux, vt]                                     (N, 6656)
//   position : sine embedding of normalized cxcywh (from vt-path boxes)
//              added to the concat vector                          (+6656)
//
// The aux branch has NO learned parameters — parity against
// torchvision.ops.roi_align alone validates it. The SimpleFPN conv tower
// itself runs inside the P1 mmproj graph; HFRE consumes its 4 levels.
#pragma once

#include "vlx_types.hpp"

struct ggml_tensor;   // device-resident level tensors (CUDA RoIAlign only)

#include <string>
#include <vector>

namespace vlx {

// One bilinear sampling tap of a RoIAlign grid point (torchvision
// pre_calc_for_bilinear_interpolate semantics: edge-snapped 4-neighbor
// interpolation). Exported so the GPU RoIAlign path can precompute the
// gather indices/weights with the exact same sampling geometry.
struct PreCalc {
    int pos1, pos2, pos3, pos4;
    float w1, w2, w3, w4;
};

// Line-for-line port of torchvision 0.25 cpu/roi_align_common.h
// pre_calc_for_bilinear_interpolate (fills bin-major: ph, pw, grid_y,
// grid_x; zero-weight taps when a sample falls fully outside).
void pre_calc_bilinear(int height, int width, int pooled, float roi_start_h,
                       float roi_start_w, float bin_h, float bin_w, int grid_h,
                       int grid_w, std::vector<PreCalc>* out);

class GgufFile;

class HfreEncoder {
public:
    explicit HfreEncoder(const HfreConfig& cfg);

    // aux-only encode (validated directly against torchvision; the vt branch
    // needs the P1 main-tower tap and is appended through encode_vt()).
    bool encode_aux(const std::vector<Box>& boxes,
                    const std::vector<FeatureLevel>& aux_levels,
                    std::vector<float>* out_aux_nxc,
                    std::string* err = nullptr);

    // vt branch: consumes the 4 SimpleFPN levels produced by the mmproj graph.
    // levels are ordered as Python's simple_fpn output (level 0 first).
    // feature_strides must match the FPN pyramid (level 0 = finest).
    bool encode_vt(const std::vector<Box>& boxes,
                   const std::vector<FeatureLevel>& vt_fpn_levels,
                   const std::vector<float>& feature_strides,
                   std::vector<float>* out_vt_nxc,
                   std::string* err = nullptr);

    // Full fusion: concat(aux, vt) + sine pos embed of the boxes in
    // ORIGINAL-image pixels. Returns (N, final_dim) row-major.
    bool fuse(const std::vector<float>& aux_nxc,
              const std::vector<float>& vt_nxc,
              const std::vector<Box>& boxes,
              int aux_channels, int vt_channels,
              const std::vector<FeatureLevel>& vt_grid_ref,
              std::vector<float>* out_object_features,
              std::string* err = nullptr);

    // Torchvision-parity RoIAlign on a single NCHW level.
    // boxes are in ORIGINAL image pixels; scaled by spatial_scale inside.
    static void roi_align_mean_pool(const FeatureLevel& level,
                                    const std::vector<Box>& boxes_px,
                                    float spatial_scale, int out_size,
                                    bool aligned,
                                    std::vector<float>* out_nxc,
                                    int n_threads = 1);

private:
    // sine embedding: input (N,4) normalized cxcywh -> (N, pos_embedding_dim)
    // (block order y, x, w, h; each block pos_embedding_dim/4 wide).
    void sine_embed(const std::vector<std::array<float, 4>>& cxcywh_norm,
                    std::vector<float>* out_nxd) const;

    HfreConfig cfg_;
};


// ---- fused CUDA RoIAlign (roi_align_cuda.cu; CUDA builds only) ----
// Drop-in GPU equivalents of encode_aux/encode_vt's pooling stage: same
// contract, same outputs (bands, not bitwise — the kernel accumulates in
// tap order like the host port). On any CUDA error these return the error
// string and the caller falls back to the host path.
std::string cuda_encode_aux(const std::vector<FeatureLevel>& aux_levels,
                            const std::vector<Box>& boxes, int out_size,
                            bool aligned, float aux_spatial_scale,
                            std::vector<float>* out_aux_nxc);
std::string cuda_encode_vt(const std::vector<FeatureLevel>& vt_levels,
                           const std::vector<float>& strides,
                           const std::vector<Box>& boxes, int out_size,
                           bool aligned, std::vector<float>* out_vt_nxc);


// ---- fused CUDA RoIAlign over GPU-resident levels (roi_align_cuda.cu;
// CUDA builds only) ---- // The levels are graph output tensors already on
// the device (RadioGraph/SimpleFpnGraph::out_level_gpu) — no host
// read-back, no re-upload, no host channel concat. Same output contracts
// as the host-path equivalents; any CUDA error returns the error string
// and the caller falls back to the host path (re-reading the levels via
// ggml_backend_tensor_get on the same tensors).
std::string cuda_encode_aux_dev(ggml_tensor* const* level_tensors, int n_levels,
                                const std::vector<Box>& boxes, int out_size,
                                bool aligned, float aux_spatial_scale,
                                std::vector<float>* out_aux_nxc,
                                float** d_out_out = nullptr);
// Fuses the device-resident pool outputs with the sine position embedding
// (HfreEncoder::fuse + sine_embed semantics) entirely on the GPU and reads
// back the (N, D) float rows run_mlp2x_gelu consumes. img_w/img_h are the
// position-embed frame extents (= vt_grid_ref[0].width/height).
std::string cuda_fuse_dev(const float* d_aux, const float* d_vt0,
                          const float* d_vt1, const float* d_vt2,
                          const float* d_vt3, const std::vector<Box>& boxes,
                          float img_w, float img_h, int aux_C, int vt_block,
                          int D, std::vector<float>* out_fused);
std::string cuda_pool_dev(ggml_tensor* level_t, const std::vector<Box>& boxes,
                          float spatial_scale, int out_size, bool aligned,
                          std::vector<float>* out_nxc,
                          float** d_out_out = nullptr);


// Frees device pointers returned by cuda_encode_aux_dev/cuda_pool_dev.
void cuda_free_dev_ptrs(const float* const* ptrs, int n);
}  // namespace vlx
