# llama-patches

Idempotent patches against the vendored llama.cpp (`e467e5b5e`, the pin whose
vendored ggml is >= v0.21-level). Replayed at CMake configure time by
`cmake/apply_patches.cmake` (stamp: `third_party/llama.cpp/.vlx-patches-applied`).

## 0001-vlx-prec-f32-strict-gemm.patch

**Why**: `GGML_PREC_F32` on an F32 × F32 `mul_mat` requests IEEE fp32
accumulation, but the CUDA handle defaults to `CUBLAS_TF32_TENSOR_OP_MATH`
(10-bit mantissa inputs) — silently degrading the f32 precision contract that
the parity gates are built on. The patch drops to default math for that call
(and restores on every exit path), and makes the matching adjustment in the
Vulkan matmul pipeline key. Without it, `--precision f32` runs would use TF32
on Ampere+ and the three-way f32 referee would fail.

## Contract

- Patches apply to the submodule working tree at configure time; the dirty
  submodule is the expected local state. Cloners get the same result
  automatically — commit nothing inside `third_party/llama.cpp`.
- To add a patch: increment the `000N-` prefix, keep it applicable to the
  pinned revision, and re-run cmake (the stamp regenerates).
