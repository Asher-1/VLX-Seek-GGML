#!/usr/bin/env python3
"""Convert the HFRE SimpleFPN (object_vp_extractor.simple_fpn) to GGUF.

Layout decisions baked here (mirroring simplefpn_graph.cpp):
  * ConvTranspose2d k2 s2 weight (in, out, 2, 2) -> merged 4-slice GEMM form
    (4*out, in), row k = 2a+b holding W[:, :, a, b]^T; the C++ graph does ONE
    mul_mat per ConvT and re-interleaves the rows in-graph (pixel shuffle).
  * Conv2d 1x1 weight (out, in) -> stored as-is (mul_mat consumes (out, in)
    row-major directly, same as every Linear in this repo).
  * Conv2d 3x3 weight (out, in, 3, 3) -> stored as-is: ggml_im2col's k order
    is c*9 + ky*3 + kx (verified in ggml-cpu/ops.cpp), which equals torch's
    contiguous (OC, IC*3*3) reshape — zero rearrangement needed.
  * LayerNorm weight/bias and ConvT bias: stored as-is (1-D broadcast).
  * nn.GELU() in SimpleFP is exact-erf -> the C++ graph must use
    ggml_gelu_erf (config key vlx.fpn.act=erf is written for the assert).

Level inventory (10B checkpoint, scale_factors=[8,4,2,1] -> simfp_2..5):
  simfp_2 (stride 4) : ConvT x3 (4096->2048->1024->512) + 1x1(512) + 3x3
  simfp_3 (stride 8) : ConvT x2 (4096->2048->1024)  + 1x1(1024->512) + 3x3
  simfp_4 (stride 16): ConvT x1 (4096->2048)        + 1x1(2048->512) + 3x3
  simfp_5 (stride 32): 1x1(4096->512) + 3x3
All convs bias=True except the 1x1/3x3 pair (use_bias = (norm == "") is False
for norm="LN"); the 1x1/3x3 bias is carried by the following LayerNorm.

Usage:
  python3 convert_fpn_to_gguf.py --model-path resources/VLX-Seek-1.5-10B \
      --output cpp_ggml/models/gguf/vlx-fpn-10b-f32.gguf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f32")
    args = ap.parse_args()

    try:
        from gguf import GGUFWriter, GGMLQuantizationType
        from gguf.quants import quantize
    except ImportError:
        sys.exit("pip install gguf is required for conversion")

    model_path = Path(args.model_path)
    tensors = {}
    with safe_open(str(model_path / "model.safetensors"), framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("model.object_vp_extractor.simple_fpn.") or \
               k.startswith("model.mm_projector_aux."):
                tensors[k] = sf.get_tensor(k).float().numpy()
    if not tensors:
        sys.exit("no object_vp_extractor.simple_fpn tensors found")
    print(f"loaded {len(tensors)} SimpleFPN + projector tensors")

    writer = GGUFWriter(args.output, "vlxfpn")
    writer.add_string("general.name", "vlx-seek-simplefpn")
    writer.add_string("general.architecture", "vlxfpn")
    writer.add_uint32("vlx.fpn.out_channels", 512)
    writer.add_uint32("vlx.fpn.vt_dim", 4096)
    # scale_factors [8,4,2,1] on stride 32 -> level strides 4/8/16/32
    writer.add_float32("vlx.fpn.act", 0.0)  # erf marker (assert-only)
    # HFRE vt-branch RoIAlign parameters (10B config: mm_roi_output_size,
    # mm_roi_algined) — single source of truth for the C++ encode_vt call.
    writer.add_uint32("vlx.hfre.roi_output_size", 14)
    writer.add_uint32("vlx.hfre.roi_aligned", 1)
    # aux projector (mm_projector_aux, mlp2x_gelu): Linear(6656->4096) -> GELU
    # -> Linear(4096->4096). Consumes the fused object features; emits the
    # <objfeat> token embeddings (LLM hidden size).
    writer.add_uint32("vlx.auxproj.out_dim", 4096)

    def add(name: str, arr: np.ndarray) -> None:
        # Every 2-D tensor here is a GEMM weight (ConvT merged / 1x1 / 3x3 /
        # aux_proj). bf16 checkpoint -> f16 is lossless; q8_0 needs ne[0]%32.
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.ndim == 2 and args.dtype != "f32":
            if args.dtype == "q8_0" and arr.shape[1] % 32 == 0:
                writer.add_tensor(name, quantize(arr, GGMLQuantizationType.Q8_0),
                                  raw_dtype=GGMLQuantizationType.Q8_0)
                return
            writer.add_tensor(name, arr.astype(np.float16),
                              raw_dtype=GGMLQuantizationType.F16)
            return
        writer.add_tensor(name, arr)

    prefix = "model.object_vp_extractor.simple_fpn."
    for stage in (2, 3, 4, 5):
        p = f"{prefix}simfp_{stage}."
        o = f"fpn.simfp_{stage}."
        n_layers = max(int(k[len(p):].split(".")[0]) for k in tensors
                       if k.startswith(p)) + 1
        for li in range(n_layers):
            q = f"{p}{li}."
            wq = tensors.get(q + "weight")
            if wq is None:
                continue  # GELU
            if wq.ndim == 4 and wq.shape[2] == 2 and wq.shape[3] == 2:
                # ConvTranspose k2s2: merge the four kernel slices into one
                # (4*out, in) GEMM matrix, row k = 2a+b = W[:, :, a, b]^T.
                merged = np.transpose(wq, (2, 3, 1, 0)).reshape(
                    -1, wq.shape[0])
                add(f"{o}{li}.weight_t", merged)
                add(f"{o}{li}.bias", tensors[q + "bias"])
                print(f"simfp_{stage}.{li}: ConvT {wq.shape} -> merged "
                      f"{merged.shape}")
            elif wq.ndim == 4 and wq.shape[2] == 1 and wq.shape[3] == 1:
                add(f"{o}{li}.weight", wq.reshape(wq.shape[0], -1))
            elif wq.ndim == 4 and wq.shape[2] == 3 and wq.shape[3] == 3:
                add(f"{o}{li}.weight", wq.reshape(wq.shape[0], -1))
            elif wq.ndim == 1:
                # standalone LayerNorm layer between ConvTs (flat keys)
                add(f"{o}{li}.weight", wq)
                add(f"{o}{li}.bias", tensors[q + "bias"])
            else:
                sys.exit(f"unrecognized weight {q}weight {wq.shape}")
            if q + "norm.weight" in tensors:
                add(f"{o}{li}.norm.weight", tensors[q + "norm.weight"])
                add(f"{o}{li}.norm.bias", tensors[q + "norm.bias"])

    for li in (0, 2):
        add(f"aux_proj.{li}.weight", tensors[f"model.mm_projector_aux.{li}.weight"].reshape(
            tensors[f"model.mm_projector_aux.{li}.weight"].shape[0], -1))
        add(f"aux_proj.{li}.bias", tensors[f"model.mm_projector_aux.{li}.bias"])

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
