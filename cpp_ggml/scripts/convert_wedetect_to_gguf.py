#!/usr/bin/env python3
"""Convert WeDetect-Uni (resources/wedetect_base_uni.pth) to GGUF.

The checkpoint is loaded into detect_tools' SimpleYOLOWorldDetector with the
EXACT key remap of inference.py::load_wedetect_proposals (copied verbatim,
then asserted strict: no missing key), so the converted weights are the
weights the official pipeline runs.

Algebraic folds baked here (all exact in real arithmetic; f32 rounding of
the folded products is ~1 ulp and covered by the abs-band gate):
  * ConvModule (Conv no-bias + BatchNorm eval + act): W' = W * g/sqrt(v+eps),
    b' = beta - mean * g/sqrt(v+eps). Neck BN eps = 1e-5 (nn default), head BN
    eps = 1e-3 (module attribute — read from the module, never assumed).
  * ConvNeXt layer-scale gamma folded into pwconv2: W2' = gamma[:,None]*W2,
    b2' = gamma*b2.
  * cls head: cls_contrasts[l].norm (BN eps 1e-3) folded into the cls 1x1
    conv; exp(logit_scale_l) folded into the prompt embeddings, bias_l
    becomes a (256,) broadcast vector -> logits = E_l @ ce + b_l in ONE
    mul_mat per level.

Layout (ggml ne is the numpy shape reversed):
  * Conv kxk weight (O, I, k, k) -> (O, I*k*k): ggml_im2col's k order
    c*k*k + ky*k + kx equals torch's contiguous reshape.
  * depthwise 7x7 (C, 1, 7, 7) stored as-is -> ne=(7, 7, 1, C), the kernel
    layout ggml_conv_2d_dw_direct expects.
  * ConvTranspose k2s2 (I, O, 2, 2) -> merged (4*O, I), row k = 2a+b holds
    W[:, :, a, b]^T (same 4-slice GEMM form as convert_fpn_to_gguf.py).

--dtype f16 / q8_0 stores the large GEMM matrices (2-D, ne[0] % 32 == 0 for
q8_0, else f16) quantized; every 1-D tensor and the depthwise kernels stay
f32 (they are added/broadcast, not multiplied by mul_mat).

Usage:
  python3 cpp_ggml/scripts/convert_wedetect_to_gguf.py \
      --checkpoint resources/wedetect_base_uni.pth \
      --output cpp_ggml/models/gguf/wedetect-base-uni-f32.gguf [--dtype f32]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "detect_tools" / "WeDetect"))


def load_official(checkpoint: str):
    """Mirror inference.py::load_wedetect_proposals key handling verbatim."""
    from generate_proposal import SimpleYOLOWorldDetector
    model = SimpleYOLOWorldDetector(backbone_size="base", prompt_dim=768,
                                    num_prompts=256, num_proposals=100)
    state_dict = torch.load(checkpoint, map_location="cpu")
    state_dict = state_dict.get("state_dict", state_dict)
    for key in list(state_dict.keys()):
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            state_dict[new_key] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            state_dict[new_key] = state_dict.pop(key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # the official call is strict=False; assert nothing the forward uses is
    # left at its random init (only the unused classifier/norm may be extra).
    assert not missing, f"official remap leaves params uninitialized: {missing}"
    assert set(unexpected) <= {"backbone.norm.weight", "backbone.norm.bias",
                               "backbone.head.weight", "backbone.head.bias"}, unexpected
    return model.eval()


def bn_fold(conv_w: np.ndarray, conv_b, bn) -> tuple[np.ndarray, np.ndarray]:
    g, beta = bn.weight.detach().double(), bn.bias.detach().double()
    s = (g / torch.sqrt(bn.running_var.double() + bn.eps)).numpy()
    t = beta.numpy() - bn.running_mean.double().numpy() * s
    w = conv_w.astype(np.float64) * s.reshape(-1, *([1] * (conv_w.ndim - 1)))
    b = t if conv_b is None else conv_b.astype(np.float64) * s + t
    return w.astype(np.float32), b.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "resources/wedetect_base_uni.pth"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--dtype", choices=("f32", "f16", "q8_0"), default="f32")
    args = ap.parse_args()

    from gguf import GGUFWriter, GGMLQuantizationType
    from gguf.quants import quantize

    det = load_official(args.checkpoint)
    torch.set_grad_enabled(False)
    writer = GGUFWriter(args.output, "wedetect")
    writer.add_string("general.name", "wedetect-base-uni")
    writer.add_uint32("wd.img_size", det.img_size[0])
    writer.add_uint32("wd.num_prompts", det.embeddings.shape[0])
    writer.add_uint32("wd.prompt_dim", det.embeddings.shape[1])
    writer.add_uint32("wd.reg_max", det.bbox_head.reg_max)
    writer.add_uint32("wd.num_proposals", 100)   # inference.py num_proposals
    n_q = [0, 0]

    def add(name: str, arr: np.ndarray, gemm: bool = False) -> None:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if gemm and arr.ndim == 2 and args.dtype != "f32":
            if args.dtype == "q8_0" and arr.shape[1] % 32 == 0:
                # quantize() returns the uint8 byte matrix; the writer derives
                # the logical shape from it (raw_shape would be read as bytes).
                writer.add_tensor(name, quantize(arr, GGMLQuantizationType.Q8_0),
                                  raw_dtype=GGMLQuantizationType.Q8_0)
                n_q[0] += 1
                return
            writer.add_tensor(name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)
            n_q[1] += 1
            return
        writer.add_tensor(name, arr)

    def npy(t: torch.Tensor) -> np.ndarray:
        return t.detach().float().numpy()

    # ---------------- backbone (ConvNeXt-B) ----------------
    bb = det.backbone
    stem_conv, stem_ln = bb.downsample_layers[0][0], bb.downsample_layers[0][1]
    add("bb.stem.w", npy(stem_conv.weight).reshape(stem_conv.out_channels, -1), gemm=True)
    add("bb.stem.b", npy(stem_conv.bias))
    add("bb.stem.ln.w", npy(stem_ln.weight))
    add("bb.stem.ln.b", npy(stem_ln.bias))
    for i in range(1, 4):
        ln, conv = bb.downsample_layers[i][0], bb.downsample_layers[i][1]
        add(f"bb.ds{i}.ln.w", npy(ln.weight))
        add(f"bb.ds{i}.ln.b", npy(ln.bias))
        add(f"bb.ds{i}.w", npy(conv.weight).reshape(conv.out_channels, -1), gemm=True)
        add(f"bb.ds{i}.b", npy(conv.bias))
    depths = []
    for si, stage in enumerate(bb.stages):
        depths.append(len(stage))
        for bi, blk in enumerate(stage):
            p = f"bb.s{si}.{bi}."
            add(p + "dw.w", npy(blk.dwconv.weight))            # (C,1,7,7) -> ne (7,7,1,C)
            add(p + "dw.b", npy(blk.dwconv.bias))
            add(p + "ln.w", npy(blk.norm.weight))
            add(p + "ln.b", npy(blk.norm.bias))
            add(p + "pw1.w", npy(blk.pwconv1.weight), gemm=True)
            add(p + "pw1.b", npy(blk.pwconv1.bias))
            g = blk.gamma.detach().double().numpy()
            w2 = (blk.pwconv2.weight.detach().double().numpy() * g[:, None]).astype(np.float32)
            b2 = (blk.pwconv2.bias.detach().double().numpy() * g).astype(np.float32)
            add(p + "pw2.w", w2, gemm=True)
            add(p + "pw2.b", b2)
    writer.add_array("wd.depths", depths)

    # ---------------- neck (CSPRepBiFPANNeck) ----------------
    # Every ConvModule_torch is emitted under its python module path, so the
    # C++ graph addresses weights by the same names as the reference module.
    n_conv = 0
    for name, mod in det.neck.named_modules():
        cls = type(mod).__name__
        if cls == "ConvModule_torch":
            # name ends with ".block"; strip it so C++ uses e.g. "Rep_p4.cv1"
            key = "neck." + name[: -len(".block")]
            w, b = bn_fold(npy(mod.conv.weight), None, mod.bn)
            add(key + ".w", w.reshape(w.shape[0], -1), gemm=True)
            add(key + ".b", b)
            n_conv += 1
        elif cls == "Transpose":
            wt = npy(mod.upsample_transpose.weight)            # (I, O, 2, 2)
            merged = np.transpose(wt, (2, 3, 1, 0)).reshape(-1, wt.shape[0])
            add("neck." + name + ".wt", merged, gemm=True)
            add("neck." + name + ".b", npy(mod.upsample_transpose.bias))
        elif cls == "BottleRep":
            add("neck." + name + ".alpha", npy(mod.alpha).reshape(1))
    print(f"neck: {n_conv} folded ConvModules")

    # ---------------- head ----------------
    E = det.embeddings.detach().double().numpy()             # (256, 768)
    for li in range(3):
        for kind in ("reg", "cls"):
            seq = (det.bbox_head.reg_preds if kind == "reg" else det.bbox_head.cls_preds)[li]
            for j, (ci, bi) in enumerate(((0, 1), (3, 4))):
                w, b = bn_fold(npy(seq[ci].weight), None, seq[bi])
                add(f"head.{li}.{kind}.{j}.w", w.reshape(w.shape[0], -1), gemm=True)
                add(f"head.{li}.{kind}.{j}.b", b)
            w = npy(seq[6].weight).reshape(seq[6].out_channels, -1)
            b = npy(seq[6].bias)
            if kind == "cls":
                w, b = bn_fold(w, b, det.bbox_head.cls_contrasts[li].norm)
            add(f"head.{li}.{kind}.2.w", w, gemm=(kind == "cls"))
            add(f"head.{li}.{kind}.2.b", b)
        cc = det.bbox_head.cls_contrasts[li]
        scale = float(torch.exp(cc.logit_scale.detach().double()))
        add(f"head.{li}.emb.w", (E * scale).astype(np.float32), gemm=True)
        add(f"head.{li}.emb.b", np.full(E.shape[0], float(cc.bias), np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"wrote {args.output} (dtype={args.dtype}: {n_q[0]} q8_0, {n_q[1]} f16 GEMM tensors)")


if __name__ == "__main__":
    main()
