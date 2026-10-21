#!/usr/bin/env python3
"""Full-chain gate: C++ `vlx-cli run --dump-dir` vs the official python run
(dump_official_ref.py -> ref.pt + result.json).

Stages, in pipeline order (abs band is the verdict, corr is auxiliary):
  input_ids       exact: same templates, same tokenizer, same <objN> numbering
  primary_pixels  exact: smart_resize + torch bicubic + (x-127.5)/127.5 in f32
  aux_input       exact: bf16(u8/255) -> AA bilinear -> bf16, (1,3,H,W)
  image_embeds    band:  mmproj output after the official bf16 cast
  object_feats    band:  HFRE + mm_projector_aux after the bf16 cast
  step0_logits    band + argmax equality (the only thing greedy decoding reads)
  step_argmax     exact: every generated token id
  result.json     exact text (visualization_path dropped: it is a local path)

Feature bands are relative to the stage's max |ref| like the other gates. The
reference itself is a bf16 run, so its own representation error is 2^-8 of
each value; a C++ run in f32/f16 weights sits inside that, q8_0 does not have
to (its verdict is the token/JSON equality, which is what the user sees).

Three-way noise verdict (--exact DIR = dump_official_ref.py --dtype float32):
the official bf16 run is itself a noisy estimate of the model, so a band
around it measures "distance to another noisy run". With the f32 run as the
judge, a float stage PASSES when
    ||cpp - exact|| / ||exact||  <=  ratio * ||official_bf16 - exact|| / ||exact||
i.e. the C++ port is no noisier than the official release (ratio 1.0).

Usage:
  python3 cpp_ggml/scripts/compare_official_parity.py --cpp /tmp/vlx-cpp \
      --ref /tmp/vlx-official [--exact /tmp/vlx-official-f32] [--rel 2e-2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def stat(name: str, a: np.ndarray, r: np.ndarray, rel: float) -> bool:
    d = np.abs(a.astype(np.float64) - r.astype(np.float64))
    scale = float(np.abs(r).max()) or 1.0
    corr = float(np.corrcoef(a.ravel(), r.ravel())[0, 1]) if a.size > 1 else 1.0
    ok = d.max() / scale <= rel
    print(f"  {name:15s} max|d|={d.max():.3e} mean|d|={d.mean():.3e} |ref|max={scale:.3e} "
          f"rel={d.max() / scale:.2e} corr={corr:.9f} {'PASS' if ok else 'FAIL'}")
    return ok


def exact(name: str, a: np.ndarray, r: np.ndarray) -> bool:
    if a.shape != r.shape:
        print(f"  {name:15s} shape {a.shape} vs {r.shape} FAIL")
        return False
    n = int((a != r).sum())
    print(f"  {name:15s} {n}/{a.size} differ {'PASS' if n == 0 else 'FAIL'}")
    return n == 0


def noise(name: str, a: np.ndarray, r: np.ndarray, e: np.ndarray, ratio: float) -> bool:
    """C++ (a) vs official bf16 (r), both judged by the f32 official run (e)."""
    e64 = e.astype(np.float64)
    n_e = np.linalg.norm(e64) or 1.0
    d_c = np.linalg.norm(a.astype(np.float64) - e64) / n_e
    d_r = np.linalg.norm(r.astype(np.float64) - e64) / n_e
    ok = d_c <= ratio * d_r
    print(f"  {name:15s} |cpp-exact|={d_c:.3e} |official_bf16-exact|={d_r:.3e} "
          f"ratio={d_c / max(d_r, 1e-30):.2f} (<= {ratio}) {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", default="/tmp/vlx-cpp")
    ap.add_argument("--ref", default="/tmp/vlx-official")
    ap.add_argument("--rel", type=float, default=2e-2, help="image_embeds/object_feats band")
    ap.add_argument("--logit-rel", type=float, default=5e-2, help="step0 logits band")
    ap.add_argument("--exact", default=None, help="official --dtype float32 dump: noise judge")
    ap.add_argument("--noise-ratio", type=float, default=1.0)
    args = ap.parse_args()
    c, rd = Path(args.cpp), Path(args.ref)
    ref = torch.load(rd / "ref.pt", weights_only=False)
    ex = torch.load(Path(args.exact) / "ref.pt", weights_only=False) if args.exact else None
    ok = True

    def band(name: str, a: np.ndarray, r: np.ndarray, key: str, rel: float) -> bool:
        if ex is None or ex.get(key) is None:
            return stat(name, a, r, rel)
        stat(name, a, r, rel)   # informational: distance to the bf16 run
        return noise(name, a, r, ex[key].float().numpy().reshape(-1), args.noise_ratio)

    ids_r = ref["input_ids"].numpy().astype(np.int32)
    ok &= exact("input_ids", np.fromfile(c / "input_ids.i32", np.int32), ids_r)
    pp = ref["primary_pixels"].numpy()
    ok &= exact("primary_pixels", np.fromfile(c / "primary_pixels.f32", np.float32).reshape(-1), pp.reshape(-1))

    for name, key in (("aux_input", "aux_input"), ("image_embeds", "image_embeds"),
                      ("hfre_fused", "hfre_fused"), ("object_feats", "object_feats")):
        p = c / f"{name}.f32"
        if not p.exists() or ref.get(key) is None:
            if not p.exists() and ref.get(key) is not None and ref[key].numel():
                print(f"  {name:15s} missing in C++ dump FAIL")
                ok = False
            continue
        r = ref[key].float().numpy().reshape(-1)
        a = np.fromfile(p, np.float32)
        if a.size != r.size:
            print(f"  {name:15s} size {a.size} vs {r.size} FAIL")
            ok = False
            continue
        ok &= exact(name, a, r) if name == "aux_input" else band(name, a, r, key, args.rel)

    lg_c = np.fromfile(c / "step0_logits.f32", np.float32)
    lg_r = ref["step0_logits"].float().numpy()
    ok &= band("step0_logits", lg_c, lg_r, "step0_logits", args.logit_rel)
    am_ok = int(lg_c.argmax()) == int(lg_r.argmax())
    print(f"  step0 argmax    {int(lg_c.argmax())} vs {int(lg_r.argmax())} {'PASS' if am_ok else 'FAIL'}")
    ok &= am_ok

    sa_c = json.load(open(c / "step_argmax.json"))
    sa_r = [int(t) for t in ref["step_argmax"]]
    first = next((i for i, (x, y) in enumerate(zip(sa_c, sa_r)) if x != y), None)
    sa_ok = sa_c == sa_r
    print(f"  step_argmax     {len(sa_c)} vs {len(sa_r)} tokens"
          f"{'' if first is None else f', first diff at {first}'} {'PASS' if sa_ok else 'FAIL'}")
    ok &= sa_ok

    res_c = json.load(open(c / "result.json"))
    res_c.pop("visualization_path", None)
    txt_c = json.dumps(res_c, ensure_ascii=False, indent=2)
    txt_r = (rd / "result.json").read_text().rstrip()
    js_ok = txt_c == txt_r
    print(f"  result.json     {'identical' if js_ok else 'DIFFERS'} {'PASS' if js_ok else 'FAIL'}")
    ok &= js_ok
    print("OFFICIAL GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
