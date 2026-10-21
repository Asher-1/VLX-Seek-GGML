#!/usr/bin/env python3
"""Generate a shrunken C-RADIOv4 checkpoint for topology parity.

The RADIO parity gate verifies GRAPH TOPOLOGY and TENSOR LAYOUT (patch gather
order, CPE resize+crop, cls/register prefix, pre-norm block structure, tap
points), not weight values. A synthetic checkpoint with identical tensor
shapes-but-tiny exercises the full tool chain (convert -> torch dump -> C++ ->
compare) on CPU in seconds, without the 10B download. Real-weight parity runs
after the full checkpoint lands.

Layout matches the real 10B checkpoint keys so the converter's fuzzy matching
is exercised end to end. Scale-down vs 10B:
  dim 1152->64, heads 16->4, depth 27->4, pos grid 128->8,
  cls 3 + reg 7 -> cls 1 + reg 1, mlp 4304->256.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

PREFIX = "language_model.model.vision_tower_aux.image_tower.radio_model.model"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dim, heads, depth, mlp, P, pos_grid = 64, 4, 4, 256, 16, 8
    hd = dim // heads
    n_skip = 2  # 1 cls + 1 register

    t: dict[str, np.ndarray] = {}
    t[f"{PREFIX}.patch_generator.embedder.weight"] = (
        rng.standard_normal((dim, 3 * P * P)) * 0.05
    ).astype(np.float32)
    t[f"{PREFIX}.patch_generator.pos_embed"] = (
        rng.standard_normal((1, pos_grid * pos_grid, dim)) * 0.02
    ).astype(np.float32)
    t[f"{PREFIX}.patch_generator.cls_token.token"] = (
        rng.standard_normal((n_skip, dim)) * 0.02
    ).astype(np.float32)

    for i in range(depth):
        p = f"{PREFIX}.blocks.{i}"
        t[f"{p}.norm1.weight"] = np.ones(dim, np.float32)
        t[f"{p}.norm1.bias"] = (rng.standard_normal(dim) * 0.01).astype(np.float32)
        t[f"{p}.norm2.weight"] = np.ones(dim, np.float32)
        t[f"{p}.norm2.bias"] = (rng.standard_normal(dim) * 0.01).astype(np.float32)
        t[f"{p}.attn.qkv.weight"] = (rng.standard_normal((3 * dim, dim)) * 0.05).astype(np.float32)
        t[f"{p}.attn.qkv.bias"] = (rng.standard_normal(3 * dim) * 0.01).astype(np.float32)
        t[f"{p}.attn.proj.weight"] = (rng.standard_normal((dim, dim)) / hd**0.5).astype(np.float32)
        t[f"{p}.attn.proj.bias"] = (rng.standard_normal(dim) * 0.01).astype(np.float32)
        t[f"{p}.ls1.gamma"] = (1.0 + rng.standard_normal(dim) * 0.01).astype(np.float32)
        t[f"{p}.ls2.gamma"] = (1.0 + rng.standard_normal(dim) * 0.01).astype(np.float32)
        t[f"{p}.mlp.fc1.weight"] = (rng.standard_normal((mlp, dim)) * 0.05).astype(np.float32)
        t[f"{p}.mlp.fc1.bias"] = (rng.standard_normal(mlp) * 0.01).astype(np.float32)
        t[f"{p}.mlp.fc2.weight"] = (rng.standard_normal((dim, mlp)) * 0.05).astype(np.float32)
        t[f"{p}.mlp.fc2.bias"] = (rng.standard_normal(dim) * 0.01).astype(np.float32)

    save_file(t, str(out / "model.safetensors"))

    # config.json mirrors the 10B fields the converter consumes.
    cfg = {
        "model_type": "vlx_seek_1_5",
        "mm_object_hidden_size": 6656,
        "mm_roi_output_size": 14,
        "mm_roi_algined": False,
        "mm_add_pos_embed": True,
        "mm_object_feature_combination": "concat",
        "mm_use_simpleFPN_for_vt": True,
        "mm_simpleFPN_out_channels_for_vt": 512,
        "vision_config": {"patch_size": 16, "spatial_merge_size": 2},
        "vision_aux_config": {
            "args": {
                "model": "vit_so400m_patch16_224",
                "cpe_max_size": pos_grid * P,
                "cpe_num_registers": None,
                "num_cls_tokens": 1,
                "register_multiple": 10,
                "dtype": "float32",
            },
            "patch_size": P,
            "max_resolution": pos_grid * P,
        },
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"synth radio written to {out} "
          f"(dim{dim} heads{heads} depth{depth} pos{pos_grid} skip{n_skip})")


if __name__ == "__main__":
    main()
