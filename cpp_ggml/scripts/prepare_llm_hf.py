#!/usr/bin/env python3
"""Extract the VLX-Seek LLM into a standard Qwen3.5-for-causal-lm HF directory
consumable by llama.cpp's convert_hf_to_gguf.py (Qwen3_5TextModel is already
registered there — no converter patches needed).

What it does:
  * copies model.language_model.* -> model.* (drops the wrapper prefix) and
    lm_head.weight, streaming tensor-by-tensor (memory-safe for 20 GB)
  * writes config.json with architectures=["Qwen3_5ForCausalLM"] and the
    text_config fields hoisted to the top level
  * writes a PLACEHOLDER tokenizer.json when the real one is unavailable
    (248182 dummy normal tokens): the converted GGUF then validates the whole
    weight/tensor pipeline but CANNOT generate text. Re-run with the real
    tokenizer.json (VLX-Seek HF repo / Qwen3.5 base + added tokens) to get a
    usable model — only the vocab section changes.

Usage:
  python3 prepare_llm_hf.py --model-path resources/VLX-Seek-1.5-10B \
      --out /tmp/vlx-llm-hf [--real-tokenizer.json path]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="resources/VLX-Seek-1.5-10B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--real-tokenizer-json", default=None,
                    help="path to the real tokenizer.json (248182-vocab BPE); "
                         "omit for a placeholder vocab")
    args = ap.parse_args()

    src = Path(args.model_path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    tc = cfg["text_config"]
    # transformers v5 nests RoPE under rope_parameters (theta 1e7 for the 10B);
    # a silent fallback here once shipped freq_base 5e6 -> every full-attention
    # layer rotated at the wrong frequencies (tokens still matched, logits not).
    rope = dict(tc.get("rope_parameters") or {})
    rope_theta = rope.get("rope_theta", tc.get("rope_theta"))
    if rope_theta is None:
        sys.exit("config.json text_config has no rope_theta / rope_parameters.rope_theta")
    new_cfg = {
        "architectures": ["Qwen3_5TextModel"],
        "model_type": "qwen35",
        "vocab_size": tc["vocab_size"],
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc.get("head_dim"),
        "hidden_act": tc.get("hidden_act", "silu"),
        "attention_bias": tc.get("attention_bias", False),
        "attn_output_gate": tc.get("attn_output_gate", True),
        "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim", 4),
        "linear_key_head_dim": tc.get("linear_key_head_dim"),
        "linear_value_head_dim": tc.get("linear_value_head_dim"),
        "linear_num_key_heads": tc.get("linear_num_key_heads"),
        "linear_num_value_heads": tc.get("linear_num_value_heads"),
        "full_attention_interval": tc.get("full_attention_interval"),
        "layer_types": tc.get("layer_types"),
        "max_position_embeddings": tc.get("max_position_embeddings", 262144),
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "rope_parameters": rope,
        "partial_rotary_factor": tc.get("partial_rotary_factor", rope.get("partial_rotary_factor")),
        "tie_word_embeddings": tc.get("tie_word_embeddings", False),
        "bos_token_id": tc.get("bos_token_id", 151643),
        "eos_token_id": tc.get("eos_token_id", 151645),
        "dtype": "float32",
    }
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2))

    # ---- tensors: stream copy with prefix strip, sharded at ~4 GB ----
    n_copied = 0
    shard, written = {}, 0
    shards, weight_map = [], {}

    def flush():
        nonlocal shard, written
        if not shard:
            return
        from safetensors.torch import save_file
        name = f"model-{len(shards):05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shards.append(name)
        print(f"  shard {name}: {written/1e9:.2f} GB cumulative")
        shard = {}

    with safe_open(src / "model.safetensors", framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("model.language_model."):
                nk = k[len("model.language_model."):]
            elif k == "lm_head.weight":
                nk = k
            else:
                continue
            t = sf.get_tensor(k)
            shard[nk] = t
            written += t.numel() * t.element_size()
            n_copied += 1
            if written > 4e9:
                flush()
    flush()
    if len(shards) > 1:
        (out / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": written},
            "weight_map": weight_map,
        }, indent=2))

    # ---- tokenizer: real if provided, else placeholder ----
    V = new_cfg["vocab_size"]
    if args.real_tokenizer_json:
        import shutil
        shutil.copy(args.real_tokenizer_json, out / "tokenizer.json")
        print(f"tokenizer: real ({args.real_tokenizer_json})")
    else:
        tok = {
            "version": "1.0",
            "truncation": None, "padding": None,
            "added_tokens": [],
            "normalizer": None, "pre_tokenizer": None,
            "post_processor": None, "decoder": None,
            "model": {
                "type": "BPE", "dropout": None, "unk_token": None,
                "continuing_subword_prefix": None, "end_of_word_suffix": None,
                "fuse_unk": False, "byte_fallback": False,
                "vocab": {f"<|placeholder_{i}|>": i for i in range(V)},
                "merges": [],
            },
        }
        (out / "tokenizer.json").write_text(json.dumps(tok))
        print(f"tokenizer: PLACEHOLDER with {V} dummy tokens — "
              f"the converted GGUF validates weights only; regenerate with the "
              f"real tokenizer.json before any text run")

    # generation config passthrough
    gen = src / "generation_config.json"
    if gen.exists():
        (out / "generation_config.json").write_text(gen.read_text())

    print(f"copied {n_copied} tensors -> {out}")
    print(f"NEXT: python3 cpp_ggml/third_party/llama.cpp/convert_hf_to_gguf.py "
          f"--outfile <out.gguf> {out} --outtype f16")


if __name__ == "__main__":
    main()
