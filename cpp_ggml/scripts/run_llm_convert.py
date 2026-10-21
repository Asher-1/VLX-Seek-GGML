#!/usr/bin/env python3
"""Offline conversion wrapper for the VLX-Seek LLM.

The placeholder tokenizer.json (248182 dummy tokens, written by
prepare_llm_hf.py when the real tokenizer is unreachable) makes
ModelBase.get_vocab_base_pre() raise "BPE pre-tokenizer not recognized" —
its probe hash matches nothing in the known table.

This wrapper monkey-patches that ONE function in-process (llama.cpp sources
stay untouched; no submodule edits, no patch files for a throwaway concern)
and then runs convert_hf_to_gguf's main(). With the REAL tokenizer.json the
patch is unnecessary and the official converter works as-is.

Usage:
  python3 run_llm_convert.py <hf_dir> <out.gguf> [f16|q8_0|bf16...]
"""
from __future__ import annotations

import sys
from pathlib import Path

LLAMA_DIR = Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp"
sys.path.insert(0, str(LLAMA_DIR))
sys.path.insert(1, str(LLAMA_DIR / "gguf-py"))

import conversion.base as conv_base  # noqa: E402
import conversion.qwen as conv_qwen  # noqa: E402


_orig_get_vocab_base_pre = conv_qwen.TextModel.get_vocab_base_pre


def _patched_get_vocab_base_pre(self, tokenizer):
    try:
        return _orig_get_vocab_base_pre(self, tokenizer)
    except NotImplementedError:
        if self.dir_model and "<|placeholder_0|>" in str(
            (self.dir_model / "tokenizer.json").read_text()[:200000]
        ):
            print("run_llm_convert: placeholder vocab detected -> "
                  "overriding BPE pre-tokenizer id (weights-only conversion)")
            return "qwen35-placeholder-offline"
        raise


conv_qwen.TextModel.get_vocab_base_pre = _patched_get_vocab_base_pre

import convert_hf_to_gguf  # noqa: E402  (imports cleanly after the patch)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    convert_hf_to_gguf.main()
