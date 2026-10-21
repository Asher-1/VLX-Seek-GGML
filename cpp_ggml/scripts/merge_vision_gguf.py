#!/usr/bin/env python3
"""Merge the three vision GGUFs (mmproj / fpn / radio) into one file per dtype.

The C++ graphs address tensors by name, and the converters already use
disjoint namespaces (`mmproj.*`, `radio.*`, fpn incl. `aux_proj.*`), so a
byte-level concat with a shared header is lossless: identical tensor data,
identical KV values, one file per precision tier. The LLM stays a separate
file on purpose: llama.cpp's `qwen35` arch loader rejects foreign tensors,
and the vision weights must free before the LLM loads (memory lifecycle).

KV policy: copy everything; rewrite the frame-level keys for the merged file;
duplicates must carry identical values (asserted — today only
vlx.hfre.roi_output_size / roi_aligned, shared by the fpn and radio docs).

Output: vlx-vision-10b-<dtype>.gguf (mmproj 333 + fpn 46 + radio 329 = 708
tensors).

Usage:
  python3 cpp_ggml/scripts/merge_vision_gguf.py --dtype f16
  python3 cpp_ggml/scripts/merge_vision_gguf.py --dtype q8_0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))

import gguf  # noqa: E402  (vendored llama.cpp gguf-py)

FRAME_KEYS = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
              "general.architecture", "general.name"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="f16", choices=("f16", "q8_0"))
    ap.add_argument("--gguf-dir", default=str(ROOT / "models" / "gguf"))
    ap.add_argument("--output", default=None,
                    help="default: <gguf-dir>/vlx-vision-10b-<dtype>.gguf")
    args = ap.parse_args()

    parts = [f"vlx-mmproj-10b-{args.dtype}.gguf", f"vlx-fpn-10b-{args.dtype}.gguf",
             f"vlx-radio-10b-{args.dtype}.gguf"]
    readers = []
    seen_kv: dict[str, object] = {}
    kv_field: dict[str, object] = {}   # first field object per key, for the value type
    n_tensors = 0
    for part in parts:
        path = Path(args.gguf_dir) / part
        r = gguf.GGUFReader(str(path))
        for field in r.fields.values():
            if field.name in FRAME_KEYS:
                continue
            val = field.contents()
            if field.name in seen_kv:
                if seen_kv[field.name] != val:
                    sys.exit(f"KV conflict for {field.name}: "
                             f"{seen_kv[field.name]!r} vs {val!r}")
            else:
                seen_kv[field.name] = val
                kv_field[field.name] = field
        n_tensors += len(r.tensors)
        readers.append((part, r))
        print(f"{part}: {len(r.tensors)} tensors")

    out = Path(args.output or Path(args.gguf_dir) / f"vlx-vision-10b-{args.dtype}.gguf")
    writer = gguf.GGUFWriter(str(out), "vlxvision")
    writer.add_architecture()
    writer.add_string("general.name", "vlx-vision-10b")
    writer.add_string("general.description",
                      "merged vision stack: mmproj (main tower) + fpn (SimpleFPN) "
                      "+ radio (C-RADIOv4 aux); tensors are byte-identical to the "
                      "per-part files")
    for name, val in seen_kv.items():
        field = kv_field[name]
        writer.add_key_value(name, val, field.types[0])
    for _, r in readers:
        for t in r.tensors:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype,
                                   t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for _, r in readers:
        for t in r.tensors:
            writer.write_tensor_data(t.data)
    writer.close()
    print(f"merged {n_tensors} tensors -> {out} ({out.stat().st_size / 2**30:.2f} GiB)")

    # verify: reopen and compare per-tensor bytes against the sources
    check = gguf.GGUFReader(str(out))
    by_name = {t.name: t for t in check.tensors}
    assert len(by_name) == n_tensors, "tensor count mismatch"
    for part, r in readers:
        for t in r.tensors:
            src, dst = t.data, by_name[t.name].data
            assert src.tobytes() == dst.tobytes(), f"byte mismatch: {t.name}"
    print(f"verify: {n_tensors}/{n_tensors} tensors byte-identical "
          f"(dims/dtype preserved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
