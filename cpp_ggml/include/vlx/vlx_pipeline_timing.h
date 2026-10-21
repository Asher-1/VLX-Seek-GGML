// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Common wall-clock timing contract for the VLX-Seek pipeline. Mirrors
// aicore_pipeline_timings so the header is drop-in for include/aicore/ at
// integration time.
//
// preprocess_ms:   input validation, decode/resize/normalize and the
//                  WeDetect-Uni proposal pass when the caller did not supply
//                  boxes.
// inference_ms:    weight upload + the vision graphs (main tower, RADIO,
//                  SimpleFPN), HFRE region encoding and the LLM
//                  prefill + greedy decode.
// postprocess_ms:  answer remap, <objN> resolution and typed-result
//                  construction.
// serialization_ms: optional PNG rendering only (render_png / visualize).
// e2e_ms:          API entry until the native result is ready; excludes
//                  optional serialization when the pipeline can expose it
//                  independently.
//
// Consumers must inspect valid_fields. A stage that cannot be measured at a
// truthful boundary is left at zero and its bit is clear.
#pragma once

#include <stdint.h>

#define VLX_PIPELINE_TIMINGS_ABI_VERSION 1u

enum vlx_pipeline_timing_field {
    VLX_TIMING_PREPROCESS = 1u << 0,
    VLX_TIMING_INFERENCE = 1u << 1,
    VLX_TIMING_POSTPROCESS = 1u << 2,
    VLX_TIMING_SERIALIZATION = 1u << 3,
    VLX_TIMING_E2E = 1u << 4
};

typedef struct vlx_pipeline_timings {
    uint32_t abi_version;
    uint32_t valid_fields;
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
    double serialization_ms;
    double e2e_ms;
} vlx_pipeline_timings;
