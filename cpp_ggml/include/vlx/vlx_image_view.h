// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// Borrowed, row-stride-aware image view used by the VLX-Seek in-memory task
// API. Layout contract mirrors aicore_image_view so the header is drop-in
// for include/aicore/ at integration time.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "vlx/vlx_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vlx_image_format {
    VLX_IMAGE_RGB8 = 1,
    VLX_IMAGE_RGBA8 = 2,
    VLX_IMAGE_GRAY8 = 3,
    VLX_IMAGE_BGR8 = 4,
    VLX_IMAGE_BGRA8 = 5
} vlx_image_format;

/** Borrowed, row-stride-aware image view. The caller owns the memory and
 *  must keep it alive for the duration of the call. */
typedef struct vlx_image_view {
    const uint8_t* data;
    int32_t width;
    int32_t height;
    size_t row_stride_bytes;
    vlx_image_format format;
} vlx_image_view;

#ifdef __cplusplus
}
#endif
