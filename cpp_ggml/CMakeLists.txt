cmake_minimum_required(VERSION 3.14)
project(vlxseekggml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Backends: forward a single GPU backend selection to ggml.
# ---------------------------------------------------------------------------
option(VLX_GGML_CUDA   "Enable CUDA backend in ggml"   OFF)
option(VLX_GGML_VULKAN "Enable Vulkan backend in ggml" OFF)
option(VLX_GGML_METAL  "Enable Metal backend in ggml"  OFF)
option(VLX_GGML_BUILD_CLI "Build the radio-cli binary" ON)

set(_vlx_gpu_backends 0)
foreach(_b VLX_GGML_CUDA VLX_GGML_VULKAN VLX_GGML_METAL)
    if(${_b})
        math(EXPR _vlx_gpu_backends "${_vlx_gpu_backends} + 1")
    endif()
endforeach()
if(_vlx_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

if(VLX_GGML_CUDA)
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
    enable_language(CUDA)
endif()
if(VLX_GGML_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
endif()
if(VLX_GGML_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

if(NOT (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/vlx_cli.cpp"))
    message(FATAL_ERROR "cpp_ggml/third_party/llama.cpp is missing — run "
        "git submodule update --init cpp_ggml/third_party/llama.cpp. It is the "
        "single ggml source (its vendored copy is >= the former v0.21 pin) and "
        "the only one this build links.")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt"
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/vlx_cli.cpp")
    # ggml-source changes to llama.cpp's vendored ggml go through
    # third_party/llama-patches/*.patch (same idempotent contract as
    # ggml-patches). Replayed at configure time: add_subdirectory below reads
    # the sources immediately, so a build-time custom command would be too late.
    find_package(Patch REQUIRED)
    file(GLOB VLX_LLAMA_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama-patches/*.patch")
    list(SORT VLX_LLAMA_PATCHES)
    execute_process(
        COMMAND ${CMAKE_COMMAND}
                -DPATCH_EXECUTABLE=${Patch_EXECUTABLE}
                -DGGML_DIR=${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp
                "-DVLX_PATCH_FILES=${VLX_LLAMA_PATCHES}"
                -DSTAMP=${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/.vlx-patches-applied
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply_patches.cmake
        RESULT_VARIABLE _vlx_llama_patch_rc)
    if(NOT _vlx_llama_patch_rc EQUAL 0)
        message(FATAL_ERROR "llama.cpp patch replay failed (${_vlx_llama_patch_rc})")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${VLX_LLAMA_PATCHES})
    # Must come BEFORE anything references the ggml target: llama.cpp brings
    # its own vendored ggml (the only one this build links).
    add_subdirectory(third_party/llama.cpp llama.cpp-sub EXCLUDE_FROM_ALL)
    add_executable(vlx-cli src/vlx_cli.cpp)

    # Pure-C smoke test of the VLX-Seek C API (AICore-contract surface).
    add_executable(vlx-seek-capi-smoke tests/vlx_capi_smoke.c)
    target_include_directories(vlx-seek-capi-smoke PRIVATE include)
    target_link_libraries(vlx-seek-capi-smoke PRIVATE vlxvision Threads::Threads)
    set_target_properties(vlx-seek-capi-smoke PROPERTIES LINKER_LANGUAGE CXX)
    target_include_directories(vlx-cli PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    )
    target_link_libraries(vlx-cli PRIVATE vlxvision llama)
endif()

# ---------------------------------------------------------------------------
# Single ggml source: llama.cpp's vendored copy (the pin that matters is the
# llama.cpp commit; its vendored ggml is >= the former v0.21 standalone pin).
# The vision graphs (vlxvision) compile against it, so radio/mmproj/simplefpn
# and the LLM share one runtime. ggml-source changes go through
# third_party/llama-patches/*.patch — never hand-edit the submodule (sync
# contract for other developers). The former third_party/ggml standalone
# submodule and ggml-patches/ were removed: the fallback was never used.
# ---------------------------------------------------------------------------

set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

find_package(Patch REQUIRED)

# ---------------------------------------------------------------------------
# Vision sub-chain sources.
# ---------------------------------------------------------------------------
add_library(vlxvision STATIC
    src/gguf_loader.cpp
    src/radio_graph.cpp
    src/mmproj_graph.cpp
    src/simplefpn_graph.cpp
    src/hfre.cpp
    src/image_prep.cpp
    src/wedetect_graph.cpp
    src/vlx_pipeline_common.cpp
    src/vlx_seek_runtime.cpp
    src/vlx_seek_capi.cpp
    src/vlx_roialign_gpu.cpp
)
# Fused CUDA RoIAlign (roi_align_cuda.cu) is built into vlxvision only for
# CUDA builds; non-CUDA builds fall back to the host pooling at the call
# sites, so the library keeps the same symbol surface everywhere.
if(VLX_GGML_CUDA)
    target_sources(vlxvision PRIVATE src/roi_align_cuda.cu)
    target_compile_definitions(vlxvision PUBLIC VLX_HAS_ROIALIGN_CUDA=1)
endif()
target_include_directories(vlxvision PUBLIC src include)
target_include_directories(vlxvision PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include)
target_link_libraries(vlxvision PUBLIC ggml llama)
# Host-side preprocessing must round exactly like torch's CPU kernels: the
# one FMA torch uses (AA resize) is spelled std::fma, every other mul+add has
# to stay unfused whatever -march the toolchain defaults to.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(src/image_prep.cpp src/hfre.cpp src/radio_graph.cpp
        src/mmproj_graph.cpp src/wedetect_graph.cpp src/vlx_cli.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()
# JPEG decode parity: PIL decodes through libjpeg (ISLOW IDCT + fancy
# upsampling); stb_image's IDCT/upsampler differ by +-1..2 LSB on real photos.
# With the system libjpeg(-turbo) the decoded RGB is byte-identical to
# Image.open(...).convert("RGB"); without it stb_image is the fallback.
find_package(JPEG)
if(JPEG_FOUND)
    target_compile_definitions(vlxvision PRIVATE VLX_HAVE_LIBJPEG=1)
    target_link_libraries(vlxvision PRIVATE JPEG::JPEG)
endif()

find_package(Threads REQUIRED)
target_link_libraries(vlxvision PUBLIC Threads::Threads)

if(VLX_GGML_BUILD_CLI)
    add_executable(radio-cli src/radio_cli.cpp)
    target_link_libraries(radio-cli PRIVATE vlxvision)
endif()

# ---------------------------------------------------------------------------
# P2+: full-chain CLI. Enabled when third_party/llama.cpp is initialized.
# The vision graph sources are kept header-only-compatible with llama.cpp's
# vendored ggml so the merge is an include-path switch, not a rewrite.
# ---------------------------------------------------------------------------
