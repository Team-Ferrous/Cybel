cmake_minimum_required(VERSION 3.10)
project(granite_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
string(TIMESTAMP ANVIL_NATIVE_BUILD_ID "%Y%m%dT%H%M%SZ" UTC)
option(ENABLE_SANITIZERS "Enable sanitizer instrumentation for native builds" OFF)
option(ENABLE_LTO "Enable Link-Time Optimization" ON)
option(ANVIL_REQUIRE_OPENMP "Require OpenMP for native CPU runtime" ON)
option(ANVIL_REQUIRE_AVX2 "Require AVX2-capable x86 CPU features" ON)
option(ANVIL_ENABLE_AMX "Enable AMX BF16 kernel dispatch when compiled into the native runtime" ON)
option(ANVIL_ENABLE_HWLOC "Enable hwloc topology+binding support" ON)
option(ANVIL_ENABLE_LIBNUMA "Enable libnuma memory policy APIs" ON)
option(ANVIL_ENABLE_NUMA_STRICT "Enable strict NUMA/CCX runtime controls" ON)
set(ANVIL_NATIVE_ISA_BASELINE "avx2" CACHE STRING "Native ISA baseline: avx2 or host")
set_property(CACHE ANVIL_NATIVE_ISA_BASELINE PROPERTY STRINGS avx2 host)
set(SANITIZERS "address,undefined" CACHE STRING "Comma-separated sanitizer list")

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
endif()

if(ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        message(WARNING "Sanitizers requested but compiler does not support GCC/Clang sanitizer flags")
    else()
        string(REPLACE "," ";" SAN_LIST "${SANITIZERS}")
        set(SAN_FLAGS "")
        foreach(SAN ${SAN_LIST})
            string(STRIP "${SAN}" SAN)
            if(SAN STREQUAL "")
                continue()
            endif()
            list(APPEND SAN_FLAGS "-fsanitize=${SAN}")
        endforeach()
        if(SAN_FLAGS)
            message(STATUS "Enabling sanitizer flags: ${SAN_FLAGS}")
            add_compile_options(-fno-omit-frame-pointer -g ${SAN_FLAGS})
            add_link_options(${SAN_FLAGS})
        endif()
    endif()
endif()

# Detect CPU capabilities
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64")
    set(ANVIL_X86_HAS_AVX2 OFF)
    set(ANVIL_X86_HAS_AMX OFF)
    string(TOLOWER "${ANVIL_NATIVE_ISA_BASELINE}" ANVIL_NATIVE_ISA_BASELINE_NORMALIZED)
    if(NOT ANVIL_NATIVE_ISA_BASELINE_NORMALIZED STREQUAL "avx2"
       AND NOT ANVIL_NATIVE_ISA_BASELINE_NORMALIZED STREQUAL "host")
        message(FATAL_ERROR "ANVIL_NATIVE_ISA_BASELINE must be one of: avx2, host")
    endif()
    if(EXISTS "/proc/cpuinfo")
        file(READ "/proc/cpuinfo" CPUINFO)
        if(CPUINFO MATCHES "avx2")
            message(STATUS "AVX2 detected")
            set(ANVIL_X86_HAS_AVX2 ON)
        else()
            message(STATUS "No advanced AVX detected")
        endif()
        if(CPUINFO MATCHES "amx_tile" OR CPUINFO MATCHES "amx_int8" OR CPUINFO MATCHES "amx_bf16" OR CPUINFO MATCHES "amx_fp16")
            message(STATUS "AMX detected")
            set(ANVIL_X86_HAS_AMX ON)
        endif()
    elseif(ANVIL_REQUIRE_AVX2)
        message(FATAL_ERROR "Cannot verify AVX2 support: /proc/cpuinfo is unavailable.")
    endif()
    if(ANVIL_REQUIRE_AVX2 AND NOT ANVIL_X86_HAS_AVX2)
        message(FATAL_ERROR "ANVIL_REQUIRE_AVX2 is ON but AVX2/AVX-512 was not detected.")
    endif()
    if(ANVIL_NATIVE_ISA_BASELINE_NORMALIZED STREQUAL "avx2")
        message(STATUS "Configuring native runtime for strict AVX2/FMA baseline")
        add_compile_options(-mavx2 -mfma)
        add_compile_options(-mtune=native -O3)
    else()
        message(STATUS "Configuring native runtime for host-tuned ISA baseline")
        add_compile_options(-march=native -O3)
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "ARM64 detected")
    add_compile_options(-march=native -O3)
endif()

set(ANVIL_NATIVE_OPTIONAL_ISA_LEAVES "")
if(ANVIL_ENABLE_AMX AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64"))
    set(ANVIL_NATIVE_OPTIONAL_ISA_LEAVES "amx")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP ${OpenMP_CXX_VERSION} detected — enabling multi-threaded kernels")
    add_compile_options(${OpenMP_CXX_FLAGS})
elseif(ANVIL_REQUIRE_OPENMP)
    message(FATAL_ERROR "ANVIL_REQUIRE_OPENMP is ON but OpenMP was not found.")
else()
    message(WARNING "OpenMP not found — kernels will run single-threaded")
endif()

find_package(PkgConfig QUIET)
set(ANVIL_HWLOC_FOUND OFF)
set(ANVIL_NUMA_FOUND OFF)
if(ANVIL_ENABLE_HWLOC AND PkgConfig_FOUND)
    pkg_check_modules(HWLOC QUIET hwloc)
    if(HWLOC_FOUND)
        set(ANVIL_HWLOC_FOUND ON)
        message(STATUS "hwloc detected via pkg-config")
    else()
        message(STATUS "hwloc not detected; topology fallback will be used")
    endif()
endif()

if(ANVIL_ENABLE_LIBNUMA)
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        set(ANVIL_NUMA_FOUND ON)
        message(STATUS "libnuma detected: ${NUMA_LIB}")
    else()
        message(STATUS "libnuma not detected; syscall-based NUMA fallback will be used")
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/../simd)
include_directories(${CMAKE_SOURCE_DIR}/../../ik_llama.cpp-main/src)
include_directories(${CMAKE_SOURCE_DIR}/vendor/llama_unicode)

set(ANVIL_NATIVE_SOURCES)
foreach(
    SRC
    fast_attention.cpp
    amx_kernels.cpp
    coconut_bridge.cpp
    simd_ops.cpp
    quantized_matmul.cpp
    thread_config.cpp
    model_graph.cpp
    native_kv_cache.cpp
    tinyblas_kernels.cpp
    interleave_weights.cpp
    tensor_decomposition_kernels.cpp
    native_tokenizer.cpp
)
    if(EXISTS "${CMAKE_SOURCE_DIR}/${SRC}")
        list(APPEND ANVIL_NATIVE_SOURCES "${SRC}")
    endif()
endforeach()

set(ANVIL_SPLIT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/split")

set(ANVIL_KERNEL_WRAPPERS
    "${ANVIL_SPLIT_ROOT}/kernels/amx_kernels_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/fast_attention_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/simd_ops_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/quantized_matmul_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/qsg_state_kernels_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/qsg_parallel_kernels_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/thread_config_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/numa_topology_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/numa_allocator_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/tinyblas_kernels_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/interleave_weights_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/kernels/tensor_decomposition_kernels_wrap.cpp"
)

if(ANVIL_ENABLE_AMX)
    set_source_files_properties(
        "${ANVIL_SPLIT_ROOT}/kernels/amx_kernels_wrap.cpp"
        PROPERTIES
        COMPILE_OPTIONS "-mamx-tile;-mamx-bf16"
    )
endif()

set(ANVIL_RUNTIME_CORE_WRAPPERS
    "${ANVIL_SPLIT_ROOT}/runtime/core/coconut_bridge_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/cpu_math_runtime_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/model_graph_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/native_kv_cache_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/native_tokenizer_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/unicode_wrap.cpp"
    "${ANVIL_SPLIT_ROOT}/runtime/core/unicode_data_wrap.cpp"
)

set(ANVIL_BACKEND_WRAPPERS
    "${ANVIL_SPLIT_ROOT}/backends/granite4_tinyh/backend_module.cpp"
    "${ANVIL_SPLIT_ROOT}/backends/qwen35/backend_module.cpp"
)

set(ANVIL_COMPAT_WRAPPERS
    "${ANVIL_SPLIT_ROOT}/compat/native_ops_compat.cpp"
)

foreach(
    SRC
    IN LISTS
    ANVIL_KERNEL_WRAPPERS
    ANVIL_RUNTIME_CORE_WRAPPERS
    ANVIL_BACKEND_WRAPPERS
    ANVIL_COMPAT_WRAPPERS
)
    if(NOT EXISTS "${SRC}")
        message(FATAL_ERROR "Missing split wrapper source: ${SRC}")
    endif()
endforeach()

if(ANVIL_NATIVE_SOURCES)
    message(STATUS "Building split native targets and compatibility anvil_native_ops...")

    add_library(anvil_kernels_obj OBJECT ${ANVIL_KERNEL_WRAPPERS})
    target_compile_definitions(
        anvil_kernels_obj
        PRIVATE
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
    )
    if(ANVIL_ENABLE_NUMA_STRICT)
        target_compile_definitions(anvil_kernels_obj PRIVATE ANVIL_ENABLE_NUMA_STRICT=1)
    endif()
    if(ANVIL_HWLOC_FOUND)
        target_compile_definitions(anvil_kernels_obj PRIVATE ANVIL_HAVE_HWLOC=1)
        target_include_directories(anvil_kernels_obj PRIVATE ${HWLOC_INCLUDE_DIRS})
    endif()
    if(ANVIL_NUMA_FOUND)
        target_compile_definitions(anvil_kernels_obj PRIVATE ANVIL_HAVE_LIBNUMA=1)
    endif()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(anvil_kernels_obj PRIVATE ANVIL_HAVE_MADV_COLLAPSE=1)
    endif()
    target_compile_options(
        anvil_kernels_obj
        PRIVATE
        $<$<CONFIG:Release>:-fopenmp-simd -ffp-contract=fast -fno-math-errno -fno-trapping-math -falign-functions=32 -falign-loops=32>
    )

    add_library(anvil_runtime_core_obj OBJECT ${ANVIL_RUNTIME_CORE_WRAPPERS})
    target_compile_definitions(
        anvil_runtime_core_obj
        PRIVATE
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
    )

    add_library(
        anvil_runtime_core
        SHARED
        $<TARGET_OBJECTS:anvil_runtime_core_obj>
        $<TARGET_OBJECTS:anvil_kernels_obj>
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(anvil_runtime_core PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(ANVIL_HWLOC_FOUND)
        target_link_libraries(anvil_runtime_core PRIVATE ${HWLOC_LIBRARIES})
    endif()
    if(ANVIL_NUMA_FOUND)
        target_link_libraries(anvil_runtime_core PRIVATE ${NUMA_LIB})
    endif()
    set_target_properties(anvil_runtime_core PROPERTIES
        OUTPUT_NAME "anvil_runtime_core"
        PREFIX "lib"
        SUFFIX ".so"
    )
    add_custom_command(TARGET anvil_runtime_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_runtime_core>
        ${CMAKE_SOURCE_DIR}/libanvil_runtime_core.so
        COMMENT "Copying runtime core shared library to core/native for backend module loading"
    )

    add_library(anvil_compat_obj OBJECT ${ANVIL_COMPAT_WRAPPERS})
    target_compile_definitions(
        anvil_compat_obj
        PRIVATE
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
        ANVIL_NATIVE_SPLIT_LAYOUT="kernels/runtime_core/backends/compat"
        ANVIL_NATIVE_PUBLIC_LOAD_TARGET="libanvil_native_ops.so"
        ANVIL_NATIVE_RUNTIME_CORE_TARGET="libanvil_runtime_core.so"
        ANVIL_NATIVE_COMPAT_ALIAS_CSV="libanvil_native_ops.so,libfast_attention.so,libcoconut_bridge.so"
        ANVIL_NATIVE_SPLIT_ABI_VERSION=1
        ANVIL_NATIVE_ISA_BASELINE="${ANVIL_NATIVE_ISA_BASELINE}"
        ANVIL_NATIVE_OPTIONAL_ISA_LEAVES="${ANVIL_NATIVE_OPTIONAL_ISA_LEAVES}"
    )

    add_library(
        anvil_backend_granite4_tinyh
        MODULE
        "${ANVIL_SPLIT_ROOT}/backends/granite4_tinyh/backend_module.cpp"
    )
    target_compile_definitions(
        anvil_backend_granite4_tinyh
        PRIVATE
        ANVIL_BACKEND_NAME="granite4_tinyh"
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
        ANVIL_NATIVE_BACKEND_ABI_VERSION=1
    )
    target_link_libraries(anvil_backend_granite4_tinyh PRIVATE anvil_runtime_core)
    set_target_properties(anvil_backend_granite4_tinyh PROPERTIES
        OUTPUT_NAME "anvil_backend_granite4_tinyh"
        PREFIX "lib"
        SUFFIX ".so"
        BUILD_RPATH "$ORIGIN;$ORIGIN/build"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/build"
    )
    add_custom_command(TARGET anvil_backend_granite4_tinyh POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_backend_granite4_tinyh>
        ${CMAKE_SOURCE_DIR}/libanvil_backend_granite4_tinyh.so
        COMMENT "Copying granite backend module to core/native for runtime discovery"
    )

    add_library(
        anvil_backend_qwen35
        MODULE
        "${ANVIL_SPLIT_ROOT}/backends/qwen35/backend_module.cpp"
    )
    target_compile_definitions(
        anvil_backend_qwen35
        PRIVATE
        ANVIL_BACKEND_NAME="qwen35"
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
        ANVIL_NATIVE_BACKEND_ABI_VERSION=1
    )
    target_link_libraries(anvil_backend_qwen35 PRIVATE anvil_runtime_core)
    set_target_properties(anvil_backend_qwen35 PROPERTIES
        OUTPUT_NAME "anvil_backend_qwen35"
        PREFIX "lib"
        SUFFIX ".so"
        BUILD_RPATH "$ORIGIN;$ORIGIN/build"
        INSTALL_RPATH "$ORIGIN;$ORIGIN/build"
    )
    add_custom_command(TARGET anvil_backend_qwen35 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_backend_qwen35>
        ${CMAKE_SOURCE_DIR}/libanvil_backend_qwen35.so
        COMMENT "Copying qwen35 backend module to core/native for runtime discovery"
    )

    add_library(
        anvil_native_ops
        SHARED
        $<TARGET_OBJECTS:anvil_runtime_core_obj>
        $<TARGET_OBJECTS:anvil_kernels_obj>
        $<TARGET_OBJECTS:anvil_compat_obj>
    )
    target_compile_definitions(
        anvil_native_ops
        PRIVATE
        ANVIL_NATIVE_BUILD_ID="${ANVIL_NATIVE_BUILD_ID}"
        ANVIL_NATIVE_PUBLIC_LOAD_TARGET=1
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(anvil_native_ops PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(ANVIL_HWLOC_FOUND)
        target_link_libraries(anvil_native_ops PRIVATE ${HWLOC_LIBRARIES})
    endif()
    if(ANVIL_NUMA_FOUND)
        target_link_libraries(anvil_native_ops PRIVATE ${NUMA_LIB})
    endif()
    set_target_properties(anvil_native_ops PROPERTIES
        OUTPUT_NAME "anvil_native_ops"
        PREFIX "lib"
        SUFFIX ".so"
    )

    add_custom_command(TARGET anvil_native_ops POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_native_ops>
        ${CMAKE_SOURCE_DIR}/libanvil_native_ops.so
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_native_ops>
        ${CMAKE_SOURCE_DIR}/libfast_attention.so
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:anvil_native_ops>
        ${CMAKE_SOURCE_DIR}/libcoconut_bridge.so
        COMMENT "Copying compatibility aliases for public native load target"
    )
else()
    message(STATUS "No native sources found, skipping native libraries")
endif()

message(STATUS "Native build system configured. Ready for ops.")
