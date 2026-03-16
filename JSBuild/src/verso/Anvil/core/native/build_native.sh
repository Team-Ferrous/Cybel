#!/bin/bash
# core/native/build_native.sh

set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$DIR/build"
REPO_ROOT="$( cd "$DIR/../.." && pwd )"
VECTOR_STORE_SRC="$DIR/saguaro_vector_store.cpp"
VECTOR_STORE_OUT="$REPO_ROOT/build/libanvil_saguaro_vector_store.so"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi

ANVIL_NATIVE_BUILD_TYPE="${ANVIL_NATIVE_BUILD_TYPE:-Release}"
ANVIL_NATIVE_ISA_BASELINE="${ANVIL_NATIVE_ISA_BASELINE:-avx2}"
ANVIL_REQUIRE_AVX2="${ANVIL_REQUIRE_AVX2:-ON}"
ANVIL_REQUIRE_OPENMP="${ANVIL_REQUIRE_OPENMP:-ON}"
ANVIL_ENABLE_AMX="${ANVIL_ENABLE_AMX:-ON}"
ANVIL_ENABLE_HWLOC="${ANVIL_ENABLE_HWLOC:-ON}"
ANVIL_ENABLE_LIBNUMA="${ANVIL_ENABLE_LIBNUMA:-ON}"
ANVIL_ENABLE_NUMA_STRICT="${ANVIL_ENABLE_NUMA_STRICT:-ON}"
ANVIL_NATIVE_EXTRA_CXX_FLAGS="${ANVIL_NATIVE_EXTRA_CXX_FLAGS:-}"

HOST_SIMD_SUMMARY="scalar"
if grep -q "avx512" /proc/cpuinfo; then
    HOST_SIMD_SUMMARY="avx512"
elif grep -q "avx2" /proc/cpuinfo; then
    HOST_SIMD_SUMMARY="avx2"
fi

echo "Host SIMD summary: $HOST_SIMD_SUMMARY"
echo "Native build type: $ANVIL_NATIVE_BUILD_TYPE"
echo "Native ISA baseline: $ANVIL_NATIVE_ISA_BASELINE"
echo "Require AVX2: $ANVIL_REQUIRE_AVX2"
echo "Require OpenMP: $ANVIL_REQUIRE_OPENMP"
echo "Enable AMX: $ANVIL_ENABLE_AMX"
echo "Enable hwloc: $ANVIL_ENABLE_HWLOC"
echo "Enable libnuma: $ANVIL_ENABLE_LIBNUMA"
echo "Enable strict NUMA: $ANVIL_ENABLE_NUMA_STRICT"
if [ -n "$ANVIL_NATIVE_EXTRA_CXX_FLAGS" ]; then
    echo "Extra CXX flags: $ANVIL_NATIVE_EXTRA_CXX_FLAGS"
fi

echo "Configuring native modules..."
cmake_args=(
    "-DCMAKE_BUILD_TYPE=$ANVIL_NATIVE_BUILD_TYPE"
    "-DANVIL_NATIVE_ISA_BASELINE=$ANVIL_NATIVE_ISA_BASELINE"
    "-DANVIL_REQUIRE_AVX2=$ANVIL_REQUIRE_AVX2"
    "-DANVIL_REQUIRE_OPENMP=$ANVIL_REQUIRE_OPENMP"
    "-DANVIL_ENABLE_AMX=$ANVIL_ENABLE_AMX"
    "-DANVIL_ENABLE_HWLOC=$ANVIL_ENABLE_HWLOC"
    "-DANVIL_ENABLE_LIBNUMA=$ANVIL_ENABLE_LIBNUMA"
    "-DANVIL_ENABLE_NUMA_STRICT=$ANVIL_ENABLE_NUMA_STRICT"
)
if [ -n "$ANVIL_NATIVE_EXTRA_CXX_FLAGS" ]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$ANVIL_NATIVE_EXTRA_CXX_FLAGS")
fi
cmake -B "$BUILD_DIR" -S "$DIR" "${cmake_args[@]}"

echo "Building native modules..."
cmake --build "$BUILD_DIR" -j$( nproc )

echo "Building SIMD modules..."
bash "$DIR/../../core/simd/build_simd.sh"

echo "Building Saguaro vector store library..."
mkdir -p "$( dirname "$VECTOR_STORE_OUT" )"
g++ \
    -std=c++17 \
    -O3 \
    -fPIC \
    -shared \
    -mavx2 \
    -mfma \
    -fopenmp \
    "$VECTOR_STORE_SRC" \
    -o "$VECTOR_STORE_OUT"

echo "Native build complete."
