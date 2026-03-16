#!/bin/bash
# core/simd/build_simd.sh

# Exit on error
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="$DIR/build"

ANVIL_SIMD_FORCE_FLAGS="${ANVIL_SIMD_FORCE_FLAGS:-}"
ANVIL_SIMD_EXTRA_FLAGS="${ANVIL_SIMD_EXTRA_FLAGS:-}"

SIMD_FLAGS="$ANVIL_SIMD_FORCE_FLAGS"
if [ -n "$SIMD_FLAGS" ]; then
    echo "Using forced SIMD flags: $SIMD_FLAGS"
elif grep -q "avx512f" /proc/cpuinfo; then
    SIMD_FLAGS="-mavx512f -mavx512dq -mfma"
    echo "Detected AVX-512 support"
elif grep -q "avx2" /proc/cpuinfo; then
    SIMD_FLAGS="-mavx2 -mfma"
    echo "Detected AVX2 support"
elif grep -q "neon" /proc/cpuinfo || grep -q "asimd" /proc/cpuinfo; then
    # ARM NEON is usually default on aarch64, but explicit doesn't hurt
    echo "Detected ARM NEON support"
else
    echo "No advanced SIMD detected, using defaults"
fi
if [ -n "$ANVIL_SIMD_EXTRA_FLAGS" ]; then
    echo "Appending SIMD extra flags: $ANVIL_SIMD_EXTRA_FLAGS"
fi

mkdir -p "$BUILD_DIR"

# Compile
echo "Compiling SIMD library..."
g++ -O3 -shared -fPIC $SIMD_FLAGS $ANVIL_SIMD_EXTRA_FLAGS -fopenmp -I"$DIR" -o "$DIR/libhnn_simd.so" "$DIR/simd_bindings.cc"

echo "Build complete: $DIR/libhnn_simd.so"
