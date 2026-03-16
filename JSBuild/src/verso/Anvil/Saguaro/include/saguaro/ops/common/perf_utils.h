// saguaro/native/ops/common/perf_utils.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file perf_utils.h
 * @brief Performance utilities for cache optimization.
 *
 * Provides platform-specific prefetch hints, alignment utilities, and
 * memory access optimization helpers for x86-64 (AVX2/AVX-512) and
 * ARM64 (NEON) architectures.
 *
 * Phase 96: GGML-Inspired Cache-Aware Memory Layout
 */

#ifndef OPS_COMMON_PERF_UTILS_H_
#define OPS_COMMON_PERF_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Platform detection for prefetch intrinsics
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>  // SSE for _mm_prefetch
#define SAGUARO_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
#define SAGUARO_ARCH_ARM64 1
#endif

namespace hsmn {
namespace ops {

// =============================================================================
// Cache Constants
// =============================================================================

/**
 * @brief Cache line size in bytes.
 *
 * 64 bytes is standard for modern x86-64 and ARM64 processors.
 * Used for alignment and prefetch stride calculations.
 */
constexpr std::size_t kCacheLineSize = 64;

/**
 * @brief L1 cache optimal block size for tiling.
 *
 * Typical L1D cache is 32-48KB. 64x64 float32 matrices fit well.
 */
constexpr int kL1BlockSize = 64;

/**
 * @brief L2 cache optimal block size for tiling.
 *
 * Typical L2 cache is 256KB-1MB. 256x256 float32 matrices fit.
 */
constexpr int kL2BlockSize = 256;

/**
 * @brief Default prefetch distance in cache lines.
 *
 * Prefetch 2-4 cache lines ahead for optimal latency hiding.
 */
constexpr int kPrefetchDistance = 3;

// =============================================================================
// Prefetch Utilities
// =============================================================================

/**
 * @brief Prefetch data into L1 cache for read (temporal, will be reused).
 *
 * Use for data that will be accessed multiple times.
 * Equivalent to _mm_prefetch(..., _MM_HINT_T0) on x86.
 *
 * @param addr Address to prefetch.
 */
inline void PrefetchT0(const void* addr) {
#if defined(SAGUARO_ARCH_X86_64)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3);  // Read, high locality
#else
    (void)addr;
#endif
}

/**
 * @brief Prefetch data into L2 cache (will be reused, but not immediately).
 *
 * Use for data that will be accessed soon but not in the immediate loop.
 * Equivalent to _mm_prefetch(..., _MM_HINT_T1) on x86.
 *
 * @param addr Address to prefetch.
 */
inline void PrefetchT1(const void* addr) {
#if defined(SAGUARO_ARCH_X86_64)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 2);  // Read, medium locality
#else
    (void)addr;
#endif
}

/**
 * @brief Prefetch data into L3/LLC cache (low locality).
 *
 * Use for data that will be accessed once and not immediately.
 * Equivalent to _mm_prefetch(..., _MM_HINT_T2) on x86.
 *
 * @param addr Address to prefetch.
 */
inline void PrefetchT2(const void* addr) {
#if defined(SAGUARO_ARCH_X86_64)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T2);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 1);  // Read, low locality
#else
    (void)addr;
#endif
}

/**
 * @brief Non-temporal prefetch (streaming, won't pollute cache).
 *
 * Use for data that will be read once and not reused.
 * Prevents cache pollution for streaming access patterns.
 * Equivalent to _mm_prefetch(..., _MM_HINT_NTA) on x86.
 *
 * @param addr Address to prefetch.
 */
inline void PrefetchNTA(const void* addr) {
#if defined(SAGUARO_ARCH_X86_64)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_NTA);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 0);  // Read, no locality
#else
    (void)addr;
#endif
}

/**
 * @brief Prefetch for write (exclusive ownership).
 *
 * Use before writing to memory to get exclusive cache line ownership.
 *
 * @param addr Address to prefetch for writing.
 */
inline void PrefetchW(void* addr) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3);  // Write, high locality
#else
    (void)addr;
#endif
}

/**
 * @brief Prefetch multiple cache lines ahead.
 *
 * Useful for prefetching a range of data in streaming access patterns.
 *
 * @param addr Base address.
 * @param cache_lines Number of cache lines to prefetch.
 */
inline void PrefetchAhead(const void* addr, int cache_lines) {
    const char* ptr = static_cast<const char*>(addr);
    for (int i = 0; i < cache_lines; ++i) {
        PrefetchT0(ptr + i * kCacheLineSize);
    }
}

/**
 * @brief Prefetch ahead for streaming access (non-temporal).
 *
 * @param addr Base address.
 * @param cache_lines Number of cache lines to prefetch.
 */
inline void PrefetchAheadNTA(const void* addr, int cache_lines) {
    const char* ptr = static_cast<const char*>(addr);
    for (int i = 0; i < cache_lines; ++i) {
        PrefetchNTA(ptr + i * kCacheLineSize);
    }
}

// =============================================================================
// Legacy Compatibility
// =============================================================================

/**
 * @brief Prefetch data into L1 cache (legacy name).
 *
 * @deprecated Use PrefetchT0 for consistency.
 */
inline void PrefetchL1(const void* addr) {
    PrefetchT0(addr);
}

// =============================================================================
// Alignment Utilities
// =============================================================================

/**
 * @brief Check if pointer is aligned to specified boundary.
 *
 * @param ptr Pointer to check.
 * @param alignment Required alignment (must be power of 2).
 * @return True if aligned.
 */
inline bool IsAligned(const void* ptr, std::size_t alignment) {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * @brief Check if pointer is cache-line aligned (64 bytes).
 *
 * @param ptr Pointer to check.
 * @return True if cache-line aligned.
 */
inline bool IsCacheLineAligned(const void* ptr) {
    return IsAligned(ptr, kCacheLineSize);
}

/**
 * @brief Check if pointer is SIMD-aligned (32 bytes for AVX, 64 for AVX-512).
 *
 * @param ptr Pointer to check.
 * @return True if SIMD-aligned.
 */
inline bool IsSimdAligned(const void* ptr) {
#if defined(__AVX512F__)
    return IsAligned(ptr, 64);
#elif defined(__AVX2__) || defined(__AVX__)
    return IsAligned(ptr, 32);
#else
    return IsAligned(ptr, 16);  // SSE/NEON
#endif
}

/**
 * @brief Round up value to next multiple of alignment.
 *
 * @param value Value to round up.
 * @param alignment Alignment (must be power of 2).
 * @return Rounded up value.
 */
inline std::size_t RoundUp(std::size_t value, std::size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const std::size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

/**
 * @brief Round up to cache line boundary.
 *
 * @param value Value to round up.
 * @return Value rounded up to next cache line.
 */
inline std::size_t RoundUpToCacheLine(std::size_t value) {
    return RoundUp(value, kCacheLineSize);
}

/**
 * @brief Calculate aligned byte size with overflow protection.
 *
 * @param elements Number of elements.
 * @param element_bytes Size of each element in bytes.
 * @param ok Optional output for success/failure.
 * @return Total byte size, or 0 on overflow.
 */
inline std::size_t SafeByteSize(int64_t elements, std::size_t element_bytes,
                                bool* ok = nullptr) {
    if (ok) {
        *ok = true;
    }
    if (elements <= 0) {
        return 0;
    }
    const auto max_value = std::numeric_limits<std::size_t>::max();
    if (static_cast<unsigned long long>(elements) >
        max_value / element_bytes) {
        if (ok) {
            *ok = false;
        }
        return 0;
    }
    return static_cast<std::size_t>(elements) * element_bytes;
}

// =============================================================================
// Memory Copy Utilities
// =============================================================================

/**
 * @brief Copy a span of elements with optimized memcpy.
 *
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param src Source pointer.
 * @param count Number of elements to copy.
 */
template <typename T>
inline void CopySpan(T* dst, const T* src, std::size_t count) {
    if (count == 0 || dst == src) {
        return;
    }
    std::memcpy(dst, src, count * sizeof(T));
}

/**
 * @brief Zero-fill a span of elements.
 *
 * @tparam T Element type.
 * @param dst Destination pointer.
 * @param count Number of elements to zero.
 */
template <typename T>
inline void ZeroSpan(T* dst, std::size_t count) {
    if (count == 0) {
        return;
    }
    std::memset(dst, 0, count * sizeof(T));
}

// =============================================================================
// Prefetch Strategy Helpers
// =============================================================================

/**
 * @brief Prefetch strategy for matrix operations.
 *
 * Prefetches data for the next iteration of a matrix loop.
 * Use at the start of the inner loop to hide memory latency.
 *
 * @param current_row Current row pointer.
 * @param next_row Next row pointer to prefetch.
 * @param row_bytes Bytes per row.
 */
inline void PrefetchMatrixRow(const void* current_row, const void* next_row,
                               std::size_t row_bytes) {
    (void)current_row;  // Current row is already cached
    
    // Prefetch next row into L1
    const char* ptr = static_cast<const char*>(next_row);
    std::size_t cache_lines = (row_bytes + kCacheLineSize - 1) / kCacheLineSize;
    
    // Limit prefetch to reasonable amount (4 cache lines)
    cache_lines = cache_lines > 4 ? 4 : cache_lines;
    
    for (std::size_t i = 0; i < cache_lines; ++i) {
        PrefetchT0(ptr + i * kCacheLineSize);
    }
}

/**
 * @brief Prefetch strategy for streaming reads.
 *
 * Use for sequential access patterns where data won't be reused.
 *
 * @param addr Current address.
 * @param stride Bytes to next element.
 * @param ahead_count How many elements ahead to prefetch.
 */
inline void PrefetchStreaming(const void* addr, std::size_t stride,
                               int ahead_count) {
    const char* ptr = static_cast<const char*>(addr);
    for (int i = 1; i <= ahead_count; ++i) {
        PrefetchNTA(ptr + i * stride);
    }
}

}  // namespace ops
}  // namespace hsmn

#endif  // OPS_COMMON_PERF_UTILS_H_
