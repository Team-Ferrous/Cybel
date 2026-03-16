// saguaro.native/ops/common/tensor_stream_pool.h
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file tensor_stream_pool.h
 * @brief Zero-Copy Inter-Kernel Tensor Streaming Pool.
 *
 * TensorStreamPool provides thread-local buffer management for zero-copy
 * data streaming between specialized C++ kernels. Instead of allocating
 * new tensors for each block's output and copying to the next block's input,
 * kernels use this pool for direct pointer handoff.
 *
 * Key Features:
 * - Thread-local storage (no lock contention)
 * - 64-byte alignment for AVX-512 compatibility
 * - Producer-consumer handoff semantics
 * - Buffer reuse with size bucketing
 * - Debug validation mode (use-after-handoff detection)
 * - Telemetry for performance monitoring
 *
 * Usage Pattern:
 *   1. Producer kernel acquires buffer: ptr = g_tensor_stream.acquire(size)
 *   2. Producer writes output to buffer
 *   3. Producer marks buffer ready: g_tensor_stream.handoff(ptr, "ConsumerName")
 *   4. Consumer kernel uses buffer directly (zero copy!)
 *   5. Consumer releases buffer: g_tensor_stream.release(ptr)
 *
 * Expected Impact:
 * - Up to 100% reduction in inter-kernel memory copy traffic
 * - 80%+ reduction in forward pass allocations
 * - 15%+ wall-clock improvement for sequences ≥32K
 *
 * @note Thread-safe. Each thread gets its own pool instance.
 * @note Part of TensorStreamPool C++ Enhancement Roadmap Phase 0.
 */

#ifndef SAGUARO_NATIVE_OPS_COMMON_TENSOR_STREAM_POOL_H_
#define SAGUARO_NATIVE_OPS_COMMON_TENSOR_STREAM_POOL_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cassert>
#include <algorithm>

// Platform-specific aligned allocation
#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

// SIMD intrinsics for prefetch operations
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace saguaro {
namespace ops {

// =============================================================================
// ALIGNMENT CONSTANTS
// =============================================================================

// AVX-512 requires 64-byte alignment, AVX2 requires 32-byte, NEON requires 16-byte
// We use 64-byte for maximum compatibility with all SIMD instruction sets
#if defined(__AVX512F__)
constexpr size_t kTensorStreamAlignment = 64;  // 512-bit vectors
#elif defined(__AVX2__)
constexpr size_t kTensorStreamAlignment = 32;  // 256-bit vectors
#elif defined(__ARM_NEON)
constexpr size_t kTensorStreamAlignment = 16;  // 128-bit vectors
#else
constexpr size_t kTensorStreamAlignment = 64;  // Default to maximum for portability
#endif

// Minimum buffer size to avoid frequent small allocations (64KB)
constexpr size_t kMinBufferSize = 65536;

// Default maximum pool size per thread (2GB fallback)
constexpr size_t kDefaultMaxPoolSizeBytes = 2ULL * 1024 * 1024 * 1024;

// Warning threshold (80% by default)
constexpr float kDefaultWarningThreshold = 0.8f;

/**
 * @brief Get configurable maximum pool size from environment variable.
 * 
 * Reads SAGUARO_TENSOR_STREAM_MAX_GB environment variable for runtime
 * configuration from Python config.TENSOR_STREAM_MAX_SIZE_GB.
 * 
 * @return Maximum pool size in bytes.
 */
inline size_t GetMaxPoolSizeBytes() {
    static size_t cached_value = 0;
    if (cached_value > 0) return cached_value;
    
    const char* env_val = std::getenv("SAGUARO_TENSOR_STREAM_MAX_GB");
    if (env_val != nullptr) {
        double gb = std::atof(env_val);
        if (gb > 0.0 && gb <= 64.0) {  // Sanity check: 0-64 GB
            cached_value = static_cast<size_t>(gb * 1024 * 1024 * 1024);
            return cached_value;
        }
    }
    cached_value = kDefaultMaxPoolSizeBytes;
    return cached_value;
}

/**
 * @brief Get configurable warning threshold from environment variable.
 * 
 * Reads SAGUARO_TENSOR_STREAM_WARN_THRESHOLD environment variable.
 * 
 * @return Warning threshold (0.0 to 1.0).
 */
inline float GetWarningThreshold() {
    const char* env_val = std::getenv("SAGUARO_TENSOR_STREAM_WARN_THRESHOLD");
    if (env_val != nullptr) {
        float threshold = static_cast<float>(std::atof(env_val));
        if (threshold > 0.0f && threshold < 1.0f) {
            return threshold;
        }
    }
    return kDefaultWarningThreshold;
}

// =============================================================================
// SIMD LEVEL DETECTION
// =============================================================================

/**
 * @brief Detected SIMD level for runtime dispatch.
 */
enum class SIMDLevel : int {
    SCALAR = 0,
    NEON = 1,
    AVX2 = 2,
    AVX512 = 3
};

/**
 * @brief Get the compile-time detected SIMD level.
 * @return SIMDLevel enum indicating available SIMD support.
 */
inline SIMDLevel GetSIMDLevel() {
#if defined(__AVX512F__)
    return SIMDLevel::AVX512;
#elif defined(__AVX2__)
    return SIMDLevel::AVX2;
#elif defined(__ARM_NEON)
    return SIMDLevel::NEON;
#else
    return SIMDLevel::SCALAR;
#endif
}

/**
 * @brief Convert SIMD level to human-readable string.
 */
inline const char* SIMDLevelToString(SIMDLevel level) {
    switch (level) {
        case SIMDLevel::AVX512: return "AVX-512";
        case SIMDLevel::AVX2: return "AVX2";
        case SIMDLevel::NEON: return "NEON";
        default: return "Scalar";
    }
}

/**
 * @brief Align size up to SIMD boundary.
 * @param bytes Size in bytes to align.
 * @return Aligned size (multiple of kTensorStreamAlignment).
 */
inline size_t AlignToSIMD(size_t bytes) {
    return (bytes + kTensorStreamAlignment - 1) & ~(kTensorStreamAlignment - 1);
}

// =============================================================================
// ALIGNED MEMORY ALLOCATION
// =============================================================================

/**
 * @brief Allocate memory aligned for optimal SIMD performance.
 *
 * @param bytes Number of bytes to allocate.
 * @param alignment Alignment requirement (default: kTensorStreamAlignment).
 * @return Aligned pointer, or nullptr on failure.
 */
inline void* AlignedAllocStream(size_t bytes, size_t alignment = kTensorStreamAlignment) {
    if (bytes == 0) return nullptr;
    
#ifdef _WIN32
    return _aligned_malloc(bytes, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Free memory allocated with AlignedAllocStream.
 * @param ptr Pointer from AlignedAllocStream.
 */
inline void AlignedFreeStream(void* ptr) {
    if (ptr == nullptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// =============================================================================
// PREFETCH UTILITIES
// =============================================================================

/**
 * @brief Prefetch memory for read access (L1 cache hint).
 * @param ptr Pointer to prefetch.
 */
inline void PrefetchRead(const void* ptr) {
    if (ptr == nullptr) return;
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE__)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);  // 0=read, 3=high temporal locality
#endif
}

/**
 * @brief Prefetch memory for write access (L1 cache hint).
 * @param ptr Pointer to prefetch.
 */
inline void PrefetchWrite(void* ptr) {
    if (ptr == nullptr) return;
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE__)
    _mm_prefetch(static_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 1, 3);  // 1=write, 3=high temporal locality
#endif
}

// =============================================================================
// TENSOR STREAM POOL
// =============================================================================

/**
 * @brief Thread-local zero-copy buffer pool for inter-kernel streaming.
 *
 * Provides producer-consumer semantics for buffer handoff between kernels.
 * Buffers are allocated on first acquire() for a given size bucket and
 * persist for thread lifetime (no malloc/free in hot path after warmup).
 *
 * Memory lifecycle:
 *   - Buffers are allocated on first acquire() for a given size bucket
 *   - Buffers persist for thread lifetime
 *   - Pool is cleared on thread exit via TLS destructor
 *   - Manual clear() available for memory pressure situations
 *
 * Thread Safety:
 *   - Each thread has its own pool instance (thread_local)
 *   - No locking required within a single thread
 *   - Cross-thread buffer sharing is NOT supported
 */
#if defined(__GNUC__) || defined(__clang__)
class __attribute__((visibility("default"))) TensorStreamPool {
#else
class TensorStreamPool {
#endif
public:
    // -------------------------------------------------------------------------
    // Statistics Structure
    // -------------------------------------------------------------------------
    
    /**
     * @brief Streaming statistics for telemetry and debugging.
     */
    struct Stats {
        size_t total_allocated_bytes = 0;   ///< Total bytes allocated by pool
        size_t num_buffers = 0;             ///< Number of buffer entries
        size_t acquire_count = 0;           ///< Total acquire() calls
        size_t reuse_count = 0;             ///< acquire() that reused existing buffer
        size_t zero_copy_handoffs = 0;      ///< Successful zero-copy handoffs
        size_t fallback_copies = 0;         ///< Size mismatch requiring copy
        size_t release_count = 0;           ///< Total release() calls
        size_t peak_usage_bytes = 0;        ///< Peak memory in use simultaneously
        size_t current_usage_bytes = 0;     ///< Current memory in use
        double avg_buffer_lifetime_ms = 0;  ///< Average buffer hold time
    };

    // -------------------------------------------------------------------------
    // Public API
    // -------------------------------------------------------------------------
    
    /**
     * @brief Acquire a buffer of given size.
     *
     * If a buffer of compatible size was previously released, returns that
     * buffer (zero allocation). Otherwise allocates new aligned memory.
     *
     * @param size_bytes Number of bytes needed.
     * @param producer_hint Optional name of producing kernel (for debugging).
     * @return Pointer to aligned buffer, or nullptr on allocation failure.
     */
    float* Acquire(size_t size_bytes, const char* producer_hint = nullptr);
    
    /**
     * @brief Mark buffer as ready for handoff to next kernel.
     *
     * This signals that the producer has finished writing and the buffer
     * is ready for consumption. Adds prefetch hints for the consumer.
     *
     * @param ptr Pointer previously returned by Acquire().
     * @param consumer_hint Optional name of expected consumer kernel.
     */
    void Handoff(float* ptr, const char* consumer_hint = nullptr);
    
    /**
     * @brief Release buffer back to pool for reuse.
     *
     * The buffer remains allocated but is marked as available for future
     * Acquire() calls of compatible size.
     *
     * @param ptr Pointer from Acquire() or received via handoff.
     */
    void Release(float* ptr);
    
    /**
     * @brief Get current streaming statistics.
     * @return Stats structure with all telemetry data.
     */
    Stats GetStats() const { return stats_; }
    
    /**
     * @brief Clear all buffers, freeing all memory.
     *
     * WARNING: Invalidates all previously acquired pointers!
     * Use this for memory pressure situations or testing.
     */
    void Clear();
    
    /**
     * @brief Reset statistics to zero (for benchmarking).
     */
    void ResetStats();
    
    // -------------------------------------------------------------------------
    // Debug Validation (enabled with SAGUARO_DEBUG_STREAMING)
    // -------------------------------------------------------------------------
    
#ifdef SAGUARO_DEBUG_STREAMING
    /**
     * @brief Validate that a pointer is valid and properly acquired.
     *
     * Throws std::runtime_error if validation fails.
     *
     * @param ptr Pointer to validate.
     */
    void ValidateAccess(float* ptr);
    
    /**
     * @brief Check if buffer was handed off (for detecting use-after-handoff).
     *
     * @param ptr Pointer to check.
     * @return true if buffer was handed off and not yet acquired by consumer.
     */
    bool WasHandedOff(float* ptr) const;
#endif

    // -------------------------------------------------------------------------
    // Constructor / Destructor
    // -------------------------------------------------------------------------
    
    TensorStreamPool() = default;
    ~TensorStreamPool() { Clear(); }
    
    // Non-copyable, non-movable (thread-local ownership)
    TensorStreamPool(const TensorStreamPool&) = delete;
    TensorStreamPool& operator=(const TensorStreamPool&) = delete;
    TensorStreamPool(TensorStreamPool&&) = delete;
    TensorStreamPool& operator=(TensorStreamPool&&) = delete;

private:
    // -------------------------------------------------------------------------
    // Internal Types
    // -------------------------------------------------------------------------
    
    /**
     * @brief Metadata for a single buffer in the pool.
     */
    struct BufferEntry {
        std::unique_ptr<char, decltype(&AlignedFreeStream)> storage{nullptr, &AlignedFreeStream};
        float* aligned_ptr = nullptr;
        size_t size_bytes = 0;
        size_t bucket_size = 0;      ///< Size bucket for reuse matching
        bool in_use = false;
        bool handed_off = false;
        const char* current_owner = nullptr;
        std::chrono::steady_clock::time_point acquired_at;
    };
    
    // -------------------------------------------------------------------------
    // Internal Methods
    // -------------------------------------------------------------------------
    
    /**
     * @brief Compute size bucket for efficient buffer reuse.
     *
     * Rounds up to power-of-2 buckets with minimum kMinBufferSize.
     * This reduces fragmentation by reusing slightly larger buffers.
     */
    static size_t ComputeBucket(size_t size_bytes);
    
    /**
     * @brief Find existing buffer that can serve the requested size.
     * @return Pointer to buffer entry, or nullptr if none available.
     */
    BufferEntry* FindReusableBuffer(size_t bucket_size);
    
    /**
     * @brief Find buffer entry by pointer.
     * @return Pointer to buffer entry, or nullptr if not found.
     */
    BufferEntry* FindByPointer(float* ptr);
    
    /**
     * @brief Update lifetime statistics when buffer is released.
     */
    void UpdateLifetimeStats(const BufferEntry& entry);
    
    // -------------------------------------------------------------------------
    // Data Members
    // -------------------------------------------------------------------------
    
    // Map from bucket size to list of buffers
    std::unordered_map<size_t, std::vector<BufferEntry>> buffers_;
    
    // Statistics
    Stats stats_;
    
    // Lifetime tracking (for avg_buffer_lifetime_ms)
    size_t lifetime_samples_ = 0;
    double lifetime_sum_ms_ = 0;
};

// =============================================================================
// GLOBAL SHARED INSTANCE
// =============================================================================

/**
 * @brief Get the global shared TensorStreamPool instance.
 *
 * Defined in tensor_stream_pool.cc to ensure exactly one instance exists
 * across all translation units. This is critical for TensorFlow ops
 * which may be compiled into the same shared library.
 * 
 * Usage:
 *   saguaro::ops::GetTensorStreamPool().Acquire(...);
 *   saguaro::ops::GetTensorStreamPool().Handoff(...);
 *   saguaro::ops::GetTensorStreamPool().Release(...);
 */
TensorStreamPool& GetTensorStreamPool();

// =============================================================================
// CONVENIENCE MACROS
// =============================================================================

/**
 * @brief Acquire buffer with automatic producer name from function.
 */
#define TENSOR_STREAM_ACQUIRE(size_bytes) \
    saguaro::ops::GetTensorStreamPool().Acquire((size_bytes), __func__)

/**
 * @brief Handoff buffer with automatic consumer hint.
 */
#define TENSOR_STREAM_HANDOFF(ptr, consumer) \
    saguaro::ops::GetTensorStreamPool().Handoff((ptr), (consumer))

/**
 * @brief Release buffer.
 */
#define TENSOR_STREAM_RELEASE(ptr) \
    saguaro::ops::GetTensorStreamPool().Release((ptr))

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_COMMON_TENSOR_STREAM_POOL_H_
