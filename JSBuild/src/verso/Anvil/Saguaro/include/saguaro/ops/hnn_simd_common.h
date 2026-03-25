// saguaro/native/ops/hnn_simd_common.h
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
 * @file hnn_simd_common.h
 * @brief Shared SIMD activation and normalization functions.
 *
 * Provides vectorized implementations of common neural network operations:
 * - Exponential (exp) with Taylor 5th order approximation
 * - Sigmoid: 1 / (1 + exp(-x))
 * - SiLU: x * sigmoid(x)
 * - GELU: Gaussian Error Linear Unit
 * - Softmax: numerically stable with max subtraction
 * - LayerNorm: fused gamma/beta normalization
 *
 * SIMD Support:
 * - AVX-512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - ARM NEON: 4-wide vectorization
 * - Scalar fallback for all architectures
 *
 * All functions use the simd_ prefix and are in namespace saguaro::ops.
 * Thread-safe: all functions are reentrant with no shared state.
 *
 * Precision: float32 only (maintains O(n) linear complexity)
 */

#ifndef SAGUARO_NATIVE_OPS_HNN_SIMD_COMMON_H_
#define SAGUARO_NATIVE_OPS_HNN_SIMD_COMMON_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include <thread>  // For std::thread::hardware_concurrency() in get_optimal_path_thread_count()

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__)
#include <immintrin.h>
#define HNN_SIMD_ALIGNMENT 64  // 512-bit vectors
#elif defined(__AVX2__)
#include <immintrin.h>
#define HNN_SIMD_ALIGNMENT 32  // 256-bit vectors
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HNN_SIMD_ALIGNMENT 16  // 128-bit vectors
#else
#define HNN_SIMD_ALIGNMENT 16  // Default alignment
#endif

// Cross-platform aligned memory allocation
#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

// =============================================================================
// SIMD-OPTIMAL MEMORY ALIGNMENT UTILITIES
// =============================================================================

/**
 * @brief Allocate memory aligned for optimal SIMD performance.
 * 
 * Aligns to 32 bytes for AVX2 (256-bit), 64 bytes for AVX-512 (512-bit),
 * or 16 bytes for NEON (128-bit) depending on detected SIMD support.
 * 
 * @param bytes Number of bytes to allocate
 * @return Aligned pointer, or nullptr on failure
 */
inline void* aligned_alloc_simd(size_t bytes) {
#ifdef _WIN32
    return _aligned_malloc(bytes, HNN_SIMD_ALIGNMENT);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, HNN_SIMD_ALIGNMENT, bytes) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Free memory allocated with aligned_alloc_simd.
 * @param ptr Pointer returned by aligned_alloc_simd
 */
inline void aligned_free_simd(void* ptr) {
    if (ptr == nullptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// =============================================================================
// PHASE V2.0-P1.3: THREAD-LOCAL PATH SCRATCH BUFFER POOL
// =============================================================================
// Reduces memory allocation overhead by reusing scratch buffers across
// QHD, COCONUT, and SuperposedExpert path computations. Each thread
// maintains its own pool to avoid lock contention.
//
// Usage:
//   float* scratch = g_path_scratch.get(needed_size);
//   // ... use scratch buffer (valid until next get() or clear())
//   // No explicit free needed - pool manages lifetime
//
// Memory alignment: Automatically aligned for optimal SIMD performance
// (64 bytes for AVX-512, 32 bytes for AVX2, 16 bytes for NEON)
//
// See HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 11.6 (P-1.1)

/**
 * @brief Thread-local scratch buffer pool for path computations.
 * 
 * Provides reusable aligned memory for QHD, COCONUT, and MoE path buffers.
 * Grows as needed but never shrinks (amortized O(1) allocation).
 */
struct PathScratchPool {
    float* buffer = nullptr;
    size_t capacity = 0;  // In floats
    
    /**
     * @brief Get a scratch buffer of at least the requested size.
     * 
     * @param size_floats Number of floats needed
     * @return Pointer to aligned scratch buffer (valid until next get())
     */
    float* get(size_t size_floats) {
        if (size_floats > capacity) {
            // Double capacity to amortize reallocations
            size_t new_capacity = std::max(size_floats, capacity * 2);
            // Minimum 64KB to avoid frequent small reallocations
            new_capacity = std::max(new_capacity, size_t(16384));
            
            if (buffer != nullptr) {
                aligned_free_simd(buffer);
            }
            buffer = static_cast<float*>(aligned_alloc_simd(new_capacity * sizeof(float)));
            capacity = (buffer != nullptr) ? new_capacity : 0;
        }
        return buffer;
    }
    
    /**
     * @brief Clear the pool, freeing all memory.
     */
    void clear() {
        if (buffer != nullptr) {
            aligned_free_simd(buffer);
            buffer = nullptr;
            capacity = 0;
        }
    }
    
    ~PathScratchPool() {
        clear();
    }
};

// Thread-local scratch pool instance for path computations
// Each thread gets its own pool to avoid lock contention
// NOTE: 'inline' is required (C++17) to avoid multiple definition errors
// when this header is included in multiple translation units
inline thread_local PathScratchPool g_path_scratch;

/**
 * @brief Secondary scratch pool for operations needing two buffers.
 * 
 * Some operations (e.g., FFT) need two scratch buffers simultaneously.
 * This provides a second pool for such cases.
 */
inline thread_local PathScratchPool g_path_scratch_secondary;

/**
 * @brief Get the optimal number of threads for parallel path computation.
 * 
 * Returns a thread count that balances parallelism with overhead:
 * - 1-4 cores: use all cores
 * - 5-8 cores: use 85% of cores (leave some for OS)
 * - 9+ cores: use 75% of cores (diminishing returns)
 * 
 * @return Recommended thread count for path parallelism
 */
inline int get_optimal_path_thread_count() {
    static int cached_count = 0;
    if (cached_count > 0) return cached_count;
    
    int cores = static_cast<int>(std::thread::hardware_concurrency());
    if (cores <= 0) cores = 4;  // Safe default
    
    if (cores <= 4) {
        cached_count = cores;
    } else if (cores <= 8) {
        cached_count = std::max(1, static_cast<int>(cores * 0.85));
    } else {
        cached_count = std::max(1, static_cast<int>(cores * 0.75));
    }
    
    return cached_count;
}

namespace saguaro {
namespace ops {

// =============================================================================
// EXPONENTIAL APPROXIMATION
// Taylor 5th order with range reduction for full float range accuracy
// exp(x) = 2^n * exp(r) where n = floor(x / ln2), r = x - n*ln2
// For r ∈ [-0.35, 0.35]: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
// =============================================================================

/**
 * @brief Vectorized in-place exp with range reduction.
 *
 * Uses range reduction: exp(x) = 2^n * exp(r)
 * where n = round(x / ln2) and r = x - n*ln2 (so |r| <= 0.35)
 * Then Taylor 5th order is accurate to ~1e-6.
 *
 * @param data Float array to apply exp in-place
 * @param size Number of elements
 */
inline void simd_exp_inplace(float* data, int64_t size) {
    // Constants
    constexpr float kLog2E = 1.442695041f;   // 1 / ln(2)
    constexpr float kLn2Hi = 0.693359375f;   // High bits of ln(2)
    constexpr float kLn2Lo = -2.12194440e-4f; // Low bits of ln(2)
    
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 log2e = _mm512_set1_ps(kLog2E);
    const __m512 ln2_hi = _mm512_set1_ps(kLn2Hi);
    const __m512 ln2_lo = _mm512_set1_ps(kLn2Lo);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 c2 = _mm512_set1_ps(0.5f);           // 1/2!
    const __m512 c3 = _mm512_set1_ps(0.16666667f);    // 1/3!
    const __m512 c4 = _mm512_set1_ps(0.04166667f);    // 1/4!
    const __m512 c5 = _mm512_set1_ps(0.00833333f);    // 1/5!
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        
        // Range reduction: n = round(x * log2(e))
        __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, log2e), 
                                         _MM_FROUND_TO_NEAREST_INT);
        
        // r = x - n * ln(2)  (using hi/lo for precision)
        __m512 r = _mm512_sub_ps(x, _mm512_mul_ps(n, ln2_hi));
        r = _mm512_sub_ps(r, _mm512_mul_ps(n, ln2_lo));
        
        // Taylor expansion: exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
        __m512 r2 = _mm512_mul_ps(r, r);
        __m512 r3 = _mm512_mul_ps(r2, r);
        __m512 r4 = _mm512_mul_ps(r2, r2);
        __m512 r5 = _mm512_mul_ps(r4, r);
        
        __m512 exp_r = _mm512_add_ps(one, r);
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r2, c2));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r3, c3));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r4, c4));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r5, c5));
        
        // Multiply by 2^n using scalef (available in AVX-512)
        __m512 result = _mm512_scalef_ps(exp_r, n);
        
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 log2e = _mm256_set1_ps(kLog2E);
    const __m256 ln2_hi = _mm256_set1_ps(kLn2Hi);
    const __m256 ln2_lo = _mm256_set1_ps(kLn2Lo);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    const __m256 c5 = _mm256_set1_ps(0.00833333f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        
        // Range reduction: n = round(x * log2(e))
        __m256 n = _mm256_round_ps(_mm256_mul_ps(x, log2e), 
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        
        // r = x - n * ln(2)
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n, ln2_lo));
        
        // Taylor expansion using Horner's method with FMA (FMA3)
        // exp(r) ≈ 1 + r(1 + r(c2 + r(c3 + r(c4 + r*c5))))
        // Evaluates as: p = c5; p = fma(p, r, c4); p = fma(p, r, c3); ...
        __m256 exp_r = _mm256_fmadd_ps(c5, r, c4);        // c5*r + c4
        exp_r = _mm256_fmadd_ps(exp_r, r, c3);            // (c5*r + c4)*r + c3
        exp_r = _mm256_fmadd_ps(exp_r, r, c2);            // ... + c2
        exp_r = _mm256_fmadd_ps(exp_r, r, one);           // ... + 1
        exp_r = _mm256_fmadd_ps(exp_r, r, one);           // ... * r + 1 = final exp(r)
        
        // Multiply by 2^n: convert n to int, add to exponent
        __m256i n_int = _mm256_cvtps_epi32(n);
        n_int = _mm256_slli_epi32(n_int, 23);  // Shift to exponent position
        __m256 pow2n = _mm256_castsi256_ps(_mm256_add_epi32(
            _mm256_castps_si256(one), n_int));
        // Correct for 2^0 = 1 bias
        pow2n = _mm256_castsi256_ps(_mm256_add_epi32(
            _mm256_castps_si256(exp_r), n_int));
        
        // Actually: manipulate float bits properly
        // For 2^n, add n<<23 to the float's exponent bits
        __m256i exp_r_bits = _mm256_castps_si256(exp_r);
        __m256i result_bits = _mm256_add_epi32(exp_r_bits, n_int);
        __m256 result = _mm256_castsi256_ps(result_bits);
        
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t log2e = vdupq_n_f32(kLog2E);
    const float32x4_t ln2_hi = vdupq_n_f32(kLn2Hi);
    const float32x4_t ln2_lo = vdupq_n_f32(kLn2Lo);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.16666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166667f);
    const float32x4_t c5 = vdupq_n_f32(0.00833333f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        
        // n = round(x * log2(e))
        float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));
        
        // r = x - n * ln(2)
        float32x4_t r = vsubq_f32(x, vmulq_f32(n, ln2_hi));
        r = vsubq_f32(r, vmulq_f32(n, ln2_lo));
        
        // Taylor expansion
        float32x4_t r2 = vmulq_f32(r, r);
        float32x4_t r3 = vmulq_f32(r2, r);
        float32x4_t r4 = vmulq_f32(r2, r2);
        float32x4_t r5 = vmulq_f32(r4, r);
        
        float32x4_t exp_r = vaddq_f32(one, r);
        exp_r = vaddq_f32(exp_r, vmulq_f32(r2, c2));
        exp_r = vaddq_f32(exp_r, vmulq_f32(r3, c3));
        exp_r = vaddq_f32(exp_r, vmulq_f32(r4, c4));
        exp_r = vaddq_f32(exp_r, vmulq_f32(r5, c5));
        
        // Multiply by 2^n via bit manipulation
        int32x4_t n_int = vcvtq_s32_f32(n);
        n_int = vshlq_n_s32(n_int, 23);
        int32x4_t exp_r_bits = vreinterpretq_s32_f32(exp_r);
        int32x4_t result_bits = vaddq_s32(exp_r_bits, n_int);
        float32x4_t result = vreinterpretq_f32_s32(result_bits);
        
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::exp(data[i]);
    }
}

// =============================================================================
// LOGARITHM APPROXIMATION
// log(x) = n*ln(2) + log(m) where x = 2^n * m, m ∈ [1, 2)
// For m ∈ [1, 2): log(m) ≈ (m-1) - (m-1)²/2 + (m-1)³/3 - (m-1)⁴/4 + (m-1)⁵/5
// =============================================================================

/**
 * @brief Vectorized in-place log with range reduction.
 *
 * Uses range reduction: log(x) = n*ln(2) + log(m)
 * where n = exponent bits and m = mantissa with exponent set to 0.
 * Then polynomial approximation for log(1+t) where t = m - 1.
 *
 * @param data Float array to apply log in-place (must be positive)
 * @param size Number of elements
 */
inline void simd_log_inplace(float* data, int64_t size) {
    // Constants for log approximation
    constexpr float kLn2 = 0.693147181f;
    constexpr int32_t kExpBias = 127;
    constexpr int32_t kMantissaMask = 0x007FFFFF;
    constexpr int32_t kOneAsInt = 0x3F800000;  // 1.0f as int
    
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 ln2 = _mm512_set1_ps(kLn2);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 c1 = _mm512_set1_ps(1.0f);
    const __m512 c2 = _mm512_set1_ps(-0.5f);           // -1/2
    const __m512 c3 = _mm512_set1_ps(0.33333333f);     // 1/3
    const __m512 c4 = _mm512_set1_ps(-0.25f);          // -1/4
    const __m512 c5 = _mm512_set1_ps(0.2f);            // 1/5
    const __m512i mantissa_mask = _mm512_set1_epi32(kMantissaMask);
    const __m512i one_as_int = _mm512_set1_epi32(kOneAsInt);
    const __m512i exp_bias = _mm512_set1_epi32(kExpBias);
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        
        // Clamp to minimum positive value to avoid log(0) or log(negative)
        x = _mm512_max_ps(x, _mm512_set1_ps(1e-38f));
        
        // Extract exponent: n = floor(log2(x)) = exponent_bits - 127
        __m512i x_int = _mm512_castps_si512(x);
        __m512i exponent = _mm512_srli_epi32(x_int, 23);
        exponent = _mm512_sub_epi32(exponent, exp_bias);
        __m512 n = _mm512_cvtepi32_ps(exponent);
        
        // Extract mantissa and set exponent to 0: m = x * 2^(-n), so m ∈ [1, 2)
        __m512i mantissa = _mm512_and_si512(x_int, mantissa_mask);
        mantissa = _mm512_or_si512(mantissa, one_as_int);
        __m512 m = _mm512_castsi512_ps(mantissa);
        
        // t = m - 1 (so t ∈ [0, 1))
        __m512 t = _mm512_sub_ps(m, one);
        
        // log(1+t) ≈ t - t²/2 + t³/3 - t⁴/4 + t⁵/5 (Taylor series)
        __m512 t2 = _mm512_mul_ps(t, t);
        __m512 t3 = _mm512_mul_ps(t2, t);
        __m512 t4 = _mm512_mul_ps(t2, t2);
        __m512 t5 = _mm512_mul_ps(t4, t);
        
        __m512 log_m = _mm512_mul_ps(t, c1);
        log_m = _mm512_add_ps(log_m, _mm512_mul_ps(t2, c2));
        log_m = _mm512_add_ps(log_m, _mm512_mul_ps(t3, c3));
        log_m = _mm512_add_ps(log_m, _mm512_mul_ps(t4, c4));
        log_m = _mm512_add_ps(log_m, _mm512_mul_ps(t5, c5));
        
        // log(x) = n * ln(2) + log(m)
        __m512 result = _mm512_add_ps(_mm512_mul_ps(n, ln2), log_m);
        
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 ln2 = _mm256_set1_ps(kLn2);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(-0.5f);
    const __m256 c3 = _mm256_set1_ps(0.33333333f);
    const __m256 c4 = _mm256_set1_ps(-0.25f);
    const __m256 c5 = _mm256_set1_ps(0.2f);
    const __m256i mantissa_mask = _mm256_set1_epi32(kMantissaMask);
    const __m256i one_as_int = _mm256_set1_epi32(kOneAsInt);
    const __m256i exp_bias = _mm256_set1_epi32(kExpBias);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        
        // Clamp to minimum positive value
        x = _mm256_max_ps(x, _mm256_set1_ps(1e-38f));
        
        // Extract exponent
        __m256i x_int = _mm256_castps_si256(x);
        __m256i exponent = _mm256_srli_epi32(x_int, 23);
        exponent = _mm256_sub_epi32(exponent, exp_bias);
        __m256 n = _mm256_cvtepi32_ps(exponent);
        
        // Extract mantissa
        __m256i mantissa = _mm256_and_si256(x_int, mantissa_mask);
        mantissa = _mm256_or_si256(mantissa, one_as_int);
        __m256 m = _mm256_castsi256_ps(mantissa);
        
        // t = m - 1
        __m256 t = _mm256_sub_ps(m, one);
        
        // Taylor series using Horner's method with FMA (FMA3)
        // log(1+t) ≈ t - t²/2 + t³/3 - t⁴/4 + t⁵/5
        // Horner form: t(1 + t(-0.5 + t(0.333 + t(-0.25 + t*0.2))))
        __m256 log_m = _mm256_fmadd_ps(c5, t, c4);     // 0.2*t + (-0.25)
        log_m = _mm256_fmadd_ps(log_m, t, c3);          // (...)*t + 0.333
        log_m = _mm256_fmadd_ps(log_m, t, c2);          // (...)*t + (-0.5)
        log_m = _mm256_fmadd_ps(log_m, t, c1);          // (...)*t + 1
        log_m = _mm256_mul_ps(log_m, t);                // (...)*t = final log(1+t)
        
        // log(x) = n * ln(2) + log(m) - use FMA for final step
        __m256 result = _mm256_fmadd_ps(n, ln2, log_m);
        
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t ln2 = vdupq_n_f32(kLn2);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(-0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.33333333f);
    const float32x4_t c4 = vdupq_n_f32(-0.25f);
    const float32x4_t c5 = vdupq_n_f32(0.2f);
    const int32x4_t mantissa_mask = vdupq_n_s32(kMantissaMask);
    const int32x4_t one_as_int = vdupq_n_s32(kOneAsInt);
    const int32x4_t exp_bias = vdupq_n_s32(kExpBias);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        
        // Clamp to minimum positive value
        x = vmaxq_f32(x, vdupq_n_f32(1e-38f));
        
        // Extract exponent
        int32x4_t x_int = vreinterpretq_s32_f32(x);
        int32x4_t exponent = vshrq_n_s32(x_int, 23);
        exponent = vsubq_s32(exponent, exp_bias);
        float32x4_t n = vcvtq_f32_s32(exponent);
        
        // Extract mantissa
        int32x4_t mantissa = vandq_s32(x_int, mantissa_mask);
        mantissa = vorrq_s32(mantissa, one_as_int);
        float32x4_t m = vreinterpretq_f32_s32(mantissa);
        
        // t = m - 1
        float32x4_t t = vsubq_f32(m, one);
        
        // Taylor series
        float32x4_t t2 = vmulq_f32(t, t);
        float32x4_t t3 = vmulq_f32(t2, t);
        float32x4_t t4 = vmulq_f32(t2, t2);
        float32x4_t t5 = vmulq_f32(t4, t);
        
        float32x4_t log_m = vmulq_f32(t, c1);
        log_m = vaddq_f32(log_m, vmulq_f32(t2, c2));
        log_m = vaddq_f32(log_m, vmulq_f32(t3, c3));
        log_m = vaddq_f32(log_m, vmulq_f32(t4, c4));
        log_m = vaddq_f32(log_m, vmulq_f32(t5, c5));
        
        // log(x) = n * ln(2) + log(m)
        float32x4_t result = vaddq_f32(vmulq_f32(n, ln2), log_m);
        
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        data[i] = std::log(data[i] > 0.0f ? data[i] : 1e-38f);
    }
}

// =============================================================================
// COMPLEX MULTIPLICATION: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// For FFT butterfly operations and other complex arithmetic
// =============================================================================

/**
 * @brief Vectorized complex multiply with separate real/imaginary arrays.
 *
 * Computes (a+bi)(c+di) = (ac-bd) + (ad+bc)i for n complex numbers.
 * Uses split real/imaginary format for efficient SIMD access.
 *
 * @param out_re Output real part [n] (can alias a_re or b_re)
 * @param out_im Output imaginary part [n] (can alias a_im or b_im)
 * @param a_re First input real part [n]
 * @param a_im First input imaginary part [n]
 * @param b_re Second input real part [n]
 * @param b_im Second input imaginary part [n]
 * @param n Number of complex elements
 */
inline void simd_complex_mul(
    float* __restrict__ out_re, float* __restrict__ out_im,
    const float* a_re, const float* a_im,
    const float* b_re, const float* b_im, int64_t n
) {
    int64_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m512 ar = _mm512_loadu_ps(&a_re[i]);
        __m512 ai = _mm512_loadu_ps(&a_im[i]);
        __m512 br = _mm512_loadu_ps(&b_re[i]);
        __m512 bi = _mm512_loadu_ps(&b_im[i]);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        __m512 ac = _mm512_mul_ps(ar, br);
        __m512 bd = _mm512_mul_ps(ai, bi);
        __m512 ad = _mm512_mul_ps(ar, bi);
        __m512 bc = _mm512_mul_ps(ai, br);

        _mm512_storeu_ps(&out_re[i], _mm512_sub_ps(ac, bd));
        _mm512_storeu_ps(&out_im[i], _mm512_add_ps(ad, bc));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 ar = _mm256_loadu_ps(&a_re[i]);
        __m256 ai = _mm256_loadu_ps(&a_im[i]);
        __m256 br = _mm256_loadu_ps(&b_re[i]);
        __m256 bi = _mm256_loadu_ps(&b_im[i]);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        __m256 ac = _mm256_mul_ps(ar, br);
        __m256 bd = _mm256_mul_ps(ai, bi);
        __m256 ad = _mm256_mul_ps(ar, bi);
        __m256 bc = _mm256_mul_ps(ai, br);

        _mm256_storeu_ps(&out_re[i], _mm256_sub_ps(ac, bd));
        _mm256_storeu_ps(&out_im[i], _mm256_add_ps(ad, bc));
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t ar = vld1q_f32(&a_re[i]);
        float32x4_t ai = vld1q_f32(&a_im[i]);
        float32x4_t br = vld1q_f32(&b_re[i]);
        float32x4_t bi = vld1q_f32(&b_im[i]);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        float32x4_t ac = vmulq_f32(ar, br);
        float32x4_t bd = vmulq_f32(ai, bi);
        float32x4_t ad = vmulq_f32(ar, bi);
        float32x4_t bc = vmulq_f32(ai, br);

        vst1q_f32(&out_re[i], vsubq_f32(ac, bd));
        vst1q_f32(&out_im[i], vaddq_f32(ad, bc));
    }
#endif
    // Scalar fallback for remainder
    for (; i < n; ++i) {
        float ar = a_re[i], ai = a_im[i];
        float br = b_re[i], bi = b_im[i];
        out_re[i] = ar * br - ai * bi;
        out_im[i] = ar * bi + ai * br;
    }
}

/**
 * @brief Vectorized complex multiply-accumulate: out += a * b
 *
 * Computes out += (a+bi)(c+di) = (ac-bd) + (ad+bc)i for n complex numbers.
 * Useful for FFT butterfly accumulation.
 *
 * @param out_re Output/accumulator real part [n]
 * @param out_im Output/accumulator imaginary part [n]
 * @param a_re First input real part [n]
 * @param a_im First input imaginary part [n]
 * @param b_re Second input real part [n]
 * @param b_im Second input imaginary part [n]
 * @param n Number of complex elements
 */
inline void simd_complex_mul_add(
    float* __restrict__ out_re, float* __restrict__ out_im,
    const float* a_re, const float* a_im,
    const float* b_re, const float* b_im, int64_t n
) {
    int64_t i = 0;

#if defined(__AVX512F__)
    for (; i + 16 <= n; i += 16) {
        __m512 ar = _mm512_loadu_ps(&a_re[i]);
        __m512 ai = _mm512_loadu_ps(&a_im[i]);
        __m512 br = _mm512_loadu_ps(&b_re[i]);
        __m512 bi = _mm512_loadu_ps(&b_im[i]);
        __m512 or_ = _mm512_loadu_ps(&out_re[i]);
        __m512 oi = _mm512_loadu_ps(&out_im[i]);

        // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        __m512 ac = _mm512_mul_ps(ar, br);
        __m512 bd = _mm512_mul_ps(ai, bi);
        __m512 ad = _mm512_mul_ps(ar, bi);
        __m512 bc = _mm512_mul_ps(ai, br);

        _mm512_storeu_ps(&out_re[i], _mm512_add_ps(or_, _mm512_sub_ps(ac, bd)));
        _mm512_storeu_ps(&out_im[i], _mm512_add_ps(oi, _mm512_add_ps(ad, bc)));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= n; i += 8) {
        __m256 ar = _mm256_loadu_ps(&a_re[i]);
        __m256 ai = _mm256_loadu_ps(&a_im[i]);
        __m256 br = _mm256_loadu_ps(&b_re[i]);
        __m256 bi = _mm256_loadu_ps(&b_im[i]);
        __m256 or_ = _mm256_loadu_ps(&out_re[i]);
        __m256 oi = _mm256_loadu_ps(&out_im[i]);

        __m256 ac = _mm256_mul_ps(ar, br);
        __m256 bd = _mm256_mul_ps(ai, bi);
        __m256 ad = _mm256_mul_ps(ar, bi);
        __m256 bc = _mm256_mul_ps(ai, br);

        _mm256_storeu_ps(&out_re[i], _mm256_add_ps(or_, _mm256_sub_ps(ac, bd)));
        _mm256_storeu_ps(&out_im[i], _mm256_add_ps(oi, _mm256_add_ps(ad, bc)));
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        float32x4_t ar = vld1q_f32(&a_re[i]);
        float32x4_t ai = vld1q_f32(&a_im[i]);
        float32x4_t br = vld1q_f32(&b_re[i]);
        float32x4_t bi = vld1q_f32(&b_im[i]);
        float32x4_t or_ = vld1q_f32(&out_re[i]);
        float32x4_t oi = vld1q_f32(&out_im[i]);

        float32x4_t ac = vmulq_f32(ar, br);
        float32x4_t bd = vmulq_f32(ai, bi);
        float32x4_t ad = vmulq_f32(ar, bi);
        float32x4_t bc = vmulq_f32(ai, br);

        vst1q_f32(&out_re[i], vaddq_f32(or_, vsubq_f32(ac, bd)));
        vst1q_f32(&out_im[i], vaddq_f32(oi, vaddq_f32(ad, bc)));
    }
#endif
    // Scalar fallback for remainder
    for (; i < n; ++i) {
        float ar = a_re[i], ai = a_im[i];
        float br = b_re[i], bi = b_im[i];
        out_re[i] += ar * br - ai * bi;
        out_im[i] += ar * bi + ai * br;
    }
}

// =============================================================================
// SIGMOID: σ(x) = 1 / (1 + exp(-x))
// =============================================================================

/**
 * @brief Vectorized in-place sigmoid activation.
 *
 * @param data Float array to apply sigmoid in-place
 * @param size Number of elements
 */
inline void simd_sigmoid_inplace(float* data, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    const __m512 log2e = _mm512_set1_ps(1.442695041f);
    const __m512 ln2_hi = _mm512_set1_ps(0.693359375f);
    const __m512 ln2_lo = _mm512_set1_ps(-2.12194440e-4f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.16666667f);
    const __m512 c4 = _mm512_set1_ps(0.04166667f);
    const __m512 c5 = _mm512_set1_ps(0.00833333f);
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        
        // Numerically stable sigmoid: if x >= 0, 1/(1+exp(-x)); else exp(x)/(1+exp(x))
        // For SIMD, we can use 1 / (1 + exp(-|x|)) and then adjust if x < 0:
        // if x < 0, result = 1 - sigmoid(|x|)
        
        __m512 abs_x = _mm512_abs_ps(x);
        __m512 neg_abs_x = _mm512_mul_ps(abs_x, neg_one);
        
        // Compute exp(-|x|) using Taylor
        __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(neg_abs_x, log2e), _MM_FROUND_TO_NEAREST_INT);
        __m512 r = _mm512_sub_ps(neg_abs_x, _mm512_mul_ps(n, ln2_hi));
        r = _mm512_sub_ps(r, _mm512_mul_ps(n, ln2_lo));
        
        __m512 r2 = _mm512_mul_ps(r, r);
        __m512 r3 = _mm512_mul_ps(r2, r);
        __m512 r4 = _mm512_mul_ps(r2, r2);
        __m512 r5 = _mm512_mul_ps(r4, r);
        
        __m512 exp_r = _mm512_add_ps(one, r);
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r2, c2));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r3, c3));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r4, c4));
        exp_r = _mm512_add_ps(exp_r, _mm512_mul_ps(r5, c5));
        
        __m512 exp_neg_abs = _mm512_scalef_ps(exp_r, n);
        
        // sig = 1 / (1 + exp(-|x|))
        __m512 sig = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_abs));
        
        // if x < 0, result = 1 - sig
        __mmask16 mask = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_LT_OS);
        __m512 result = _mm512_mask_sub_ps(sig, mask, one, sig);
        
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 log2e = _mm256_set1_ps(1.442695041f);
    const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
    const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    const __m256 c5 = _mm256_set1_ps(0.00833333f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        
        // sig(x) = 1 / (1 + exp(-|x|)) if x >= 0 else 1 - sig(|x|)
        __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
        __m256 neg_abs_x = _mm256_mul_ps(abs_x, neg_one);
        
        // Compute exp(-|x|)
        __m256 n = _mm256_round_ps(_mm256_mul_ps(neg_abs_x, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 r = _mm256_sub_ps(neg_abs_x, _mm256_mul_ps(n, ln2_hi));
        r = _mm256_sub_ps(r, _mm256_mul_ps(n, ln2_lo));
        
        // Taylor (Horner)
        __m256 exp_r = _mm256_fmadd_ps(c5, r, c4);
        exp_r = _mm256_fmadd_ps(exp_r, r, c3);
        exp_r = _mm256_fmadd_ps(exp_r, r, c2);
        exp_r = _mm256_fmadd_ps(exp_r, r, one);
        exp_r = _mm256_fmadd_ps(exp_r, r, one);
        
        // Scale by 2^n
        __m256i n_int = _mm256_cvtps_epi32(n);
        n_int = _mm256_slli_epi32(n_int, 23);
        __m256 exp_neg_abs = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(exp_r), n_int));
        
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_abs));
        
        // Handle x < 0
        __m256 lt_zero_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LT_OS);
        __m256 result = _mm256_blendv_ps(sig, _mm256_sub_ps(one, sig), lt_zero_mask);
        
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.16666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166667f);
    const float32x4_t c5 = vdupq_n_f32(0.00833333f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        float32x4_t neg_x = vmulq_f32(x, neg_one);
        
        float32x4_t x2 = vmulq_f32(neg_x, neg_x);
        float32x4_t x3 = vmulq_f32(x2, neg_x);
        float32x4_t x4 = vmulq_f32(x2, x2);
        float32x4_t x5 = vmulq_f32(x4, neg_x);
        
        float32x4_t exp_neg = vaddq_f32(one, neg_x);
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x2, c2));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x3, c3));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x4, c4));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x5, c5));
        
        float32x4_t denom = vaddq_f32(one, exp_neg);
        float32x4_t result = vdivq_f32(one, denom);
        
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        data[i] = 1.0f / (1.0f + std::exp(-data[i]));
    }
}

// =============================================================================
// SILU: x * sigmoid(x)
// =============================================================================

/**
 * @brief Vectorized in-place SiLU (Swish) activation.
 *
 * @param data Float array to apply SiLU in-place
 * @param size Number of elements
 */
inline void simd_silu_inplace(float* data, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    const __m512 c2 = _mm512_set1_ps(0.5f);
    const __m512 c3 = _mm512_set1_ps(0.16666667f);
    const __m512 c4 = _mm512_set1_ps(0.04166667f);
    const __m512 c5 = _mm512_set1_ps(0.00833333f);
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 neg_x = _mm512_mul_ps(x, neg_one);
        
        // Compute sigmoid(x)
        __m512 x2 = _mm512_mul_ps(neg_x, neg_x);
        __m512 x3 = _mm512_mul_ps(x2, neg_x);
        __m512 x4 = _mm512_mul_ps(x2, x2);
        __m512 x5 = _mm512_mul_ps(x4, neg_x);
        
        __m512 exp_neg = _mm512_add_ps(one, neg_x);
        exp_neg = _mm512_add_ps(exp_neg, _mm512_mul_ps(x2, c2));
        exp_neg = _mm512_add_ps(exp_neg, _mm512_mul_ps(x3, c3));
        exp_neg = _mm512_add_ps(exp_neg, _mm512_mul_ps(x4, c4));
        exp_neg = _mm512_add_ps(exp_neg, _mm512_mul_ps(x5, c5));
        
        __m512 denom = _mm512_add_ps(one, exp_neg);
        __m512 sigmoid = _mm512_div_ps(one, denom);
        
        // x * sigmoid(x)
        __m512 result = _mm512_mul_ps(x, sigmoid);
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    const __m256 c2 = _mm256_set1_ps(0.5f);
    const __m256 c3 = _mm256_set1_ps(0.16666667f);
    const __m256 c4 = _mm256_set1_ps(0.04166667f);
    const __m256 c5 = _mm256_set1_ps(0.00833333f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 neg_x = _mm256_mul_ps(x, neg_one);
        
        __m256 x2 = _mm256_mul_ps(neg_x, neg_x);
        __m256 x3 = _mm256_mul_ps(x2, neg_x);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 x5 = _mm256_mul_ps(x4, neg_x);
        
        __m256 exp_neg = _mm256_add_ps(one, neg_x);
        exp_neg = _mm256_add_ps(exp_neg, _mm256_mul_ps(x2, c2));
        exp_neg = _mm256_add_ps(exp_neg, _mm256_mul_ps(x3, c3));
        exp_neg = _mm256_add_ps(exp_neg, _mm256_mul_ps(x4, c4));
        exp_neg = _mm256_add_ps(exp_neg, _mm256_mul_ps(x5, c5));
        
        __m256 denom = _mm256_add_ps(one, exp_neg);
        __m256 sigmoid = _mm256_div_ps(one, denom);
        
        __m256 result = _mm256_mul_ps(x, sigmoid);
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.16666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166667f);
    const float32x4_t c5 = vdupq_n_f32(0.00833333f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        float32x4_t neg_x = vmulq_f32(x, neg_one);
        
        float32x4_t x2 = vmulq_f32(neg_x, neg_x);
        float32x4_t x3 = vmulq_f32(x2, neg_x);
        float32x4_t x4 = vmulq_f32(x2, x2);
        float32x4_t x5 = vmulq_f32(x4, neg_x);
        
        float32x4_t exp_neg = vaddq_f32(one, neg_x);
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x2, c2));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x3, c3));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x4, c4));
        exp_neg = vaddq_f32(exp_neg, vmulq_f32(x5, c5));
        
        float32x4_t denom = vaddq_f32(one, exp_neg);
        float32x4_t sigmoid = vdivq_f32(one, denom);
        
        float32x4_t result = vmulq_f32(x, sigmoid);
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        float sig = 1.0f / (1.0f + std::exp(-data[i]));
        data[i] = data[i] * sig;
    }
}

// =============================================================================
// GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Using tanh approximation for SIMD
// =============================================================================

/**
 * @brief Vectorized in-place GELU activation.
 *
 * Uses the tanh approximation:
 * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * @param data Float array to apply GELU in-place
 * @param size Number of elements
 */
inline void simd_gelu_inplace(float* data, int64_t size) {
    // Constants for GELU
    constexpr float kSqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
    constexpr float kGELUCoeff = 0.044715f;
    
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 sqrt2overpi = _mm512_set1_ps(kSqrt2OverPi);
    const __m512 gelu_coeff = _mm512_set1_ps(kGELUCoeff);
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 x3 = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
        
        // inner = sqrt(2/π) * (x + 0.044715 * x³)
        __m512 inner = _mm512_add_ps(x, _mm512_mul_ps(gelu_coeff, x3));
        inner = _mm512_mul_ps(sqrt2overpi, inner);
        
        // tanh approximation: tanh(x) ≈ x for small x, clamp for large
        // For better accuracy, use polynomial: tanh(x) ≈ x - x³/3 + 2x⁵/15
        __m512 inner2 = _mm512_mul_ps(inner, inner);
        __m512 inner3 = _mm512_mul_ps(inner2, inner);
        __m512 inner5 = _mm512_mul_ps(inner3, inner2);
        __m512 tanh_approx = _mm512_sub_ps(inner, 
            _mm512_mul_ps(inner3, _mm512_set1_ps(0.33333333f)));
        tanh_approx = _mm512_add_ps(tanh_approx,
            _mm512_mul_ps(inner5, _mm512_set1_ps(0.13333333f)));
        
        // Clamp tanh to [-1, 1]
        tanh_approx = _mm512_max_ps(tanh_approx, _mm512_set1_ps(-1.0f));
        tanh_approx = _mm512_min_ps(tanh_approx, one);
        
        // GELU = 0.5 * x * (1 + tanh)
        __m512 result = _mm512_mul_ps(half, 
            _mm512_mul_ps(x, _mm512_add_ps(one, tanh_approx)));
        
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 sqrt2overpi = _mm256_set1_ps(kSqrt2OverPi);
    const __m256 gelu_coeff = _mm256_set1_ps(kGELUCoeff);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 x3 = _mm256_mul_ps(_mm256_mul_ps(x, x), x);
        
        // inner = sqrt(2/π) * (x + 0.044715 * x³) using FMA
        __m256 inner = _mm256_fmadd_ps(gelu_coeff, x3, x);  // x + coeff*x³
        inner = _mm256_mul_ps(sqrt2overpi, inner);
        
        // tanh approximation: tanh(z) ≈ z - z³/3 + 2z⁵/15 using FMA
        const __m256 c_tanh3 = _mm256_set1_ps(-0.33333333f);
        const __m256 c_tanh5 = _mm256_set1_ps(0.13333333f);
        __m256 inner2 = _mm256_mul_ps(inner, inner);
        __m256 inner3 = _mm256_mul_ps(inner2, inner);
        __m256 inner5 = _mm256_mul_ps(inner3, inner2);
        __m256 tanh_approx = _mm256_fmadd_ps(c_tanh5, inner5,
                             _mm256_fmadd_ps(c_tanh3, inner3, inner));
        
        tanh_approx = _mm256_max_ps(tanh_approx, _mm256_set1_ps(-1.0f));
        tanh_approx = _mm256_min_ps(tanh_approx, one);
        
        // GELU = 0.5 * x * (1 + tanh) using FMA
        __m256 one_plus_tanh = _mm256_add_ps(one, tanh_approx);
        __m256 result = _mm256_mul_ps(half, _mm256_mul_ps(x, one_plus_tanh));
        
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t half = vdupq_n_f32(0.5f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t sqrt2overpi = vdupq_n_f32(kSqrt2OverPi);
    const float32x4_t gelu_coeff = vdupq_n_f32(kGELUCoeff);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        float32x4_t x3 = vmulq_f32(vmulq_f32(x, x), x);
        
        float32x4_t inner = vaddq_f32(x, vmulq_f32(gelu_coeff, x3));
        inner = vmulq_f32(sqrt2overpi, inner);
        
        float32x4_t inner2 = vmulq_f32(inner, inner);
        float32x4_t inner3 = vmulq_f32(inner2, inner);
        float32x4_t inner5 = vmulq_f32(inner3, inner2);
        float32x4_t tanh_approx = vsubq_f32(inner,
            vmulq_f32(inner3, vdupq_n_f32(0.33333333f)));
        tanh_approx = vaddq_f32(tanh_approx,
            vmulq_f32(inner5, vdupq_n_f32(0.13333333f)));
        
        tanh_approx = vmaxq_f32(tanh_approx, vdupq_n_f32(-1.0f));
        tanh_approx = vminq_f32(tanh_approx, one);
        
        float32x4_t result = vmulq_f32(half,
            vmulq_f32(x, vaddq_f32(one, tanh_approx)));
        
        vst1q_f32(&data[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        float x = data[i];
        float inner = kSqrt2OverPi * (x + kGELUCoeff * x * x * x);
        float tanh_val = std::tanh(inner);
        data[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

// =============================================================================
// SOFTMAX: numerically stable with max subtraction
// =============================================================================

/**
 * @brief Find maximum value in array using SIMD.
 */
inline float simd_reduce_max(const float* data, int64_t size) {
    float max_val = -std::numeric_limits<float>::infinity();
    int64_t i = 0;

#if defined(__AVX512F__)
    if (size >= 16) {
        __m512 max_vec = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 16 <= size; i += 16) {
            __m512 v = _mm512_loadu_ps(&data[i]);
            max_vec = _mm512_max_ps(max_vec, v);
        }
        max_val = _mm512_reduce_max_ps(max_vec);
    }
#elif defined(__AVX2__)
    if (size >= 8) {
        __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        for (; i + 8 <= size; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            max_vec = _mm256_max_ps(max_vec, v);
        }
        // Horizontal max
        __m128 lo = _mm256_castps256_ps128(max_vec);
        __m128 hi = _mm256_extractf128_ps(max_vec, 1);
        __m128 max128 = _mm_max_ps(lo, hi);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
        max_val = _mm_cvtss_f32(max128);
    }
#elif defined(__ARM_NEON)
    if (size >= 4) {
        float32x4_t max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        for (; i + 4 <= size; i += 4) {
            float32x4_t v = vld1q_f32(&data[i]);
            max_vec = vmaxq_f32(max_vec, v);
        }
        max_val = vmaxvq_f32(max_vec);
    }
#endif
    for (; i < size; ++i) {
        max_val = std::max(max_val, data[i]);
    }
    return max_val;
}

/**
 * @brief Sum array elements using SIMD.
 */
inline float simd_reduce_sum(const float* data, int64_t size) {
    float sum = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        acc = _mm512_add_ps(acc, v);
    }
    sum = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        acc = _mm256_add_ps(acc, v);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum = _mm_cvtss_f32(sum128);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        acc = vaddq_f32(acc, v);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < size; ++i) {
        sum += data[i];
    }
    return sum;
}

/**
 * @brief Vectorized in-place softmax (numerically stable).
 *
 * @param data Float array to apply softmax in-place
 * @param size Number of elements
 */
inline void simd_softmax_inplace(float* data, int64_t size) {
    // Step 1: Find max for numerical stability
    float max_val = simd_reduce_max(data, size);
    
    // Step 2: Subtract max and compute exp
    int64_t i = 0;
#if defined(__AVX512F__)
    const __m512 max_vec = _mm512_set1_ps(max_val);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        v = _mm512_sub_ps(v, max_vec);
        _mm512_storeu_ps(&data[i], v);
    }
#elif defined(__AVX2__)
    const __m256 max_vec = _mm256_set1_ps(max_val);
    for (; i + 8 <= size; i += 8) {
        // Prefetch next cache line (64 bytes = 16 floats ahead)
        _mm_prefetch(reinterpret_cast<const char*>(&data[i + 64]), _MM_HINT_T0);
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_sub_ps(v, max_vec);
        _mm256_storeu_ps(&data[i], v);
    }
#elif defined(__ARM_NEON)
    const float32x4_t max_vec = vdupq_n_f32(max_val);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        v = vsubq_f32(v, max_vec);
        vst1q_f32(&data[i], v);
    }
#endif
    for (; i < size; ++i) {
        data[i] -= max_val;
    }
    
    // Apply exp using SIMD-accelerated Taylor approximation with range reduction
    simd_exp_inplace(data, size);
    
    // Step 3: Normalize by sum
    float sum = simd_reduce_sum(data, size);
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
    
    i = 0;
#if defined(__AVX512F__)
    const __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        _mm512_storeu_ps(&data[i], _mm512_mul_ps(v, inv_sum_vec));
    }
#elif defined(__AVX2__)
    const __m256 inv_sum_vec8 = _mm256_set1_ps(inv_sum);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, inv_sum_vec8));
    }
#elif defined(__ARM_NEON)
    const float32x4_t inv_sum_vec4 = vdupq_n_f32(inv_sum);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vmulq_f32(v, inv_sum_vec4));
    }
#endif
    for (; i < size; ++i) {
        data[i] *= inv_sum;
    }
}

// =============================================================================
// LAYERNORM: output = gamma * (x - mean) / sqrt(var + eps) + beta
// =============================================================================

/**
 * @brief Vectorized LayerNorm.
 *
 * @param input Input array [batch_seq * dim]
 * @param gamma Scale parameter [dim]
 * @param beta Shift parameter [dim]
 * @param output Output array [batch_seq * dim]
 * @param batch_seq Number of samples (batch * seq_len)
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
inline void simd_layernorm(
    const float* input, const float* gamma, const float* beta,
    float* output, int64_t batch_seq, int64_t dim, float eps = 1e-5f) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_seq; ++b) {
        const float* x_row = input + b * dim;
        float* out_row = output + b * dim;
        
        // Compute mean and variance
        float mean = 0.0f;
        float var = 0.0f;
        
#if defined(__AVX512F__)
        if (dim >= 16) {
            __m512 sum_vec = _mm512_setzero_ps();
            int64_t d = 0;
            for (; d + 16 <= dim; d += 16) {
                sum_vec = _mm512_add_ps(sum_vec, _mm512_loadu_ps(&x_row[d]));
            }
            mean = _mm512_reduce_add_ps(sum_vec);
            for (; d < dim; ++d) mean += x_row[d];
            mean /= static_cast<float>(dim);

            __m512 mean_vec = _mm512_set1_ps(mean);
            __m512 var_vec = _mm512_setzero_ps();
            d = 0;
            for (; d + 16 <= dim; d += 16) {
                __m512 diff = _mm512_sub_ps(_mm512_loadu_ps(&x_row[d]), mean_vec);
                var_vec = _mm512_add_ps(var_vec, _mm512_mul_ps(diff, diff));
            }
            var = _mm512_reduce_add_ps(var_vec);
            for (; d < dim; ++d) {
                float diff = x_row[d] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(dim);
        } else {
#elif defined(__AVX2__)
        if (dim >= 8) {
            __m256 sum_vec = _mm256_setzero_ps();
            int64_t d = 0;
            for (; d + 8 <= dim; d += 8) {
                sum_vec = _mm256_add_ps(sum_vec, _mm256_loadu_ps(&x_row[d]));
            }
            // Horizontal sum for AVX2
            __m128 vlow = _mm256_castps256_ps128(sum_vec);
            __m128 vhigh = _mm256_extractf128_ps(sum_vec, 1);
            __m128 res = _mm_add_ps(vlow, vhigh);
            res = _mm_hadd_ps(res, res);
            res = _mm_hadd_ps(res, res);
            mean = _mm_cvtss_f32(res);
            
            for (; d < dim; ++d) mean += x_row[d];
            mean /= static_cast<float>(dim);

            __m256 mean_vec = _mm256_set1_ps(mean);
            __m256 var_vec = _mm256_setzero_ps();
            d = 0;
            for (; d + 8 <= dim; d += 8) {
                __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(&x_row[d]), mean_vec);
                var_vec = _mm256_add_ps(var_vec, _mm256_mul_ps(diff, diff));
            }
            // Horizontal sum for AVX2
            vlow = _mm256_castps256_ps128(var_vec);
            vhigh = _mm256_extractf128_ps(var_vec, 1);
            res = _mm_add_ps(vlow, vhigh);
            res = _mm_hadd_ps(res, res);
            res = _mm_hadd_ps(res, res);
            var = _mm_cvtss_f32(res);
            
            for (; d < dim; ++d) {
                float diff = x_row[d] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(dim);
        } else {
#endif
            for (int64_t d = 0; d < dim; ++d) mean += x_row[d];
            mean /= static_cast<float>(dim);
            for (int64_t d = 0; d < dim; ++d) {
                float diff = x_row[d] - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(dim);
#if defined(__AVX512F__) || defined(__AVX2__)
        }
#endif
        
        // Normalize with SIMD
        float inv_std = 1.0f / std::sqrt(var + eps);
        int64_t d = 0;
        
#if defined(__AVX512F__)
        const __m512 mean_vec = _mm512_set1_ps(mean);
        const __m512 inv_std_vec = _mm512_set1_ps(inv_std);
        for (; d + 16 <= dim; d += 16) {
            __m512 x = _mm512_loadu_ps(&x_row[d]);
            __m512 g = _mm512_loadu_ps(&gamma[d]);
            __m512 bt = _mm512_loadu_ps(&beta[d]);
            
            __m512 normalized = _mm512_mul_ps(_mm512_sub_ps(x, mean_vec), inv_std_vec);
            __m512 result = _mm512_add_ps(_mm512_mul_ps(g, normalized), bt);
            _mm512_storeu_ps(&out_row[d], result);
        }
#elif defined(__AVX2__)
        const __m256 mean_vec = _mm256_set1_ps(mean);
        const __m256 inv_std_vec = _mm256_set1_ps(inv_std);
        const __m256 neg_mean_inv_std = _mm256_set1_ps(-mean * inv_std);
        for (; d + 8 <= dim; d += 8) {
            __m256 x = _mm256_loadu_ps(&x_row[d]);
            __m256 g = _mm256_loadu_ps(&gamma[d]);
            __m256 bt = _mm256_loadu_ps(&beta[d]);
            
            // normalized = (x - mean) * inv_std = x * inv_std - mean * inv_std
            // result = gamma * normalized + beta
            // Fused: result = gamma * (x * inv_std + neg_mean_inv_std) + beta
            __m256 x_scaled = _mm256_mul_ps(x, inv_std_vec);
            __m256 normalized = _mm256_add_ps(x_scaled, neg_mean_inv_std);
            __m256 result = _mm256_fmadd_ps(g, normalized, bt);
            _mm256_storeu_ps(&out_row[d], result);
        }
#elif defined(__ARM_NEON)
        const float32x4_t mean_vec = vdupq_n_f32(mean);
        const float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t x = vld1q_f32(&x_row[d]);
            float32x4_t g = vld1q_f32(&gamma[d]);
            float32x4_t bt = vld1q_f32(&beta[d]);
            
            float32x4_t normalized = vmulq_f32(vsubq_f32(x, mean_vec), inv_std_vec);
            float32x4_t result = vaddq_f32(vmulq_f32(g, normalized), bt);
            vst1q_f32(&out_row[d], result);
        }
#endif
        // Scalar fallback
        for (; d < dim; ++d) {
            out_row[d] = gamma[d] * (x_row[d] - mean) * inv_std + beta[d];
        }
    }
}

// =============================================================================
// ELEMENT-WISE OPERATIONS (from hnn_core_helpers.h)
// =============================================================================

/**
 * @brief Vectorized element-wise Hadamard product.
 */
inline void simd_hadamard_product(const float* a, const float* b, float* out, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_mul_ps(va, vb));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vb));
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vmulq_f32(va, vb));
    }
#endif
    for (; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

/**
 * @brief Vectorized element-wise add.
 */
inline void simd_add(const float* a, const float* b, float* out, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&out[i], _mm512_add_ps(va, vb));
    }
#elif defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_add_ps(va, vb));
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&out[i], vaddq_f32(va, vb));
    }
#endif
    for (; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

/**
 * @brief Vectorized scale in-place: data[i] *= scale
 */
inline void simd_scale_inplace(float* data, float scale, int64_t size) {
    int64_t i = 0;
#if defined(__AVX512F__)
    const __m512 scale_vec = _mm512_set1_ps(scale);
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        _mm512_storeu_ps(&data[i], _mm512_mul_ps(v, scale_vec));
    }
#elif defined(__AVX2__)
    const __m256 scale_vec = _mm256_set1_ps(scale);
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        _mm256_storeu_ps(&data[i], _mm256_mul_ps(v, scale_vec));
    }
#elif defined(__ARM_NEON)
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vmulq_f32(v, scale_vec));
    }
#endif
    for (; i < size; ++i) {
        data[i] *= scale;
    }
}

}  // namespace ops
}  // namespace saguaro

// =============================================================================
// ADDITIONAL CONSOLIDATED SIMD UTILITIES
// These functions consolidate duplicate implementations from 56+ source files
// Organized in hsmn::simd namespace for cleaner API
// =============================================================================

namespace hsmn {
namespace simd {

/**
 * @brief Vectorized dot product of two float arrays.
 *
 * Computes: sum(a[i] * b[i]) for i in [0, size)
 *
 * @param a First input array
 * @param b Second input array
 * @param size Number of elements
 * @return Dot product scalar value
 */
inline float simd_dot_product(const float* a, const float* b, int64_t size) {
    float result = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        acc = _mm512_fmadd_ps(va, vb, acc);  // Fused multiply-add
    }
    result = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc = _mm256_fmadd_ps(va, vb, acc);  // Fused multiply-add (FMA3)
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    result = _mm_cvtss_f32(sum128);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc = vmlaq_f32(acc, va, vb);  // Fused multiply-add
    }
    result = vaddvq_f32(acc);
#endif
    // Scalar fallback for remainder
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

/**
 * @brief Vectorized L2 norm of a float array.
 *
 * Computes: sqrt(sum(x[i]^2)) for i in [0, size)
 *
 * @param x Input array
 * @param size Number of elements
 * @return L2 norm scalar value
 */
inline float simd_norm(const float* x, int64_t size) {
    float sum_sq = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum_sq = _mm_cvtss_f32(sum128);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        acc = vmlaq_f32(acc, v, v);
    }
    sum_sq = vaddvq_f32(acc);
#endif
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    return std::sqrt(sum_sq);
}

/**
 * @brief Vectorized cosine similarity between two float arrays.
 *
 * Computes: dot(a, b) / (norm(a) * norm(b))
 * Returns 0 if either norm is zero to avoid division by zero.
 *
 * @param a First input array
 * @param b Second input array
 * @param size Number of elements
 * @return Cosine similarity in [-1, 1]
 */
inline float simd_cosine_similarity(const float* a, const float* b, int64_t size) {
    float dot = 0.0f;
    float norm_a_sq = 0.0f;
    float norm_b_sq = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc_dot = _mm512_setzero_ps();
    __m512 acc_norm_a = _mm512_setzero_ps();
    __m512 acc_norm_b = _mm512_setzero_ps();
    
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        acc_dot = _mm512_fmadd_ps(va, vb, acc_dot);
        acc_norm_a = _mm512_fmadd_ps(va, va, acc_norm_a);
        acc_norm_b = _mm512_fmadd_ps(vb, vb, acc_norm_b);
    }
    dot = _mm512_reduce_add_ps(acc_dot);
    norm_a_sq = _mm512_reduce_add_ps(acc_norm_a);
    norm_b_sq = _mm512_reduce_add_ps(acc_norm_b);
#elif defined(__AVX2__)
    __m256 acc_dot = _mm256_setzero_ps();
    __m256 acc_norm_a = _mm256_setzero_ps();
    __m256 acc_norm_b = _mm256_setzero_ps();
    
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        acc_dot = _mm256_fmadd_ps(va, vb, acc_dot);
        acc_norm_a = _mm256_fmadd_ps(va, va, acc_norm_a);
        acc_norm_b = _mm256_fmadd_ps(vb, vb, acc_norm_b);
    }
    // Horizontal sums
    auto reduce_m256 = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    };
    dot = reduce_m256(acc_dot);
    norm_a_sq = reduce_m256(acc_norm_a);
    norm_b_sq = reduce_m256(acc_norm_b);
#elif defined(__ARM_NEON)
    float32x4_t acc_dot = vdupq_n_f32(0.0f);
    float32x4_t acc_norm_a = vdupq_n_f32(0.0f);
    float32x4_t acc_norm_b = vdupq_n_f32(0.0f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        acc_dot = vmlaq_f32(acc_dot, va, vb);
        acc_norm_a = vmlaq_f32(acc_norm_a, va, va);
        acc_norm_b = vmlaq_f32(acc_norm_b, vb, vb);
    }
    dot = vaddvq_f32(acc_dot);
    norm_a_sq = vaddvq_f32(acc_norm_a);
    norm_b_sq = vaddvq_f32(acc_norm_b);
#endif
    // Scalar remainder
    for (; i < size; ++i) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }
    
    float denom = std::sqrt(norm_a_sq) * std::sqrt(norm_b_sq);
    return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

/**
 * @brief Compute cosine similarities of a query against multiple keys.
 *
 * For each key k in [0, num_keys): out[k] = cosine_sim(query, keys + k*dim)
 *
 * @param query Query vector [dim]
 * @param keys Key matrix [num_keys, dim] (row-major)
 * @param out Output similarities [num_keys]
 * @param num_keys Number of keys
 * @param dim Dimension of each vector
 */
inline void simd_batch_cosine_similarity(
    const float* query, const float* keys, float* out,
    int64_t num_keys, int64_t dim) {
    
    // Pre-compute query norm
    float query_norm = simd_norm(query, dim);
    float inv_query_norm = (query_norm > 1e-8f) ? (1.0f / query_norm) : 0.0f;
    
    #pragma omp parallel for if(num_keys > 64)
    for (int64_t k = 0; k < num_keys; ++k) {
        const float* key = keys + k * dim;
        float dot = simd_dot_product(query, key, dim);
        float key_norm = simd_norm(key, dim);
        float inv_key_norm = (key_norm > 1e-8f) ? (1.0f / key_norm) : 0.0f;
        out[k] = dot * inv_query_norm * inv_key_norm;
    }
}

/**
 * @brief Vectorized scaled add: out[i] = a[i] + scale * b[i]
 *
 * @param a First input array
 * @param b Second input array (to be scaled)
 * @param scale Scale factor for b
 * @param out Output array (can alias a for in-place)
 * @param size Number of elements
 */
inline void simd_add_scaled(const float* a, const float* b, float scale,
                            float* out, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 scale_vec = _mm512_set1_ps(scale);
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 result = _mm512_fmadd_ps(vb, scale_vec, va);
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 scale_vec = _mm256_set1_ps(scale);
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 result = _mm256_fmadd_ps(vb, scale_vec, va);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t scale_vec = vdupq_n_f32(scale);
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t result = vmlaq_f32(va, vb, scale_vec);
        vst1q_f32(&out[i], result);
    }
#endif
    for (; i < size; ++i) {
        out[i] = a[i] + scale * b[i];
    }
}

/**
 * @brief Vectorized gated update: out[i] = gate[i] * update[i] + (1 - gate[i]) * current[i]
 *
 * Useful for GRU-style updates, memory gating, and Titans surprise-gated writes.
 *
 * @param gate Gate values [size], typically in [0, 1]
 * @param current Current/old values [size]
 * @param update New/update values [size]
 * @param out Output array [size] (can alias current for in-place)
 * @param size Number of elements
 */
inline void simd_gated_update(const float* gate, const float* current,
                              const float* update, float* out, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= size; i += 16) {
        __m512 g = _mm512_loadu_ps(&gate[i]);
        __m512 c = _mm512_loadu_ps(&current[i]);
        __m512 u = _mm512_loadu_ps(&update[i]);
        __m512 one_minus_g = _mm512_sub_ps(one, g);
        // out = g * u + (1-g) * c
        __m512 result = _mm512_fmadd_ps(g, u, _mm512_mul_ps(one_minus_g, c));
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(&gate[i]);
        __m256 c = _mm256_loadu_ps(&current[i]);
        __m256 u = _mm256_loadu_ps(&update[i]);
        __m256 one_minus_g = _mm256_sub_ps(one, g);
        __m256 result = _mm256_fmadd_ps(g, u, _mm256_mul_ps(one_minus_g, c));
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t g = vld1q_f32(&gate[i]);
        float32x4_t c = vld1q_f32(&current[i]);
        float32x4_t u = vld1q_f32(&update[i]);
        float32x4_t one_minus_g = vsubq_f32(one, g);
        // out = g * u + (1-g) * c
        float32x4_t result = vmlaq_f32(vmulq_f32(one_minus_g, c), g, u);
        vst1q_f32(&out[i], result);
    }
#endif
    for (; i < size; ++i) {
        out[i] = gate[i] * update[i] + (1.0f - gate[i]) * current[i];
    }
}

/**
 * @brief Vectorized EMA (Exponential Moving Average) blend.
 *
 * out[i] = alpha * new_val[i] + (1 - alpha) * old_val[i]
 *
 * @param old_val Previous values [size]
 * @param new_val New values [size]
 * @param alpha Blend factor (0 = all old, 1 = all new)
 * @param out Output array [size] (can alias old_val for in-place)
 * @param size Number of elements
 */
inline void simd_ema_blend(const float* old_val, const float* new_val,
                           float alpha, float* out, int64_t size) {
    float one_minus_alpha = 1.0f - alpha;
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 alpha_vec = _mm512_set1_ps(alpha);
    const __m512 oma_vec = _mm512_set1_ps(one_minus_alpha);
    for (; i + 16 <= size; i += 16) {
        __m512 old_v = _mm512_loadu_ps(&old_val[i]);
        __m512 new_v = _mm512_loadu_ps(&new_val[i]);
        // out = alpha * new + (1-alpha) * old
        __m512 result = _mm512_fmadd_ps(alpha_vec, new_v, _mm512_mul_ps(oma_vec, old_v));
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 alpha_vec = _mm256_set1_ps(alpha);
    const __m256 oma_vec = _mm256_set1_ps(one_minus_alpha);
    for (; i + 8 <= size; i += 8) {
        __m256 old_v = _mm256_loadu_ps(&old_val[i]);
        __m256 new_v = _mm256_loadu_ps(&new_val[i]);
        __m256 result = _mm256_fmadd_ps(alpha_vec, new_v, _mm256_mul_ps(oma_vec, old_v));
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t alpha_vec = vdupq_n_f32(alpha);
    const float32x4_t oma_vec = vdupq_n_f32(one_minus_alpha);
    for (; i + 4 <= size; i += 4) {
        float32x4_t old_v = vld1q_f32(&old_val[i]);
        float32x4_t new_v = vld1q_f32(&new_val[i]);
        float32x4_t result = vmlaq_f32(vmulq_f32(oma_vec, old_v), alpha_vec, new_v);
        vst1q_f32(&out[i], result);
    }
#endif
    for (; i < size; ++i) {
        out[i] = alpha * new_val[i] + one_minus_alpha * old_val[i];
    }
}

/**
 * @brief Vectorized RMS (Root Mean Square) normalization.
 *
 * out[i] = x[i] * gamma[i] / sqrt(mean(x^2) + eps)
 *
 * @param x Input array [size]
 * @param gamma Scale parameter [size]
 * @param out Output array [size]
 * @param size Number of elements
 * @param eps Small epsilon for numerical stability
 */
inline void simd_rms_norm(const float* x, const float* gamma, float* out,
                          int64_t size, float eps = 1e-5f) {
    // Compute mean of squares
    float sum_sq = 0.0f;
    int64_t i = 0;
    
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(&x[i]);
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum_sq = _mm_cvtss_f32(sum128);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        acc = vmlaq_f32(acc, v, v);
    }
    sum_sq = vaddvq_f32(acc);
#endif
    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    
    float mean_sq = sum_sq / static_cast<float>(size);
    float inv_rms = 1.0f / std::sqrt(mean_sq + eps);
    
    // Apply normalization with scale
    i = 0;
#if defined(__AVX512F__)
    const __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
    for (; i + 16 <= size; i += 16) {
        __m512 vx = _mm512_loadu_ps(&x[i]);
        __m512 vg = _mm512_loadu_ps(&gamma[i]);
        __m512 result = _mm512_mul_ps(_mm512_mul_ps(vx, inv_rms_vec), vg);
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
    for (; i + 8 <= size; i += 8) {
        __m256 vx = _mm256_loadu_ps(&x[i]);
        __m256 vg = _mm256_loadu_ps(&gamma[i]);
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(vx, inv_rms_vec), vg);
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    for (; i + 4 <= size; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        float32x4_t vg = vld1q_f32(&gamma[i]);
        float32x4_t result = vmulq_f32(vmulq_f32(vx, inv_rms_vec), vg);
        vst1q_f32(&out[i], result);
    }
#endif
    for (; i < size; ++i) {
        out[i] = x[i] * inv_rms * gamma[i];
    }
}

/**
 * @brief Vectorized tanh activation.
 *
 * Uses Padé approximant for accuracy: tanh(x) ≈ x(27 + x²) / (27 + 9x²) for |x| < 4.37
 * Falls back to ±1 for large values.
 *
 * @param data Float array to apply tanh in-place
 * @param size Number of elements
 */
inline void simd_tanh_inplace(float* data, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 c27 = _mm512_set1_ps(27.0f);
    const __m512 c9 = _mm512_set1_ps(9.0f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 neg_one = _mm512_set1_ps(-1.0f);
    const __m512 clamp = _mm512_set1_ps(4.37f);
    
    for (; i + 16 <= size; i += 16) {
        __m512 x = _mm512_loadu_ps(&data[i]);
        __m512 x2 = _mm512_mul_ps(x, x);
        
        // Padé approximant
        __m512 numer = _mm512_mul_ps(x, _mm512_add_ps(c27, x2));
        __m512 denom = _mm512_add_ps(c27, _mm512_mul_ps(c9, x2));
        __m512 result = _mm512_div_ps(numer, denom);
        
        // Clamp to [-1, 1]
        result = _mm512_max_ps(result, neg_one);
        result = _mm512_min_ps(result, one);
        
        _mm512_storeu_ps(&data[i], result);
    }
#elif defined(__AVX2__)
    const __m256 c27 = _mm256_set1_ps(27.0f);
    const __m256 c9 = _mm256_set1_ps(9.0f);
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 neg_one = _mm256_set1_ps(-1.0f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 x = _mm256_loadu_ps(&data[i]);
        __m256 x2 = _mm256_mul_ps(x, x);
        
        __m256 numer = _mm256_mul_ps(x, _mm256_add_ps(c27, x2));
        __m256 denom = _mm256_add_ps(c27, _mm256_mul_ps(c9, x2));
        __m256 result = _mm256_div_ps(numer, denom);
        
        result = _mm256_max_ps(result, neg_one);
        result = _mm256_min_ps(result, one);
        
        _mm256_storeu_ps(&data[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t c27 = vdupq_n_f32(27.0f);
    const float32x4_t c9 = vdupq_n_f32(9.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t x = vld1q_f32(&data[i]);
        float32x4_t x2 = vmulq_f32(x, x);
        
        float32x4_t numer = vmulq_f32(x, vaddq_f32(c27, x2));
        float32x4_t denom = vaddq_f32(c27, vmulq_f32(c9, x2));
        float32x4_t result = vdivq_f32(numer, denom);
        
        result = vmaxq_f32(result, neg_one);
        result = vminq_f32(result, one);
        
        vst1q_f32(&data[i], result);
    }
#endif
    for (; i < size; ++i) {
        data[i] = std::tanh(data[i]);
    }
}

}  // namespace simd
}  // namespace hsmn

#endif  // SAGUARO_NATIVE_OPS_HNN_SIMD_COMMON_H_
