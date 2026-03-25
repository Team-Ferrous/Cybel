// highnoon/_native/ops/fused_parallel_scan_op.h
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
 * @file fused_parallel_scan_op.h
 * @brief SIMD-optimized parallel prefix scan primitives for MinGRU.
 *
 * Implements the Blelloch parallel scan algorithm for O(n) work, O(log n) depth
 * computation of cumulative operations. This enables 10-175x training speedup
 * for MinGRU by parallelizing the recurrence computation.
 *
 * Key Operations:
 * - Log-space arithmetic for numerical stability
 * - Up-sweep (reduce) phase: tree reduction
 * - Down-sweep (propagate) phase: cumulative propagation
 * - Chunked processing for memory efficiency
 *
 * SIMD Support:
 * - AVX-512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - ARM NEON: 4-wide vectorization
 * - Scalar fallback
 *
 * Reference: Blelloch, G. E. (1990). Prefix Sums and Their Applications
 * Reference: minGRU (Bengio et al. 2024)
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_PARALLEL_SCAN_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_PARALLEL_SCAN_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr float kLogEpsilon = 1e-10f;  // Epsilon for log stability
constexpr float kLogClampMin = -88.0f; // Clamp to prevent underflow
constexpr float kLogClampMax = 88.0f;  // Clamp to prevent overflow

// =============================================================================
// LOG-SPACE ARITHMETIC
// For minGRU: a_t = log(1 - g_t), b_t = log(g_t * c_t)
// Using log-space prevents numerical overflow in cumulative products
// =============================================================================

/**
 * @brief Compute log(1 - x) safely with Taylor expansion for small x.
 * 
 * For x close to 0, log(1-x) ≈ -x - x²/2 - x³/3 - x⁴/4 - x⁵/5
 * For larger x, use standard log(1-x) with clamping.
 *
 * @param x Input array (gate values, 0 < x < 1)
 * @param out Output array for log(1-x)
 * @param size Number of elements
 */
inline void simd_log1p_neg_safe(const float* x, float* out, int64_t size) {
    int64_t i = 0;

#if defined(__AVX512F__)
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 eps = _mm512_set1_ps(kLogEpsilon);
    const __m512 clamp_min = _mm512_set1_ps(kLogClampMin);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 third = _mm512_set1_ps(0.33333333f);
    const __m512 quarter = _mm512_set1_ps(0.25f);
    const __m512 fifth = _mm512_set1_ps(0.2f);
    const __m512 threshold = _mm512_set1_ps(0.1f);
    
    for (; i + 16 <= size; i += 16) {
        __m512 xv = _mm512_loadu_ps(&x[i]);
        
        // For small x, use Taylor: log(1-x) ≈ -x - x²/2 - x³/3 - x⁴/4 - x⁵/5
        __m512 x2 = _mm512_mul_ps(xv, xv);
        __m512 x3 = _mm512_mul_ps(x2, xv);
        __m512 x4 = _mm512_mul_ps(x2, x2);
        __m512 x5 = _mm512_mul_ps(x4, xv);
        
        __m512 taylor = _mm512_mul_ps(xv, _mm512_set1_ps(-1.0f));
        taylor = _mm512_sub_ps(taylor, _mm512_mul_ps(x2, half));
        taylor = _mm512_sub_ps(taylor, _mm512_mul_ps(x3, third));
        taylor = _mm512_sub_ps(taylor, _mm512_mul_ps(x4, quarter));
        taylor = _mm512_sub_ps(taylor, _mm512_mul_ps(x5, fifth));
        
        // For larger x, use log(1-x) directly
        __m512 one_minus_x = _mm512_sub_ps(one, xv);
        one_minus_x = _mm512_max_ps(one_minus_x, eps);  // Prevent log(0)
        
        // Use intrinsic log approximation (AVX512 has svml support on many systems)
        // Fallback: use range reduction
        __m512 log_val = _mm512_set1_ps(0.0f);
        float temp[16];
        _mm512_storeu_ps(temp, one_minus_x);
        for (int j = 0; j < 16; ++j) {
            temp[j] = std::log(temp[j]);
        }
        log_val = _mm512_loadu_ps(temp);
        
        // Blend based on x magnitude
        __mmask16 small_mask = _mm512_cmp_ps_mask(xv, threshold, _CMP_LT_OS);
        __m512 result = _mm512_mask_blend_ps(small_mask, log_val, taylor);
        result = _mm512_max_ps(result, clamp_min);
        
        _mm512_storeu_ps(&out[i], result);
    }
#elif defined(__AVX2__)
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 eps = _mm256_set1_ps(kLogEpsilon);
    const __m256 clamp_min = _mm256_set1_ps(kLogClampMin);
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 third = _mm256_set1_ps(0.33333333f);
    const __m256 quarter = _mm256_set1_ps(0.25f);
    const __m256 fifth = _mm256_set1_ps(0.2f);
    const __m256 threshold = _mm256_set1_ps(0.1f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 xv = _mm256_loadu_ps(&x[i]);
        
        // Taylor expansion for small x
        __m256 x2 = _mm256_mul_ps(xv, xv);
        __m256 x3 = _mm256_mul_ps(x2, xv);
        __m256 x4 = _mm256_mul_ps(x2, x2);
        __m256 x5 = _mm256_mul_ps(x4, xv);
        
        __m256 taylor = _mm256_mul_ps(xv, _mm256_set1_ps(-1.0f));
        taylor = _mm256_sub_ps(taylor, _mm256_mul_ps(x2, half));
        taylor = _mm256_sub_ps(taylor, _mm256_mul_ps(x3, third));
        taylor = _mm256_sub_ps(taylor, _mm256_mul_ps(x4, quarter));
        taylor = _mm256_sub_ps(taylor, _mm256_mul_ps(x5, fifth));
        
        // Standard log for larger x
        __m256 one_minus_x = _mm256_sub_ps(one, xv);
        one_minus_x = _mm256_max_ps(one_minus_x, eps);
        
        float temp[8];
        _mm256_storeu_ps(temp, one_minus_x);
        for (int j = 0; j < 8; ++j) {
            temp[j] = std::log(temp[j]);
        }
        __m256 log_val = _mm256_loadu_ps(temp);
        
        // Blend based on threshold
        __m256 cmp = _mm256_cmp_ps(xv, threshold, _CMP_LT_OS);
        __m256 result = _mm256_blendv_ps(log_val, taylor, cmp);
        result = _mm256_max_ps(result, clamp_min);
        
        _mm256_storeu_ps(&out[i], result);
    }
#elif defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t eps = vdupq_n_f32(kLogEpsilon);
    const float32x4_t clamp_min = vdupq_n_f32(kLogClampMin);
    
    for (; i + 4 <= size; i += 4) {
        float32x4_t xv = vld1q_f32(&x[i]);
        float32x4_t one_minus_x = vsubq_f32(one, xv);
        one_minus_x = vmaxq_f32(one_minus_x, eps);
        
        float temp[4];
        vst1q_f32(temp, one_minus_x);
        for (int j = 0; j < 4; ++j) {
            temp[j] = std::log(temp[j]);
        }
        float32x4_t result = vld1q_f32(temp);
        result = vmaxq_f32(result, clamp_min);
        
        vst1q_f32(&out[i], result);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        float val = 1.0f - x[i];
        val = std::max(val, kLogEpsilon);
        out[i] = std::max(std::log(val), kLogClampMin);
    }
}

/**
 * @brief Compute log(x * y) = log(x) + log(y) safely.
 * 
 * Used for computing log(g * c) where g is gate and c is candidate.
 *
 * @param g Gate values (0 < g < 1)
 * @param c Candidate values
 * @param out Output array for log(g * c)
 * @param size Number of elements
 */
inline void simd_log_product_safe(const float* g, const float* c, float* out, int64_t size) {
    int64_t i = 0;
    
#if defined(__AVX512F__)
    const __m512 eps = _mm512_set1_ps(kLogEpsilon);
    const __m512 clamp_min = _mm512_set1_ps(kLogClampMin);
    const __m512 clamp_max = _mm512_set1_ps(kLogClampMax);
    
    for (; i + 16 <= size; i += 16) {
        __m512 gv = _mm512_loadu_ps(&g[i]);
        __m512 cv = _mm512_loadu_ps(&c[i]);
        
        // Compute |g * c| for log
        __m512 product = _mm512_mul_ps(gv, cv);
        __m512 abs_product = _mm512_abs_ps(product);
        abs_product = _mm512_max_ps(abs_product, eps);
        
        // Compute log
        float temp[16];
        _mm512_storeu_ps(temp, abs_product);
        for (int j = 0; j < 16; ++j) {
            temp[j] = std::log(temp[j]);
        }
        __m512 log_val = _mm512_loadu_ps(temp);
        
        // Clamp
        log_val = _mm512_max_ps(log_val, clamp_min);
        log_val = _mm512_min_ps(log_val, clamp_max);
        
        _mm512_storeu_ps(&out[i], log_val);
    }
#elif defined(__AVX2__)
    const __m256 eps = _mm256_set1_ps(kLogEpsilon);
    const __m256 clamp_min = _mm256_set1_ps(kLogClampMin);
    const __m256 clamp_max = _mm256_set1_ps(kLogClampMax);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    
    for (; i + 8 <= size; i += 8) {
        __m256 gv = _mm256_loadu_ps(&g[i]);
        __m256 cv = _mm256_loadu_ps(&c[i]);
        
        __m256 product = _mm256_mul_ps(gv, cv);
        __m256 abs_product = _mm256_and_ps(product, sign_mask);
        abs_product = _mm256_max_ps(abs_product, eps);
        
        float temp[8];
        _mm256_storeu_ps(temp, abs_product);
        for (int j = 0; j < 8; ++j) {
            temp[j] = std::log(temp[j]);
        }
        __m256 log_val = _mm256_loadu_ps(temp);
        
        log_val = _mm256_max_ps(log_val, clamp_min);
        log_val = _mm256_min_ps(log_val, clamp_max);
        
        _mm256_storeu_ps(&out[i], log_val);
    }
#elif defined(__ARM_NEON)
    for (; i + 4 <= size; i += 4) {
        float32x4_t gv = vld1q_f32(&g[i]);
        float32x4_t cv = vld1q_f32(&c[i]);
        
        float32x4_t product = vmulq_f32(gv, cv);
        float32x4_t abs_product = vabsq_f32(product);
        abs_product = vmaxq_f32(abs_product, vdupq_n_f32(kLogEpsilon));
        
        float temp[4];
        vst1q_f32(temp, abs_product);
        for (int j = 0; j < 4; ++j) {
            temp[j] = std::log(temp[j]);
        }
        float32x4_t log_val = vld1q_f32(temp);
        
        log_val = vmaxq_f32(log_val, vdupq_n_f32(kLogClampMin));
        log_val = vminq_f32(log_val, vdupq_n_f32(kLogClampMax));
        
        vst1q_f32(&out[i], log_val);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        float product = std::abs(g[i] * c[i]);
        product = std::max(product, kLogEpsilon);
        float log_val = std::log(product);
        out[i] = std::max(kLogClampMin, std::min(kLogClampMax, log_val));
    }
}

// =============================================================================
// BLELLOCH PARALLEL PREFIX SCAN
// Two-phase algorithm: Up-sweep (reduce) + Down-sweep (propagate)
// For minGRU: computes cumulative sum in log-space
// =============================================================================

/**
 * @brief Associative binary operator for minGRU scan in log-space.
 * 
 * For minGRU recurrence h_t = g_t * h_{t-1} + (1-g_t) * c_t
 * In log-space with a = log(1-g), b = log(g*c):
 * 
 * Combine (a1, b1) ⊕ (a2, b2) = (a1 + a2, log(exp(b1 + a2) + exp(b2)))
 *                              = (a1 + a2, log_sum_exp(b1 + a2, b2))
 *
 * @param a1 First element's log-decay
 * @param b1 First element's log-contribution
 * @param a2 Second element's log-decay
 * @param b2 Second element's log-contribution
 * @param a_out Combined log-decay
 * @param b_out Combined log-contribution
 */
inline void scan_binary_op(float a1, float b1, float a2, float b2, 
                           float* a_out, float* b_out) {
    *a_out = a1 + a2;
    
    // log_sum_exp(b1 + a2, b2) for numerical stability
    float x = b1 + a2;
    float y = b2;
    float max_val = std::max(x, y);
    *b_out = max_val + std::log(std::exp(x - max_val) + std::exp(y - max_val));
}

/**
 * @brief SIMD binary operator for batch processing.
 */
inline void simd_scan_binary_op(
    const float* a1, const float* b1,
    const float* a2, const float* b2,
    float* a_out, float* b_out,
    int64_t size) {
    
    int64_t i = 0;
    
#if defined(__AVX2__)
    for (; i + 8 <= size; i += 8) {
        __m256 va1 = _mm256_loadu_ps(&a1[i]);
        __m256 vb1 = _mm256_loadu_ps(&b1[i]);
        __m256 va2 = _mm256_loadu_ps(&a2[i]);
        __m256 vb2 = _mm256_loadu_ps(&b2[i]);
        
        // a_out = a1 + a2
        __m256 va_out = _mm256_add_ps(va1, va2);
        _mm256_storeu_ps(&a_out[i], va_out);
        
        // log_sum_exp(b1 + a2, b2)
        __m256 x = _mm256_add_ps(vb1, va2);
        __m256 max_val = _mm256_max_ps(x, vb2);
        
        // exp(x - max), exp(b2 - max)
        __m256 x_shifted = _mm256_sub_ps(x, max_val);
        __m256 b2_shifted = _mm256_sub_ps(vb2, max_val);
        
        float x_arr[8], b2_arr[8], max_arr[8];
        _mm256_storeu_ps(x_arr, x_shifted);
        _mm256_storeu_ps(b2_arr, b2_shifted);
        _mm256_storeu_ps(max_arr, max_val);
        
        for (int j = 0; j < 8; ++j) {
            float sum = std::exp(x_arr[j]) + std::exp(b2_arr[j]);
            b2_arr[j] = max_arr[j] + std::log(sum);
        }
        
        __m256 vb_out = _mm256_loadu_ps(b2_arr);
        _mm256_storeu_ps(&b_out[i], vb_out);
    }
#endif
    // Scalar fallback
    for (; i < size; ++i) {
        scan_binary_op(a1[i], b1[i], a2[i], b2[i], &a_out[i], &b_out[i]);
    }
}

/**
 * @brief Sequential scan for a single sequence (baseline/fallback).
 * 
 * Processes one sequence of length L for hidden dimension H.
 * Output: h[t] = exp(cumsum_a[t]) * h0 + exp(cumsum_b[t])
 *
 * @param log_decay log(1-g) array [L, H]
 * @param log_contrib log(g*c) array [L, H]
 * @param initial_h Initial hidden state [H] (or nullptr for zeros)
 * @param output Output hidden states [L, H]
 * @param L Sequence length
 * @param H Hidden dimension
 */
inline void sequential_scan(
    const float* log_decay,    // [L, H]
    const float* log_contrib,  // [L, H]
    const float* initial_h,    // [H] or nullptr
    float* output,             // [L, H]
    int L, int H) {
    
    std::vector<float> cumsum_a(H, 0.0f);  // Cumulative log-decay
    std::vector<float> log_h(H);           // log(h) in log-space
    
    // Initialize log_h from initial_h
    if (initial_h) {
        for (int i = 0; i < H; ++i) {
            float h0 = std::abs(initial_h[i]);
            log_h[i] = (h0 > kLogEpsilon) ? std::log(h0) : kLogClampMin;
        }
    } else {
        std::fill(log_h.begin(), log_h.end(), kLogClampMin);
    }
    
    for (int t = 0; t < L; ++t) {
        const float* a_t = log_decay + t * H;
        const float* b_t = log_contrib + t * H;
        float* h_t = output + t * H;
        
        for (int i = 0; i < H; ++i) {
            cumsum_a[i] += a_t[i];
            
            // h[t] = (1-g[t]) * h[t-1] + g[t] * c[t]
            // In log-space: log(h[t]) = log_sum_exp(log_h + a_t, b_t)
            float term1 = log_h[i] + a_t[i];  // log((1-g) * h_prev)
            float term2 = b_t[i];              // log(g * c)
            
            float max_val = std::max(term1, term2);
            float new_log_h = max_val + std::log(
                std::exp(term1 - max_val) + std::exp(term2 - max_val));
            
            log_h[i] = new_log_h;
            h_t[i] = std::exp(new_log_h);
        }
    }
}

/**
 * @brief Blelloch parallel prefix scan (work-efficient).
 * 
 * Two phases:
 * 1. Up-sweep (reduce): Build partial sums in tree structure
 * 2. Down-sweep (propagate): Distribute results back
 *
 * @param log_decay log(1-g) array [L, H]
 * @param log_contrib log(g*c) array [L, H]  
 * @param initial_h Initial hidden state [H] or nullptr
 * @param output Output hidden states [L, H]
 * @param L Sequence length (should be power of 2, padded if needed)
 * @param H Hidden dimension
 */
inline void parallel_blelloch_scan(
    const float* log_decay,
    const float* log_contrib,
    const float* initial_h,
    float* output,
    int L, int H) {
    
    // For short sequences, use sequential scan
    if (L <= 16) {
        sequential_scan(log_decay, log_contrib, initial_h, output, L, H);
        return;
    }
    
    // Allocate workspace for tree operations
    // Each level needs L/2^level elements
    std::vector<float> tree_a(L * H);
    std::vector<float> tree_b(L * H);
    
    // Initialize leaves with input data
    std::copy(log_decay, log_decay + L * H, tree_a.begin());
    std::copy(log_contrib, log_contrib + L * H, tree_b.begin());
    
    // Compute number of levels
    int levels = 0;
    int temp = L;
    while (temp > 1) {
        temp >>= 1;
        levels++;
    }
    
    // === UP-SWEEP (Reduce) ===
    // Combine pairs at each level
    for (int level = 0; level < levels; ++level) {
        int stride = 1 << (level + 1);
        int num_ops = L >> (level + 1);
        
        #pragma omp parallel for if(num_ops > 4)
        for (int j = 0; j < num_ops; ++j) {
            int left_idx = j * stride + (1 << level) - 1;
            int right_idx = j * stride + stride - 1;
            
            // Combine (tree[left], tree[right]) -> tree[right]
            for (int h = 0; h < H; ++h) {
                float a1 = tree_a[left_idx * H + h];
                float b1 = tree_b[left_idx * H + h];
                float a2 = tree_a[right_idx * H + h];
                float b2 = tree_b[right_idx * H + h];
                
                scan_binary_op(a1, b1, a2, b2,
                              &tree_a[right_idx * H + h],
                              &tree_b[right_idx * H + h]);
            }
        }
    }
    
    // Clear last element (identity for down-sweep)
    for (int h = 0; h < H; ++h) {
        tree_a[(L - 1) * H + h] = 0.0f;
        tree_b[(L - 1) * H + h] = kLogClampMin;
    }
    
    // === DOWN-SWEEP (Propagate) ===
    for (int level = levels - 1; level >= 0; --level) {
        int stride = 1 << (level + 1);
        int num_ops = L >> (level + 1);
        
        #pragma omp parallel for if(num_ops > 4)
        for (int j = 0; j < num_ops; ++j) {
            int left_idx = j * stride + (1 << level) - 1;
            int right_idx = j * stride + stride - 1;
            
            for (int h = 0; h < H; ++h) {
                // Save right value
                float old_right_a = tree_a[right_idx * H + h];
                float old_right_b = tree_b[right_idx * H + h];
                
                // Right = combine(left, right)
                float a1 = tree_a[left_idx * H + h];
                float b1 = tree_b[left_idx * H + h];
                
                scan_binary_op(a1, b1, old_right_a, old_right_b,
                              &tree_a[right_idx * H + h],
                              &tree_b[right_idx * H + h]);
                
                // Left = old right
                tree_a[left_idx * H + h] = old_right_a;
                tree_b[left_idx * H + h] = old_right_b;
            }
        }
    }
    
    // === RECONSTRUCT OUTPUT ===
    // h[t] = exp(cumsum) applied to initial_h
    std::vector<float> log_h0(H);
    if (initial_h) {
        for (int h = 0; h < H; ++h) {
            float val = std::abs(initial_h[h]);
            log_h0[h] = (val > kLogEpsilon) ? std::log(val) : kLogClampMin;
        }
    } else {
        std::fill(log_h0.begin(), log_h0.end(), kLogClampMin);
    }
    
    #pragma omp parallel for
    for (int t = 0; t < L; ++t) {
        for (int h = 0; h < H; ++h) {
            float cumsum_a = tree_a[t * H + h];
            float cumsum_b = tree_b[t * H + h];
            
            // h[t] = h0 * exp(cumsum_a) + contribution from cumsum_b
            float term1 = log_h0[h] + cumsum_a;
            float term2 = cumsum_b;
            
            float max_val = std::max(term1, term2);
            float log_ht = max_val + std::log(
                std::exp(term1 - max_val) + std::exp(term2 - max_val));
            
            output[t * H + h] = std::exp(log_ht);
        }
    }
}

/**
 * @brief Chunked parallel scan for memory efficiency.
 * 
 * Processes sequence in chunks, maintaining state between chunks.
 *
 * @param log_decay log(1-g) array [L, H]
 * @param log_contrib log(g*c) array [L, H]
 * @param initial_h Initial hidden state [H] or nullptr
 * @param output Output hidden states [L, H]
 * @param L Sequence length
 * @param H Hidden dimension
 * @param chunk_size Chunk size for processing
 */
inline void chunked_parallel_scan(
    const float* log_decay,
    const float* log_contrib,
    const float* initial_h,
    float* output,
    int L, int H, int chunk_size) {
    
    std::vector<float> current_h(H);
    if (initial_h) {
        std::copy(initial_h, initial_h + H, current_h.begin());
    } else {
        std::fill(current_h.begin(), current_h.end(), 0.0f);
    }
    
    for (int chunk_start = 0; chunk_start < L; chunk_start += chunk_size) {
        int chunk_len = std::min(chunk_size, L - chunk_start);
        
        // Pad to power of 2 for Blelloch scan
        int padded_len = 1;
        while (padded_len < chunk_len) padded_len <<= 1;
        
        std::vector<float> padded_decay(padded_len * H, 0.0f);
        std::vector<float> padded_contrib(padded_len * H, kLogClampMin);
        std::vector<float> chunk_output(padded_len * H);
        
        // Copy chunk data
        std::copy(log_decay + chunk_start * H,
                  log_decay + (chunk_start + chunk_len) * H,
                  padded_decay.begin());
        std::copy(log_contrib + chunk_start * H,
                  log_contrib + (chunk_start + chunk_len) * H,
                  padded_contrib.begin());
        
        // Run scan on padded chunk
        parallel_blelloch_scan(
            padded_decay.data(),
            padded_contrib.data(),
            current_h.data(),
            chunk_output.data(),
            padded_len, H);
        
        // Copy valid results to output
        std::copy(chunk_output.begin(),
                  chunk_output.begin() + chunk_len * H,
                  output + chunk_start * H);
        
        // Update current_h for next chunk
        std::copy(output + (chunk_start + chunk_len - 1) * H,
                  output + (chunk_start + chunk_len) * H,
                  current_h.begin());
    }
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_PARALLEL_SCAN_OP_H_
