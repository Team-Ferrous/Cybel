// highnoon/_native/ops/simd_circular_conv.h
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
 * @file simd_circular_conv.h
 * @brief PHASE V2.0-P2.2: SIMD-Optimized Circular Convolution
 *
 * Circular convolution for hyperdimensional computing operations:
 *   c[k] = Σⱼ a[j] × b[(k-j) mod D]
 *
 * Used in:
 *   - HD binding operations
 *   - Holographic routing correlation
 *   - Quantum holographic memory
 *
 * Provides AVX2, AVX-512, and ARM NEON implementations.
 * AVX2 is the primary target (most compatible).
 *
 * Performance: 2-3× speedup vs naive O(D²), complementing FFT for small D.
 *
 * Reference: HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 6.2
 */

#ifndef HIGHNOON_NATIVE_OPS_SIMD_CIRCULAR_CONV_H_
#define HIGHNOON_NATIVE_OPS_SIMD_CIRCULAR_CONV_H_

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#include "hnn_simd_common.h"

namespace highnoon {
namespace ops {
namespace circular {

// =============================================================================
// AVX2 CIRCULAR CONVOLUTION (Primary)
// =============================================================================

#if defined(__AVX2__)
/**
 * @brief Circular convolution using AVX2 intrinsics.
 *
 * Computes: c[k] = Σⱼ a[j] × b[(k-j) mod hd_dim]
 *
 * For small hd_dim (< 256), direct convolution may be faster than FFT.
 * For large hd_dim, consider using FFT-based convolution instead.
 *
 * @param a First input vector [hd_dim]
 * @param b Second input vector [hd_dim]
 * @param c Output vector [hd_dim]
 * @param workspace Scratch space [hd_dim] for rotated b values
 * @param hd_dim Hyperdimensional dimension
 */
inline void simd_circular_conv_avx2(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
    // Zero output
    std::memset(c, 0, hd_dim * sizeof(float));
    
    // For each output position k
    for (int k = 0; k < hd_dim; ++k) {
        float sum = 0.0f;
        __m256 acc = _mm256_setzero_ps();
        
        // Process in chunks of 8
        int j = 0;
        for (; j + 8 <= hd_dim; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(&a[j]);
            
            // Load b[(k-j) mod hd_dim] - need to gather
            float b_vals[8];
            for (int i = 0; i < 8; ++i) {
                int idx = (k - (j + i) % hd_dim + hd_dim) % hd_dim;
                b_vals[i] = b[idx];
            }
            __m256 b_vec = _mm256_loadu_ps(b_vals);
            
            acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
        }
        
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 s = _mm_add_ps(hi, lo);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum = _mm_cvtss_f32(s);
        
        // Scalar remainder
        for (; j < hd_dim; ++j) {
            int idx = (k - j % hd_dim + hd_dim) % hd_dim;
            sum += a[j] * b[idx];
        }
        
        c[k] = sum;
    }
}
#endif  // __AVX2__

// =============================================================================
// AVX-512 CIRCULAR CONVOLUTION (Secondary)
// =============================================================================

#if defined(__AVX512F__)
inline void simd_circular_conv_avx512(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
    // Zero output
    std::memset(c, 0, hd_dim * sizeof(float));
    
    for (int k = 0; k < hd_dim; ++k) {
        float sum = 0.0f;
        __m512 acc = _mm512_setzero_ps();
        
        int j = 0;
        for (; j + 16 <= hd_dim; j += 16) {
            __m512 a_vec = _mm512_loadu_ps(&a[j]);
            
            // Gather b values
            float b_vals[16];
            for (int i = 0; i < 16; ++i) {
                int idx = (k - (j + i) % hd_dim + hd_dim) % hd_dim;
                b_vals[i] = b[idx];
            }
            __m512 b_vec = _mm512_loadu_ps(b_vals);
            
            acc = _mm512_fmadd_ps(a_vec, b_vec, acc);
        }
        
        // Horizontal sum
        sum = _mm512_reduce_add_ps(acc);
        
        // Scalar remainder
        for (; j < hd_dim; ++j) {
            int idx = (k - j % hd_dim + hd_dim) % hd_dim;
            sum += a[j] * b[idx];
        }
        
        c[k] = sum;
    }
}
#endif  // __AVX512F__

// =============================================================================
// ARM NEON CIRCULAR CONVOLUTION (Tertiary)
// =============================================================================

#if defined(__ARM_NEON)
inline void simd_circular_conv_neon(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
    std::memset(c, 0, hd_dim * sizeof(float));
    
    for (int k = 0; k < hd_dim; ++k) {
        float sum = 0.0f;
        float32x4_t acc = vdupq_n_f32(0.0f);
        
        int j = 0;
        for (; j + 4 <= hd_dim; j += 4) {
            float32x4_t a_vec = vld1q_f32(&a[j]);
            
            float b_vals[4];
            for (int i = 0; i < 4; ++i) {
                int idx = (k - (j + i) % hd_dim + hd_dim) % hd_dim;
                b_vals[i] = b[idx];
            }
            float32x4_t b_vec = vld1q_f32(b_vals);
            
            acc = vmlaq_f32(acc, a_vec, b_vec);
        }
        
        // Horizontal sum
        float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        sum2 = vpadd_f32(sum2, sum2);
        sum = vget_lane_f32(sum2, 0);
        
        for (; j < hd_dim; ++j) {
            int idx = (k - j % hd_dim + hd_dim) % hd_dim;
            sum += a[j] * b[idx];
        }
        
        c[k] = sum;
    }
}
#endif  // __ARM_NEON

// =============================================================================
// SCALAR FALLBACK
// =============================================================================

inline void simd_circular_conv_scalar(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
    for (int k = 0; k < hd_dim; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < hd_dim; ++j) {
            int idx = (k - j % hd_dim + hd_dim) % hd_dim;
            sum += a[j] * b[idx];
        }
        c[k] = sum;
    }
}

// =============================================================================
// DISPATCH FUNCTION
// =============================================================================

/**
 * @brief Auto-dispatch circular convolution to best available SIMD.
 */
inline void simd_circular_conv(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
#if defined(__AVX512F__)
    simd_circular_conv_avx512(a, b, c, workspace, hd_dim);
#elif defined(__AVX2__)
    simd_circular_conv_avx2(a, b, c, workspace, hd_dim);
#elif defined(__ARM_NEON)
    simd_circular_conv_neon(a, b, c, workspace, hd_dim);
#else
    simd_circular_conv_scalar(a, b, c, workspace, hd_dim);
#endif
}

// =============================================================================
// CIRCULAR CORRELATION (Inverse operation)
// =============================================================================

/**
 * @brief Circular correlation: c[k] = Σⱼ a[j] × b[(j+k) mod D]
 *
 * Used for:
 *   - HD unbinding: a ⊛ b⁻¹ where b⁻¹[i] = b[-i mod D]
 *   - Holographic retrieval
 */
inline void simd_circular_corr(
    const float* a, const float* b, float* c, float* workspace, int hd_dim) {
    
    // Correlation is convolution with reversed b
    // b_rev[i] = b[(hd_dim - i) mod hd_dim]
    for (int i = 0; i < hd_dim; ++i) {
        workspace[i] = b[(hd_dim - i) % hd_dim];
    }
    
    simd_circular_conv(a, workspace, c, nullptr, hd_dim);
}

}  // namespace circular
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_SIMD_CIRCULAR_CONV_H_
