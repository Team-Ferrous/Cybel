// highnoon/_native/ops/quantum_kernel_embed_op.h
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
 * @file quantum_kernel_embed_op.h
 * @brief SIMD-optimized quantum kernel embedding kernels.
 *
 * Implements RBF, polynomial, and linear kernels with:
 * - AVX-512 for x86_64 with AVX-512
 * - AVX2/FMA for x86_64 with AVX2
 * - ARM NEON for ARM64/aarch64
 *
 * Complexity: O(B * M * D) where B=batch, M=num_support, D=input_dim
 * All implementations are LINEAR in batch and input dimensions.
 */

#ifndef HIGHNOON_QUANTUM_KERNEL_EMBED_OP_H_
#define HIGHNOON_QUANTUM_KERNEL_EMBED_OP_H_

#include <cmath>
#include <cstring>
#include <algorithm>

// SIMD intrinsics
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define HIGHNOON_USE_AVX512
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define HIGHNOON_USE_AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define HIGHNOON_USE_NEON
#endif

namespace highnoon {
namespace ops {

// =============================================================================
// RBF Kernel: K(x, y) = exp(-gamma * ||x - y||²)
// =============================================================================

/**
 * Compute pairwise squared Euclidean distances with SIMD.
 * dist[i,j] = ||x[i] - support[j]||²
 */
inline void compute_squared_distances_simd(
    const float* x,           // [batch, input_dim]
    const float* support,     // [num_support, input_dim]
    float* distances,         // [batch, num_support] output
    int batch_size,
    int num_support,
    int input_dim
) {
    #ifdef HIGHNOON_USE_AVX512
    // AVX-512 implementation: 16 floats per iteration
    const int simd_width = 16;
    const int vec_iters = input_dim / simd_width;
    const int remainder = input_dim % simd_width;

    for (int b = 0; b < batch_size; ++b) {
        const float* x_row = x + b * input_dim;

        for (int m = 0; m < num_support; ++m) {
            const float* support_row = support + m * input_dim;

            __m512 sum_vec = _mm512_setzero_ps();

            // Vectorized main loop
            for (int i = 0; i < vec_iters; ++i) {
                __m512 x_vec = _mm512_loadu_ps(x_row + i * simd_width);
                __m512 s_vec = _mm512_loadu_ps(support_row + i * simd_width);
                __m512 diff = _mm512_sub_ps(x_vec, s_vec);
                sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
            }

            // Horizontal sum
            float sum = _mm512_reduce_add_ps(sum_vec);

            // Handle remainder
            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                float diff = x_row[i] - support_row[i];
                sum += diff * diff;
            }

            distances[b * num_support + m] = sum;
        }
    }

    #elif defined(HIGHNOON_USE_AVX2)
    // AVX2 implementation: 8 floats per iteration
    const int simd_width = 8;
    const int vec_iters = input_dim / simd_width;
    const int remainder = input_dim % simd_width;

    for (int b = 0; b < batch_size; ++b) {
        const float* x_row = x + b * input_dim;

        for (int m = 0; m < num_support; ++m) {
            const float* support_row = support + m * input_dim;

            __m256 sum_vec = _mm256_setzero_ps();

            // Vectorized main loop
            for (int i = 0; i < vec_iters; ++i) {
                __m256 x_vec = _mm256_loadu_ps(x_row + i * simd_width);
                __m256 s_vec = _mm256_loadu_ps(support_row + i * simd_width);
                __m256 diff = _mm256_sub_ps(x_vec, s_vec);
                sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
            }

            // Horizontal sum
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            // Handle remainder
            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                float diff = x_row[i] - support_row[i];
                sum += diff * diff;
            }

            distances[b * num_support + m] = sum;
        }
    }

    #elif defined(HIGHNOON_USE_NEON)
    // ARM NEON implementation: 4 floats per iteration
    const int simd_width = 4;
    const int vec_iters = input_dim / simd_width;
    const int remainder = input_dim % simd_width;

    for (int b = 0; b < batch_size; ++b) {
        const float* x_row = x + b * input_dim;

        for (int m = 0; m < num_support; ++m) {
            const float* support_row = support + m * input_dim;

            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            // Vectorized main loop
            for (int i = 0; i < vec_iters; ++i) {
                float32x4_t x_vec = vld1q_f32(x_row + i * simd_width);
                float32x4_t s_vec = vld1q_f32(support_row + i * simd_width);
                float32x4_t diff = vsubq_f32(x_vec, s_vec);
                sum_vec = vfmaq_f32(sum_vec, diff, diff);
            }

            // Horizontal sum
            float32x2_t sum_low = vget_low_f32(sum_vec);
            float32x2_t sum_high = vget_high_f32(sum_vec);
            float32x2_t sum_pair = vpadd_f32(sum_low, sum_high);
            float sum = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

            // Handle remainder
            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                float diff = x_row[i] - support_row[i];
                sum += diff * diff;
            }

            distances[b * num_support + m] = sum;
        }
    }

    #else
    // Scalar fallback
    for (int b = 0; b < batch_size; ++b) {
        const float* x_row = x + b * input_dim;

        for (int m = 0; m < num_support; ++m) {
            const float* support_row = support + m * input_dim;

            float sum = 0.0f;
            for (int d = 0; d < input_dim; ++d) {
                float diff = x_row[d] - support_row[d];
                sum += diff * diff;
            }

            distances[b * num_support + m] = sum;
        }
    }
    #endif
}

/**
 * RBF kernel: K(x, y) = exp(-gamma * ||x - y||²)
 */
inline void rbf_kernel(
    const float* x,
    const float* support,
    float* output,
    int batch_size,
    int num_support,
    int input_dim,
    float gamma
) {
    // Step 1: Compute squared distances
    compute_squared_distances_simd(x, support, output, batch_size, num_support, input_dim);

    // Step 2: Apply exp(-gamma * dist²) with SIMD
    const int total_size = batch_size * num_support;

    #ifdef HIGHNOON_USE_AVX512
    const int simd_width = 16;
    const int vec_iters = total_size / simd_width;
    const int remainder = total_size % simd_width;

    __m512 gamma_vec = _mm512_set1_ps(-gamma);

    for (int i = 0; i < vec_iters; ++i) {
        __m512 dist_vec = _mm512_loadu_ps(output + i * simd_width);
        __m512 exp_arg = _mm512_mul_ps(gamma_vec, dist_vec);

        // Fast exp approximation (accurate to 5 decimal places)
        // exp(x) ≈ 2^(x/ln(2)) using AVX-512 exp2
        __m512 scale = _mm512_set1_ps(1.442695f);  // 1/ln(2)
        __m512 scaled = _mm512_mul_ps(exp_arg, scale);
        __m512 result = _mm512_exp2_ps(scaled);

        _mm512_storeu_ps(output + i * simd_width, result);
    }

    for (int i = vec_iters * simd_width; i < total_size; ++i) {
        output[i] = std::exp(-gamma * output[i]);
    }

    #elif defined(HIGHNOON_USE_AVX2)
    const int simd_width = 8;
    const int vec_iters = total_size / simd_width;

    __m256 gamma_vec = _mm256_set1_ps(-gamma);

    for (int i = 0; i < vec_iters; ++i) {
        __m256 dist_vec = _mm256_loadu_ps(output + i * simd_width);
        __m256 exp_arg = _mm256_mul_ps(gamma_vec, dist_vec);

        // Use FMA for better numerical accuracy
        // Store and call scalar exp (AVX2 doesn't have native exp)
        float temp[8];
        _mm256_storeu_ps(temp, exp_arg);
        for (int j = 0; j < 8; ++j) {
            temp[j] = std::exp(temp[j]);
        }
        _mm256_storeu_ps(output + i * simd_width, _mm256_loadu_ps(temp));
    }

    for (int i = vec_iters * simd_width; i < total_size; ++i) {
        output[i] = std::exp(-gamma * output[i]);
    }

    #else
    // Scalar or NEON (no native exp in NEON, use scalar)
    for (int i = 0; i < total_size; ++i) {
        output[i] = std::exp(-gamma * output[i]);
    }
    #endif
}

// =============================================================================
// Polynomial Kernel: K(x, y) = (gamma * <x, y> + coef0)^degree
// =============================================================================

inline void polynomial_kernel(
    const float* x,
    const float* support,
    float* output,
    int batch_size,
    int num_support,
    int input_dim,
    float gamma,
    int degree,
    float coef0
) {
    // Compute dot products: output[b,m] = <x[b], support[m]>
    for (int b = 0; b < batch_size; ++b) {
        const float* x_row = x + b * input_dim;

        for (int m = 0; m < num_support; ++m) {
            const float* support_row = support + m * input_dim;

            #ifdef HIGHNOON_USE_AVX512
            const int simd_width = 16;
            const int vec_iters = input_dim / simd_width;

            __m512 sum_vec = _mm512_setzero_ps();

            for (int i = 0; i < vec_iters; ++i) {
                __m512 x_vec = _mm512_loadu_ps(x_row + i * simd_width);
                __m512 s_vec = _mm512_loadu_ps(support_row + i * simd_width);
                sum_vec = _mm512_fmadd_ps(x_vec, s_vec, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);

            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                sum += x_row[i] * support_row[i];
            }

            #elif defined(HIGHNOON_USE_AVX2)
            const int simd_width = 8;
            const int vec_iters = input_dim / simd_width;

            __m256 sum_vec = _mm256_setzero_ps();

            for (int i = 0; i < vec_iters; ++i) {
                __m256 x_vec = _mm256_loadu_ps(x_row + i * simd_width);
                __m256 s_vec = _mm256_loadu_ps(support_row + i * simd_width);
                sum_vec = _mm256_fmadd_ps(x_vec, s_vec, sum_vec);
            }

            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum128 = _mm_add_ps(low, high);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float sum = _mm_cvtss_f32(sum128);

            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                sum += x_row[i] * support_row[i];
            }

            #elif defined(HIGHNOON_USE_NEON)
            const int simd_width = 4;
            const int vec_iters = input_dim / simd_width;

            float32x4_t sum_vec = vdupq_n_f32(0.0f);

            for (int i = 0; i < vec_iters; ++i) {
                float32x4_t x_vec = vld1q_f32(x_row + i * simd_width);
                float32x4_t s_vec = vld1q_f32(support_row + i * simd_width);
                sum_vec = vfmaq_f32(sum_vec, x_vec, s_vec);
            }

            float32x2_t sum_low = vget_low_f32(sum_vec);
            float32x2_t sum_high = vget_high_f32(sum_vec);
            float32x2_t sum_pair = vpadd_f32(sum_low, sum_high);
            float sum = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);

            for (int i = vec_iters * simd_width; i < input_dim; ++i) {
                sum += x_row[i] * support_row[i];
            }

            #else
            float sum = 0.0f;
            for (int d = 0; d < input_dim; ++d) {
                sum += x_row[d] * support_row[d];
            }
            #endif

            // Apply polynomial: (gamma * dot + coef0)^degree
            float poly_val = gamma * sum + coef0;

            // Fast integer power
            float result = 1.0f;
            for (int p = 0; p < degree; ++p) {
                result *= poly_val;
            }

            output[b * num_support + m] = result;
        }
    }
}

// =============================================================================
// Linear Kernel: K(x, y) = <x, y>
// =============================================================================

inline void linear_kernel(
    const float* x,
    const float* support,
    float* output,
    int batch_size,
    int num_support,
    int input_dim
) {
    // Just compute dot products (same as polynomial with gamma=1, coef0=0, degree=1)
    polynomial_kernel(x, support, output, batch_size, num_support, input_dim, 1.0f, 1, 0.0f);
}

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_QUANTUM_KERNEL_EMBED_OP_H_
