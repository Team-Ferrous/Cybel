// saguaro.native/ops/fused_superposition_slots_op.h
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
 * @file fused_superposition_slots_op.h
 * @brief SIMD-optimized helpers for superposition slot operations.
 *
 * Enhancement 4: Quantum-Inspired Slot Superposition
 * Provides collapse_read and superposition_write kernels.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_SUPERPOSITION_SLOTS_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_SUPERPOSITION_SLOTS_OP_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace saguaro {
namespace ops {

/**
 * @brief Softmax over dimension with temperature scaling.
 *
 * @param data In/out: logits to apply softmax
 * @param size Number of elements
 * @param temperature Softmax temperature (lower = sharper)
 */
inline void superposition_softmax(float* data, int64_t size, float temperature) {
    if (size <= 0 || temperature <= 0.0f) return;

    // Apply temperature scaling
    float inv_temp = 1.0f / temperature;
    for (int64_t i = 0; i < size; ++i) {
        data[i] *= inv_temp;
    }

    // Find max for numerical stability
    float max_val = data[0];
    for (int64_t i = 1; i < size; ++i) {
        if (data[i] > max_val) max_val = data[i];
    }

    // Exp and sum
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    // Normalize
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        for (int64_t i = 0; i < size; ++i) {
            data[i] *= inv_sum;
        }
    }
}

/**
 * @brief SIMD-optimized dot product.
 */
inline float simd_dot(const float* a, const float* b, int64_t size) {
#ifdef __AVX2__
    __m256 sum_vec = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    // Horizontal sum
    __m128 low = _mm256_castps256_ps128(sum_vec);
    __m128 high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);
    // Remainder
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

/**
 * @brief SIMD-optimized scaled add: out = alpha * a + beta * b
 */
inline void simd_scaled_add(
    const float* a, float alpha,
    const float* b, float beta,
    float* out, int64_t size
) {
#ifdef __AVX2__
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    int64_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 result = _mm256_fmadd_ps(alpha_vec, a_vec,
                                        _mm256_mul_ps(beta_vec, b_vec));
        _mm256_storeu_ps(out + i, result);
    }
    for (; i < size; ++i) {
        out[i] = alpha * a[i] + beta * b[i];
    }
#else
    for (int64_t i = 0; i < size; ++i) {
        out[i] = alpha * a[i] + beta * b[i];
    }
#endif
}

/**
 * @brief SIMD-optimized weighted sum: out = sum_i(weights[i] * vectors[i])
 *
 * @param vectors Pointer to [num_vectors, dim] array
 * @param weights Pointer to [num_vectors] array (should sum to 1)
 * @param out Output pointer [dim]
 * @param num_vectors Number of vectors to combine
 * @param dim Dimension of each vector
 */
inline void simd_weighted_sum(
    const float* vectors,
    const float* weights,
    float* out,
    int64_t num_vectors,
    int64_t dim
) {
    // Zero output
    for (int64_t d = 0; d < dim; ++d) {
        out[d] = 0.0f;
    }

#ifdef __AVX2__
    for (int64_t v = 0; v < num_vectors; ++v) {
        const float* vec = vectors + v * dim;
        __m256 w_vec = _mm256_set1_ps(weights[v]);
        int64_t d = 0;
        for (; d + 8 <= dim; d += 8) {
            __m256 curr = _mm256_loadu_ps(out + d);
            __m256 vec_d = _mm256_loadu_ps(vec + d);
            curr = _mm256_fmadd_ps(w_vec, vec_d, curr);
            _mm256_storeu_ps(out + d, curr);
        }
        for (; d < dim; ++d) {
            out[d] += weights[v] * vec[d];
        }
    }
#else
    for (int64_t v = 0; v < num_vectors; ++v) {
        for (int64_t d = 0; d < dim; ++d) {
            out[d] += weights[v] * vectors[v * dim + d];
        }
    }
#endif
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_SUPERPOSITION_SLOTS_OP_H_
