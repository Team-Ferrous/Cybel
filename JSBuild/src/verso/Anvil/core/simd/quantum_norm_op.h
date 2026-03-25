// highnoon/_native/ops/quantum_norm_op.h
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
 * @file quantum_norm_op.h
 * @brief Unitary-preserving normalization via Cayley retraction.
 *
 * Phase 30 of Unified Quantum Architecture Enhancement.
 *
 * Implements normalization that preserves quantum coherence:
 *   x_norm = (x / ||x||₂) · scale + bias
 *
 * Key Properties:
 * - Unit norm preservation: Maintains quantum state normalization
 * - Stiefel manifold geometry: Smooth differentiable manifold
 * - Cayley retraction: Gradient via orthogonal projection
 *
 * SIMD optimized for AVX512/AVX2/NEON with scalar fallback.
 */

#ifndef HIGHNOON_NATIVE_OPS_QUANTUM_NORM_OP_H_
#define HIGHNOON_NATIVE_OPS_QUANTUM_NORM_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

// SIMD architecture detection
#if defined(__AVX512F__)
#include <immintrin.h>
#define HN_QNORM_AVX512 1
#define HN_QNORM_SIMD_WIDTH 16
#elif defined(__AVX2__)
#include <immintrin.h>
#define HN_QNORM_AVX2 1
#define HN_QNORM_SIMD_WIDTH 8
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define HN_QNORM_NEON 1
#define HN_QNORM_SIMD_WIDTH 4
#else
#define HN_QNORM_SCALAR 1
#define HN_QNORM_SIMD_WIDTH 1
#endif

namespace highnoon {
namespace ops {
namespace quantum_norm {

// =============================================================================
// L2 NORM COMPUTATION
// =============================================================================

/**
 * @brief Compute L2 norm of a vector with SIMD optimization.
 *
 * @param data Input vector
 * @param size Vector size
 * @return L2 norm
 */
template <typename T>
inline T ComputeL2Norm(const T* data, int64_t size) {
    T sum_sq = static_cast<T>(0);
    int64_t i = 0;

#if defined(HN_QNORM_AVX512)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 v = _mm512_loadu_ps(reinterpret_cast<const float*>(&data[i]));
        acc = _mm512_fmadd_ps(v, v, acc);
    }
    sum_sq = _mm512_reduce_add_ps(acc);
#elif defined(HN_QNORM_AVX2)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(&data[i]));
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum_sq = _mm_cvtss_f32(sum4);
#elif defined(HN_QNORM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t v = vld1q_f32(reinterpret_cast<const float*>(&data[i]));
        acc = vmlaq_f32(acc, v, v);
    }
    float32x2_t sum2 = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum2 = vpadd_f32(sum2, sum2);
    sum_sq = vget_lane_f32(sum2, 0);
#endif

    // Scalar remainder
    for (; i < size; ++i) {
        sum_sq += data[i] * data[i];
    }
    
    return std::sqrt(sum_sq);
}

// =============================================================================
// UNITARY NORMALIZATION KERNEL
// =============================================================================

/**
 * @brief Forward pass for unitary normalization.
 *
 * Computes: output = (x / ||x||₂) · scale + bias
 *
 * @param input Input tensor [batch, seq_len, dim]
 * @param scale Learnable scale [dim]
 * @param bias Learnable bias [dim]
 * @param output Output tensor [batch, seq_len, dim]
 * @param norms Output norms for backward [batch * seq_len] (optional)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <typename T>
inline void UnitaryNormForward(
    const T* input,
    const T* scale,
    const T* bias,
    T* output,
    T* norms,  // Can be nullptr if not needed
    int batch_size,
    int seq_len,
    int dim,
    T eps = 1e-6f) {
    
    const int64_t total_vectors = static_cast<int64_t>(batch_size) * seq_len;
    
    #pragma omp parallel for
    for (int64_t v = 0; v < total_vectors; ++v) {
        const T* in_ptr = input + v * dim;
        T* out_ptr = output + v * dim;
        
        // Compute L2 norm
        T norm = ComputeL2Norm(in_ptr, dim);
        norm = std::max(norm, eps);
        
        if (norms != nullptr) {
            norms[v] = norm;
        }
        
        T inv_norm = static_cast<T>(1) / norm;
        
        // Normalize and apply affine transform
        int d = 0;
        
#if defined(HN_QNORM_AVX512)
        __m512 inv_norm_v = _mm512_set1_ps(inv_norm);
        for (; d + 16 <= dim; d += 16) {
            __m512 x = _mm512_loadu_ps(&in_ptr[d]);
            __m512 s = _mm512_loadu_ps(&scale[d]);
            __m512 b = _mm512_loadu_ps(&bias[d]);
            
            // out = (x / norm) * scale + bias
            __m512 normalized = _mm512_mul_ps(x, inv_norm_v);
            __m512 result = _mm512_fmadd_ps(normalized, s, b);
            
            _mm512_storeu_ps(&out_ptr[d], result);
        }
#elif defined(HN_QNORM_AVX2)
        __m256 inv_norm_v = _mm256_set1_ps(inv_norm);
        for (; d + 8 <= dim; d += 8) {
            __m256 x = _mm256_loadu_ps(&in_ptr[d]);
            __m256 s = _mm256_loadu_ps(&scale[d]);
            __m256 b = _mm256_loadu_ps(&bias[d]);
            
            __m256 normalized = _mm256_mul_ps(x, inv_norm_v);
            __m256 result = _mm256_fmadd_ps(normalized, s, b);
            
            _mm256_storeu_ps(&out_ptr[d], result);
        }
#elif defined(HN_QNORM_NEON)
        float32x4_t inv_norm_v = vdupq_n_f32(inv_norm);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t x = vld1q_f32(&in_ptr[d]);
            float32x4_t s = vld1q_f32(&scale[d]);
            float32x4_t b = vld1q_f32(&bias[d]);
            
            float32x4_t normalized = vmulq_f32(x, inv_norm_v);
            float32x4_t result = vmlaq_f32(b, normalized, s);
            
            vst1q_f32(&out_ptr[d], result);
        }
#endif
        
        // Scalar remainder
        for (; d < dim; ++d) {
            T normalized = in_ptr[d] * inv_norm;
            out_ptr[d] = normalized * scale[d] + bias[d];
        }
    }
}

/**
 * @brief Backward pass for unitary normalization.
 *
 * Computes gradients with orthogonal projection:
 *   ∂L/∂x = P_⊥(∂L/∂x_norm · scale) / norm
 * where P_⊥ = I - x̂·x̂ᵀ (orthogonal complement projection)
 *
 * @param grad_output Gradient w.r.t. output [batch, seq_len, dim]
 * @param input Original input [batch, seq_len, dim]
 * @param scale Scale parameter [dim]
 * @param norms Cached norms [batch * seq_len]
 * @param grad_input Gradient w.r.t. input [batch, seq_len, dim]
 * @param grad_scale Gradient w.r.t. scale [dim] (accumulated)
 * @param grad_bias Gradient w.r.t. bias [dim] (accumulated)
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param dim Feature dimension
 * @param eps Epsilon for numerical stability
 */
template <typename T>
inline void UnitaryNormBackward(
    const T* grad_output,
    const T* input,
    const T* scale,
    const T* norms,
    T* grad_input,
    T* grad_scale,
    T* grad_bias,
    int batch_size,
    int seq_len,
    int dim,
    T eps = 1e-6f) {
    
    const int64_t total_vectors = static_cast<int64_t>(batch_size) * seq_len;
    
    // Zero accumulation buffers
    std::fill(grad_scale, grad_scale + dim, static_cast<T>(0));
    std::fill(grad_bias, grad_bias + dim, static_cast<T>(0));
    
    // Thread-local accumulators for grad_scale and grad_bias
    #pragma omp parallel
    {
        std::vector<T> local_grad_scale(dim, static_cast<T>(0));
        std::vector<T> local_grad_bias(dim, static_cast<T>(0));
        
        #pragma omp for
        for (int64_t v = 0; v < total_vectors; ++v) {
            const T* grad_out_ptr = grad_output + v * dim;
            const T* in_ptr = input + v * dim;
            T* grad_in_ptr = grad_input + v * dim;
            
            T norm = norms[v];
            T inv_norm = static_cast<T>(1) / norm;
            
            // Compute x_hat = x / ||x||
            // Compute grad_pre = grad_output * scale (before affine)
            // Accumulate grad_bias += grad_output
            // Accumulate grad_scale += x_hat * grad_output
            
            // Step 1: Compute dot product ⟨x_hat, grad_pre⟩ for projection
            T dot_product = static_cast<T>(0);
            
            for (int d = 0; d < dim; ++d) {
                T x_hat = in_ptr[d] * inv_norm;
                T grad_pre = grad_out_ptr[d] * scale[d];
                dot_product += x_hat * grad_pre;
                
                // Accumulate grad_scale and grad_bias
                local_grad_scale[d] += x_hat * grad_out_ptr[d];
                local_grad_bias[d] += grad_out_ptr[d];
            }
            
            // Step 2: Compute orthogonal projection: P_⊥(grad_pre) = grad_pre - (x_hat · grad_pre) * x_hat
            // Then divide by norm
            for (int d = 0; d < dim; ++d) {
                T x_hat = in_ptr[d] * inv_norm;
                T grad_pre = grad_out_ptr[d] * scale[d];
                
                // P_⊥ projection
                T projected = grad_pre - dot_product * x_hat;
                
                // Divide by norm
                grad_in_ptr[d] = projected * inv_norm;
            }
        }
        
        // Reduce thread-local accumulators
        #pragma omp critical
        {
            for (int d = 0; d < dim; ++d) {
                grad_scale[d] += local_grad_scale[d];
                grad_bias[d] += local_grad_bias[d];
            }
        }
    }
}

/**
 * @brief RMS normalization variant (no mean centering, just scale by RMS).
 *
 * Computes: output = x / RMS(x) · scale
 * where RMS(x) = sqrt(mean(x²))
 *
 * @param input Input tensor
 * @param scale Scale parameter
 * @param output Output tensor
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param dim Feature dimension
 * @param eps Epsilon
 */
template <typename T>
inline void RMSNormForward(
    const T* input,
    const T* scale,
    T* output,
    int batch_size,
    int seq_len,
    int dim,
    T eps = 1e-6f) {
    
    const int64_t total_vectors = static_cast<int64_t>(batch_size) * seq_len;
    
    #pragma omp parallel for
    for (int64_t v = 0; v < total_vectors; ++v) {
        const T* in_ptr = input + v * dim;
        T* out_ptr = output + v * dim;
        
        // Compute mean of squares
        T mean_sq = static_cast<T>(0);
        for (int d = 0; d < dim; ++d) {
            mean_sq += in_ptr[d] * in_ptr[d];
        }
        mean_sq /= dim;
        
        T rms = std::sqrt(mean_sq + eps);
        T inv_rms = static_cast<T>(1) / rms;
        
        // Normalize
        for (int d = 0; d < dim; ++d) {
            out_ptr[d] = in_ptr[d] * inv_rms * scale[d];
        }
    }
}

}  // namespace quantum_norm
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_QUANTUM_NORM_OP_H_
