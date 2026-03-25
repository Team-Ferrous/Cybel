// saguaro.native/ops/fused_streaming_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// SIMD helpers for KV-free streaming inference.
// Provides state compression and efficient state updates.

#ifndef SAGUARO_NATIVE_OPS_FUSED_STREAMING_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_STREAMING_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace saguaro {
namespace ops {

/**
 * @brief Compress state using low-rank approximation.
 *
 * Simple compression via truncation. For full SVD, use TensorFlow's linear algebra.
 */
inline void streaming_compress_state(
    const float* input, float* output,
    int64_t batch_size, int64_t state_dim, int64_t target_dim) {
    
    int64_t copy_dim = std::min(state_dim, target_dim);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* src = input + b * state_dim;
        float* dst = output + b * target_dim;
        
        std::copy(src, src + copy_dim, dst);
        
        // Zero-pad if target is larger
        if (target_dim > state_dim) {
            std::fill(dst + state_dim, dst + target_dim, 0.0f);
        }
    }
}

/**
 * @brief Update streaming state with new chunk.
 *
 * Blends old state with new state using exponential moving average.
 */
inline void streaming_update_state(
    float* state, const float* new_state,
    int64_t batch_size, int64_t state_dim, float alpha = 0.9f) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        float* s = state + b * state_dim;
        const float* ns = new_state + b * state_dim;
        
        int64_t d = 0;
#if defined(__AVX512F__)
        __m512 a = _mm512_set1_ps(alpha);
        __m512 one_minus_a = _mm512_set1_ps(1.0f - alpha);
        for (; d + 16 <= state_dim; d += 16) {
            __m512 old = _mm512_loadu_ps(&s[d]);
            __m512 nw = _mm512_loadu_ps(&ns[d]);
            __m512 result = _mm512_fmadd_ps(a, old, _mm512_mul_ps(one_minus_a, nw));
            _mm512_storeu_ps(&s[d], result);
        }
#elif defined(__AVX2__)
        __m256 a = _mm256_set1_ps(alpha);
        __m256 one_minus_a = _mm256_set1_ps(1.0f - alpha);
        for (; d + 8 <= state_dim; d += 8) {
            __m256 old = _mm256_loadu_ps(&s[d]);
            __m256 nw = _mm256_loadu_ps(&ns[d]);
            __m256 result = _mm256_fmadd_ps(a, old, _mm256_mul_ps(one_minus_a, nw));
            _mm256_storeu_ps(&s[d], result);
        }
#elif defined(__ARM_NEON)
        float32x4_t a = vdupq_n_f32(alpha);
        float32x4_t one_minus_a = vdupq_n_f32(1.0f - alpha);
        for (; d + 4 <= state_dim; d += 4) {
            float32x4_t old = vld1q_f32(&s[d]);
            float32x4_t nw = vld1q_f32(&ns[d]);
            float32x4_t result = vmlaq_f32(vmulq_f32(one_minus_a, nw), a, old);
            vst1q_f32(&s[d], result);
        }
#endif
        for (; d < state_dim; ++d) {
            s[d] = alpha * s[d] + (1.0f - alpha) * ns[d];
        }
    }
}

/**
 * @brief Initialize zero state.
 */
inline void streaming_init_state(
    float* state, int64_t batch_size, int64_t state_dim) {
    
    std::fill(state, state + batch_size * state_dim, 0.0f);
}

/**
 * @brief Copy state.
 */
inline void streaming_copy_state(
    const float* src, float* dst, int64_t total_size) {
    
    std::copy(src, src + total_size, dst);
}

/**
 * @brief Compute L2 norm of state (for compression quality assessment).
 */
inline float streaming_state_norm(
    const float* state, int64_t size) {
    
    float sum = 0.0f;
    for (int64_t i = 0; i < size; ++i) {
        sum += state[i] * state[i];
    }
    return std::sqrt(sum);
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_STREAMING_OP_H_
