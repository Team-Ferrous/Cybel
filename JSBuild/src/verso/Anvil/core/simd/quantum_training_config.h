// highnoon/_native/ops/quantum_training_config.h
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

#ifndef HIGHNOON_NATIVE_OPS_QUANTUM_TRAINING_CONFIG_H_
#define HIGHNOON_NATIVE_OPS_QUANTUM_TRAINING_CONFIG_H_

#include <cmath>
#include <cstdint>
#include <algorithm>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace hsmn {
namespace quantum_training {

// =============================================================================
// Quantum Training Configuration Structure
// =============================================================================

struct QuantumTrainingConfig {
    // Barren Plateau Monitor (T4)
    bool enable_barren_plateau;
    float barren_plateau_threshold;
    float barren_plateau_hysteresis;
    float barren_plateau_lr_scale;
    
    // Quantum Natural Gradient (T1)
    bool enable_qng;
    float qng_damping;
    float qng_ema_decay;
    
    // Tensor-GaLore (T2) - Note: Complex compression done in Python
    // C++ side handles scaled gradient application
    bool enable_galore;
    float galore_scale;
    
    // Evolution Time Optimizer (T5)
    bool enable_evolution_time_opt;
    float evolution_time_lr;
    float evolution_time_min;
    float evolution_time_max;
    
    // Default configuration
    static QuantumTrainingConfig Default() {
        return {
            .enable_barren_plateau = true,
            .barren_plateau_threshold = 1e-6f,
            .barren_plateau_hysteresis = 5.0f,
            .barren_plateau_lr_scale = 10.0f,
            .enable_qng = false,
            .qng_damping = 1e-4f,
            .qng_ema_decay = 0.99f,
            .enable_galore = false,
            .galore_scale = 0.25f,
            .enable_evolution_time_opt = false,
            .evolution_time_lr = 0.01f,
            .evolution_time_min = 1e-6f,
            .evolution_time_max = 1e3f,
        };
    }
};

// =============================================================================
// SIMD-Optimized Quantum Training Functions
// =============================================================================

// Vectorized gradient norm computation for barren plateau detection
inline float VectorizedGradientNorm(const float* grad, int64_t size) {
    float norm_sq = 0.0f;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    for (; i + 16 <= size; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        acc = _mm512_fmadd_ps(g, g, acc);
    }
    norm_sq = _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        acc = _mm256_fmadd_ps(g, g, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    norm_sq = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i + 4 <= size; i += 4) {
        float32x4_t g = vld1q_f32(&grad[i]);
        acc = vmlaq_f32(acc, g, g);
    }
    float32x2_t sum = vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
    sum = vpadd_f32(sum, sum);
    norm_sq = vget_lane_f32(sum, 0);
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        norm_sq += grad[i] * grad[i];
    }

    return std::sqrt(norm_sq);
}

// Vectorized QFIM diagonal update (EMA of squared gradients)
inline void VectorizedQFIMUpdate(const float* grad, float* qfim, int64_t size,
                                  float ema_decay) {
    const float one_minus_decay = 1.0f - ema_decay;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 decay_vec = _mm512_set1_ps(ema_decay);
    __m512 one_minus_decay_vec = _mm512_set1_ps(one_minus_decay);
    for (; i + 16 <= size; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 q = _mm512_loadu_ps(&qfim[i]);
        __m512 g_sq = _mm512_mul_ps(g, g);
        // q_new = decay * q + (1 - decay) * g^2
        __m512 q_new = _mm512_fmadd_ps(decay_vec, q,
                                       _mm512_mul_ps(one_minus_decay_vec, g_sq));
        _mm512_storeu_ps(&qfim[i], q_new);
    }
#elif defined(__AVX2__)
    __m256 decay_vec = _mm256_set1_ps(ema_decay);
    __m256 one_minus_decay_vec = _mm256_set1_ps(one_minus_decay);
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 q = _mm256_loadu_ps(&qfim[i]);
        __m256 g_sq = _mm256_mul_ps(g, g);
        // q_new = decay * q + (1 - decay) * g^2
        __m256 q_new = _mm256_fmadd_ps(decay_vec, q,
                                       _mm256_mul_ps(one_minus_decay_vec, g_sq));
        _mm256_storeu_ps(&qfim[i], q_new);
    }
#elif defined(__ARM_NEON)
    float32x4_t decay_vec = vdupq_n_f32(ema_decay);
    float32x4_t one_minus_decay_vec = vdupq_n_f32(one_minus_decay);
    for (; i + 4 <= size; i += 4) {
        float32x4_t g = vld1q_f32(&grad[i]);
        float32x4_t q = vld1q_f32(&qfim[i]);
        float32x4_t g_sq = vmulq_f32(g, g);
        // q_new = decay * q + (1 - decay) * g^2
        float32x4_t q_new = vmlaq_f32(vmulq_f32(one_minus_decay_vec, g_sq),
                                       decay_vec, q);
        vst1q_f32(&qfim[i], q_new);
    }
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        float g_sq = grad[i] * grad[i];
        qfim[i] = ema_decay * qfim[i] + one_minus_decay * g_sq;
    }
}

// Vectorized QNG preconditioning: grad_precond = grad / (sqrt(qfim) + damping)
inline void VectorizedQNGPrecondition(float* grad, const float* qfim, int64_t size,
                                       float damping) {
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 damp_vec = _mm512_set1_ps(damping);
    for (; i + 16 <= size; i += 16) {
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 q = _mm512_loadu_ps(&qfim[i]);
        __m512 precond = _mm512_add_ps(_mm512_sqrt_ps(q), damp_vec);
        __m512 g_new = _mm512_div_ps(g, precond);
        _mm512_storeu_ps(&grad[i], g_new);
    }
#elif defined(__AVX2__)
    __m256 damp_vec = _mm256_set1_ps(damping);
    for (; i + 8 <= size; i += 8) {
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 q = _mm256_loadu_ps(&qfim[i]);
        __m256 precond = _mm256_add_ps(_mm256_sqrt_ps(q), damp_vec);
        __m256 g_new = _mm256_div_ps(g, precond);
        _mm256_storeu_ps(&grad[i], g_new);
    }
#elif defined(__ARM_NEON)
    float32x4_t damp_vec = vdupq_n_f32(damping);
    for (; i + 4 <= size; i += 4) {
        float32x4_t g = vld1q_f32(&grad[i]);
        float32x4_t q = vld1q_f32(&qfim[i]);
        // ARM NEON sqrt approximation
        float32x4_t q_sqrt;
        q_sqrt = vrsqrteq_f32(q);
        q_sqrt = vmulq_f32(q_sqrt, vrsqrtsq_f32(vmulq_f32(q, q_sqrt), q_sqrt));
        q_sqrt = vmulq_f32(q, q_sqrt); // This gives sqrt(q)
        float32x4_t precond = vaddq_f32(q_sqrt, damp_vec);
        // NEON division approximation
        float32x4_t reciprocal = vrecpeq_f32(precond);
        reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(precond, reciprocal));
        float32x4_t g_new = vmulq_f32(g, reciprocal);
        vst1q_f32(&grad[i], g_new);
    }
#endif

    // Scalar fallback
    for (; i < size; ++i) {
        float precond = std::sqrt(qfim[i]) + damping;
        grad[i] = grad[i] / precond;
    }
}

// Check for barren plateau condition
inline bool IsBarrenPlateau(float grad_norm, float threshold) {
    return grad_norm < threshold;
}

// Compute LR scale for barren plateau mitigation
inline float BarrenPlateauLRScale(float grad_norm, float avg_grad_norm,
                                   float threshold, float hysteresis,
                                   float base_scale) {
    if (grad_norm < threshold) {
        // In barren plateau - scale up LR
        return base_scale;
    } else if (grad_norm < threshold * hysteresis) {
        // In hysteresis zone - gradually reduce scale
        float t = (grad_norm - threshold) / (threshold * (hysteresis - 1.0f));
        return 1.0f + (base_scale - 1.0f) * (1.0f - t);
    }
    return 1.0f;
}

// =============================================================================
// PHASE 5: QNG GEODESIC CORRECTIONS FOR MoE ROUTING
// Applies quantum natural gradient with Riemannian exponential map.
// Ensures updates follow geodesics on the parameter manifold.
// =============================================================================

/**
 * @brief First-order geodesic update: params_new = params - lr * QFIM^{-1} * grad
 * 
 * Standard QNG update without geodesic corrections.
 */
inline void VectorizedQNGGeodesicOrder1(
    float* params, const float* grad, const float* qfim,
    float step_size, float damping, int64_t size) {
    
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 step_vec = _mm512_set1_ps(step_size);
    __m512 damp_vec = _mm512_set1_ps(damping);
    for (; i + 16 <= size; i += 16) {
        __m512 p = _mm512_loadu_ps(&params[i]);
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 q = _mm512_loadu_ps(&qfim[i]);
        
        // QNG: params -= step * grad / (sqrt(qfim) + damping)
        __m512 precond = _mm512_add_ps(_mm512_sqrt_ps(q), damp_vec);
        __m512 natural_grad = _mm512_div_ps(g, precond);
        p = _mm512_fnmadd_ps(step_vec, natural_grad, p);
        
        _mm512_storeu_ps(&params[i], p);
    }
#elif defined(__AVX2__)
    __m256 step_vec = _mm256_set1_ps(step_size);
    __m256 damp_vec = _mm256_set1_ps(damping);
    for (; i + 8 <= size; i += 8) {
        __m256 p = _mm256_loadu_ps(&params[i]);
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 q = _mm256_loadu_ps(&qfim[i]);
        
        __m256 precond = _mm256_add_ps(_mm256_sqrt_ps(q), damp_vec);
        __m256 natural_grad = _mm256_div_ps(g, precond);
        p = _mm256_fnmadd_ps(step_vec, natural_grad, p);
        
        _mm256_storeu_ps(&params[i], p);
    }
#elif defined(__ARM_NEON)
    float32x4_t step_vec = vdupq_n_f32(step_size);
    float32x4_t damp_vec = vdupq_n_f32(damping);
    for (; i + 4 <= size; i += 4) {
        float32x4_t p = vld1q_f32(&params[i]);
        float32x4_t g = vld1q_f32(&grad[i]);
        float32x4_t q = vld1q_f32(&qfim[i]);
        
        // NEON sqrt approximation
        float32x4_t q_sqrt = vrsqrteq_f32(q);
        q_sqrt = vmulq_f32(q_sqrt, vrsqrtsq_f32(vmulq_f32(q, q_sqrt), q_sqrt));
        q_sqrt = vmulq_f32(q, q_sqrt);
        
        float32x4_t precond = vaddq_f32(q_sqrt, damp_vec);
        float32x4_t reciprocal = vrecpeq_f32(precond);
        reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(precond, reciprocal));
        float32x4_t natural_grad = vmulq_f32(g, reciprocal);
        
        p = vmlsq_f32(p, step_vec, natural_grad);
        vst1q_f32(&params[i], p);
    }
#endif

    for (; i < size; ++i) {
        float precond = std::sqrt(qfim[i]) + damping;
        params[i] -= step_size * grad[i] / precond;
    }
}

/**
 * @brief Second-order geodesic QNG with Christoffel symbol correction.
 * 
 * Adds correction term proportional to gradient squared and curvature:
 *   params_new = params - lr * g_nat + 0.5 * lr^2 * Γ * g_nat^2
 * 
 * where Γ approximates the Christoffel symbol via QFIM gradient.
 * 
 * @param params Parameters to update [size]
 * @param grad Gradient [size]
 * @param qfim QFIM diagonal [size]
 * @param qfim_prev Previous QFIM (for Christoffel approximation) [size]
 * @param step_size Learning rate
 * @param damping Damping factor
 * @param size Number of parameters
 */
inline void VectorizedQNGGeodesicOrder2(
    float* params, const float* grad, const float* qfim,
    const float* qfim_prev, float step_size, float damping, int64_t size) {
    
    const float lr_sq_half = 0.5f * step_size * step_size;
    int64_t i = 0;

#if defined(__AVX512F__)
    __m512 step_vec = _mm512_set1_ps(step_size);
    __m512 lr_sq_half_vec = _mm512_set1_ps(lr_sq_half);
    __m512 damp_vec = _mm512_set1_ps(damping);
    __m512 eps_vec = _mm512_set1_ps(1e-8f);
    
    for (; i + 16 <= size; i += 16) {
        __m512 p = _mm512_loadu_ps(&params[i]);
        __m512 g = _mm512_loadu_ps(&grad[i]);
        __m512 q = _mm512_loadu_ps(&qfim[i]);
        __m512 q_prev = _mm512_loadu_ps(&qfim_prev[i]);
        
        // First-order natural gradient
        __m512 precond = _mm512_add_ps(_mm512_sqrt_ps(q), damp_vec);
        __m512 g_nat = _mm512_div_ps(g, precond);
        
        // Christoffel approximation: (q - q_prev) / (q + eps)
        __m512 dq = _mm512_sub_ps(q, q_prev);
        __m512 christoffel = _mm512_div_ps(dq, _mm512_add_ps(q, eps_vec));
        
        // Geodesic correction: 0.5 * lr^2 * Γ * g_nat^2
        __m512 g_nat_sq = _mm512_mul_ps(g_nat, g_nat);
        __m512 correction = _mm512_mul_ps(lr_sq_half_vec, 
                             _mm512_mul_ps(christoffel, g_nat_sq));
        
        // Final update: params -= lr * g_nat - correction
        p = _mm512_fnmadd_ps(step_vec, g_nat, p);
        p = _mm512_add_ps(p, correction);
        
        _mm512_storeu_ps(&params[i], p);
    }
#elif defined(__AVX2__)
    __m256 step_vec = _mm256_set1_ps(step_size);
    __m256 lr_sq_half_vec = _mm256_set1_ps(lr_sq_half);
    __m256 damp_vec = _mm256_set1_ps(damping);
    __m256 eps_vec = _mm256_set1_ps(1e-8f);
    
    for (; i + 8 <= size; i += 8) {
        __m256 p = _mm256_loadu_ps(&params[i]);
        __m256 g = _mm256_loadu_ps(&grad[i]);
        __m256 q = _mm256_loadu_ps(&qfim[i]);
        __m256 q_prev = _mm256_loadu_ps(&qfim_prev[i]);
        
        __m256 precond = _mm256_add_ps(_mm256_sqrt_ps(q), damp_vec);
        __m256 g_nat = _mm256_div_ps(g, precond);
        
        __m256 dq = _mm256_sub_ps(q, q_prev);
        __m256 christoffel = _mm256_div_ps(dq, _mm256_add_ps(q, eps_vec));
        
        __m256 g_nat_sq = _mm256_mul_ps(g_nat, g_nat);
        __m256 correction = _mm256_mul_ps(lr_sq_half_vec,
                             _mm256_mul_ps(christoffel, g_nat_sq));
        
        p = _mm256_fnmadd_ps(step_vec, g_nat, p);
        p = _mm256_add_ps(p, correction);
        
        _mm256_storeu_ps(&params[i], p);
    }
#endif

    for (; i < size; ++i) {
        float precond = std::sqrt(qfim[i]) + damping;
        float g_nat = grad[i] / precond;
        
        // Christoffel approximation
        float dq = qfim[i] - qfim_prev[i];
        float christoffel = dq / (qfim[i] + 1e-8f);
        
        // Geodesic correction
        float correction = lr_sq_half * christoffel * g_nat * g_nat;
        
        params[i] -= step_size * g_nat - correction;
    }
}

/**
 * @brief Apply geodesic QNG update with automatic order selection.
 * 
 * @param params Parameters to update
 * @param grad Gradient
 * @param qfim Current QFIM diagonal
 * @param qfim_prev Previous QFIM (can be nullptr for order 1)
 * @param step_size Learning rate
 * @param damping Damping factor
 * @param size Number of parameters
 * @param order Geodesic order (1 or 2)
 */
inline void VectorizedQNGGeodesic(
    float* params, const float* grad, const float* qfim,
    const float* qfim_prev, float step_size, float damping,
    int64_t size, int order = 2) {
    
    if (order >= 2 && qfim_prev != nullptr) {
        VectorizedQNGGeodesicOrder2(params, grad, qfim, qfim_prev,
                                     step_size, damping, size);
    } else {
        VectorizedQNGGeodesicOrder1(params, grad, qfim,
                                     step_size, damping, size);
    }
}

}  // namespace quantum_training
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_QUANTUM_TRAINING_CONFIG_H_
