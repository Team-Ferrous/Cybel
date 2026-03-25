// saguaro.native/ops/sympflow_optimizer_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");

/**
 * @file sympflow_optimizer_op.h
 * @brief Phase 46: SympFlow Hamiltonian Optimizer
 *
 * Treats gradient descent as Hamiltonian dynamics with symplectic integrators
 * for the training loop.
 *
 * Key Features:
 *   - Leapfrog Integration: 2nd-order symplectic integrator
 *   - Momentum as Conjugate: Natural momentum from Hamilton's equations
 *   - Friction Term: Controlled dissipation for convergence
 *
 * Research Basis: "Symplectic Optimization" (arXiv 2023)
 *
 * Integration Points:
 *   - Training loop optimizer
 */

#ifndef SAGUARO_NATIVE_OPS_SYMPFLOW_OPTIMIZER_OP_H_
#define SAGUARO_NATIVE_OPS_SYMPFLOW_OPTIMIZER_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace saguaro {
namespace sympflow {

// =============================================================================
// CONFIGURATION
// =============================================================================

struct SympFlowConfig {
    float mass;                   // Effective mass for momentum
    float friction;               // Dissipation rate γ
    float step_size;              // Leapfrog step size (learning rate)
    int num_leapfrog_steps;       // Leapfrog sub-steps
    
    SympFlowConfig()
        : mass(1.0f)
        , friction(0.1f)
        , step_size(0.01f)
        , num_leapfrog_steps(1) {}
};

// =============================================================================
// SYMPLECTIC INTEGRATION
// =============================================================================

/**
 * @brief Leapfrog step for symplectic optimization.
 *
 * Hamilton's equations:
 *   dθ/dt = p/m         (position update)
 *   dp/dt = -∇L - γp    (momentum update with friction)
 *
 * Leapfrog integrator (2nd order):
 *   p_{1/2} = p_n - (h/2) * (∇L + γ*p_n)
 *   θ_{n+1} = θ_n + h * p_{1/2} / m
 *   p_{n+1} = p_{1/2} - (h/2) * (∇L + γ*p_{1/2})
 *
 * @param params Parameters θ [num_params]
 * @param momentum Momentum p [num_params]
 * @param gradients Gradients ∇L [num_params]
 * @param config SympFlow configuration
 * @param num_params Number of parameters
 */
inline void LeapfrogStep(
    float* params, float* momentum, const float* gradients,
    const SympFlowConfig& config,
    int num_params) {
    
    const float h = config.step_size;
    const float m = config.mass;
    const float gamma = config.friction;
    const float half_h = h * 0.5f;
    const float inv_m = 1.0f / m;
    
    for (int step = 0; step < config.num_leapfrog_steps; ++step) {
        int i = 0;
        
#if defined(__AVX2__)
        __m256 h_half = _mm256_set1_ps(half_h);
        __m256 h_full = _mm256_set1_ps(h);
        __m256 inv_m_v = _mm256_set1_ps(inv_m);
        __m256 gamma_v = _mm256_set1_ps(gamma);
        
        for (; i + 8 <= num_params; i += 8) {
            __m256 p = _mm256_loadu_ps(&momentum[i]);
            __m256 g = _mm256_loadu_ps(&gradients[i]);
            __m256 theta = _mm256_loadu_ps(&params[i]);
            
            // p_{1/2} = p - (h/2) * (g + γ*p)
            __m256 friction_term = _mm256_mul_ps(gamma_v, p);
            __m256 force = _mm256_add_ps(g, friction_term);
            __m256 dp_half = _mm256_mul_ps(h_half, force);
            __m256 p_half = _mm256_sub_ps(p, dp_half);
            
            // θ = θ + h * p_{1/2} / m
            __m256 dtheta = _mm256_mul_ps(h_full, _mm256_mul_ps(p_half, inv_m_v));
            theta = _mm256_add_ps(theta, dtheta);
            
            // p = p_{1/2} - (h/2) * (g + γ*p_{1/2})
            friction_term = _mm256_mul_ps(gamma_v, p_half);
            force = _mm256_add_ps(g, friction_term);
            __m256 dp = _mm256_mul_ps(h_half, force);
            p = _mm256_sub_ps(p_half, dp);
            
            _mm256_storeu_ps(&params[i], theta);
            _mm256_storeu_ps(&momentum[i], p);
        }
#endif
        // Scalar remainder
        for (; i < num_params; ++i) {
            float p = momentum[i];
            float g = gradients[i];
            
            // Half momentum step
            float force = g + gamma * p;
            float p_half = p - half_h * force;
            
            // Full position step
            params[i] += h * p_half * inv_m;
            
            // Half momentum step
            force = g + gamma * p_half;
            momentum[i] = p_half - half_h * force;
        }
    }
}

/**
 * @brief Initialize momentum with damped noise.
 *
 * @param momentum Output momentum [num_params]
 * @param gradients Initial gradients [num_params]
 * @param mass Effective mass
 * @param num_params Number of parameters
 */
inline void InitializeMomentum(
    float* momentum, const float* gradients,
    float mass, int num_params) {
    
    // Initialize momentum opposite to gradient for initial descent
    for (int i = 0; i < num_params; ++i) {
        momentum[i] = -gradients[i] * mass * 0.1f;
    }
}

/**
 * @brief Compute kinetic energy (momentum-based).
 *
 * KE = 0.5 * Σ p_i² / m
 *
 * @param momentum Momentum [num_params]
 * @param mass Effective mass
 * @param num_params Number of parameters
 * @return Kinetic energy
 */
inline float ComputeKineticEnergy(
    const float* momentum, float mass, int num_params) {
    
    float ke = 0.0f;
    
    int i = 0;
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= num_params; i += 8) {
        __m256 p = _mm256_loadu_ps(&momentum[i]);
        sum = _mm256_fmadd_ps(p, p, sum);
    }
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    ke = _mm_cvtss_f32(sum4);
#endif
    for (; i < num_params; ++i) {
        ke += momentum[i] * momentum[i];
    }
    
    return 0.5f * ke / mass;
}

}  // namespace sympflow
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_SYMPFLOW_OPTIMIZER_OP_H_
