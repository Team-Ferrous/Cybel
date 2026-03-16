// saguaro.native/ops/quantum_galore_op.h
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
 * @file quantum_galore_op.h
 * @brief Phase 91: Tensor-GaLore v2 with Quantum Rank Selection
 *
 * Implements quantum-inspired dynamic rank selection for gradient compression:
 * - Entropy-based effective rank computation from gradient eigenvalue spectrum
 * - Block-wise rank allocation using Taylor expansion influence scores
 * - Quantum random feature projection matrix updates (integrates with QuantumFeatureMapKernel)
 *
 * Complexity: O(rank · d) for projected gradients
 * Memory: O(rank · d) << O(d²) for full gradients
 *
 * Reference: "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection" (2024)
 */

#ifndef SAGUARO_NATIVE_OPS_QUANTUM_GALORE_OP_H_
#define SAGUARO_NATIVE_OPS_QUANTUM_GALORE_OP_H_

#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#include "common/perf_utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD intrinsics
#if defined(__AVX512F__)
#include <immintrin.h>
#define QUANTUM_GALORE_USE_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define QUANTUM_GALORE_USE_AVX2 1
#elif defined(__SSE4_2__)
#include <emmintrin.h>
#include <smmintrin.h>
#define QUANTUM_GALORE_USE_SSE4 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define QUANTUM_GALORE_USE_NEON 1
#endif

namespace saguaro {
namespace quantum_galore {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr float kEntropyEpsilon = 1e-8f;
constexpr float kMinRankFraction = 0.1f;  // Minimum 10% of max rank
constexpr float kMaxRankFraction = 1.0f;  // Maximum 100% of max rank

// -----------------------------------------------------------------------------
// SIMD Utilities
// -----------------------------------------------------------------------------

#if defined(QUANTUM_GALORE_USE_AVX512)
/**
 * @brief AVX-512 horizontal sum of 16 floats.
 */
inline float _mm512_reduce_add_ps_safe(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

#if defined(QUANTUM_GALORE_USE_AVX2)
/**
 * @brief AVX2 horizontal sum of 8 floats.
 */
inline float _mm256_reduce_add_ps_manual(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float _mm256_reduce_max_ps_manual(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 maxs = _mm_max_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);
    maxs = _mm_max_ss(maxs, shuf);
    return _mm_cvtss_f32(maxs);
}
#endif

// -----------------------------------------------------------------------------
// Core Kernels
// -----------------------------------------------------------------------------

/**
 * @brief Compute Shannon entropy of normalized eigenvalue spectrum.
 *
 * H = -Σ p_i log(p_i) where p_i = λ_i / Σλ_j (normalized eigenvalues)
 *
 * @param eigenvalues Sorted eigenvalues (descending order)
 * @param num_eigenvalues Number of eigenvalues
 * @return Shannon entropy of the spectrum
 */
template <typename T>
T ComputeSpectrumEntropy(const T* eigenvalues, int num_eigenvalues) {
    if (num_eigenvalues <= 0) return static_cast<T>(0);

    // First pass: compute sum for normalization
    T sum = static_cast<T>(0);
    
#if defined(QUANTUM_GALORE_USE_AVX2)
    if (num_eigenvalues >= 8) {
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < num_eigenvalues; i += 8) {
            __m256 ev = _mm256_loadu_ps(reinterpret_cast<const float*>(eigenvalues + i));
            ev = _mm256_max_ps(ev, _mm256_setzero_ps());  // Clamp to non-negative
            sum_vec = _mm256_add_ps(sum_vec, ev);
        }
        sum = static_cast<T>(_mm256_reduce_add_ps_manual(sum_vec));
        for (; i < num_eigenvalues; ++i) {
            T val = eigenvalues[i];
            if (val > static_cast<T>(0)) sum += val;
        }
    } else
#endif
    {
        for (int i = 0; i < num_eigenvalues; ++i) {
            T val = eigenvalues[i];
            if (val > static_cast<T>(0)) sum += val;
        }
    }

    if (sum < static_cast<T>(kEntropyEpsilon)) {
        return static_cast<T>(0);
    }

    // Second pass: compute entropy
    T inv_sum = static_cast<T>(1) / sum;
    T entropy = static_cast<T>(0);

#if defined(QUANTUM_GALORE_USE_AVX2)
    if (num_eigenvalues >= 8) {
        __m256 entropy_vec = _mm256_setzero_ps();
        __m256 inv_sum_vec = _mm256_set1_ps(static_cast<float>(inv_sum));
        __m256 eps_vec = _mm256_set1_ps(kEntropyEpsilon);
        
        int i = 0;
        for (; i + 7 < num_eigenvalues; i += 8) {
            __m256 ev = _mm256_loadu_ps(reinterpret_cast<const float*>(eigenvalues + i));
            ev = _mm256_max_ps(ev, _mm256_setzero_ps());
            __m256 p = _mm256_mul_ps(ev, inv_sum_vec);
            p = _mm256_add_ps(p, eps_vec);  // Add epsilon for log stability
            
            // Compute -p * log(p) using scalar fallback for log
            // (AVX2 doesn't have native log, would need Intel SVML or approximation)
            alignas(32) float p_arr[8];
            _mm256_store_ps(p_arr, p);
            for (int j = 0; j < 8; ++j) {
                if (p_arr[j] > kEntropyEpsilon) {
                    entropy -= static_cast<T>(p_arr[j]) * std::log(static_cast<T>(p_arr[j]));
                }
            }
        }
        for (; i < num_eigenvalues; ++i) {
            T val = eigenvalues[i];
            if (val > static_cast<T>(0)) {
                T p = val * inv_sum + static_cast<T>(kEntropyEpsilon);
                entropy -= p * std::log(p);
            }
        }
    } else
#endif
    {
        for (int i = 0; i < num_eigenvalues; ++i) {
            T val = eigenvalues[i];
            if (val > static_cast<T>(0)) {
                T p = val * inv_sum + static_cast<T>(kEntropyEpsilon);
                entropy -= p * std::log(p);
            }
        }
    }

    return entropy;
}

/**
 * @brief Compute effective rank from entropy via exp(H).
 *
 * Effective rank = exp(entropy) is the quantum analog of participation ratio,
 * indicating how many dimensions are "active" in the gradient.
 *
 * @param eigenvalues Sorted eigenvalues (descending)
 * @param num_eigenvalues Number of eigenvalues
 * @param max_rank Maximum allowable rank
 * @param min_rank Minimum rank (default: max_rank * 0.1)
 * @return Effective rank clamped to [min_rank, max_rank]
 */
template <typename T>
int ComputeEffectiveRank(
    const T* eigenvalues,
    int num_eigenvalues,
    int max_rank,
    int min_rank = -1
) {
    if (min_rank < 0) {
        min_rank = std::max(1, static_cast<int>(max_rank * kMinRankFraction));
    }

    T entropy = ComputeSpectrumEntropy(eigenvalues, num_eigenvalues);
    T effective_rank_f = std::exp(entropy);
    
    int effective_rank = static_cast<int>(std::round(effective_rank_f));
    effective_rank = std::max(min_rank, std::min(max_rank, effective_rank));
    
    return effective_rank;
}

/**
 * @brief Compute Taylor expansion influence scores for block-wise rank allocation.
 *
 * Influence score = ||∇W||² / ||W||² approximates the Fisher information diagonal,
 * indicating block importance for learning.
 *
 * @param block_gradients Array of gradient norms per block
 * @param block_weights Array of weight norms per block
 * @param influence_scores Output influence scores (normalized)
 * @param num_blocks Number of blocks
 */
template <typename T>
void ComputeBlockInfluenceScores(
    const T* block_gradients,
    const T* block_weights,
    T* influence_scores,
    int num_blocks
) {
    T total_influence = static_cast<T>(0);
    
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:total_influence) schedule(static)
    #endif
    for (int i = 0; i < num_blocks; ++i) {
        T w_norm = block_weights[i] + static_cast<T>(kEntropyEpsilon);
        T g_norm = block_gradients[i];
        T score = (g_norm * g_norm) / (w_norm * w_norm);
        influence_scores[i] = score;
        total_influence += score;
    }
    
    // Normalize scores to sum to 1
    if (total_influence > static_cast<T>(kEntropyEpsilon)) {
        T inv_total = static_cast<T>(1) / total_influence;
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < num_blocks; ++i) {
            influence_scores[i] *= inv_total;
        }
    } else {
        // Uniform allocation if no gradient signal
        T uniform = static_cast<T>(1) / static_cast<T>(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            influence_scores[i] = uniform;
        }
    }
}

/**
 * @brief Allocate rank budget across blocks based on influence scores.
 *
 * Critical blocks (first/last layers) receive minimum 1.5x average allocation.
 *
 * @param influence_scores Normalized influence scores
 * @param num_blocks Number of blocks
 * @param total_rank_budget Total rank budget to distribute
 * @param rank_allocations Output: rank per block
 * @param min_rank Minimum rank per block
 * @param critical_block_ids Indices of critical blocks (first/last)
 * @param num_critical Number of critical blocks
 */
template <typename T>
void AllocateBlockRanks(
    const T* influence_scores,
    int num_blocks,
    int total_rank_budget,
    int* rank_allocations,
    int min_rank,
    const int* critical_block_ids = nullptr,
    int num_critical = 0
) {
    // Base allocation proportional to influence
    int allocated = 0;
    
    for (int i = 0; i < num_blocks; ++i) {
        int rank = std::max(
            min_rank,
            static_cast<int>(std::round(influence_scores[i] * total_rank_budget))
        );
        rank_allocations[i] = rank;
        allocated += rank;
    }
    
    // Boost critical blocks (first/last layers)
    if (critical_block_ids != nullptr && num_critical > 0) {
        T avg_rank = static_cast<T>(total_rank_budget) / static_cast<T>(num_blocks);
        int critical_min = static_cast<int>(std::ceil(avg_rank * 1.5f));
        
        for (int c = 0; c < num_critical; ++c) {
            int idx = critical_block_ids[c];
            if (idx >= 0 && idx < num_blocks && rank_allocations[idx] < critical_min) {
                allocated -= rank_allocations[idx];
                rank_allocations[idx] = critical_min;
                allocated += critical_min;
            }
        }
    }
    
    // Redistribute excess/deficit proportionally
    int diff = total_rank_budget - allocated;
    if (diff != 0 && num_blocks > 0) {
        int adjustment = diff / num_blocks;
        int remainder = diff % num_blocks;
        
        for (int i = 0; i < num_blocks; ++i) {
            rank_allocations[i] += adjustment;
            if (i < std::abs(remainder)) {
                rank_allocations[i] += (diff > 0 ? 1 : -1);
            }
            rank_allocations[i] = std::max(min_rank, rank_allocations[i]);
        }
    }
}

/**
 * @brief Apply quantum random feature projection to gradient.
 *
 * Uses cos(Rx + b) feature map inspired by quantum RBF kernel,
 * providing more stable projection updates than standard random projections.
 *
 * @param gradient Input gradient matrix [rows x cols]
 * @param rotation_matrix Random rotation parameters [rank x cols]
 * @param bias Random bias values [rank]
 * @param output Projected gradient [rank x cols] or [rows x rank]
 * @param rows Number of rows
 * @param cols Number of columns
 * @param rank Target projection rank
 * @param project_rows If true, project rows; otherwise project columns
 */
template <typename T>
void QuantumRandomProjection(
    const T* gradient,
    const T* rotation_matrix,
    const T* bias,
    T* output,
    int rows,
    int cols,
    int rank,
    bool project_rows = true
) {
    if (project_rows) {
        // Output shape: [rank, cols]
        // G_c = φ(R)^T @ G where φ(x) = cos(Rx + b)
        
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int r = 0; r < rank; ++r) {
            for (int c = 0; c < cols; ++c) {
                T sum = static_cast<T>(0);
                
                // Inner product with quantum feature
                for (int i = 0; i < rows; ++i) {
                    // Phase 96: Prefetch next rows for cache efficiency
                    if (i + 4 < rows) {
                        saguaro::ops::PrefetchT0(gradient + (i + 4) * cols + c);
                        saguaro::ops::PrefetchT0(rotation_matrix + r * rows + i + 4);
                    }
                    T rot_val = rotation_matrix[r * rows + i];
                    T grad_val = gradient[i * cols + c];
                    T feature = std::cos(rot_val * grad_val + bias[r]);
                    sum += feature * grad_val;
                }
                
                output[r * cols + c] = sum / std::sqrt(static_cast<T>(rows));
            }
        }
    } else {
        // Output shape: [rows, rank]
        // G_c = G @ φ(R) where φ(x) = cos(Rx + b)
        
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int i = 0; i < rows; ++i) {
            for (int r = 0; r < rank; ++r) {
                T sum = static_cast<T>(0);
                
                // Phase 96: Prefetch next row and rotation vector
                if (i + 1 < rows) {
                    saguaro::ops::PrefetchT0(gradient + (i + 1) * cols);
                }
                if (r + 1 < rank) {
                    saguaro::ops::PrefetchT1(rotation_matrix + (r + 1) * cols);
                }
                
                for (int c = 0; c < cols; ++c) {
                    T rot_val = rotation_matrix[r * cols + c];
                    T grad_val = gradient[i * cols + c];
                    T feature = std::cos(rot_val * grad_val + bias[r]);
                    sum += grad_val * feature;
                }
                
                output[i * rank + r] = sum / std::sqrt(static_cast<T>(cols));
            }
        }
    }
}

/**
 * @brief Deproject from low-rank space back to full gradient.
 *
 * Inverse of quantum random projection using adjoint feature map.
 *
 * @param compressed Compressed gradient [rank x cols] or [rows x rank]
 * @param rotation_matrix Same rotation parameters used in projection
 * @param bias Same bias values used in projection
 * @param output Reconstructed gradient [rows x cols]
 * @param rows Original number of rows
 * @param cols Original number of columns
 * @param rank Projection rank
 * @param row_projection If true, was projected along rows
 */
template <typename T>
void QuantumRandomDeproject(
    const T* compressed,
    const T* rotation_matrix,
    const T* bias,
    T* output,
    int rows,
    int cols,
    int rank,
    bool row_projection = true
) {
    if (row_projection) {
        // Input shape: [rank, cols], Output: [rows, cols]
        // G = φ(R) @ G_c
        
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int i = 0; i < rows; ++i) {
            for (int c = 0; c < cols; ++c) {
                T sum = static_cast<T>(0);
                
                for (int r = 0; r < rank; ++r) {
                    T rot_val = rotation_matrix[r * rows + i];
                    T comp_val = compressed[r * cols + c];
                    T feature = std::cos(rot_val + bias[r]);
                    sum += feature * comp_val;
                }
                
                output[i * cols + c] = sum / std::sqrt(static_cast<T>(rank));
            }
        }
    } else {
        // Input shape: [rows, rank], Output: [rows, cols]
        // G = G_c @ φ(R)^T
        
        #ifdef _OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
        #endif
        for (int i = 0; i < rows; ++i) {
            for (int c = 0; c < cols; ++c) {
                T sum = static_cast<T>(0);
                
                for (int r = 0; r < rank; ++r) {
                    T rot_val = rotation_matrix[r * cols + c];
                    T comp_val = compressed[i * rank + r];
                    T feature = std::cos(rot_val + bias[r]);
                    sum += comp_val * feature;
                }
                
                output[i * cols + c] = sum / std::sqrt(static_cast<T>(rank));
            }
        }
    }
}

/**
 * @brief Initialize quantum random feature parameters.
 *
 * Uses Gaussian random initialization scaled for quantum feature stability.
 *
 * @param rotation_matrix Output rotation parameters
 * @param bias Output bias values
 * @param rotation_size Size of rotation matrix
 * @param bias_size Size of bias vector
 * @param seed Random seed
 * @param scale Scaling factor (default: 1.0 / sqrt(dim))
 */
template <typename T>
void InitQuantumRandomFeatures(
    T* rotation_matrix,
    T* bias,
    int rotation_size,
    int bias_size,
    uint32_t seed,
    T scale = static_cast<T>(-1)
) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 2.0f * M_PI);
    
    // Auto-scale if not specified
    if (scale < static_cast<T>(0)) {
        scale = static_cast<T>(1) / std::sqrt(static_cast<T>(bias_size));
    }
    
    for (int i = 0; i < rotation_size; ++i) {
        rotation_matrix[i] = static_cast<T>(normal(rng)) * scale;
    }
    
    for (int i = 0; i < bias_size; ++i) {
        bias[i] = static_cast<T>(uniform(rng));
    }
}

/**
 * @brief Full Quantum GaLore forward projection with dynamic rank.
 *
 * Combines entropy-based rank selection with quantum random projection.
 *
 * @param gradient Input gradient [rows x cols]
 * @param eigenvalues Pre-computed singular values (or nullptr for auto-compute)
 * @param num_eigenvalues Number of eigenvalues
 * @param rotation_matrix Quantum random rotation parameters
 * @param bias Quantum random bias values
 * @param output Compressed gradient output
 * @param rows Number of rows
 * @param cols Number of columns
 * @param max_rank Maximum allowable rank
 * @param min_rank Minimum rank
 * @return Actual rank used for compression
 */
template <typename T>
int QuantumGaLoreProjectFull(
    const T* gradient,
    const T* eigenvalues,
    int num_eigenvalues,
    const T* rotation_matrix,
    const T* bias,
    T* output,
    int rows,
    int cols,
    int max_rank,
    int min_rank = 1
) {
    // Compute effective rank from eigenvalue spectrum
    int effective_rank = ComputeEffectiveRank(
        eigenvalues, num_eigenvalues, max_rank, min_rank
    );
    
    // Determine projection direction (tall vs wide matrix)
    bool project_rows = (rows >= cols);
    
    // Apply quantum random projection
    QuantumRandomProjection(
        gradient,
        rotation_matrix,
        bias,
        output,
        rows,
        cols,
        effective_rank,
        project_rows
    );
    
    return effective_rank;
}

}  // namespace quantum_galore
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_QUANTUM_GALORE_OP_H_
