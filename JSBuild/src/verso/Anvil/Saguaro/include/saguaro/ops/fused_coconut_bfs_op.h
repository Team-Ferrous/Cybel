// saguaro/native/ops/fused_coconut_bfs_op.h
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
 * @file fused_coconut_bfs_op.h
 * @brief Phase 87: CoCoNut Multi-path BFS Exploration SIMD Helpers.
 *
 * Implements core operations for multi-path BFS thought exploration:
 *   - Path expansion from single hidden state to N parallel paths
 *   - Simultaneous thought evolution across all paths
 *   - Grover-inspired amplitude scoring for path selection
 *   - Top-k pruning of low-amplitude paths
 *
 * SIMD optimizations:
 * - AVX512: 16-wide vectorization
 * - AVX2: 8-wide vectorization
 * - NEON: 4-wide vectorization (ARM)
 * - Scalar fallback for all architectures
 *
 * Complexity: O(k * num_paths * d²) where k=thought_steps, d=dim
 *
 * @note Thread-safe. All functions are reentrant with no shared state.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_COCONUT_BFS_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_COCONUT_BFS_OP_H_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

// Include shared SIMD library for common operations
#include "hnn_simd_common.h"
#include "fused_fft_projector_op.h"
#include "common/tensor_stream_pool.h"  // Phase 3: Zero-copy inter-kernel streaming

#if defined(__AVX512F__)
#include <immintrin.h>
#define COCONUT_SIMD_WIDTH 16
#elif defined(__AVX2__)
#include <immintrin.h>
#define COCONUT_SIMD_WIDTH 8
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define COCONUT_SIMD_WIDTH 4
#else
#define COCONUT_SIMD_WIDTH 1
#endif

namespace saguaro {
namespace ops {

// =============================================================================
// PHASE V2.0-P1.5: SCRATCH BUFFER POOLING INTEGRATION
// =============================================================================
// Uses thread-local PathScratchPool from hnn_simd_common.h to reduce
// memory allocation overhead in COCONUT BFS operations.
//
// Key functions that benefit from pooling:
// - coconut_evolve_paths(): normalized buffer
// - coconut_prune_paths(): indices buffer
//
// IMPORTANT: COCONUT amplitudes are gradient-connected via Grover rotation.
// Scratch pooling only applies to intermediate buffers, NOT amplitudes.
//
// See HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 11.6 (P-1.1)
// =============================================================================

// Import scratch pool from hnn_simd_common.h
using ::g_path_scratch;
using ::g_path_scratch_secondary;

// =============================================================================
// COMMON HELPERS (used by coconut and other ops)
// =============================================================================

/**
 * @brief Mean pool hidden states along sequence dimension.
 *
 * Computes: output[b, d] = mean(input[b, :, d])
 *
 * @param input Input hidden states [batch, seq_len, dim]
 * @param output Output mean-pooled states [batch, dim]
 * @param batch_size Batch dimension
 * @param seq_len Sequence length
 * @param dim Hidden dimension
 */
inline void continuous_thought_mean_pool(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t seq_len,
    int64_t dim) {

    const float inv_seq = 1.0f / static_cast<float>(seq_len);

    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_in = input + b * seq_len * dim;
        float* batch_out = output + b * dim;

        // Initialize output to zero
        for (int64_t d = 0; d < dim; ++d) {
            batch_out[d] = 0.0f;
        }

        // Sum across sequence
        for (int64_t s = 0; s < seq_len; ++s) {
            const float* seq_in = batch_in + s * dim;
            int64_t d = 0;
#if defined(__AVX512F__)
            for (; d + 16 <= dim; d += 16) {
                __m512 ov = _mm512_loadu_ps(&batch_out[d]);
                __m512 iv = _mm512_loadu_ps(&seq_in[d]);
                _mm512_storeu_ps(&batch_out[d], _mm512_add_ps(ov, iv));
            }
#elif defined(__AVX2__)
            for (; d + 8 <= dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(&batch_out[d]);
                __m256 iv = _mm256_loadu_ps(&seq_in[d]);
                _mm256_storeu_ps(&batch_out[d], _mm256_add_ps(ov, iv));
            }
#elif defined(__ARM_NEON)
            for (; d + 4 <= dim; d += 4) {
                float32x4_t ov = vld1q_f32(&batch_out[d]);
                float32x4_t iv = vld1q_f32(&seq_in[d]);
                vst1q_f32(&batch_out[d], vaddq_f32(ov, iv));
            }
#endif
            for (; d < dim; ++d) {
                batch_out[d] += seq_in[d];
            }
        }

        // Divide by sequence length
        int64_t d = 0;
#if defined(__AVX512F__)
        __m512 inv_v = _mm512_set1_ps(inv_seq);
        for (; d + 16 <= dim; d += 16) {
            __m512 ov = _mm512_loadu_ps(&batch_out[d]);
            _mm512_storeu_ps(&batch_out[d], _mm512_mul_ps(ov, inv_v));
        }
#elif defined(__AVX2__)
        __m256 inv_v = _mm256_set1_ps(inv_seq);
        for (; d + 8 <= dim; d += 8) {
            __m256 ov = _mm256_loadu_ps(&batch_out[d]);
            _mm256_storeu_ps(&batch_out[d], _mm256_mul_ps(ov, inv_v));
        }
#elif defined(__ARM_NEON)
        float32x4_t inv_v = vdupq_n_f32(inv_seq);
        for (; d + 4 <= dim; d += 4) {
            float32x4_t ov = vld1q_f32(&batch_out[d]);
            vst1q_f32(&batch_out[d], vmulq_f32(ov, inv_v));
        }
#endif
        for (; d < dim; ++d) {
            batch_out[d] *= inv_seq;
        }
    }
}

namespace coconut {

// =============================================================================
// PATH EXPANSION: hidden_state -> N parallel thought paths
// =============================================================================

/**
 * @brief Expand a single hidden state into N parallel thought paths.
 *
 * Each path starts from the same hidden state but will be evolved
 * independently. Adds small noise for path diversity.
 *
 * @param hidden_state Input hidden state [batch, dim]
 * @param paths Output paths [batch, num_paths, dim]
 * @param batch_size Batch dimension
 * @param num_paths Number of parallel paths to create
 * @param dim Hidden dimension
 * @param noise_scale Scale of diversity noise (default 0.01)
 */
inline void coconut_expand_paths(
    const float* hidden_state,
    float* paths,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    float noise_scale = 0.01f) {
    
    #pragma omp parallel for collapse(2)
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t p = 0; p < num_paths; ++p) {
            const float* src = hidden_state + b * dim;
            float* dst = paths + (b * num_paths + p) * dim;
            
            // Copy hidden state with small path-specific noise
            // Use deterministic noise based on path index for reproducibility
            float path_offset = static_cast<float>(p) / static_cast<float>(num_paths);
            
            int64_t d = 0;
#if defined(__AVX512F__)
            __m512 noise_base = _mm512_set1_ps(path_offset * noise_scale);
            for (; d + 16 <= dim; d += 16) {
                __m512 x = _mm512_loadu_ps(&src[d]);
                // Add small deterministic perturbation
                __m512 noise = _mm512_mul_ps(noise_base, 
                    _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));
                _mm512_storeu_ps(&dst[d], _mm512_add_ps(x, noise));
            }
#elif defined(__AVX2__)
            __m256 noise_base = _mm256_set1_ps(path_offset * noise_scale);
            for (; d + 8 <= dim; d += 8) {
                __m256 x = _mm256_loadu_ps(&src[d]);
                __m256 noise = _mm256_mul_ps(noise_base,
                    _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0));
                _mm256_storeu_ps(&dst[d], _mm256_add_ps(x, noise));
            }
#elif defined(__ARM_NEON)
            float32x4_t noise_base = vdupq_n_f32(path_offset * noise_scale);
            float noise_mult[4] = {0, 1, 2, 3};
            float32x4_t noise_mult_v = vld1q_f32(noise_mult);
            for (; d + 4 <= dim; d += 4) {
                float32x4_t x = vld1q_f32(&src[d]);
                float32x4_t noise = vmulq_f32(noise_base, noise_mult_v);
                vst1q_f32(&dst[d], vaddq_f32(x, noise));
            }
#endif
            // Scalar fallback
            for (; d < dim; ++d) {
                float noise = path_offset * noise_scale * static_cast<float>(d % 16);
                dst[d] = src[d] + noise;
            }
        }
    }
}

// =============================================================================
// PARALLEL PATH EVOLUTION: Apply thought projector to all paths
// =============================================================================

/**
 * @brief Evolve all thought paths through a single projector step.
 *
 * Applies: path = path + MLP(LayerNorm(path)) for each path.
 * This is the core thought step, run in parallel across all paths.
 *
 * @param paths Input/output paths [batch, num_paths, dim]
 * @param norm_gamma LayerNorm gamma [dim]
 * @param norm_beta LayerNorm beta [dim]
 * @param dense1_weight First dense weight [dim, hidden_dim]
 * @param dense1_bias First dense bias [hidden_dim]
 * @param dense2_weight Second dense weight [hidden_dim, dim]
 * @param dense2_bias Second dense bias [dim]
 * @param batch_size Batch dimension
 * @param num_paths Number of paths
 * @param dim Hidden dimension
 * @param hidden_dim MLP hidden dimension
 * @param work_buffer Scratch space [batch * num_paths * hidden_dim]
 */

inline void coconut_evolve_paths(
    float* paths,
    const float* norm_gamma,
    const float* norm_beta,
    const float* dense1_weight,
    const float* dense1_bias,
    const float* dense2_weight,
    const float* dense2_bias,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    int64_t hidden_dim,
    float* work_buffer,
    bool use_fft = false,
    bool input_is_freq = false,
    bool output_is_freq = false,
    int64_t path_stride = 0) {
    
    const int64_t total_paths = batch_size * num_paths;
    // Default stride to dim if not specified
    if (path_stride == 0) path_stride = dim;
    
    if (use_fft) {
        // UQHA Phase 2.1 + Phase 2.2: FFT-based path evolution with persistent state
        // When input_is_freq=true: paths are already in frequency domain
        // When output_is_freq=true: keep output in frequency domain
        // This eliminates k-2 FFT/IFFT pairs during multi-step reasoning
        fft_projector_forward(
            paths,
            dense1_weight,
            dense1_bias,
            dense2_weight,
            dense2_bias,
            norm_gamma,
            norm_beta,
            total_paths,
            dim,
            input_is_freq,
            output_is_freq,
            path_stride
        );
        return;
    }
    
    // Legacy Dense Path (O(D²))
    // V2.0-P1.5: Use scratch pool for normalized buffer
    const size_t normalized_size = total_paths * dim;
    float* normalized = g_path_scratch.get(normalized_size);
    
    // Step 1: LayerNorm across all paths
    saguaro::ops::simd_layernorm(paths, norm_gamma, norm_beta, normalized, 
                   total_paths, dim, 1e-6f);
    
    // Step 2: Dense1 + GELU for each path
    #pragma omp parallel for
    for (int64_t idx = 0; idx < total_paths; ++idx) {
        const float* norm_row = normalized + idx * dim;
        float* hidden_row = work_buffer + idx * hidden_dim;
        
        // Matrix-vector: hidden = normalized @ dense1_weight + dense1_bias
        for (int64_t h = 0; h < hidden_dim; ++h) {
            float sum = dense1_bias[h];
            for (int64_t d = 0; d < dim; ++d) {
                sum += norm_row[d] * dense1_weight[d * hidden_dim + h];
            }
            hidden_row[h] = sum;
        }
    }
    
    // GELU activation in-place
    saguaro::ops::simd_gelu_inplace(work_buffer, total_paths * hidden_dim);
    
    // Step 3: Dense2 + residual connection
    #pragma omp parallel for
    for (int64_t idx = 0; idx < total_paths; ++idx) {
        const float* hidden_row = work_buffer + idx * hidden_dim;
        float* path_row = paths + idx * dim;
        
        for (int64_t d = 0; d < dim; ++d) {
            float sum = dense2_bias[d];
            for (int64_t h = 0; h < hidden_dim; ++h) {
                sum += hidden_row[h] * dense2_weight[h * dim + d];
            }
            // Residual connection
            path_row[d] += sum;
        }
    }
}

// =============================================================================
// AMPLITUDE SCORING: Grover-inspired path quality scoring
// =============================================================================

/**
 * @brief Compute amplitude scores for each path using Grover-inspired scoring.
 *
 * Scores are based on semantic coherence with the original context.
 * Higher amplitude = more promising path.
 *
 * @param paths Current path states [batch, num_paths, dim]
 * @param context Original context [batch, dim]
 * @param amplitudes Output amplitudes [batch, num_paths]
 * @param batch_size Batch dimension
 * @param num_paths Number of paths
 * @param dim Hidden dimension
 */
inline void coconut_amplitude_score(
    const float* paths,
    const float* context,
    float* amplitudes,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* ctx = context + b * dim;
        
        // Compute context norm
        float ctx_norm_sq = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            ctx_norm_sq += ctx[d] * ctx[d];
        }
        float ctx_norm = std::sqrt(ctx_norm_sq + 1e-8f);
        
        // Score each path by cosine similarity with context
        float max_score = -1e9f;
        for (int64_t p = 0; p < num_paths; ++p) {
            const float* path = paths + (b * num_paths + p) * dim;
            
            float dot = 0.0f;
            float path_norm_sq = 0.0f;
            
            int64_t d = 0;
#if defined(__AVX512F__)
            __m512 dot_acc = _mm512_setzero_ps();
            __m512 norm_acc = _mm512_setzero_ps();
            for (; d + 16 <= dim; d += 16) {
                __m512 pv = _mm512_loadu_ps(&path[d]);
                __m512 cv = _mm512_loadu_ps(&ctx[d]);
                dot_acc = _mm512_fmadd_ps(pv, cv, dot_acc);
                norm_acc = _mm512_fmadd_ps(pv, pv, norm_acc);
            }
            dot = _mm512_reduce_add_ps(dot_acc);
            path_norm_sq = _mm512_reduce_add_ps(norm_acc);
#elif defined(__AVX2__)
            __m256 dot_acc = _mm256_setzero_ps();
            __m256 norm_acc = _mm256_setzero_ps();
            for (; d + 8 <= dim; d += 8) {
                __m256 pv = _mm256_loadu_ps(&path[d]);
                __m256 cv = _mm256_loadu_ps(&ctx[d]);
                dot_acc = _mm256_fmadd_ps(pv, cv, dot_acc);
                norm_acc = _mm256_fmadd_ps(pv, pv, norm_acc);
            }
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(dot_acc, 1);
            __m128 lo = _mm256_castps256_ps128(dot_acc);
            __m128 sum = _mm_add_ps(hi, lo);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            dot = _mm_cvtss_f32(sum);
            
            hi = _mm256_extractf128_ps(norm_acc, 1);
            lo = _mm256_castps256_ps128(norm_acc);
            sum = _mm_add_ps(hi, lo);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            path_norm_sq = _mm_cvtss_f32(sum);
#elif defined(__ARM_NEON)
            float32x4_t dot_acc = vdupq_n_f32(0);
            float32x4_t norm_acc = vdupq_n_f32(0);
            for (; d + 4 <= dim; d += 4) {
                float32x4_t pv = vld1q_f32(&path[d]);
                float32x4_t cv = vld1q_f32(&ctx[d]);
                dot_acc = vmlaq_f32(dot_acc, pv, cv);
                norm_acc = vmlaq_f32(norm_acc, pv, pv);
            }
            float32x2_t d_sum = vadd_f32(vget_low_f32(dot_acc), vget_high_f32(dot_acc));
            d_sum = vpadd_f32(d_sum, d_sum);
            dot = vget_lane_f32(d_sum, 0);
            
            float32x2_t n_sum = vadd_f32(vget_low_f32(norm_acc), vget_high_f32(norm_acc));
            n_sum = vpadd_f32(n_sum, n_sum);
            path_norm_sq = vget_lane_f32(n_sum, 0);
#endif
            // Scalar remainder
            for (; d < dim; ++d) {
                dot += path[d] * ctx[d];
                path_norm_sq += path[d] * path[d];
            }
            
            float path_norm = std::sqrt(path_norm_sq + 1e-8f);
            float score = dot / (path_norm * ctx_norm);
            amplitudes[b * num_paths + p] = score;
            max_score = std::max(max_score, score);
        }
        
        // Normalize to [0, 1] range and apply softmax-like scaling
        float sum_exp = 0.0f;
        for (int64_t p = 0; p < num_paths; ++p) {
            float& amp = amplitudes[b * num_paths + p];
            amp = std::exp(amp - max_score);  // Numerical stability
            sum_exp += amp;
        }
        for (int64_t p = 0; p < num_paths; ++p) {
            amplitudes[b * num_paths + p] /= sum_exp;
        }
    }
}

// =============================================================================
// PATH PRUNING: Keep top-k paths based on amplitude
// =============================================================================

/**
 * @brief Prune low-amplitude paths, keeping only top-k.
 *
 * @param paths Input paths [batch, num_paths, dim]
 * @param amplitudes Path amplitudes [batch, num_paths]
 * @param pruned_paths Output pruned paths [batch, k, dim]
 * @param pruned_indices Output indices of kept paths [batch, k]
 * @param batch_size Batch dimension
 * @param num_paths Number of input paths
 * @param dim Hidden dimension
 * @param k Number of paths to keep
 */
inline void coconut_prune_paths(
    const float* paths,
    const float* amplitudes,
    float* pruned_paths,
    int32_t* pruned_indices,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    int64_t k) {
    
    k = std::min(k, num_paths);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_amps = amplitudes + b * num_paths;
        
        // Create indices and sort by amplitude (descending)
        std::vector<int32_t> indices(num_paths);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [batch_amps](int32_t a, int32_t b) {
                return batch_amps[a] > batch_amps[b];
            });
        
        // Copy top-k paths
        for (int64_t i = 0; i < k; ++i) {
            int32_t src_idx = indices[i];
            pruned_indices[b * k + i] = src_idx;
            
            const float* src = paths + (b * num_paths + src_idx) * dim;
            float* dst = pruned_paths + (b * k + i) * dim;
            std::copy(src, src + dim, dst);
        }
    }
}

// =============================================================================
// PATH AGGREGATION: Combine paths into single output
// =============================================================================

/**
 * @brief Aggregate multiple paths into single output using amplitude weighting.
 *
 * @param paths Input paths [batch, num_paths, dim]
 * @param amplitudes Path amplitudes [batch, num_paths]
 * @param output Output aggregated state [batch, dim]
 * @param batch_size Batch dimension
 * @param num_paths Number of paths
 * @param dim Hidden dimension
 */
inline void coconut_aggregate_paths(
    const float* paths,
    const float* amplitudes,
    float* output,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        float* out = output + b * dim;
        
        // Initialize output to zero
        for (int64_t d = 0; d < dim; ++d) {
            out[d] = 0.0f;
        }
        
        // Weighted sum of paths
        for (int64_t p = 0; p < num_paths; ++p) {
            float amp = amplitudes[b * num_paths + p];
            const float* path = paths + (b * num_paths + p) * dim;
            
            int64_t d = 0;
#if defined(__AVX512F__)
            __m512 amp_v = _mm512_set1_ps(amp);
            for (; d + 16 <= dim; d += 16) {
                __m512 ov = _mm512_loadu_ps(&out[d]);
                __m512 pv = _mm512_loadu_ps(&path[d]);
                _mm512_storeu_ps(&out[d], _mm512_fmadd_ps(amp_v, pv, ov));
            }
#elif defined(__AVX2__)
            __m256 amp_v = _mm256_set1_ps(amp);
            for (; d + 8 <= dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(&out[d]);
                __m256 pv = _mm256_loadu_ps(&path[d]);
                _mm256_storeu_ps(&out[d], _mm256_fmadd_ps(amp_v, pv, ov));
            }
#elif defined(__ARM_NEON)
            float32x4_t amp_v = vdupq_n_f32(amp);
            for (; d + 4 <= dim; d += 4) {
                float32x4_t ov = vld1q_f32(&out[d]);
                float32x4_t pv = vld1q_f32(&path[d]);
                vst1q_f32(&out[d], vmlaq_f32(ov, amp_v, pv));
            }
#endif
            for (; d < dim; ++d) {
                out[d] += amp * path[d];
            }
        }
    }
}

// =============================================================================
// BACKWARD HELPERS: Analytic gradients for training support
// =============================================================================

/**
 * @brief Backward pass for path aggregation.
 *
 * Forward: output[d] = sum_p(amp[p] * path[p,d])
 * Backward: grad_path[p,d] = amp[p] * grad_output[d]
 *           grad_amp[p] = sum_d(path[p,d] * grad_output[d])
 */
inline void coconut_aggregate_paths_backward(
    const float* grad_output,  // [batch, dim]
    const float* paths,        // [batch, num_paths, dim]
    const float* amplitudes,   // [batch, num_paths]
    float* grad_paths,         // [batch, num_paths, dim]
    float* grad_amplitudes,    // [batch, num_paths]
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* g_out = grad_output + b * dim;
        
        for (int64_t p = 0; p < num_paths; ++p) {
            float amp = amplitudes[b * num_paths + p];
            const float* path = paths + (b * num_paths + p) * dim;
            float* g_path = grad_paths + (b * num_paths + p) * dim;
            
            // grad_path = amp * grad_output
            float dot = 0.0f;
            for (int64_t d = 0; d < dim; ++d) {
                g_path[d] = amp * g_out[d];
                dot += path[d] * g_out[d];
            }
            // grad_amp = sum(path * grad_output)
            grad_amplitudes[b * num_paths + p] = dot;
        }
    }
}

/**
 * @brief Backward pass for Dense layer.
 *
 * Forward: y = x @ W + b
 * Backward: grad_x = grad_y @ W^T
 *           grad_W = x^T @ grad_y
 *           grad_b = sum(grad_y, axis=0)
 */
inline void dense_backward(
    const float* grad_y,      // [batch, out_dim]
    const float* x,           // [batch, in_dim]
    const float* weight,      // [in_dim, out_dim]
    float* grad_x,            // [batch, in_dim]
    float* grad_weight,       // [in_dim, out_dim]
    float* grad_bias,         // [out_dim]
    int64_t batch_size,
    int64_t in_dim,
    int64_t out_dim) {
    
    // Initialize grad_weight and grad_bias to zero
    std::fill(grad_weight, grad_weight + in_dim * out_dim, 0.0f);
    std::fill(grad_bias, grad_bias + out_dim, 0.0f);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* g_y = grad_y + b * out_dim;
        const float* x_row = x + b * in_dim;
        float* g_x = grad_x + b * in_dim;
        
        // grad_x = grad_y @ W^T
        for (int64_t i = 0; i < in_dim; ++i) {
            float sum = 0.0f;
            for (int64_t o = 0; o < out_dim; ++o) {
                sum += g_y[o] * weight[i * out_dim + o];
            }
            g_x[i] = sum;
        }
        
        // Accumulate grad_W and grad_b (need atomic or reduction)
        #pragma omp critical
        {
            for (int64_t i = 0; i < in_dim; ++i) {
                for (int64_t o = 0; o < out_dim; ++o) {
                    grad_weight[i * out_dim + o] += x_row[i] * g_y[o];
                }
            }
            for (int64_t o = 0; o < out_dim; ++o) {
                grad_bias[o] += g_y[o];
            }
        }
    }
}

/**
 * @brief Backward pass for LayerNorm.
 *
 * Computes gradients for gamma, beta, and input.
 */
inline void layernorm_backward(
    const float* grad_output,  // [batch, dim]
    const float* input,        // [batch, dim]
    const float* gamma,        // [dim]
    float* grad_input,         // [batch, dim]
    float* grad_gamma,         // [dim]
    float* grad_beta,          // [dim]
    int64_t batch_size,
    int64_t dim) {
    
    // Initialize gradients
    std::fill(grad_gamma, grad_gamma + dim, 0.0f);
    std::fill(grad_beta, grad_beta + dim, 0.0f);
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* g_out = grad_output + b * dim;
        const float* x = input + b * dim;
        float* g_x = grad_input + b * dim;
        
        // Compute mean and variance
        float mean = 0.0f, var = 0.0f;
        for (int64_t d = 0; d < dim; ++d) mean += x[d];
        mean /= dim;
        for (int64_t d = 0; d < dim; ++d) {
            float diff = x[d] - mean;
            var += diff * diff;
        }
        var = var / dim + 1e-6f;
        float inv_std = 1.0f / std::sqrt(var);
        
        // Compute normalized values and gradients
        float d_var = 0.0f, d_mean = 0.0f;
        for (int64_t d = 0; d < dim; ++d) {
            float x_hat = (x[d] - mean) * inv_std;
            
            #pragma omp atomic
            grad_gamma[d] += g_out[d] * x_hat;
            #pragma omp atomic
            grad_beta[d] += g_out[d];
            
            float d_xhat = g_out[d] * gamma[d];
            d_var += d_xhat * (x[d] - mean) * (-0.5f) * inv_std * inv_std * inv_std;
            d_mean += d_xhat * (-inv_std);
        }
        
        d_mean += d_var * (-2.0f / dim) * (0.0f);  // sum(x - mean) = 0
        
        // Compute grad_input
        for (int64_t d = 0; d < dim; ++d) {
            float d_xhat = g_out[d] * gamma[d];
            g_x[d] = d_xhat * inv_std + d_var * 2.0f * (x[d] - mean) / dim + d_mean / dim;
        }
    }
}

/**
 * @brief Backward pass for broadcast projection (add to sequence).
 *
 * Forward: output[b,s,d] = input[b,s,d] + projected[b,d]
 * Backward: grad_input = grad_output
 *           grad_projected = sum(grad_output, axis=seq)
 */
inline void broadcast_backward(
    const float* grad_output,    // [batch, seq_len, dim]
    float* grad_input,           // [batch, seq_len, dim]
    float* grad_projected,       // [batch, dim]
    int64_t batch_size,
    int64_t seq_len,
    int64_t dim) {
    
    #pragma omp parallel for
    for (int64_t b = 0; b < batch_size; ++b) {
        float* g_proj = grad_projected + b * dim;
        std::fill(g_proj, g_proj + dim, 0.0f);
        
        for (int64_t s = 0; s < seq_len; ++s) {
            const float* g_out = grad_output + (b * seq_len + s) * dim;
            float* g_in = grad_input + (b * seq_len + s) * dim;
            
            for (int64_t d = 0; d < dim; ++d) {
                g_in[d] = g_out[d];
                g_proj[d] += g_out[d];
            }
        }
    }
}

}  // namespace coconut

// =============================================================================
// PHASE 3: STREAMING COCONUT BFS FORWARD (TensorStreamPool Integration)
// =============================================================================
// Zero-copy streaming variant that uses TensorStreamPool for path buffers.
// Eliminates memory copy overhead between expand → evolve → aggregate stages.

/**
 * @brief Streaming COCONUT BFS forward pass with TensorStreamPool integration.
 *
 * Performs full thought iteration: expand → evolve × k → score → aggregate.
 * Uses TensorStreamPool for inter-stage buffer handoff (zero-copy).
 *
 * @param context Input context [batch, dim]
 * @param output Output aggregated thought [batch, dim]
 * @param norm_gamma LayerNorm gamma
 * @param norm_beta LayerNorm beta
 * @param dense1_weight First MLP weight [dim, hidden_dim]
 * @param dense1_bias First MLP bias
 * @param dense2_weight Second MLP weight [hidden_dim, dim]
 * @param dense2_bias Second MLP bias
 * @param config BFS configuration
 * @param use_streaming If true, use TensorStreamPool (default: true)
 */
inline void coconut_bfs_forward_streaming(
    const float* context,
    float* output,
    const float* norm_gamma,
    const float* norm_beta,
    const float* dense1_weight,
    const float* dense1_bias,
    const float* dense2_weight,
    const float* dense2_bias,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    int64_t hidden_dim,
    int thought_steps = 1,
    bool use_streaming = true
) {
    using namespace hsmn::ops;
    
    // Calculate buffer sizes
    size_t paths_size = batch_size * num_paths * dim * sizeof(float);
    size_t amplitudes_size = batch_size * num_paths * sizeof(float);
    size_t work_buffer_size = batch_size * num_paths * hidden_dim * sizeof(float);
    
    float* paths = nullptr;
    float* amplitudes = nullptr;
    float* work_buffer = nullptr;
    
    if (use_streaming) {
        // Stage 1: Acquire paths buffer from pool
        paths = GetTensorStreamPool().Acquire(paths_size, "coconut_expand");
        amplitudes = GetTensorStreamPool().Acquire(amplitudes_size, "coconut_amplitudes");
        work_buffer = GetTensorStreamPool().Acquire(work_buffer_size, "coconut_evolve");
    } else {
        // Fallback: use PathScratchPool
        paths = g_path_scratch.get(batch_size * num_paths * dim);
        amplitudes = g_path_scratch_secondary.get(batch_size * num_paths);
        work_buffer = g_path_scratch.get(batch_size * num_paths * hidden_dim);
    }
    
    if (!paths || !amplitudes || !work_buffer) {
        // Emergency fallback to heap
        std::vector<float> paths_vec(batch_size * num_paths * dim);
        std::vector<float> amp_vec(batch_size * num_paths);
        std::vector<float> work_vec(batch_size * num_paths * hidden_dim);
        paths = paths_vec.data();
        amplitudes = amp_vec.data();
        work_buffer = work_vec.data();
        use_streaming = false;  // Can't handoff heap memory
    }
    
    // Stage 1: Expand paths from context
    coconut::coconut_expand_paths(context, paths, batch_size, num_paths, dim);
    
    if (use_streaming) {
        GetTensorStreamPool().Handoff(paths, "coconut_evolve");
    }
    
    // Stage 2: Evolve paths through thought iterations
    for (int step = 0; step < thought_steps; ++step) {
        coconut::coconut_evolve_paths(
            paths, norm_gamma, norm_beta,
            dense1_weight, dense1_bias,
            dense2_weight, dense2_bias,
            batch_size, num_paths, dim, hidden_dim,
            work_buffer
        );
    }
    
    if (use_streaming) {
        GetTensorStreamPool().Handoff(paths, "coconut_score");
    }
    
    // Stage 3: Score paths using amplitude coherence with context
    coconut::coconut_amplitude_score(paths, context, amplitudes, batch_size, num_paths, dim);
    
    if (use_streaming) {
        GetTensorStreamPool().Handoff(amplitudes, "coconut_aggregate");
    }
    
    // Stage 4: Aggregate paths weighted by amplitudes
    coconut::coconut_aggregate_paths(paths, amplitudes, output, batch_size, num_paths, dim);
    
    // Release buffers back to pool
    if (use_streaming) {
        GetTensorStreamPool().Release(work_buffer);
        GetTensorStreamPool().Release(amplitudes);
        GetTensorStreamPool().Release(paths);
    }
}

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_COCONUT_BFS_OP_H_
