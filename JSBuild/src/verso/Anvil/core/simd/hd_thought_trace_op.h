// highnoon/_native/ops/hd_thought_trace_op.h
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
 * @file hd_thought_trace_op.h
 * @brief Phase 300+: HD thought trace storage for COCONUT reasoning.
 *
 * hd_upgrade.md Phase 4 - HD MoE + Reasoning Enhancements.
 *
 * Stores thought evolution in HD superposition for retrievable history.
 * This enables COCONUT's continuous thought blocks to maintain a
 * compressed record of the reasoning process.
 *
 * Key operations:
 * - HDThoughtTraceUpdate: Add new thought to accumulated trace
 * - HDThoughtTraceRetrieve: Retrieve specific step via unbinding
 * - HDThoughtTracePath: Score multiple paths for BFS selection
 */

#ifndef HIGHNOON_NATIVE_OPS_HD_THOUGHT_TRACE_OP_H_
#define HIGHNOON_NATIVE_OPS_HD_THOUGHT_TRACE_OP_H_

#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

namespace hsmn {
namespace hd_thought {

/**
 * HD Thought Trace Configuration.
 */
struct HDThoughtConfig {
    int hd_dim = 4096;           // HD dimension
    float epsilon = 1e-8f;       // Numerical stability
    int max_steps = 64;          // Maximum reasoning steps
};

// Reuse FFT
inline void fft_thought(float* data, int n, bool inverse = false) {
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[2*i], data[2*j]);
            std::swap(data[2*i + 1], data[2*j + 1]);
        }
        int k = n >> 1;
        while (k <= j) { j -= k; k >>= 1; }
        j += k;
    }

    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float angle = sign * 2.0f * M_PI / static_cast<float>(len);
        float wlen_re = std::cos(angle);
        float wlen_im = std::sin(angle);

        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int jj = 0; jj < len / 2; ++jj) {
                int u_idx = 2 * (i + jj);
                int v_idx = 2 * (i + jj + len / 2);
                float u_re = data[u_idx], u_im = data[u_idx + 1];
                float v_re = data[v_idx], v_im = data[v_idx + 1];
                float tv_re = v_re * w_re - v_im * w_im;
                float tv_im = v_re * w_im + v_im * w_re;
                data[u_idx] = u_re + tv_re;
                data[u_idx + 1] = u_im + tv_im;
                data[v_idx] = u_re - tv_re;
                data[v_idx + 1] = u_im - tv_im;
                float nw_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
            }
        }
    }
    if (inverse) {
        float scale = 1.0f / static_cast<float>(n);
        for (int i = 0; i < 2 * n; ++i) data[i] *= scale;
    }
}

/**
 * Generate step key for thought trace (deterministic).
 *
 * @param step_key Output step key [hd_dim]
 * @param step Step index
 * @param hd_dim HD dimension
 */
inline void generate_step_key(
    float* step_key,
    int step,
    int hd_dim
) {
    // Use same Floquet-inspired encoding as position keys
    for (int d = 0; d < hd_dim; ++d) {
        float freq = 1.0f / std::pow(10000.0f, 
            2.0f * static_cast<float>(d / 2) / static_cast<float>(hd_dim));
        
        if (d % 2 == 0) {
            step_key[d] = std::sin(step * freq);
        } else {
            step_key[d] = std::cos(step * freq);
        }
    }
}

/**
 * Holographic bind for thought trace.
 */
inline void hd_bind_thought(
    const float* x,
    const float* key,
    float* result,
    int dim
) {
    std::vector<float> x_freq(2 * dim), k_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        x_freq[2*i] = x[i]; x_freq[2*i + 1] = 0.0f;
        k_freq[2*i] = key[i]; k_freq[2*i + 1] = 0.0f;
    }
    
    fft_thought(x_freq.data(), dim, false);
    fft_thought(k_freq.data(), dim, false);
    
    for (int i = 0; i < dim; ++i) {
        float xr = x_freq[2*i], xi = x_freq[2*i + 1];
        float kr = k_freq[2*i], ki = k_freq[2*i + 1];
        x_freq[2*i] = xr * kr - xi * ki;
        x_freq[2*i + 1] = xr * ki + xi * kr;
    }
    
    fft_thought(x_freq.data(), dim, true);
    
    for (int i = 0; i < dim; ++i) {
        result[i] = x_freq[2*i];
    }
}

/**
 * Holographic unbind for thought retrieval.
 */
inline void hd_unbind_thought(
    const float* bundle,
    const float* key,
    float* result,
    int dim,
    float epsilon = 1e-8f
) {
    std::vector<float> b_freq(2 * dim), k_freq(2 * dim);
    
    for (int i = 0; i < dim; ++i) {
        b_freq[2*i] = bundle[i]; b_freq[2*i + 1] = 0.0f;
        k_freq[2*i] = key[i]; k_freq[2*i + 1] = 0.0f;
    }
    
    fft_thought(b_freq.data(), dim, false);
    fft_thought(k_freq.data(), dim, false);
    
    for (int i = 0; i < dim; ++i) {
        float br = b_freq[2*i], bi = b_freq[2*i + 1];
        float kr = k_freq[2*i], ki = k_freq[2*i + 1];
        float denom = kr * kr + ki * ki + epsilon;
        
        b_freq[2*i] = (br * kr + bi * ki) / denom;
        b_freq[2*i + 1] = (bi * kr - br * ki) / denom;
    }
    
    fft_thought(b_freq.data(), dim, true);
    
    for (int i = 0; i < dim; ++i) {
        result[i] = b_freq[2*i];
    }
}

/**
 * Update thought trace with new thought.
 *
 * trace = trace + holographic_bind(thought, step_key)
 *
 * @param trace In/out thought trace [batch, hd_dim]
 * @param thought New thought [batch, hd_dim]
 * @param step Current step index
 * @param batch_size Batch size
 * @param hd_dim HD dimension
 */
inline void HDThoughtTraceUpdate(
    float* trace,
    const float* thought,
    int step,
    int batch_size,
    int hd_dim
) {
    std::vector<float> step_key(hd_dim);
    generate_step_key(step_key.data(), step, hd_dim);
    
    std::vector<float> bound(hd_dim);
    
    #pragma omp parallel for firstprivate(bound)
    for (int b = 0; b < batch_size; ++b) {
        const float* thought_ptr = thought + b * hd_dim;
        float* trace_ptr = trace + b * hd_dim;
        
        hd_bind_thought(thought_ptr, step_key.data(), bound.data(), hd_dim);
        
        for (int d = 0; d < hd_dim; ++d) {
            trace_ptr[d] += bound[d];
        }
    }
}

/**
 * Retrieve specific step from thought trace.
 *
 * retrieved = holographic_unbind(trace, step_key)
 *
 * @param trace Thought trace [batch, hd_dim]
 * @param retrieved Output retrieved thought [batch, hd_dim]
 * @param step Step index to retrieve
 * @param batch_size Batch size
 * @param hd_dim HD dimension
 * @param epsilon Numerical stability
 */
inline void HDThoughtTraceRetrieve(
    const float* trace,
    float* retrieved,
    int step,
    int batch_size,
    int hd_dim,
    float epsilon = 1e-8f
) {
    std::vector<float> step_key(hd_dim);
    generate_step_key(step_key.data(), step, hd_dim);
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* trace_ptr = trace + b * hd_dim;
        float* out_ptr = retrieved + b * hd_dim;
        
        hd_unbind_thought(trace_ptr, step_key.data(), out_ptr, hd_dim, epsilon);
    }
}

/**
 * Compute similarity between path candidates and target.
 *
 * Used for BFS path selection in COCONUT.
 *
 * @param paths Candidate paths [batch, num_paths, hd_dim]
 * @param target Target thought [batch, hd_dim]
 * @param scores Output similarity scores [batch, num_paths]
 * @param batch_size Batch size
 * @param num_paths Number of candidate paths
 * @param hd_dim HD dimension
 */
inline void HDThoughtPathScores(
    const float* paths,
    const float* target,
    float* scores,
    int batch_size,
    int num_paths,
    int hd_dim
) {
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int p = 0; p < num_paths; ++p) {
            const float* path_ptr = paths + (b * num_paths + p) * hd_dim;
            const float* target_ptr = target + b * hd_dim;
            
            // Cosine similarity
            float dot = 0.0f, norm_p = 0.0f, norm_t = 0.0f;
            for (int d = 0; d < hd_dim; ++d) {
                dot += path_ptr[d] * target_ptr[d];
                norm_p += path_ptr[d] * path_ptr[d];
                norm_t += target_ptr[d] * target_ptr[d];
            }
            
            float denom = std::sqrt(norm_p) * std::sqrt(norm_t) + 1e-8f;
            scores[b * num_paths + p] = dot / denom;
        }
    }
}

/**
 * Gradient of thought trace update w.r.t. inputs.
 *
 * @param grad_trace Gradient w.r.t. trace output [batch, hd_dim]
 * @param thought Forward pass thought [batch, hd_dim]
 * @param grad_thought Output gradient w.r.t. thought [batch, hd_dim]
 * @param step Step index
 * @param batch_size Batch size
 * @param hd_dim HD dimension
 */
inline void HDThoughtTraceUpdateGrad(
    const float* grad_trace,
    const float* thought,
    float* grad_thought,
    int step,
    int batch_size,
    int hd_dim
) {
    std::vector<float> step_key(hd_dim);
    generate_step_key(step_key.data(), step, hd_dim);
    
    // grad_thought = unbind(grad_trace, step_key)
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const float* grad_ptr = grad_trace + b * hd_dim;
        float* out_ptr = grad_thought + b * hd_dim;
        
        hd_unbind_thought(grad_ptr, step_key.data(), out_ptr, hd_dim);
    }
}

}  // namespace hd_thought
}  // namespace hsmn

#endif  // HIGHNOON_NATIVE_OPS_HD_THOUGHT_TRACE_OP_H_
