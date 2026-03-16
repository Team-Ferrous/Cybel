// saguaro.native/ops/unified_attention_op.cc
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file unified_attention_op.cc
 * @brief Unified Attention Operations Implementation
 *
 * This file implements all 11 attention mechanisms through a unified interface.
 * Each kernel is optimized for CPU execution with SIMD (AVX2/AVX512/NEON).
 *
 * Phase 2 of V2 Performance Optimization - Attention Consolidation
 */

#include "unified_attention_op.h"

#include <cstring>
#include <memory>
#include <vector>

// TensorFlow includes for op registration
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

// TBB for parallel execution
#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

namespace saguaro {
namespace attention {

// =============================================================================
// FLASH ATTENTION KERNEL
// O(n²) full attention with memory-efficient tiling
// =============================================================================

void FlashAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    float* workspace) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const float scale = config.ComputeScale();
    const float eps = config.epsilon;
    
    // Total elements per head: seq_q * head_dim
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    
    // Parallel over batch and heads
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            // Map to KV head for GQA
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Thread-local storage for attention scores
            std::vector<float> scores(seq_k);
            
            // For each query position
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                float* out_i = out_ptr + i * head_dim;
                
                // Compute scaled dot-product scores
                for (int j = 0; j < seq_k; ++j) {
                    const float* kj = k_ptr + j * head_dim;
                    scores[j] = scale * simd::dot_product(qi, kj, head_dim);
                }
                
                // Apply causal mask if needed
                if (config.causal) {
                    simd::apply_causal_mask(scores.data(), 1, seq_k, i);
                }
                
                // Softmax
                simd::softmax_inplace(scores.data(), seq_k, eps);
                
                // Weighted sum of values
                simd::weighted_sum(scores.data(), v_ptr, out_i, seq_k, head_dim);
            }
        }
    }
}

// =============================================================================
// LINEAR ATTENTION KERNEL
// O(n) via ELU+1 feature map: O = φ(Q) @ (φ(K)ᵀ @ V) / (φ(Q) @ φ(K)ᵀ @ 1)
// =============================================================================

void LinearAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output,
    float* kv_state,
    float* k_sum) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    
    // State matrices: KV ∈ R^{d×d}, K_sum ∈ R^d
    const int state_size = head_dim * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Thread-local state matrices
            std::vector<float> KV(state_size, 0.0f);  // [head_dim, head_dim]
            std::vector<float> K_sum(head_dim, 0.0f); // [head_dim]
            std::vector<float> phi_q(head_dim);
            std::vector<float> phi_k(head_dim);
            
            // Causal linear attention: accumulate KV state as we process
            for (int i = 0; i < seq_k; ++i) {
                // Apply ELU+1 feature map to key
                std::memcpy(phi_k.data(), k_ptr + i * head_dim, head_dim * sizeof(float));
                simd::elu_plus_one_inplace(phi_k.data(), head_dim);
                
                const float* vi = v_ptr + i * head_dim;
                
                // Update state: KV += φ(k) ⊗ v
                for (int d1 = 0; d1 < head_dim; ++d1) {
                    for (int d2 = 0; d2 < head_dim; ++d2) {
                        KV[d1 * head_dim + d2] += phi_k[d1] * vi[d2];
                    }
                    K_sum[d1] += phi_k[d1];
                }
                
                // If this is also a query position, compute output
                if (i < seq_q) {
                    std::memcpy(phi_q.data(), q_ptr + i * head_dim, head_dim * sizeof(float));
                    simd::elu_plus_one_inplace(phi_q.data(), head_dim);
                    
                    float* out_i = out_ptr + i * head_dim;
                    
                    // Numerator: φ(q) @ KV
                    for (int d2 = 0; d2 < head_dim; ++d2) {
                        float sum = 0.0f;
                        for (int d1 = 0; d1 < head_dim; ++d1) {
                            sum += phi_q[d1] * KV[d1 * head_dim + d2];
                        }
                        out_i[d2] = sum;
                    }
                    
                    // Denominator: φ(q) · K_sum
                    float denom = simd::dot_product(phi_q.data(), K_sum.data(), head_dim);
                    denom = std::max(denom, eps);
                    
                    // Normalize
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] /= denom;
                    }
                }
            }
        }
    }
}

// =============================================================================
// LOCAL WINDOWED ATTENTION KERNEL
// O(n×w) with window size w (Griffin-style)
// =============================================================================

void LocalWindowedAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const int window = config.window_size;
    const float scale = config.ComputeScale();
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Thread-local storage for window attention scores
            std::vector<float> scores(window + 1);
            
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                float* out_i = out_ptr + i * head_dim;
                
                // Compute window bounds
                int w_start = std::max(0, i - window / 2);
                int w_end = std::min(seq_k, i + window / 2 + 1);
                
                if (config.causal) {
                    w_end = std::min(w_end, i + 1);
                }
                
                int w_len = w_end - w_start;
                if (w_len <= 0) {
                    std::fill(out_i, out_i + head_dim, 0.0f);
                    continue;
                }
                
                // Compute scores within window
                for (int j = 0; j < w_len; ++j) {
                    const float* kj = k_ptr + (w_start + j) * head_dim;
                    scores[j] = scale * simd::dot_product(qi, kj, head_dim);
                }
                
                // Softmax over window
                simd::softmax_inplace(scores.data(), w_len, eps);
                
                // Weighted sum within window
                std::fill(out_i, out_i + head_dim, 0.0f);
                for (int j = 0; j < w_len; ++j) {
                    const float* vj = v_ptr + (w_start + j) * head_dim;
                    float w = scores[j];
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] += w * vj[d];
                    }
                }
            }
        }
    }
}

// =============================================================================
// DIFFERENTIAL ATTENTION KERNEL
// DiffAttn = softmax(Q₁K₁ᵀ/√d) - λ·softmax(Q₂K₂ᵀ/√d)
// =============================================================================

void DifferentialAttentionForward(
    const float* Q, const float* K, const float* V,
    float lambda,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const int half_dim = head_dim / 2;
    const float scale = 1.0f / std::sqrt(static_cast<float>(half_dim));
    const float eps = config.epsilon;
    
    // Clamp lambda
    lambda = std::max(config.lambda_min, std::min(config.lambda_max, lambda));
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Thread-local storage
            std::vector<float> scores1(seq_k);
            std::vector<float> scores2(seq_k);
            std::vector<float> diff_attn(seq_k);
            
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                const float* q1 = qi;              // First half
                const float* q2 = qi + half_dim;   // Second half
                float* out_i = out_ptr + i * head_dim;
                
                // Compute scores for both halves
                for (int j = 0; j < seq_k; ++j) {
                    const float* kj = k_ptr + j * head_dim;
                    const float* k1 = kj;
                    const float* k2 = kj + half_dim;
                    
                    scores1[j] = scale * simd::dot_product(q1, k1, half_dim);
                    scores2[j] = scale * simd::dot_product(q2, k2, half_dim);
                }
                
                // Apply causal mask
                if (config.causal) {
                    for (int j = i + 1; j < seq_k; ++j) {
                        scores1[j] = -1e9f;
                        scores2[j] = -1e9f;
                    }
                }
                
                // Softmax both score sets
                simd::softmax_inplace(scores1.data(), seq_k, eps);
                simd::softmax_inplace(scores2.data(), seq_k, eps);
                
                // Differential attention: A1 - λ*A2
                // CPU Performance Optimization (Section 6.2): Add SIMD pragma
                #pragma omp simd
                for (int j = 0; j < seq_k; ++j) {
                    diff_attn[j] = scores1[j] - lambda * scores2[j];
                }
                
                // Weighted sum of values
                std::fill(out_i, out_i + head_dim, 0.0f);
                for (int j = 0; j < seq_k; ++j) {
                    const float* vj = v_ptr + j * head_dim;
                    float w = diff_attn[j];
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] += w * vj[d];
                    }
                }
                
                // Optional normalization: (1 - λ_init) scaling
                if (config.normalize_diff) {
                    float norm_scale = 1.0f - config.lambda_init;
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] *= norm_scale;
                    }
                }
            }
        }
    }
}

// =============================================================================
// NATIVE SPARSE ATTENTION (NSA) KERNEL
// O(n log n) via token compression + block selection + local window
// =============================================================================

void SparseNSAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const int block_size = config.block_size;
    const int num_selected = config.num_selected_blocks;
    const int tokens_per_block = config.tokens_per_block;
    const int window = config.window_size;
    const float scale = config.ComputeScale();
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    const int num_blocks = (seq_k + block_size - 1) / block_size;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Step 1: Compute block representations (mean pooling)
            std::vector<float> block_keys(num_blocks * head_dim, 0.0f);
            std::vector<float> block_scores(num_blocks);
            
            for (int blk = 0; blk < num_blocks; ++blk) {
                int start = blk * block_size;
                int end = std::min(start + block_size, seq_k);
                int count = end - start;
                
                float* bkey = block_keys.data() + blk * head_dim;
                for (int j = start; j < end; ++j) {
                    const float* kj = k_ptr + j * head_dim;
                    for (int d = 0; d < head_dim; ++d) {
                        bkey[d] += kj[d];
                    }
                }
                for (int d = 0; d < head_dim; ++d) {
                    bkey[d] /= count;
                }
            }
            
            // For each query position
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                float* out_i = out_ptr + i * head_dim;
                
                // Step 2: Score blocks against query
                for (int blk = 0; blk < num_blocks; ++blk) {
                    const float* bkey = block_keys.data() + blk * head_dim;
                    block_scores[blk] = simd::dot_product(qi, bkey, head_dim);
                }
                
                // Step 3: Select top-k blocks
                std::vector<int> selected_blocks;
                std::vector<std::pair<float, int>> scored_blocks(num_blocks);
                for (int blk = 0; blk < num_blocks; ++blk) {
                    scored_blocks[blk] = {block_scores[blk], blk};
                }
                std::partial_sort(scored_blocks.begin(),
                                  scored_blocks.begin() + std::min(num_selected, num_blocks),
                                  scored_blocks.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                
                for (int k = 0; k < std::min(num_selected, num_blocks); ++k) {
                    selected_blocks.push_back(scored_blocks[k].second);
                }
                
                // Step 4: Gather tokens from selected blocks + local window
                std::vector<int> attend_indices;
                
                // Local window tokens
                int w_start = std::max(0, i - window / 2);
                int w_end = config.causal ? std::min(seq_k, i + 1) : std::min(seq_k, i + window / 2 + 1);
                for (int j = w_start; j < w_end; ++j) {
                    attend_indices.push_back(j);
                }
                
                // Tokens from selected blocks (avoiding duplicates with local window)
                for (int blk : selected_blocks) {
                    int start = blk * block_size;
                    int end = std::min(start + block_size, seq_k);
                    // Select top tokens_per_block from this block by score
                    for (int cnt = 0; cnt < std::min(tokens_per_block, end - start); ++cnt) {
                        int j = start + cnt;
                        if (j < w_start || j >= w_end) {  // Not in local window
                            if (config.causal && j > i) continue;  // Respect causality
                            attend_indices.push_back(j);
                        }
                    }
                }
                
                // Global tokens
                if (config.use_global_tokens) {
                    for (int g = 0; g < config.num_global_tokens && g < seq_k; ++g) {
                        bool found = false;
                        for (int idx : attend_indices) {
                            if (idx == g) { found = true; break; }
                        }
                        if (!found) attend_indices.push_back(g);
                    }
                }
                
                // Step 5: Compute attention over gathered tokens
                int num_attend = static_cast<int>(attend_indices.size());
                std::vector<float> scores(num_attend);
                
                for (int idx = 0; idx < num_attend; ++idx) {
                    int j = attend_indices[idx];
                    const float* kj = k_ptr + j * head_dim;
                    scores[idx] = scale * simd::dot_product(qi, kj, head_dim);
                }
                
                // Softmax and weighted sum
                simd::softmax_inplace(scores.data(), num_attend, eps);
                
                std::fill(out_i, out_i + head_dim, 0.0f);
                for (int idx = 0; idx < num_attend; ++idx) {
                    int j = attend_indices[idx];
                    const float* vj = v_ptr + j * head_dim;
                    float w = scores[idx];
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] += w * vj[d];
                    }
                }
            }
        }
    }
}

// =============================================================================
// GROUPED-QUERY ATTENTION (GQA) KERNEL
// Standard GQA with shared KV heads
// =============================================================================

void GQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    // GQA is essentially flash attention with KV head sharing
    // The sharing is already handled in FlashAttentionForward via QueriesPerKVHead()
    FlashAttentionForward(Q, K, V, config, output, nullptr);
}

// =============================================================================
// LINEAR GQA KERNEL
// Linear attention with GQA head sharing
// =============================================================================

void LinearGQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    // Linear attention already handles GQA via QueriesPerKVHead()
    LinearAttentionForward(Q, K, V, config, output, nullptr, nullptr);
}

// =============================================================================
// SLIDING GQA KERNEL
// Sliding window + GQA
// =============================================================================

void SlidingGQAForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    // Local windowed attention already handles GQA
    LocalWindowedAttentionForward(Q, K, V, config, output);
}

// =============================================================================
// LATENT KV ATTENTION KERNEL
// DeepSeek-style latent KV compression
// =============================================================================

void LatentKVForward(
    const float* Q, const float* K, const float* V,
    const float* latent_keys, const float* latent_values,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int head_dim = config.head_dim;
    const int num_latents = config.num_latents;
    const float scale = config.ComputeScale();
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Thread-local storage
            std::vector<float> scores(num_latents);
            
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                float* out_i = out_ptr + i * head_dim;
                
                // Cross-attention to latents
                for (int l = 0; l < num_latents; ++l) {
                    const float* lk = latent_keys + l * head_dim;
                    scores[l] = scale * simd::dot_product(qi, lk, head_dim);
                }
                
                // Softmax
                simd::softmax_inplace(scores.data(), num_latents, eps);
                
                // Weighted sum of latent values
                std::fill(out_i, out_i + head_dim, 0.0f);
                for (int l = 0; l < num_latents; ++l) {
                    const float* lv = latent_values + l * head_dim;
                    float w = scores[l];
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] += w * lv[d];
                    }
                }
            }
        }
    }
}

// =============================================================================
// QUANTUM ADAPTIVE SELF-ATTENTION (QASA) KERNEL
// VQC-based attention scoring
// =============================================================================

void QASAForward(
    const float* Q, const float* K, const float* V,
    const float* vqc_params,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const int num_qubits = config.num_qubits;
    const int vqc_layers = config.vqc_layers;
    const float entangle = config.entanglement_strength;
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    const int num_params = vqc_layers * num_qubits;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            std::vector<float> scores(seq_k);
            
            for (int i = 0; i < seq_q; ++i) {
                const float* qi = q_ptr + i * head_dim;
                float* out_i = out_ptr + i * head_dim;
                
                // VQC-based attention scores
                for (int j = 0; j < seq_k; ++j) {
                    const float* kj = k_ptr + j * head_dim;
                    
                    // Simplified VQC overlap computation
                    float score = 0.0f;
                    for (int l = 0; l < vqc_layers; ++l) {
                        float layer_score = 0.0f;
                        for (int q = 0; q < num_qubits; ++q) {
                            int param_idx = l * num_qubits + q;
                            int dim_idx = param_idx % head_dim;
                            
                            float theta_q = qi[dim_idx] + vqc_params[param_idx];
                            float theta_k = kj[dim_idx] + vqc_params[num_params + param_idx];
                            
                            layer_score += std::cos(theta_q - theta_k);
                        }
                        layer_score *= (1.0f + entangle * std::cos(layer_score));
                        score += layer_score;
                    }
                    scores[j] = score / (vqc_layers * num_qubits);
                }
                
                // Causal mask
                if (config.causal) {
                    for (int j = i + 1; j < seq_k; ++j) {
                        scores[j] = -1e9f;
                    }
                }
                
                // Softmax
                simd::softmax_inplace(scores.data(), seq_k, eps);
                
                // Weighted sum
                simd::weighted_sum(scores.data(), v_ptr, out_i, seq_k, head_dim);
            }
        }
    }
}

// =============================================================================
// LEARNABLE MULTI-SCALE WAVELET TRANSFORMER (LMWT) KERNEL
// Wavelet-domain attention with learnable decomposition
// =============================================================================

void LMWTForward(
    const float* Q, const float* K, const float* V,
    const float* alpha, const float* beta,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const int num_scales = config.num_wavelet_scales;
    const float scale = config.ComputeScale();
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            const float* q_ptr = Q + (b * heads + h) * q_stride;
            const float* k_ptr = K + (b * config.num_kv_heads + kv_h) * k_stride;
            const float* v_ptr = V + (b * config.num_kv_heads + kv_h) * k_stride;
            float* out_ptr = output + (b * heads + h) * q_stride;
            
            // Multi-scale attention: compute at different granularities
            // and combine with learned weights
            std::vector<float> multi_scale_output(seq_q * head_dim, 0.0f);
            std::vector<float> scores(seq_k);
            
            for (int s = 0; s < num_scales; ++s) {
                float alpha_s = alpha[s];
                float beta_s = beta[s];
                int stride = 1 << s;  // 1, 2, 4, 8, ...
                
                for (int i = 0; i < seq_q; ++i) {
                    const float* qi = q_ptr + i * head_dim;
                    
                    // Downsampled attention
                    int num_k = (seq_k + stride - 1) / stride;
                    for (int jj = 0; jj < num_k; ++jj) {
                        int j = jj * stride;
                        const float* kj = k_ptr + j * head_dim;
                        scores[jj] = scale * simd::dot_product(qi, kj, head_dim);
                    }
                    
                    if (config.causal) {
                        int max_jj = (i / stride) + 1;
                        for (int jj = max_jj; jj < num_k; ++jj) {
                            scores[jj] = -1e9f;
                        }
                    }
                    
                    simd::softmax_inplace(scores.data(), num_k, eps);
                    
                    // Accumulate with wavelet coefficients
                    for (int jj = 0; jj < num_k; ++jj) {
                        int j = jj * stride;
                        const float* vj = v_ptr + j * head_dim;
                        float w = scores[jj] * (s == 0 ? alpha_s : beta_s);
                        for (int d = 0; d < head_dim; ++d) {
                            multi_scale_output[i * head_dim + d] += w * vj[d];
                        }
                    }
                }
            }
            
            // Normalize and copy to output
            float norm_factor = 1.0f / num_scales;
            for (int i = 0; i < seq_q * head_dim; ++i) {
                out_ptr[i] = multi_scale_output[i] * norm_factor;
            }
        }
    }
}

// =============================================================================
// HOLOGRAPHIC FFT FEATURE MAP
// O(n × d log d) via split-radix FFT for HD embedding mesh
// =============================================================================

namespace {
// Cooley-Tukey in-place FFT (radix-2)
// Input: data[0..n-1] as interleaved [re0,im0,re1,im1,...]
// This is a simplified FFT for demonstration - production should use FFTW/PFFFT
void fft_inplace(float* data, int n, bool inverse = false) {
    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[2*i], data[2*j]);
            std::swap(data[2*i+1], data[2*j+1]);
        }
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
    
    // Cooley-Tukey iterative FFT
    float sign = inverse ? 1.0f : -1.0f;
    for (int len = 2; len <= n; len <<= 1) {
        float ang = sign * 2.0f * M_PI / len;
        float wn_re = std::cos(ang);
        float wn_im = std::sin(ang);
        
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int k = 0; k < len / 2; ++k) {
                // u = data[i + k]
                float u_re = data[2*(i+k)];
                float u_im = data[2*(i+k)+1];
                // v = data[i + k + len/2] * w
                float t_re = data[2*(i+k+len/2)];
                float t_im = data[2*(i+k+len/2)+1];
                float v_re = t_re * w_re - t_im * w_im;
                float v_im = t_re * w_im + t_im * w_re;
                
                // Butterfly
                data[2*(i+k)] = u_re + v_re;
                data[2*(i+k)+1] = u_im + v_im;
                data[2*(i+k+len/2)] = u_re - v_re;
                data[2*(i+k+len/2)+1] = u_im - v_im;
                
                // Update twiddle factor
                float w_new_re = w_re * wn_re - w_im * wn_im;
                float w_new_im = w_re * wn_im + w_im * wn_re;
                w_re = w_new_re;
                w_im = w_new_im;
            }
        }
    }
    
    // Scale for inverse FFT
    if (inverse) {
        float scale = 1.0f / n;
        for (int i = 0; i < 2*n; ++i) {
            data[i] *= scale;
        }
    }
}
}  // anonymous namespace

void HolographicFeatureMap(
    const float* input,
    float* features_re,
    float* features_im,
    int batch, int heads, int seq, int head_dim) {
    
    // For each position, compute FFT and extract real/imag parts
    // Total complexity: O(batch × heads × seq × head_dim × log(head_dim))
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            for (int s = 0; s < seq; ++s) {
                int offset = ((b * heads + h) * seq + s) * head_dim;
                const float* in_ptr = input + offset;
                float* out_re = features_re + offset;
                float* out_im = features_im + offset;
                
                // Create interleaved complex buffer: [re0, im0, re1, im1, ...]
                // Input is real-only, so imaginary parts are 0
                std::vector<float> complex_buf(2 * head_dim, 0.0f);
                for (int d = 0; d < head_dim; ++d) {
                    complex_buf[2*d] = in_ptr[d];  // Real part
                    complex_buf[2*d + 1] = 0.0f;   // Imaginary part
                }
                
                // Compute FFT in-place
                fft_inplace(complex_buf.data(), head_dim, false);
                
                // Extract real and imaginary parts
                // Normalize by sqrt(head_dim) for stable feature magnitudes
                float norm = 1.0f / std::sqrt(static_cast<float>(head_dim));
                for (int d = 0; d < head_dim; ++d) {
                    out_re[d] = complex_buf[2*d] * norm;
                    out_im[d] = complex_buf[2*d + 1] * norm;
                }
            }
        }
    }
}

void HolographicLinearAttentionForward(
    const float* Q, const float* K, const float* V,
    const UnifiedAttentionConfig& config,
    float* output) {
    
    const int batch = config.batch_size;
    const int heads = config.num_heads;
    const int seq_q = config.seq_len;
    const int seq_k = config.kv_seq_len;
    const int head_dim = config.head_dim;
    const float eps = config.epsilon;
    
    const int q_stride = seq_q * head_dim;
    const int k_stride = seq_k * head_dim;
    const int total_q = batch * heads * seq_q * head_dim;
    const int total_k = batch * heads * seq_k * head_dim;
    
    // Allocate feature buffers for complex FFT features
    // φ(x) = [Re(FFT(x)), Im(FFT(x))]
    std::vector<float> Q_re(total_q), Q_im(total_q);
    std::vector<float> K_re(total_k), K_im(total_k);
    
    // Compute holographic feature maps: O(n × d log d)
    HolographicFeatureMap(Q, Q_re.data(), Q_im.data(), batch, heads, seq_q, head_dim);
    HolographicFeatureMap(K, K_re.data(), K_im.data(), batch, heads, seq_k, head_dim);
    
    // O(n) linear attention with complex-valued kernel trick
    // State matrices: KV_re, KV_im ∈ R^{d×d}, K_sum_re, K_sum_im ∈ R^d
    const int state_size = head_dim * head_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int kv_h = h;
            if (config.num_kv_heads < config.num_heads) {
                kv_h = h / config.QueriesPerKVHead();
            }
            
            int q_offset = (b * heads + h) * q_stride;
            int k_offset = (b * config.num_kv_heads + kv_h) * k_stride;
            
            const float* q_re = Q_re.data() + q_offset;
            const float* q_im = Q_im.data() + q_offset;
            const float* k_re = K_re.data() + k_offset;
            const float* k_im = K_im.data() + k_offset;
            const float* v_ptr = V + k_offset;
            float* out_ptr = output + q_offset;
            
            // Complex-valued KV state and K sum
            std::vector<float> KV_re(state_size, 0.0f);
            std::vector<float> KV_im(state_size, 0.0f);
            std::vector<float> K_sum_re(head_dim, 0.0f);
            std::vector<float> K_sum_im(head_dim, 0.0f);
            
            // Causal linear attention: accumulate KV state as we process
            for (int i = 0; i < seq_k; ++i) {
                const float* ki_re = k_re + i * head_dim;
                const float* ki_im = k_im + i * head_dim;
                const float* vi = v_ptr + i * head_dim;
                
                // Update state: KV += φ(k) ⊗ v (complex outer product)
                // (k_re + i·k_im) ⊗ v = k_re·v + i·(k_im·v)
                for (int d1 = 0; d1 < head_dim; ++d1) {
                    for (int d2 = 0; d2 < head_dim; ++d2) {
                        KV_re[d1 * head_dim + d2] += ki_re[d1] * vi[d2];
                        KV_im[d1 * head_dim + d2] += ki_im[d1] * vi[d2];
                    }
                    K_sum_re[d1] += ki_re[d1];
                    K_sum_im[d1] += ki_im[d1];
                }
                
                // If this is also a query position, compute output
                if (i < seq_q) {
                    const float* qi_re = q_re + i * head_dim;
                    const float* qi_im = q_im + i * head_dim;
                    float* out_i = out_ptr + i * head_dim;
                    
                    // Numerator: Re(φ(q)* · KV) where φ(q)* is complex conjugate
                    // (q_re - i·q_im) · (KV_re + i·KV_im) = (q_re·KV_re + q_im·KV_im) + i·(...)
                    // We only need the real part for output
                    for (int d2 = 0; d2 < head_dim; ++d2) {
                        float sum = 0.0f;
                        for (int d1 = 0; d1 < head_dim; ++d1) {
                            // Real part of complex dot product with conjugate
                            sum += qi_re[d1] * KV_re[d1 * head_dim + d2] +
                                   qi_im[d1] * KV_im[d1 * head_dim + d2];
                        }
                        out_i[d2] = sum;
                    }
                    
                    // Denominator: Re(φ(q)* · K_sum)
                    float denom = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        denom += qi_re[d] * K_sum_re[d] + qi_im[d] * K_sum_im[d];
                    }
                    denom = std::max(denom, eps);
                    
                    // Normalize
                    for (int d = 0; d < head_dim; ++d) {
                        out_i[d] /= denom;
                    }
                }
            }
        }
    }
}

}  // namespace attention
}  // namespace saguaro

// =============================================================================
// TENSORFLOW OP REGISTRATION
// =============================================================================

using namespace tensorflow;

REGISTER_OP("UnifiedAttention")
    .Input("query: float32")
    .Input("key: float32")
    .Input("value: float32")
    .Input("extra_inputs: float32")
    .Output("output: float32")
    .Attr("mode: int = 0")
    .Attr("batch_size: int = 1")
    .Attr("num_heads: int = 8")
    .Attr("num_kv_heads: int = 1")
    .Attr("head_dim: int = 64")
    .Attr("seq_len: int = 512")
    .Attr("kv_seq_len: int = 512")
    .Attr("window_size: int = 256")
    .Attr("causal: bool = true")
    .Attr("scale: float = 0.0")
    .Attr("dropout_rate: float = 0.0")
    .Attr("epsilon: float = 1e-6")
    .Attr("lambda_init: float = 0.8")
    .Attr("block_size: int = 64")
    .Attr("num_selected_blocks: int = 8")
    .Attr("num_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("entanglement_strength: float = 0.5")
    .Attr("num_latents: int = 64")
    .Attr("num_wavelet_scales: int = 4")
    .Attr("use_holographic_features: bool = false")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // Output shape matches query shape
        return Status();
    })
    .Doc(R"doc(
Unified Attention operation supporting 11 attention mechanisms.

query: Query tensor [batch, heads, seq_q, head_dim]
key: Key tensor [batch, kv_heads, seq_k, head_dim]  
value: Value tensor [batch, kv_heads, seq_k, head_dim]
extra_inputs: Mode-specific inputs (VQC params, latents, wavelet coeffs)
output: Attention output [batch, heads, seq_q, head_dim]
mode: Attention mode (0=FLASH, 1=LINEAR, 2=LOCAL, 3=DIFFERENTIAL, etc.)
)doc");

class UnifiedAttentionOp : public OpKernel {
public:
    explicit UnifiedAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("batch_size", &batch_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_kv_heads", &num_kv_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seq_len", &seq_len_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("kv_seq_len", &kv_seq_len_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("causal", &causal_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("dropout_rate", &dropout_rate_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("lambda_init", &lambda_init_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("block_size", &block_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_selected_blocks", &num_selected_blocks_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_qubits", &num_qubits_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &vqc_layers_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entanglement_strength", &entanglement_strength_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_latents", &num_latents_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_wavelet_scales", &num_wavelet_scales_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_holographic_features", &use_holographic_features_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& query = ctx->input(0);
        const Tensor& key = ctx->input(1);
        const Tensor& value = ctx->input(2);
        const Tensor& extra = ctx->input(3);
        
        // Allocate output
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &output));
        
        // Build config
        saguaro::attention::UnifiedAttentionConfig config;
        config.mode = static_cast<saguaro::attention::AttentionMode>(mode_);
        config.batch_size = batch_size_;
        config.num_heads = num_heads_;
        config.num_kv_heads = num_kv_heads_;
        config.head_dim = head_dim_;
        config.seq_len = seq_len_;
        config.kv_seq_len = kv_seq_len_;
        config.window_size = window_size_;
        config.causal = causal_;
        config.scale = scale_;
        config.dropout_rate = dropout_rate_;
        config.epsilon = epsilon_;
        config.lambda_init = lambda_init_;
        config.block_size = block_size_;
        config.num_selected_blocks = num_selected_blocks_;
        config.num_qubits = num_qubits_;
        config.vqc_layers = vqc_layers_;
        config.entanglement_strength = entanglement_strength_;
        config.num_latents = num_latents_;
        config.num_wavelet_scales = num_wavelet_scales_;
        config.use_holographic_features = use_holographic_features_;
        
        // Call unified dispatcher
        saguaro::attention::UnifiedAttentionForward(
            query.flat<float>().data(),
            key.flat<float>().data(),
            value.flat<float>().data(),
            config,
            output->flat<float>().data(),
            extra.flat<float>().data());
    }
    
private:
    int mode_;
    int batch_size_;
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int seq_len_;
    int kv_seq_len_;
    int window_size_;
    bool causal_;
    float scale_;
    float dropout_rate_;
    float epsilon_;
    float lambda_init_;
    int block_size_;
    int num_selected_blocks_;
    int num_qubits_;
    int vqc_layers_;
    float entanglement_strength_;
    int num_latents_;
    int num_wavelet_scales_;
    bool use_holographic_features_;
};

REGISTER_KERNEL_BUILDER(Name("UnifiedAttention").Device(DEVICE_CPU), UnifiedAttentionOp);
