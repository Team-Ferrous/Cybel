// saguaro.native/ops/fused_collapse_op.cc
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
 * @file fused_collapse_op.cc
 * @brief TensorFlow op implementation for Fused Contextual Gating Collapse.
 *
 * Enterprise-grade fused collapse operator with SIMD optimization.
 * Implements cross-attention collapse for superposition states with
 * Gumbel-Softmax unified training/inference.
 *
 * Migration from: saguaro/models/layers/collapse.py
 * Phase 16: Contextual Gating Collapse Enhancement
 */

#include "fused_collapse_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op.h"

#include <cmath>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

namespace {

// =============================================================================
// CORE KERNEL IMPLEMENTATIONS
// =============================================================================

/**
 * @brief Scalar forward pass implementation for fused collapse.
 *
 * Computes multi-head cross-attention with Gumbel-Softmax:
 * 1. Q = context @ W_q + b_q
 * 2. K = superposed @ W_k + b_k  (for each superposition state)
 * 3. V = superposed @ W_v + b_v
 * 4. attention = gumbel_softmax(Q @ K^T / sqrt(d_head), temperature)
 * 5. context_vec = attention @ V
 * 6. output = context_vec @ W_o + b_o
 */
template <typename T>
void FusedCollapseForwardScalar(
    const T* context,           // [B, d_in]
    const T* superposed,        // [B, S, d_out]
    const T* q_weights,         // [d_in, d_out]
    const T* k_weights,         // [d_out, d_out]
    const T* v_weights,         // [d_out, d_out]
    const T* o_weights,         // [d_out, d_out]
    const T* q_bias,            // [d_out]
    const T* k_bias,            // [d_out]
    const T* v_bias,            // [d_out]
    const T* o_bias,            // [d_out]
    T* output,                  // [B, d_out]
    T* attention_cache,         // [B, H, S]
    int batch,
    int superposition_dim,
    int d_in,
    int d_out,
    int num_heads,
    float temperature,
    bool training,
    bool use_kernel_attention,
    int feature_map) {
    
    const int head_dim = d_out / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Pre-allocate per-thread buffers outside parallel region
    #pragma omp parallel
    {
        // Thread-local buffers
        std::vector<T> query(d_out);
        std::vector<T> key(superposition_dim * d_out);
        std::vector<T> value(superposition_dim * d_out);
        std::vector<T> scores(superposition_dim);
        std::vector<T> gumbel_noise(superposition_dim);
        std::vector<T> attention_weights(superposition_dim);
        std::vector<T> context_vec(d_out);
        
        #pragma omp for
        for (int b = 0; b < batch; ++b) {
            const T* ctx_b = context + b * d_in;
            const T* sup_b = superposed + b * superposition_dim * d_out;
            T* out_b = output + b * d_out;
            T* attn_cache_b = attention_cache + b * num_heads * superposition_dim;
            
            // Step 1: Compute Query from context
            // Q = ctx @ W_q^T + b_q  (W_q is [d_in, d_out])
            for (int o = 0; o < d_out; ++o) {
                T sum = static_cast<T>(0);
                for (int i = 0; i < d_in; ++i) {
                    sum += ctx_b[i] * q_weights[i * d_out + o];
                }
                query[o] = sum + q_bias[o];
            }
            
            // Step 2: Compute Key and Value for each superposition state
            for (int s = 0; s < superposition_dim; ++s) {
                const T* sup_s = sup_b + s * d_out;
                T* key_s = key.data() + s * d_out;
                T* val_s = value.data() + s * d_out;
                
                for (int o = 0; o < d_out; ++o) {
                    T k_sum = static_cast<T>(0);
                    T v_sum = static_cast<T>(0);
                    for (int i = 0; i < d_out; ++i) {
                        k_sum += sup_s[i] * k_weights[i * d_out + o];
                        v_sum += sup_s[i] * v_weights[i * d_out + o];
                    }
                    key_s[o] = k_sum + k_bias[o];
                    val_s[o] = v_sum + v_bias[o];
                }
            }
            
            // Step 3: Multi-head attention with Gumbel-Softmax
            std::fill(context_vec.begin(), context_vec.end(), static_cast<T>(0));
            
            for (int h = 0; h < num_heads; ++h) {
                const int head_offset = h * head_dim;
                
                // Compute attention scores for this head
                for (int s = 0; s < superposition_dim; ++s) {
                    const T* key_s = key.data() + s * d_out + head_offset;
                    const T* query_h = query.data() + head_offset;
                    
                    T dot = static_cast<T>(0);
                    for (int d = 0; d < head_dim; ++d) {
                        dot += query_h[d] * key_s[d];
                    }
                    scores[s] = dot * scale;
                }
                
                // Apply kernel attention feature map if enabled
                if (use_kernel_attention) {
                    if (feature_map == 1) {
                        // ELU+1 feature map
                        for (int s = 0; s < superposition_dim; ++s) {
                            T x = scores[s];
                            scores[s] = (x > static_cast<T>(0)) ? (x + static_cast<T>(1)) : std::exp(x);
                        }
                    } else if (feature_map == 2) {
                        // ReLU² feature map
                        for (int s = 0; s < superposition_dim; ++s) {
                            T x = std::max(static_cast<T>(0), scores[s]);
                            scores[s] = x * x;
                        }
                    }
                    
                    // Normalize (linear attention)
                    T sum = static_cast<T>(0);
                    for (int s = 0; s < superposition_dim; ++s) {
                        sum += scores[s];
                    }
                    T inv_sum = (sum > static_cast<T>(0)) ? (static_cast<T>(1) / sum) : static_cast<T>(0);
                    for (int s = 0; s < superposition_dim; ++s) {
                        attention_weights[s] = scores[s] * inv_sum;
                    }
                } else {
                    // Gumbel-Softmax
                    saguaro::ops::generate_gumbel_noise(gumbel_noise.data(), superposition_dim);
                    saguaro::ops::simd_gumbel_softmax(
                        scores.data(),
                        gumbel_noise.data(),
                        attention_weights.data(),
                        superposition_dim,
                        temperature,
                        !training  // hard samples at inference
                    );
                }
                
                // Cache attention weights for backward
                for (int s = 0; s < superposition_dim; ++s) {
                    attn_cache_b[h * superposition_dim + s] = attention_weights[s];
                }
                
                // Aggregate values weighted by attention
                for (int s = 0; s < superposition_dim; ++s) {
                    const T* val_s = value.data() + s * d_out + head_offset;
                    T weight = attention_weights[s];
                    
                    for (int d = 0; d < head_dim; ++d) {
                        context_vec[head_offset + d] += weight * val_s[d];
                    }
                }
            }
            
            // Step 4: Output projection
            for (int o = 0; o < d_out; ++o) {
                T sum = static_cast<T>(0);
                for (int i = 0; i < d_out; ++i) {
                    sum += context_vec[i] * o_weights[i * d_out + o];
                }
                out_b[o] = sum + o_bias[o];
            }
        }
    }  // end parallel
}

/**
 * @brief Backward pass for fused collapse.
 */
template <typename T>
void FusedCollapseBackwardScalar(
    const T* grad_output,       // [B, d_out]
    const T* context,           // [B, d_in]
    const T* superposed,        // [B, S, d_out]
    const T* attention_cache,   // [B, H, S]
    const T* q_weights,         // [d_in, d_out]
    const T* k_weights,         // [d_out, d_out]
    const T* v_weights,         // [d_out, d_out]
    const T* o_weights,         // [d_out, d_out]
    const T* q_bias,
    const T* k_bias,
    const T* v_bias,
    T* grad_context,            // [B, d_in]
    T* grad_superposed,         // [B, S, d_out]
    T* grad_q_weights,          // [d_in, d_out]
    T* grad_k_weights,          // [d_out, d_out]
    T* grad_v_weights,          // [d_out, d_out]
    T* grad_o_weights,          // [d_out, d_out]
    T* grad_q_bias,             // [d_out]
    T* grad_k_bias,             // [d_out]
    T* grad_v_bias,             // [d_out]
    T* grad_o_bias,             // [d_out]
    int batch,
    int superposition_dim,
    int d_in,
    int d_out,
    int num_heads) {
    
    const int head_dim = d_out / num_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    
    // Zero out gradient accumulators
    std::fill(grad_q_weights, grad_q_weights + d_in * d_out, static_cast<T>(0));
    std::fill(grad_k_weights, grad_k_weights + d_out * d_out, static_cast<T>(0));
    std::fill(grad_v_weights, grad_v_weights + d_out * d_out, static_cast<T>(0));
    std::fill(grad_o_weights, grad_o_weights + d_out * d_out, static_cast<T>(0));
    std::fill(grad_q_bias, grad_q_bias + d_out, static_cast<T>(0));
    std::fill(grad_k_bias, grad_k_bias + d_out, static_cast<T>(0));
    std::fill(grad_v_bias, grad_v_bias + d_out, static_cast<T>(0));
    std::fill(grad_o_bias, grad_o_bias + d_out, static_cast<T>(0));
    
    #pragma omp parallel
    {
        // Thread-local gradient accumulators
        std::vector<T> local_grad_q_weights(d_in * d_out, static_cast<T>(0));
        std::vector<T> local_grad_k_weights(d_out * d_out, static_cast<T>(0));
        std::vector<T> local_grad_v_weights(d_out * d_out, static_cast<T>(0));
        std::vector<T> local_grad_o_weights(d_out * d_out, static_cast<T>(0));
        std::vector<T> local_grad_q_bias(d_out, static_cast<T>(0));
        std::vector<T> local_grad_k_bias(d_out, static_cast<T>(0));
        std::vector<T> local_grad_v_bias(d_out, static_cast<T>(0));
        std::vector<T> local_grad_o_bias(d_out, static_cast<T>(0));
        
        // Thread-local buffers
        std::vector<T> query(d_out);
        std::vector<T> key(superposition_dim * d_out);
        std::vector<T> value(superposition_dim * d_out);
        std::vector<T> context_vec(d_out);
        std::vector<T> grad_context_vec(d_out);
        std::vector<T> grad_value(superposition_dim * d_out);
        std::vector<T> grad_query(d_out);
        std::vector<T> grad_key(superposition_dim * d_out);
        
        #pragma omp for
        for (int b = 0; b < batch; ++b) {
            const T* ctx_b = context + b * d_in;
            const T* sup_b = superposed + b * superposition_dim * d_out;
            const T* go_b = grad_output + b * d_out;
            const T* attn_b = attention_cache + b * num_heads * superposition_dim;
            T* gc_b = grad_context + b * d_in;
            T* gs_b = grad_superposed + b * superposition_dim * d_out;
            
            // Recompute forward values
            for (int o = 0; o < d_out; ++o) {
                T sum = static_cast<T>(0);
                for (int i = 0; i < d_in; ++i) {
                    sum += ctx_b[i] * q_weights[i * d_out + o];
                }
                query[o] = sum + q_bias[o];
            }
            
            for (int s = 0; s < superposition_dim; ++s) {
                const T* sup_s = sup_b + s * d_out;
                T* key_s = key.data() + s * d_out;
                T* val_s = value.data() + s * d_out;
                
                for (int o = 0; o < d_out; ++o) {
                    T k_sum = static_cast<T>(0);
                    T v_sum = static_cast<T>(0);
                    for (int i = 0; i < d_out; ++i) {
                        k_sum += sup_s[i] * k_weights[i * d_out + o];
                        v_sum += sup_s[i] * v_weights[i * d_out + o];
                    }
                    key_s[o] = k_sum + k_bias[o];
                    val_s[o] = v_sum + v_bias[o];
                }
            }
            
            // Reconstruct context_vec
            std::fill(context_vec.begin(), context_vec.end(), static_cast<T>(0));
            for (int h = 0; h < num_heads; ++h) {
                const int head_offset = h * head_dim;
                for (int s = 0; s < superposition_dim; ++s) {
                    T weight = attn_b[h * superposition_dim + s];
                    const T* val_s = value.data() + s * d_out + head_offset;
                    for (int d = 0; d < head_dim; ++d) {
                        context_vec[head_offset + d] += weight * val_s[d];
                    }
                }
            }
            
            // Backward through output projection
            // grad_context_vec = grad_output @ W_o
            for (int i = 0; i < d_out; ++i) {
                T sum = static_cast<T>(0);
                for (int o = 0; o < d_out; ++o) {
                    sum += go_b[o] * o_weights[i * d_out + o];
                }
                grad_context_vec[i] = sum;
            }
            
            // Accumulate grad_o_weights and grad_o_bias
            for (int i = 0; i < d_out; ++i) {
                for (int o = 0; o < d_out; ++o) {
                    local_grad_o_weights[i * d_out + o] += context_vec[i] * go_b[o];
                }
            }
            for (int o = 0; o < d_out; ++o) {
                local_grad_o_bias[o] += go_b[o];
            }
            
            // Backward through value aggregation
            std::fill(grad_value.begin(), grad_value.end(), static_cast<T>(0));
            std::fill(grad_query.begin(), grad_query.end(), static_cast<T>(0));
            std::fill(grad_key.begin(), grad_key.end(), static_cast<T>(0));
            
            for (int h = 0; h < num_heads; ++h) {
                const int head_offset = h * head_dim;
                
                for (int s = 0; s < superposition_dim; ++s) {
                    T weight = attn_b[h * superposition_dim + s];
                    T* gv_s = grad_value.data() + s * d_out + head_offset;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        gv_s[d] += weight * grad_context_vec[head_offset + d];
                    }
                }
                
                // Gradient through attention (simplified - assumes soft attention)
                // For Gumbel-Softmax gradient, this is a straight-through approximation
                for (int s = 0; s < superposition_dim; ++s) {
                    T grad_weight = static_cast<T>(0);
                    const T* val_s = value.data() + s * d_out + head_offset;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        grad_weight += val_s[d] * grad_context_vec[head_offset + d];
                    }
                    
                    // Gradient through scaled dot-product
                    T grad_score = grad_weight * attn_b[h * superposition_dim + s] * 
                                  (static_cast<T>(1) - attn_b[h * superposition_dim + s]);
                    grad_score *= scale;
                    
                    // Accumulate gradients for query and key
                    const T* key_s = key.data() + s * d_out + head_offset;
                    for (int d = 0; d < head_dim; ++d) {
                        grad_query[head_offset + d] += grad_score * key_s[d];
                        grad_key[s * d_out + head_offset + d] += grad_score * query[head_offset + d];
                    }
                }
            }
            
            // Backward through V projection
            for (int s = 0; s < superposition_dim; ++s) {
                const T* sup_s = sup_b + s * d_out;
                const T* gv_s = grad_value.data() + s * d_out;
                T* gs_s = gs_b + s * d_out;
                
                // grad_superposed += grad_value @ W_v^T
                for (int i = 0; i < d_out; ++i) {
                    T sum = static_cast<T>(0);
                    for (int o = 0; o < d_out; ++o) {
                        sum += gv_s[o] * v_weights[i * d_out + o];
                    }
                    gs_s[i] += sum;
                }
                
                // Accumulate grad_v_weights
                for (int i = 0; i < d_out; ++i) {
                    for (int o = 0; o < d_out; ++o) {
                        local_grad_v_weights[i * d_out + o] += sup_s[i] * gv_s[o];
                    }
                }
                
                // Accumulate grad_v_bias
                for (int o = 0; o < d_out; ++o) {
                    local_grad_v_bias[o] += gv_s[o];
                }
            }
            
            // Backward through K projection
            for (int s = 0; s < superposition_dim; ++s) {
                const T* sup_s = sup_b + s * d_out;
                const T* gk_s = grad_key.data() + s * d_out;
                T* gs_s = gs_b + s * d_out;
                
                for (int i = 0; i < d_out; ++i) {
                    T sum = static_cast<T>(0);
                    for (int o = 0; o < d_out; ++o) {
                        sum += gk_s[o] * k_weights[i * d_out + o];
                    }
                    gs_s[i] += sum;
                }
                
                for (int i = 0; i < d_out; ++i) {
                    for (int o = 0; o < d_out; ++o) {
                        local_grad_k_weights[i * d_out + o] += sup_s[i] * gk_s[o];
                    }
                }
                
                for (int o = 0; o < d_out; ++o) {
                    local_grad_k_bias[o] += gk_s[o];
                }
            }
            
            // Backward through Q projection
            for (int i = 0; i < d_in; ++i) {
                T sum = static_cast<T>(0);
                for (int o = 0; o < d_out; ++o) {
                    sum += grad_query[o] * q_weights[i * d_out + o];
                }
                gc_b[i] = sum;
            }
            
            for (int i = 0; i < d_in; ++i) {
                for (int o = 0; o < d_out; ++o) {
                    local_grad_q_weights[i * d_out + o] += ctx_b[i] * grad_query[o];
                }
            }
            
            for (int o = 0; o < d_out; ++o) {
                local_grad_q_bias[o] += grad_query[o];
            }
        }
        
        // Reduce thread-local gradients
        #pragma omp critical
        {
            for (int64_t i = 0; i < d_in * d_out; ++i) {
                grad_q_weights[i] += local_grad_q_weights[i];
            }
            for (int64_t i = 0; i < d_out * d_out; ++i) {
                grad_k_weights[i] += local_grad_k_weights[i];
                grad_v_weights[i] += local_grad_v_weights[i];
                grad_o_weights[i] += local_grad_o_weights[i];
            }
            for (int64_t i = 0; i < d_out; ++i) {
                grad_q_bias[i] += local_grad_q_bias[i];
                grad_k_bias[i] += local_grad_k_bias[i];
                grad_v_bias[i] += local_grad_v_bias[i];
                grad_o_bias[i] += local_grad_o_bias[i];
            }
        }
    }  // end parallel
}

}  // namespace

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedCollapse")
    .Input("context: T")                 // [B, d_in]
    .Input("superposed: T")              // [B, S, d_out]
    .Input("q_weights: T")               // [d_in, d_out]
    .Input("k_weights: T")               // [d_out, d_out]
    .Input("v_weights: T")               // [d_out, d_out]
    .Input("o_weights: T")               // [d_out, d_out]
    .Input("q_bias: T")                  // [d_out]
    .Input("k_bias: T")                  // [d_out]
    .Input("v_bias: T")                  // [d_out]
    .Input("o_bias: T")                  // [d_out]
    .Output("output: T")                 // [B, d_out]
    .Output("attention_cache: T")        // [B, H, S]
    .Attr("T: {float32, float64}")
    .Attr("num_heads: int = 4")
    .Attr("temperature: float = 1.0")
    .Attr("training: bool = true")
    .Attr("use_kernel_attention: bool = false")
    .Attr("feature_map: int = 0")        // 0=softmax, 1=elu+1, 2=relu²
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle context_shape, superposed_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &context_shape));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &superposed_shape));
        
        DimensionHandle batch = c->Dim(context_shape, 0);
        DimensionHandle d_out = c->Dim(superposed_shape, 2);
        DimensionHandle superposition_dim = c->Dim(superposed_shape, 1);
        
        int num_heads;
        TF_RETURN_IF_ERROR(c->GetAttr("num_heads", &num_heads));
        
        // output: [B, d_out]
        c->set_output(0, c->MakeShape({batch, d_out}));
        // attention_cache: [B, H, S]
        c->set_output(1, c->MakeShape({batch, num_heads, superposition_dim}));
        
        return OkStatus();
    });

REGISTER_OP("FusedCollapseGrad")
    .Input("grad_output: T")             // [B, d_out]
    .Input("context: T")                 // [B, d_in]
    .Input("superposed: T")              // [B, S, d_out]
    .Input("attention_cache: T")         // [B, H, S]
    .Input("q_weights: T")               // [d_in, d_out]
    .Input("k_weights: T")               // [d_out, d_out]
    .Input("v_weights: T")               // [d_out, d_out]
    .Input("o_weights: T")               // [d_out, d_out]
    .Input("q_bias: T")                  // [d_out]
    .Input("k_bias: T")                  // [d_out]
    .Input("v_bias: T")                  // [d_out]
    .Output("grad_context: T")           // [B, d_in]
    .Output("grad_superposed: T")        // [B, S, d_out]
    .Output("grad_q_weights: T")         // [d_in, d_out]
    .Output("grad_k_weights: T")         // [d_out, d_out]
    .Output("grad_v_weights: T")         // [d_out, d_out]
    .Output("grad_o_weights: T")         // [d_out, d_out]
    .Output("grad_q_bias: T")            // [d_out]
    .Output("grad_k_bias: T")            // [d_out]
    .Output("grad_v_bias: T")            // [d_out]
    .Output("grad_o_bias: T")            // [d_out]
    .Attr("T: {float32, float64}")
    .Attr("num_heads: int = 4")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));   // grad_context same as context
        c->set_output(1, c->input(2));   // grad_superposed same as superposed
        c->set_output(2, c->input(4));   // grad_q_weights same as q_weights
        c->set_output(3, c->input(5));   // grad_k_weights same as k_weights
        c->set_output(4, c->input(6));   // grad_v_weights same as v_weights
        c->set_output(5, c->input(7));   // grad_o_weights same as o_weights
        c->set_output(6, c->input(8));   // grad_q_bias same as q_bias
        c->set_output(7, c->input(9));   // grad_k_bias same as k_bias
        c->set_output(8, c->input(10));  // grad_v_bias same as v_bias
        c->set_output(9, c->input(10));  // grad_o_bias same as o_bias (approx)
        return OkStatus();
    });

// =============================================================================
// OPKERNEL IMPLEMENTATIONS
// =============================================================================

template <typename T>
class FusedCollapseOp : public OpKernel {
public:
    explicit FusedCollapseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("training", &training_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_kernel_attention", &use_kernel_attention_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_map", &feature_map_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& context = ctx->input(0);
        const Tensor& superposed = ctx->input(1);
        const Tensor& q_weights = ctx->input(2);
        const Tensor& k_weights = ctx->input(3);
        const Tensor& v_weights = ctx->input(4);
        const Tensor& o_weights = ctx->input(5);
        const Tensor& q_bias = ctx->input(6);
        const Tensor& k_bias = ctx->input(7);
        const Tensor& v_bias = ctx->input(8);
        const Tensor& o_bias = ctx->input(9);
        
        // Validate shapes
        OP_REQUIRES(ctx, context.dims() == 2,
            errors::InvalidArgument("context must be 2D [B, d_in]"));
        OP_REQUIRES(ctx, superposed.dims() == 3,
            errors::InvalidArgument("superposed must be 3D [B, S, d_out]"));
        
        const int batch = context.dim_size(0);
        const int d_in = context.dim_size(1);
        const int superposition_dim = superposed.dim_size(1);
        const int d_out = superposed.dim_size(2);
        
        OP_REQUIRES(ctx, superposed.dim_size(0) == batch,
            errors::InvalidArgument("batch size mismatch"));
        OP_REQUIRES(ctx, d_out % num_heads_ == 0,
            errors::InvalidArgument("d_out must be divisible by num_heads"));
        
        // Allocate outputs
        Tensor* output = nullptr;
        TensorShape output_shape({batch, d_out});
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
        
        Tensor* attention_cache = nullptr;
        TensorShape cache_shape({batch, num_heads_, superposition_dim});
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, cache_shape, &attention_cache));
        
        // Call kernel
        FusedCollapseForwardScalar<T>(
            context.flat<T>().data(),
            superposed.flat<T>().data(),
            q_weights.flat<T>().data(),
            k_weights.flat<T>().data(),
            v_weights.flat<T>().data(),
            o_weights.flat<T>().data(),
            q_bias.flat<T>().data(),
            k_bias.flat<T>().data(),
            v_bias.flat<T>().data(),
            o_bias.flat<T>().data(),
            output->flat<T>().data(),
            attention_cache->flat<T>().data(),
            batch,
            superposition_dim,
            d_in,
            d_out,
            num_heads_,
            temperature_,
            training_,
            use_kernel_attention_,
            feature_map_
        );
    }
    
private:
    int num_heads_;
    float temperature_;
    bool training_;
    bool use_kernel_attention_;
    int feature_map_;
};

template <typename T>
class FusedCollapseGradOp : public OpKernel {
public:
    explicit FusedCollapseGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& context = ctx->input(1);
        const Tensor& superposed = ctx->input(2);
        const Tensor& attention_cache = ctx->input(3);
        const Tensor& q_weights = ctx->input(4);
        const Tensor& k_weights = ctx->input(5);
        const Tensor& v_weights = ctx->input(6);
        const Tensor& o_weights = ctx->input(7);
        const Tensor& q_bias = ctx->input(8);
        const Tensor& k_bias = ctx->input(9);
        const Tensor& v_bias = ctx->input(10);
        
        const int batch = context.dim_size(0);
        const int d_in = context.dim_size(1);
        const int superposition_dim = superposed.dim_size(1);
        const int d_out = superposed.dim_size(2);
        
        // Allocate outputs
        Tensor* grad_context = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, context.shape(), &grad_context));
        
        Tensor* grad_superposed = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, superposed.shape(), &grad_superposed));
        
        Tensor* grad_q_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, q_weights.shape(), &grad_q_weights));
        
        Tensor* grad_k_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, k_weights.shape(), &grad_k_weights));
        
        Tensor* grad_v_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, v_weights.shape(), &grad_v_weights));
        
        Tensor* grad_o_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(5, o_weights.shape(), &grad_o_weights));
        
        Tensor* grad_q_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(6, q_bias.shape(), &grad_q_bias));
        
        Tensor* grad_k_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(7, k_bias.shape(), &grad_k_bias));
        
        Tensor* grad_v_bias = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(8, v_bias.shape(), &grad_v_bias));
        
        Tensor* grad_o_bias = nullptr;
        TensorShape o_bias_shape({d_out});
        OP_REQUIRES_OK(ctx, ctx->allocate_output(9, o_bias_shape, &grad_o_bias));
        
        // Initialize grad_superposed to zero
        std::fill(grad_superposed->flat<T>().data(), 
                  grad_superposed->flat<T>().data() + batch * superposition_dim * d_out,
                  static_cast<T>(0));
        
        // Call backward kernel
        FusedCollapseBackwardScalar<T>(
            grad_output.flat<T>().data(),
            context.flat<T>().data(),
            superposed.flat<T>().data(),
            attention_cache.flat<T>().data(),
            q_weights.flat<T>().data(),
            k_weights.flat<T>().data(),
            v_weights.flat<T>().data(),
            o_weights.flat<T>().data(),
            q_bias.flat<T>().data(),
            k_bias.flat<T>().data(),
            v_bias.flat<T>().data(),
            grad_context->flat<T>().data(),
            grad_superposed->flat<T>().data(),
            grad_q_weights->flat<T>().data(),
            grad_k_weights->flat<T>().data(),
            grad_v_weights->flat<T>().data(),
            grad_o_weights->flat<T>().data(),
            grad_q_bias->flat<T>().data(),
            grad_k_bias->flat<T>().data(),
            grad_v_bias->flat<T>().data(),
            grad_o_bias->flat<T>().data(),
            batch,
            superposition_dim,
            d_in,
            d_out,
            num_heads_
        );
    }
    
private:
    int num_heads_;
};

// Register kernels for float32 and float64
REGISTER_KERNEL_BUILDER(
    Name("FusedCollapse").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedCollapseOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedCollapse").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    FusedCollapseOp<double>);

REGISTER_KERNEL_BUILDER(
    Name("FusedCollapseGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    FusedCollapseGradOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("FusedCollapseGrad").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    FusedCollapseGradOp<double>);

}  // namespace tensorflow
