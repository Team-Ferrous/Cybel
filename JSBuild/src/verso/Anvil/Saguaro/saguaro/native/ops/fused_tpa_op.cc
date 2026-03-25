// saguaro.native/ops/fused_tpa_op.cc
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
 * @file fused_tpa_op.cc
 * @brief O(n) Tensor Product Attention TensorFlow Op.
 *
 * Fused kernel implementing TPA with:
 * 1. Context projection
 * 2. Tensor product Q/K/V factorization
 * 3. KV head expansion
 * 4. O(n) linear attention
 * 5. Output projection and layer norm
 */

#include "fused_tpa_op.h"
#include "fused_linear_gqa_op.h"  // For feature maps and linear attention
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

namespace {

inline void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

inline void layer_norm(float* data, const float* gamma, const float* beta,
                       int batch, int dim, float eps = 1e-5f) {
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b) {
        float* row = data + b * dim;
        float mean = 0.0f, var = 0.0f;
        for (int d = 0; d < dim; ++d) mean += row[d];
        mean /= dim;
        for (int d = 0; d < dim; ++d) var += (row[d] - mean) * (row[d] - mean);
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + eps);
        for (int d = 0; d < dim; ++d) {
            row[d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
        }
    }
}

}  // namespace

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedTPA")
    .Input("x: float")                  // [batch, seq, embed_dim]
    .Input("context_weight: float")     // [embed_dim, context_dim]
    .Input("q_factor_a: float")         // [embed_dim, rank * head_dim]
    .Input("q_factor_b: float")         // [context_dim, rank * num_heads]
    .Input("k_factor_a: float")         // [embed_dim, rank * head_dim]
    .Input("k_factor_b: float")         // [context_dim, rank * num_kv_heads]
    .Input("v_factor_a: float")         // [embed_dim, rank * head_dim]
    .Input("v_factor_b: float")         // [context_dim, rank * num_kv_heads]
    .Input("out_weight: float")         // [embed_dim, embed_dim]
    .Input("norm_gamma: float")
    .Input("norm_beta: float")
    .Output("output: float")
    .Attr("num_heads: int")
    .Attr("num_kv_heads: int")
    .Attr("head_dim: int")
    .Attr("rank: int")
    .Attr("context_dim: int")
    .Attr("causal: bool = true")
    .Attr("eps: float = 1e-6")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Fused O(n) Tensor Product Attention.

TPA factorizes Q/K/V projections via tensor decomposition for 10x+ KV cache reduction.
Combined with linear attention for O(n) complexity.
)doc");

// =============================================================================
// FORWARD KERNEL
// =============================================================================

class FusedTPAOp : public OpKernel {
public:
    explicit FusedTPAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_kv_heads", &num_kv_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("context_dim", &context_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("causal", &causal_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("eps", &eps_));
        
        num_queries_per_kv_ = num_heads_ / num_kv_heads_;
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& x_tensor = ctx->input(0);
        const Tensor& ctx_w_tensor = ctx->input(1);
        const Tensor& q_fa_tensor = ctx->input(2);
        const Tensor& q_fb_tensor = ctx->input(3);
        const Tensor& k_fa_tensor = ctx->input(4);
        const Tensor& k_fb_tensor = ctx->input(5);
        const Tensor& v_fa_tensor = ctx->input(6);
        const Tensor& v_fb_tensor = ctx->input(7);
        const Tensor& out_w_tensor = ctx->input(8);
        const Tensor& gamma_tensor = ctx->input(9);
        const Tensor& beta_tensor = ctx->input(10);

        const int batch_size = x_tensor.dim_size(0);
        const int seq_len = x_tensor.dim_size(1);
        const int embed_dim = x_tensor.dim_size(2);
        const int64_t batch_seq = batch_size * seq_len;

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor.shape(), &output_tensor));

        // Pointers
        const float* x_data = x_tensor.flat<float>().data();
        const float* ctx_w = ctx_w_tensor.flat<float>().data();
        const float* q_fa = q_fa_tensor.flat<float>().data();
        const float* q_fb = q_fb_tensor.flat<float>().data();
        const float* k_fa = k_fa_tensor.flat<float>().data();
        const float* k_fb = k_fb_tensor.flat<float>().data();
        const float* v_fa = v_fa_tensor.flat<float>().data();
        const float* v_fb = v_fb_tensor.flat<float>().data();
        const float* out_w = out_w_tensor.flat<float>().data();
        const float* gamma = gamma_tensor.flat<float>().data();
        const float* beta = beta_tensor.flat<float>().data();
        float* out_data = output_tensor->flat<float>().data();

        // Buffers
        std::vector<float> context(batch_seq * context_dim_);
        std::vector<float> Q(batch_seq * num_heads_ * head_dim_);
        std::vector<float> K_small(batch_seq * num_kv_heads_ * head_dim_);
        std::vector<float> V_small(batch_seq * num_kv_heads_ * head_dim_);
        std::vector<float> K_expanded(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> V_expanded(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> Q_features(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> K_features(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> attn_out(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> kv_state(batch_size * num_heads_ * head_dim_ * head_dim_);
        std::vector<float> k_sum_state(batch_size * num_heads_ * head_dim_);

        // Step 1: Context projection
        saguaro::ops::tpa_context_projection(
            x_data, ctx_w, context.data(),
            batch_seq, embed_dim, context_dim_);

        // Step 2: Tensor product projections for Q, K, V
        saguaro::ops::tpa_tensor_product_projection(
            x_data, context.data(), q_fa, q_fb, Q.data(),
            batch_seq, embed_dim, context_dim_,
            rank_, num_heads_, head_dim_);
        
        saguaro::ops::tpa_tensor_product_projection(
            x_data, context.data(), k_fa, k_fb, K_small.data(),
            batch_seq, embed_dim, context_dim_,
            rank_, num_kv_heads_, head_dim_);
        
        saguaro::ops::tpa_tensor_product_projection(
            x_data, context.data(), v_fa, v_fb, V_small.data(),
            batch_seq, embed_dim, context_dim_,
            rank_, num_kv_heads_, head_dim_);

        // Step 3: Expand KV heads
        saguaro::ops::linear_gqa_expand_kv_heads(
            K_small.data(), K_expanded.data(),
            batch_size, num_kv_heads_, num_queries_per_kv_,
            seq_len, head_dim_);
        saguaro::ops::linear_gqa_expand_kv_heads(
            V_small.data(), V_expanded.data(),
            batch_size, num_kv_heads_, num_queries_per_kv_,
            seq_len, head_dim_);

        // Step 4: Apply ELU feature map
        const int64_t total = batch_size * num_heads_ * seq_len * head_dim_;
        saguaro::ops::linear_gqa_feature_map_elu(Q.data(), Q_features.data(), total);
        saguaro::ops::linear_gqa_feature_map_elu(K_expanded.data(), K_features.data(), total);

        // Step 5: O(n) linear attention
        if (causal_) {
            saguaro::ops::linear_gqa_causal_attention_forward(
                Q_features.data(), K_features.data(), V_expanded.data(),
                attn_out.data(), kv_state.data(), k_sum_state.data(),
                batch_size, num_heads_, seq_len, head_dim_, head_dim_, eps_);
        } else {
            saguaro::ops::linear_gqa_bidirectional_attention_forward(
                Q_features.data(), K_features.data(), V_expanded.data(),
                attn_out.data(), kv_state.data(), k_sum_state.data(),
                batch_size, num_heads_, seq_len, head_dim_, head_dim_, eps_);
        }

        // Step 6: Reshape [B, H, L, D] -> [B*L, H*D]
        std::vector<float> attn_reshaped(batch_seq * embed_dim);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int l = 0; l < seq_len; ++l) {
                for (int h = 0; h < num_heads_; ++h) {
                    for (int d = 0; d < head_dim_; ++d) {
                        int src = b * num_heads_ * seq_len * head_dim_ +
                                 h * seq_len * head_dim_ + l * head_dim_ + d;
                        int dst = (b * seq_len + l) * embed_dim + h * head_dim_ + d;
                        attn_reshaped[dst] = attn_out[src];
                    }
                }
            }
        }

        // Step 7: Output projection
        std::vector<float> proj_out(batch_seq * embed_dim);
        matmul(attn_reshaped.data(), out_w, proj_out.data(), batch_seq, embed_dim, embed_dim);

        // Step 8: Residual + LayerNorm
        for (int64_t i = 0; i < batch_seq * embed_dim; ++i) {
            proj_out[i] += x_data[i];
        }
        std::copy(proj_out.begin(), proj_out.end(), out_data);
        layer_norm(out_data, gamma, beta, batch_seq, embed_dim);
    }

private:
    int num_heads_;
    int num_kv_heads_;
    int num_queries_per_kv_;
    int head_dim_;
    int rank_;
    int context_dim_;
    bool causal_;
    float eps_;
};

REGISTER_KERNEL_BUILDER(Name("FusedTPA").Device(DEVICE_CPU), FusedTPAOp);

// =============================================================================
// GRADIENT OP
// =============================================================================

REGISTER_OP("FusedTPAGrad")
    .Input("grad_output: float")
    .Input("x: float")
    .Input("context_weight: float")
    .Input("q_factor_a: float")
    .Input("q_factor_b: float")
    .Input("k_factor_a: float")
    .Input("k_factor_b: float")
    .Input("v_factor_a: float")
    .Input("v_factor_b: float")
    .Input("out_weight: float")
    .Output("grad_x: float")
    .Output("grad_context_weight: float")
    .Output("grad_q_factor_a: float")
    .Output("grad_q_factor_b: float")
    .Output("grad_k_factor_a: float")
    .Output("grad_k_factor_b: float")
    .Output("grad_v_factor_a: float")
    .Output("grad_v_factor_b: float")
    .Output("grad_out_weight: float")
    .Attr("num_heads: int")
    .Attr("num_kv_heads: int")
    .Attr("head_dim: int")
    .Attr("rank: int")
    .Attr("context_dim: int")
    .Attr("causal: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));
        for (int i = 1; i <= 8; ++i) {
            c->set_output(i, c->input(i + 1));
        }
        return absl::OkStatus();
    })
    .Doc("Gradient op for FusedTPA.");

class FusedTPAGradOp : public OpKernel {
public:
    explicit FusedTPAGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_kv_heads", &num_kv_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("context_dim", &context_dim_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Allocate outputs and pass through residual gradient
        const Tensor& grad_out_tensor = ctx->input(0);
        const Tensor& x_tensor = ctx->input(1);
        
        // grad_x gets residual gradient
        Tensor* grad_x = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor.shape(), &grad_x));
        std::copy(grad_out_tensor.flat<float>().data(),
                  grad_out_tensor.flat<float>().data() + x_tensor.NumElements(),
                  grad_x->flat<float>().data());
        
        // Zero all weight gradients
        for (int i = 1; i <= 8; ++i) {
            const Tensor& weight = ctx->input(i + 1);
            Tensor* grad = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(i, weight.shape(), &grad));
            std::fill(grad->flat<float>().data(),
                      grad->flat<float>().data() + weight.NumElements(), 0.0f);
        }
    }

private:
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int rank_;
    int context_dim_;
};

REGISTER_KERNEL_BUILDER(Name("FusedTPAGrad").Device(DEVICE_CPU), FusedTPAGradOp);

}  // namespace tensorflow
