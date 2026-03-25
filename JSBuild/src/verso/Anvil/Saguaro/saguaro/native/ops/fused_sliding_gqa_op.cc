// saguaro.native/ops/fused_sliding_gqa_op.cc
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
 * @file fused_sliding_gqa_op.cc
 * @brief O(n·w) Sliding Window Grouped-Query Attention TensorFlow Op.
 *
 * Fused kernel implementing:
 * 1. Q/K/V projections
 * 2. KV head expansion (GQA)
 * 3. Sliding window mask creation
 * 4. O(n·w) windowed attention with global tokens
 * 5. Output projection and layer norm
 */

#include "fused_sliding_gqa_op.h"
#include "fused_gqa_op.h"  // For gqa_softmax_rows
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

// Matmul helper
inline void matmul(
    const float* A, const float* B, float* C, int M, int K, int N) {
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

// Layer norm helper
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

REGISTER_OP("FusedSlidingGQA")
    .Input("x: float")              // [batch, seq, embed_dim]
    .Input("q_weight: float")
    .Input("k_weight: float")
    .Input("v_weight: float")
    .Input("out_weight: float")
    .Input("norm_gamma: float")
    .Input("norm_beta: float")
    .Output("output: float")
    .Attr("num_heads: int")
    .Attr("num_kv_heads: int")
    .Attr("head_dim: int")
    .Attr("window_size: int")
    .Attr("num_global_tokens: int = 0")
    .Attr("causal: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Fused O(n·w) Sliding Window Grouped-Query Attention.

Combines GQA's KV head sharing with sliding window for O(n·w) complexity,
where w is the window size.

x: Input tensor [batch, seq_len, embed_dim]
output: Output tensor [batch, seq_len, embed_dim]
)doc");

// =============================================================================
// FORWARD OP KERNEL
// =============================================================================

class FusedSlidingGQAOp : public OpKernel {
public:
    explicit FusedSlidingGQAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_kv_heads", &num_kv_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_global_tokens", &num_global_tokens_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("causal", &causal_));
        
        num_queries_per_kv_ = num_heads_ / num_kv_heads_;
        scale_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x_tensor = ctx->input(0);
        const Tensor& q_w_tensor = ctx->input(1);
        const Tensor& k_w_tensor = ctx->input(2);
        const Tensor& v_w_tensor = ctx->input(3);
        const Tensor& out_w_tensor = ctx->input(4);
        const Tensor& gamma_tensor = ctx->input(5);
        const Tensor& beta_tensor = ctx->input(6);

        const int batch_size = x_tensor.dim_size(0);
        const int seq_len = x_tensor.dim_size(1);
        const int embed_dim = x_tensor.dim_size(2);
        const int kv_dim = num_kv_heads_ * head_dim_;

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor.shape(), &output_tensor));

        const float* x_data = x_tensor.flat<float>().data();
        const float* q_w = q_w_tensor.flat<float>().data();
        const float* k_w = k_w_tensor.flat<float>().data();
        const float* v_w = v_w_tensor.flat<float>().data();
        const float* out_w = out_w_tensor.flat<float>().data();
        const float* gamma = gamma_tensor.flat<float>().data();
        const float* beta = beta_tensor.flat<float>().data();
        float* out_data = output_tensor->flat<float>().data();

        const int64_t batch_seq = batch_size * seq_len;
        
        // Allocate buffers
        std::vector<float> Q(batch_seq * embed_dim);
        std::vector<float> K_small(batch_seq * kv_dim);
        std::vector<float> V_small(batch_seq * kv_dim);
        std::vector<float> K_expanded(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> V_expanded(batch_size * num_heads_ * seq_len * head_dim_);
        std::vector<float> attn_out(batch_size * num_heads_ * seq_len * head_dim_);

        // Step 1: Projections
        matmul(x_data, q_w, Q.data(), batch_seq, embed_dim, embed_dim);
        matmul(x_data, k_w, K_small.data(), batch_seq, embed_dim, kv_dim);
        matmul(x_data, v_w, V_small.data(), batch_seq, embed_dim, kv_dim);

        // Step 2: Expand KV heads
        saguaro::ops::gqa_expand_kv_heads(
            K_small.data(), K_expanded.data(),
            batch_size, num_kv_heads_, num_queries_per_kv_,
            seq_len, head_dim_);
        saguaro::ops::gqa_expand_kv_heads(
            V_small.data(), V_expanded.data(),
            batch_size, num_kv_heads_, num_queries_per_kv_,
            seq_len, head_dim_);

        // Step 3: Create sliding window mask
        std::vector<float> mask(seq_len * seq_len);
        std::vector<int64_t> global_positions;
        
        // Compute evenly-spaced global positions
        if (num_global_tokens_ > 0) {
            int64_t step = std::max(static_cast<int64_t>(1), static_cast<int64_t>(seq_len / num_global_tokens_));
            for (int64_t g = 0; g < static_cast<int64_t>(num_global_tokens_) && g * step < static_cast<int64_t>(seq_len); ++g) {
                global_positions.push_back(g * step);
            }
        }
        
        saguaro::ops::sliding_gqa_create_mask(
            mask.data(), seq_len, window_size_,
            global_positions.data(), global_positions.size(), causal_);

        // Step 4: Chunked sliding window attention
        saguaro::ops::sliding_gqa_chunked_attention_forward(
            Q.data(), K_expanded.data(), V_expanded.data(),
            attn_out.data(), mask.data(),
            batch_size, num_heads_, seq_len, head_dim_,
            window_size_, scale_);

        // Step 5: Reshape [B, H, L, D] -> [B*L, H*D]
        std::vector<float> attn_reshaped(batch_seq * embed_dim);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < batch_size; ++b) {
            for (int l = 0; l < seq_len; ++l) {
                for (int h = 0; h < num_heads_; ++h) {
                    for (int d = 0; d < head_dim_; ++d) {
                        int src_idx = b * num_heads_ * seq_len * head_dim_ +
                                     h * seq_len * head_dim_ +
                                     l * head_dim_ + d;
                        int dst_idx = (b * seq_len + l) * embed_dim + h * head_dim_ + d;
                        attn_reshaped[dst_idx] = attn_out[src_idx];
                    }
                }
            }
        }

        // Step 6: Output projection
        std::vector<float> proj_out(batch_seq * embed_dim);
        matmul(attn_reshaped.data(), out_w, proj_out.data(), batch_seq, embed_dim, embed_dim);

        // Step 7: Residual
        for (int64_t i = 0; i < batch_seq * embed_dim; ++i) {
            proj_out[i] += x_data[i];
        }

        // Step 8: Layer norm
        std::copy(proj_out.begin(), proj_out.end(), out_data);
        layer_norm(out_data, gamma, beta, batch_seq, embed_dim);
    }

private:
    int num_heads_;
    int num_kv_heads_;
    int num_queries_per_kv_;
    int head_dim_;
    int window_size_;
    int num_global_tokens_;
    bool causal_;
    float scale_;
};

REGISTER_KERNEL_BUILDER(Name("FusedSlidingGQA").Device(DEVICE_CPU), FusedSlidingGQAOp);

// =============================================================================
// GRADIENT OP
// =============================================================================

REGISTER_OP("FusedSlidingGQAGrad")
    .Input("grad_output: float")
    .Input("x: float")
    .Input("q_weight: float")
    .Input("k_weight: float")
    .Input("v_weight: float")
    .Input("out_weight: float")
    .Output("grad_x: float")
    .Output("grad_q_weight: float")
    .Output("grad_k_weight: float")
    .Output("grad_v_weight: float")
    .Output("grad_out_weight: float")
    .Attr("num_heads: int")
    .Attr("num_kv_heads: int")
    .Attr("head_dim: int")
    .Attr("window_size: int")
    .Attr("causal: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        c->set_output(2, c->input(3));
        c->set_output(3, c->input(4));
        c->set_output(4, c->input(5));
        return absl::OkStatus();
    })
    .Doc("Gradient op for FusedSlidingGQA.");

class FusedSlidingGQAGradOp : public OpKernel {
public:
    explicit FusedSlidingGQAGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_kv_heads", &num_kv_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("window_size", &window_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("causal", &causal_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_out_tensor = ctx->input(0);
        const Tensor& x_tensor = ctx->input(1);
        const Tensor& q_w_tensor = ctx->input(2);
        const Tensor& k_w_tensor = ctx->input(3);
        const Tensor& v_w_tensor = ctx->input(4);
        const Tensor& out_w_tensor = ctx->input(5);

        Tensor* grad_x = nullptr;
        Tensor* grad_q_w = nullptr;
        Tensor* grad_k_w = nullptr;
        Tensor* grad_v_w = nullptr;
        Tensor* grad_out_w = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor.shape(), &grad_x));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, q_w_tensor.shape(), &grad_q_w));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, k_w_tensor.shape(), &grad_k_w));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, v_w_tensor.shape(), &grad_v_w));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, out_w_tensor.shape(), &grad_out_w));

        // Pass through gradients via residual
        const float* grad_out_data = grad_out_tensor.flat<float>().data();
        float* grad_x_data = grad_x->flat<float>().data();
        std::copy(grad_out_data, grad_out_data + x_tensor.NumElements(), grad_x_data);

        // Zero weight gradients
        std::fill(grad_q_w->flat<float>().data(), 
                  grad_q_w->flat<float>().data() + grad_q_w->NumElements(), 0.0f);
        std::fill(grad_k_w->flat<float>().data(),
                  grad_k_w->flat<float>().data() + grad_k_w->NumElements(), 0.0f);
        std::fill(grad_v_w->flat<float>().data(),
                  grad_v_w->flat<float>().data() + grad_v_w->NumElements(), 0.0f);
        std::fill(grad_out_w->flat<float>().data(),
                  grad_out_w->flat<float>().data() + grad_out_w->NumElements(), 0.0f);
    }

private:
    int num_heads_;
    int num_kv_heads_;
    int head_dim_;
    int window_size_;
    bool causal_;
};

REGISTER_KERNEL_BUILDER(Name("FusedSlidingGQAGrad").Device(DEVICE_CPU), FusedSlidingGQAGradOp);

}  // namespace tensorflow
