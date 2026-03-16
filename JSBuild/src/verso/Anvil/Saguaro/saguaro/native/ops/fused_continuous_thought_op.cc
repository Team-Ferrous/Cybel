// saguaro.native/ops/fused_continuous_thought_op.cc
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
 * @file fused_continuous_thought_op.cc
 * @brief COCONUT Continuous Thought Block TensorFlow Op.
 *
 * Implements Chain of Continuous Thought (COCONUT) for latent-space
 * reasoning without generating intermediate tokens. The model reasons
 * in continuous embedding space rather than discrete token space.
 *
 * Key Features:
 *   - Mean pooling for thought extraction
 *   - Iterative thought refinement (num_thought_steps iterations)
 *   - Gated broadcast back to sequence
 *   - SIMD-optimized GELU and LayerNorm
 *
 * Complexity: O(n + k·d²) where n=seq_len, k=thought_steps, d=dim
 */

#include "fused_continuous_thought_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "common/edition_limits.h"

#include <vector>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedContinuousThought")
    .Input("x: float32")
    .Input("input_norm_gamma: float32")
    .Input("input_norm_beta: float32")
    .Input("aggregator_weight: float32")
    .Input("aggregator_bias: float32")
    .Input("projector_norm_gamma: float32")
    .Input("projector_norm_beta: float32")
    .Input("projector_dense1_weight: float32")
    .Input("projector_dense1_bias: float32")
    .Input("projector_dense2_weight: float32")
    .Input("projector_dense2_bias: float32")
    .Input("broadcast_weight: float32")
    .Input("broadcast_bias: float32")
    .Input("gate_weight: float32")
    .Input("gate_bias: float32")
    .Input("output_norm_gamma: float32")
    .Input("output_norm_beta: float32")
    .Output("output: float32")
    .Output("thought_state: float32")
    .Attr("num_thought_steps: int = 4")
    .Attr("use_gating: bool = true")
    .SetShapeFn([](InferenceContext* c) {
        // output: same as input x [batch, seq_len, embed_dim]
        ShapeHandle input = c->input(0);
        c->set_output(0, input);
        // thought_state: [batch, embed_dim]
        auto batch = c->Dim(input, 0);
        auto embed_dim = c->Dim(input, 2);
        c->set_output(1, c->MakeShape({batch, embed_dim}));
        return Status();
    })
    .Doc(R"doc(
Fused Continuous Thought Block with SIMD optimization.
Performs COCONUT-style continuous thought reasoning in latent space.
)doc");

REGISTER_OP("FusedContinuousThoughtGrad")
    .Input("grad_output: float32")
    .Input("grad_thought_state: float32")
    .Input("x: float32")
    .Input("input_norm_gamma: float32")
    .Input("aggregator_weight: float32")
    .Input("projector_dense1_weight: float32")
    .Input("projector_dense2_weight: float32")
    .Input("broadcast_weight: float32")
    .Input("gate_weight: float32")
    .Attr("num_thought_steps: int = 4")
    .Attr("use_gating: bool = true")
    .Output("grad_x: float32")
    .Output("grad_input_norm_gamma: float32")
    .Output("grad_input_norm_beta: float32")
    .Output("grad_aggregator_weight: float32")
    .Output("grad_aggregator_bias: float32")
    .Output("grad_projector_norm_gamma: float32")
    .Output("grad_projector_norm_beta: float32")
    .Output("grad_projector_dense1_weight: float32")
    .Output("grad_projector_dense1_bias: float32")
    .Output("grad_projector_dense2_weight: float32")
    .Output("grad_projector_dense2_bias: float32")
    .Output("grad_broadcast_weight: float32")
    .Output("grad_broadcast_bias: float32")
    .Output("grad_gate_weight: float32")
    .Output("grad_gate_bias: float32")
    .Output("grad_output_norm_gamma: float32")
    .Output("grad_output_norm_beta: float32")
    .SetShapeFn([](InferenceContext* c) {
        // Gradients have same shapes as corresponding inputs
        c->set_output(0, c->input(2));   // grad_x
        c->set_output(1, c->input(3));   // grad_input_norm_gamma
        c->set_output(2, c->input(3));   // grad_input_norm_beta (same shape)
        c->set_output(3, c->input(4));   // grad_aggregator_weight
        // ... remaining shapes would need explicit handling
        return Status();
    });

// =============================================================================
// FORWARD KERNEL
// =============================================================================

namespace {

// Simple matrix multiply: C[M,N] = A[M,K] @ B[K,N]
void matmul(const float* A, const float* B, float* C,
            int64_t M, int64_t K, int64_t N) {
    #pragma omp parallel for
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Add bias to matrix: C[i,j] += bias[j]
void add_bias(float* C, const float* bias, int64_t rows, int64_t cols) {
    #pragma omp parallel for
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            C[i * cols + j] += bias[j];
        }
    }
}

}  // anonymous namespace

class FusedContinuousThoughtOp : public OpKernel {
 public:
    explicit FusedContinuousThoughtOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_thought_steps", &num_thought_steps_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_gating", &use_gating_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& x = ctx->input(0);
        const Tensor& input_norm_gamma = ctx->input(1);
        const Tensor& input_norm_beta = ctx->input(2);
        const Tensor& aggregator_weight = ctx->input(3);
        const Tensor& aggregator_bias = ctx->input(4);
        const Tensor& projector_norm_gamma = ctx->input(5);
        const Tensor& projector_norm_beta = ctx->input(6);
        const Tensor& projector_dense1_weight = ctx->input(7);
        const Tensor& projector_dense1_bias = ctx->input(8);
        const Tensor& projector_dense2_weight = ctx->input(9);
        const Tensor& projector_dense2_bias = ctx->input(10);
        const Tensor& broadcast_weight = ctx->input(11);
        const Tensor& broadcast_bias = ctx->input(12);
        const Tensor& gate_weight = ctx->input(13);
        const Tensor& gate_bias = ctx->input(14);
        const Tensor& output_norm_gamma = ctx->input(15);
        const Tensor& output_norm_beta = ctx->input(16);

        // Get dimensions
        const int64_t batch_size = x.dim_size(0);
        const int64_t seq_len = x.dim_size(1);
        const int64_t embed_dim = x.dim_size(2);
        const int64_t hidden_dim = projector_dense1_weight.dim_size(1);
        
        // HighNoon Lite Edition: Enforce embedding dimension limit (max 4096)
        SAGUARO_CHECK_EMBEDDING_DIM(ctx, embed_dim);

        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &output));
        
        Tensor* thought_state = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({batch_size, embed_dim}), &thought_state));

        // Get raw pointers
        const float* x_data = x.flat<float>().data();
        const float* ing = input_norm_gamma.flat<float>().data();
        const float* inb = input_norm_beta.flat<float>().data();
        const float* agg_w = aggregator_weight.flat<float>().data();
        const float* agg_b = aggregator_bias.flat<float>().data();
        const float* png = projector_norm_gamma.flat<float>().data();
        const float* pnb = projector_norm_beta.flat<float>().data();
        const float* pd1_w = projector_dense1_weight.flat<float>().data();
        const float* pd1_b = projector_dense1_bias.flat<float>().data();
        const float* pd2_w = projector_dense2_weight.flat<float>().data();
        const float* pd2_b = projector_dense2_bias.flat<float>().data();
        const float* bc_w = broadcast_weight.flat<float>().data();
        const float* bc_b = broadcast_bias.flat<float>().data();
        const float* gate_w = gate_weight.flat<float>().data();
        const float* gate_b = gate_bias.flat<float>().data();
        const float* ong = output_norm_gamma.flat<float>().data();
        const float* onb = output_norm_beta.flat<float>().data();
        
        float* out_data = output->flat<float>().data();
        float* thought_data = thought_state->flat<float>().data();

        // Allocate temporary buffers
        std::vector<float> normed(batch_size * seq_len * embed_dim);
        std::vector<float> thought(batch_size * embed_dim);
        std::vector<float> thought_norm(batch_size * embed_dim);
        std::vector<float> thought_hidden(batch_size * hidden_dim);
        std::vector<float> thought_refined(batch_size * embed_dim);
        std::vector<float> thought_broadcast(batch_size * embed_dim);
        std::vector<float> gate_values(batch_size * seq_len * embed_dim);

        // Step 1: Input layer normalization
        saguaro::ops::continuous_thought_layer_norm(
            x_data, ing, inb, normed.data(),
            batch_size * seq_len, embed_dim);

        // Step 2: Mean pool over sequence to get thought seed
        saguaro::ops::continuous_thought_mean_pool(
            normed.data(), thought.data(),
            batch_size, seq_len, embed_dim);

        // Step 3: Aggregator projection
        matmul(thought.data(), agg_w, thought_refined.data(),
               batch_size, embed_dim, embed_dim);
        add_bias(thought_refined.data(), agg_b, batch_size, embed_dim);

        // Copy refined thought for iterative processing
        std::copy(thought_refined.begin(), thought_refined.end(), thought.begin());

        // Step 4: Iterative thought refinement
        for (int step = 0; step < num_thought_steps_; ++step) {
            // Pre-norm
            saguaro::ops::continuous_thought_layer_norm(
                thought.data(), png, pnb, thought_norm.data(),
                batch_size, embed_dim);

            // Dense 1: up projection with GELU
            matmul(thought_norm.data(), pd1_w, thought_hidden.data(),
                   batch_size, embed_dim, hidden_dim);
            add_bias(thought_hidden.data(), pd1_b, batch_size, hidden_dim);
            saguaro::ops::continuous_thought_gelu(
                thought_hidden.data(), thought_hidden.data(),
                batch_size * hidden_dim);

            // Dense 2: down projection
            matmul(thought_hidden.data(), pd2_w, thought_refined.data(),
                   batch_size, hidden_dim, embed_dim);
            add_bias(thought_refined.data(), pd2_b, batch_size, embed_dim);

            // Residual connection
            saguaro::ops::continuous_thought_add(
                thought.data(), thought_refined.data(), thought.data(),
                batch_size * embed_dim);
        }

        // Step 5: Broadcast projection
        matmul(thought.data(), bc_w, thought_broadcast.data(),
               batch_size, embed_dim, embed_dim);
        add_bias(thought_broadcast.data(), bc_b, batch_size, embed_dim);

        // Step 6: Gated residual or simple addition
        if (use_gating_) {
            // Compute gate values from original input
            matmul(x_data, gate_w, gate_values.data(),
                   batch_size * seq_len, embed_dim, embed_dim);
            add_bias(gate_values.data(), gate_b, batch_size * seq_len, embed_dim);
            saguaro::ops::continuous_thought_sigmoid(
                gate_values.data(), gate_values.data(),
                batch_size * seq_len * embed_dim);

            // Gated broadcast: out = x + gate * thought_broadcast
            saguaro::ops::continuous_thought_gated_broadcast(
                x_data, thought_broadcast.data(), gate_values.data(),
                out_data, batch_size, seq_len, embed_dim);
        } else {
            // Simple broadcast: out = x + thought_broadcast
            saguaro::ops::continuous_thought_broadcast_add(
                x_data, thought_broadcast.data(),
                out_data, batch_size, seq_len, embed_dim);
        }

        // Step 7: Output layer normalization
        saguaro::ops::continuous_thought_layer_norm(
            out_data, ong, onb, out_data,
            batch_size * seq_len, embed_dim);

        // Copy final thought state to output
        std::copy(thought.begin(), thought.end(), thought_data);
    }

 private:
    int num_thought_steps_;
    bool use_gating_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedContinuousThought").Device(DEVICE_CPU),
    FusedContinuousThoughtOp);

// =============================================================================
// GRADIENT KERNEL
// =============================================================================

class FusedContinuousThoughtGradOp : public OpKernel {
 public:
    explicit FusedContinuousThoughtGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_thought_steps", &num_thought_steps_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_gating", &use_gating_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& grad_output = ctx->input(0);
        const Tensor& grad_thought_state = ctx->input(1);
        const Tensor& x = ctx->input(2);
        const Tensor& input_norm_gamma = ctx->input(3);
        const Tensor& aggregator_weight = ctx->input(4);
        const Tensor& projector_dense1_weight = ctx->input(5);
        const Tensor& projector_dense2_weight = ctx->input(6);
        const Tensor& broadcast_weight = ctx->input(7);
        const Tensor& gate_weight = ctx->input(8);

        const int64_t batch_size = x.dim_size(0);
        const int64_t seq_len = x.dim_size(1);
        const int64_t embed_dim = x.dim_size(2);
        const int64_t hidden_dim = projector_dense1_weight.dim_size(1);

        // Allocate all gradient outputs
        Tensor* grad_x = nullptr;
        Tensor* grad_input_norm_gamma = nullptr;
        Tensor* grad_input_norm_beta = nullptr;
        Tensor* grad_aggregator_weight = nullptr;
        Tensor* grad_aggregator_bias = nullptr;
        Tensor* grad_projector_norm_gamma = nullptr;
        Tensor* grad_projector_norm_beta = nullptr;
        Tensor* grad_projector_dense1_weight = nullptr;
        Tensor* grad_projector_dense1_bias = nullptr;
        Tensor* grad_projector_dense2_weight = nullptr;
        Tensor* grad_projector_dense2_bias = nullptr;
        Tensor* grad_broadcast_weight = nullptr;
        Tensor* grad_broadcast_bias = nullptr;
        Tensor* grad_gate_weight = nullptr;
        Tensor* grad_gate_bias = nullptr;
        Tensor* grad_output_norm_gamma = nullptr;
        Tensor* grad_output_norm_beta = nullptr;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &grad_x));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input_norm_gamma.shape(), &grad_input_norm_gamma));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, input_norm_gamma.shape(), &grad_input_norm_beta));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, aggregator_weight.shape(), &grad_aggregator_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, TensorShape({embed_dim}), &grad_aggregator_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(5, input_norm_gamma.shape(), &grad_projector_norm_gamma));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(6, input_norm_gamma.shape(), &grad_projector_norm_beta));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(7, projector_dense1_weight.shape(), &grad_projector_dense1_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(8, TensorShape({hidden_dim}), &grad_projector_dense1_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(9, projector_dense2_weight.shape(), &grad_projector_dense2_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(10, TensorShape({embed_dim}), &grad_projector_dense2_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(11, broadcast_weight.shape(), &grad_broadcast_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(12, TensorShape({embed_dim}), &grad_broadcast_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(13, gate_weight.shape(), &grad_gate_weight));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(14, TensorShape({embed_dim}), &grad_gate_bias));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(15, input_norm_gamma.shape(), &grad_output_norm_gamma));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(16, input_norm_gamma.shape(), &grad_output_norm_beta));

        // Initialize all gradients to zero
        // For now, use identity gradient through residual path
        // Full gradient implementation would require storing intermediates
        
        // Copy grad_output to grad_x as identity gradient for residual connection
        const float* grad_out_data = grad_output.flat<float>().data();
        float* grad_x_data = grad_x->flat<float>().data();
        std::copy(grad_out_data, grad_out_data + batch_size * seq_len * embed_dim, grad_x_data);

        // Zero out all weight/bias gradients (simplified - full implementation needed)
        auto zero_tensor = [](Tensor* t) {
            std::fill(t->flat<float>().data(), 
                      t->flat<float>().data() + t->NumElements(), 0.0f);
        };
        
        zero_tensor(grad_input_norm_gamma);
        zero_tensor(grad_input_norm_beta);
        zero_tensor(grad_aggregator_weight);
        zero_tensor(grad_aggregator_bias);
        zero_tensor(grad_projector_norm_gamma);
        zero_tensor(grad_projector_norm_beta);
        zero_tensor(grad_projector_dense1_weight);
        zero_tensor(grad_projector_dense1_bias);
        zero_tensor(grad_projector_dense2_weight);
        zero_tensor(grad_projector_dense2_bias);
        zero_tensor(grad_broadcast_weight);
        zero_tensor(grad_broadcast_bias);
        zero_tensor(grad_gate_weight);
        zero_tensor(grad_gate_bias);
        zero_tensor(grad_output_norm_gamma);
        zero_tensor(grad_output_norm_beta);
    }

 private:
    int num_thought_steps_;
    bool use_gating_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedContinuousThoughtGrad").Device(DEVICE_CPU),
    FusedContinuousThoughtGradOp);

}  // namespace tensorflow
