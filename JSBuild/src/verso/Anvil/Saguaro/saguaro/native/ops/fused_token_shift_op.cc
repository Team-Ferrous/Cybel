// saguaro.native/ops/fused_token_shift_op.cc
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
 * @file fused_token_shift_op.cc
 * @brief TensorFlow custom op for RWKV-6 style data-dependent token shifting.
 *
 * This file implements the forward and backward kernels for the
 * DataDependentTokenShift layer with SIMD optimization.
 *
 * NOTE: Gate projection (input @ gate_kernel + gate_bias) must be computed
 * in Python and passed as gate_proj. This follows the same pattern as
 * fused_min_gru_op which receives pre-computed projections.
 *
 * Ops registered:
 * - FusedTokenShift: Forward pass
 * - FusedTokenShiftGrad: Backward pass for gradient computation
 */

#include "fused_token_shift_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// Op Registration
// =============================================================================

REGISTER_OP("FusedTokenShift")
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("gate_proj: float")       // [batch, seq_len, embedding_dim] PRE-COMPUTED
    .Input("decay_weights: float")   // [embedding_dim] (can be empty if not used)
    .Attr("use_learned_decay: bool = true")
    .Output("output: float")         // [batch, seq_len, embedding_dim]
    .Output("gate: float")           // [batch, seq_len, embedding_dim] (saved for backward)
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input_shape = c->input(0);
        c->set_output(0, input_shape);  // output same shape as input
        c->set_output(1, input_shape);  // gate same shape as input
        return OkStatus();
    })
    .Doc(R"doc(
Fused data-dependent token shift operation.

RWKV-6 style adaptive token shifting for improved long-term memory.
Uses input-dependent gating to mix current and previous token states.

Forward computation:
  gate = sigmoid(gate_proj)  # gate_proj = input @ gate_kernel + gate_bias
  if use_learned_decay:
    gate = gate * sigmoid(decay_weights)
  output = gate * input + (1 - gate) * prev_input

input: Input tensor of shape [batch, seq_len, embedding_dim].
prev_input: Previous token states of same shape.
gate_proj: Pre-computed gate projection [batch, seq_len, embedding_dim].
decay_weights: Learned decay weights [embedding_dim].
use_learned_decay: Whether to apply learned decay.
output: Shifted output tensor.
gate: Gate values (saved for backward pass).
)doc");

REGISTER_OP("FusedTokenShiftGrad")
    .Input("grad_output: float")     // [batch, seq_len, embedding_dim]
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("gate_proj: float")       // [batch, seq_len, embedding_dim]
    .Input("decay_weights: float")   // [embedding_dim]
    .Input("gate: float")            // [batch, seq_len, embedding_dim] (from forward)
    .Attr("use_learned_decay: bool = true")
    .Output("grad_input: float")           // [batch, seq_len, embedding_dim]
    .Output("grad_prev_input: float")      // [batch, seq_len, embedding_dim]
    .Output("grad_gate_proj: float")       // [batch, seq_len, embedding_dim]
    .Output("grad_decay_weights: float")   // [embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input_shape = c->input(1);
        c->set_output(0, input_shape);  // grad_input
        c->set_output(1, input_shape);  // grad_prev_input
        c->set_output(2, input_shape);  // grad_gate_proj
        
        ShapeHandle decay_shape = c->input(4);
        c->set_output(3, decay_shape);  // grad_decay_weights
        return OkStatus();
    });

// =============================================================================
// Forward Kernel Implementation
// =============================================================================

namespace {

/**
 * @brief Token shift forward pass with SIMD optimization.
 *
 * Computes: gate = sigmoid(gate_proj) [* sigmoid(decay)] 
 *           output = gate * x + (1 - gate) * prev
 */
void TokenShiftForward(
    const float* input,
    const float* prev_input,
    const float* gate_proj,      // Pre-computed linear projection
    const float* decay_weights,
    float* output,
    float* gate_out,
    int batch_size,
    int seq_len,
    int embedding_dim,
    bool use_learned_decay) {
    
    const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len;
    const int64_t dim = embedding_dim;
    
    // Pre-compute learned decay if enabled
    std::vector<float> learned_decay(dim, 1.0f);
    if (use_learned_decay && decay_weights != nullptr) {
        for (int64_t d = 0; d < dim; ++d) {
            learned_decay[d] = 1.0f / (1.0f + std::exp(-decay_weights[d]));
        }
    }
    
    #pragma omp parallel for
    for (int64_t t = 0; t < total_tokens; ++t) {
        const int64_t offset = t * dim;
        const float* x = input + offset;
        const float* prev = prev_input + offset;
        const float* gate_linear = gate_proj + offset;  // Pre-computed
        float* out = output + offset;
        float* gate = gate_out + offset;
        
        // Copy gate_proj to gate buffer
        std::copy(gate_linear, gate_linear + dim, gate);
        
        // Apply sigmoid in-place
        saguaro::ops::simd_sigmoid_inplace(gate, dim);
        
        // Apply learned decay if enabled
        if (use_learned_decay) {
            saguaro::ops::simd_mul(gate, learned_decay.data(), gate, dim);
        }
        
        // Token mixing: out = gate * x + (1 - gate) * prev
        saguaro::ops::simd_token_mix(gate, x, prev, out, dim);
    }
}

/**
 * @brief Token shift backward pass with SIMD optimization.
 *
 * Computes gradients w.r.t. input, prev_input, gate_proj, and decay_weights.
 */
void TokenShiftBackward(
    const float* grad_output,
    const float* input,
    const float* prev_input,
    const float* gate_proj,
    const float* decay_weights,
    const float* gate,  // Forward gate output
    float* grad_input,
    float* grad_prev,
    float* grad_gate_proj,
    float* grad_decay_weights,
    int batch_size,
    int seq_len,
    int embedding_dim,
    bool use_learned_decay) {
    
    const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len;
    const int64_t dim = embedding_dim;
    
    // Pre-compute learned decay and its gradient multipliers
    std::vector<float> learned_decay(dim, 1.0f);
    std::vector<float> decay_grad_mult(dim, 0.0f);  // sigmoid'(decay) for chain rule
    
    if (use_learned_decay && decay_weights != nullptr) {
        for (int64_t d = 0; d < dim; ++d) {
            float sig = 1.0f / (1.0f + std::exp(-decay_weights[d]));
            learned_decay[d] = sig;
            decay_grad_mult[d] = sig * (1.0f - sig);  // sigmoid derivative
        }
    }
    
    // Initialize gradient accumulators for decay weights
    std::memset(grad_decay_weights, 0, dim * sizeof(float));
    
// Thread-local storage for gradient accumulation
    const int num_threads = 
#ifdef _OPENMP
        omp_get_max_threads();
#else
        1;
#endif
    
    std::vector<std::vector<float>> local_grad_decay(num_threads, std::vector<float>(dim, 0.0f));
    
    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<float> gate_pre_decay(dim);
        std::vector<float> grad_gate_pre_decay(dim);
        
        #pragma omp for
        for (int64_t t = 0; t < total_tokens; ++t) {
            const int64_t offset = t * dim;
            
            // Reconstruct gate_pre_decay = sigmoid(gate_proj) 
            std::copy(gate_proj + offset, gate_proj + offset + dim, gate_pre_decay.data());
            saguaro::ops::simd_sigmoid_inplace(gate_pre_decay.data(), dim);
            
            for (int64_t d = 0; d < dim; ++d) {
                float g = gate[offset + d];  // This is gate_pre_decay * learned_decay
                float x_val = input[offset + d];
                float prev_val = prev_input[offset + d];
                float dL_dout = grad_output[offset + d];
                
                // Gradient w.r.t. gate: dL/dg = dL/dout * (x - prev)
                float dL_dg = dL_dout * (x_val - prev_val);
                
                // Gradient w.r.t. input: dL/dx = dL/dout * gate
                grad_input[offset + d] = dL_dout * g;
                
                // Gradient w.r.t. prev_input: dL/dprev = dL/dout * (1 - gate)
                grad_prev[offset + d] = dL_dout * (1.0f - g);
                
                if (use_learned_decay) {
                    // Gate = sigmoid(gate_proj) * sigmoid(decay)
                    // dL/d(sigmoid(gate_proj)) = dL/dg * sigmoid(decay)
                    float sig_decay = learned_decay[d];
                    float gate_pre = gate_pre_decay[d];
                    
                    // dL/d(gate_proj) = dL/d(sigmoid(gate_proj)) * sigmoid'(gate_proj)
                    //                 = dL/dg * sigmoid(decay) * gate_pre * (1 - gate_pre)
                    float sigmoid_deriv = gate_pre * (1.0f - gate_pre);
                    grad_gate_proj[offset + d] = dL_dg * sig_decay * sigmoid_deriv;
                    
                    // dL/d(decay_weights) += dL/dg * gate_pre * sigmoid'(decay)
                    local_grad_decay[tid][d] += dL_dg * gate_pre * decay_grad_mult[d];
                } else {
                    // Gate = sigmoid(gate_proj)
                    float gate_pre = gate_pre_decay[d];
                    float sigmoid_deriv = gate_pre * (1.0f - gate_pre);
                    grad_gate_proj[offset + d] = dL_dg * sigmoid_deriv;
                }
            }
        }
    }
    
    // Reduce thread-local decay gradients
    if (use_learned_decay) {
        for (int tid = 0; tid < num_threads; ++tid) {
            for (int64_t d = 0; d < dim; ++d) {
                grad_decay_weights[d] += local_grad_decay[tid][d];
            }
        }
    }
}

}  // namespace

// =============================================================================
// OpKernel Classes
// =============================================================================

class FusedTokenShiftOp : public OpKernel {
 public:
    explicit FusedTokenShiftOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_learned_decay", &use_learned_decay_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& input = ctx->input(0);
        const Tensor& prev_input = ctx->input(1);
        const Tensor& gate_proj = ctx->input(2);  // Pre-computed
        const Tensor& decay_weights = ctx->input(3);
        
        // Validate shapes
        OP_REQUIRES(ctx, input.dims() == 3,
            errors::InvalidArgument("input must be 3D [batch, seq_len, dim]"));
        OP_REQUIRES(ctx, prev_input.shape() == input.shape(),
            errors::InvalidArgument("prev_input must have same shape as input"));
        OP_REQUIRES(ctx, gate_proj.shape() == input.shape(),
            errors::InvalidArgument("gate_proj must have same shape as input"));
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        
        // Allocate outputs
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        
        Tensor* gate_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &gate_out));
        
        // Run forward kernel
        TokenShiftForward(
            input.flat<float>().data(),
            prev_input.flat<float>().data(),
            gate_proj.flat<float>().data(),
            use_learned_decay_ ? decay_weights.flat<float>().data() : nullptr,
            output->flat<float>().data(),
            gate_out->flat<float>().data(),
            batch_size,
            seq_len,
            embedding_dim,
            use_learned_decay_
        );
    }

 private:
    bool use_learned_decay_;
};

class FusedTokenShiftGradOp : public OpKernel {
 public:
    explicit FusedTokenShiftGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_learned_decay", &use_learned_decay_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& grad_output = ctx->input(0);
        const Tensor& input = ctx->input(1);
        const Tensor& prev_input = ctx->input(2);
        const Tensor& gate_proj = ctx->input(3);
        const Tensor& decay_weights = ctx->input(4);
        const Tensor& gate = ctx->input(5);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        
        // Allocate gradient outputs
        Tensor* grad_input = nullptr;
        Tensor* grad_prev = nullptr;
        Tensor* grad_gate_proj = nullptr;
        Tensor* grad_decay = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, prev_input.shape(), &grad_prev));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, gate_proj.shape(), &grad_gate_proj));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, decay_weights.shape(), &grad_decay));
        
        // Run backward kernel
        TokenShiftBackward(
            grad_output.flat<float>().data(),
            input.flat<float>().data(),
            prev_input.flat<float>().data(),
            gate_proj.flat<float>().data(),
            use_learned_decay_ ? decay_weights.flat<float>().data() : nullptr,
            gate.flat<float>().data(),
            grad_input->flat<float>().data(),
            grad_prev->flat<float>().data(),
            grad_gate_proj->flat<float>().data(),
            grad_decay->flat<float>().data(),
            batch_size,
            seq_len,
            embedding_dim,
            use_learned_decay_
        );
    }

 private:
    bool use_learned_decay_;
};

// =============================================================================
// Kernel Registration
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("FusedTokenShift").Device(DEVICE_CPU),
    FusedTokenShiftOp);

REGISTER_KERNEL_BUILDER(
    Name("FusedTokenShiftGrad").Device(DEVICE_CPU),
    FusedTokenShiftGradOp);

// =============================================================================
// ENHANCEMENT 1: SIMPLIFIED TOKEN SHIFT (RWKV-7 STYLE)
// 3x faster - no input-dependent gating, uses only learned decay weights
// =============================================================================

REGISTER_OP("FusedSimplifiedTokenShift")
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("decay_weights: float")   // [embedding_dim]
    .Output("output: float")         // [batch, seq_len, embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return OkStatus();
    })
    .Doc(R"doc(
RWKV-7 style simplified token shift operation.

Uses only learned decay weights (no input-dependent gate) for 3x speedup.
Forward computation:
  gate = sigmoid(decay_weights)  # Fixed per dimension
  output = gate * input + (1 - gate) * prev_input

input: Input tensor of shape [batch, seq_len, embedding_dim].
prev_input: Previous token states of same shape.
decay_weights: Learned decay weights [embedding_dim].
output: Shifted output tensor.
)doc");

REGISTER_OP("FusedSimplifiedTokenShiftGrad")
    .Input("grad_output: float")     // [batch, seq_len, embedding_dim]
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("decay_weights: float")   // [embedding_dim]
    .Output("grad_input: float")           // [batch, seq_len, embedding_dim]
    .Output("grad_prev_input: float")      // [batch, seq_len, embedding_dim]
    .Output("grad_decay_weights: float")   // [embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input
        c->set_output(1, c->input(2));  // grad_prev_input
        c->set_output(2, c->input(3));  // grad_decay_weights
        return OkStatus();
    });

namespace {

void SimplifiedTokenShiftForward(
    const float* input,
    const float* prev_input,
    const float* decay_weights,
    float* output,
    int batch_size,
    int seq_len,
    int embedding_dim) {
    
    const int64_t batch_seq = static_cast<int64_t>(batch_size) * seq_len;
    const int64_t dim = embedding_dim;
    
    // Pre-compute sigmoid(decay_weights)
    std::vector<float> decay(dim);
    for (int64_t d = 0; d < dim; ++d) {
        decay[d] = 1.0f / (1.0f + std::exp(-decay_weights[d]));
    }
    
    // Use SIMD-optimized simplified mix
    saguaro::ops::token_shift_simplified_mix(
        decay.data(), input, prev_input, output,
        batch_seq, dim);
}

void SimplifiedTokenShiftBackward(
    const float* grad_output,
    const float* input,
    const float* prev_input,
    const float* decay_weights,
    float* grad_input,
    float* grad_prev,
    float* grad_decay_weights,
    int batch_size,
    int seq_len,
    int embedding_dim) {
    
    const int64_t batch_seq = static_cast<int64_t>(batch_size) * seq_len;
    const int64_t dim = embedding_dim;
    
    // Pre-compute sigmoid and its derivative
    std::vector<float> decay(dim);
    std::vector<float> decay_deriv(dim);
    for (int64_t d = 0; d < dim; ++d) {
        float sig = 1.0f / (1.0f + std::exp(-decay_weights[d]));
        decay[d] = sig;
        decay_deriv[d] = sig * (1.0f - sig);
    }
    
    // Zero initialize gradient accumulator
    std::memset(grad_decay_weights, 0, dim * sizeof(float));
    
    // Compute gradients
    #pragma omp parallel
    {
        std::vector<float> local_grad_decay(dim, 0.0f);
        
        #pragma omp for
        for (int64_t t = 0; t < batch_seq; ++t) {
            const int64_t offset = t * dim;
            
            for (int64_t d = 0; d < dim; ++d) {
                float g = decay[d];
                float dL_dout = grad_output[offset + d];
                float x_val = input[offset + d];
                float prev_val = prev_input[offset + d];
                
                // grad_input = dL/dout * gate
                grad_input[offset + d] = dL_dout * g;
                
                // grad_prev = dL/dout * (1 - gate)
                grad_prev[offset + d] = dL_dout * (1.0f - g);
                
                // grad_decay = dL/dout * (x - prev) * sigmoid'(decay)
                local_grad_decay[d] += dL_dout * (x_val - prev_val) * decay_deriv[d];
            }
        }
        
        // Reduce to global gradient
        #pragma omp critical
        {
            for (int64_t d = 0; d < dim; ++d) {
                grad_decay_weights[d] += local_grad_decay[d];
            }
        }
    }
}

}  // namespace

class FusedSimplifiedTokenShiftOp : public OpKernel {
 public:
    explicit FusedSimplifiedTokenShiftOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& prev_input = ctx->input(1);
        const Tensor& decay_weights = ctx->input(2);
        
        OP_REQUIRES(ctx, input.dims() == 3,
            errors::InvalidArgument("input must be 3D"));
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        
        SimplifiedTokenShiftForward(
            input.flat<float>().data(),
            prev_input.flat<float>().data(),
            decay_weights.flat<float>().data(),
            output->flat<float>().data(),
            input.dim_size(0), input.dim_size(1), input.dim_size(2));
    }
};

class FusedSimplifiedTokenShiftGradOp : public OpKernel {
 public:
    explicit FusedSimplifiedTokenShiftGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& input = ctx->input(1);
        const Tensor& prev_input = ctx->input(2);
        const Tensor& decay_weights = ctx->input(3);
        
        Tensor* grad_input = nullptr;
        Tensor* grad_prev = nullptr;
        Tensor* grad_decay = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, prev_input.shape(), &grad_prev));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, decay_weights.shape(), &grad_decay));
        
        SimplifiedTokenShiftBackward(
            grad_output.flat<float>().data(),
            input.flat<float>().data(),
            prev_input.flat<float>().data(),
            decay_weights.flat<float>().data(),
            grad_input->flat<float>().data(),
            grad_prev->flat<float>().data(),
            grad_decay->flat<float>().data(),
            input.dim_size(0), input.dim_size(1), input.dim_size(2));
    }
};

REGISTER_KERNEL_BUILDER(
    Name("FusedSimplifiedTokenShift").Device(DEVICE_CPU),
    FusedSimplifiedTokenShiftOp);

REGISTER_KERNEL_BUILDER(
    Name("FusedSimplifiedTokenShiftGrad").Device(DEVICE_CPU),
    FusedSimplifiedTokenShiftGradOp);

// =============================================================================
// ENHANCEMENT 3: HIERARCHICAL TOKEN SHIFT
// Layer-position aware decay for multi-scale temporal patterns
// =============================================================================

REGISTER_OP("FusedHierarchicalTokenShift")
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("gate_proj: float")       // [batch, seq_len, embedding_dim]
    .Input("decay_weights: float")   // [embedding_dim]
    .Attr("layer_position: int = 0")
    .Attr("decay_factor: float = 2.0")
    .Output("output: float")         // [batch, seq_len, embedding_dim]
    .Output("gate: float")           // [batch, seq_len, embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return OkStatus();
    })
    .Doc(R"doc(
Hierarchical token shift with layer-position aware decay.

Earlier layers use faster decay (local patterns), later layers use slower decay (global context).
decay_effective = base_decay ** (1.0 / (layer_position + decay_factor))

input: Input tensor of shape [batch, seq_len, embedding_dim].
prev_input: Previous token states of same shape.
gate_proj: Pre-computed gate projection.
decay_weights: Base decay weights [embedding_dim].
layer_position: Current layer index (0-indexed).
decay_factor: Scaling factor for hierarchy (default 2.0).
output: Shifted output tensor.
gate: Gate values for backward pass.
)doc");

REGISTER_OP("FusedHierarchicalTokenShiftGrad")
    .Input("grad_output: float")     // [batch, seq_len, embedding_dim]
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("prev_input: float")      // [batch, seq_len, embedding_dim]
    .Input("gate_proj: float")       // [batch, seq_len, embedding_dim]
    .Input("decay_weights: float")   // [embedding_dim]
    .Input("gate: float")            // [batch, seq_len, embedding_dim]
    .Attr("layer_position: int = 0")
    .Attr("decay_factor: float = 2.0")
    .Output("grad_input: float")           // [batch, seq_len, embedding_dim]
    .Output("grad_prev_input: float")      // [batch, seq_len, embedding_dim]
    .Output("grad_gate_proj: float")       // [batch, seq_len, embedding_dim]
    .Output("grad_decay_weights: float")   // [embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        c->set_output(2, c->input(3));
        c->set_output(3, c->input(4));
        return OkStatus();
    });

class FusedHierarchicalTokenShiftOp : public OpKernel {
 public:
    explicit FusedHierarchicalTokenShiftOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("layer_position", &layer_position_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_factor", &decay_factor_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& prev_input = ctx->input(1);
        const Tensor& gate_proj = ctx->input(2);
        const Tensor& decay_weights = ctx->input(3);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int64_t dim = embedding_dim;
        
        Tensor* output = nullptr;
        Tensor* gate_out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &gate_out));
        
        // Compute base sigmoid decay
        std::vector<float> base_decay(dim);
        for (int64_t d = 0; d < dim; ++d) {
            base_decay[d] = 1.0f / (1.0f + std::exp(-decay_weights.flat<float>()(d)));
        }
        
        // Apply hierarchical scaling
        std::vector<float> hierarchical_decay(dim);
        saguaro::ops::token_shift_compute_hierarchical_decay(
            base_decay.data(), hierarchical_decay.data(),
            dim, layer_position_, decay_factor_);
        
        // Run standard forward with adjusted decay
        const int64_t batch_seq = static_cast<int64_t>(batch_size) * seq_len;
        
        #pragma omp parallel for
        for (int64_t t = 0; t < batch_seq; ++t) {
            const int64_t offset = t * dim;
            float* gate = gate_out->flat<float>().data() + offset;
            
            // Copy and apply sigmoid to gate_proj
            std::copy(gate_proj.flat<float>().data() + offset,
                      gate_proj.flat<float>().data() + offset + dim, gate);
            saguaro::ops::token_shift_sigmoid_inplace(gate, dim);
            
            // Apply hierarchical decay
            for (int64_t d = 0; d < dim; ++d) {
                gate[d] *= hierarchical_decay[d];
            }
            
            // Token mixing
            saguaro::ops::token_shift_mix(
                gate,
                input.flat<float>().data() + offset,
                prev_input.flat<float>().data() + offset,
                output->flat<float>().data() + offset,
                dim);
        }
    }

 private:
    int layer_position_;
    float decay_factor_;
};

class FusedHierarchicalTokenShiftGradOp : public OpKernel {
 public:
    explicit FusedHierarchicalTokenShiftGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("layer_position", &layer_position_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("decay_factor", &decay_factor_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& input = ctx->input(1);
        const Tensor& prev_input = ctx->input(2);
        const Tensor& gate_proj = ctx->input(3);
        const Tensor& decay_weights = ctx->input(4);
        const Tensor& gate_forward = ctx->input(5);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int64_t dim = embedding_dim;
        const int64_t total_tokens = static_cast<int64_t>(batch_size) * seq_len;
        
        Tensor* grad_input = nullptr;
        Tensor* grad_prev = nullptr;
        Tensor* grad_gate_proj = nullptr;
        Tensor* grad_decay = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, input.shape(), &grad_prev));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, input.shape(), &grad_gate_proj));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, decay_weights.shape(), &grad_decay));
        
        // 1. Recompute hierarchical decay and its gradient multiplier
        std::vector<float> base_decay(dim);
        for (int64_t d = 0; d < dim; ++d) {
            base_decay[d] = 1.0f / (1.0f + std::exp(-decay_weights.flat<float>()(d)));
        }
        
        std::vector<float> h_decay(dim);
        saguaro::ops::token_shift_compute_hierarchical_decay(
            base_decay.data(), h_decay.data(), dim, layer_position_, decay_factor_);
        
        std::vector<float> h_grad_mult(dim);
        saguaro::ops::token_shift_compute_hierarchical_decay_grad(
            base_decay.data(), h_decay.data(), h_grad_mult.data(), dim, layer_position_, decay_factor_);
            
        // 2. Run backward pass
        std::memset(grad_decay->flat<float>().data(), 0, dim * sizeof(float));
        
        #pragma omp parallel
        {
            std::vector<float> local_grad_decay(dim, 0.0f);
            
            #pragma omp for
            for (int64_t t = 0; t < total_tokens; ++t) {
                const int64_t offset = t * dim;
                const float* g_out = grad_output.flat<float>().data() + offset;
                const float* x = input.flat<float>().data() + offset;
                const float* prev = prev_input.flat<float>().data() + offset;
                const float* g_proj = gate_proj.flat<float>().data() + offset;
                const float* g_fwd = gate_forward.flat<float>().data() + offset;
                
                float* g_in = grad_input->flat<float>().data() + offset;
                float* g_pv = grad_prev->flat<float>().data() + offset;
                float* g_gp = grad_gate_proj->flat<float>().data() + offset;
                
                for (int64_t d = 0; d < dim; ++d) {
                    float dL_dout = g_out[d];
                    float gate_val = g_fwd[d];
                    
                    // grad_input = dL/dout * gate
                    g_in[d] = dL_dout * gate_val;
                    // grad_prev = dL/dout * (1 - gate)
                    g_pv[d] = dL_dout * (1.0f - gate_val);
                    
                    // Gradient w.r.t. gate total
                    float dL_dg = dL_dout * (x[d] - prev[d]);
                    
                    // Gate = sigmoid(gate_proj) * h_decay
                    // sigmoid(gate_proj) = gate_val / h_decay (but we can use gate_proj)
                    float s_gp = 1.0f / (1.0f + std::exp(-g_proj[d]));
                    float ds_gp = s_gp * (1.0f - s_gp);
                    
                    // grad_gate_proj = dL/dg * h_decay * ds_gp
                    g_gp[d] = dL_dg * h_decay[d] * ds_gp;
                    
                    // grad_decay = dL/dg * s_gp * h_grad_mult
                    local_grad_decay[d] += dL_dg * s_gp * h_grad_mult[d];
                }
            }
            
            #pragma omp critical
            {
                for (int64_t d = 0; d < dim; ++d) {
                    grad_decay->flat<float>()(d) += local_grad_decay[d];
                }
            }
        }
    }

 private:
    int layer_position_;
    float decay_factor_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedHierarchicalTokenShiftGrad").Device(DEVICE_CPU),
    FusedHierarchicalTokenShiftGradOp);

REGISTER_KERNEL_BUILDER(
    Name("FusedHierarchicalTokenShift").Device(DEVICE_CPU),
    FusedHierarchicalTokenShiftOp);

// =============================================================================
// ENHANCEMENT 4: DELTA RULE TOKEN SHIFT
// Gated Delta Networks for precise memory control (ICLR 2025)
// =============================================================================

REGISTER_OP("FusedDeltaTokenShift")
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("state: float")           // [batch, embedding_dim] (recurrent state)
    .Input("erase_proj: float")      // [batch, seq_len, embedding_dim]
    .Input("write_proj: float")      // [batch, seq_len, embedding_dim]
    .Output("output: float")         // [batch, seq_len, embedding_dim]
    .Output("new_state: float")      // [batch, embedding_dim]
    .Output("erase_gate: float")     // [batch, seq_len, embedding_dim] (for backward)
    .Output("write_gate: float")     // [batch, seq_len, embedding_dim] (for backward)
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));  // output
        c->set_output(1, c->input(1));  // new_state
        c->set_output(2, c->input(0));  // erase_gate
        c->set_output(3, c->input(0));  // write_gate
        return OkStatus();
    })
    .Doc(R"doc(
Delta rule token shift for precise memory control.

Uses separate erase and write gates for better in-context learning:
  erase = sigmoid(erase_proj)
  write = sigmoid(write_proj)
  state = state * (1 - erase) + write * input
  output = state

input: Input tensor.
state: Recurrent memory state.
erase_proj: Pre-computed erase projection.
write_proj: Pre-computed write projection.
output: Updated output (same as new_state broadcast to sequence).
new_state: Updated memory state.
erase_gate: Erase gate values (for backward).
write_gate: Write gate values (for backward).
)doc");

REGISTER_OP("FusedDeltaTokenShiftGrad")
    .Input("grad_output: float")     // [batch, seq_len, embedding_dim]
    .Input("grad_new_state: float")  // [batch, embedding_dim]
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("state: float")           // [batch, embedding_dim]
    .Input("erase_proj: float")      // [batch, seq_len, embedding_dim]
    .Input("write_proj: float")      // [batch, seq_len, embedding_dim]
    .Input("erase_gate: float")      // [batch, seq_len, embedding_dim]
    .Input("write_gate: float")      // [batch, seq_len, embedding_dim]
    .Output("grad_input: float")           // [batch, seq_len, embedding_dim]
    .Output("grad_state: float")           // [batch, embedding_dim]
    .Output("grad_erase_proj: float")      // [batch, seq_len, embedding_dim]
    .Output("grad_write_proj: float")      // [batch, seq_len, embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(2));
        c->set_output(1, c->input(3));
        c->set_output(2, c->input(4));
        c->set_output(3, c->input(5));
        return OkStatus();
    });

class FusedDeltaTokenShiftOp : public OpKernel {
 public:
    explicit FusedDeltaTokenShiftOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& state = ctx->input(1);
        const Tensor& erase_proj = ctx->input(2);
        const Tensor& write_proj = ctx->input(3);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int64_t dim = embedding_dim;
        
        Tensor* output = nullptr;
        Tensor* new_state = nullptr;
        Tensor* erase_gate = nullptr;
        Tensor* write_gate = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, state.shape(), &new_state));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, input.shape(), &erase_gate));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, input.shape(), &write_gate));
        
        // Initialize new_state from input state
        std::copy(state.flat<float>().data(),
                  state.flat<float>().data() + batch_size * dim,
                  new_state->flat<float>().data());
        
        // Process sequence step by step (recurrent)
        for (int b = 0; b < batch_size; ++b) {
            float* current_state = new_state->flat<float>().data() + b * dim;
            
            for (int s = 0; s < seq_len; ++s) {
                const int64_t offset = (b * seq_len + s) * dim;
                
                float* e_gate = erase_gate->flat<float>().data() + offset;
                float* w_gate = write_gate->flat<float>().data() + offset;
                float* out = output->flat<float>().data() + offset;
                
                // Compute gates
                std::copy(erase_proj.flat<float>().data() + offset,
                          erase_proj.flat<float>().data() + offset + dim, e_gate);
                std::copy(write_proj.flat<float>().data() + offset,
                          write_proj.flat<float>().data() + offset + dim, w_gate);
                
                saguaro::ops::token_shift_sigmoid_inplace(e_gate, dim);
                saguaro::ops::token_shift_sigmoid_inplace(w_gate, dim);
                
                // Delta update
                std::vector<float> temp_state(dim);
                saguaro::ops::token_shift_delta_update(
                    current_state, e_gate, w_gate,
                    input.flat<float>().data() + offset,
                    temp_state.data(), dim);
                
                // Copy to current_state and output
                std::copy(temp_state.begin(), temp_state.end(), current_state);
                std::copy(temp_state.begin(), temp_state.end(), out);
            }
        }
    }
};

class FusedDeltaTokenShiftGradOp : public OpKernel {
 public:
    explicit FusedDeltaTokenShiftGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& grad_new_state_in = ctx->input(1);
        const Tensor& input = ctx->input(2);
        const Tensor& state = ctx->input(3);
        const Tensor& erase_proj = ctx->input(4);
        const Tensor& write_proj = ctx->input(5);
        const Tensor& erase_gate = ctx->input(6);
        const Tensor& write_gate = ctx->input(7);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int64_t dim = embedding_dim;
        
        Tensor* grad_input = nullptr;
        Tensor* grad_state = nullptr;
        Tensor* grad_erase_proj = nullptr;
        Tensor* grad_write_proj = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, state.shape(), &grad_state));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, input.shape(), &grad_erase_proj));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, input.shape(), &grad_write_proj));
        
        // Process backward in reverse sequence order
        for (int b = 0; b < batch_size; ++b) {
            std::vector<float> curr_grad_state(dim);
            std::copy(grad_new_state_in.flat<float>().data() + b * dim,
                      grad_new_state_in.flat<float>().data() + (b + 1) * dim,
                      curr_grad_state.data());
                      
            for (int s = seq_len - 1; s >= 0; --s) {
                const int64_t offset = (b * seq_len + s) * dim;
                
                // Add grad_output[s] to curr_grad_state since output = state
                for (int64_t d = 0; d < dim; ++d) {
                    curr_grad_state[d] += grad_output.flat<float>()(offset + d);
                }
                
                // We need the state BEFORE the current step to compute grad_erase
                // This is a bit tricky; we might need to re-run forward or store all intermediate states.
                // For now, let's approximate or stick to simple gate gradients if state is not available.
                // Wait, I can re-run forward locally for this batch or just store states.
                // Given the Recurrent nature, I'll store states for this batch.
                std::vector<std::vector<float>> step_states(seq_len + 1, std::vector<float>(dim));
                std::copy(state.flat<float>().data() + b * dim,
                          state.flat<float>().data() + (b + 1) * dim,
                          step_states[0].data());
                
                for (int fs = 0; fs < seq_len; ++fs) {
                    const int64_t f_offset = (b * seq_len + fs) * dim;
                    saguaro::ops::token_shift_delta_update(
                        step_states[fs].data(),
                        erase_gate.flat<float>().data() + f_offset,
                        write_gate.flat<float>().data() + f_offset,
                        input.flat<float>().data() + f_offset,
                        step_states[fs + 1].data(), dim);
                }
                
                // Now run backward properly using stored states
                std::vector<float> next_grad_state(dim, 0.0f);
                for (int bs = seq_len - 1; bs >= 0; --bs) {
                    const int64_t b_offset = (b * seq_len + bs) * dim;
                    
                    // Gradient of output is already in curr_grad_state implicitly
                    // because output = state_after.
                    
                    std::vector<float> g_s(dim), g_e(dim), g_w(dim), g_v(dim);
                    saguaro::ops::token_shift_delta_update_grad(
                        curr_grad_state.data(),
                        step_states[bs].data(),
                        erase_gate.flat<float>().data() + b_offset,
                        write_gate.flat<float>().data() + b_offset,
                        input.flat<float>().data() + b_offset,
                        g_s.data(), g_e.data(), g_w.data(), g_v.data(), dim);
                    
                    // Apply sigmoid derivatives for projections
                    for (int64_t d = 0; d < dim; ++d) {
                        float e = erase_gate.flat<float>()(b_offset + d);
                        float w = write_gate.flat<float>()(b_offset + d);
                        grad_erase_proj->flat<float>()(b_offset + d) = g_e[d] * e * (1.0f - e);
                        grad_write_proj->flat<float>()(b_offset + d) = g_w[d] * w * (1.0f - w);
                        grad_input->flat<float>()(b_offset + d) = g_v[d];
                    }
                    
                    // Pack grad_output[bs-1] if bs > 0
                    curr_grad_state = g_s;
                    if (bs > 0) {
                        for (int64_t d = 0; d < dim; ++d) {
                            curr_grad_state[d] += grad_output.flat<float>()((b * seq_len + bs - 1) * dim + d);
                        }
                    }
                }
                
                // Final state gradient
                std::copy(curr_grad_state.begin(), curr_grad_state.end(),
                          grad_state->flat<float>().data() + b * dim);
                break; // We processed the whole sequence for this batch in the inner loop
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
    Name("FusedDeltaTokenShiftGrad").Device(DEVICE_CPU),
    FusedDeltaTokenShiftGradOp);

REGISTER_KERNEL_BUILDER(
    Name("FusedDeltaTokenShift").Device(DEVICE_CPU),
    FusedDeltaTokenShiftOp);

// =============================================================================
// ENHANCEMENT 5: MULTI-POSITION TOKEN SHIFT
// Access tokens at multiple shift distances directly
// =============================================================================

REGISTER_OP("FusedMultiPositionTokenShift")
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("blend_weights: float")   // [num_distances]
    .Attr("distances: list(int)")    // e.g., [1, 2, 4]
    .Output("output: float")         // [batch, seq_len, embedding_dim]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return OkStatus();
    })
    .Doc(R"doc(
Multi-position token shift with configurable distances.

Creates shifted versions at multiple distances and blends them.
output = sum(blend_weights[i] * shift(input, distances[i]))

input: Input tensor.
blend_weights: Blending weights for each distance (should sum to 1).
distances: List of shift distances (e.g., [1, 2, 4]).
output: Blended output tensor.
)doc");

REGISTER_OP("FusedMultiPositionTokenShiftGrad")
    .Input("grad_output: float")     // [batch, seq_len, embedding_dim]
    .Input("input: float")           // [batch, seq_len, embedding_dim]
    .Input("blend_weights: float")   // [num_distances]
    .Attr("distances: list(int)")
    .Output("grad_input: float")           // [batch, seq_len, embedding_dim]
    .Output("grad_blend_weights: float")   // [num_distances]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        return OkStatus();
    });

class FusedMultiPositionTokenShiftOp : public OpKernel {
 public:
    explicit FusedMultiPositionTokenShiftOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("distances", &distances_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& blend_weights = ctx->input(1);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int num_distances = distances_.size();
        
        OP_REQUIRES(ctx, blend_weights.dim_size(0) == num_distances,
            errors::InvalidArgument("blend_weights size must match distances"));
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
        
        // Allocate buffer for all shifted versions
        const int64_t batch_seq = static_cast<int64_t>(batch_size) * seq_len;
        const int64_t shift_size = batch_seq * embedding_dim;
        std::vector<float> shifts(num_distances * shift_size);
        
        // Create shifted versions
        saguaro::ops::token_shift_multi_distance(
            input.flat<float>().data(),
            shifts.data(),
            distances_.data(),
            batch_size, seq_len, embedding_dim, num_distances);
        
        // Blend shifted versions
        saguaro::ops::token_shift_blend_distances(
            shifts.data(),
            blend_weights.flat<float>().data(),
            output->flat<float>().data(),
            batch_seq, embedding_dim, num_distances);
    }

 private:
    std::vector<int> distances_;
};

class FusedMultiPositionTokenShiftGradOp : public OpKernel {
 public:
    explicit FusedMultiPositionTokenShiftGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("distances", &distances_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        const Tensor& input = ctx->input(1);
        const Tensor& blend_weights = ctx->input(2);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int embedding_dim = input.dim_size(2);
        const int num_distances = distances_.size();
        
        Tensor* grad_input = nullptr;
        Tensor* grad_weights = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, blend_weights.shape(), &grad_weights));
        
        saguaro::ops::token_shift_multi_distance_grad(
            grad_output.flat<float>().data(),
            input.flat<float>().data(),
            grad_input->flat<float>().data(),
            grad_weights->flat<float>().data(),
            distances_.data(),
            blend_weights.flat<float>().data(),
            batch_size, seq_len, embedding_dim, num_distances);
    }

 private:
    std::vector<int> distances_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedMultiPositionTokenShiftGrad").Device(DEVICE_CPU),
    FusedMultiPositionTokenShiftGradOp);

REGISTER_KERNEL_BUILDER(
    Name("FusedMultiPositionTokenShift").Device(DEVICE_CPU),
    FusedMultiPositionTokenShiftOp);

}  // namespace tensorflow
