// highnoon/_native/ops/circular_conv_op.cc
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
 * @file circular_conv_op.cc
 * @brief TensorFlow Op for in-place circular convolution (Phase 900.2).
 *
 * Memory-optimized holographic binding for DualPathEmbedding.
 * Replaces tf.signal.fft-based circular convolution with in-place FFT.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "circular_conv_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registration: CircularConvForward
// =============================================================================

REGISTER_OP("CircularConvForward")
    .Input("tokens_hd: float32")      // [batch, seq, hd_dim]
    .Input("position_vectors: float32") // [seq, hd_dim] or [1, seq, hd_dim]
    .Output("bound_hd: float32")      // [batch, seq, hd_dim]
    .Attr("hd_dim: int")
    .SetShapeFn([](InferenceContext* c) {
        // Output shape matches tokens_hd input
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Circular convolution for holographic position binding.

Computes tokens_hd ⊛ position_vectors using in-place FFT.
Memory efficient: 4× reduction vs TensorFlow's tf.signal.fft.

tokens_hd: Input token embeddings [batch, seq, hd_dim].
position_vectors: Position vectors [seq, hd_dim] broadcasted to all batches.
bound_hd: Output bound embeddings [batch, seq, hd_dim].
hd_dim: Hyperdimensional dimension (must be power of 2).
)doc");

// =============================================================================
// Op Registration: CircularConvBackward
// =============================================================================

REGISTER_OP("CircularConvBackward")
    .Input("grad_output: float32")     // [batch, seq, hd_dim]
    .Input("tokens_hd: float32")       // [batch, seq, hd_dim]
    .Input("position_vectors: float32") // [seq, hd_dim]
    .Output("grad_tokens: float32")    // [batch, seq, hd_dim]
    .Output("grad_positions: float32") // [seq, hd_dim]
    .Attr("hd_dim: int")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_tokens matches tokens_hd
        c->set_output(1, c->input(2));  // grad_positions matches position_vectors
        return Status();
    })
    .Doc(R"doc(
Backward pass for circular convolution.

Computes gradients for tokens_hd and position_vectors.

grad_output: Gradient of loss w.r.t. bound_hd.
tokens_hd: Forward input tokens.
position_vectors: Forward input positions.
grad_tokens: Gradient w.r.t. tokens_hd.
grad_positions: Gradient w.r.t. position_vectors (accumulated across batch).
hd_dim: Hyperdimensional dimension.
)doc");

// =============================================================================
// Kernel: CircularConvForwardOp
// =============================================================================

class CircularConvForwardOp : public OpKernel {
public:
    explicit CircularConvForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &hd_dim_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& tokens_hd = ctx->input(0);
        const Tensor& position_vectors = ctx->input(1);

        // Validate shapes
        OP_REQUIRES(ctx, tokens_hd.dims() == 3,
            errors::InvalidArgument("tokens_hd must be 3D [batch, seq, hd_dim]"));
        
        const int batch_size = tokens_hd.dim_size(0);
        const int seq_len = tokens_hd.dim_size(1);
        const int hd_dim = tokens_hd.dim_size(2);

        OP_REQUIRES(ctx, hd_dim == hd_dim_,
            errors::InvalidArgument("hd_dim mismatch: got ", hd_dim, " expected ", hd_dim_));

        // Validate position_vectors shape: [seq, hd_dim] or [1, seq, hd_dim]
        if (position_vectors.dims() == 2) {
            OP_REQUIRES(ctx, position_vectors.dim_size(0) == seq_len &&
                             position_vectors.dim_size(1) == hd_dim,
                errors::InvalidArgument("position_vectors shape mismatch"));
        } else if (position_vectors.dims() == 3) {
            OP_REQUIRES(ctx, position_vectors.dim_size(0) == 1 &&
                             position_vectors.dim_size(1) == seq_len &&
                             position_vectors.dim_size(2) == hd_dim,
                errors::InvalidArgument("position_vectors shape mismatch"));
        } else {
            OP_REQUIRES(ctx, false,
                errors::InvalidArgument("position_vectors must be 2D or 3D"));
        }

        // Allocate output
        Tensor* bound_hd = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tokens_hd.shape(), &bound_hd));

        // Get raw pointers
        const float* tokens_ptr = tokens_hd.flat<float>().data();
        const float* pos_ptr = position_vectors.dims() == 3 ?
            position_vectors.flat<float>().data() :
            position_vectors.flat<float>().data();
        float* output_ptr = bound_hd->flat<float>().data();

        // Call optimized kernel
        highnoon::ops::circular_convolution_batched(
            tokens_ptr, pos_ptr, output_ptr,
            batch_size, seq_len, hd_dim
        );
    }

private:
    int hd_dim_;
};

// =============================================================================
// Kernel: CircularConvBackwardOp
// =============================================================================

class CircularConvBackwardOp : public OpKernel {
public:
    explicit CircularConvBackwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &hd_dim_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get inputs
        const Tensor& grad_output = ctx->input(0);
        const Tensor& tokens_hd = ctx->input(1);
        const Tensor& position_vectors = ctx->input(2);

        const int batch_size = tokens_hd.dim_size(0);
        const int seq_len = tokens_hd.dim_size(1);
        const int hd_dim = tokens_hd.dim_size(2);

        // Allocate outputs
        Tensor* grad_tokens = nullptr;
        Tensor* grad_positions = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tokens_hd.shape(), &grad_tokens));
        
        // grad_positions is [seq, hd_dim] (accumulated across batch)
        TensorShape pos_grad_shape;
        pos_grad_shape.AddDim(seq_len);
        pos_grad_shape.AddDim(hd_dim);
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, pos_grad_shape, &grad_positions));

        // Get raw pointers
        const float* grad_out_ptr = grad_output.flat<float>().data();
        const float* tokens_ptr = tokens_hd.flat<float>().data();
        const float* pos_ptr = position_vectors.dims() == 3 ?
            position_vectors.flat<float>().data() :
            position_vectors.flat<float>().data();
        float* grad_tokens_ptr = grad_tokens->flat<float>().data();
        float* grad_pos_ptr = grad_positions->flat<float>().data();

        // Call optimized backward kernel
        highnoon::ops::circular_convolution_batched_backward(
            grad_out_ptr, tokens_ptr, pos_ptr,
            grad_tokens_ptr, grad_pos_ptr,
            batch_size, seq_len, hd_dim
        );
    }

private:
    int hd_dim_;
};

// =============================================================================
// Kernel Registration
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("CircularConvForward").Device(DEVICE_CPU),
    CircularConvForwardOp);

REGISTER_KERNEL_BUILDER(
    Name("CircularConvBackward").Device(DEVICE_CPU),
    CircularConvBackwardOp);

}  // namespace tensorflow
