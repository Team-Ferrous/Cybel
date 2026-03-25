// saguaro.native/ops/hd_streaming_adapter_op.cc
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
 * HD Streaming Adapter Op - Bridges HD bundles to model architecture.
 * 
 * Phase 200+: Quantum-Enhanced HD Streaming Mode
 * 
 * This op projects holographic bundles from HolographicCorpus to the model's
 * hidden dimension, adding a sequence dimension for ReasoningModule compatibility.
 * 
 * Shape: (batch, hd_dim) -> (batch, 1, hidden_dim)
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "hd_streaming_adapter_op.h"

using namespace tensorflow;

// =============================================================================
// HDStreamingProject Op - Forward Pass
// =============================================================================

REGISTER_OP("HDStreamingProject")
    .Input("hd_bundles: float")           // [batch, hd_dim]
    .Input("projection_weights: float")   // [hd_dim, hidden_dim]
    .Input("projection_bias: float")      // [hidden_dim]
    .Output("output: float")              // [batch, 1, hidden_dim]
    .Attr("hd_dim: int = 1024")
    .Attr("hidden_dim: int = 256")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int hidden_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("hidden_dim", &hidden_dim));
        auto batch = c->Dim(c->input(0), 0);
        // Output shape: (batch, 1, hidden_dim) - sequence dim added for ReasoningModule
        c->set_output(0, c->MakeShape({batch, 1, hidden_dim}));
        return Status();
    })
    .Doc(R"doc(
HD Streaming Project Op - Projects HD bundles to model hidden dimension.

Phase 200+: Bridges HolographicCorpus HD bundles to ReasoningModule.

hd_bundles: [batch, hd_dim] float32 - Holographic bundles from corpus
projection_weights: [hd_dim, hidden_dim] float32 - Learned projection matrix
projection_bias: [hidden_dim] float32 - Learned bias vector
output: [batch, 1, hidden_dim] float32 - Projected features with sequence dim

The sequence dimension (1) is added for compatibility with ReasoningModule which
expects input shape (batch, seq_len, hidden_dim). For HD streaming, seq_len=1
since the entire context is compressed into a single holographic bundle.
)doc");


class HDStreamingProjectOp : public OpKernel {
 public:
  explicit HDStreamingProjectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_dim", &config_.hidden_dim));
    config_.add_sequence_dim = true;
  }

  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& hd_bundles = ctx->input(0);
    const Tensor& projection_weights = ctx->input(1);
    const Tensor& projection_bias = ctx->input(2);
    
    // Validate shapes
    const int batch_size = hd_bundles.dim_size(0);
    const int input_hd_dim = hd_bundles.dim_size(1);
    
    OP_REQUIRES(ctx, input_hd_dim == config_.hd_dim,
        errors::InvalidArgument(
            "hd_bundles dim 1 (", input_hd_dim, ") must match hd_dim attr (", 
            config_.hd_dim, ")"));
    
    OP_REQUIRES(ctx, projection_weights.dim_size(0) == config_.hd_dim,
        errors::InvalidArgument(
            "projection_weights dim 0 must match hd_dim"));
    
    OP_REQUIRES(ctx, projection_weights.dim_size(1) == config_.hidden_dim,
        errors::InvalidArgument(
            "projection_weights dim 1 must match hidden_dim"));
            
    OP_REQUIRES(ctx, projection_bias.dim_size(0) == config_.hidden_dim,
        errors::InvalidArgument(
            "projection_bias dim 0 must match hidden_dim"));
    
    // Allocate output: (batch, 1, hidden_dim)
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({batch_size, 1, config_.hidden_dim}), &output));
    
    // Call kernel
    saguaro::hd_streaming::HDStreamProject(
        hd_bundles.flat<float>().data(),
        projection_weights.flat<float>().data(),
        projection_bias.flat<float>().data(),
        output->flat<float>().data(),
        config_,
        batch_size
    );
  }

 private:
  saguaro::hd_streaming::HDStreamingConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HDStreamingProject").Device(DEVICE_CPU), 
                        HDStreamingProjectOp);


// =============================================================================
// HDStreamingProjectGrad Op - Backward Pass
// =============================================================================

REGISTER_OP("HDStreamingProjectGrad")
    .Input("grad_output: float")          // [batch, 1, hidden_dim] or [batch, hidden_dim]
    .Input("hd_bundles: float")           // [batch, hd_dim] - forward pass input
    .Input("projection_weights: float")   // [hd_dim, hidden_dim] - forward pass weights
    .Output("grad_bundles: float")        // [batch, hd_dim]
    .Output("grad_weights: float")        // [hd_dim, hidden_dim]
    .Output("grad_bias: float")           // [hidden_dim]
    .Attr("hd_dim: int = 1024")
    .Attr("hidden_dim: int = 256")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int hd_dim, hidden_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("hd_dim", &hd_dim));
        TF_RETURN_IF_ERROR(c->GetAttr("hidden_dim", &hidden_dim));
        auto batch = c->Dim(c->input(1), 0);  // Get batch from hd_bundles
        c->set_output(0, c->MakeShape({batch, hd_dim}));        // grad_bundles
        c->set_output(1, c->MakeShape({hd_dim, hidden_dim}));   // grad_weights  
        c->set_output(2, c->MakeShape({hidden_dim}));           // grad_bias
        return Status();
    })
    .Doc(R"doc(
HD Streaming Project Gradient Op - Computes gradients for backward pass.

grad_output: [batch, hidden_dim] or [batch, 1, hidden_dim] - Upstream gradient
hd_bundles: [batch, hd_dim] - Forward pass input
projection_weights: [hd_dim, hidden_dim] - Forward pass weights
grad_bundles: [batch, hd_dim] - Gradient w.r.t. input bundles
grad_weights: [hd_dim, hidden_dim] - Gradient w.r.t. projection weights
grad_bias: [hidden_dim] - Gradient w.r.t. bias
)doc");


class HDStreamingProjectGradOp : public OpKernel {
 public:
  explicit HDStreamingProjectGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_dim", &config_.hidden_dim));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& grad_output = ctx->input(0);
    const Tensor& hd_bundles = ctx->input(1);
    const Tensor& projection_weights = ctx->input(2);
    
    const int batch_size = hd_bundles.dim_size(0);
    
    // Allocate outputs
    Tensor* grad_bundles = nullptr;
    Tensor* grad_weights = nullptr;
    Tensor* grad_bias = nullptr;
    
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        0, TensorShape({batch_size, config_.hd_dim}), &grad_bundles));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        1, TensorShape({config_.hd_dim, config_.hidden_dim}), &grad_weights));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
        2, TensorShape({config_.hidden_dim}), &grad_bias));
    
    // Handle grad_output shape - may be (batch, hidden_dim) or (batch, 1, hidden_dim)
    // Flatten sequence dimension if present
    const float* grad_out_ptr = grad_output.flat<float>().data();
    
    // Call gradient kernel
    saguaro::hd_streaming::HDStreamProjectGrad(
        grad_out_ptr,
        hd_bundles.flat<float>().data(),
        projection_weights.flat<float>().data(),
        grad_bundles->flat<float>().data(),
        grad_weights->flat<float>().data(),
        grad_bias->flat<float>().data(),
        config_,
        batch_size
    );
  }

 private:
  saguaro::hd_streaming::HDStreamingConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HDStreamingProjectGrad").Device(DEVICE_CPU),
                        HDStreamingProjectGradOp);
