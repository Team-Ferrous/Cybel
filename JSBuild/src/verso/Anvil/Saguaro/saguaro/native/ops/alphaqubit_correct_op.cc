// saguaro.native/ops/alphaqubit_correct_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

/**
 * @file alphaqubit_correct_op.cc
 * @brief TensorFlow kernel registration for AlphaQubitCorrect op
 *
 * Phase 61 + S11: Unified quantum layer output error correction.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "alphaqubit_correct_op.h"

using namespace tensorflow;

REGISTER_OP("AlphaQubitCorrect")
    .Input("quantum_output: float")
    .Input("qkv_weights: float")
    .Input("proj_weights: float")
    .Input("corr_w1: float")
    .Input("corr_w2: float")
    .Input("gate_w: float")
    .Input("gate_b: float")
    .Output("corrected_output: float")
    .Attr("feature_dim: int = 256")
    .Attr("hidden_dim: int = 64")
    .Attr("num_attn_layers: int = 2")
    .Attr("num_heads: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape matches input shape
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 61 + S11: AlphaQubit Unified Error Correction

Takes quantum layer output and returns error-corrected output using
syndrome detection (self-attention) and learned correction with
gated residual connection.

quantum_output: Input from quantum layer [batch, feature_dim]
qkv_weights: Attention Q/K/V weights [num_layers, 3, dim, hidden]
proj_weights: Attention projection weights [num_layers, hidden, dim]
corr_w1: Correction dense layer 1 [dim, hidden]
corr_w2: Correction dense layer 2 [hidden, dim]
gate_w: Gate weights [dim, dim]
gate_b: Gate bias [dim]
corrected_output: Error-corrected output [batch, feature_dim]
)doc");

class AlphaQubitCorrectOp : public OpKernel {
 public:
  explicit AlphaQubitCorrectOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_dim", &config_.feature_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_attn_layers", &config_.num_attn_layers));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &config_.num_heads));
  }
  
  void Compute(OpKernelContext* ctx) override {
    // Get inputs
    const Tensor& quantum_output = ctx->input(0);
    const Tensor& qkv_weights = ctx->input(1);
    const Tensor& proj_weights = ctx->input(2);
    const Tensor& corr_w1 = ctx->input(3);
    const Tensor& corr_w2 = ctx->input(4);
    const Tensor& gate_w = ctx->input(5);
    const Tensor& gate_b = ctx->input(6);
    
    int batch = quantum_output.dim_size(0);
    
    // Allocate output
    Tensor* corrected_output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, quantum_output.shape(), 
                                              &corrected_output));
    
    // Call implementation
    saguaro::alphaqubit::AlphaQubitCorrect(
        quantum_output.flat<float>().data(),
        qkv_weights.flat<float>().data(),
        proj_weights.flat<float>().data(),
        corr_w1.flat<float>().data(),
        corr_w2.flat<float>().data(),
        gate_w.flat<float>().data(),
        gate_b.flat<float>().data(),
        corrected_output->flat<float>().data(),
        config_, batch);
  }
  
 private:
  saguaro::alphaqubit::AlphaQubitCorrectConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("AlphaQubitCorrect").Device(DEVICE_CPU), 
                        AlphaQubitCorrectOp);
