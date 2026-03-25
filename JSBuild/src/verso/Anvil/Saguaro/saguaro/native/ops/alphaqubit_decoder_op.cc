// saguaro.native/ops/alphaqubit_decoder_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "alphaqubit_decoder_op.h"

using namespace tensorflow;

REGISTER_OP("AlphaQubitDecode")
    .Input("syndrome: float")
    .Input("embed_weights: float")
    .Input("attention_weights: float")
    .Input("output_weights: float")
    .Output("error_probs: float")
    .Attr("syndrome_dim: int = 64")
    .Attr("hidden_dim: int = 128")
    .Attr("num_layers: int = 2")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch, 4}));  // 4 error classes
        return Status();
    })
    .Doc("Phase 61: AlphaQubit-2 neural syndrome decoder.");

class AlphaQubitDecodeOp : public OpKernel {
 public:
  explicit AlphaQubitDecodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("syndrome_dim", &config_.syndrome_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_layers", &config_.num_layers));
  }
  void Compute(OpKernelContext* ctx) override {
    auto syndrome = ctx->input(0);
    auto embed_w = ctx->input(1);
    auto attn_w = ctx->input(2);
    auto out_w = ctx->input(3);
    int batch = syndrome.dim_size(0);
    
    Tensor* error_probs = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, 4}, &error_probs));
    
    saguaro::alphaqubit::AlphaQubitDecode(syndrome.flat<float>().data(),
        embed_w.flat<float>().data(), attn_w.flat<float>().data(),
        out_w.flat<float>().data(), error_probs->flat<float>().data(),
        config_, batch);
  }
 private:
  saguaro::alphaqubit::AlphaQubitConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("AlphaQubitDecode").Device(DEVICE_CPU), AlphaQubitDecodeOp);
