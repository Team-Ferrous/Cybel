// saguaro.native/ops/td_moe_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/edition_limits.h"
#include "td_moe_op.h"

using namespace tensorflow;

REGISTER_OP("TDMoEForward")
    .Input("input: float")
    .Input("router_weights: float")
    .Input("expert_cores: float")
    .Input("expert_modes_a: float")
    .Input("expert_modes_b: float")
    .Output("output: float")
    .Attr("num_experts: int = 8")
    .Attr("tucker_rank: int = 16")
    .Attr("top_k: int = 2")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto output_dim = c->Dim(c->input(3), 1);  // from modes_a shape
        c->set_output(0, c->MakeShape({batch, output_dim}));
        return Status();
    })
    .Doc("Phase 57: Tucker Decomposition MoE with parameter-efficient experts.");

class TDMoEForwardOp : public OpKernel {
 public:
  explicit TDMoEForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_experts", &config_.num_experts));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tucker_rank", &config_.tucker_rank));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_k", &config_.top_k));
  }
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    auto router = ctx->input(1);
    auto cores = ctx->input(2);
    auto modes_a = ctx->input(3);
    auto modes_b = ctx->input(4);
    
    // HighNoon Lite Edition: Enforce MoE expert limit (max 12)
    SAGUARO_CHECK_MOE_EXPERTS(ctx, config_.num_experts);
    
    int batch = input.dim_size(0);
    config_.input_dim = input.dim_size(1);
    config_.output_dim = modes_a.dim_size(1) / config_.tucker_rank;
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, config_.output_dim}, &output));
    
    saguaro::tdmoe::TDMoEForward(input.flat<float>().data(), router.flat<float>().data(),
        cores.flat<float>().data(), modes_a.flat<float>().data(),
        modes_b.flat<float>().data(), output->flat<float>().data(), config_, batch);
  }
 private:
  saguaro::tdmoe::TDMoEConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("TDMoEForward").Device(DEVICE_CPU), TDMoEForwardOp);
