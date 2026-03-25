// saguaro.native/ops/adiabatic_optimizer_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "adiabatic_optimizer_op.h"

using namespace tensorflow;

REGISTER_OP("AdiabaticOptimizerStep")
    .Input("params: float")
    .Input("gradients: float")
    .Input("velocity: float")
    .Output("updated_params: float")
    .Output("updated_velocity: float")
    .Attr("schedule_s: float = 0.5")
    .Attr("initial_temp: float = 10.0")
    .Attr("final_temp: float = 0.01")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(2));
        return Status();
    })
    .Doc("Phase 59: Quantum adiabatic optimizer step with tunneling.");

class AdiabaticOptimizerStepOp : public OpKernel {
 public:
  explicit AdiabaticOptimizerStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("schedule_s", &schedule_s_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("initial_temp", &config_.initial_temp));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("final_temp", &config_.final_temp));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto params = ctx->input(0);
    auto gradients = ctx->input(1);
    auto velocity = ctx->input(2);
    int num_params = params.NumElements();
    
    Tensor* out_params = nullptr;
    Tensor* out_velocity = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, params.shape(), &out_params));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, velocity.shape(), &out_velocity));
    
    std::copy_n(params.flat<float>().data(), num_params, out_params->flat<float>().data());
    std::copy_n(velocity.flat<float>().data(), num_params, out_velocity->flat<float>().data());
    
    saguaro::qao::AdiabaticOptimizerStep(out_params->flat<float>().data(),
        gradients.flat<float>().data(), out_velocity->flat<float>().data(),
        schedule_s_, config_, num_params, seed_);
  }
 private:
  float schedule_s_;
  saguaro::qao::QAOConfig config_;
  int seed_;
};
REGISTER_KERNEL_BUILDER(Name("AdiabaticOptimizerStep").Device(DEVICE_CPU), AdiabaticOptimizerStepOp);
