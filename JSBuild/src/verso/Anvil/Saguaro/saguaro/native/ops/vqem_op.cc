// saguaro.native/ops/vqem_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <random>
#include "vqem_op.h"

using namespace tensorflow;

REGISTER_OP("VQEMForward")
    .Input("input_state: float")
    .Input("mitigation_params: float")
    .Output("output_state: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Phase 62: VQEM error mitigation forward pass.");

class VQEMForwardOp : public OpKernel {
 public:
  explicit VQEMForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    auto params = ctx->input(1);
    int batch = input.dim_size(0), dim = input.dim_size(1);
    int num_params = params.NumElements();
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    
    saguaro::vqem::VQEMForward(input.flat<float>().data(), params.flat<float>().data(),
        output->flat<float>().data(), batch, dim, num_params);
  }
};
REGISTER_KERNEL_BUILDER(Name("VQEMForward").Device(DEVICE_CPU), VQEMForwardOp);

REGISTER_OP("VQEMTrainStep")
    .Input("mitigation_params: float")
    .Input("noisy_output: float")
    .Input("ideal_output: float")
    .Output("updated_params: float")
    .Attr("learning_rate: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Phase 62: VQEM parameter training step.");

class VQEMTrainStepOp : public OpKernel {
 public:
  explicit VQEMTrainStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("learning_rate", &lr_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto params = ctx->input(0);
    auto noisy = ctx->input(1);
    auto ideal = ctx->input(2);
    int batch = noisy.dim_size(0), dim = noisy.dim_size(1);
    int num_params = params.NumElements();
    
    Tensor* updated = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, params.shape(), &updated));
    std::copy_n(params.flat<float>().data(), num_params, updated->flat<float>().data());
    
    saguaro::vqem::VQEMTrainStep(updated->flat<float>().data(),
        noisy.flat<float>().data(), ideal.flat<float>().data(),
        lr_, batch, dim, num_params);
  }
 private:
  float lr_;
};
REGISTER_KERNEL_BUILDER(Name("VQEMTrainStep").Device(DEVICE_CPU), VQEMTrainStepOp);
