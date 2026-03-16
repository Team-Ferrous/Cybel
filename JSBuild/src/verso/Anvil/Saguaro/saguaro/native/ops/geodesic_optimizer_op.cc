// saguaro.native/ops/geodesic_optimizer_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "geodesic_optimizer_op.h"

using namespace tensorflow;

REGISTER_OP("GeodesicOptimizerStep")
    .Input("params: float")
    .Input("gradients: float")
    .Input("velocity: float")
    .Output("updated_params: float")
    .Output("updated_velocity: float")
    .Attr("learning_rate: float = 0.001")
    .Attr("momentum: float = 0.9")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(2));
        return Status();
    })
    .Doc("Phase 60: Geodesic optimizer with natural gradients on parameter manifold.");

class GeodesicOptimizerStepOp : public OpKernel {
 public:
  explicit GeodesicOptimizerStepOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("learning_rate", &lr_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("momentum", &momentum_));
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
    
    std::vector<float> qgt(num_params);
    saguaro::gqgo::GeodesicOptimizerStep(out_params->flat<float>().data(),
        gradients.flat<float>().data(), out_velocity->flat<float>().data(),
        qgt.data(), lr_, momentum_, num_params);
  }
 private:
  float lr_, momentum_;
};
REGISTER_KERNEL_BUILDER(Name("GeodesicOptimizerStep").Device(DEVICE_CPU), GeodesicOptimizerStepOp);
