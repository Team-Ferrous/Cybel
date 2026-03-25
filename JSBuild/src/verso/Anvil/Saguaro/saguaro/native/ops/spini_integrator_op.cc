// saguaro.native/ops/spini_integrator_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "spini_integrator_op.h"
using namespace tensorflow;

REGISTER_OP("SPINIOptimizer")
    .Input("params: float").Input("velocity: float").Input("gradients: float")
    .Output("updated_params: float").Output("updated_velocity: float")
    .Attr("learning_rate: float = 0.001").Attr("friction: float = 0.1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); c->set_output(1, c->input(1)); return Status();
    }).Doc("Phase 78: SPINI symplectic optimizer step.");

class SPINIOptimizerOp : public OpKernel {
 public:
  explicit SPINIOptimizerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("learning_rate", &lr_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("friction", &friction_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto params = ctx->input(0), vel = ctx->input(1), grads = ctx->input(2);
    int np = params.NumElements();
    Tensor *out_p, *out_v;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, params.shape(), &out_p));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, vel.shape(), &out_v));
    std::copy_n(params.flat<float>().data(), np, out_p->flat<float>().data());
    std::copy_n(vel.flat<float>().data(), np, out_v->flat<float>().data());
    saguaro::spini::SPINIOptimizerStep(out_p->flat<float>().data(),
        out_v->flat<float>().data(), grads.flat<float>().data(), lr_, friction_, np);
  }
 private:
  float lr_, friction_;
};
REGISTER_KERNEL_BUILDER(Name("SPINIOptimizer").Device(DEVICE_CPU), SPINIOptimizerOp);
