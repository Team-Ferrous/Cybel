// saguaro.native/ops/random_natural_gradient_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "random_natural_gradient_op.h"
using namespace tensorflow;

REGISTER_OP("RandomNaturalGradient")
    .Input("params: float").Input("gradients: float")
    .Output("updated_params: float")
    .Attr("learning_rate: float = 0.001").Attr("num_samples: int = 10").Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status();
    }).Doc("Phase 72: Random natural gradient update.");

class RandomNaturalGradientOp : public OpKernel {
 public:
  explicit RandomNaturalGradientOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("learning_rate", &lr_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_samples", &samples_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto params = ctx->input(0), grads = ctx->input(1);
    int np = params.NumElements();
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, params.shape(), &out));
    std::copy_n(params.flat<float>().data(), np, out->flat<float>().data());
    saguaro::rng::RandomNaturalGradient(out->flat<float>().data(),
        grads.flat<float>().data(), lr_, np, samples_, seed_);
  }
 private:
  float lr_; int samples_, seed_;
};
REGISTER_KERNEL_BUILDER(Name("RandomNaturalGradient").Device(DEVICE_CPU), RandomNaturalGradientOp);
