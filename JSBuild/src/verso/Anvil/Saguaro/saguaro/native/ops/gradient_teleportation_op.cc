// saguaro.native/ops/gradient_teleportation_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "gradient_teleportation_op.h"

using namespace tensorflow;

REGISTER_OP("TeleportGradients")
    .Input("local_grads: float").Input("bell_channel: float")
    .Output("teleported: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); return Status();
    })
    .Doc("Phase 64: Teleport gradients via Bell channel.");

class TeleportGradientsOp : public OpKernel {
 public:
  explicit TeleportGradientsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto lg = ctx->input(0), bc = ctx->input(1);
    int batch = lg.dim_size(0), np = lg.dim_size(1);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, lg.shape(), &out));
    saguaro::gradtele::TeleportGradients(lg.flat<float>().data(), bc.flat<float>().data(),
        out->flat<float>().data(), batch, np);
  }
};
REGISTER_KERNEL_BUILDER(Name("TeleportGradients").Device(DEVICE_CPU), TeleportGradientsOp);
