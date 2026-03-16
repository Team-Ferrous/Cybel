// saguaro.native/ops/fused_qgan_op.cc
#include "fused_qgan_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FusedQGANWasserstein")
    .Input("real: float32").Input("fake: float32")
    .Output("distance: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc("Compute Wasserstein distance.");

class FusedQGANWassersteinOp : public OpKernel {
 public:
    explicit FusedQGANWassersteinOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<float>()() = saguaro::ops::qgan_wasserstein(
            ctx->input(0).flat<float>().data(), ctx->input(1).flat<float>().data(),
            ctx->input(0).NumElements());
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedQGANWasserstein").Device(DEVICE_CPU), FusedQGANWassersteinOp);
} // namespace tensorflow
