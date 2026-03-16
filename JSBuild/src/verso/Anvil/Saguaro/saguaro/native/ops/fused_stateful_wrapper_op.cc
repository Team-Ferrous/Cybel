// saguaro.native/ops/fused_stateful_wrapper_op.cc
#include "fused_stateful_wrapper_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FusedStatefulReset")
    .Input("state: float32")
    .Output("reset_state: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Reset stateful wrapper state to zeros.");

class FusedStatefulResetOp : public OpKernel {
 public:
    explicit FusedStatefulResetOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ctx->input(0).shape(), &out));
        saguaro::ops::stateful_reset(out->flat<float>().data(), out->NumElements());
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedStatefulReset").Device(DEVICE_CPU), FusedStatefulResetOp);
} // namespace tensorflow
