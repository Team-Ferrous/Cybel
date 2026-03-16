// saguaro.native/ops/fused_qbm_op.cc
#include "fused_qbm_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FusedQBMEnergy")
    .Input("visible: float32").Input("hidden: float32").Input("weights: float32")
    .Output("energy: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc("Compute QBM energy.");

class FusedQBMEnergyOp : public OpKernel {
 public:
    explicit FusedQBMEnergyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
    void Compute(OpKernelContext* ctx) override {
        const Tensor& v = ctx->input(0);
        const Tensor& h = ctx->input(1);
        const Tensor& w = ctx->input(2);
        Tensor* out = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
        out->scalar<float>()() = saguaro::ops::qbm_energy(
            v.flat<float>().data(), h.flat<float>().data(), w.flat<float>().data(),
            v.NumElements(), h.NumElements());
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedQBMEnergy").Device(DEVICE_CPU), FusedQBMEnergyOp);
} // namespace tensorflow
