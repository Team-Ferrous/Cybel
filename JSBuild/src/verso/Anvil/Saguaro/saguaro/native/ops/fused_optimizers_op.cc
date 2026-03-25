// saguaro.native/ops/fused_optimizers_op.cc
#include "fused_optimizers_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
REGISTER_OP("FusedAdamWUpdate")
    .Input("param: float32").Input("m: float32").Input("v: float32").Input("grad: float32")
    .Output("new_param: float32").Output("new_m: float32").Output("new_v: float32")
    .Attr("lr: float = 0.001").Attr("beta1: float = 0.9").Attr("beta2: float = 0.999")
    .Attr("eps: float = 1e-8").Attr("wd: float = 0.01").Attr("t: int = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); c->set_output(1, c->input(1)); c->set_output(2, c->input(2));
        return Status();
    })
    .Doc("Fused AdamW optimizer step.");

class FusedAdamWUpdateOp : public OpKernel {
    float lr_, beta1_, beta2_, eps_, wd_; int t_;
 public:
    explicit FusedAdamWUpdateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("lr", &lr_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta1", &beta1_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta2", &beta2_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("eps", &eps_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("wd", &wd_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("t", &t_));
    }
    void Compute(OpKernelContext* ctx) override {
        auto size = ctx->input(0).NumElements();
        Tensor *p, *m, *v;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, ctx->input(0).shape(), &p));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, ctx->input(1).shape(), &m));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, ctx->input(2).shape(), &v));
        std::copy(ctx->input(0).flat<float>().data(), ctx->input(0).flat<float>().data()+size, p->flat<float>().data());
        std::copy(ctx->input(1).flat<float>().data(), ctx->input(1).flat<float>().data()+size, m->flat<float>().data());
        std::copy(ctx->input(2).flat<float>().data(), ctx->input(2).flat<float>().data()+size, v->flat<float>().data());
        saguaro::ops::optimizer_adamw_update(
            p->flat<float>().data(), m->flat<float>().data(), v->flat<float>().data(),
            ctx->input(3).flat<float>().data(), size, lr_, beta1_, beta2_, eps_, wd_, t_);
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedAdamWUpdate").Device(DEVICE_CPU), FusedAdamWUpdateOp);
} // namespace tensorflow
