// saguaro.native/ops/multi_stage_hamiltonian_op.cc
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "multi_stage_hamiltonian_op.h"
using namespace tensorflow;

REGISTER_OP("MultiStageHamiltonian")
    .Input("state_q: float").Input("state_p: float").Input("params: float")
    .Output("evolved_q: float").Output("evolved_p: float")
    .Attr("dt: float = 0.01").Attr("num_stages: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0)); c->set_output(1, c->input(1)); return Status();
    }).Doc("Phase 70: Multi-stage Hamiltonian evolution.");

class MultiStageHamiltonianOp : public OpKernel {
 public:
  explicit MultiStageHamiltonianOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dt", &dt_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_stages", &stages_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto q = ctx->input(0), p = ctx->input(1), params = ctx->input(2);
    int batch = q.dim_size(0), dim = q.dim_size(1);
    Tensor *out_q, *out_p;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, q.shape(), &out_q));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, p.shape(), &out_p));
    std::copy_n(q.flat<float>().data(), batch*dim, out_q->flat<float>().data());
    std::copy_n(p.flat<float>().data(), batch*dim, out_p->flat<float>().data());
    saguaro::msham::MultiStageHamiltonianForward(out_q->flat<float>().data(),
        out_p->flat<float>().data(), params.flat<float>().data(), dt_, stages_, batch, dim);
  }
 private:
  float dt_; int stages_;
};
REGISTER_KERNEL_BUILDER(Name("MultiStageHamiltonian").Device(DEVICE_CPU), MultiStageHamiltonianOp);
