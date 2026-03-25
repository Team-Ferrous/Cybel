// saguaro.native/ops/majorana_position_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "majorana_position_op.h"

using namespace tensorflow;

REGISTER_OP("MajoranaPositionEncode")
    .Input("positions: int32")
    .Output("encoding: float")
    .Attr("dim: int")
    .Attr("floquet_period: int = 4")
    .Attr("majorana_mass: float = 0.1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int dim;
        TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));
        auto batch = c->Dim(c->input(0), 0);
        auto seq = c->Dim(c->input(0), 1);
        c->set_output(0, c->MakeShape({batch, seq, dim}));
        return Status();
    })
    .Doc("Phase 50: Majorana position encoding with Floquet drive.");

class MajoranaPositionEncodeOp : public OpKernel {
 public:
  explicit MajoranaPositionEncodeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("floquet_period", &period_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("majorana_mass", &mass_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto positions = ctx->input(0);
    int batch = positions.dim_size(0), seq = positions.dim_size(1);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, seq, dim_}, &output));
    saguaro::majorana::MajoranaEncode(positions.flat<int>().data(),
        output->flat<float>().data(), batch, seq, dim_, period_, mass_);
  }
 private:
  int dim_, period_;
  float mass_;
};
REGISTER_KERNEL_BUILDER(Name("MajoranaPositionEncode").Device(DEVICE_CPU), MajoranaPositionEncodeOp);
