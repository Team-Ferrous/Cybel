// saguaro.native/ops/topological_wavelet_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "topological_wavelet_op.h"

using namespace tensorflow;

REGISTER_OP("TopologicalWaveletAttention")
    .Input("input: float")
    .Input("values: float")
    .Output("output: float")
    .Attr("num_scales: int = 4")
    .Attr("threshold: float = 0.1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        auto dim = c->Dim(c->input(0), 2);
        c->set_output(0, c->MakeShape({batch, dim}));
        return Status();
    })
    .Doc("Phase 56: Topological wavelet attention with Betti number bias.");

class TopologicalWaveletAttentionOp : public OpKernel {
 public:
  explicit TopologicalWaveletAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_scales", &num_scales_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    auto values = ctx->input(1);
    int batch = input.dim_size(0), seq = input.dim_size(1), dim = input.dim_size(2);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, dim}, &output));
    saguaro::twa::TopologicalWaveletAttention(input.flat<float>().data(),
        values.flat<float>().data(), output->flat<float>().data(),
        batch, seq, dim, num_scales_, threshold_);
  }
 private:
  int num_scales_;
  float threshold_;
};
REGISTER_KERNEL_BUILDER(Name("TopologicalWaveletAttention").Device(DEVICE_CPU),
                        TopologicalWaveletAttentionOp);
