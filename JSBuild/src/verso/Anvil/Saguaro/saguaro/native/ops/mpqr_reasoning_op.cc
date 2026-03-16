// saguaro.native/ops/mpqr_reasoning_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "mpqr_reasoning_op.h"

using namespace tensorflow;

REGISTER_OP("MPQRReasoning")
    .Input("initial_thought: float")
    .Input("quality_oracle: float")
    .Output("path_states: float")
    .Output("path_amplitudes: float")
    .Attr("num_paths: int = 8")
    .Attr("grover_iterations: int = 3")
    .Attr("quality_threshold: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_paths;
        TF_RETURN_IF_ERROR(c->GetAttr("num_paths", &num_paths));
        auto batch = c->Dim(c->input(0), 0);
        auto dim = c->Dim(c->input(0), 1);
        c->set_output(0, c->MakeShape({batch, dim}));
        c->set_output(1, c->MakeShape({batch, num_paths}));
        return Status();
    })
    .Doc("Phase 55: Multi-path quantum reasoning with Grover amplification.");

class MPQRReasoningOp : public OpKernel {
 public:
  explicit MPQRReasoningOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_paths", &num_paths_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("grover_iterations", &iters_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("quality_threshold", &threshold_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto initial = ctx->input(0);
    auto quality = ctx->input(1);
    int batch = initial.dim_size(0), dim = initial.dim_size(1);
    Tensor* path_states = nullptr;
    Tensor* path_amps = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, dim}, &path_states));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {batch, num_paths_}, &path_amps));
    saguaro::mpqr::AmplifiedPathReasoning(initial.flat<float>().data(),
        path_states->flat<float>().data(), path_amps->flat<float>().data(),
        quality.flat<float>().data(), iters_, threshold_, batch, num_paths_, dim);
  }
 private:
  int num_paths_, iters_;
  float threshold_;
};
REGISTER_KERNEL_BUILDER(Name("MPQRReasoning").Device(DEVICE_CPU), MPQRReasoningOp);
