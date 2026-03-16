// saguaro.native/ops/hyperdimensional_embedding_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "hyperdimensional_embedding_op.h"

using namespace tensorflow;

REGISTER_OP("HolographicBundle")
    .Input("token_ids: int32")
    .Input("base_vectors: float")
    .Input("position_keys: float")
    .Output("output: float")
    .Attr("hd_dim: int = 4096")
    .Attr("model_dim: int = 256")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int model_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("model_dim", &model_dim));
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch, model_dim}));
        return Status();
    })
    .Doc("Phase 48: Holographic bundling with FFT-based circular convolution binding.");

class HolographicBundleOp : public OpKernel {
 public:
  explicit HolographicBundleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("model_dim", &config_.model_dim));
  }
  void Compute(OpKernelContext* ctx) override {
    auto token_ids = ctx->input(0);
    auto base_vectors = ctx->input(1);
    auto position_keys = ctx->input(2);
    int batch = token_ids.dim_size(0), seq = token_ids.dim_size(1);
    int vocab_size = base_vectors.dim_size(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, config_.model_dim}, &output));
    saguaro::hqe::HolographicBundle(token_ids.flat<int>().data(),
        base_vectors.flat<float>().data(), position_keys.flat<float>().data(),
        nullptr, output->flat<float>().data(), config_, batch, seq, vocab_size);
  }
 private:
  saguaro::hqe::HQEConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("HolographicBundle").Device(DEVICE_CPU), HolographicBundleOp);

REGISTER_OP("CTQWSpread")
    .Input("embeddings: float")
    .Output("output: float")
    .Attr("steps: int = 3")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Phase 48: Continuous-time quantum walk semantic spreading.");

class CTQWSpreadOp : public OpKernel {
 public:
  explicit CTQWSpreadOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("steps", &steps_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    int batch = input.dim_size(0), dim = input.dim_size(1);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    std::copy_n(input.flat<float>().data(), batch * dim, output->flat<float>().data());
    saguaro::hqe::CTQWSpread(output->flat<float>().data(), batch, dim, steps_);
  }
 private:
  int steps_;
};
REGISTER_KERNEL_BUILDER(Name("CTQWSpread").Device(DEVICE_CPU), CTQWSpreadOp);
