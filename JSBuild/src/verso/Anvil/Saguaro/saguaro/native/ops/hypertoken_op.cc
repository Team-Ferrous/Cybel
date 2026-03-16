// saguaro.native/ops/hypertoken_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "hypertoken_op.h"

using namespace tensorflow;

REGISTER_OP("EncodeHypertoken")
    .Input("subword_ids: int32")
    .Input("subword_lengths: int32")
    .Input("embedding_table: float")
    .Output("hypertoken: float")
    .Attr("token_dim: int = 512")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int token_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("token_dim", &token_dim));
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch, token_dim}));
        return Status();
    })
    .Doc("Phase 49: Encode subwords into holographic hypertoken.");

class EncodeHypertokenOp : public OpKernel {
 public:
  explicit EncodeHypertokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("token_dim", &config_.token_dim));
  }
  void Compute(OpKernelContext* ctx) override {
    auto subword_ids = ctx->input(0);
    auto subword_lengths = ctx->input(1);
    auto embedding_table = ctx->input(2);
    int batch = subword_ids.dim_size(0), max_subwords = subword_ids.dim_size(1);
    int vocab_size = embedding_table.dim_size(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch, config_.token_dim}, &output));
    saguaro::hypertoken::EncodeHypertoken(
        subword_ids.flat<int>().data(), subword_lengths.flat<int>().data(),
        embedding_table.flat<float>().data(), output->flat<float>().data(),
        config_, batch, max_subwords, vocab_size);
  }
 private:
  saguaro::hypertoken::HypertokenConfig config_;
};
REGISTER_KERNEL_BUILDER(Name("EncodeHypertoken").Device(DEVICE_CPU), EncodeHypertokenOp);

REGISTER_OP("GroverRetrieve")
    .Input("hypertoken: float")
    .Output("retrieved: float")
    .Attr("attribute_index: int = 0")
    .Attr("grover_iterations: int = 3")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc("Phase 49: Grover-style attribute retrieval from hypertoken.");

class GroverRetrieveOp : public OpKernel {
 public:
  explicit GroverRetrieveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("attribute_index", &attr_idx_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("grover_iterations", &iters_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto input = ctx->input(0);
    int batch = input.dim_size(0), dim = input.dim_size(1);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &output));
    for (int b = 0; b < batch; ++b) {
      saguaro::hypertoken::GroverRetrieve(
          input.flat<float>().data() + b * dim, attr_idx_,
          output->flat<float>().data() + b * dim, iters_, dim);
    }
  }
 private:
  int attr_idx_, iters_;
};
REGISTER_KERNEL_BUILDER(Name("GroverRetrieve").Device(DEVICE_CPU), GroverRetrieveOp);
