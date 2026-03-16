// saguaro.native/ops/born_rule_loss_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "born_rule_loss_op.h"

using namespace tensorflow;

REGISTER_OP("BornRuleLoss")
    .Input("logits: float")
    .Input("targets: int32")
    .Output("loss: float")
    .Output("grad_logits: float")
    .Attr("temperature: float = 1.0")
    .Attr("use_qfim: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch}));
        c->set_output(1, c->input(0));
        return Status();
    })
    .Doc("Phase 51: Born rule loss with QFIM gradients.");

class BornRuleLossOp : public OpKernel {
 public:
  explicit BornRuleLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temp_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("use_qfim", &use_qfim_));
  }
  void Compute(OpKernelContext* ctx) override {
    auto logits = ctx->input(0);
    auto targets = ctx->input(1);
    int batch = logits.dim_size(0), seq = logits.dim_size(1), vocab = logits.dim_size(2);
    Tensor* loss = nullptr;
    Tensor* grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch}, &loss));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, logits.shape(), &grad));
    saguaro::qbrl::BornRuleLoss(logits.flat<float>().data(), targets.flat<int>().data(),
        loss->flat<float>().data(), grad->flat<float>().data(),
        batch, seq, vocab, temp_, use_qfim_);
  }
 private:
  float temp_;
  bool use_qfim_;
};
REGISTER_KERNEL_BUILDER(Name("BornRuleLoss").Device(DEVICE_CPU), BornRuleLossOp);
