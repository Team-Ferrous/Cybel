// saguaro.native/ops/quantum_fidelity_loss_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantum_fidelity_loss_op.h"

using namespace tensorflow;

REGISTER_OP("QuantumFidelityLoss")
    .Input("pred_states: float")
    .Input("true_states: float")
    .Output("loss: float")
    .Output("grad: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        auto batch = c->Dim(c->input(0), 0);
        c->set_output(0, c->MakeShape({batch}));
        c->set_output(1, c->input(0));
        return Status();
    })
    .Doc("Phase 52: Quantum fidelity loss between predicted and true states.");

class QuantumFidelityLossOp : public OpKernel {
 public:
  explicit QuantumFidelityLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    auto pred = ctx->input(0);
    auto true_s = ctx->input(1);
    int batch = pred.dim_size(0), dim = pred.dim_size(1);
    Tensor* loss = nullptr;
    Tensor* grad = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, {batch}, &loss));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, pred.shape(), &grad));
    saguaro::qfidelity::QuantumFidelityLoss(pred.flat<float>().data(),
        true_s.flat<float>().data(), loss->flat<float>().data(),
        grad->flat<float>().data(), batch, dim);
  }
};
REGISTER_KERNEL_BUILDER(Name("QuantumFidelityLoss").Device(DEVICE_CPU), QuantumFidelityLossOp);
