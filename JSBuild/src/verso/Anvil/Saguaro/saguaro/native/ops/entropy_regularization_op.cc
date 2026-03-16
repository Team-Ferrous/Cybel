// saguaro.native/ops/entropy_regularization_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "entropy_regularization_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: VonNeumannEntropyLoss
// =============================================================================

REGISTER_OP("VonNeumannEntropyLoss")
    .Input("activations: float")          // [batch, dim]
    .Output("loss: float")                // scalar
    .Output("entropy: float")             // scalar (metric)
    .Output("flatness: float")            // scalar (metric)
    .Attr("entropy_weight: float = 0.01")
    .Attr("spectral_weight: float = 0.01")
    .Attr("target_entropy: float = 0.5")
    .Attr("spectral_flatness_target: float = 0.8")
    .Attr("power_iter_steps: int = 10")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Phase 45: Von Neumann Entropy Regularization Loss.

Computes entropy-based regularization from activation covariance:
1. Compute covariance matrix
2. Extract eigenvalues via power iteration
3. Von Neumann entropy: S = -Σ λ log λ
4. Spectral flatness penalty

activations: Activations [batch, dim]

loss: Total regularization loss
entropy: Von Neumann entropy metric
flatness: Spectral flatness metric
)doc");

class VonNeumannEntropyLossOp : public OpKernel {
 public:
  explicit VonNeumannEntropyLossOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("entropy_weight", &config_.entropy_weight));
    OP_REQUIRES_OK(context, context->GetAttr("spectral_weight", &config_.spectral_weight));
    OP_REQUIRES_OK(context, context->GetAttr("target_entropy", &config_.target_entropy));
    OP_REQUIRES_OK(context, context->GetAttr("spectral_flatness_target", 
                                              &config_.spectral_flatness_target));
    OP_REQUIRES_OK(context, context->GetAttr("power_iter_steps", &config_.power_iter_steps));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& activations = context->input(0);

    const int batch_size = activations.dim_size(0);
    const int dim = activations.dim_size(1);

    Tensor* loss = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &loss));
    
    Tensor* entropy = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}), &entropy));
    
    Tensor* flatness = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &flatness));

    float loss_val = saguaro::entropy_reg::ComputeEntropyRegularization(
        activations.flat<float>().data(),
        batch_size, dim, config_
    );

    float entropy_val, flatness_val;
    saguaro::entropy_reg::ComputeEntropyMetrics(
        activations.flat<float>().data(),
        &entropy_val, &flatness_val,
        batch_size, dim
    );

    loss->scalar<float>()() = loss_val;
    entropy->scalar<float>()() = entropy_val;
    flatness->scalar<float>()() = flatness_val;
  }

 private:
  saguaro::entropy_reg::EntropyRegConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("VonNeumannEntropyLoss").Device(DEVICE_CPU),
                        VonNeumannEntropyLossOp);

// =============================================================================
// OP REGISTRATION: ComputeCovariance
// =============================================================================

REGISTER_OP("ComputeActivationCovariance")
    .Input("activations: float")          // [batch, dim]
    .Output("covariance: float")          // [dim, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle act = c->input(0);
        if (c->RankKnown(act) && c->Rank(act) == 2) {
            auto dim = c->Dim(act, 1);
            c->set_output(0, c->MakeShape({dim, dim}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Compute covariance matrix from activations.

activations: Input activations [batch, dim]

covariance: Covariance matrix [dim, dim]
)doc");

class ComputeActivationCovarianceOp : public OpKernel {
 public:
  explicit ComputeActivationCovarianceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& activations = context->input(0);

    const int batch_size = activations.dim_size(0);
    const int dim = activations.dim_size(1);

    Tensor* covariance = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({dim, dim}), &covariance));

    saguaro::entropy_reg::ComputeCovariance(
        activations.flat<float>().data(),
        covariance->flat<float>().data(),
        batch_size, dim
    );
  }
};

REGISTER_KERNEL_BUILDER(Name("ComputeActivationCovariance").Device(DEVICE_CPU),
                        ComputeActivationCovarianceOp);
