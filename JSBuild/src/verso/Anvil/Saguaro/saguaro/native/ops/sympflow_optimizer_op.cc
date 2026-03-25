// saguaro.native/ops/sympflow_optimizer_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "sympflow_optimizer_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: SympFlowStep
// =============================================================================

REGISTER_OP("SympFlowStep")
    .Input("params: float")               // [num_params]
    .Input("momentum: float")             // [num_params]
    .Input("gradients: float")            // [num_params]
    .Output("params_new: float")          // [num_params]
    .Output("momentum_new: float")        // [num_params]
    .Attr("mass: float = 1.0")
    .Attr("friction: float = 0.1")
    .Attr("step_size: float = 0.01")
    .Attr("num_leapfrog_steps: int = 1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status();
    })
    .Doc(R"doc(
Phase 46: SympFlow Hamiltonian Optimizer Step.

Treats optimization as Hamiltonian dynamics with Leapfrog integration:
1. Half momentum update
2. Full position (parameter) update
3. Half momentum update

params: Current parameters [num_params]
momentum: Current momentum [num_params]
gradients: Parameter gradients [num_params]

params_new: Updated parameters [num_params]
momentum_new: Updated momentum [num_params]
)doc");

class SympFlowStepOp : public OpKernel {
 public:
  explicit SympFlowStepOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mass", &config_.mass));
    OP_REQUIRES_OK(context, context->GetAttr("friction", &config_.friction));
    OP_REQUIRES_OK(context, context->GetAttr("step_size", &config_.step_size));
    OP_REQUIRES_OK(context, context->GetAttr("num_leapfrog_steps", &config_.num_leapfrog_steps));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& params = context->input(0);
    const Tensor& momentum = context->input(1);
    const Tensor& gradients = context->input(2);

    const int num_params = params.NumElements();

    Tensor* params_new = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, params.shape(), &params_new));
    
    Tensor* momentum_new = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, momentum.shape(), &momentum_new));

    // Copy inputs to outputs
    std::memcpy(params_new->flat<float>().data(),
                params.flat<float>().data(),
                num_params * sizeof(float));
    std::memcpy(momentum_new->flat<float>().data(),
                momentum.flat<float>().data(),
                num_params * sizeof(float));

    saguaro::sympflow::LeapfrogStep(
        params_new->flat<float>().data(),
        momentum_new->flat<float>().data(),
        gradients.flat<float>().data(),
        config_, num_params
    );
  }

 private:
  saguaro::sympflow::SympFlowConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("SympFlowStep").Device(DEVICE_CPU), SympFlowStepOp);

// =============================================================================
// OP REGISTRATION: KineticEnergy
// =============================================================================

REGISTER_OP("SympFlowKineticEnergy")
    .Input("momentum: float")             // [num_params]
    .Output("kinetic_energy: float")      // scalar
    .Attr("mass: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return Status();
    })
    .Doc(R"doc(
Compute kinetic energy from momentum.

KE = 0.5 * Σ p² / m

momentum: Momentum vector [num_params]

kinetic_energy: Scalar kinetic energy
)doc");

class SympFlowKineticEnergyOp : public OpKernel {
 public:
  explicit SympFlowKineticEnergyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("mass", &mass_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& momentum = context->input(0);
    const int num_params = momentum.NumElements();

    Tensor* ke = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &ke));

    float energy = saguaro::sympflow::ComputeKineticEnergy(
        momentum.flat<float>().data(), mass_, num_params
    );

    ke->scalar<float>()() = energy;
  }

 private:
  float mass_;
};

REGISTER_KERNEL_BUILDER(Name("SympFlowKineticEnergy").Device(DEVICE_CPU),
                        SympFlowKineticEnergyOp);
