// saguaro.native/ops/quantum_teleport_bus_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "quantum_teleport_bus_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: TeleportState
// =============================================================================

REGISTER_OP("QuantumTeleportState")
    .Input("input: float")                // [batch, dim]
    .Output("output: float")              // [batch, dim]
    .Output("fidelity: float")            // [batch]
    .Attr("entanglement_dim: int = 64")
    .Attr("fidelity_threshold: float = 0.9")
    .Attr("use_error_correction: bool = true")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        shape_inference::ShapeHandle input = c->input(0);
        if (c->RankKnown(input) && c->Rank(input) == 2) {
            c->set_output(1, c->MakeShape({c->Dim(input, 0)}));
        } else {
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 44: Quantum State Teleportation for Cross-Block Communication.

Teleportation Steps:
  1. Create Bell pair shared between source and destination
  2. Bell measurement at source produces 2 classical bits
  3. Apply Pauli corrections at destination

input: State to teleport [batch, dim]

output: Teleported state [batch, dim]
fidelity: Teleportation fidelity [batch]
)doc");

class QuantumTeleportStateOp : public OpKernel {
 public:
  explicit QuantumTeleportStateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_dim", &config_.entanglement_dim));
    OP_REQUIRES_OK(context, context->GetAttr("fidelity_threshold", &config_.fidelity_threshold));
    OP_REQUIRES_OK(context, context->GetAttr("use_error_correction", &config_.use_error_correction));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const int batch_size = input.dim_size(0);
    const int dim = input.dim_size(1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
    
    Tensor* fidelity = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size}), &fidelity));

    saguaro::teleport_bus::TeleportState(
        input.flat<float>().data(),
        output->flat<float>().data(),
        config_, batch_size, dim
    );

    saguaro::teleport_bus::ComputeFidelity(
        input.flat<float>().data(),
        output->flat<float>().data(),
        fidelity->flat<float>().data(),
        batch_size, dim
    );
  }

 private:
  saguaro::teleport_bus::TeleportConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumTeleportState").Device(DEVICE_CPU),
                        QuantumTeleportStateOp);

// =============================================================================
// OP REGISTRATION: BellMeasurement
// =============================================================================

REGISTER_OP("BellMeasurement")
    .Input("state_a: float")              // [batch, dim]
    .Input("state_b: float")              // [batch, dim]
    .Output("classical_bits: int32")      // [batch, 2]
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle state = c->input(0);
        if (c->RankKnown(state) && c->Rank(state) == 2) {
            c->set_output(0, c->MakeShape({c->Dim(state, 0), 2}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Perform Bell measurement on two states.

state_a: First state [batch, dim]
state_b: Second state [batch, dim]

classical_bits: 2-bit measurement outcome [batch, 2]
)doc");

class BellMeasurementOp : public OpKernel {
 public:
  explicit BellMeasurementOp(OpKernelConstruction* context) : OpKernel(context) {
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    seed_ = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& state_a = context->input(0);
    const Tensor& state_b = context->input(1);

    const int batch_size = state_a.dim_size(0);
    const int dim = state_a.dim_size(1);

    Tensor* classical_bits = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, 2}), &classical_bits));

    saguaro::teleport_bus::BellMeasurement(
        state_a.flat<float>().data(),
        state_b.flat<float>().data(),
        classical_bits->flat<int>().data(),
        batch_size, dim, seed_
    );
  }

 private:
  uint32_t seed_;
};

REGISTER_KERNEL_BUILDER(Name("BellMeasurement").Device(DEVICE_CPU),
                        BellMeasurementOp);
