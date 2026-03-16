// saguaro/_native/ops/q_ssm_gating_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "q_ssm_gating_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QSSMForward
// =============================================================================

REGISTER_OP("QSSMForward")
    .Input("input: float")                // [batch, seq, input_dim]
    .Input("state: float")                // [batch, state_dim]
    .Input("vqc_params: float")           // [vqc_layers, vqc_qubits, 2]
    .Output("output: float")              // [batch, seq, input_dim]
    .Output("final_state: float")         // [batch, state_dim]
    .Attr("state_dim: int = 16")
    .Attr("input_dim: int = 64")
    .Attr("vqc_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("use_born_rule: bool = true")
    .Attr("measurement_temp: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        return Status();
    })
    .Doc(R"doc(
Phase 69: Q-SSM Forward pass with VQC gating.

Implements quantum-optimized selective state space model:
  S_t = σ_VQC(x_t) ⊙ S_{t-1} + (1 - σ_VQC(x_t)) ⊙ Update(x_t)

input: Input sequence [batch, seq, input_dim]
state: Initial SSM state [batch, state_dim]
vqc_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2]

output: Output sequence [batch, seq, input_dim]
final_state: Updated state [batch, state_dim]
)doc");

class QSSMForwardOp : public OpKernel {
 public:
  explicit QSSMForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("input_dim", &config_.input_dim));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_qubits", &config_.vqc_qubits));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(context, context->GetAttr("use_born_rule", &config_.use_born_rule));
    OP_REQUIRES_OK(context, context->GetAttr("measurement_temp", &config_.measurement_temp));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& state = context->input(1);
    const Tensor& vqc_params = context->input(2);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
    
    Tensor* final_state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, state.shape(), &final_state));

    // Copy initial state
    std::copy_n(state.flat<float>().data(),
                batch_size * config_.state_dim,
                final_state->flat<float>().data());

    saguaro::qssm::QSSMForward(
        input.flat<float>().data(),
        final_state->flat<float>().data(),
        vqc_params.flat<float>().data(),
        output->flat<float>().data(),
        config_,
        batch_size, seq_len
    );
  }

 private:
  saguaro::qssm::QSSMConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QSSMForward").Device(DEVICE_CPU), QSSMForwardOp);

// =============================================================================
// OP REGISTRATION: VQCGateExpectation
// =============================================================================

REGISTER_OP("VQCGateExpectation")
    .Input("encoded_input: float")        // [batch, vqc_qubits]
    .Input("rotation_params: float")      // [vqc_layers, vqc_qubits, 2]
    .Output("gate_values: float")         // [batch]
    .Attr("vqc_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("use_born_rule: bool = true")
    .Attr("measurement_temp: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input = c->input(0);
        if (c->RankKnown(input) && c->Rank(input) == 2) {
            c->set_output(0, c->MakeShape({c->Dim(input, 0)}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 69: Compute VQC expectation values for gating.

Simulates RY-RX VQC with CNOT entanglement and measures ⟨Z⟩.

encoded_input: Encoded input features [batch, vqc_qubits]
rotation_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2]

gate_values: Gate values in [0, 1] range [batch]
)doc");

class VQCGateExpectationOp : public OpKernel {
 public:
  explicit VQCGateExpectationOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("vqc_qubits", &config_.vqc_qubits));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(context, context->GetAttr("use_born_rule", &config_.use_born_rule));
    OP_REQUIRES_OK(context, context->GetAttr("measurement_temp", &config_.measurement_temp));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& encoded_input = context->input(0);
    const Tensor& rotation_params = context->input(1);

    const int batch_size = encoded_input.dim_size(0);

    Tensor* gate_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size}), &gate_values));

    saguaro::qssm::VQCGateExpectation(
        encoded_input.flat<float>().data(),
        rotation_params.flat<float>().data(),
        gate_values->flat<float>().data(),
        config_,
        batch_size
    );
  }

 private:
  saguaro::qssm::QSSMConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("VQCGateExpectation").Device(DEVICE_CPU), VQCGateExpectationOp);

// =============================================================================
// OP REGISTRATION: QSSMComputeGates
// =============================================================================

REGISTER_OP("QSSMComputeGates")
    .Input("input: float")                // [batch, seq, input_dim]
    .Input("vqc_params: float")           // [vqc_layers, vqc_qubits, 2]
    .Output("gate_values: float")         // [batch, seq]
    .Attr("input_dim: int = 64")
    .Attr("vqc_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("use_born_rule: bool = true")
    .Attr("measurement_temp: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input = c->input(0);
        if (c->RankKnown(input) && c->Rank(input) == 3) {
            c->set_output(0, c->MakeShape({c->Dim(input, 0), c->Dim(input, 1)}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 69: Compute Q-SSM gate values for full sequence.

For monitoring and visualization of gating behavior.

input: Input sequence [batch, seq, input_dim]
vqc_params: VQC rotation parameters [vqc_layers, vqc_qubits, 2]

gate_values: Gate values for each position [batch, seq]
)doc");

class QSSMComputeGatesOp : public OpKernel {
 public:
  explicit QSSMComputeGatesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("input_dim", &config_.input_dim));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_qubits", &config_.vqc_qubits));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(context, context->GetAttr("use_born_rule", &config_.use_born_rule));
    OP_REQUIRES_OK(context, context->GetAttr("measurement_temp", &config_.measurement_temp));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& vqc_params = context->input(1);

    const int batch_size = input.dim_size(0);
    const int seq_len = input.dim_size(1);

    Tensor* gate_values = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len}), &gate_values));

    saguaro::qssm::ComputeGateValues(
        input.flat<float>().data(),
        vqc_params.flat<float>().data(),
        gate_values->flat<float>().data(),
        config_,
        batch_size, seq_len
    );
  }

 private:
  saguaro::qssm::QSSMConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QSSMComputeGates").Device(DEVICE_CPU), QSSMComputeGatesOp);
