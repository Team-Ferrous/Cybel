// saguaro.native/ops/qmoe_routing_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file qmoe_routing_op.cc
 * @brief Phase 42: QMoE TensorFlow custom operations.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "qmoe_routing_op.h"
#include "common/edition_limits.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QMoERouting
// =============================================================================

REGISTER_OP("QMoERouting")
    .Input("token_embeddings: float")     // [batch, seq, dim]
    .Input("vqc_angles: float")           // [num_layers, num_qubits, 2]
    .Output("expert_probs: float")        // [batch, seq, num_experts]
    .Output("top_k_indices: int32")       // [batch, seq, top_k]
    .Output("top_k_weights: float")       // [batch, seq, top_k]
    .Attr("num_qubits: int = 4")
    .Attr("vqc_layers: int = 2")
    .Attr("num_experts: int = 8")
    .Attr("top_k: int = 2")
    .Attr("temperature: float = 1.0")
    .Attr("use_entanglement: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle embeddings = c->input(0);
        if (c->RankKnown(embeddings) && c->Rank(embeddings) == 3) {
            auto batch = c->Dim(embeddings, 0);
            auto seq = c->Dim(embeddings, 1);
            
            int num_experts, top_k;
            c->GetAttr("num_experts", &num_experts);
            c->GetAttr("top_k", &top_k);
            
            c->set_output(0, c->MakeShape({batch, seq, num_experts}));
            c->set_output(1, c->MakeShape({batch, seq, top_k}));
            c->set_output(2, c->MakeShape({batch, seq, top_k}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
            c->set_output(2, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 42: Quantum Mixture of Experts Routing.

VQC-based routing for expert selection:
1. Amplitude encode tokens into quantum state
2. Apply variational quantum circuit with learnable angles
3. Born rule measurement for expert probabilities
4. Top-K selection with normalized weights

token_embeddings: Token embeddings [batch, seq, dim]
vqc_angles: VQC rotation angles [num_layers, num_qubits, 2]

expert_probs: Expert probabilities [batch, seq, num_experts]
top_k_indices: Top-K expert indices [batch, seq, top_k]
top_k_weights: Top-K normalized weights [batch, seq, top_k]
)doc");

class QMoERoutingOp : public OpKernel {
 public:
  explicit QMoERoutingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_qubits", &config_.num_qubits));
    OP_REQUIRES_OK(context, context->GetAttr("vqc_layers", &config_.vqc_layers));
    OP_REQUIRES_OK(context, context->GetAttr("num_experts", &config_.num_experts));
    OP_REQUIRES_OK(context, context->GetAttr("top_k", &config_.top_k));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &config_.measurement_temperature));
    OP_REQUIRES_OK(context, context->GetAttr("use_entanglement", &config_.use_entanglement));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& embeddings = context->input(0);
    const Tensor& vqc_angles = context->input(1);

    OP_REQUIRES(context, embeddings.dims() == 3,
                errors::InvalidArgument("embeddings must be 3D [batch, seq, dim]"));

    const int batch_size = embeddings.dim_size(0);
    const int seq_len = embeddings.dim_size(1);
    const int dim = embeddings.dim_size(2);

    // HighNoon Lite Edition: Enforce MoE expert limit (max 12)
    SAGUARO_CHECK_MOE_EXPERTS(context, config_.num_experts);

    // Allocate outputs
    Tensor* expert_probs = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, config_.num_experts}), &expert_probs));
    
    Tensor* top_k_indices = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, seq_len, config_.top_k}), &top_k_indices));
    
    Tensor* top_k_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        2, TensorShape({batch_size, seq_len, config_.top_k}), &top_k_weights));

    saguaro::qmoe::QMoERouting(
        embeddings.flat<float>().data(),
        vqc_angles.flat<float>().data(),
        expert_probs->flat<float>().data(),
        top_k_indices->flat<int>().data(),
        top_k_weights->flat<float>().data(),
        config_,
        batch_size, seq_len, dim
    );
  }

 private:
  saguaro::qmoe::QMoEConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QMoERouting").Device(DEVICE_CPU), QMoERoutingOp);

// =============================================================================
// OP REGISTRATION: AmplitudeEncode
// =============================================================================

REGISTER_OP("QuantumAmplitudeEncode")
    .Input("features: float")             // [batch, dim]
    .Output("state: float")               // [batch, 2^num_qubits]
    .Attr("num_qubits: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle features = c->input(0);
        if (c->RankKnown(features) && c->Rank(features) == 2) {
            int num_qubits;
            c->GetAttr("num_qubits", &num_qubits);
            auto batch = c->Dim(features, 0);
            c->set_output(0, c->MakeShape({batch, 1 << num_qubits}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Amplitude encode features into quantum state.

|ψ⟩ = Σ_i (f_i / ||f||) |i⟩

features: Input features [batch, dim]

state: Quantum state amplitudes [batch, 2^num_qubits]
)doc");

class QuantumAmplitudeEncodeOp : public OpKernel {
 public:
  explicit QuantumAmplitudeEncodeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_qubits", &num_qubits_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& features = context->input(0);

    const int batch_size = features.dim_size(0);
    const int dim = features.dim_size(1);
    const int state_dim = 1 << num_qubits_;

    Tensor* state = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, state_dim}), &state));

    const float* feat_data = features.flat<float>().data();
    float* state_data = state->flat<float>().data();

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
      saguaro::qmoe::AmplitudeEncode(
          feat_data + b * dim,
          state_data + b * state_dim,
          dim, num_qubits_
      );
    }
  }

 private:
  int num_qubits_;
};

REGISTER_KERNEL_BUILDER(Name("QuantumAmplitudeEncode").Device(DEVICE_CPU),
                        QuantumAmplitudeEncodeOp);

// =============================================================================
// OP REGISTRATION: BornRuleMeasurement
// =============================================================================

REGISTER_OP("BornRuleMeasurement")
    .Input("state: float")                // [batch, num_experts]
    .Output("probs: float")               // [batch, num_experts]
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Born rule measurement: P(i) = |⟨ψ|i⟩|²

state: Quantum state amplitudes [batch, num_experts]

probs: Expert probabilities [batch, num_experts]
)doc");

class BornRuleMeasurementOp : public OpKernel {
 public:
  explicit BornRuleMeasurementOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& state = context->input(0);

    const int batch_size = state.dim_size(0);
    const int num_experts = state.dim_size(1);

    Tensor* probs = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, state.shape(), &probs));

    const float* state_data = state.flat<float>().data();
    float* probs_data = probs->flat<float>().data();

    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
      saguaro::qmoe::BornRuleMeasurement(
          state_data + b * num_experts,
          probs_data + b * num_experts,
          num_experts, temperature_
      );
    }
  }

 private:
  float temperature_;
};

REGISTER_KERNEL_BUILDER(Name("BornRuleMeasurement").Device(DEVICE_CPU),
                        BornRuleMeasurementOp);
