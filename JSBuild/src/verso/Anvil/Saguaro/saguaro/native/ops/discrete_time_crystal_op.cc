// saguaro.native/ops/discrete_time_crystal_op.cc
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
 * @file discrete_time_crystal_op.cc
 * @brief Phase 38: Discrete Time Crystal TensorFlow custom operations.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "discrete_time_crystal_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: DTCStabilizedEvolution
// =============================================================================

REGISTER_OP("DTCStabilizedEvolution")
    .Input("hidden_state: float")         // [batch, seq, state_dim]
    .Input("h_evolution: float")          // [state_dim, state_dim]
    .Output("stabilized_state: float")    // [batch, seq, state_dim]
    .Attr("floquet_period: int = 4")
    .Attr("coupling_j: float = 1.0")
    .Attr("disorder_w: float = 0.5")
    .Attr("pi_pulse_error: float = 0.01")
    .Attr("use_prethermal: bool = true")
    .Attr("num_cycles: int = 1")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Phase 38: Discrete Time Crystal Stabilized Evolution.

Applies DTC dynamics to stabilize TimeCrystal hidden states:
1. Floquet driving with half-period Hamiltonian evolution
2. π-pulse rotation for period-doubling
3. Many-body localization disorder for thermalization prevention

hidden_state: Input hidden state [batch, seq, state_dim]
h_evolution: Effective Hamiltonian [state_dim, state_dim]

stabilized_state: DTC-stabilized hidden state [batch, seq, state_dim]
)doc");

class DTCStabilizedEvolutionOp : public OpKernel {
 public:
  explicit DTCStabilizedEvolutionOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("floquet_period", &config_.floquet_period));
    OP_REQUIRES_OK(context, context->GetAttr("coupling_j", &config_.coupling_J));
    OP_REQUIRES_OK(context, context->GetAttr("disorder_w", &config_.disorder_W));
    OP_REQUIRES_OK(context, context->GetAttr("pi_pulse_error", &config_.pi_pulse_error));
    OP_REQUIRES_OK(context, context->GetAttr("use_prethermal", &config_.use_prethermal_phase));
    OP_REQUIRES_OK(context, context->GetAttr("num_cycles", &config_.num_floquet_cycles));
    
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& hidden_state = context->input(0);
    const Tensor& h_evolution = context->input(1);

    OP_REQUIRES(context, hidden_state.dims() == 3,
                errors::InvalidArgument("hidden_state must be 3D [batch, seq, state_dim]"));
    OP_REQUIRES(context, h_evolution.dims() == 2,
                errors::InvalidArgument("h_evolution must be 2D [state_dim, state_dim]"));

    const int batch_size = hidden_state.dim_size(0);
    const int seq_len = hidden_state.dim_size(1);
    const int state_dim = hidden_state.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, hidden_state.shape(), &output));

    // Copy input to output
    std::memcpy(output->flat<float>().data(),
                hidden_state.flat<float>().data(),
                hidden_state.NumElements() * sizeof(float));

    // Apply DTC stabilization
    saguaro::dtc::DTCStabilizedEvolution(
        output->flat<float>().data(),
        h_evolution.flat<float>().data(),
        config_,
        batch_size, seq_len, state_dim
    );
  }

 private:
  saguaro::dtc::DTCConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("DTCStabilizedEvolution").Device(DEVICE_CPU),
                        DTCStabilizedEvolutionOp);

// =============================================================================
// OP REGISTRATION: DTCOrderParameter
// =============================================================================

REGISTER_OP("DTCOrderParameter")
    .Input("state: float")                // [batch, seq, state_dim]
    .Output("magnetization: float")       // [batch, seq]
    .Output("dtc_order: float")           // [batch]
    .Attr("floquet_period: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle state = c->input(0);
        if (c->RankKnown(state) && c->Rank(state) == 3) {
            shape_inference::DimensionHandle batch = c->Dim(state, 0);
            shape_inference::DimensionHandle seq = c->Dim(state, 1);
            c->set_output(0, c->MakeShape({batch, seq}));
            c->set_output(1, c->MakeShape({batch}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Compute DTC order parameter from hidden state magnetization.

state: Hidden state [batch, seq, state_dim]

magnetization: Magnetization time series [batch, seq]
dtc_order: DTC phase order strength [batch] (0 to 1, higher = more DTC)
)doc");

class DTCOrderParameterOp : public OpKernel {
 public:
  explicit DTCOrderParameterOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("floquet_period", &floquet_period_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& state = context->input(0);

    const int batch_size = state.dim_size(0);
    const int seq_len = state.dim_size(1);
    const int state_dim = state.dim_size(2);

    Tensor* magnetization = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len}), &magnetization));
    
    Tensor* dtc_order = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size}), &dtc_order));

    // Compute magnetization
    saguaro::dtc::ComputeDTCOrderParameter(
        state.flat<float>().data(),
        magnetization->flat<float>().data(),
        batch_size, seq_len, state_dim
    );

    // Compute DTC order for each batch
    float* mag_data = magnetization->flat<float>().data();
    float* order_data = dtc_order->flat<float>().data();

    for (int b = 0; b < batch_size; ++b) {
      order_data[b] = saguaro::dtc::ComputeDTCPhaseOrder(
          mag_data + b * seq_len,
          seq_len, floquet_period_
      );
    }
  }

 private:
  int floquet_period_;
};

REGISTER_KERNEL_BUILDER(Name("DTCOrderParameter").Device(DEVICE_CPU),
                        DTCOrderParameterOp);

// =============================================================================
// OP REGISTRATION: ApplyPiPulse (standalone for testing/debugging)
// =============================================================================

REGISTER_OP("ApplyPiPulse")
    .Input("state: float")                // [batch, seq, state_dim]
    .Output("rotated_state: float")       // [batch, seq, state_dim]
    .Attr("error: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Apply π-pulse rotation with controlled error for DTC dynamics.

state: Input state [batch, seq, state_dim]
error: π-pulse imperfection ε (default: 0.01)

rotated_state: Rotated state [batch, seq, state_dim]
)doc");

class ApplyPiPulseOp : public OpKernel {
 public:
  explicit ApplyPiPulseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("error", &error_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& state = context->input(0);

    const int batch_size = state.dim_size(0);
    const int seq_len = state.dim_size(1);
    const int state_dim = state.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, state.shape(), &output));

    std::memcpy(output->flat<float>().data(),
                state.flat<float>().data(),
                state.NumElements() * sizeof(float));

    saguaro::dtc::ApplyPiPulse(
        output->flat<float>().data(),
        error_,
        batch_size, seq_len, state_dim
    );
  }

 private:
  float error_;
};

REGISTER_KERNEL_BUILDER(Name("ApplyPiPulse").Device(DEVICE_CPU),
                        ApplyPiPulseOp);
