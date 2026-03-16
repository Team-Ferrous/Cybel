// saguaro.native/ops/qmamba_op.cc
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
 * @file qmamba_op.cc
 * @brief Phase 37: QMamba TensorFlow custom operations.
 *
 * Registers TensorFlow ops for quantum-enhanced selective state space model.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "qmamba_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QMambaSelectiveScan
// =============================================================================

REGISTER_OP("QMambaSelectiveScan")
    .Input("x: float")                    // [batch, seq_len, d_inner]
    .Input("a_log: float")                // [d_inner, state_dim]
    .Input("b: float")                    // [batch, seq_len, state_dim]
    .Input("c: float")                    // [batch, seq_len, state_dim]
    .Input("dt: float")                   // [batch, seq_len, d_inner]
    .Input("rotation_angles: float")      // [entanglement_depth, num_states]
    .Output("output: float")              // [batch, seq_len, d_inner]
    .Output("h_super: float")             // [batch, K, d_inner, state_dim]
    .Attr("num_superposition_states: int = 4")
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.5")
    .Attr("use_amplitude_selection: bool = true")
    .Attr("gumbel_temperature: float = 1.0")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // x: [batch, seq_len, d_inner]
        shape_inference::ShapeHandle x_shape = c->input(0);
        
        if (c->RankKnown(x_shape) && c->Rank(x_shape) == 3) {
            shape_inference::DimensionHandle batch = c->Dim(x_shape, 0);
            shape_inference::DimensionHandle seq_len = c->Dim(x_shape, 1);
            shape_inference::DimensionHandle d_inner = c->Dim(x_shape, 2);
            
            // Output: [batch, seq_len, d_inner]
            c->set_output(0, c->MakeShape({batch, seq_len, d_inner}));
            
            // h_super: [batch, K, d_inner, state_dim] - state_dim from a_log
            int K;
            c->GetAttr("num_superposition_states", &K);
            shape_inference::ShapeHandle a_shape = c->input(1);
            if (c->RankKnown(a_shape) && c->Rank(a_shape) == 2) {
                shape_inference::DimensionHandle state_dim = c->Dim(a_shape, 1);
                c->set_output(1, c->MakeShape({batch, K, d_inner, state_dim}));
            } else {
                c->set_output(1, c->UnknownShape());
            }
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Phase 37: QMamba Quantum-Enhanced Selective State Space Model.

Extends Mamba SSM with quantum superposition states for enhanced context
understanding. Each of K parallel state paths evolves independently with
entanglement-induced correlations, then collapses via Born rule or
Gumbel-Softmax.

x: Input sequence [batch, seq_len, d_inner]
a_log: Log of decay rates [d_inner, state_dim]
b: B projections [batch, seq_len, state_dim]
c: C projections [batch, seq_len, state_dim]
dt: Discretization timesteps [batch, seq_len, d_inner]
rotation_angles: VQC angles [entanglement_depth, num_states]

output: SSM output [batch, seq_len, d_inner]
h_super: Final superposition states [batch, K, d_inner, state_dim]
)doc");

// =============================================================================
// KERNEL: QMambaSelectiveScan
// =============================================================================

class QMambaSelectiveScanOp : public OpKernel {
 public:
  explicit QMambaSelectiveScanOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("num_superposition_states", 
                                              &config_.num_superposition_states));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", 
                                              &config_.entanglement_depth));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", 
                                              &config_.entanglement_strength));
    OP_REQUIRES_OK(context, context->GetAttr("use_amplitude_selection", 
                                              &config_.use_amplitude_selection));
    OP_REQUIRES_OK(context, context->GetAttr("gumbel_temperature", 
                                              &config_.gumbel_temperature));
    
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    config_.seed = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& x = context->input(0);
    const Tensor& a_log = context->input(1);
    const Tensor& b = context->input(2);
    const Tensor& c = context->input(3);
    const Tensor& dt = context->input(4);
    const Tensor& rotation_angles = context->input(5);

    // Validate shapes
    OP_REQUIRES(context, x.dims() == 3,
                errors::InvalidArgument("x must be 3D [batch, seq, d_inner]"));
    OP_REQUIRES(context, a_log.dims() == 2,
                errors::InvalidArgument("a_log must be 2D [d_inner, state_dim]"));
    
    const int batch_size = x.dim_size(0);
    const int seq_len = x.dim_size(1);
    const int d_inner = x.dim_size(2);
    const int state_dim = a_log.dim_size(1);
    const int K = config_.num_superposition_states;

    // Allocate output tensors
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, d_inner}), &output));
    
    Tensor* h_super_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, K, d_inner, state_dim}), &h_super_out));

    // Get data pointers
    const float* x_data = x.flat<float>().data();
    const float* a_log_data = a_log.flat<float>().data();
    const float* b_data = b.flat<float>().data();
    const float* c_data = c.flat<float>().data();
    const float* dt_data = dt.flat<float>().data();
    const float* angles_data = rotation_angles.flat<float>().data();
    float* output_data = output->flat<float>().data();
    float* h_super_data = h_super_out->flat<float>().data();

    // Allocate skip connection (D) - use zeros for now, can be added as input
    std::vector<float> D(d_inner, 0.0f);

    // Run QMamba selective scan
    saguaro::qmamba::QMambaSelectiveScan(
        x_data, h_super_data,
        a_log_data, b_data, c_data, dt_data,
        angles_data, config_,
        batch_size, seq_len, d_inner, state_dim
    );

    // Compute outputs at each timestep
    for (int t = 0; t < seq_len; ++t) {
      const float* c_t = c_data + t * state_dim;  // Simplified: use same C for all batch
      const float* x_t = x_data + t * d_inner;
      float* out_t = output_data + t * d_inner;
      
      saguaro::qmamba::QMambaOutput(
          h_super_data,
          c_t, D.data(), x_t,
          out_t, config_,
          batch_size, d_inner, state_dim
      );
    }
  }

 private:
  saguaro::qmamba::QMambaConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QMambaSelectiveScan").Device(DEVICE_CPU),
                        QMambaSelectiveScanOp);

// =============================================================================
// OP REGISTRATION: QMambaEntangle
// =============================================================================

REGISTER_OP("QMambaEntangle")
    .Input("states: float")               // [batch, K, state_dim]
    .Input("rotation_angles: float")      // [entanglement_depth, K]
    .Output("entangled_states: float")    // [batch, K, state_dim]
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Apply VQC-inspired entanglement layers to superposition states.

states: Input states [batch, K, state_dim]
rotation_angles: VQC angles [entanglement_depth, K]

entangled_states: Output states [batch, K, state_dim]
)doc");

class QMambaEntangleOp : public OpKernel {
 public:
  explicit QMambaEntangleOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", &depth_));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", &strength_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& states = context->input(0);
    const Tensor& angles = context->input(1);

    const int batch_size = states.dim_size(0);
    const int num_states = states.dim_size(1);
    const int state_dim = states.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, states.shape(), &output));

    // Copy input to output
    std::memcpy(output->flat<float>().data(),
                states.flat<float>().data(),
                states.NumElements() * sizeof(float));

    const float* angles_data = angles.flat<float>().data();
    float* out_data = output->flat<float>().data();

    // Apply entanglement layers
    for (int layer = 0; layer < depth_; ++layer) {
#if defined(__AVX2__)
      saguaro::qmamba::ApplyEntanglementLayerAVX2(
          out_data, angles_data, strength_,
          batch_size, num_states, state_dim, layer
      );
#else
      saguaro::qmamba::ApplyEntanglementLayer(
          out_data, angles_data, strength_,
          batch_size, num_states, state_dim, layer
      );
#endif
    }
  }

 private:
  int depth_;
  float strength_;
};

REGISTER_KERNEL_BUILDER(Name("QMambaEntangle").Device(DEVICE_CPU),
                        QMambaEntangleOp);

// =============================================================================
// OP REGISTRATION: QMambaCollapse
// =============================================================================

REGISTER_OP("QMambaCollapse")
    .Input("h_super: float")              // [batch, K, state_dim]
    .Input("path_logits: float")          // [batch, K] (optional, can be zeros)
    .Output("h_collapsed: float")         // [batch, state_dim]
    .Attr("use_born_rule: bool = true")
    .Attr("temperature: float = 1.0")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle h_super = c->input(0);
        if (c->RankKnown(h_super) && c->Rank(h_super) == 3) {
            c->set_output(0, c->MakeShape({c->Dim(h_super, 0), c->Dim(h_super, 2)}));
        } else {
            c->set_output(0, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
Collapse superposition states via Born rule or Gumbel-Softmax.

h_super: Superposition states [batch, K, state_dim]
path_logits: Path selection logits [batch, K]

h_collapsed: Collapsed state [batch, state_dim]
)doc");

class QMambaCollapseOp : public OpKernel {
 public:
  explicit QMambaCollapseOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_born_rule", &use_born_rule_));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    int seed;
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed));
    seed_ = static_cast<uint32_t>(seed);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& h_super = context->input(0);
    const Tensor& path_logits = context->input(1);

    const int batch_size = h_super.dim_size(0);
    const int num_states = h_super.dim_size(1);
    const int state_dim = h_super.dim_size(2);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, state_dim}), &output));

    const float* h_data = h_super.flat<float>().data();
    const float* logits_data = path_logits.flat<float>().data();
    float* out_data = output->flat<float>().data();

    if (use_born_rule_) {
#if defined(__AVX512F__)
      saguaro::qmamba::BornRuleCollapseAVX512(
          h_data, out_data,
          batch_size, num_states, state_dim
      );
#else
      saguaro::qmamba::BornRuleCollapse(
          h_data, out_data,
          batch_size, num_states, state_dim
      );
#endif
    } else {
      saguaro::qmamba::GumbelSoftmaxCollapse(
          h_data, logits_data, out_data,
          batch_size, num_states, state_dim,
          temperature_, seed_
      );
    }
  }

 private:
  bool use_born_rule_;
  float temperature_;
  uint32_t seed_;
};

REGISTER_KERNEL_BUILDER(Name("QMambaCollapse").Device(DEVICE_CPU),
                        QMambaCollapseOp);
