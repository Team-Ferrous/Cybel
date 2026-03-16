// saguaro.native/ops/qhd_spatial_block_op.cc
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
 * @file qhd_spatial_block_op.cc
 * @brief Phase 600+: Quantum HD Spatial Block TensorFlow custom operations.
 *
 * Registers TensorFlow ops for QHD-space Mamba SSM processing with
 * quantum superposition, VQC entanglement, and Born rule collapse.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "qhd_spatial_block_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: QHDSpatialBlockForward
// =============================================================================

REGISTER_OP("QHDSpatialBlockForward")
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("a_log: float")              // [state_dim]
    .Input("b_proj: float")             // [hd_dim, state_dim]
    .Input("c_proj: float")             // [hd_dim, state_dim]
    .Input("dt: float")                 // [hd_dim] Phase 900.2: 1D (broadcasted internally)
    .Input("skip_proj: float")          // [hd_dim, hd_dim]
    .Input("amplitudes_real: float")    // [num_paths]
    .Input("amplitudes_imag: float")    // [num_paths]
    .Input("rotation_angles: float")    // [entanglement_depth, num_paths]
    .Input("walk_hamiltonian: float")   // [num_paths, num_paths] UQHA quantum walk
    .Output("hd_output: float")         // [batch, seq_len, hd_dim]
    .Output("h_final: float")           // [batch, num_paths, state_dim, hd_dim]
    .Output("coherence: float")         // [batch]
    .Attr("hd_dim: int = 4096")
    .Attr("state_dim: int = 16")
    .Attr("hidden_dim: int = 512")
    .Attr("num_paths: int = 2")
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.3")
    // UQHA Phase 850-860 attributes
    .Attr("use_frequency_stratification: bool = true")
    .Attr("freq_mask_mode: int = 0")          // 0=exponential, 1=linear, 2=learned
    .Attr("freq_overlap: float = 0.25")
    .Attr("entanglement_topology: int = 2")   // 0=adjacent, 1=hierarchical, 2=walk
    .Attr("walk_evolution_time: float = 1.0")
    // UQHA v3.1 P0: Diagonal Skip Connection
    .Attr("skip_connection_type: int = 1")     // 0=dense, 1=diagonal, 2=identity, 3=scalar
    .Attr("skip_diagonal_init: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);

        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 3) {
            auto batch = c->Dim(input_shape, 0);
            auto seq_len = c->Dim(input_shape, 1);
            auto hd_dim = c->Dim(input_shape, 2);

            int state_dim, num_paths;
            c->GetAttr("state_dim", &state_dim);
            c->GetAttr("num_paths", &num_paths);

            // Output: [batch, seq_len, hd_dim]
            c->set_output(0, c->MakeShape({batch, seq_len, hd_dim}));

            // h_final: [batch, num_paths, state_dim, hd_dim]
            c->set_output(1, c->MakeShape({batch, num_paths, state_dim, hd_dim}));

            // coherence: [batch]
            c->set_output(2, c->MakeShape({batch}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
            c->set_output(2, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
UQHA v3.0 Quantum HD Spatial Block Forward Pass.
Combines FFT-domain SSM with quantum superposition (K paths), VQC entanglement, and Born rule collapse.
Phase 850: Frequency stratification encodes hierarchy in VQC frequency paths.
Phase 860: Quantum walk entanglement replaces O(D²) cross-attention with O(K²) unitary evolution.
Complexity is O(K * L * D log D) where K is num_paths.
)doc");


// =============================================================================
// KERNEL: QHDSpatialBlockForward
// =============================================================================

class QHDSpatialBlockForwardOp : public OpKernel {
 public:
  explicit QHDSpatialBlockForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("num_paths", &config_.num_paths));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", &config_.entanglement_depth));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", &config_.entanglement_strength));
    // UQHA Phase 850-860 attributes
    OP_REQUIRES_OK(context, context->GetAttr("use_frequency_stratification", &config_.use_frequency_stratification));
    OP_REQUIRES_OK(context, context->GetAttr("freq_mask_mode", &config_.freq_mask_mode));
    OP_REQUIRES_OK(context, context->GetAttr("freq_overlap", &config_.freq_overlap));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_topology", &config_.entanglement_topology));
    OP_REQUIRES_OK(context, context->GetAttr("walk_evolution_time", &config_.walk_evolution_time));
    // UQHA v3.1 P0 attributes
    OP_REQUIRES_OK(context, context->GetAttr("skip_connection_type", &config_.skip_connection_type));
    OP_REQUIRES_OK(context, context->GetAttr("skip_diagonal_init", &config_.skip_diagonal_init));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_input = context->input(0);
    const Tensor& a_log = context->input(1);
    const Tensor& b_proj = context->input(2);
    const Tensor& c_proj = context->input(3);
    const Tensor& dt = context->input(4);
    const Tensor& skip_proj = context->input(5);
    const Tensor& amplitudes_real = context->input(6);
    const Tensor& amplitudes_imag = context->input(7);
    const Tensor& rotation_angles = context->input(8);
    const Tensor& walk_hamiltonian = context->input(9);  // UQHA

    // Validate shapes
    OP_REQUIRES(context, hd_input.dims() == 3,
                errors::InvalidArgument("hd_input must be 3D [batch, seq, hd_dim]"));
    OP_REQUIRES(context, a_log.dims() == 1,
                errors::InvalidArgument("a_log must be 1D [state_dim]"));
    OP_REQUIRES(context, amplitudes_real.dims() == 1,
                errors::InvalidArgument("amplitudes_real must be 1D [num_paths]"));
    OP_REQUIRES(context, rotation_angles.dims() == 2,
                errors::InvalidArgument("rotation_angles must be 2D [ent_depth, num_paths]"));
    OP_REQUIRES(context, walk_hamiltonian.dims() == 2,
                errors::InvalidArgument("walk_hamiltonian must be 2D [num_paths, num_paths]"));

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);
    const int state_dim = a_log.dim_size(0);
    const int num_paths = amplitudes_real.dim_size(0);

    // Validate dimensions match config
    OP_REQUIRES(context, hd_dim == config_.hd_dim,
                errors::InvalidArgument("hd_dim mismatch: ", hd_dim, " vs ", config_.hd_dim));
    OP_REQUIRES(context, state_dim == config_.state_dim,
                errors::InvalidArgument("state_dim mismatch: ", state_dim, " vs ", config_.state_dim));
    OP_REQUIRES(context, num_paths == config_.num_paths,
                errors::InvalidArgument("num_paths mismatch: ", num_paths, " vs ", config_.num_paths));

    // Allocate output tensors
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, hd_dim}), &hd_output));

    Tensor* h_final = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, num_paths, state_dim, hd_dim}), &h_final));

    Tensor* coherence = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        2, TensorShape({batch_size}), &coherence));

    // Call kernel with UQHA walk_hamiltonian
    saguaro::qhd_spatial::QHDSpatialForward(
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        amplitudes_real.flat<float>().data(),
        amplitudes_imag.flat<float>().data(),
        rotation_angles.flat<float>().data(),
        walk_hamiltonian.flat<float>().data(),  // UQHA
        hd_output->flat<float>().data(),
        h_final->flat<float>().data(),
        coherence->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::qhd_spatial::QHDSpatialConfig config_;
};


REGISTER_KERNEL_BUILDER(Name("QHDSpatialBlockForward").Device(DEVICE_CPU),
                        QHDSpatialBlockForwardOp);

// =============================================================================
// OP REGISTRATION: QHDSpatialBlockBackward
// =============================================================================

REGISTER_OP("QHDSpatialBlockBackward")
    .Input("grad_output: float")        // [batch, seq_len, hd_dim]
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("a_log: float")              // [state_dim]
    .Input("b_proj: float")             // [hd_dim, state_dim]
    .Input("c_proj: float")             // [hd_dim, state_dim]
    .Input("dt: float")                 // [hd_dim] Phase 900.2: 1D (broadcasted internally)
    .Input("skip_proj: float")          // [hd_dim, hd_dim]
    .Input("amplitudes_real: float")    // [num_paths]
    .Input("amplitudes_imag: float")    // [num_paths]
    .Input("rotation_angles: float")    // [entanglement_depth, num_paths]
    .Input("walk_hamiltonian: float")    // [num_paths, num_paths] - UQHA
    .Output("grad_input: float")        // [batch, seq_len, hd_dim]
    .Output("grad_a_log: float")        // [state_dim]
    .Output("grad_b_proj: float")       // [hd_dim, state_dim]
    .Output("grad_c_proj: float")       // [hd_dim, state_dim]
    .Output("grad_dt: float")           // [hd_dim] Phase 900.2: 1D gradient
    .Output("grad_skip: float")         // [hd_dim, hd_dim]
    .Output("grad_amplitudes_real: float")  // [num_paths]
    .Output("grad_amplitudes_imag: float")  // [num_paths]
    .Output("grad_rotation_angles: float")  // [entanglement_depth, num_paths]
    .Output("grad_walk_hamiltonian: float") // [num_paths, num_paths] - UQHA
    .Attr("hd_dim: int = 4096")
    .Attr("state_dim: int = 16")
    .Attr("hidden_dim: int = 512")
    .Attr("num_paths: int = 2")
    .Attr("entanglement_depth: int = 2")
    .Attr("entanglement_strength: float = 0.3")
    // UQHA Phase 850-860 attributes (must match Forward)
    .Attr("use_frequency_stratification: bool = true")
    .Attr("freq_mask_mode: int = 0")
    .Attr("freq_overlap: float = 0.25")
    .Attr("entanglement_topology: int = 2")
    .Attr("walk_evolution_time: float = 1.0")
    // UQHA v3.1 P0: Diagonal Skip Connection
    .Attr("skip_connection_type: int = 1")
    .Attr("skip_diagonal_init: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));   // grad_input same as hd_input
        c->set_output(1, c->input(2));   // grad_a_log same as a_log
        c->set_output(2, c->input(3));   // grad_b_proj same as b_proj
        c->set_output(3, c->input(4));   // grad_c_proj same as c_proj
        c->set_output(4, c->input(5));   // grad_dt same as dt
        c->set_output(5, c->input(6));   // grad_skip same as skip_proj
        c->set_output(6, c->input(7));   // grad_amplitudes_real
        c->set_output(7, c->input(8));   // grad_amplitudes_imag
        c->set_output(8, c->input(9));   // grad_rotation_angles
        c->set_output(9, c->input(10));  // grad_walk_hamiltonian
        return Status();
    })
    .Doc("Quantum HD Spatial Block Backward Pass - Gradient computation for BPTT.");

// =============================================================================
// KERNEL: QHDSpatialBlockBackward
// =============================================================================

class QHDSpatialBlockBackwardOp : public OpKernel {
 public:
  explicit QHDSpatialBlockBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("num_paths", &config_.num_paths));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_depth", &config_.entanglement_depth));
    OP_REQUIRES_OK(context, context->GetAttr("entanglement_strength", &config_.entanglement_strength));
    // UQHA P0 & P1 attributes
    OP_REQUIRES_OK(context, context->GetAttr("skip_connection_type", &config_.skip_connection_type));
    OP_REQUIRES_OK(context, context->GetAttr("skip_diagonal_init", &config_.skip_diagonal_init));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& grad_output = context->input(0);
    const Tensor& hd_input = context->input(1);
    const Tensor& a_log = context->input(2);
    const Tensor& b_proj = context->input(3);
    const Tensor& c_proj = context->input(4);
    const Tensor& dt = context->input(5);
    const Tensor& skip_proj = context->input(6);
    const Tensor& amplitudes_real = context->input(7);
    const Tensor& amplitudes_imag = context->input(8);
    const Tensor& rotation_angles = context->input(9);
    const Tensor& walk_hamiltonian = context->input(10);  // UQHA

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);

    // Allocate gradient tensors
    Tensor* grad_input = nullptr;
    Tensor* grad_a_log = nullptr;
    Tensor* grad_b_proj = nullptr;
    Tensor* grad_c_proj = nullptr;
    Tensor* grad_dt = nullptr;
    Tensor* grad_skip = nullptr;
    Tensor* grad_amplitudes_real = nullptr;
    Tensor* grad_amplitudes_imag = nullptr;
    Tensor* grad_rotation_angles = nullptr;
    Tensor* grad_walk_hamiltonian = nullptr;  // UQHA

    OP_REQUIRES_OK(context, context->allocate_output(0, hd_input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, a_log.shape(), &grad_a_log));
    OP_REQUIRES_OK(context, context->allocate_output(2, b_proj.shape(), &grad_b_proj));
    OP_REQUIRES_OK(context, context->allocate_output(3, c_proj.shape(), &grad_c_proj));
    OP_REQUIRES_OK(context, context->allocate_output(4, dt.shape(), &grad_dt));
    OP_REQUIRES_OK(context, context->allocate_output(5, skip_proj.shape(), &grad_skip));
    OP_REQUIRES_OK(context, context->allocate_output(6, amplitudes_real.shape(), &grad_amplitudes_real));
    OP_REQUIRES_OK(context, context->allocate_output(7, amplitudes_imag.shape(), &grad_amplitudes_imag));
    OP_REQUIRES_OK(context, context->allocate_output(8, rotation_angles.shape(), &grad_rotation_angles));
    OP_REQUIRES_OK(context, context->allocate_output(9, walk_hamiltonian.shape(), &grad_walk_hamiltonian));  // UQHA

    // Call gradient kernel
    saguaro::qhd_spatial::QHDSpatialBackward(
        grad_output.flat<float>().data(),
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        amplitudes_real.flat<float>().data(),
        amplitudes_imag.flat<float>().data(),
        rotation_angles.flat<float>().data(),
        walk_hamiltonian.flat<float>().data(),  // UQHA input
        grad_input->flat<float>().data(),
        grad_a_log->flat<float>().data(),
        grad_b_proj->flat<float>().data(),
        grad_c_proj->flat<float>().data(),
        grad_dt->flat<float>().data(),
        grad_skip->flat<float>().data(),
        grad_amplitudes_real->flat<float>().data(),
        grad_amplitudes_imag->flat<float>().data(),
        grad_rotation_angles->flat<float>().data(),
        grad_walk_hamiltonian->flat<float>().data(),  // UQHA output
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::qhd_spatial::QHDSpatialConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("QHDSpatialBlockBackward").Device(DEVICE_CPU),
                        QHDSpatialBlockBackwardOp);
