// saguaro.native/ops/hd_timecrystal_op.cc
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
 * @file hd_timecrystal_op.cc
 * @brief Phase 200+: HD TimeCrystal Block TensorFlow custom operations.
 *
 * SAGUARO_UPGRADE_ROADMAP.md Phase 2.2 - Block-level HD integration.
 *
 * Registers TensorFlow ops for Floquet domain evolution in HD space.
 * Replaces TimeCrystalBlock when USE_HD_STREAMING is enabled.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "hd_timecrystal_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: HDTimeCrystalForward
// =============================================================================

REGISTER_OP("HDTimeCrystalForward")
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("floquet_energies: float")   // [floquet_modes, hd_dim]
    .Input("drive_weights: float")      // [floquet_modes]
    .Input("coupling_matrix: float")    // [floquet_modes, floquet_modes]
    .Output("hd_output: float")         // [batch, seq_len, hd_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("floquet_modes: int = 16")
    .Attr("drive_frequency: float = 1.0")
    .Attr("drive_amplitude: float = 0.1")
    .Attr("dt: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // Output same shape as input
        return Status();
    })
    .Doc(R"doc(
HD TimeCrystal Forward Pass - Floquet domain evolution for HD bundles.

Phase 200+: Block-integrated HD streaming. Replaces TimeCrystalBlock
when USE_HD_STREAMING=True. Performs Floquet harmonic decomposition,
quasi-energy evolution, and DTC inter-mode coupling.

hd_input: HD bundle input [batch, seq_len, hd_dim]
floquet_energies: Quasi-energy spectrum [floquet_modes, hd_dim]
drive_weights: Periodic drive coupling [floquet_modes]
coupling_matrix: DTC mode-mode coupling [floquet_modes, floquet_modes]

hd_output: Evolved HD bundles [batch, seq_len, hd_dim]
)doc");

// =============================================================================
// KERNEL: HDTimeCrystalForward
// =============================================================================

class HDTimeCrystalForwardOp : public OpKernel {
 public:
  explicit HDTimeCrystalForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("floquet_modes", &config_.floquet_modes));
    OP_REQUIRES_OK(context, context->GetAttr("drive_frequency", &config_.drive_frequency));
    OP_REQUIRES_OK(context, context->GetAttr("drive_amplitude", &config_.drive_amplitude));
    OP_REQUIRES_OK(context, context->GetAttr("dt", &config_.dt));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_input = context->input(0);
    const Tensor& floquet_energies = context->input(1);
    const Tensor& drive_weights = context->input(2);
    const Tensor& coupling_matrix = context->input(3);

    // Validate shapes
    OP_REQUIRES(context, hd_input.dims() == 3,
                errors::InvalidArgument("hd_input must be 3D [batch, seq, hd_dim]"));

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);

    OP_REQUIRES(context, hd_dim == config_.hd_dim,
                errors::InvalidArgument("hd_dim mismatch"));

    // Allocate output
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_input.shape(), &hd_output));

    // Call kernel
    saguaro::hd_timecrystal::HDTimeCrystalForward(
        hd_input.flat<float>().data(),
        floquet_energies.flat<float>().data(),
        drive_weights.flat<float>().data(),
        coupling_matrix.flat<float>().data(),
        hd_output->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::hd_timecrystal::HDTimeCrystalConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HDTimeCrystalForward").Device(DEVICE_CPU),
                        HDTimeCrystalForwardOp);

// =============================================================================
// OP REGISTRATION: HDTimeCrystalBackward
// =============================================================================

REGISTER_OP("HDTimeCrystalBackward")
    .Input("grad_output: float")        // [batch, seq_len, hd_dim]
    .Input("hd_input: float")           // [batch, seq_len, hd_dim]
    .Input("floquet_energies: float")   // [floquet_modes, hd_dim]
    .Input("drive_weights: float")      // [floquet_modes]
    .Input("coupling_matrix: float")    // [floquet_modes, floquet_modes]
    .Output("grad_input: float")        // [batch, seq_len, hd_dim]
    .Output("grad_energies: float")     // [floquet_modes, hd_dim]
    .Output("grad_drive: float")        // [floquet_modes]
    .Output("grad_coupling: float")     // [floquet_modes, floquet_modes]
    .Attr("hd_dim: int = 4096")
    .Attr("floquet_modes: int = 16")
    .Attr("drive_frequency: float = 1.0")
    .Attr("drive_amplitude: float = 0.1")
    .Attr("dt: float = 0.01")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input same as hd_input
        c->set_output(1, c->input(2));  // grad_energies same as floquet_energies
        c->set_output(2, c->input(3));  // grad_drive same as drive_weights
        c->set_output(3, c->input(4));  // grad_coupling same as coupling_matrix
        return Status();
    })
    .Doc("HD TimeCrystal Backward Pass - Gradient computation via adjoint method.");

// =============================================================================
// KERNEL: HDTimeCrystalBackward
// =============================================================================

class HDTimeCrystalBackwardOp : public OpKernel {
 public:
  explicit HDTimeCrystalBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("floquet_modes", &config_.floquet_modes));
    OP_REQUIRES_OK(context, context->GetAttr("drive_frequency", &config_.drive_frequency));
    OP_REQUIRES_OK(context, context->GetAttr("drive_amplitude", &config_.drive_amplitude));
    OP_REQUIRES_OK(context, context->GetAttr("dt", &config_.dt));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& grad_output = context->input(0);
    const Tensor& hd_input = context->input(1);
    const Tensor& floquet_energies = context->input(2);
    const Tensor& drive_weights = context->input(3);
    const Tensor& coupling_matrix = context->input(4);

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);

    // Allocate gradient tensors
    Tensor* grad_input = nullptr;
    Tensor* grad_energies = nullptr;
    Tensor* grad_drive = nullptr;
    Tensor* grad_coupling = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, floquet_energies.shape(), &grad_energies));
    OP_REQUIRES_OK(context, context->allocate_output(
        2, drive_weights.shape(), &grad_drive));
    OP_REQUIRES_OK(context, context->allocate_output(
        3, coupling_matrix.shape(), &grad_coupling));

    // Call gradient kernel
    saguaro::hd_timecrystal::HDTimeCrystalBackward(
        grad_output.flat<float>().data(),
        hd_input.flat<float>().data(),
        floquet_energies.flat<float>().data(),
        drive_weights.flat<float>().data(),
        coupling_matrix.flat<float>().data(),
        grad_input->flat<float>().data(),
        grad_energies->flat<float>().data(),
        grad_drive->flat<float>().data(),
        grad_coupling->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::hd_timecrystal::HDTimeCrystalConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HDTimeCrystalBackward").Device(DEVICE_CPU),
                        HDTimeCrystalBackwardOp);
