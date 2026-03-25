// saguaro.native/ops/hd_spatial_block_op.cc
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
 * @file hd_spatial_block_op.cc
 * @brief Phase 200+: HD Spatial Block TensorFlow custom operations.
 *
 * SAGUARO_UPGRADE_ROADMAP.md Phase 2.2 - Block-level HD integration.
 *
 * Registers TensorFlow ops for HD-space Mamba SSM processing.
 * Replaces QMamba/SpatialBlock when USE_HD_STREAMING is enabled.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "hd_spatial_block_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: HDSpatialBlockForward
// =============================================================================

REGISTER_OP("HDSpatialBlockForward")
    .Input("hd_input: float")       // [batch, seq_len, hd_dim]
    .Input("a_log: float")          // [state_dim]
    .Input("b_proj: float")         // [hd_dim, state_dim]
    .Input("c_proj: float")         // [hd_dim, state_dim]
    .Input("dt: float")             // [hd_dim] Phase 900.2: 1D (broadcasted internally)
    .Input("skip_proj: float")      // [hd_dim, hd_dim]
    .Input("floquet_phases: float") // [hd_dim] - Phase 700 position encoding
    .Output("hd_output: float")     // [batch, seq_len, hd_dim]
    .Output("h_final: float")       // [batch, state_dim, hd_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("state_dim: int = 16")
    .Attr("hidden_dim: int = 512")
    .Attr("use_floquet: bool = true")  // Enable/disable Floquet position encoding
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);

        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 3) {
            auto batch = c->Dim(input_shape, 0);
            auto seq_len = c->Dim(input_shape, 1);
            auto hd_dim = c->Dim(input_shape, 2);

            // Output: [batch, seq_len, hd_dim]
            c->set_output(0, c->MakeShape({batch, seq_len, hd_dim}));

            // h_final: [batch, state_dim, hd_dim]
            int state_dim;
            c->GetAttr("state_dim", &state_dim);
            c->set_output(1, c->MakeShape({batch, state_dim, hd_dim}));
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->UnknownShape());
        }
        return Status();
    })
    .Doc(R"doc(
HD Spatial Block Forward Pass - FFT-domain Mamba SSM for HD bundles.

Phase 200+: Block-integrated HD streaming. Replaces QMamba/SpatialBlock
when USE_HD_STREAMING=True. Processes HD bundles directly using FFT-domain
operations for O(D log D) complexity.

hd_input: HD bundle input [batch, seq_len, hd_dim]
a_log: Log of SSM decay rates [state_dim]
b_proj: B projection weights [hd_dim, state_dim]
c_proj: C projection weights [hd_dim, state_dim]
dt: Discretization timesteps [hd_dim] (Phase 900.2: C++ broadcasts internally)
skip_proj: Skip connection projection [hd_dim, hd_dim]
floquet_phases: Floquet position encoding phases [hd_dim] (Phase 700)

hd_output: Processed HD bundles [batch, seq_len, hd_dim]
h_final: Final SSM hidden states [batch, state_dim, hd_dim]
)doc");

// =============================================================================
// KERNEL: HDSpatialBlockForward
// =============================================================================

class HDSpatialBlockForwardOp : public OpKernel {
 public:
  explicit HDSpatialBlockForwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("use_floquet", &use_floquet_));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_input = context->input(0);
    const Tensor& a_log = context->input(1);
    const Tensor& b_proj = context->input(2);
    const Tensor& c_proj = context->input(3);
    const Tensor& dt = context->input(4);
    const Tensor& skip_proj = context->input(5);
    const Tensor& floquet_phases = context->input(6);

    // Validate shapes
    OP_REQUIRES(context, hd_input.dims() == 3,
                errors::InvalidArgument("hd_input must be 3D [batch, seq, hd_dim]"));
    OP_REQUIRES(context, a_log.dims() == 1,
                errors::InvalidArgument("a_log must be 1D [state_dim]"));
    OP_REQUIRES(context, b_proj.dims() == 2,
                errors::InvalidArgument("b_proj must be 2D [hd_dim, state_dim]"));
    OP_REQUIRES(context, c_proj.dims() == 2,
                errors::InvalidArgument("c_proj must be 2D [hd_dim, state_dim]"));
    OP_REQUIRES(context, floquet_phases.dims() == 1,
                errors::InvalidArgument("floquet_phases must be 1D [hd_dim]"));

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);
    const int state_dim = a_log.dim_size(0);

    // Validate dimensions match config
    OP_REQUIRES(context, hd_dim == config_.hd_dim,
                errors::InvalidArgument("hd_dim mismatch: ", hd_dim, " vs ", config_.hd_dim));
    OP_REQUIRES(context, state_dim == config_.state_dim,
                errors::InvalidArgument("state_dim mismatch: ", state_dim, " vs ", config_.state_dim));

    // Allocate output tensors
    Tensor* hd_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len, hd_dim}), &hd_output));

    Tensor* h_final = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({batch_size, state_dim, hd_dim}), &h_final));

    // Call kernel with Floquet phases (Phase 700)
    saguaro::hd_spatial::HDSpatialForward(
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        use_floquet_ ? floquet_phases.flat<float>().data() : nullptr,
        hd_output->flat<float>().data(),
        h_final->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::hd_spatial::HDSpatialConfig config_;
  bool use_floquet_ = true;
};

REGISTER_KERNEL_BUILDER(Name("HDSpatialBlockForward").Device(DEVICE_CPU),
                        HDSpatialBlockForwardOp);

// =============================================================================
// OP REGISTRATION: HDSpatialBlockBackward
// =============================================================================

REGISTER_OP("HDSpatialBlockBackward")
    .Input("grad_output: float")    // [batch, seq_len, hd_dim]
    .Input("hd_input: float")       // [batch, seq_len, hd_dim]
    .Input("a_log: float")          // [state_dim]
    .Input("b_proj: float")         // [hd_dim, state_dim]
    .Input("c_proj: float")         // [hd_dim, state_dim]
    .Input("dt: float")             // [hd_dim] Phase 900.2: 1D (broadcasted internally)
    .Input("skip_proj: float")      // [hd_dim, hd_dim]
    .Input("floquet_phases: float") // [hd_dim] - Phase 700 position encoding
    .Output("grad_input: float")    // [batch, seq_len, hd_dim]
    .Output("grad_a_log: float")    // [state_dim]
    .Output("grad_b_proj: float")   // [hd_dim, state_dim]
    .Output("grad_c_proj: float")   // [hd_dim, state_dim]
    .Output("grad_dt: float")       // [hd_dim] Phase 900.2: 1D gradient
    .Output("grad_skip: float")     // [hd_dim, hd_dim]
    .Output("grad_floquet: float")  // [hd_dim] - Phase 700
    .Attr("hd_dim: int = 4096")
    .Attr("state_dim: int = 16")
    .Attr("hidden_dim: int = 512")
    .Attr("use_floquet: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input same as hd_input
        c->set_output(1, c->input(2));  // grad_a_log same as a_log
        c->set_output(2, c->input(3));  // grad_b_proj same as b_proj
        c->set_output(3, c->input(4));  // grad_c_proj same as c_proj
        c->set_output(4, c->input(5));  // grad_dt same as dt
        c->set_output(5, c->input(6));  // grad_skip same as skip_proj
        c->set_output(6, c->input(7));  // grad_floquet same as floquet_phases
        return Status();
    })
    .Doc("HD Spatial Block Backward Pass - Gradient computation for BPTT with Floquet position encoding.");

// =============================================================================
// KERNEL: HDSpatialBlockBackward
// =============================================================================

class HDSpatialBlockBackwardOp : public OpKernel {
 public:
  explicit HDSpatialBlockBackwardOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("state_dim", &config_.state_dim));
    OP_REQUIRES_OK(context, context->GetAttr("hidden_dim", &config_.hidden_dim));
    OP_REQUIRES_OK(context, context->GetAttr("use_floquet", &use_floquet_));
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
    const Tensor& floquet_phases = context->input(7);  // Phase 700

    const int batch_size = hd_input.dim_size(0);
    const int seq_len = hd_input.dim_size(1);
    const int hd_dim = hd_input.dim_size(2);

    // Allocate gradient tensors
    Tensor* grad_input = nullptr;
    Tensor* grad_a_log = nullptr;
    Tensor* grad_b_proj = nullptr;
    Tensor* grad_c_proj = nullptr;
    Tensor* grad_dt = nullptr;
    Tensor* grad_skip = nullptr;
    Tensor* grad_floquet = nullptr;  // Phase 700

    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, a_log.shape(), &grad_a_log));
    OP_REQUIRES_OK(context, context->allocate_output(
        2, b_proj.shape(), &grad_b_proj));
    OP_REQUIRES_OK(context, context->allocate_output(
        3, c_proj.shape(), &grad_c_proj));
    OP_REQUIRES_OK(context, context->allocate_output(
        4, dt.shape(), &grad_dt));
    OP_REQUIRES_OK(context, context->allocate_output(
        5, skip_proj.shape(), &grad_skip));
    OP_REQUIRES_OK(context, context->allocate_output(
        6, TensorShape({hd_dim}), &grad_floquet));  // Phase 700

    // Call gradient kernel with Floquet phases (Phase 700)
    saguaro::hd_spatial::HDSpatialBackward(
        grad_output.flat<float>().data(),
        hd_input.flat<float>().data(),
        a_log.flat<float>().data(),
        b_proj.flat<float>().data(),
        c_proj.flat<float>().data(),
        dt.flat<float>().data(),
        skip_proj.flat<float>().data(),
        use_floquet_ ? floquet_phases.flat<float>().data() : nullptr,
        grad_input->flat<float>().data(),
        grad_a_log->flat<float>().data(),
        grad_b_proj->flat<float>().data(),
        grad_c_proj->flat<float>().data(),
        grad_dt->flat<float>().data(),
        grad_skip->flat<float>().data(),
        use_floquet_ ? grad_floquet->flat<float>().data() : nullptr,
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::hd_spatial::HDSpatialConfig config_;
  bool use_floquet_ = true;
};

REGISTER_KERNEL_BUILDER(Name("HDSpatialBlockBackward").Device(DEVICE_CPU),
                        HDSpatialBlockBackwardOp);
