// saguaro.native/ops/holographic_loss_op.cc
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
 * @file holographic_loss_op.cc
 * @brief Phase 200+: Holographic Cross-Entropy Loss TensorFlow operations.
 *
 * SAGUARO_UPGRADE_ROADMAP.md Phase 3.2 - QULS Native Ops.
 *
 * Registers TensorFlow ops for HD-space cross-entropy computation.
 * Enables memory-efficient loss calculation without materializing full logits.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

#include "holographic_loss_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION: HolographicCrossEntropy
// =============================================================================

REGISTER_OP("HolographicCrossEntropy")
    .Input("hd_states: float")          // [batch, seq_len, hd_dim]
    .Input("token_bases: float")        // [vocab_size, hd_dim]
    .Input("targets: int32")            // [batch, seq_len]
    .Input("negative_samples: int32")   // [batch, seq_len, num_negatives]
    .Output("losses: float")            // [batch, seq_len]
    .Output("total_loss: float")        // scalar
    .Attr("hd_dim: int = 4096")
    .Attr("vocab_size: int = 50000")
    .Attr("label_smoothing: float = 0.1")
    .Attr("temperature: float = 1.0")
    .Attr("num_negatives: int = 64")
    .Attr("use_nce: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);

        if (c->RankKnown(input_shape) && c->Rank(input_shape) == 3) {
            auto batch = c->Dim(input_shape, 0);
            auto seq_len = c->Dim(input_shape, 1);

            c->set_output(0, c->MakeShape({batch, seq_len}));
            c->set_output(1, c->Scalar());
        } else {
            c->set_output(0, c->UnknownShape());
            c->set_output(1, c->Scalar());
        }
        return Status();
    })
    .Doc(R"doc(
Holographic Cross-Entropy Loss - Compute CE directly in HD space.

Phase 200+: QULS Native Ops. Computes cross-entropy without materializing
the full [batch, seq, vocab] logits tensor. Uses NCE for large vocabularies
to achieve O(D + k) complexity instead of O(D + V).

hd_states: HD state vectors from model [batch, seq_len, hd_dim]
token_bases: Token HD base vectors (embeddings) [vocab_size, hd_dim]
targets: Target token indices [batch, seq_len]
negative_samples: Pre-sampled negative indices [batch, seq_len, num_negatives]

losses: Per-token loss values [batch, seq_len]
total_loss: Mean loss (scalar)
)doc");

// =============================================================================
// KERNEL: HolographicCrossEntropy
// =============================================================================

class HolographicCrossEntropyOp : public OpKernel {
 public:
  explicit HolographicCrossEntropyOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &config_.vocab_size));
    OP_REQUIRES_OK(context, context->GetAttr("label_smoothing", &config_.label_smoothing));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &config_.temperature));
    OP_REQUIRES_OK(context, context->GetAttr("num_negatives", &config_.num_negatives));
    OP_REQUIRES_OK(context, context->GetAttr("use_nce", &config_.use_nce));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_states = context->input(0);
    const Tensor& token_bases = context->input(1);
    const Tensor& targets = context->input(2);
    const Tensor& negative_samples = context->input(3);

    // Validate shapes
    OP_REQUIRES(context, hd_states.dims() == 3,
                errors::InvalidArgument("hd_states must be 3D [batch, seq, hd_dim]"));

    const int batch_size = hd_states.dim_size(0);
    const int seq_len = hd_states.dim_size(1);
    const int hd_dim = hd_states.dim_size(2);

    OP_REQUIRES(context, hd_dim == config_.hd_dim,
                errors::InvalidArgument("hd_dim mismatch"));

    // Allocate outputs
    Tensor* losses = nullptr;
    Tensor* total_loss = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(
        0, TensorShape({batch_size, seq_len}), &losses));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, TensorShape({}), &total_loss));

    // Get negative samples pointer (may be null if not using NCE)
    const int* neg_samples_ptr = nullptr;
    if (negative_samples.NumElements() > 0) {
        neg_samples_ptr = negative_samples.flat<int>().data();
    }

    // Call kernel
    saguaro::holographic_loss::HolographicLossForward(
        hd_states.flat<float>().data(),
        token_bases.flat<float>().data(),
        targets.flat<int>().data(),
        neg_samples_ptr,
        losses->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );

    // Compute mean loss
    float sum_loss = 0.0f;
    const float* loss_data = losses->flat<float>().data();
    for (int i = 0; i < batch_size * seq_len; ++i) {
        sum_loss += loss_data[i];
    }
    total_loss->scalar<float>()() = sum_loss / static_cast<float>(batch_size * seq_len);
  }

 private:
  saguaro::holographic_loss::HolographicLossConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HolographicCrossEntropy").Device(DEVICE_CPU),
                        HolographicCrossEntropyOp);

// =============================================================================
// OP REGISTRATION: HolographicCrossEntropyGrad
// =============================================================================

REGISTER_OP("HolographicCrossEntropyGrad")
    .Input("hd_states: float")          // [batch, seq_len, hd_dim]
    .Input("token_bases: float")        // [vocab_size, hd_dim]
    .Input("targets: int32")            // [batch, seq_len]
    .Input("negative_samples: int32")   // [batch, seq_len, num_negatives]
    .Output("grad_states: float")       // [batch, seq_len, hd_dim]
    .Output("grad_bases: float")        // [vocab_size, hd_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("vocab_size: int = 50000")
    .Attr("label_smoothing: float = 0.1")
    .Attr("temperature: float = 1.0")
    .Attr("num_negatives: int = 64")
    .Attr("use_nce: bool = true")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // grad_states same as hd_states
        c->set_output(1, c->input(1));  // grad_bases same as token_bases
        return Status();
    })
    .Doc("Holographic Cross-Entropy Gradient - Compute backprop gradients.");

// =============================================================================
// KERNEL: HolographicCrossEntropyGrad
// =============================================================================

class HolographicCrossEntropyGradOp : public OpKernel {
 public:
  explicit HolographicCrossEntropyGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("hd_dim", &config_.hd_dim));
    OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &config_.vocab_size));
    OP_REQUIRES_OK(context, context->GetAttr("label_smoothing", &config_.label_smoothing));
    OP_REQUIRES_OK(context, context->GetAttr("temperature", &config_.temperature));
    OP_REQUIRES_OK(context, context->GetAttr("num_negatives", &config_.num_negatives));
    OP_REQUIRES_OK(context, context->GetAttr("use_nce", &config_.use_nce));
  }

  void Compute(OpKernelContext* context) override {
    // Get input tensors
    const Tensor& hd_states = context->input(0);
    const Tensor& token_bases = context->input(1);
    const Tensor& targets = context->input(2);
    const Tensor& negative_samples = context->input(3);

    const int batch_size = hd_states.dim_size(0);
    const int seq_len = hd_states.dim_size(1);

    // Allocate gradient tensors
    Tensor* grad_states = nullptr;
    Tensor* grad_bases = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(
        0, hd_states.shape(), &grad_states));
    OP_REQUIRES_OK(context, context->allocate_output(
        1, token_bases.shape(), &grad_bases));

    // Get negative samples pointer
    const int* neg_samples_ptr = nullptr;
    if (negative_samples.NumElements() > 0) {
        neg_samples_ptr = negative_samples.flat<int>().data();
    }

    // Call gradient kernel
    saguaro::holographic_loss::HolographicLossBackward(
        hd_states.flat<float>().data(),
        token_bases.flat<float>().data(),
        targets.flat<int>().data(),
        neg_samples_ptr,
        grad_states->flat<float>().data(),
        grad_bases->flat<float>().data(),
        config_,
        batch_size,
        seq_len
    );
  }

 private:
  saguaro::holographic_loss::HolographicLossConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("HolographicCrossEntropyGrad").Device(DEVICE_CPU),
                        HolographicCrossEntropyGradOp);
