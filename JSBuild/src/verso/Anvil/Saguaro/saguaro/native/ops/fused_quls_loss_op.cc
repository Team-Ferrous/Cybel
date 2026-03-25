// saguaro.native/ops/fused_quls_loss_op.cc
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
 * @file fused_quls_loss_op.cc
 * @brief TensorFlow custom op registration for Fused QULS Loss.
 *
 * Registers the fused quantum unified loss system operator with TensorFlow,
 * providing both forward and backward passes for end-to-end training.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "fused_quls_loss_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;

// =============================================================================
// FORWARD OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedQulsLossForward")
    .Input("logits: float32")           // [batch, vocab_size]
    .Input("labels: int32")             // [batch]
    .Attr("ce_weight: float = 1.0")
    .Attr("fidelity_weight: float = 0.01")
    .Attr("born_weight: float = 0.005")
    .Attr("entropy_weight: float = 0.01")
    .Attr("spectral_weight: float = 0.01")
    .Attr("label_smoothing: float = 0.1")
    .Attr("target_entropy: float = 0.5")
    .Output("total_loss: float32")      // scalar
    .Output("ce_loss: float32")         // scalar
    .Output("fidelity_loss: float32")   // scalar
    .Output("entropy_loss: float32")    // scalar
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->Scalar());
        c->set_output(1, c->Scalar());
        c->set_output(2, c->Scalar());
        c->set_output(3, c->Scalar());
        return OkStatus();
    })
    .Doc(R"doc(
Fused Quantum Unified Loss System forward pass.

Computes all QULS loss components in a single fused kernel:
  - Cross-entropy with label smoothing
  - Quantum fidelity loss
  - Entropy regularization

logits: Input logits tensor [batch, vocab_size].
labels: Target label indices [batch].
ce_weight: Weight for cross-entropy loss.
fidelity_weight: Weight for quantum fidelity loss.
born_weight: Weight for Born rule regularization (unused in this op).
entropy_weight: Weight for entropy regularization.
spectral_weight: Weight for spectral flatness (unused in this op).
label_smoothing: Label smoothing factor epsilon.
target_entropy: Target entropy for regularization.
total_loss: Weighted sum of all loss components.
ce_loss: Cross-entropy loss value.
fidelity_loss: Quantum fidelity loss value.
entropy_loss: Entropy regularization loss value.
)doc");

// =============================================================================
// FORWARD OP KERNEL
// =============================================================================

class FusedQulsLossForwardOp : public OpKernel {
 public:
    explicit FusedQulsLossForwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ce_weight", &config_.ce_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("fidelity_weight", &config_.fidelity_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("born_weight", &config_.born_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entropy_weight", &config_.entropy_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("spectral_weight", &config_.spectral_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("label_smoothing", &config_.label_smoothing));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("target_entropy", &config_.target_entropy));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& logits_tensor = ctx->input(0);
        const Tensor& labels_tensor = ctx->input(1);
        
        // Validate shapes
        OP_REQUIRES(ctx, logits_tensor.dims() == 2,
            errors::InvalidArgument("logits must be 2D [batch, vocab_size]"));
        OP_REQUIRES(ctx, labels_tensor.dims() == 1,
            errors::InvalidArgument("labels must be 1D [batch]"));
        
        const int64_t batch_size = logits_tensor.dim_size(0);
        const int64_t vocab_size = logits_tensor.dim_size(1);
        
        OP_REQUIRES(ctx, labels_tensor.dim_size(0) == batch_size,
            errors::InvalidArgument("labels batch size must match logits"));
        
        // Get pointers
        const float* logits = logits_tensor.flat<float>().data();
        const int32_t* labels = labels_tensor.flat<int32_t>().data();
        
        // Configure
        saguaro::ops::quls::QULSLossConfig config = config_;
        config.vocab_size = vocab_size;
        
        // Compute loss
        saguaro::ops::quls::QULSLossOutput output;
        saguaro::ops::quls::quls_loss_forward(
            logits,
            labels,
            nullptr,  // amplitudes
            nullptr,  // coherence_scores
            nullptr,  // h_init
            nullptr,  // h_final
            nullptr,  // eigenvalues
            output,
            config,
            batch_size
        );
        
        // Allocate outputs
        Tensor* total_loss_tensor = nullptr;
        Tensor* ce_loss_tensor = nullptr;
        Tensor* fidelity_loss_tensor = nullptr;
        Tensor* entropy_loss_tensor = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &total_loss_tensor));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({}), &ce_loss_tensor));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({}), &fidelity_loss_tensor));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({}), &entropy_loss_tensor));
        
        total_loss_tensor->scalar<float>()() = output.total_loss;
        ce_loss_tensor->scalar<float>()() = output.ce_loss;
        fidelity_loss_tensor->scalar<float>()() = output.fidelity_loss;
        entropy_loss_tensor->scalar<float>()() = output.entropy_loss;
    }

 private:
    saguaro::ops::quls::QULSLossConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("FusedQulsLossForward").Device(DEVICE_CPU),
                        FusedQulsLossForwardOp);

// =============================================================================
// BACKWARD OP REGISTRATION
// =============================================================================

REGISTER_OP("FusedQulsLossBackward")
    .Input("logits: float32")           // [batch, vocab_size]
    .Input("labels: int32")             // [batch]
    .Attr("ce_weight: float = 1.0")
    .Attr("fidelity_weight: float = 0.01")
    .Attr("label_smoothing: float = 0.1")
    .Output("grad_logits: float32")     // [batch, vocab_size]
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));  // Same shape as logits
        return OkStatus();
    })
    .Doc(R"doc(
Fused Quantum Unified Loss System backward pass.

Computes gradients w.r.t. logits for CE and fidelity losses in a single pass.

logits: Input logits tensor [batch, vocab_size].
labels: Target label indices [batch].
ce_weight: Weight for cross-entropy loss.
fidelity_weight: Weight for quantum fidelity loss.
label_smoothing: Label smoothing factor epsilon.
grad_logits: Gradient w.r.t. logits [batch, vocab_size].
)doc");

// =============================================================================
// BACKWARD OP KERNEL
// =============================================================================

class FusedQulsLossBackwardOp : public OpKernel {
 public:
    explicit FusedQulsLossBackwardOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ce_weight", &config_.ce_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("fidelity_weight", &config_.fidelity_weight));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("label_smoothing", &config_.label_smoothing));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& logits_tensor = ctx->input(0);
        const Tensor& labels_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, logits_tensor.dims() == 2,
            errors::InvalidArgument("logits must be 2D"));
        OP_REQUIRES(ctx, labels_tensor.dims() == 1,
            errors::InvalidArgument("labels must be 1D"));
        
        const int64_t batch_size = logits_tensor.dim_size(0);
        const int64_t vocab_size = logits_tensor.dim_size(1);
        
        const float* logits = logits_tensor.flat<float>().data();
        const int32_t* labels = labels_tensor.flat<int32_t>().data();
        
        // Allocate output
        Tensor* grad_logits_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, logits_tensor.shape(), &grad_logits_tensor));
        float* grad_logits = grad_logits_tensor->flat<float>().data();
        
        // Configure
        saguaro::ops::quls::QULSLossConfig config = config_;
        config.vocab_size = vocab_size;
        
        // Compute gradients
        saguaro::ops::quls::quls_loss_backward(
            grad_logits,
            logits,
            labels,
            config,
            batch_size
        );
    }

 private:
    saguaro::ops::quls::QULSLossConfig config_;
};

REGISTER_KERNEL_BUILDER(Name("FusedQulsLossBackward").Device(DEVICE_CPU),
                        FusedQulsLossBackwardOp);

}  // namespace tensorflow
