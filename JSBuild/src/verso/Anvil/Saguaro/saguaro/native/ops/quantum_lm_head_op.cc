// saguaro.native/ops/quantum_lm_head_op.cc
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
 * @file quantum_lm_head_op.cc
 * @brief TensorFlow custom ops for quantum LM head.
 *
 * Phase 33 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - QuantumLMHeadForward: VQC-based output projection
 *   - QuantumLMHeadBackward: Parameter-shift gradient
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_lm_head_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("QuantumLMHeadForward")
    .Input("hidden_states: float")
    .Input("rotation_params: float")
    .Input("token_weights: float")
    .Output("logits: float")
    .Attr("vocab_size: int")
    .Attr("num_layers: int = 2")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int vocab_size;
        TF_RETURN_IF_ERROR(c->GetAttr("vocab_size", &vocab_size));
        
        shape_inference::ShapeHandle input_shape = c->input(0);
        int rank = c->Rank(input_shape);
        
        if (rank == 2) {
            // [seq_len, d_model] -> [seq_len, vocab_size]
            c->set_output(0, c->MakeShape({
                c->Dim(input_shape, 0),
                vocab_size
            }));
        } else if (rank == 3) {
            // [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
            c->set_output(0, c->MakeShape({
                c->Dim(input_shape, 0),
                c->Dim(input_shape, 1),
                vocab_size
            }));
        }
        return Status();
    })
    .Doc(R"doc(
Quantum LM head: VQC-based output projection.

Uses variational quantum circuit simulation for expressive output
transformation with Born rule probability extraction.

hidden_states: Input hidden states [batch, seq_len, d_model]
rotation_params: VQC rotation parameters [num_layers, d_model/2, 2]
token_weights: Token-qubit contribution weights [vocab_size, d_model/2]
vocab_size: Vocabulary size
num_layers: VQC circuit depth
logits: Output logits [batch, seq_len, vocab_size]
)doc");

REGISTER_OP("QuantumLMHeadBackward")
    .Input("grad_logits: float")
    .Input("hidden_states: float")
    .Input("rotation_params: float")
    .Input("token_weights: float")
    .Output("grad_rotation: float")
    .Output("grad_token_weights: float")
    .Attr("vocab_size: int")
    .Attr("num_layers: int = 2")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));  // grad_rotation same as rotation_params
        c->set_output(1, c->input(3));  // grad_token_weights same as token_weights
        return Status();
    })
    .Doc(R"doc(
Backward pass for quantum LM head.

Uses parameter-shift rule approximation for gradient computation.

grad_logits: Gradient w.r.t. logits
hidden_states: Original hidden states
rotation_params: VQC parameters
token_weights: Token weights
grad_rotation: Gradient w.r.t. rotation params
grad_token_weights: Gradient w.r.t. token weights
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class QuantumLMHeadForwardOp : public OpKernel {
public:
    explicit QuantumLMHeadForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
        OP_REQUIRES_OK(context, context->GetAttr("num_layers", &num_layers_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& hidden_states = context->input(0);
        const Tensor& rotation_params = context->input(1);
        const Tensor& token_weights = context->input(2);
        
        const TensorShape& hidden_shape = hidden_states.shape();
        
        int batch_size = 1;
        int seq_len;
        int d_model;
        
        if (hidden_shape.dims() == 2) {
            seq_len = hidden_shape.dim_size(0);
            d_model = hidden_shape.dim_size(1);
        } else {
            batch_size = hidden_shape.dim_size(0);
            seq_len = hidden_shape.dim_size(1);
            d_model = hidden_shape.dim_size(2);
        }
        
        TensorShape output_shape;
        if (hidden_shape.dims() == 2) {
            output_shape = TensorShape({seq_len, vocab_size_});
        } else {
            output_shape = TensorShape({batch_size, seq_len, vocab_size_});
        }
        
        Tensor* logits = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &logits));
        
        const float* hidden_data = hidden_states.flat<float>().data();
        const float* rotation_data = rotation_params.flat<float>().data();
        const float* weights_data = token_weights.flat<float>().data();
        float* logits_data = logits->flat<float>().data();
        
        saguaro::ops::quantum_lm_head::QuantumLMHeadForward(
            hidden_data, rotation_data, weights_data, logits_data,
            batch_size, seq_len, d_model, vocab_size_, num_layers_);
    }

private:
    int vocab_size_;
    int num_layers_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumLMHeadForward").Device(DEVICE_CPU),
    QuantumLMHeadForwardOp);

class QuantumLMHeadBackwardOp : public OpKernel {
public:
    explicit QuantumLMHeadBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
        OP_REQUIRES_OK(context, context->GetAttr("num_layers", &num_layers_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_logits = context->input(0);
        const Tensor& hidden_states = context->input(1);
        const Tensor& rotation_params = context->input(2);
        const Tensor& token_weights = context->input(3);
        
        const TensorShape& hidden_shape = hidden_states.shape();
        
        int batch_size = 1;
        int seq_len;
        int d_model;
        
        if (hidden_shape.dims() == 2) {
            seq_len = hidden_shape.dim_size(0);
            d_model = hidden_shape.dim_size(1);
        } else {
            batch_size = hidden_shape.dim_size(0);
            seq_len = hidden_shape.dim_size(1);
            d_model = hidden_shape.dim_size(2);
        }
        
        Tensor* grad_rotation = nullptr;
        Tensor* grad_token_weights = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, rotation_params.shape(), &grad_rotation));
        OP_REQUIRES_OK(context, context->allocate_output(1, token_weights.shape(), &grad_token_weights));
        
        const float* grad_logits_data = grad_logits.flat<float>().data();
        const float* hidden_data = hidden_states.flat<float>().data();
        const float* rotation_data = rotation_params.flat<float>().data();
        const float* weights_data = token_weights.flat<float>().data();
        float* grad_rot_data = grad_rotation->flat<float>().data();
        float* grad_weights_data = grad_token_weights->flat<float>().data();
        
        saguaro::ops::quantum_lm_head::QuantumLMHeadBackward(
            grad_logits_data, hidden_data, rotation_data, weights_data,
            grad_rot_data, grad_weights_data,
            batch_size, seq_len, d_model, vocab_size_, num_layers_);
    }

private:
    int vocab_size_;
    int num_layers_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumLMHeadBackward").Device(DEVICE_CPU),
    QuantumLMHeadBackwardOp);
