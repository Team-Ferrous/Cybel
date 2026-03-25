// saguaro.native/ops/quantum_lm_head_hd_op.cc
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
 * @file quantum_lm_head_hd_op.cc
 * @brief TensorFlow op registration for entropy-aware QuantumLMHead.
 *
 * VQC-HD Integration Enhancement #2.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "quantum_lm_head_hd_op.h"

namespace tensorflow {

using namespace saguaro::quantum_lm_head_hd;

// =============================================================================
// Forward Op
// =============================================================================

REGISTER_OP("QuantumLMHeadHD")
    .Input("hidden_states: float32")     // [batch, seq, hidden_dim]
    .Input("vqc_params: float32")        // [layers, qubits, 3]
    .Input("entangle_params: float32")   // [layers, qubits-1]
    .Input("input_proj: float32")        // [hidden_dim, vqc_dim]
    .Input("output_proj: float32")       // [vqc_dim, vocab_size]
    .Attr("vqc_qubits: int = 8")
    .Attr("vqc_layers: int = 2")
    .Attr("entropy_scale: float = 0.1")
    .Attr("entropy_target_qubit: int = 0")
    .Output("logits: float32")           // [batch, seq, vocab_size]
    .Output("entropy: float32")          // [batch, seq]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle hidden_shape = c->input(0);
        shape_inference::ShapeHandle output_proj_shape = c->input(4);
        
        auto batch = c->Dim(hidden_shape, 0);
        auto seq = c->Dim(hidden_shape, 1);
        auto vocab = c->Dim(output_proj_shape, 1);
        
        c->set_output(0, c->MakeShape({batch, seq, vocab}));
        c->set_output(1, c->MakeShape({batch, seq}));
        return Status();
    })
    .Doc(R"doc(
Entropy-aware QuantumLMHead.

Applies VQC with HD spectral entropy injection for improved rare token prediction.
The entropy is computed from the FFT power spectrum of hidden states and injected
as additional phase rotation in the first VQC layer.

hidden_states: Input hidden states.
vqc_params: VQC rotation parameters.
entangle_params: Entangling layer parameters.
input_proj: Projection from hidden_dim to VQC amplitude space.
output_proj: Projection from VQC probabilities to vocab logits.
logits: Output vocabulary logits.
entropy: Computed spectral entropy per position.
)doc");

class QuantumLMHeadHDOp : public OpKernel {
private:
    QuantumLMHeadHDConfig config_;

public:
    explicit QuantumLMHeadHDOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_qubits", &config_.vqc_qubits));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &config_.vqc_layers));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entropy_scale", &config_.entropy_scale));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entropy_target_qubit", &config_.entropy_target_qubit));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& hidden_t = ctx->input(0);
        const Tensor& vqc_params_t = ctx->input(1);
        const Tensor& entangle_t = ctx->input(2);
        const Tensor& input_proj_t = ctx->input(3);
        const Tensor& output_proj_t = ctx->input(4);

        const int batch_size = hidden_t.dim_size(0);
        const int seq_len = hidden_t.dim_size(1);
        const int hidden_dim = hidden_t.dim_size(2);
        const int vocab_size = output_proj_t.dim_size(1);

        // Allocate outputs
        Tensor* logits = nullptr;
        Tensor* entropy = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, seq_len, vocab_size}), &logits));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            1, TensorShape({batch_size, seq_len}), &entropy));

        // Execute forward pass
        QuantumLMHeadHDForward(
            hidden_t.flat<float>().data(),
            vqc_params_t.flat<float>().data(),
            entangle_t.flat<float>().data(),
            input_proj_t.flat<float>().data(),
            output_proj_t.flat<float>().data(),
            logits->flat<float>().data(),
            entropy->flat<float>().data(),
            config_,
            batch_size, seq_len, hidden_dim, vocab_size
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("QuantumLMHeadHD").Device(DEVICE_CPU), QuantumLMHeadHDOp);

// =============================================================================
// Gradient Op
// =============================================================================

REGISTER_OP("QuantumLMHeadHDGrad")
    .Input("grad_logits: float32")       // [batch, seq, vocab_size]
    .Input("hidden_states: float32")     // [batch, seq, hidden_dim]
    .Input("vqc_params: float32")        // [layers, qubits, 3]
    .Input("entangle_params: float32")   // [layers, qubits-1]
    .Input("input_proj: float32")        // [hidden_dim, vqc_dim]
    .Input("output_proj: float32")       // [vqc_dim, vocab_size]
    .Attr("vqc_qubits: int = 8")
    .Attr("vqc_layers: int = 2")
    .Attr("entropy_scale: float = 0.1")
    .Attr("entropy_target_qubit: int = 0")
    .Output("grad_hidden: float32")      // [batch, seq, hidden_dim]
    .Output("grad_vqc_params: float32")  // [layers, qubits, 3]
    .Output("grad_entangle: float32")    // [layers, qubits-1]
    .Output("grad_input_proj: float32")  // [hidden_dim, vqc_dim]
    .Output("grad_output_proj: float32") // [vqc_dim, vocab_size]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // Same as hidden_states
        c->set_output(1, c->input(2));  // Same as vqc_params
        c->set_output(2, c->input(3));  // Same as entangle_params
        c->set_output(3, c->input(4));  // Same as input_proj
        c->set_output(4, c->input(5));  // Same as output_proj
        return Status();
    })
    .Doc("Gradient for QuantumLMHeadHD.");

class QuantumLMHeadHDGradOp : public OpKernel {
private:
    QuantumLMHeadHDConfig config_;

public:
    explicit QuantumLMHeadHDGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_qubits", &config_.vqc_qubits));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_layers", &config_.vqc_layers));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entropy_scale", &config_.entropy_scale));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entropy_target_qubit", &config_.entropy_target_qubit));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_logits_t = ctx->input(0);
        const Tensor& hidden_t = ctx->input(1);
        const Tensor& vqc_params_t = ctx->input(2);
        const Tensor& entangle_t = ctx->input(3);
        const Tensor& input_proj_t = ctx->input(4);
        const Tensor& output_proj_t = ctx->input(5);

        const int batch_size = hidden_t.dim_size(0);
        const int seq_len = hidden_t.dim_size(1);
        const int hidden_dim = hidden_t.dim_size(2);
        const int vocab_size = output_proj_t.dim_size(1);

        // Allocate output gradients
        Tensor* grad_hidden = nullptr;
        Tensor* grad_vqc = nullptr;
        Tensor* grad_entangle = nullptr;
        Tensor* grad_input_proj = nullptr;
        Tensor* grad_output_proj = nullptr;

        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, hidden_t.shape(), &grad_hidden));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, vqc_params_t.shape(), &grad_vqc));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, entangle_t.shape(), &grad_entangle));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(3, input_proj_t.shape(), &grad_input_proj));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(4, output_proj_t.shape(), &grad_output_proj));

        // Execute backward pass
        QuantumLMHeadHDBackward(
            grad_logits_t.flat<float>().data(),
            hidden_t.flat<float>().data(),
            vqc_params_t.flat<float>().data(),
            entangle_t.flat<float>().data(),
            input_proj_t.flat<float>().data(),
            output_proj_t.flat<float>().data(),
            grad_hidden->flat<float>().data(),
            grad_vqc->flat<float>().data(),
            grad_entangle->flat<float>().data(),
            grad_input_proj->flat<float>().data(),
            grad_output_proj->flat<float>().data(),
            config_,
            batch_size, seq_len, hidden_dim, vocab_size
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("QuantumLMHeadHDGrad").Device(DEVICE_CPU), QuantumLMHeadHDGradOp);

}  // namespace tensorflow
