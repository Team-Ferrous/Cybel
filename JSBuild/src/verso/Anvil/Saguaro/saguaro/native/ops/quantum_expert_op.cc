// saguaro.native/ops/quantum_expert_op.cc
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
 * @file quantum_expert_op.cc
 * @brief TensorFlow custom ops for unitary expert networks.
 *
 * Phase 29 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - UnitaryExpertForward: x → U₁·x → σ_quantum → U₂·x
 *   - UnitaryExpertBackward: Custom gradients for Cayley-parameterized matrices
 *   - QuantumActivation: Parametric rotation activation
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_expert_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("UnitaryExpertForward")
    .Input("input: float")
    .Input("u1_weights: float")
    .Input("u2_weights: float")
    .Input("activation_angle: float")
    .Output("output: float")
    .Output("hidden_cache: float")
    .Attr("d_ff: int")
    .Attr("neumann_terms: int = 6")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));  // output same shape as input
        
        // hidden_cache: [batch * seq_len, d_ff]
        int d_ff;
        TF_RETURN_IF_ERROR(c->GetAttr("d_ff", &d_ff));
        
        shape_inference::ShapeHandle input_shape = c->input(0);
        if (c->Rank(input_shape) >= 2) {
            shape_inference::DimensionHandle num_tokens = c->Dim(input_shape, 0);
            c->set_output(1, c->MakeShape({num_tokens, d_ff}));
        }
        return Status();
    })
    .Doc(R"doc(
Unitary expert network forward pass with architecture x -> U1*x -> quantum_activation -> U2*x.
Provides gradient and information preservation via unitary transforms.

input: Input tensor with shape [num_tokens, d_model].
u1_weights: First projection weights with shape [d_ff, d_model].
u2_weights: Second projection weights with shape [d_model, d_ff].
activation_angle: Quantum activation angle as scalar.
d_ff: Intermediate feedforward dimension.
neumann_terms: Number of Neumann series terms for Cayley transform.
output: Output tensor with shape [num_tokens, d_model].
hidden_cache: Cached hidden activations for backward with shape [num_tokens, d_ff].
)doc");

REGISTER_OP("UnitaryExpertBackward")
    .Input("grad_output: float")
    .Input("input: float")
    .Input("u1_weights: float")
    .Input("u2_weights: float")
    .Input("activation_angle: float")
    .Input("hidden_cache: float")
    .Output("grad_input: float")
    .Output("grad_u1: float")
    .Output("grad_u2: float")
    .Output("grad_angle: float")
    .Attr("d_ff: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_input same as input
        c->set_output(1, c->input(2));  // grad_u1 same as u1_weights
        c->set_output(2, c->input(3));  // grad_u2 same as u2_weights
        c->set_output(3, c->MakeShape({}));  // grad_angle is scalar
        return Status();
    })
    .Doc(R"doc(
Backward pass for unitary expert network.
Computes gradients for all parameters including activation angle.

grad_output: Gradient w.r.t. output.
input: Original input.
u1_weights: First projection weights.
u2_weights: Second projection weights.
activation_angle: Activation angle.
hidden_cache: Cached hidden activations from forward.
grad_input: Gradient w.r.t. input.
grad_u1: Gradient w.r.t. U1 weights.
grad_u2: Gradient w.r.t. U2 weights.
grad_angle: Gradient w.r.t. activation angle.
)doc");

REGISTER_OP("QuantumActivation")
    .Input("input: float")
    .Input("angle: float")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Quantum activation applies parametric rotation to pairs of dimensions.

input: Input tensor where last dimension must be even.
angle: Rotation angle as scalar.
output: Rotated output, same shape as input.
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class UnitaryExpertForwardOp : public OpKernel {
public:
    explicit UnitaryExpertForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("d_ff", &d_ff_));
        OP_REQUIRES_OK(context, context->GetAttr("neumann_terms", &neumann_terms_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& u1_weights = context->input(1);
        const Tensor& u2_weights = context->input(2);
        const Tensor& activation_angle = context->input(3);
        
        const TensorShape& input_shape = input.shape();
        OP_REQUIRES(context, input_shape.dims() == 2,
            errors::InvalidArgument("Input must be 2D [num_tokens, d_model]"));
        
        const int num_tokens = input_shape.dim_size(0);
        const int d_model = input_shape.dim_size(1);
        
        // Allocate outputs
        Tensor* output = nullptr;
        Tensor* hidden_cache = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
        OP_REQUIRES_OK(context, context->allocate_output(1, 
            TensorShape({num_tokens, d_ff_}), &hidden_cache));
        
        const float* input_data = input.flat<float>().data();
        const float* u1_data = u1_weights.flat<float>().data();
        const float* u2_data = u2_weights.flat<float>().data();
        const float angle = activation_angle.scalar<float>()();
        float* output_data = output->flat<float>().data();
        float* hidden_data = hidden_cache->flat<float>().data();
        
        // Forward pass with hidden caching
        std::vector<float> U1(d_ff_ * d_model);
        std::vector<float> U2(d_model * d_ff_);
        std::copy(u1_data, u1_data + d_ff_ * d_model, U1.begin());
        std::copy(u2_data, u2_data + d_model * d_ff_, U2.begin());
        
        #pragma omp parallel for
        for (int t = 0; t < num_tokens; ++t) {
            const float* x = input_data + t * d_model;
            float* y = output_data + t * d_model;
            float* h = hidden_data + t * d_ff_;
            
            // Step 1: Project through U₁
            for (int i = 0; i < d_ff_; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    sum += U1[i * d_model + j] * x[j];
                }
                h[i] = sum;
            }
            
            // Step 2: Quantum activation
            saguaro::ops::quantum_expert::QuantumActivation(h, angle, d_ff_);
            
            // Step 3: Project back through U₂
            for (int i = 0; i < d_model; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < d_ff_; ++j) {
                    sum += U2[i * d_ff_ + j] * h[j];
                }
                y[i] = sum;
            }
        }
    }

private:
    int d_ff_;
    int neumann_terms_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryExpertForward").Device(DEVICE_CPU),
    UnitaryExpertForwardOp);

class UnitaryExpertBackwardOp : public OpKernel {
public:
    explicit UnitaryExpertBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("d_ff", &d_ff_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output = context->input(0);
        const Tensor& input = context->input(1);
        const Tensor& u1_weights = context->input(2);
        const Tensor& u2_weights = context->input(3);
        const Tensor& activation_angle = context->input(4);
        const Tensor& hidden_cache = context->input(5);
        
        const TensorShape& input_shape = input.shape();
        const int num_tokens = input_shape.dim_size(0);
        const int d_model = input_shape.dim_size(1);
        
        // Allocate outputs
        Tensor* grad_input = nullptr;
        Tensor* grad_u1 = nullptr;
        Tensor* grad_u2 = nullptr;
        Tensor* grad_angle = nullptr;
        
        OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
        OP_REQUIRES_OK(context, context->allocate_output(1, u1_weights.shape(), &grad_u1));
        OP_REQUIRES_OK(context, context->allocate_output(2, u2_weights.shape(), &grad_u2));
        OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape({}), &grad_angle));
        
        const float* grad_out_data = grad_output.flat<float>().data();
        const float* input_data = input.flat<float>().data();
        const float* u1_data = u1_weights.flat<float>().data();
        const float* u2_data = u2_weights.flat<float>().data();
        const float angle = activation_angle.scalar<float>()();
        const float* hidden_data = hidden_cache.flat<float>().data();
        
        float* grad_input_data = grad_input->flat<float>().data();
        float* grad_u1_data = grad_u1->flat<float>().data();
        float* grad_u2_data = grad_u2->flat<float>().data();
        
        float total_grad_angle = saguaro::ops::quantum_expert::UnitaryExpertBackward(
            grad_out_data, input_data, u1_data, u2_data, angle, hidden_data,
            grad_input_data, grad_u1_data, grad_u2_data,
            num_tokens, d_model, d_ff_);
        
        grad_angle->scalar<float>()() = total_grad_angle;
    }

private:
    int d_ff_;
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryExpertBackward").Device(DEVICE_CPU),
    UnitaryExpertBackwardOp);

class QuantumActivationOp : public OpKernel {
public:
    explicit QuantumActivationOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const Tensor& angle_tensor = context->input(1);
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &output));
        
        // Copy input to output, then apply in-place
        auto input_flat = input.flat<float>();
        auto output_flat = output->flat<float>();
        
        std::copy(input_flat.data(), input_flat.data() + input_flat.size(),
                  output_flat.data());
        
        const float angle = angle_tensor.scalar<float>()();
        const int64_t size = input_flat.size();
        
        saguaro::ops::quantum_expert::QuantumActivation(
            output_flat.data(), angle, static_cast<int>(size));
    }
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumActivation").Device(DEVICE_CPU),
    QuantumActivationOp);
