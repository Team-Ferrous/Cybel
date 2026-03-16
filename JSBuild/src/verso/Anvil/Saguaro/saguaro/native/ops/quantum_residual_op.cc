// saguaro.native/ops/quantum_residual_op.cc
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
 * @file quantum_residual_op.cc
 * @brief TensorFlow custom ops for unitary residual connections.
 *
 * Phase 34 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - UnitaryResidualForward: y = cos(θ)·x + sin(θ)·f(x)
 *   - UnitaryResidualBackward: Custom gradients for Cayley-parameterized angles
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_residual_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("UnitaryResidualForward")
    .Input("x: float")
    .Input("f_x: float")
    .Input("angle: float")
    .Output("output: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Unitary residual connection via rotation blending: y = cos(angle)*x + sin(angle)*f_x.
Provides gradient preservation and norm preservation when inputs are orthogonal.

x: Input tensor of any shape.
f_x: Block output tensor, same shape as x.
angle: Blend angle theta as scalar.
output: Blended output, same shape as x.
)doc");

REGISTER_OP("UnitaryResidualBackward")
    .Input("grad_output: float")
    .Input("x: float")
    .Input("f_x: float")
    .Input("angle: float")
    .Output("grad_x: float")
    .Output("grad_f_x: float")
    .Output("grad_angle: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // grad_x same shape as x
        c->set_output(1, c->input(2));  // grad_f_x same shape as f_x
        c->set_output(2, c->MakeShape({}));  // grad_angle is scalar
        return Status();
    })
    .Doc(R"doc(
Backward pass for unitary residual connection.

Gradient formulas for the rotation blend operation.

grad_output: Gradient w.r.t. output, same shape as x.
x: Original input tensor.
f_x: Original block output tensor.
angle: Blend angle theta as scalar.
grad_x: Gradient w.r.t. x.
grad_f_x: Gradient w.r.t. f_x.
grad_angle: Gradient w.r.t. angle as scalar.
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class UnitaryResidualForwardOp : public OpKernel {
public:
    explicit UnitaryResidualForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& x = context->input(0);
        const Tensor& f_x = context->input(1);
        const Tensor& angle_tensor = context->input(2);

        OP_REQUIRES(context, x.shape() == f_x.shape(),
            errors::InvalidArgument("x and f_x must have the same shape"));
        OP_REQUIRES(context, angle_tensor.dims() == 0,
            errors::InvalidArgument("angle must be a scalar"));

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &output));

        const float* x_data = x.flat<float>().data();
        const float* f_x_data = f_x.flat<float>().data();
        float* output_data = output->flat<float>().data();
        const float angle = angle_tensor.scalar<float>()();
        const int64_t size = x.NumElements();

        saguaro::ops::quantum_residual::UnitaryResidualForward(
            x_data, f_x_data, output_data, angle, size);
    }
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryResidualForward").Device(DEVICE_CPU),
    UnitaryResidualForwardOp);

class UnitaryResidualBackwardOp : public OpKernel {
public:
    explicit UnitaryResidualBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output = context->input(0);
        const Tensor& x = context->input(1);
        const Tensor& f_x = context->input(2);
        const Tensor& angle_tensor = context->input(3);

        OP_REQUIRES(context, grad_output.shape() == x.shape(),
            errors::InvalidArgument("grad_output must match x shape"));
        OP_REQUIRES(context, x.shape() == f_x.shape(),
            errors::InvalidArgument("x and f_x must have the same shape"));
        OP_REQUIRES(context, angle_tensor.dims() == 0,
            errors::InvalidArgument("angle must be a scalar"));

        Tensor* grad_x = nullptr;
        Tensor* grad_f_x = nullptr;
        Tensor* grad_angle = nullptr;

        OP_REQUIRES_OK(context, context->allocate_output(0, x.shape(), &grad_x));
        OP_REQUIRES_OK(context, context->allocate_output(1, f_x.shape(), &grad_f_x));
        OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({}), &grad_angle));

        const float* grad_output_data = grad_output.flat<float>().data();
        const float* x_data = x.flat<float>().data();
        const float* f_x_data = f_x.flat<float>().data();
        float* grad_x_data = grad_x->flat<float>().data();
        float* grad_f_x_data = grad_f_x->flat<float>().data();
        const float angle = angle_tensor.scalar<float>()();
        const int64_t size = x.NumElements();

        float grad_angle_value = saguaro::ops::quantum_residual::UnitaryResidualBackward(
            grad_output_data, x_data, f_x_data,
            grad_x_data, grad_f_x_data, angle, size);

        grad_angle->scalar<float>()() = grad_angle_value;
    }
};

REGISTER_KERNEL_BUILDER(
    Name("UnitaryResidualBackward").Device(DEVICE_CPU),
    UnitaryResidualBackwardOp);
