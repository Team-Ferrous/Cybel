// saguaro.native/ops/hd_spectral_entropy_op.cc
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
 * @file hd_spectral_entropy_op.cc
 * @brief TensorFlow ops for HD spectral entropy (QULS integration).
 *
 * Registers:
 * - HDSpectralEntropy: Compute spectral entropy from hidden states
 * - HDSpectralFlatness: Compute spectral flatness metric
 * - HDSpectralEntropyGrad: Gradient op for spectral entropy
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_spectral_entropy_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDSpectralEntropy")
    .Input("hidden_states: float32")
    .Attr("epsilon: float = 1e-8")
    .Output("entropy: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input_shape = c->input(0);
        int rank = c->Rank(input_shape);
        
        if (rank == 2) {
            // [batch, dim] -> [batch]
            c->set_output(0, c->Vector(c->Dim(input_shape, 0)));
        } else if (rank == 3) {
            // [batch, seq, dim] -> [batch]
            c->set_output(0, c->Vector(c->Dim(input_shape, 0)));
        } else {
            c->set_output(0, c->Scalar());
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute HD spectral entropy from hidden states.

Uses FFT-based O(d log d) spectral analysis instead of O(d²) eigenvalue computation.

hidden_states: Input tensor [batch, dim] or [batch, seq, dim].
epsilon: Numerical stability constant.
entropy: Output entropy values [batch].
)doc");

REGISTER_OP("HDSpectralFlatness")
    .Input("hidden_states: float32")
    .Attr("epsilon: float = 1e-8")
    .Output("flatness: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input_shape = c->input(0);
        int rank = c->Rank(input_shape);
        
        if (rank == 2) {
            c->set_output(0, c->Vector(c->Dim(input_shape, 0)));
        } else if (rank == 3) {
            c->set_output(0, c->Vector(c->Dim(input_shape, 0)));
        } else {
            c->set_output(0, c->Scalar());
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute spectral flatness (Wiener entropy) from hidden states.

Spectral flatness = geometric_mean(power) / arithmetic_mean(power).
Ranges from 0 (pure tone) to 1 (white noise).

hidden_states: Input tensor [batch, dim] or [batch, seq, dim].
epsilon: Numerical stability constant.
flatness: Output flatness values [batch].
)doc");

REGISTER_OP("HDSpectralEntropyGrad")
    .Input("hidden_states: float32")
    .Input("grad_output: float32")
    .Attr("epsilon: float = 1e-8")
    .Output("grad_input: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute gradient of HD spectral entropy w.r.t. input.

hidden_states: Input tensor [batch, dim].
grad_output: Gradient from downstream [batch].
epsilon: Numerical stability constant.
grad_input: Gradient w.r.t. hidden_states [batch, dim].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDSpectralEntropyOp : public OpKernel {
public:
    explicit HDSpectralEntropyOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input_tensor = ctx->input(0);
        const int rank = input_tensor.dims();
        
        OP_REQUIRES(ctx, rank == 2 || rank == 3,
            errors::InvalidArgument(
                "HDSpectralEntropy expects 2D or 3D input, got rank ", rank));
        
        saguaro::hd_spectral::HDSpectralConfig config;
        config.epsilon = epsilon_;
        
        if (rank == 2) {
            const int batch_size = input_tensor.dim_size(0);
            const int dim = input_tensor.dim_size(1);
            
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
                TensorShape({batch_size}), &output_tensor));
            
            const float* input = input_tensor.flat<float>().data();
            float* output = output_tensor->flat<float>().data();
            
            saguaro::hd_spectral::HDSpectralEntropyBatch2D(
                input, output, batch_size, dim, config);
        } else {  // rank == 3
            const int batch_size = input_tensor.dim_size(0);
            const int seq_len = input_tensor.dim_size(1);
            const int dim = input_tensor.dim_size(2);
            
            Tensor* output_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({batch_size}), &output_tensor));
            
            const float* input = input_tensor.flat<float>().data();
            float* output = output_tensor->flat<float>().data();
            
            saguaro::hd_spectral::HDSpectralEntropyBatch3D(
                input, output, batch_size, seq_len, dim, config);
        }
    }

private:
    float epsilon_;
};

class HDSpectralFlatnessOp : public OpKernel {
public:
    explicit HDSpectralFlatnessOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input_tensor = ctx->input(0);
        const int rank = input_tensor.dims();
        
        OP_REQUIRES(ctx, rank == 2,
            errors::InvalidArgument(
                "HDSpectralFlatness expects 2D input, got rank ", rank));
        
        const int batch_size = input_tensor.dim_size(0);
        const int dim = input_tensor.dim_size(1);
        
        saguaro::hd_spectral::HDSpectralConfig config;
        config.epsilon = epsilon_;
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size}), &output_tensor));
        
        const float* input = input_tensor.flat<float>().data();
        float* output = output_tensor->flat<float>().data();
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            output[b] = saguaro::hd_spectral::HDSpectralFlatness(
                input + b * dim, dim, config);
        }
    }

private:
    float epsilon_;
};

class HDSpectralEntropyGradOp : public OpKernel {
public:
    explicit HDSpectralEntropyGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("epsilon", &epsilon_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input_tensor = ctx->input(0);
        const Tensor& grad_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, input_tensor.dims() == 2,
            errors::InvalidArgument(
                "HDSpectralEntropyGrad expects 2D input"));
        
        const int batch_size = input_tensor.dim_size(0);
        const int dim = input_tensor.dim_size(1);
        
        saguaro::hd_spectral::HDSpectralConfig config;
        config.epsilon = epsilon_;
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, dim}), &output_tensor));
        
        const float* input = input_tensor.flat<float>().data();
        const float* grad_out = grad_tensor.flat<float>().data();
        float* grad_in = output_tensor->flat<float>().data();
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            saguaro::hd_spectral::HDSpectralEntropyGrad(
                input + b * dim,
                grad_out[b],
                grad_in + b * dim,
                dim,
                config
            );
        }
    }

private:
    float epsilon_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDSpectralEntropy").Device(DEVICE_CPU),
    HDSpectralEntropyOp);

REGISTER_KERNEL_BUILDER(
    Name("HDSpectralFlatness").Device(DEVICE_CPU),
    HDSpectralFlatnessOp);

REGISTER_KERNEL_BUILDER(
    Name("HDSpectralEntropyGrad").Device(DEVICE_CPU),
    HDSpectralEntropyGradOp);

}  // namespace tensorflow
