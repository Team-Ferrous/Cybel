// saguaro.native/ops/hd_gradient_compression_op.cc
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
 * @file hd_gradient_compression_op.cc
 * @brief TensorFlow ops for HD-native FFT gradient compression.
 *
 * Phase 300+: GaLore integrated into HD architecture.
 *
 * Registers:
 * - HDGradientFFTCompress: Compress gradient via FFT top-K filtering
 * - HDGradientFFTDecompress: Decompress gradient from FFT coefficients
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_gradient_compression_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDGradientFFTCompress")
    .Input("gradient: float32")
    .Attr("bandwidth: int = 256")
    .Attr("preserve_dc: bool = true")
    .Output("compressed_real: float32")
    .Output("compressed_imag: float32")
    .Output("indices: int32")
    .SetShapeFn([](InferenceContext* c) {
        int bandwidth;
        TF_RETURN_IF_ERROR(c->GetAttr("bandwidth", &bandwidth));
        
        ShapeHandle grad_shape = c->input(0);
        if (c->Rank(grad_shape) == 1) {
            c->set_output(0, c->Vector(bandwidth));
            c->set_output(1, c->Vector(bandwidth));
            c->set_output(2, c->Vector(bandwidth));
        } else if (c->Rank(grad_shape) == 2) {
            DimensionHandle batch = c->Dim(grad_shape, 0);
            c->set_output(0, c->Matrix(batch, bandwidth));
            c->set_output(1, c->Matrix(batch, bandwidth));
            c->set_output(2, c->Matrix(batch, bandwidth));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compress gradient via FFT frequency-domain filtering.

Applies FFT, keeps top-K frequency components by magnitude,
discards the rest. This is the HD-native version of GaLore.

gradient: Input gradient [dim] or [batch, dim].
bandwidth: Number of frequency components to keep.
preserve_dc: Always keep DC component.
compressed_real: Real parts of kept frequencies [bandwidth] or [batch, bandwidth].
compressed_imag: Imaginary parts of kept frequencies [bandwidth] or [batch, bandwidth].
indices: Indices of kept frequencies [bandwidth] or [batch, bandwidth].
)doc");

REGISTER_OP("HDGradientFFTDecompress")
    .Input("compressed_real: float32")
    .Input("compressed_imag: float32")
    .Input("indices: int32")
    .Attr("original_dim: int")
    .Attr("scale: float = 1.0")
    .Output("gradient: float32")
    .SetShapeFn([](InferenceContext* c) {
        int original_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("original_dim", &original_dim));
        
        ShapeHandle comp_shape = c->input(0);
        if (c->Rank(comp_shape) == 1) {
            c->set_output(0, c->Vector(original_dim));
        } else if (c->Rank(comp_shape) == 2) {
            DimensionHandle batch = c->Dim(comp_shape, 0);
            c->set_output(0, c->Matrix(batch, original_dim));
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Decompress gradient from FFT frequency representation.

Scatters kept frequencies back to full FFT, applies inverse FFT
to reconstruct the gradient.

compressed_real: Real parts of kept frequencies.
compressed_imag: Imaginary parts of kept frequencies.
indices: Indices of kept frequencies.
original_dim: Original gradient dimension.
scale: Output scaling factor.
gradient: Reconstructed gradient [original_dim] or [batch, original_dim].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDGradientFFTCompressOp : public OpKernel {
public:
    explicit HDGradientFFTCompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bandwidth", &bandwidth_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("preserve_dc", &preserve_dc_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_tensor = ctx->input(0);
        
        saguaro::hd_grad_compress::HDGradCompressConfig config;
        config.bandwidth = bandwidth_;
        config.preserve_dc = preserve_dc_;
        
        const bool is_batched = grad_tensor.dims() == 2;
        
        if (is_batched) {
            const int batch_size = grad_tensor.dim_size(0);
            const int dim = grad_tensor.dim_size(1);
            int padded_dim = 1;
            while (padded_dim < dim) {
                padded_dim <<= 1;
            }
            const int bandwidth = std::min(bandwidth_, padded_dim);
            config.bandwidth = bandwidth;
            
            Tensor* real_tensor = nullptr;
            Tensor* imag_tensor = nullptr;
            Tensor* indices_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({batch_size, bandwidth}), &real_tensor));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1,
                TensorShape({batch_size, bandwidth}), &imag_tensor));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(2,
                TensorShape({batch_size, bandwidth}), &indices_tensor));
            
            const float* grad = grad_tensor.flat<float>().data();
            float* comp_real = real_tensor->flat<float>().data();
            float* comp_imag = imag_tensor->flat<float>().data();
            int* indices = indices_tensor->flat<int>().data();
            
            saguaro::hd_grad_compress::HDGradientBatchCompress(
                grad, comp_real, comp_imag, indices,
                batch_size, dim, config
            );
        } else {
            const int dim = grad_tensor.dim_size(0);
            int padded_dim = 1;
            while (padded_dim < dim) {
                padded_dim <<= 1;
            }
            const int bandwidth = std::min(bandwidth_, padded_dim);
            config.bandwidth = bandwidth;
            
            Tensor* real_tensor = nullptr;
            Tensor* imag_tensor = nullptr;
            Tensor* indices_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({bandwidth}), &real_tensor));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1,
                TensorShape({bandwidth}), &imag_tensor));
            OP_REQUIRES_OK(ctx, ctx->allocate_output(2,
                TensorShape({bandwidth}), &indices_tensor));
            
            const float* grad = grad_tensor.flat<float>().data();
            float* comp_real = real_tensor->flat<float>().data();
            float* comp_imag = imag_tensor->flat<float>().data();
            int* indices = indices_tensor->flat<int>().data();
            
            saguaro::hd_grad_compress::HDGradientCompress(
                grad, comp_real, comp_imag, indices,
                dim, config
            );
        }
    }

private:
    int bandwidth_;
    bool preserve_dc_;
};

class HDGradientFFTDecompressOp : public OpKernel {
public:
    explicit HDGradientFFTDecompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("original_dim", &original_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &scale_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& real_tensor = ctx->input(0);
        const Tensor& imag_tensor = ctx->input(1);
        const Tensor& indices_tensor = ctx->input(2);
        
        saguaro::hd_grad_compress::HDGradCompressConfig config;
        config.scale = scale_;
        
        const bool is_batched = real_tensor.dims() == 2;
        
        if (is_batched) {
            const int batch_size = real_tensor.dim_size(0);
            const int bandwidth = real_tensor.dim_size(1);
            
            Tensor* grad_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({batch_size, original_dim_}), &grad_tensor));
            
            const float* comp_real = real_tensor.flat<float>().data();
            const float* comp_imag = imag_tensor.flat<float>().data();
            const int* indices = indices_tensor.flat<int>().data();
            float* grad = grad_tensor->flat<float>().data();
            
            saguaro::hd_grad_compress::HDGradientBatchDecompress(
                comp_real, comp_imag, indices, grad,
                batch_size, bandwidth, original_dim_, config
            );
        } else {
            const int bandwidth = real_tensor.dim_size(0);
            
            Tensor* grad_tensor = nullptr;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
                TensorShape({original_dim_}), &grad_tensor));
            
            const float* comp_real = real_tensor.flat<float>().data();
            const float* comp_imag = imag_tensor.flat<float>().data();
            const int* indices = indices_tensor.flat<int>().data();
            float* grad = grad_tensor->flat<float>().data();
            
            saguaro::hd_grad_compress::HDGradientDecompress(
                comp_real, comp_imag, indices, grad,
                bandwidth, original_dim_, config
            );
        }
    }

private:
    int original_dim_;
    float scale_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDGradientFFTCompress").Device(DEVICE_CPU),
    HDGradientFFTCompressOp);

REGISTER_KERNEL_BUILDER(
    Name("HDGradientFFTDecompress").Device(DEVICE_CPU),
    HDGradientFFTDecompressOp);

}  // namespace tensorflow
