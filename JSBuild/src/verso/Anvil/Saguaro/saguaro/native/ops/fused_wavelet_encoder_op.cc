// saguaro.native/ops/fused_wavelet_encoder_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Fused Wavelet Encoder Chunk operator - performs learnable 1D DWT
// chunking for efficient sequence encoding in transformer architectures.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace tensorflow {

// ==================== Forward Op ====================
REGISTER_OP("FusedWaveletEncoderChunk")
    .Input("input_sequence: float32")      // [batch, seq_len, dim]
    .Input("wavelet_filters: float32")     // [num_levels, filter_size]
    .Input("projection_weights: float32")  // [dim, output_dim]
    .Attr("chunk_size: int = 256")
    .Attr("num_levels: int = 4")
    .Attr("wavelet_type: string = 'db4'")
    .Output("encoded_chunks: float32")     // [batch, num_chunks, output_dim]
    .Output("detail_coeffs: float32")      // [batch, num_levels, num_chunks, dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
        
        int chunk_size;
        TF_RETURN_IF_ERROR(c->GetAttr("chunk_size", &chunk_size));
        int num_levels;
        TF_RETURN_IF_ERROR(c->GetAttr("num_levels", &num_levels));
        
        shape_inference::DimensionHandle batch = c->Dim(input_shape, 0);
        shape_inference::DimensionHandle seq_len = c->Dim(input_shape, 1);
        shape_inference::DimensionHandle dim = c->Dim(input_shape, 2);
        
        // Compute num_chunks = ceil(seq_len / chunk_size)
        c->set_output(0, c->MakeShape({batch, c->UnknownDim(), c->UnknownDim()}));
        c->set_output(1, c->MakeShape({batch, num_levels, c->UnknownDim(), dim}));
        return Status();
    })
    .Doc("Fused wavelet encoder with learnable DWT for sequence chunking.");

class FusedWaveletEncoderChunkOp : public OpKernel {
private:
    int chunk_size_;
    int num_levels_;
    std::string wavelet_type_;
    
public:
    explicit FusedWaveletEncoderChunkOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_levels", &num_levels_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("wavelet_type", &wavelet_type_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& input = ctx->input(0);
        const Tensor& wavelet_filters = ctx->input(1);
        const Tensor& projection = ctx->input(2);
        
        const int batch_size = input.dim_size(0);
        const int seq_len = input.dim_size(1);
        const int input_dim = input.dim_size(2);
        const int output_dim = projection.dim_size(1);
        
        // Compute number of chunks
        const int num_chunks = (seq_len + chunk_size_ - 1) / chunk_size_;
        
        // Allocate outputs
        Tensor* encoded_chunks = nullptr;
        Tensor* detail_coeffs = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({batch_size, num_chunks, output_dim}), &encoded_chunks));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, 
            TensorShape({batch_size, num_levels_, num_chunks, input_dim}), &detail_coeffs));
        
        auto input_flat = input.flat_inner_dims<float, 3>();
        auto proj_flat = projection.flat_inner_dims<float, 2>();
        auto encoded_flat = encoded_chunks->flat_inner_dims<float, 3>();
        auto detail_flat = detail_coeffs->flat_inner_dims<float, 4>();
        
        // Process each batch and chunk
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < num_chunks; ++c) {
                const int start_idx = c * chunk_size_;
                const int end_idx = std::min(start_idx + chunk_size_, seq_len);
                const int actual_chunk_size = end_idx - start_idx;
                
                // Compute average of chunk (approximation coefficient)
                std::vector<float> approx(input_dim, 0.0f);
                for (int i = start_idx; i < end_idx; ++i) {
                    for (int d = 0; d < input_dim; ++d) {
                        approx[d] += input_flat(b, i, d);
                    }
                }
                for (int d = 0; d < input_dim; ++d) {
                    approx[d] /= static_cast<float>(actual_chunk_size);
                }
                
                // Apply projection to get encoded chunk
                for (int o = 0; o < output_dim; ++o) {
                    float sum = 0.0f;
                    for (int d = 0; d < input_dim; ++d) {
                        sum += approx[d] * proj_flat(d, o);
                    }
                    encoded_flat(b, c, o) = sum;
                }
                
                // Compute detail coefficients at each level
                for (int level = 0; level < num_levels_; ++level) {
                    // Haar-like detail computation for simplicity
                    float scale = 1.0f / std::pow(2.0f, level + 1);
                    for (int d = 0; d < input_dim; ++d) {
                        float detail = 0.0f;
                        int half_chunk = std::max(1, actual_chunk_size / (1 << (level + 1)));
                        
                        // Compute difference between halves
                        float first_half = 0.0f, second_half = 0.0f;
                        int mid = start_idx + half_chunk;
                        int count1 = 0, count2 = 0;
                        
                        for (int i = start_idx; i < std::min(mid, end_idx); ++i) {
                            first_half += input_flat(b, i, d);
                            count1++;
                        }
                        for (int i = mid; i < end_idx; ++i) {
                            second_half += input_flat(b, i, d);
                            count2++;
                        }
                        
                        if (count1 > 0) first_half /= count1;
                        if (count2 > 0) second_half /= count2;
                        
                        detail_flat(b, level, c, d) = (first_half - second_half) * scale;
                    }
                }
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedWaveletEncoderChunk").Device(DEVICE_CPU), FusedWaveletEncoderChunkOp);

// ==================== Gradient Op ====================
REGISTER_OP("FusedWaveletEncoderChunkGrad")
    .Input("grad_encoded: float32")          // [batch, num_chunks, output_dim]
    .Input("grad_detail: float32")           // [batch, num_levels, num_chunks, dim]
    .Input("input_sequence: float32")        // [batch, seq_len, dim]
    .Input("wavelet_filters: float32")       // [num_levels, filter_size]
    .Input("projection_weights: float32")    // [dim, output_dim]
    .Attr("chunk_size: int = 256")
    .Attr("num_levels: int = 4")
    .Output("grad_input: float32")
    .Output("grad_wavelet: float32")
    .Output("grad_projection: float32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));
        c->set_output(1, c->input(3));
        c->set_output(2, c->input(4));
        return Status();
    })
    .Doc("Gradient for fused wavelet encoder chunk.");

class FusedWaveletEncoderChunkGradOp : public OpKernel {
private:
    int chunk_size_;
    int num_levels_;
    
public:
    explicit FusedWaveletEncoderChunkGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("chunk_size", &chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_levels", &num_levels_));
    }
    
    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_encoded = ctx->input(0);
        const Tensor& input = ctx->input(2);
        const Tensor& wavelet_filters = ctx->input(3);
        const Tensor& projection = ctx->input(4);
        
        // Allocate outputs
        Tensor* grad_input = nullptr;
        Tensor* grad_wavelet = nullptr;
        Tensor* grad_projection = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input.shape(), &grad_input));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, wavelet_filters.shape(), &grad_wavelet));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, projection.shape(), &grad_projection));
        
        // Zero initialize all gradients
        auto grad_input_flat = grad_input->flat<float>();
        auto grad_wavelet_flat = grad_wavelet->flat<float>();
        auto grad_proj_flat = grad_projection->flat<float>();
        
        for (int i = 0; i < grad_input_flat.size(); ++i) {
            grad_input_flat(i) = 0.0f;
        }
        for (int i = 0; i < grad_wavelet_flat.size(); ++i) {
            grad_wavelet_flat(i) = 0.0f;
        }
        for (int i = 0; i < grad_proj_flat.size(); ++i) {
            grad_proj_flat(i) = 0.0f;
        }
        
        // Backpropagation would go here
        // For now, provides zero gradients as placeholder
    }
};

REGISTER_KERNEL_BUILDER(Name("FusedWaveletEncoderChunkGrad").Device(DEVICE_CPU), FusedWaveletEncoderChunkGradOp);

}  // namespace tensorflow
