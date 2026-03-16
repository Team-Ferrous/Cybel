// saguaro.native/ops/hd_holographic_similarity_op.cc
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
 * @file hd_holographic_similarity_op.cc
 * @brief TensorFlow ops for holographic attention similarity.
 *
 * Registers:
 * - HolographicAttentionScores: Compute Q·K scores via FFT correlation
 * - HolographicPositionBind: Apply position-aware holographic binding
 * - GeneratePositionKeys: Generate Floquet-inspired position keys
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_holographic_similarity_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HolographicAttentionScores")
    .Input("queries: float32")
    .Input("keys: float32")
    .Attr("temperature: float = 1.0")
    .Output("scores: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle q_shape = c->input(0);  // [batch, heads, seq_q, head_dim]
        ShapeHandle k_shape = c->input(1);  // [batch, heads, seq_k, head_dim]
        
        DimensionHandle batch = c->Dim(q_shape, 0);
        DimensionHandle heads = c->Dim(q_shape, 1);
        DimensionHandle seq_q = c->Dim(q_shape, 2);
        DimensionHandle seq_k = c->Dim(k_shape, 2);
        
        c->set_output(0, c->MakeShape({batch, heads, seq_q, seq_k}));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute holographic attention scores via FFT-based circular correlation.

queries: Query tensor [batch, heads, seq_q, head_dim].
keys: Key tensor [batch, heads, seq_k, head_dim].
temperature: Softmax temperature.
scores: Attention scores [batch, heads, seq_q, seq_k].
)doc");

REGISTER_OP("HolographicPositionBind")
    .Input("tensor: float32")
    .Input("pos_keys: float32")
    .Output("bound: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Apply position-aware holographic binding to Q/K tensors.

tensor: Input [batch, heads, seq, head_dim].
pos_keys: Position keys [max_seq, head_dim].
bound: Position-bound output [batch, heads, seq, head_dim].
)doc");

REGISTER_OP("GeneratePositionKeys")
    .Attr("max_seq: int")
    .Attr("head_dim: int")
    .Attr("base_freq: float = 10000.0")
    .Output("pos_keys: float32")
    .SetShapeFn([](InferenceContext* c) {
        int max_seq, head_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("max_seq", &max_seq));
        TF_RETURN_IF_ERROR(c->GetAttr("head_dim", &head_dim));
        c->set_output(0, c->Matrix(max_seq, head_dim));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Generate Floquet-inspired position keys for holographic attention.

max_seq: Maximum sequence length.
head_dim: Head dimension.
base_freq: Base frequency for sinusoidal encoding.
pos_keys: Generated position keys [max_seq, head_dim].
)doc");

REGISTER_OP("HolographicBind")
    .Input("x: float32")
    .Input("y: float32")
    .Output("bound: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Holographic bind: x ⊛ y = IFFT(FFT(x) * FFT(y)).

x: First tensor [..., dim].
y: Second tensor [..., dim].
bound: Bound result [..., dim].
)doc");

REGISTER_OP("HolographicSimilarity")
    .Input("x: float32")
    .Input("y: float32")
    .Output("similarity: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle shape = c->input(0);
        int rank = c->Rank(shape);
        if (rank > 1) {
            std::vector<DimensionHandle> dims;
            for (int i = 0; i < rank - 1; ++i) {
                dims.push_back(c->Dim(shape, i));
            }
            c->set_output(0, c->MakeShape(dims));
        } else {
            c->set_output(0, c->Scalar());
        }
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compute holographic similarity between vectors.

x: First tensor [..., dim].
y: Second tensor [..., dim].
similarity: Similarity scores [...].
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HolographicAttentionScoresOp : public OpKernel {
public:
    explicit HolographicAttentionScoresOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& q_tensor = ctx->input(0);
        const Tensor& k_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, q_tensor.dims() == 4,
            errors::InvalidArgument("Queries must be 4D"));
        OP_REQUIRES(ctx, k_tensor.dims() == 4,
            errors::InvalidArgument("Keys must be 4D"));
        
        const int batch_size = q_tensor.dim_size(0);
        const int num_heads = q_tensor.dim_size(1);
        const int seq_q = q_tensor.dim_size(2);
        const int seq_k = k_tensor.dim_size(2);
        const int head_dim = q_tensor.dim_size(3);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, num_heads, seq_q, seq_k}),
            &output_tensor));
        
        saguaro::hd_attention::HDAttentionConfig config;
        config.temperature = temperature_;
        
        saguaro::hd_attention::HolographicAttentionScores(
            q_tensor.flat<float>().data(),
            k_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            batch_size, num_heads, seq_q, seq_k, head_dim,
            config
        );
    }

private:
    float temperature_;
};

class HolographicPositionBindOp : public OpKernel {
public:
    explicit HolographicPositionBindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& tensor = ctx->input(0);
        const Tensor& pos_keys = ctx->input(1);
        
        OP_REQUIRES(ctx, tensor.dims() == 4,
            errors::InvalidArgument("Input must be 4D"));
        
        const int batch_size = tensor.dim_size(0);
        const int num_heads = tensor.dim_size(1);
        const int seq_len = tensor.dim_size(2);
        const int head_dim = tensor.dim_size(3);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, tensor.shape(), &output_tensor));
        
        saguaro::hd_attention::HolographicPositionBind(
            tensor.flat<float>().data(),
            pos_keys.flat<float>().data(),
            output_tensor->flat<float>().data(),
            batch_size, num_heads, seq_len, head_dim
        );
    }
};

class GeneratePositionKeysOp : public OpKernel {
public:
    explicit GeneratePositionKeysOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_seq", &max_seq_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("head_dim", &head_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("base_freq", &base_freq_));
    }

    void Compute(OpKernelContext* ctx) override {
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({max_seq_, head_dim_}), &output_tensor));
        
        saguaro::hd_attention::GeneratePositionKeys(
            output_tensor->flat<float>().data(),
            max_seq_, head_dim_, base_freq_
        );
    }

private:
    int max_seq_;
    int head_dim_;
    float base_freq_;
};

class HolographicBindOp : public OpKernel {
public:
    explicit HolographicBindOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x_tensor = ctx->input(0);
        const Tensor& y_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, x_tensor.shape() == y_tensor.shape(),
            errors::InvalidArgument("Shapes must match"));
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x_tensor.shape(), &output_tensor));
        
        const int total_size = x_tensor.NumElements();
        const int dim = x_tensor.dim_size(x_tensor.dims() - 1);
        const int num_vectors = total_size / dim;
        
        const float* x = x_tensor.flat<float>().data();
        const float* y = y_tensor.flat<float>().data();
        float* out = output_tensor->flat<float>().data();
        
        #pragma omp parallel for
        for (int i = 0; i < num_vectors; ++i) {
            saguaro::hd_attention::holographic_bind(
                x + i * dim, y + i * dim, out + i * dim, dim
            );
        }
    }
};

class HolographicSimilarityOp : public OpKernel {
public:
    explicit HolographicSimilarityOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x_tensor = ctx->input(0);
        const Tensor& y_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, x_tensor.shape() == y_tensor.shape(),
            errors::InvalidArgument("Shapes must match"));
        
        const int rank = x_tensor.dims();
        const int dim = x_tensor.dim_size(rank - 1);
        
        // Output shape is input shape without last dimension
        TensorShape output_shape;
        for (int i = 0; i < rank - 1; ++i) {
            output_shape.AddDim(x_tensor.dim_size(i));
        }
        if (output_shape.dims() == 0) {
            output_shape.AddDim(1);
        }
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));
        
        const int num_vectors = x_tensor.NumElements() / dim;
        const float* x = x_tensor.flat<float>().data();
        const float* y = y_tensor.flat<float>().data();
        float* out = output_tensor->flat<float>().data();
        
        #pragma omp parallel for
        for (int i = 0; i < num_vectors; ++i) {
            out[i] = saguaro::hd_attention::holographic_similarity(
                x + i * dim, y + i * dim, dim
            );
        }
    }
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HolographicAttentionScores").Device(DEVICE_CPU),
    HolographicAttentionScoresOp);

REGISTER_KERNEL_BUILDER(
    Name("HolographicPositionBind").Device(DEVICE_CPU),
    HolographicPositionBindOp);

REGISTER_KERNEL_BUILDER(
    Name("GeneratePositionKeys").Device(DEVICE_CPU),
    GeneratePositionKeysOp);

REGISTER_KERNEL_BUILDER(
    Name("HolographicBind").Device(DEVICE_CPU),
    HolographicBindOp);

REGISTER_KERNEL_BUILDER(
    Name("HolographicSimilarity").Device(DEVICE_CPU),
    HolographicSimilarityOp);

}  // namespace tensorflow
