// saguaro.native/ops/qwt_continuous_op.cc
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
 * @file qwt_continuous_op.cc
 * @brief TensorFlow op registration for continuous QWT→HD path.
 *
 * VQC-HD Integration Enhancement #3.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "qwt_continuous_op.h"

namespace tensorflow {

using namespace saguaro::qwt_continuous;

// =============================================================================
// Forward Op
// =============================================================================

REGISTER_OP("QWTContinuousEmbed")
    .Input("input_bytes: uint8")         // [batch, seq_len]
    .Input("hd_base_vectors: float32")   // [vqc_dim, hd_dim]
    .Attr("vqc_dim: int = 256")
    .Attr("hd_dim: int = 4096")
    .Attr("amplitude_scale: float = 1.0")
    .Attr("normalize_output: bool = true")
    .Output("hd_output: float32")        // [batch, seq_len, hd_dim]
    .Output("amplitudes: float32")       // [batch, seq_len, vqc_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int vqc_dim, hd_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("vqc_dim", &vqc_dim));
        TF_RETURN_IF_ERROR(c->GetAttr("hd_dim", &hd_dim));
        
        shape_inference::ShapeHandle input_shape = c->input(0);
        auto batch = c->Dim(input_shape, 0);
        auto seq = c->Dim(input_shape, 1);
        
        c->set_output(0, c->MakeShape({batch, seq, hd_dim}));
        c->set_output(1, c->MakeShape({batch, seq, vqc_dim}));
        return Status();
    })
    .Doc(R"doc(
Continuous QWT Embedding.

Converts input bytes to continuous VQC amplitudes that modulate HD base vectors,
enabling gradient flow through the tokenization process.

input_bytes: Input byte sequence.
hd_base_vectors: HD base vectors for amplitude modulation.
hd_output: Output HD embeddings.
amplitudes: VQC amplitudes used for modulation (for gradient computation).
)doc");

class QWTContinuousEmbedOp : public OpKernel {
private:
    QWTContinuousConfig config_;

public:
    explicit QWTContinuousEmbedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_dim", &config_.vqc_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("amplitude_scale", &config_.amplitude_scale));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize_output", &config_.normalize_output));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& input_t = ctx->input(0);
        const Tensor& base_t = ctx->input(1);

        const int batch_size = input_t.dim_size(0);
        const int seq_len = input_t.dim_size(1);

        // Validate base vectors shape
        OP_REQUIRES(ctx, base_t.dim_size(0) == config_.vqc_dim,
            errors::InvalidArgument("hd_base_vectors dim 0 must match vqc_dim"));
        OP_REQUIRES(ctx, base_t.dim_size(1) == config_.hd_dim,
            errors::InvalidArgument("hd_base_vectors dim 1 must match hd_dim"));

        // Allocate outputs
        Tensor* hd_output = nullptr;
        Tensor* amplitudes = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({batch_size, seq_len, config_.hd_dim}), &hd_output));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            1, TensorShape({batch_size, seq_len, config_.vqc_dim}), &amplitudes));

        // Execute forward pass
        QWTContinuousForward(
            input_t.flat<uint8>().data(),
            base_t.flat<float>().data(),
            hd_output->flat<float>().data(),
            amplitudes->flat<float>().data(),
            config_,
            batch_size, seq_len
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("QWTContinuousEmbed").Device(DEVICE_CPU), QWTContinuousEmbedOp);

// =============================================================================
// Gradient Op
// =============================================================================

REGISTER_OP("QWTContinuousEmbedGrad")
    .Input("grad_hd_output: float32")    // [batch, seq_len, hd_dim]
    .Input("input_bytes: uint8")         // [batch, seq_len]
    .Input("hd_base_vectors: float32")   // [vqc_dim, hd_dim]
    .Input("amplitudes: float32")        // [batch, seq_len, vqc_dim]
    .Attr("vqc_dim: int = 256")
    .Attr("hd_dim: int = 4096")
    .Attr("amplitude_scale: float = 1.0")
    .Attr("normalize_output: bool = true")
    .Output("grad_base_vectors: float32") // [vqc_dim, hd_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));  // Same as hd_base_vectors
        return Status();
    })
    .Doc("Gradient for QWTContinuousEmbed.");

class QWTContinuousEmbedGradOp : public OpKernel {
private:
    QWTContinuousConfig config_;

public:
    explicit QWTContinuousEmbedGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_dim", &config_.vqc_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("amplitude_scale", &config_.amplitude_scale));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize_output", &config_.normalize_output));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_hd_t = ctx->input(0);
        const Tensor& input_t = ctx->input(1);
        const Tensor& base_t = ctx->input(2);
        const Tensor& amp_t = ctx->input(3);

        const int batch_size = input_t.dim_size(0);
        const int seq_len = input_t.dim_size(1);

        // Allocate output gradient
        Tensor* grad_base = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, base_t.shape(), &grad_base));

        // Execute backward pass
        QWTContinuousBackward(
            grad_hd_t.flat<float>().data(),
            input_t.flat<uint8>().data(),
            base_t.flat<float>().data(),
            amp_t.flat<float>().data(),
            grad_base->flat<float>().data(),
            config_,
            batch_size, seq_len
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("QWTContinuousEmbedGrad").Device(DEVICE_CPU), QWTContinuousEmbedGradOp);

// =============================================================================
// Base Vector Initialization Op
// =============================================================================

REGISTER_OP("InitHDBaseVectors")
    .Attr("vqc_dim: int = 256")
    .Attr("hd_dim: int = 4096")
    .Attr("seed: int = 42")
    .Output("base_vectors: float32")  // [vqc_dim, hd_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int vqc_dim, hd_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("vqc_dim", &vqc_dim));
        TF_RETURN_IF_ERROR(c->GetAttr("hd_dim", &hd_dim));
        c->set_output(0, c->Matrix(vqc_dim, hd_dim));
        return Status();
    })
    .Doc("Initialize HD base vectors for QWT continuous embedding.");

class InitHDBaseVectorsOp : public OpKernel {
private:
    int vqc_dim_;
    int hd_dim_;
    uint32_t seed_;

public:
    explicit InitHDBaseVectorsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("vqc_dim", &vqc_dim_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &hd_dim_));
        int seed_int;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_int));
        seed_ = static_cast<uint32_t>(seed_int);
    }

    void Compute(OpKernelContext* ctx) override {
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(
            0, TensorShape({vqc_dim_, hd_dim_}), &output));

        InitializeHDBaseVectors(
            output->flat<float>().data(),
            vqc_dim_, hd_dim_, seed_
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("InitHDBaseVectors").Device(DEVICE_CPU), InitHDBaseVectorsOp);

}  // namespace tensorflow
