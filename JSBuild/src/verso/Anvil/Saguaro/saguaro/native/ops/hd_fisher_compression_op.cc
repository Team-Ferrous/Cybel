// saguaro.native/ops/hd_fisher_compression_op.cc
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
 * @file hd_fisher_compression_op.cc
 * @brief TensorFlow op registration for HD Fisher Compression.
 *
 * VQC-HD Integration Enhancement #1.
 */

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "hd_fisher_compression_op.h"

namespace tensorflow {

using namespace saguaro::hd_fisher;

// =============================================================================
// Forward Op
// =============================================================================

REGISTER_OP("HDFisherCompress")
    .Input("fisher_values: float32")    // [batch, num_layers] or [num_layers]
    .Input("pos_keys: float32")         // [num_layers, hd_dim]
    .Input("proj_weights: float32")     // [hd_dim, out_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("out_dim: int = 64")
    .Attr("normalize: bool = true")
    .Attr("scale: float = 1.0")
    .Output("output: float32")          // [batch, out_dim] or [out_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int out_dim;
        TF_RETURN_IF_ERROR(c->GetAttr("out_dim", &out_dim));
        
        shape_inference::ShapeHandle fisher_shape = c->input(0);
        int rank = c->Rank(fisher_shape);
        
        if (rank == 1) {
            // Unbatched: [num_layers] -> [out_dim]
            c->set_output(0, c->Vector(out_dim));
        } else {
            // Batched: [batch, num_layers] -> [batch, out_dim]
            c->set_output(0, c->Matrix(c->Dim(fisher_shape, 0), out_dim));
        }
        return Status();
    })
    .Doc(R"doc(
HD Fisher Compression.

Compresses layer-wise Fisher information into a fixed-size HD vector using
holographic bundling. This enables efficient VQC encoding regardless of
model layer count.

fisher_values: Fisher information per layer.
pos_keys: Position binding keys for each layer.
proj_weights: Projection from HD space to output dimension.
output: Compressed representation for VQC encoding.
)doc");

class HDFisherCompressOp : public OpKernel {
private:
    HDFisherConfig config_;

public:
    explicit HDFisherCompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("out_dim", &config_.out_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize", &config_.normalize));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &config_.scale));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& fisher_t = ctx->input(0);
        const Tensor& keys_t = ctx->input(1);
        const Tensor& proj_t = ctx->input(2);

        const int rank = fisher_t.dims();
        const bool is_batched = (rank == 2);
        
        int batch_size = 1;
        int num_layers;
        
        if (is_batched) {
            batch_size = fisher_t.dim_size(0);
            num_layers = fisher_t.dim_size(1);
        } else {
            num_layers = fisher_t.dim_size(0);
        }

        // Validate inputs
        OP_REQUIRES(ctx, keys_t.dim_size(0) == num_layers,
            errors::InvalidArgument("pos_keys dim 0 must match num_layers"));
        OP_REQUIRES(ctx, keys_t.dim_size(1) == config_.hd_dim,
            errors::InvalidArgument("pos_keys dim 1 must match hd_dim"));
        OP_REQUIRES(ctx, proj_t.dim_size(0) == config_.hd_dim,
            errors::InvalidArgument("proj_weights dim 0 must match hd_dim"));
        OP_REQUIRES(ctx, proj_t.dim_size(1) == config_.out_dim,
            errors::InvalidArgument("proj_weights dim 1 must match out_dim"));

        // Allocate output
        Tensor* output = nullptr;
        if (is_batched) {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(
                0, TensorShape({batch_size, config_.out_dim}), &output));
        } else {
            OP_REQUIRES_OK(ctx, ctx->allocate_output(
                0, TensorShape({config_.out_dim}), &output));
        }

        // Get data pointers
        auto fisher_ptr = fisher_t.flat<float>().data();
        auto keys_ptr = keys_t.flat<float>().data();
        auto proj_ptr = proj_t.flat<float>().data();
        auto out_ptr = output->flat<float>().data();

        // Execute
        if (is_batched) {
            HDFisherCompressForwardBatch(
                fisher_ptr, keys_ptr, proj_ptr, out_ptr,
                batch_size, num_layers, config_
            );
        } else {
            HDFisherCompressForward(
                fisher_ptr, keys_ptr, proj_ptr, out_ptr,
                num_layers, config_
            );
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("HDFisherCompress").Device(DEVICE_CPU), HDFisherCompressOp);

// =============================================================================
// Gradient Op
// =============================================================================

REGISTER_OP("HDFisherCompressGrad")
    .Input("grad_output: float32")      // [batch, out_dim] or [out_dim]
    .Input("fisher_values: float32")    // [batch, num_layers] or [num_layers]
    .Input("pos_keys: float32")         // [num_layers, hd_dim]
    .Input("proj_weights: float32")     // [hd_dim, out_dim]
    .Attr("hd_dim: int = 4096")
    .Attr("out_dim: int = 64")
    .Attr("normalize: bool = true")
    .Attr("scale: float = 1.0")
    .Output("grad_fisher: float32")     // [batch, num_layers] or [num_layers]
    .Output("grad_keys: float32")       // [num_layers, hd_dim]
    .Output("grad_proj: float32")       // [hd_dim, out_dim]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));  // Same as fisher_values
        c->set_output(1, c->input(2));  // Same as pos_keys
        c->set_output(2, c->input(3));  // Same as proj_weights
        return Status();
    })
    .Doc("Gradient for HDFisherCompress.");

class HDFisherCompressGradOp : public OpKernel {
private:
    HDFisherConfig config_;

public:
    explicit HDFisherCompressGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("hd_dim", &config_.hd_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("out_dim", &config_.out_dim));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("normalize", &config_.normalize));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("scale", &config_.scale));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_out_t = ctx->input(0);
        const Tensor& fisher_t = ctx->input(1);
        const Tensor& keys_t = ctx->input(2);
        const Tensor& proj_t = ctx->input(3);

        const int rank = fisher_t.dims();
        const bool is_batched = (rank == 2);
        
        int batch_size = 1;
        int num_layers;
        
        if (is_batched) {
            batch_size = fisher_t.dim_size(0);
            num_layers = fisher_t.dim_size(1);
        } else {
            num_layers = fisher_t.dim_size(0);
        }

        // Allocate outputs
        Tensor* grad_fisher = nullptr;
        Tensor* grad_keys = nullptr;
        Tensor* grad_proj = nullptr;
        
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, fisher_t.shape(), &grad_fisher));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, keys_t.shape(), &grad_keys));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, proj_t.shape(), &grad_proj));

        // Get data pointers
        auto grad_out_ptr = grad_out_t.flat<float>().data();
        auto fisher_ptr = fisher_t.flat<float>().data();
        auto keys_ptr = keys_t.flat<float>().data();
        auto proj_ptr = proj_t.flat<float>().data();
        
        auto grad_fisher_ptr = grad_fisher->flat<float>().data();
        auto grad_keys_ptr = grad_keys->flat<float>().data();
        auto grad_proj_ptr = grad_proj->flat<float>().data();

        // Zero-initialize gradient accumulators
        std::fill_n(grad_keys_ptr, num_layers * config_.hd_dim, 0.0f);
        std::fill_n(grad_proj_ptr, config_.hd_dim * config_.out_dim, 0.0f);

        // Execute backward pass
        if (is_batched) {
            #pragma omp parallel for
            for (int b = 0; b < batch_size; ++b) {
                std::vector<float> local_grad_keys(num_layers * config_.hd_dim, 0.0f);
                std::vector<float> local_grad_proj(config_.hd_dim * config_.out_dim, 0.0f);
                
                HDFisherCompressBackward(
                    grad_out_ptr + b * config_.out_dim,
                    fisher_ptr + b * num_layers,
                    keys_ptr, proj_ptr,
                    grad_fisher_ptr + b * num_layers,
                    local_grad_keys.data(),
                    local_grad_proj.data(),
                    num_layers, config_
                );

                // Accumulate (thread-safe via atomic or reduction)
                #pragma omp critical
                {
                    for (int i = 0; i < num_layers * config_.hd_dim; ++i) {
                        grad_keys_ptr[i] += local_grad_keys[i];
                    }
                    for (int i = 0; i < config_.hd_dim * config_.out_dim; ++i) {
                        grad_proj_ptr[i] += local_grad_proj[i];
                    }
                }
            }
        } else {
            HDFisherCompressBackward(
                grad_out_ptr, fisher_ptr, keys_ptr, proj_ptr,
                grad_fisher_ptr, grad_keys_ptr, grad_proj_ptr,
                num_layers, config_
            );
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("HDFisherCompressGrad").Device(DEVICE_CPU), HDFisherCompressGradOp);

}  // namespace tensorflow
