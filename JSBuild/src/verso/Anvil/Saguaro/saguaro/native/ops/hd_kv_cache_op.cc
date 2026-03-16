// saguaro.native/ops/hd_kv_cache_op.cc
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
 * @file hd_kv_cache_op.cc
 * @brief TensorFlow ops for HD compressed KV cache.
 *
 * Registers:
 * - HDKVCacheCompress: Compress KV cache via holographic bundling
 * - HDKVCacheDecompress: Decompress specific positions
 * - HDKVCacheAppend: Incrementally append to compressed cache
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "hd_kv_cache_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// Op Registrations
// =============================================================================

REGISTER_OP("HDKVCacheCompress")
    .Input("kv_cache: float32")
    .Input("pos_keys: float32")
    .Attr("compression_ratio: int = 8")
    .Output("compressed: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle kv_shape = c->input(0);
        int ratio;
        TF_RETURN_IF_ERROR(c->GetAttr("compression_ratio", &ratio));
        
        DimensionHandle batch = c->Dim(kv_shape, 0);
        DimensionHandle heads = c->Dim(kv_shape, 1);
        DimensionHandle seq_len = c->Dim(kv_shape, 2);
        DimensionHandle head_dim = c->Dim(kv_shape, 3);
        
        // num_bundles = ceil(seq_len / ratio)
        DimensionHandle num_bundles;
        if (c->ValueKnown(seq_len)) {
            int64_t sl = c->Value(seq_len);
            num_bundles = c->MakeDim((sl + ratio - 1) / ratio);
        } else {
            num_bundles = c->UnknownDim();
        }
        
        c->set_output(0, c->MakeShape({batch, heads, num_bundles, head_dim}));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Compress KV cache using holographic bundling.

kv_cache: Input KV cache [batch, heads, seq_len, head_dim].
pos_keys: Position keys [max_seq, head_dim].
compression_ratio: Tokens per HD bundle.
compressed: Compressed cache [batch, heads, num_bundles, head_dim].
)doc");

REGISTER_OP("HDKVCacheDecompress")
    .Input("compressed: float32")
    .Input("pos_keys: float32")
    .Attr("seq_len: int")
    .Attr("compression_ratio: int = 8")
    .Output("kv_cache: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle comp_shape = c->input(0);
        int seq_len;
        TF_RETURN_IF_ERROR(c->GetAttr("seq_len", &seq_len));
        
        DimensionHandle batch = c->Dim(comp_shape, 0);
        DimensionHandle heads = c->Dim(comp_shape, 1);
        DimensionHandle head_dim = c->Dim(comp_shape, 3);
        
        c->set_output(0, c->MakeShape({batch, heads, seq_len, head_dim}));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Decompress KV cache by unbinding specific positions.

compressed: Compressed cache [batch, heads, num_bundles, head_dim].
pos_keys: Position keys [max_seq, head_dim].
seq_len: Target sequence length.
compression_ratio: Tokens per HD bundle.
kv_cache: Decompressed cache [batch, heads, seq_len, head_dim].
)doc");

REGISTER_OP("HDKVCacheAppend")
    .Input("compressed: float32")
    .Input("new_kv: float32")
    .Input("pos_key: float32")
    .Attr("position: int")
    .Attr("compression_ratio: int = 8")
    .Output("updated: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return absl::OkStatus();
    })
    .Doc(R"doc(
Append new K/V to compressed cache (incremental update).

compressed: Current compressed cache [batch, heads, num_bundles, head_dim].
new_kv: New K/V vector [batch, heads, head_dim].
pos_key: Position key for new token [head_dim].
position: Token position.
compression_ratio: Tokens per HD bundle.
updated: Updated compressed cache.
)doc");

// =============================================================================
// Kernel Implementations
// =============================================================================

class HDKVCacheCompressOp : public OpKernel {
public:
    explicit HDKVCacheCompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("compression_ratio", &compression_ratio_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& kv_tensor = ctx->input(0);
        const Tensor& pos_tensor = ctx->input(1);
        
        OP_REQUIRES(ctx, kv_tensor.dims() == 4,
            errors::InvalidArgument("KV cache must be 4D"));
        
        const int batch_size = kv_tensor.dim_size(0);
        const int num_heads = kv_tensor.dim_size(1);
        const int seq_len = kv_tensor.dim_size(2);
        const int head_dim = kv_tensor.dim_size(3);
        const int num_bundles = (seq_len + compression_ratio_ - 1) / compression_ratio_;
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, num_heads, num_bundles, head_dim}),
            &output_tensor));
        
        saguaro::hd_kv::HDKVCacheConfig config;
        config.compression_ratio = compression_ratio_;
        
        saguaro::hd_kv::HDKVCacheCompress(
            kv_tensor.flat<float>().data(),
            pos_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            batch_size, num_heads, seq_len, head_dim,
            config
        );
    }

private:
    int compression_ratio_;
};

class HDKVCacheDecompressOp : public OpKernel {
public:
    explicit HDKVCacheDecompressOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seq_len", &seq_len_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("compression_ratio", &compression_ratio_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& comp_tensor = ctx->input(0);
        const Tensor& pos_tensor = ctx->input(1);
        
        const int batch_size = comp_tensor.dim_size(0);
        const int num_heads = comp_tensor.dim_size(1);
        const int head_dim = comp_tensor.dim_size(3);
        
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, num_heads, seq_len_, head_dim}),
            &output_tensor));
        
        saguaro::hd_kv::HDKVCacheConfig config;
        config.compression_ratio = compression_ratio_;
        
        saguaro::hd_kv::HDKVCacheDecompress(
            comp_tensor.flat<float>().data(),
            pos_tensor.flat<float>().data(),
            output_tensor->flat<float>().data(),
            batch_size, num_heads, seq_len_, head_dim,
            config
        );
    }

private:
    int seq_len_;
    int compression_ratio_;
};

class HDKVCacheAppendOp : public OpKernel {
public:
    explicit HDKVCacheAppendOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("position", &position_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("compression_ratio", &compression_ratio_));
    }

    void Compute(OpKernelContext* ctx) override {
        ctx->set_output(0, ctx->input(0));
        Tensor* output_tensor = ctx->mutable_output(0);
        
        const Tensor& new_kv = ctx->input(1);
        const Tensor& pos_key = ctx->input(2);
        
        const int batch_size = new_kv.dim_size(0);
        const int num_heads = new_kv.dim_size(1);
        const int head_dim = new_kv.dim_size(2);
        
        saguaro::hd_kv::HDKVCacheConfig config;
        config.compression_ratio = compression_ratio_;
        
        saguaro::hd_kv::HDKVCacheAppend(
            output_tensor->flat<float>().data(),
            new_kv.flat<float>().data(),
            pos_key.flat<float>().data(),
            position_,
            batch_size, num_heads, head_dim,
            config
        );
    }

private:
    int position_;
    int compression_ratio_;
};

// =============================================================================
// Kernel Registrations
// =============================================================================

REGISTER_KERNEL_BUILDER(
    Name("HDKVCacheCompress").Device(DEVICE_CPU),
    HDKVCacheCompressOp);

REGISTER_KERNEL_BUILDER(
    Name("HDKVCacheDecompress").Device(DEVICE_CPU),
    HDKVCacheDecompressOp);

REGISTER_KERNEL_BUILDER(
    Name("HDKVCacheAppend").Device(DEVICE_CPU),
    HDKVCacheAppendOp);

}  // namespace tensorflow
