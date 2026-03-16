// saguaro.native/ops/fused_memory_builder_enhancements_op.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// TensorFlow C++ custom ops for Memory Builder enhancements.
// Implements Enhancements 3-7 from the Memory Builder roadmap.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "memory_builder_enhancements.h"

#include <vector>
#include <cmath>

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

// =============================================================================
// OP REGISTRATION
// =============================================================================

// Enhancement 3: CTQW Aggregation
REGISTER_OP("FusedCTQWAggregate")
    .Input("x: float32")
    .Attr("time: float = 1.0")
    .Attr("use_cayley: bool = true")
    .Attr("sigma: float = -1.0")
    .Output("weights: float32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
        DimensionHandle batch = c->Dim(input, 0);
        DimensionHandle num_nodes = c->Dim(input, 1);
        c->set_output(0, c->MakeShape({batch, num_nodes, num_nodes}));
        return Status();
    })
    .Doc(R"doc(
CTQW-based aggregation weights computation.
Uses Cayley-approximated matrix exponential for continuous-time quantum walk.
)doc");

// Enhancement 4: Multi-Rate EMA
REGISTER_OP("FusedMultiRateEMA")
    .Input("memory: float32")
    .Input("aggregated: float32")
    .Attr("level: int = 0")
    .Attr("base_rate: float = 0.1")
    .Attr("level_decay: float = 0.5")
    .Output("output: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Multi-rate EMA update for hierarchical memory.
memory_new = α * memory + (1-α) * aggregated where α decays with level.
)doc");

REGISTER_OP("FusedMultiRateEMAGrad")
    .Input("grad_output: float32")
    .Attr("level: int = 0")
    .Attr("base_rate: float = 0.1")
    .Attr("level_decay: float = 0.5")
    .Output("grad_memory: float32")
    .Output("grad_aggregated: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(0));
        return Status();
    });

// Enhancement 5: Cross-Level Attention
REGISTER_OP("FusedCrossLevelAttention")
    .Input("query: float32")
    .Input("key: float32")
    .Input("value: float32")
    .Attr("num_heads: int = 4")
    .Attr("residual_scale: float = 1.0")
    .Output("output: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
O(n) linear cross-level attention using ELU+1 kernel.
Enables bidirectional information flow across hierarchy levels.
)doc");

// Enhancement 6: Adaptive Chunking
REGISTER_OP("FusedAdaptiveChunk")
    .Input("x: float32")
    .Attr("min_chunk_size: int = 2")
    .Attr("max_chunk_size: int = 16")
    .Attr("boundary_threshold: float = 0.5")
    .Output("chunk_ids: int32")
    .Output("num_chunks: int32")
    .SetShapeFn([](InferenceContext* c) {
        ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
        DimensionHandle batch = c->Dim(input, 0);
        DimensionHandle seq_len = c->Dim(input, 1);
        c->set_output(0, c->MakeShape({batch, seq_len}));
        c->set_output(1, c->MakeShape({batch}));
        return Status();
    })
    .Doc(R"doc(
Adaptive content-based chunking using semantic boundaries.
Returns chunk assignments for each token.
)doc");

REGISTER_OP("FusedChunkPool")
    .Input("x: float32")
    .Input("chunk_ids: int32")
    .Input("num_chunks: int32")
    .Output("pooled: float32")
    .SetShapeFn([](InferenceContext* c) {
        // Output shape depends on num_chunks, use unknown for seq dim
        ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input));
        DimensionHandle batch = c->Dim(input, 0);
        DimensionHandle embed_dim = c->Dim(input, 2);
        c->set_output(0, c->MakeShape({batch, c->UnknownDim(), embed_dim}));
        return Status();
    })
    .Doc(R"doc(
Pool within adaptive chunks using mean aggregation.
)doc");

// Enhancement 7: Quantum Noise
REGISTER_OP("FusedQuantumNoise")
    .Input("shape: int32")
    .Attr("entanglement_strength: float = 0.1")
    .Attr("seed: int = 42")
    .Output("noise: float32")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
Generate structured quantum noise for QGAN training.
Uses rotation matrices and entanglement correlations.
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

// Enhancement 3: CTQW Aggregation
class FusedCTQWAggregateOp : public OpKernel {
public:
    explicit FusedCTQWAggregateOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("time", &time_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("use_cayley", &use_cayley_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("sigma", &sigma_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x = ctx->input(0);
        
        OP_REQUIRES(ctx, x.dims() == 3,
            errors::InvalidArgument("Input must be 3D [batch, nodes, embed_dim]"));
        
        const int64_t batch = x.dim_size(0);
        const int64_t num_nodes = x.dim_size(1);
        const int64_t embed_dim = x.dim_size(2);
        
        // Output: [batch, num_nodes, num_nodes]
        Tensor* weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({batch, num_nodes, num_nodes}), &weights));
        
        auto x_flat = x.flat<float>();
        auto weights_flat = weights->flat<float>();
        
        saguaro::ops::CTQWAggregator aggregator(time_, use_cayley_);
        
        // Process each batch
        for (int64_t b = 0; b < batch; ++b) {
            aggregator.compute_weights(
                x_flat.data() + b * num_nodes * embed_dim,
                weights_flat.data() + b * num_nodes * num_nodes,
                num_nodes,
                embed_dim,
                sigma_
            );
        }
    }

private:
    float time_;
    bool use_cayley_;
    float sigma_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedCTQWAggregate").Device(DEVICE_CPU),
    FusedCTQWAggregateOp);

// Enhancement 4: Multi-Rate EMA
class FusedMultiRateEMAOp : public OpKernel {
public:
    explicit FusedMultiRateEMAOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("level", &level_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("base_rate", &base_rate_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("level_decay", &level_decay_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& memory = ctx->input(0);
        const Tensor& aggregated = ctx->input(1);
        
        OP_REQUIRES(ctx, memory.shape() == aggregated.shape(),
            errors::InvalidArgument("Memory and aggregated must have same shape"));
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, memory.shape(), &output));
        
        const int64_t batch = memory.dim_size(0);
        const int64_t num_tokens = memory.dim_size(1);
        const int64_t embed_dim = memory.dim_size(2);
        
        saguaro::ops::MultiRateProcessor processor(base_rate_, level_decay_);
        
        for (int64_t b = 0; b < batch; ++b) {
            processor.apply(
                memory.flat<float>().data() + b * num_tokens * embed_dim,
                aggregated.flat<float>().data() + b * num_tokens * embed_dim,
                output->flat<float>().data() + b * num_tokens * embed_dim,
                num_tokens,
                embed_dim,
                level_
            );
        }
    }

private:
    int level_;
    float base_rate_;
    float level_decay_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedMultiRateEMA").Device(DEVICE_CPU),
    FusedMultiRateEMAOp);

class FusedMultiRateEMAGradOp : public OpKernel {
public:
    explicit FusedMultiRateEMAGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("level", &level_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("base_rate", &base_rate_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("level_decay", &level_decay_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& grad_output = ctx->input(0);
        
        Tensor* grad_memory = nullptr;
        Tensor* grad_aggregated = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, grad_output.shape(), &grad_memory));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, grad_output.shape(), &grad_aggregated));
        
        const int64_t batch = grad_output.dim_size(0);
        const int64_t num_tokens = grad_output.dim_size(1);
        const int64_t embed_dim = grad_output.dim_size(2);
        
        saguaro::ops::MultiRateProcessor processor(base_rate_, level_decay_);
        
        for (int64_t b = 0; b < batch; ++b) {
            processor.grad(
                grad_output.flat<float>().data() + b * num_tokens * embed_dim,
                grad_memory->flat<float>().data() + b * num_tokens * embed_dim,
                grad_aggregated->flat<float>().data() + b * num_tokens * embed_dim,
                num_tokens,
                embed_dim,
                level_
            );
        }
    }

private:
    int level_;
    float base_rate_;
    float level_decay_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedMultiRateEMAGrad").Device(DEVICE_CPU),
    FusedMultiRateEMAGradOp);

// Enhancement 5: Cross-Level Attention
class FusedCrossLevelAttentionOp : public OpKernel {
public:
    explicit FusedCrossLevelAttentionOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_heads", &num_heads_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("residual_scale", &residual_scale_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& query = ctx->input(0);
        const Tensor& key = ctx->input(1);
        const Tensor& value = ctx->input(2);
        
        OP_REQUIRES(ctx, query.dims() == 3 && key.dims() == 3 && value.dims() == 3,
            errors::InvalidArgument("All inputs must be 3D"));
        
        const int64_t batch = query.dim_size(0);
        const int64_t num_query = query.dim_size(1);
        const int64_t num_kv = key.dim_size(1);
        const int64_t embed_dim = query.dim_size(2);
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &output));
        
        saguaro::ops::CrossLevelAttention attention(num_heads_, 
            static_cast<int>(embed_dim / num_heads_), true);
        
        for (int64_t b = 0; b < batch; ++b) {
            attention.apply_residual(
                query.flat<float>().data() + b * num_query * embed_dim,
                key.flat<float>().data() + b * num_kv * embed_dim,
                value.flat<float>().data() + b * num_kv * embed_dim,
                output->flat<float>().data() + b * num_query * embed_dim,
                num_query,
                num_kv,
                embed_dim,
                residual_scale_
            );
        }
    }

private:
    int num_heads_;
    float residual_scale_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedCrossLevelAttention").Device(DEVICE_CPU),
    FusedCrossLevelAttentionOp);

// Enhancement 6: Adaptive Chunking
class FusedAdaptiveChunkOp : public OpKernel {
public:
    explicit FusedAdaptiveChunkOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("min_chunk_size", &min_chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("max_chunk_size", &max_chunk_size_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("boundary_threshold", &boundary_threshold_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x = ctx->input(0);
        
        OP_REQUIRES(ctx, x.dims() == 3,
            errors::InvalidArgument("Input must be 3D [batch, seq, embed_dim]"));
        
        const int64_t batch = x.dim_size(0);
        const int64_t seq_len = x.dim_size(1);
        const int64_t embed_dim = x.dim_size(2);
        
        Tensor* chunk_ids = nullptr;
        Tensor* num_chunks = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({batch, seq_len}), &chunk_ids));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, 
            TensorShape({batch}), &num_chunks));
        
        saguaro::ops::AdaptiveChunker chunker(
            min_chunk_size_, max_chunk_size_, boundary_threshold_);
        
        auto x_flat = x.flat<float>();
        auto chunk_ids_flat = chunk_ids->flat<int32>();
        auto num_chunks_flat = num_chunks->flat<int32>();
        
        for (int64_t b = 0; b < batch; ++b) {
            int n_chunks = chunker.compute_chunks(
                x_flat.data() + b * seq_len * embed_dim,
                chunk_ids_flat.data() + b * seq_len,
                seq_len,
                embed_dim
            );
            num_chunks_flat(b) = n_chunks;
        }
    }

private:
    int min_chunk_size_;
    int max_chunk_size_;
    float boundary_threshold_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedAdaptiveChunk").Device(DEVICE_CPU),
    FusedAdaptiveChunkOp);

class FusedChunkPoolOp : public OpKernel {
public:
    explicit FusedChunkPoolOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& x = ctx->input(0);
        const Tensor& chunk_ids = ctx->input(1);
        const Tensor& num_chunks_tensor = ctx->input(2);
        
        const int64_t batch = x.dim_size(0);
        const int64_t seq_len = x.dim_size(1);
        const int64_t embed_dim = x.dim_size(2);
        
        // Find max chunks across batch for output allocation
        auto num_chunks_flat = num_chunks_tensor.flat<int32>();
        int max_chunks = 0;
        for (int64_t b = 0; b < batch; ++b) {
            max_chunks = std::max(max_chunks, static_cast<int>(num_chunks_flat(b)));
        }
        
        Tensor* pooled = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, 
            TensorShape({batch, max_chunks, embed_dim}), &pooled));
        
        // Initialize to zero
        auto pooled_flat = pooled->flat<float>();
        std::fill(pooled_flat.data(), pooled_flat.data() + batch * max_chunks * embed_dim, 0.0f);
        
        saguaro::ops::AdaptiveChunker chunker;
        
        auto x_flat = x.flat<float>();
        auto chunk_ids_flat = chunk_ids.flat<int32>();
        
        for (int64_t b = 0; b < batch; ++b) {
            chunker.pool_chunks(
                x_flat.data() + b * seq_len * embed_dim,
                chunk_ids_flat.data() + b * seq_len,
                pooled_flat.data() + b * max_chunks * embed_dim,
                seq_len,
                num_chunks_flat(b),
                embed_dim
            );
        }
    }
};

REGISTER_KERNEL_BUILDER(
    Name("FusedChunkPool").Device(DEVICE_CPU),
    FusedChunkPoolOp);

// Enhancement 7: Quantum Noise
class FusedQuantumNoiseOp : public OpKernel {
public:
    explicit FusedQuantumNoiseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("entanglement_strength", &entanglement_strength_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("seed", &seed_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& shape_tensor = ctx->input(0);
        
        OP_REQUIRES(ctx, shape_tensor.dims() == 1,
            errors::InvalidArgument("Shape must be 1D"));
        
        auto shape_flat = shape_tensor.flat<int32>();
        TensorShape output_shape;
        for (int i = 0; i < shape_tensor.dim_size(0); ++i) {
            output_shape.AddDim(shape_flat(i));
        }
        
        Tensor* noise = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &noise));
        
        // Assume 2D: [batch, dim]
        int64_t batch = 1;
        int64_t dim = noise->NumElements();
        if (output_shape.dims() >= 2) {
            batch = output_shape.dim_size(0);
            dim = noise->NumElements() / batch;
        }
        
        saguaro::ops::QuantumNoiseGenerator generator(entanglement_strength_, seed_);
        generator.generate(noise->flat<float>().data(), batch, dim);
    }

private:
    float entanglement_strength_;
    int seed_;
};

REGISTER_KERNEL_BUILDER(
    Name("FusedQuantumNoise").Device(DEVICE_CPU),
    FusedQuantumNoiseOp);

}  // namespace tensorflow
