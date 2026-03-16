// saguaro_proposal/src/ops/saguaro_memory_ops.cc
// Copyright 2026 Verso Industries
//
// Wraps HighNoon memory kernels for SAGUARO.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"

#include "quantum_holographic_memory.h"
#include "quantum_crystallization_op.h"

namespace tensorflow {

using shape_inference::InferenceContext;

// =============================================================================
// Op: HolographicBundle
// =============================================================================

REGISTER_OP("SAGUAROHolographicBundle")
    .Input("vectors: float32")     // [num_vectors, dim]
    .Output("bundle: float32")     // [dim]
    .SetShapeFn([](InferenceContext* c) {
        shape_inference::ShapeHandle input_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
        
        // Output is [dim]
        shape_inference::DimensionHandle dim = c->Dim(input_shape, 1);
        c->set_output(0, c->Vector(dim));
        return Status();
    })
    .Doc(R"doc(
Holographic bundling (superposition).
Compresses N vectors into a single holographic superposition state.

vectors: Input vectors [N, D]
bundle: Output superposition vector [D]
)doc");

class HolographicBundleOp : public OpKernel {
public:
    explicit HolographicBundleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
        const Tensor& vectors = ctx->input(0);
        OP_REQUIRES(ctx, vectors.dims() == 2,
            errors::InvalidArgument("vectors must be 2D [num_vectors, dim]"));

        const int num_vectors = vectors.dim_size(0);
        const int dim = vectors.dim_size(1);

        Tensor* bundle = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({dim}), &bundle));

        hsmn::qhpm::HolographicBundle(
            vectors.flat<float>().data(),
            num_vectors,
            bundle->flat<float>().data(),
            dim
        );
    }
};

REGISTER_KERNEL_BUILDER(Name("SAGUAROHolographicBundle").Device(DEVICE_CPU), HolographicBundleOp);


// =============================================================================
// Op: ModernHopfieldRetrieve
// =============================================================================

REGISTER_OP("ModernHopfieldRetrieve")
    .Input("query: float32")       // [batch, dim]
    .Input("memory: float32")      // [num_patterns, dim]
    .Output("retrieved: float32")  // [batch, dim]
    .Attr("beta: float = 1.0")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Modern Hopfield Network Retrieval.
Retrieves patterns from memory with exponential capacity.

query: Query vectors [batch, D]
memory: Memory patterns [M, D]
retrieved: Retrieved patterns [batch, D]
beta: Inverse temperature (higher = sharper)
)doc");

class ModernHopfieldRetrieveOp : public OpKernel {
public:
    explicit ModernHopfieldRetrieveOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& query = ctx->input(0);
        const Tensor& memory = ctx->input(1);

        OP_REQUIRES(ctx, query.dims() == 2, errors::InvalidArgument("query must be [batch, dim]"));
        OP_REQUIRES(ctx, memory.dims() == 2, errors::InvalidArgument("memory must be [patterns, dim]"));

        const int batch_size = query.dim_size(0);
        const int dim = query.dim_size(1);
        const int num_patterns = memory.dim_size(0);

        OP_REQUIRES(ctx, memory.dim_size(1) == dim,
            errors::InvalidArgument("memory dim mismatch"));

        Tensor* retrieved = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, query.shape(), &retrieved));

        const float* query_ptr = query.flat<float>().data();
        const float* memory_ptr = memory.flat<float>().data();
        float* out_ptr = retrieved->flat<float>().data();

        // Process batch (simple loop since core func is single query)
        // Or construct batched version. Core func is:
        // ModernHopfieldRetrieve(query, memory, num_patterns, result, dim, beta)
        // It takes single query.
        
        // Parallelize over batch
        auto worker = [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                hsmn::qhpm::ModernHopfieldRetrieve(
                    query_ptr + i * dim,
                    memory_ptr,
                    num_patterns,
                    out_ptr + i * dim,
                    dim,
                    beta_
                );
            }
        };
        
        // Use TF threadpool
        auto thread_pool = ctx->device()->tensorflow_cpu_worker_threads()->workers;
        thread_pool->ParallelFor(batch_size, 
            tensorflow::thread::ThreadPool::SchedulingParams(
                tensorflow::thread::ThreadPool::SchedulingStrategy::kAdaptive, 
                /*cost_per_unit=*/ num_patterns * dim * 2, // approximate cost
                /*block_size=*/ 1),
            worker);
    }

private:
    float beta_;
};

REGISTER_KERNEL_BUILDER(Name("ModernHopfieldRetrieve").Device(DEVICE_CPU), ModernHopfieldRetrieveOp);

// =============================================================================
// Op: CrystallizeMemory
// =============================================================================

REGISTER_OP("CrystallizeMemory")
    .Input("knowledge: float32")   // [batch, dim]
    .Input("importance: float32")  // [batch, dim]
    .Output("crystal: float32")    // [batch, dim]
    .Attr("threshold: float = 0.5")
    .SetShapeFn([](InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Crystallizes memory patterns based on importance.
Locks high-importance patterns and decays low-importance ones.

knowledge: Raw memory patterns.
importance: Importance weights.
crystal: Output crystallized memory.
)doc");

class CrystallizeMemoryOp : public OpKernel {
public:
    explicit CrystallizeMemoryOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("threshold", &threshold_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& knowledge = ctx->input(0);
        const Tensor& importance = ctx->input(1);

        OP_REQUIRES(ctx, knowledge.shape() == importance.shape(),
            errors::InvalidArgument("knowledge and importance shapes must match"));

        const int batch = knowledge.dim_size(0);
        const int dim = knowledge.dim_size(1);
        
        // Flattening for the kernel call which expects [batch * dim] effectively or handles indexing
        // The header sig: CrystallizeMemory(know, imp, out, threshold, batch, dim)
        
        Tensor* crystal = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, knowledge.shape(), &crystal));

        hsmn::qcrystal::CrystallizeMemory(
            knowledge.flat<float>().data(),
            importance.flat<float>().data(),
            crystal->flat<float>().data(),
            threshold_,
            batch,
            dim
        );
    }

private:
    float threshold_;
};

REGISTER_KERNEL_BUILDER(Name("CrystallizeMemory").Device(DEVICE_CPU), CrystallizeMemoryOp);

} // namespace tensorflow
