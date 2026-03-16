// saguaro.native/ops/tensor_stream_pool_op.cc
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file tensor_stream_pool_op.cc
 * @brief TensorFlow custom ops exposing TensorStreamPool functionality.
 *
 * These ops provide Python-accessible interface to the zero-copy
 * inter-kernel streaming pool for testing and telemetry.
 *
 * Ops registered:
 * - TensorStreamAcquire: Acquire buffer from pool
 * - TensorStreamHandoff: Mark buffer ready for consumer
 * - TensorStreamRelease: Return buffer to pool
 * - TensorStreamGetStats: Get statistics dictionary
 * - TensorStreamClear: Clear all buffers
 *
 * Part of TensorStreamPool C++ Enhancement Roadmap Phase 0.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "common/tensor_stream_pool.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATION
// =============================================================================

REGISTER_OP("TensorStreamAcquire")
    .Input("size_bytes: int64")
    .Attr("producer_hint: string = ''")
    .Output("buffer_ptr: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Scalar());
        return OkStatus();
    })
    .Doc(R"doc(
Acquire a buffer from the TensorStreamPool.

Returns an int64 representing the buffer pointer address.
The buffer is aligned for optimal SIMD performance (64-byte alignment).
Pass size_bytes for the buffer size and optional producer_hint for debugging.
)doc");

REGISTER_OP("TensorStreamHandoff")
    .Input("buffer_ptr: int64")
    .Attr("consumer_hint: string = ''")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        return OkStatus();
    })
    .SetIsStateful()  // Has side effects - must be executed
    .Doc(R"doc(
Mark a buffer as ready for handoff to consumer.

This signals that the producer has finished writing and the buffer
is ready for zero-copy consumption by the next kernel.
Pass buffer_ptr from TensorStreamAcquire and optional consumer_hint.
)doc");

REGISTER_OP("TensorStreamRelease")
    .Input("buffer_ptr: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        return OkStatus();
    })
    .SetIsStateful()  // Has side effects - must be executed
    .Doc(R"doc(
Release a buffer back to the pool for reuse.

The buffer remains allocated but is marked available for future
TensorStreamAcquire calls of compatible size.
Pass buffer_ptr from TensorStreamAcquire.
)doc");

REGISTER_OP("TensorStreamGetStats")
    .Output("total_allocated_bytes: int64")
    .Output("num_buffers: int64")
    .Output("acquire_count: int64")
    .Output("reuse_count: int64")
    .Output("zero_copy_handoffs: int64")
    .Output("release_count: int64")
    .Output("peak_usage_bytes: int64")
    .Output("current_usage_bytes: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        for (int i = 0; i < 8; ++i) {
            c->set_output(i, c->Scalar());
        }
        return OkStatus();
    })
    .SetIsStateful()  // Reads mutable global state - must be re-executed each time
    .Doc(R"doc(
Get streaming statistics from TensorStreamPool.

Returns multiple scalar outputs with telemetry data.
)doc");

REGISTER_OP("TensorStreamClear")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        return OkStatus();
    })
    .SetIsStateful()  // Has side effects - must be executed
    .Doc(R"doc(
Clear all buffers from the pool, freeing memory.

Note that this invalidates all previously acquired pointers!
)doc");

// =============================================================================
// OP KERNELS
// =============================================================================

class TensorStreamAcquireOp : public OpKernel {
public:
    explicit TensorStreamAcquireOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("producer_hint", &producer_hint_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& size_tensor = context->input(0);
        int64_t size_bytes = size_tensor.scalar<int64_t>()();
        
        const char* hint = producer_hint_.empty() ? nullptr : producer_hint_.c_str();
        float* ptr = saguaro::ops::GetTensorStreamPool().Acquire(
            static_cast<size_t>(size_bytes), hint
        );
        
        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({}), &output));
        output->scalar<int64_t>()() = reinterpret_cast<int64_t>(ptr);
    }

private:
    std::string producer_hint_;
};

class TensorStreamHandoffOp : public OpKernel {
public:
    explicit TensorStreamHandoffOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("consumer_hint", &consumer_hint_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& ptr_tensor = context->input(0);
        int64_t ptr_int = ptr_tensor.scalar<int64_t>()();
        float* ptr = reinterpret_cast<float*>(ptr_int);
        
        const char* hint = consumer_hint_.empty() ? nullptr : consumer_hint_.c_str();
        saguaro::ops::GetTensorStreamPool().Handoff(ptr, hint);
    }

private:
    std::string consumer_hint_;
};

class TensorStreamReleaseOp : public OpKernel {
public:
    explicit TensorStreamReleaseOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& ptr_tensor = context->input(0);
        int64_t ptr_int = ptr_tensor.scalar<int64_t>()();
        float* ptr = reinterpret_cast<float*>(ptr_int);
        
        saguaro::ops::GetTensorStreamPool().Release(ptr);
    }
};

class TensorStreamGetStatsOp : public OpKernel {
public:
    explicit TensorStreamGetStatsOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        auto stats = saguaro::ops::GetTensorStreamPool().GetStats();
        
        // Allocate all outputs
        for (int i = 0; i < 8; ++i) {
            Tensor* output;
            OP_REQUIRES_OK(context, context->allocate_output(i, TensorShape({}), &output));
            
            switch (i) {
                case 0: output->scalar<int64_t>()() = static_cast<int64_t>(stats.total_allocated_bytes); break;
                case 1: output->scalar<int64_t>()() = static_cast<int64_t>(stats.num_buffers); break;
                case 2: output->scalar<int64_t>()() = static_cast<int64_t>(stats.acquire_count); break;
                case 3: output->scalar<int64_t>()() = static_cast<int64_t>(stats.reuse_count); break;
                case 4: output->scalar<int64_t>()() = static_cast<int64_t>(stats.zero_copy_handoffs); break;
                case 5: output->scalar<int64_t>()() = static_cast<int64_t>(stats.release_count); break;
                case 6: output->scalar<int64_t>()() = static_cast<int64_t>(stats.peak_usage_bytes); break;
                case 7: output->scalar<int64_t>()() = static_cast<int64_t>(stats.current_usage_bytes); break;
            }
        }
    }
};

class TensorStreamClearOp : public OpKernel {
public:
    explicit TensorStreamClearOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        saguaro::ops::GetTensorStreamPool().Clear();
    }
};

// =============================================================================
// KERNEL REGISTRATION
// =============================================================================

REGISTER_KERNEL_BUILDER(Name("TensorStreamAcquire").Device(DEVICE_CPU), TensorStreamAcquireOp);
REGISTER_KERNEL_BUILDER(Name("TensorStreamHandoff").Device(DEVICE_CPU), TensorStreamHandoffOp);
REGISTER_KERNEL_BUILDER(Name("TensorStreamRelease").Device(DEVICE_CPU), TensorStreamReleaseOp);
REGISTER_KERNEL_BUILDER(Name("TensorStreamGetStats").Device(DEVICE_CPU), TensorStreamGetStatsOp);
REGISTER_KERNEL_BUILDER(Name("TensorStreamClear").Device(DEVICE_CPU), TensorStreamClearOp);
