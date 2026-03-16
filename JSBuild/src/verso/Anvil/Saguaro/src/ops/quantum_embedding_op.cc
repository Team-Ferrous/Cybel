// saguaro/native/ops/quantum_embedding_op.cc
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
 * @file quantum_embedding_op.cc
 * @brief TensorFlow custom ops for quantum-enhanced embeddings.
 *
 * Phase 26 of Unified Quantum Architecture Enhancement.
 *
 * Registers the following ops:
 *   - QuantumEmbeddingForward: Holographic unbind for embedding lookup
 *   - QuantumEmbeddingBackward: Gradient via holographic bind
 *   - HaarRandomKeyInit: Initialize orthogonal token keys
 *   - InitHolographicStore: Bundle standard embeddings into holographic form
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "quantum_embedding_op.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("QuantumEmbeddingForward")
    .Input("token_ids: int32")
    .Input("holographic_store: float")
    .Input("token_keys: float")
    .Output("output: float")
    .Attr("vocab_size: int")
    .Attr("dim: int")
    .Attr("num_bundles: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int dim;
        TF_RETURN_IF_ERROR(c->GetAttr("dim", &dim));
        
        shape_inference::ShapeHandle input_shape = c->input(0);
        
        // Output shape: input_shape + [dim]
        if (c->Rank(input_shape) == 1) {
            // [seq_len] -> [seq_len, dim]
            c->set_output(0, c->MakeShape({c->Dim(input_shape, 0), dim}));
        } else if (c->Rank(input_shape) == 2) {
            // [batch, seq_len] -> [batch, seq_len, dim]
            c->set_output(0, c->MakeShape({
                c->Dim(input_shape, 0),
                c->Dim(input_shape, 1),
                dim
            }));
        }
        return Status();
    })
    .Doc(R"doc(
Quantum-enhanced embedding lookup via holographic unbinding.

Uses FFT-based circular correlation to retrieve embeddings from
a bundled holographic representation.

token_ids: Token IDs [batch, seq_len] or [seq_len]
holographic_store: Bundled representations [num_bundles, dim]
token_keys: Token-specific orthogonal keys [vocab_size, dim]
vocab_size: Vocabulary size
dim: Embedding dimension (should be power of 2)
num_bundles: Number of holographic bundles
output: Embeddings [batch, seq_len, dim] or [seq_len, dim]
)doc");

REGISTER_OP("QuantumEmbeddingBackward")
    .Input("grad_output: float")
    .Input("token_ids: int32")
    .Input("token_keys: float")
    .Input("holographic_store: float")  // Added for dynamic shape inference
    .Output("grad_store: float")
    .Attr("vocab_size: int")
    .Attr("dim: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Infer output shape from holographic_store input (index 3)
        // This enables dynamic num_bundles instead of hardcoded value
        shape_inference::ShapeHandle store_shape = c->input(3);
        c->set_output(0, store_shape);
        return Status();
    })
    .Doc(R"doc(
Backward pass for quantum embedding.

Gradients flow back to holographic store via binding operation.
Shape is inferred dynamically from holographic_store input.

grad_output: Gradient w.r.t. output embeddings
token_ids: Token IDs from forward pass
token_keys: Token keys
holographic_store: Holographic store tensor for shape inference [num_bundles, dim]
vocab_size: Vocabulary size
dim: Embedding dimension
grad_store: Gradient w.r.t. holographic store (Patched with Clamping)
)doc");

REGISTER_OP("HaarRandomKeyInit")
    .Input("shape: int32")
    .Output("keys: float")
    .Attr("seed: int = 42")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape from input shape tensor
        c->set_output(0, c->UnknownShape());
        return Status();
    })
    .Doc(R"doc(
Initialize Haar-random orthogonal token keys.

Generates unit vectors distributed uniformly on the hypersphere.

shape: Shape tensor [vocab_size, dim]
seed: Random seed
keys: Orthogonal keys [vocab_size, dim]
)doc");

REGISTER_OP("InitHolographicStore")
    .Input("embeddings: float")
    .Input("token_keys: float")
    .Output("store: float")
    .Attr("num_bundles: int = 4")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        int num_bundles;
        TF_RETURN_IF_ERROR(c->GetAttr("num_bundles", &num_bundles));
        
        shape_inference::ShapeHandle emb_shape = c->input(0);
        shape_inference::DimensionHandle dim = c->Dim(emb_shape, 1);
        
        c->set_output(0, c->MakeShape({num_bundles, dim}));
        return Status();
    })
    .Doc(R"doc(
Initialize holographic store from standard embeddings.

Binds each embedding with its key and bundles them.

embeddings: Standard embeddings [vocab_size, dim]
token_keys: Token keys [vocab_size, dim]
num_bundles: Number of bundles
store: Holographic store [num_bundles, dim]
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class QuantumEmbeddingForwardOp : public OpKernel {
public:
    explicit QuantumEmbeddingForwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
        OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
        OP_REQUIRES_OK(context, context->GetAttr("num_bundles", &num_bundles_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& token_ids = context->input(0);
        const Tensor& holographic_store = context->input(1);
        const Tensor& token_keys = context->input(2);
        
        const TensorShape& ids_shape = token_ids.shape();
        int batch_size = 1;
        int seq_len = ids_shape.dim_size(0);
        
        if (ids_shape.dims() == 2) {
            batch_size = ids_shape.dim_size(0);
            seq_len = ids_shape.dim_size(1);
        }
        
        TensorShape output_shape;
        if (ids_shape.dims() == 1) {
            output_shape = TensorShape({seq_len, dim_});
        } else {
            output_shape = TensorShape({batch_size, seq_len, dim_});
        }
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
        
        const int32_t* ids_data = token_ids.flat<int32_t>().data();
        const float* store_data = holographic_store.flat<float>().data();
        const float* keys_data = token_keys.flat<float>().data();
        float* output_data = output->flat<float>().data();
        
        saguaro::ops::quantum_embedding::QuantumEmbeddingForward(
            ids_data, store_data, keys_data, output_data,
            batch_size, seq_len, vocab_size_, dim_, num_bundles_);
    }

private:
    int vocab_size_;
    int dim_;
    int num_bundles_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumEmbeddingForward").Device(DEVICE_CPU),
    QuantumEmbeddingForwardOp);

class QuantumEmbeddingBackwardOp : public OpKernel {
public:
    explicit QuantumEmbeddingBackwardOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("vocab_size", &vocab_size_));
        OP_REQUIRES_OK(context, context->GetAttr("dim", &dim_));
        // num_bundles is now inferred from holographic_store input, not an attribute
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_output = context->input(0);
        const Tensor& token_ids = context->input(1);
        const Tensor& token_keys = context->input(2);
        const Tensor& holographic_store = context->input(3);  // New input for shape
        
        // Infer num_bundles from holographic_store shape
        const TensorShape& store_shape = holographic_store.shape();
        OP_REQUIRES(context, store_shape.dims() == 2,
            errors::InvalidArgument("holographic_store must be 2D [num_bundles, dim]"));
        const int num_bundles = store_shape.dim_size(0);
        const int dim_from_store = store_shape.dim_size(1);
        
        const TensorShape& grad_shape = grad_output.shape();
        int batch_size = 1;
        int seq_len;
        
        if (grad_shape.dims() == 2) {
            seq_len = grad_shape.dim_size(0);
        } else {
            batch_size = grad_shape.dim_size(0);
            seq_len = grad_shape.dim_size(1);
        }
        
        Tensor* grad_store = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
            TensorShape({num_bundles, dim_from_store}), &grad_store));
        
        const float* grad_out_data = grad_output.flat<float>().data();
        const int32_t* ids_data = token_ids.flat<int32_t>().data();
        const float* keys_data = token_keys.flat<float>().data();
        float* grad_store_data = grad_store->flat<float>().data();
        
        saguaro::ops::quantum_embedding::QuantumEmbeddingBackward(
            grad_out_data, ids_data, keys_data, grad_store_data,
            batch_size, seq_len, vocab_size_, dim_from_store, num_bundles);
    }

private:
    int vocab_size_;
    int dim_;
};

REGISTER_KERNEL_BUILDER(
    Name("QuantumEmbeddingBackward").Device(DEVICE_CPU),
    QuantumEmbeddingBackwardOp);

class HaarRandomKeyInitOp : public OpKernel {
public:
    explicit HaarRandomKeyInitOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& shape_tensor = context->input(0);
        
        OP_REQUIRES(context, shape_tensor.dims() == 1,
            errors::InvalidArgument("Shape must be 1D"));
        OP_REQUIRES(context, shape_tensor.dim_size(0) == 2,
            errors::InvalidArgument("Shape must have 2 elements [vocab_size, dim]"));
        
        auto shape_vec = shape_tensor.flat<int32_t>();
        int vocab_size = shape_vec(0);
        int dim = shape_vec(1);
        
        TensorShape output_shape({vocab_size, dim});
        Tensor* keys = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &keys));
        
        float* keys_data = keys->flat<float>().data();
        
        saguaro::ops::quantum_embedding::InitHaarRandomKeys(
            keys_data, vocab_size, dim, static_cast<uint64_t>(seed_));
    }

private:
    int seed_;
};

REGISTER_KERNEL_BUILDER(
    Name("HaarRandomKeyInit").Device(DEVICE_CPU),
    HaarRandomKeyInitOp);

class InitHolographicStoreOp : public OpKernel {
public:
    explicit InitHolographicStoreOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("num_bundles", &num_bundles_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& embeddings = context->input(0);
        const Tensor& token_keys = context->input(1);
        
        OP_REQUIRES(context, embeddings.dims() == 2,
            errors::InvalidArgument("Embeddings must be 2D [vocab_size, dim]"));
        
        int vocab_size = embeddings.dim_size(0);
        int dim = embeddings.dim_size(1);
        
        Tensor* store = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, 
            TensorShape({num_bundles_, dim}), &store));
        
        const float* emb_data = embeddings.flat<float>().data();
        const float* keys_data = token_keys.flat<float>().data();
        float* store_data = store->flat<float>().data();
        
        saguaro::ops::quantum_embedding::InitHolographicStore(
            emb_data, keys_data, store_data, vocab_size, dim, num_bundles_);
    }

private:
    int num_bundles_;
};

REGISTER_KERNEL_BUILDER(
    Name("InitHolographicStore").Device(DEVICE_CPU),
    InitHolographicStoreOp);
