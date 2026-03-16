// saguaro.native/ops/fused_qsg_op.cc
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
 * @file fused_qsg_op.cc
 * @brief TensorFlow custom ops for Quantum Superposition Generation (QSG).
 *
 * Registers the following ops:
 *   - QSGEntangledCoherence: Bidirectional position coherence
 *   - QSGGroverAmplify: Grover-inspired amplitude amplification
 *   - QSGSemanticOracle: Semantic consistency scoring
 *   - QSGJacobiRefine: Local consistency refinement
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "qsg_ops.h"

using namespace tensorflow;

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("QSGEntangledCoherence")
    .Input("position_states: float")
    .Output("output: float")
    .Attr("coherence_range: int = -1")
    .Attr("temperature: float = 1.0")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Compute entangled bidirectional coherence between all position pairs.

Unlike standard attention, this allows each position to influence and be
influenced by ALL other positions simultaneously, including future positions.

position_states: Input states [batch, seq_len, dim] or [seq_len, dim]
coherence_range: Maximum distance for coherence (-1 = all pairs)
temperature: Softmax temperature for coherence weights
output: Updated states with same shape as input
)doc");

REGISTER_OP("QSGGroverAmplify")
    .Input("logits: float")
    .Input("oracle_scores: float")
    .Output("amplified: float")
    .Attr("iterations: int = 3")
    .Attr("amplification_strength: float = 1.5")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Grover-inspired amplitude amplification for token selection.

Amplifies "good" tokens (high oracle score) and suppresses "bad" tokens
through iterative reflection about the mean and oracle phase kicks.

logits: Input logits [batch, seq_len, vocab_size] or [seq_len, vocab_size]
oracle_scores: Semantic consistency scores [same shape as logits]
iterations: Number of Grover iterations (typically 2-4)
amplification_strength: How strongly to amplify good tokens (1.0-2.0)
amplified: Amplified logits with same shape as input
)doc");

REGISTER_OP("QSGSemanticOracle")
    .Input("vocab_embeddings: float")
    .Input("context_embedding: float")
    .Output("oracle_scores: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // Output shape: [seq_len, vocab_size]
        shape_inference::ShapeHandle vocab_shape = c->input(0);
        shape_inference::ShapeHandle context_shape = c->input(1);

        shape_inference::DimensionHandle vocab_size = c->Dim(vocab_shape, 0);
        shape_inference::DimensionHandle seq_len = c->Dim(context_shape, 0);

        c->set_output(0, c->MakeShape({seq_len, vocab_size}));
        return Status();
    })
    .Doc(R"doc(
Compute semantic consistency oracle scores for Grover amplification.

Evaluates cosine similarity between each vocabulary token embedding and
the context representation at each position.

vocab_embeddings: Vocabulary embedding matrix [vocab_size, dim]
context_embedding: Context representation [seq_len, dim] or [batch, seq_len, dim]
oracle_scores: Consistency scores [seq_len, vocab_size] in range [0, 1]
)doc");

REGISTER_OP("QSGJacobiRefine")
    .Input("token_logits: float")
    .Input("context_embedding: float")
    .Input("vocab_embeddings: float")
    .Output("refined: float")
    .Attr("iterations: int = 2")
    .Attr("neighbor_window: int = 3")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Jacobi fixed-point iteration for local consistency refinement.

After parallel generation, refines each position based on neighbor context
to fix local inconsistencies.

token_logits: Current logits [seq_len, vocab_size]
context_embedding: Context [seq_len, dim]
vocab_embeddings: Vocabulary [vocab_size, dim]
iterations: Number of refinement iterations
neighbor_window: Size of neighbor window for averaging
refined: Refined logits with same shape as token_logits
)doc");

// [NEW] Phase A1: MPS Context Entanglement
REGISTER_OP("QSGMPSContextEntangle")
    .Input("embeddings: float")
    .Input("site_weights: float")
    .Output("context: float")
    .Output("entropy: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        // embeddings: [batch, seq_len, dim]
        // site_weights: [batch, seq_len, bond_dim, phys_dim, bond_dim]
        // output context: [batch, seq_len, dim]
        // output entropy: [batch, seq_len - 1]
        
        shape_inference::ShapeHandle embeddings;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &embeddings));
        
        shape_inference::ShapeHandle sites;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 5, &sites));
        
        // Output 0: Context has same shape as embeddings
        c->set_output(0, embeddings);
        
        // Output 1: Entropy has [batch, seq_len - 1]
        shape_inference::DimensionHandle batch = c->Dim(embeddings, 0);
        shape_inference::DimensionHandle seq_len = c->Dim(embeddings, 1);
        
        shape_inference::DimensionHandle seq_minus_1;
        TF_RETURN_IF_ERROR(c->Subtract(seq_len, 1, &seq_minus_1));
        
        c->set_output(1, c->MakeShape({batch, seq_minus_1}));
        
        return Status();
    })
    .Doc(R"doc(
Compute MPS Context Entanglement with O(N·chi²) complexity.

Embeds long-range dependencies by contracting a Matrix Product State
sweeping across the sequence.

embeddings: Input embeddings [batch, seq_len, dim]
site_weights: MPS site tensors [batch, seq_len, bond_dim, phys_dim, bond_dim]
context: Entangled context representation [batch, seq_len, dim]
entropy: Bond entanglement entropy [batch, seq_len - 1]
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class QSGEntangledCoherenceOp : public OpKernel {
public:
    explicit QSGEntangledCoherenceOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("coherence_range", &coherence_range_));
        OP_REQUIRES_OK(context, context->GetAttr("temperature", &temperature_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        const TensorShape& shape = input.shape();

        OP_REQUIRES(context, shape.dims() >= 2,
            errors::InvalidArgument("position_states must be at least 2D"));

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

        const float* input_data = input.flat<float>().data();
        float* output_data = output->flat<float>().data();

        if (shape.dims() == 2) {
            // [seq_len, dim]
            int seq_len = shape.dim_size(0);
            int dim = shape.dim_size(1);
            saguaro::ops::qsg::EntangledPositionCoherence(
                input_data, output_data, seq_len, dim,
                coherence_range_, temperature_);
        } else if (shape.dims() == 3) {
            // [batch, seq_len, dim]
            int batch_size = shape.dim_size(0);
            int seq_len = shape.dim_size(1);
            int dim = shape.dim_size(2);
            saguaro::ops::qsg::BatchEntangledCoherence(
                input_data, output_data, batch_size, seq_len, dim,
                coherence_range_);
        }
    }

private:
    int coherence_range_;
    float temperature_;
};

REGISTER_KERNEL_BUILDER(
    Name("QSGEntangledCoherence").Device(DEVICE_CPU),
    QSGEntangledCoherenceOp);

class QSGGroverAmplifyOp : public OpKernel {
public:
    explicit QSGGroverAmplifyOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("iterations", &iterations_));
        OP_REQUIRES_OK(context, context->GetAttr("amplification_strength", &amp_strength_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& logits = context->input(0);
        const Tensor& oracle = context->input(1);
        const TensorShape& shape = logits.shape();

        OP_REQUIRES(context, shape.dims() >= 2,
            errors::InvalidArgument("logits must be at least 2D"));
        OP_REQUIRES(context, oracle.shape() == shape,
            errors::InvalidArgument("oracle_scores must match logits shape"));

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

        const float* logits_data = logits.flat<float>().data();
        const float* oracle_data = oracle.flat<float>().data();
        float* output_data = output->flat<float>().data();

        if (shape.dims() == 2) {
            // [seq_len, vocab_size]
            int seq_len = shape.dim_size(0);
            int vocab_size = shape.dim_size(1);
            saguaro::ops::qsg::GroverAmplitudeAmplify(
                logits_data, oracle_data, output_data,
                seq_len, vocab_size, iterations_, amp_strength_);
        } else if (shape.dims() == 3) {
            // [batch, seq_len, vocab_size]
            int batch_size = shape.dim_size(0);
            int seq_len = shape.dim_size(1);
            int vocab_size = shape.dim_size(2);
            saguaro::ops::qsg::BatchGroverAmplify(
                logits_data, oracle_data, output_data,
                batch_size, seq_len, vocab_size, iterations_);
        }
    }

private:
    int iterations_;
    float amp_strength_;
};

REGISTER_KERNEL_BUILDER(
    Name("QSGGroverAmplify").Device(DEVICE_CPU),
    QSGGroverAmplifyOp);

class QSGSemanticOracleOp : public OpKernel {
public:
    explicit QSGSemanticOracleOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& vocab_emb = context->input(0);
        const Tensor& context_emb = context->input(1);

        OP_REQUIRES(context, vocab_emb.dims() == 2,
            errors::InvalidArgument("vocab_embeddings must be 2D [vocab, dim]"));
        OP_REQUIRES(context, context_emb.dims() >= 2,
            errors::InvalidArgument("context_embedding must be at least 2D"));

        int vocab_size = vocab_emb.dim_size(0);
        int dim = vocab_emb.dim_size(1);
        int seq_len = context_emb.dim_size(context_emb.dims() == 3 ? 1 : 0);

        OP_REQUIRES(context, context_emb.dim_size(context_emb.dims() - 1) == dim,
            errors::InvalidArgument("Dimension mismatch between vocab and context"));

        // Output shape
        TensorShape output_shape;
        if (context_emb.dims() == 3) {
            int batch = context_emb.dim_size(0);
            output_shape = TensorShape({batch, seq_len, vocab_size});
        } else {
            output_shape = TensorShape({seq_len, vocab_size});
        }

        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

        const float* vocab_data = vocab_emb.flat<float>().data();
        const float* ctx_data = context_emb.flat<float>().data();
        float* output_data = output->flat<float>().data();

        if (context_emb.dims() == 2) {
            saguaro::ops::qsg::SemanticConsistencyOracle(
                vocab_data, ctx_data, output_data,
                seq_len, vocab_size, dim);
        } else {
            // Batch processing
            int batch = context_emb.dim_size(0);
            size_t ctx_stride = seq_len * dim;
            size_t out_stride = seq_len * vocab_size;

            #pragma omp parallel for
            for (int b = 0; b < batch; ++b) {
                saguaro::ops::qsg::SemanticConsistencyOracle(
                    vocab_data,
                    ctx_data + b * ctx_stride,
                    output_data + b * out_stride,
                    seq_len, vocab_size, dim);
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
    Name("QSGSemanticOracle").Device(DEVICE_CPU),
    QSGSemanticOracleOp);

class QSGJacobiRefineOp : public OpKernel {
public:
    explicit QSGJacobiRefineOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("iterations", &iterations_));
        OP_REQUIRES_OK(context, context->GetAttr("neighbor_window", &window_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& logits = context->input(0);
        const Tensor& ctx_emb = context->input(1);
        const Tensor& vocab_emb = context->input(2);

        const TensorShape& shape = logits.shape();

        OP_REQUIRES(context, shape.dims() == 2 || shape.dims() == 3,
            errors::InvalidArgument(
                "token_logits must be 2D [seq_len, vocab_size] or "
                "3D [batch, seq_len, vocab_size]"));

        int dim = vocab_emb.dim_size(1);
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, shape, &output));

        if (shape.dims() == 2) {
            // [seq_len, vocab_size]
            int seq_len = shape.dim_size(0);
            int vocab_size = shape.dim_size(1);

            saguaro::ops::qsg::JacobiRefine(
                logits.flat<float>().data(),
                ctx_emb.flat<float>().data(),
                vocab_emb.flat<float>().data(),
                output->flat<float>().data(),
                seq_len, vocab_size, dim,
                iterations_, window_);
        } else if (shape.dims() == 3) {
            // [batch, seq_len, vocab_size]
            int batch_size = shape.dim_size(0);
            int seq_len = shape.dim_size(1);
            int vocab_size = shape.dim_size(2);

            saguaro::ops::qsg::BatchJacobiRefine(
                logits.flat<float>().data(),
                ctx_emb.flat<float>().data(),
                vocab_emb.flat<float>().data(),
                output->flat<float>().data(),
                batch_size, seq_len, vocab_size, dim,
                iterations_, window_);
        }
    }

private:
    int iterations_;
    int window_;
};

REGISTER_KERNEL_BUILDER(
    Name("QSGJacobiRefine").Device(DEVICE_CPU),
    QSGJacobiRefineOp);

// [NEW] Phase A1: MPS Context Entangle Kernel
#include "qsg_mps_entangle.h"

class QSGMPSContextEntangleOp : public OpKernel {
public:
    explicit QSGMPSContextEntangleOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& embeddings = context->input(0);
        const Tensor& site_weights = context->input(1);
        
        const TensorShape& emb_shape = embeddings.shape();
        const TensorShape& site_shape = site_weights.shape();
        
        // Validate shapes
        OP_REQUIRES(context, emb_shape.dims() == 3,
            errors::InvalidArgument("embeddings must be 3D [batch, seq, dim]"));
        OP_REQUIRES(context, site_shape.dims() == 5,
            errors::InvalidArgument(
                "site_weights must be 5D [batch, seq, bond, phys, bond]"));
        
        int batch_size = emb_shape.dim_size(0);
        int seq_len = emb_shape.dim_size(1);
        int embedding_dim = emb_shape.dim_size(2);
        
        int bond_dim = site_shape.dim_size(2);
        int phys_dim = site_shape.dim_size(3);
        
        OP_REQUIRES(context, site_shape.dim_size(4) == bond_dim,
           errors::InvalidArgument("site_weights bond dimensions must match"));
        OP_REQUIRES(context, site_shape.dim_size(0) == batch_size,
           errors::InvalidArgument("Batch size mismatch"));
        OP_REQUIRES(context, site_shape.dim_size(1) == seq_len,
           errors::InvalidArgument("Sequence length mismatch"));

        // Alloc outputs
        Tensor* context_out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, emb_shape, &context_out));
        
        Tensor* entropy_out = nullptr;
        TensorShape entropy_shape({batch_size, seq_len - 1});
        if (seq_len <= 1) entropy_shape = TensorShape({batch_size, 0});
        OP_REQUIRES_OK(context, context->allocate_output(1, entropy_shape, &entropy_out));
        
        // Call Kernel
        saguaro::ops::qsg_mps_context_entangle(
            embeddings.flat<float>().data(),
            site_weights.flat<float>().data(),
            context_out->flat<float>().data(),
            (seq_len > 1) ? entropy_out->flat<float>().data() : nullptr,
            batch_size, seq_len, embedding_dim, bond_dim, phys_dim
        );
    }
};

REGISTER_KERNEL_BUILDER(
    Name("QSGMPSContextEntangle").Device(DEVICE_CPU),
    QSGMPSContextEntangleOp);

// =============================================================================
// [NEW] Phase A1: MPS Context Entangle GRADIENT Op (for training)
// =============================================================================
// Gradient for context output dL/d_embeddings, dL/d_site_weights
// Note: The MPS contraction is O[d] = E[d] + state[d % chi], so gradient is straightforward:
//   dL/dE = dL/dO (identity pass-through for embedding component)
//   dL/d_site_weights propagates through the MPS contraction

REGISTER_OP("QSGMPSContextEntangleGrad")
    .Input("grad_context: float")    // dL/d_context [batch, seq, dim]
    .Input("grad_entropy: float")    // dL/d_entropy [batch, seq-1] (typically zeros, entropy is for info)
    .Input("embeddings: float")      // Original input [batch, seq, dim]
    .Input("site_weights: float")    // Original site tensors [batch, seq, bond, phys, bond]
    .Output("grad_embeddings: float")    // [batch, seq, dim]
    .Output("grad_site_weights: float")  // [batch, seq, bond, phys, bond]
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(2));  // grad_embeddings shape = embeddings
        c->set_output(1, c->input(3));  // grad_site_weights shape = site_weights
        return Status();
    })
    .Doc(R"doc(
Gradient for QSGMPSContextEntangle.

Computes gradients for embeddings and site_weights given upstream gradients.
)doc");

class QSGMPSContextEntangleGradOp : public OpKernel {
public:
    explicit QSGMPSContextEntangleGradOp(OpKernelConstruction* context)
        : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_context = context->input(0);
        const Tensor& grad_entropy = context->input(1);
        const Tensor& embeddings = context->input(2);
        const Tensor& site_weights = context->input(3);
        
        const TensorShape& emb_shape = embeddings.shape();
        const TensorShape& site_shape = site_weights.shape();
        
        int batch_size = emb_shape.dim_size(0);
        int seq_len = emb_shape.dim_size(1);
        int embedding_dim = emb_shape.dim_size(2);
        int bond_dim = site_shape.dim_size(2);
        int phys_dim = site_shape.dim_size(3);
        
        // Allocate gradient outputs
        Tensor* grad_emb = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, emb_shape, &grad_emb));
        
        Tensor* grad_sites = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, site_shape, &grad_sites));
        
        // Get data pointers
        const float* g_ctx = grad_context.flat<float>().data();
        const float* emb = embeddings.flat<float>().data();
        const float* sites = site_weights.flat<float>().data();
        float* g_emb = grad_emb->flat<float>().data();
        float* g_sites = grad_sites->flat<float>().data();
        
        // Initialize gradients to zero
        std::fill(g_emb, g_emb + emb_shape.num_elements(), 0.0f);
        std::fill(g_sites, g_sites + site_shape.num_elements(), 0.0f);
        
        // The forward pass: context[d] = embedding[d] + state[d % chi]
        // So: dL/d_embedding[d] = dL/d_context[d]  (direct pass-through)
        // And: dL/d_state[k] = sum over d where d%chi==k of dL/d_context[d]
        
        // For site_weights gradient, we need the chain rule through MPS contraction
        // This is more complex. For efficiency, we use a simplified approximation:
        // dL/d_site[t,i,j,k] ≈ state[i] * dL/d_state_next[k] * (contribution weight)
        
        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            std::vector<float> state(bond_dim, 0.0f);
            std::vector<float> grad_state(bond_dim, 0.0f);
            state[0] = 1.0f;  // Initial state
            
            const size_t emb_batch_offset = (size_t)b * seq_len * embedding_dim;
            const size_t site_batch_offset = (size_t)b * seq_len * bond_dim * phys_dim * bond_dim;
            
            // Forward pass to recompute states (needed for gradient)
            std::vector<std::vector<float>> saved_states(seq_len);
            for (int t = 0; t < seq_len; ++t) {
                saved_states[t] = state;
                
                // Recompute forward step
                std::vector<float> next_state(bond_dim, 0.0f);
                const float* A_t = sites + site_batch_offset + (size_t)t * bond_dim * phys_dim * bond_dim;
                
                for (int i = 0; i < bond_dim; ++i) {
                    float s_val = state[i];
                    if (std::abs(s_val) < 1e-9f) continue;
                    for (int j = 0; j < phys_dim; ++j) {
                        const float* A_slice = A_t + (i * phys_dim + j) * bond_dim;
                        for (int k = 0; k < bond_dim; ++k) {
                            next_state[k] += s_val * A_slice[k];
                        }
                    }
                }
                
                // Normalize
                float norm_sq = 0.0f;
                for (float v : next_state) norm_sq += v * v;
                if (norm_sq > 1e-9f) {
                    float inv_norm = 1.0f / std::sqrt(norm_sq);
                    for (int k = 0; k < bond_dim; ++k) next_state[k] *= inv_norm;
                }
                state = next_state;
            }
            
            // Backward pass
            std::fill(grad_state.begin(), grad_state.end(), 0.0f);
            
            for (int t = seq_len - 1; t >= 0; --t) {
                const float* g_ctx_t = g_ctx + emb_batch_offset + t * embedding_dim;
                float* g_emb_t = g_emb + emb_batch_offset + t * embedding_dim;
                
                // Gradient from context output: dL/d_emb = dL/d_ctx (direct term)
                for (int d = 0; d < embedding_dim; ++d) {
                    g_emb_t[d] += g_ctx_t[d];  // Residual connection gradient
                    // Accumulate gradient to state
                    grad_state[d % bond_dim] += g_ctx_t[d];
                }
                
                // Backprop through state normalization and contraction
                // Simplified: propagate grad_state to site_weights
                if (t > 0) {
                    const std::vector<float>& prev_state = saved_states[t];
                    float* g_A_t = g_sites + site_batch_offset + (size_t)t * bond_dim * phys_dim * bond_dim;
                    
                    for (int i = 0; i < bond_dim; ++i) {
                        float s_val = prev_state[i];
                        if (std::abs(s_val) < 1e-9f) continue;
                        for (int j = 0; j < phys_dim; ++j) {
                            float* g_A_slice = g_A_t + (i * phys_dim + j) * bond_dim;
                            for (int k = 0; k < bond_dim; ++k) {
                                g_A_slice[k] += s_val * grad_state[k];
                            }
                        }
                    }
                }
                
                // Carry gradient to previous time step (simplified)
                // In full implementation, this would be more involved
            }
        }
    }
};

REGISTER_KERNEL_BUILDER(
    Name("QSGMPSContextEntangleGrad").Device(DEVICE_CPU),
    QSGMPSContextEntangleGradOp);

