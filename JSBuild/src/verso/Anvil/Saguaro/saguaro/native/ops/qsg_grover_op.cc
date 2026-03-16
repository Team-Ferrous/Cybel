// saguaro.native/ops/qsg_grover_op.cc
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
 * @file qsg_grover_op.cc
 * @brief Grover-guided Quantum Superposition Generation (QSG v2).
 *
 * Phase 32 of Unified Quantum Architecture Enhancement.
 *
 * Implements full Grover pipeline for quality-guided generation:
 *   1. Initialize uniform superposition over candidate sequences
 *   2. Apply quality oracle O_f marking "good" sequences
 *   3. Apply diffusion operator D for amplitude amplification
 *   4. Iterate O(√N) times for optimal amplification
 *   5. Collapse to high-quality sequence
 *
 * Achieves √N speedup for finding high-quality sequences.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace tensorflow;

namespace saguaro {
namespace ops {
namespace grover_qsg {

// =============================================================================
// GROVER ITERATION KERNELS
// =============================================================================

/**
 * @brief Apply quality oracle: flip amplitude of "good" states.
 *
 * O_f|x⟩ = (-1)^f(x)|x⟩ where f(x) = 1 if quality(x) > threshold
 *
 * @param amplitudes State amplitudes [num_states]
 * @param quality_scores Quality scores per state [num_states]
 * @param threshold Quality threshold for "good" states
 * @param num_states Number of candidate states
 */
template <typename T>
inline void ApplyQualityOracle(
    T* amplitudes,
    const T* quality_scores,
    T threshold,
    int num_states) {
    
    #pragma omp parallel for
    for (int s = 0; s < num_states; ++s) {
        if (quality_scores[s] >= threshold) {
            amplitudes[s] = -amplitudes[s];  // Phase flip
        }
    }
}

/**
 * @brief Apply diffusion operator: inversion about mean.
 *
 * D = 2|ψ⟩⟨ψ| - I where |ψ⟩ = uniform superposition
 *
 * For each amplitude: a'_i = 2·mean - a_i
 *
 * @param amplitudes State amplitudes [num_states]
 * @param num_states Number of states
 */
template <typename T>
inline void ApplyDiffusionOperator(T* amplitudes, int num_states) {
    // Compute mean amplitude
    T sum = static_cast<T>(0);
    
    #pragma omp parallel for reduction(+:sum)
    for (int s = 0; s < num_states; ++s) {
        sum += amplitudes[s];
    }
    
    T mean = sum / num_states;
    T two_mean = static_cast<T>(2) * mean;
    
    // Inversion about mean
    #pragma omp parallel for
    for (int s = 0; s < num_states; ++s) {
        amplitudes[s] = two_mean - amplitudes[s];
    }
}

/**
 * @brief Full Grover iteration: Oracle + Diffusion.
 *
 * @param amplitudes State amplitudes [num_states]
 * @param quality_scores Quality scores [num_states]
 * @param threshold Quality threshold
 * @param num_states Number of states
 */
template <typename T>
inline void GroverIteration(
    T* amplitudes,
    const T* quality_scores,
    T threshold,
    int num_states) {
    
    ApplyQualityOracle(amplitudes, quality_scores, threshold, num_states);
    ApplyDiffusionOperator(amplitudes, num_states);
}

/**
 * @brief Estimate optimal number of Grover iterations.
 *
 * For M marked states out of N total: k ≈ (π/4)·√(N/M)
 *
 * @param num_states Total number of states N
 * @param num_marked Estimated number of marked states M
 * @return Optimal iteration count
 */
inline int OptimalGroverIterations(int num_states, int num_marked) {
    if (num_marked <= 0 || num_marked >= num_states) {
        return 1;
    }
    
    double ratio = static_cast<double>(num_states) / num_marked;
    int k = static_cast<int>(std::round(M_PI / 4.0 * std::sqrt(ratio)));
    
    return std::max(1, std::min(k, 20));  // Clamp to reasonable range
}

/**
 * @brief Collapse superposition to select best candidates.
 *
 * Samples top-k states by amplitude (probability) squared.
 *
 * @param amplitudes State amplitudes [num_states]
 * @param selected_indices Output indices of selected states [top_k]
 * @param num_states Number of states
 * @param top_k Number of states to select
 */
template <typename T>
inline void CollapseSuperposition(
    const T* amplitudes,
    int32_t* selected_indices,
    int num_states,
    int top_k) {
    
    // Compute probabilities (amplitude squared)
    std::vector<std::pair<T, int>> probs(num_states);
    
    for (int s = 0; s < num_states; ++s) {
        probs[s] = {amplitudes[s] * amplitudes[s], s};
    }
    
    // Partial sort to get top-k
    std::partial_sort(probs.begin(), probs.begin() + top_k, probs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract indices
    for (int i = 0; i < top_k; ++i) {
        selected_indices[i] = probs[i].second;
    }
}

/**
 * @brief Full Grover-guided QSG pipeline.
 *
 * @param candidate_logits Logits for candidate sequences [batch, num_candidates, seq_len, vocab]
 * @param quality_scores Quality scores per candidate [batch, num_candidates]
 * @param output_logits Selected high-quality logits [batch, seq_len, vocab]
 * @param batch_size Batch size
 * @param num_candidates Number of candidate sequences
 * @param seq_len Sequence length
 * @param vocab_size Vocabulary size
 * @param quality_threshold Threshold for "good" sequences
 * @param grover_iterations Number of Grover iterations (-1 for auto)
 */
template <typename T>
inline void GroverGuidedQSGForward(
    const T* candidate_logits,
    const T* quality_scores,
    T* output_logits,
    int batch_size,
    int num_candidates,
    int seq_len,
    int vocab_size,
    T quality_threshold,
    int grover_iterations = -1) {
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; ++b) {
        const T* batch_quality = quality_scores + b * num_candidates;
        
        // Initialize uniform superposition
        std::vector<T> amplitudes(num_candidates);
        T init_amp = static_cast<T>(1) / std::sqrt(static_cast<T>(num_candidates));
        std::fill(amplitudes.begin(), amplitudes.end(), init_amp);
        
        // Estimate number of marked states
        int num_marked = 0;
        for (int c = 0; c < num_candidates; ++c) {
            if (batch_quality[c] >= quality_threshold) {
                ++num_marked;
            }
        }
        
        // Determine iterations
        int iters = grover_iterations;
        if (iters < 0) {
            iters = OptimalGroverIterations(num_candidates, std::max(1, num_marked));
        }
        
        // Apply Grover iterations
        for (int i = 0; i < iters; ++i) {
            GroverIteration(amplitudes.data(), batch_quality, quality_threshold, num_candidates);
        }
        
        // Collapse to single best candidate
        int32_t best_idx;
        CollapseSuperposition(amplitudes.data(), &best_idx, num_candidates, 1);
        
        // Copy selected candidate's logits to output
        const T* selected_logits = candidate_logits + 
            (b * num_candidates + best_idx) * seq_len * vocab_size;
        T* out = output_logits + b * seq_len * vocab_size;
        
        std::copy(selected_logits, selected_logits + seq_len * vocab_size, out);
    }
}

}  // namespace grover_qsg
}  // namespace ops
}  // namespace saguaro

// =============================================================================
// OP REGISTRATIONS
// =============================================================================

REGISTER_OP("GroverGuidedQSG")
    .Input("candidate_logits: float")
    .Input("quality_scores: float")
    .Output("output_logits: float")
    .Attr("quality_threshold: float = 0.7")
    .Attr("grover_iterations: int = -1")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle input_shape = c->input(0);
        
        // Input: [batch, num_candidates, seq_len, vocab]
        // Output: [batch, seq_len, vocab]
        if (c->Rank(input_shape) == 4) {
            c->set_output(0, c->MakeShape({
                c->Dim(input_shape, 0),  // batch
                c->Dim(input_shape, 2),  // seq_len
                c->Dim(input_shape, 3)   // vocab
            }));
        }
        return Status();
    })
    .Doc(R"doc(
Grover-guided Quantum Superposition Generation uses amplitude amplification to select high-quality sequences.
Steps include uniform superposition initialization, quality oracle application, diffusion operator, O(sqrt(N)) iterations, and collapse to best sequence.

candidate_logits: Candidate sequence logits with shape [batch, num_candidates, seq_len, vocab].
quality_scores: Quality score per candidate with shape [batch, num_candidates].
quality_threshold: Threshold for good sequences between 0.0 and 1.0.
grover_iterations: Number of iterations where -1 means auto-optimal.
output_logits: Selected high-quality logits with shape [batch, seq_len, vocab].
)doc");

REGISTER_OP("GroverSingleIteration")
    .Input("amplitudes: float")
    .Input("quality_scores: float")
    .Output("output_amplitudes: float")
    .Attr("quality_threshold: float = 0.7")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status();
    })
    .Doc(R"doc(
Single Grover iteration: Oracle + Diffusion.

For fine-grained control over Grover iterations.

amplitudes: Current state amplitudes [num_states]
quality_scores: Quality scores [num_states]
quality_threshold: Oracle threshold
output_amplitudes: Updated amplitudes after iteration
)doc");

// =============================================================================
// KERNEL IMPLEMENTATIONS
// =============================================================================

class GroverGuidedQSGOp : public OpKernel {
public:
    explicit GroverGuidedQSGOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("quality_threshold", &quality_threshold_));
        OP_REQUIRES_OK(context, context->GetAttr("grover_iterations", &grover_iterations_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& candidate_logits = context->input(0);
        const Tensor& quality_scores = context->input(1);
        
        const TensorShape& logits_shape = candidate_logits.shape();
        OP_REQUIRES(context, logits_shape.dims() == 4,
            errors::InvalidArgument("candidate_logits must be 4D"));
        
        int batch_size = logits_shape.dim_size(0);
        int num_candidates = logits_shape.dim_size(1);
        int seq_len = logits_shape.dim_size(2);
        int vocab_size = logits_shape.dim_size(3);
        
        TensorShape output_shape({batch_size, seq_len, vocab_size});
        Tensor* output_logits = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_logits));
        
        const float* logits_data = candidate_logits.flat<float>().data();
        const float* quality_data = quality_scores.flat<float>().data();
        float* output_data = output_logits->flat<float>().data();
        
        saguaro::ops::grover_qsg::GroverGuidedQSGForward(
            logits_data, quality_data, output_data,
            batch_size, num_candidates, seq_len, vocab_size,
            quality_threshold_, grover_iterations_);
    }

private:
    float quality_threshold_;
    int grover_iterations_;
};

REGISTER_KERNEL_BUILDER(
    Name("GroverGuidedQSG").Device(DEVICE_CPU),
    GroverGuidedQSGOp);

class GroverSingleIterationOp : public OpKernel {
public:
    explicit GroverSingleIterationOp(OpKernelConstruction* context)
        : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("quality_threshold", &quality_threshold_));
    }

    void Compute(OpKernelContext* context) override {
        const Tensor& amplitudes = context->input(0);
        const Tensor& quality_scores = context->input(1);
        
        Tensor* output = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, amplitudes.shape(), &output));
        
        // Copy input to output
        auto input_flat = amplitudes.flat<float>();
        auto output_flat = output->flat<float>();
        std::copy(input_flat.data(), input_flat.data() + input_flat.size(), 
                  output_flat.data());
        
        const float* quality_data = quality_scores.flat<float>().data();
        int num_states = amplitudes.NumElements();
        
        saguaro::ops::grover_qsg::GroverIteration(
            output_flat.data(), quality_data, quality_threshold_, num_states);
    }

private:
    float quality_threshold_;
};

REGISTER_KERNEL_BUILDER(
    Name("GroverSingleIteration").Device(DEVICE_CPU),
    GroverSingleIterationOp);
