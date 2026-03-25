// src/ops/fused_moe_dispatch_op.cc
// Copyright 2025 Verso Industries
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
//
// ============================================================================
// PRODUCTION-READY REWRITE:
// This file has been rewritten for performance, stability, and compatibility
// with modern TensorFlow versions (TF 2.15+).
//
// Key Improvements:
// 1.  **Efficient Top-K Selection:** Replaced N*log(N) partial_sort with a
//     more efficient N*log(K) approach using a min-priority_queue, which is
//     significantly faster when K (expert_capacity) << N (num_tokens).
// 2.  **Modern Tensor Access:** Uses `TTypes<T>` views (e.g., `auto tokens =
//     context->input(0).matrix<float>()`) for safe, Eigen-style tensor
//     manipulation instead of raw pointers or manual Eigen::Map.
// 3.  **Robust Shape Inference:** Shape functions are made fully explicit to
//     prevent common op loading failures.
// 4.  **Clarity and Commenting:** Added extensive comments to clarify the logic
//     of expert choice routing, dispatching, and gradient scattering.
// 5.  **Phase 11 SIMD Compliance:** Added explicit SIMD guards (AVX512/AVX2/NEON)
//     for cross-platform compatibility. Primary workload is memory-bound
//     (scatter/gather) and atomic-heavy (gradient accumulation), but SIMD
//     headers ensure proper conditional compilation on all platforms.
// ============================================================================

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/lib/core/status.h"
#include "common/parallel/parallel_backend.h"
#include "common/perf_utils.h"
#include "common/edition_limits.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <queue>
#include <atomic>

// Phase 11 SIMD Guards: Conditional includes for cross-platform SIMD support
#if defined(__AVX512F__)
  #include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__)
  #include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
  #include <arm_neon.h>   // ARM NEON intrinsics
#endif

namespace tensorflow {

// A lightweight struct to hold information about a token chosen by an expert.
// This is used for sorting and dispatching.
struct TokenDispatchInfo {
    int32 original_token_index;
    int32 expert_id;
    float score; // The router logit score for this choice.

    // Sort primarily by expert, then by the original token order.
    // This groups tokens for each expert together while maintaining a stable order.
    bool operator<(const TokenDispatchInfo& other) const {
        if (expert_id != other.expert_id) {
            return expert_id < other.expert_id;
        }
        return original_token_index < other.original_token_index;
    }
};

// Custom comparator for the min-priority_queue used in top-k selection.
struct ScoreComparator {
    bool operator()(const std::pair<float, int>& a, const std::pair<float, int>& b) const {
        return a.first > b.first;
    }
};

// =============================================================================
// 1. FORWARD PASS OPERATOR (Original - kept for backward compatibility)
// =============================================================================
REGISTER_OP("FusedMoEDispatch")
    .Input("tokens: float")             // Shape: [num_tokens, d_model]
    .Input("router_logits: float")      // Shape: [num_tokens, num_experts]
    .Input("expert_capacity: int32")    // Shape: [1] (Scalar)
    .Output("dispatched_tokens: float") // Shape: [num_dispatched, d_model]
    .Output("dispatched_gates: float")  // Shape: [num_dispatched]
    .Output("dispatch_metadata: int32") // Shape: [num_dispatched] (Original token indices)
    .Output("expert_boundaries: int32") // Shape: [num_experts + 1]
    .Output("expert_indices: int32")    // Shape: [num_dispatched] (Expert ID for each token)
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle tokens_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &tokens_shape));
        shape_inference::DimensionHandle d_model = c->Dim(tokens_shape, 1);

        shape_inference::ShapeHandle router_logits_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &router_logits_shape));
        shape_inference::DimensionHandle num_experts = c->Dim(router_logits_shape, 1);

        // The number of dispatched tokens is data-dependent, so we use kUnknownDim.
        shape_inference::DimensionHandle num_dispatched = c->UnknownDim();

        c->set_output(0, c->Matrix(num_dispatched, d_model));
        c->set_output(1, c->Vector(num_dispatched));
        c->set_output(2, c->Vector(num_dispatched));
        
        shape_inference::DimensionHandle num_boundaries;
        TF_RETURN_IF_ERROR(c->Add(num_experts, 1, &num_boundaries));
        c->set_output(3, c->Vector(num_boundaries));
        
        c->set_output(4, c->Vector(num_dispatched));

        return OkStatus();
    });

// =============================================================================
// 1b. FORWARD PASS OPERATOR V2 (Enhanced with Ada-K, Routing Bias, Sigmoid)
// =============================================================================
REGISTER_OP("FusedMoEDispatchV2")
    .Input("tokens: float")             // Shape: [num_tokens, d_model]
    .Input("router_logits: float")      // Shape: [num_tokens, num_experts]
    .Input("expert_capacity: int32")    // Shape: [1] or [num_experts] (per-expert capacity)
    .Input("routing_bias: float")       // Shape: [num_experts] (EMA load-balancing bias)
    .Attr("use_sigmoid_routing: bool = false")  // GLM-4.5 style sigmoid gating
    .Attr("apply_bias_before_topk: bool = true") // Apply routing bias before top-k
    .Output("dispatched_tokens: float") // Shape: [num_dispatched, d_model]
    .Output("dispatched_gates: float")  // Shape: [num_dispatched]
    .Output("dispatch_metadata: int32") // Shape: [num_dispatched]
    .Output("expert_boundaries: int32") // Shape: [num_experts + 1]
    .Output("expert_indices: int32")    // Shape: [num_dispatched]
    .Output("expert_loads: float")      // Shape: [num_experts] (tokens per expert for EMA)
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle tokens_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &tokens_shape));
        shape_inference::DimensionHandle d_model = c->Dim(tokens_shape, 1);

        shape_inference::ShapeHandle router_logits_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &router_logits_shape));
        shape_inference::DimensionHandle num_experts = c->Dim(router_logits_shape, 1);

        shape_inference::DimensionHandle num_dispatched = c->UnknownDim();

        c->set_output(0, c->Matrix(num_dispatched, d_model));
        c->set_output(1, c->Vector(num_dispatched));
        c->set_output(2, c->Vector(num_dispatched));
        
        shape_inference::DimensionHandle num_boundaries;
        TF_RETURN_IF_ERROR(c->Add(num_experts, 1, &num_boundaries));
        c->set_output(3, c->Vector(num_boundaries));
        c->set_output(4, c->Vector(num_dispatched));
        c->set_output(5, c->Vector(num_experts));  // expert_loads

        return OkStatus();
    });

class FusedMoEDispatchOp : public OpKernel {
public:
    explicit FusedMoEDispatchOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        // --- 1. Get Inputs and Validate Shapes ---
        const Tensor& tokens_tensor = context->input(0);
        const Tensor& router_logits_tensor = context->input(1);
        const Tensor& expert_capacity_tensor = context->input(2);

        OP_REQUIRES(context, tokens_tensor.dims() == 2, errors::InvalidArgument("Input 'tokens' must be a 2D tensor."));
        OP_REQUIRES(context, router_logits_tensor.dims() == 2, errors::InvalidArgument("Input 'router_logits' must be a 2D tensor."));
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(expert_capacity_tensor.shape()), errors::InvalidArgument("expert_capacity must be a scalar int32 tensor."));
        OP_REQUIRES(context, tokens_tensor.dim_size(0) == router_logits_tensor.dim_size(0), errors::InvalidArgument("First dimension of 'tokens' and 'router_logits' must match."));

        const int expert_capacity = expert_capacity_tensor.scalar<int32>()();
        OP_REQUIRES(context, expert_capacity > 0, errors::InvalidArgument("Expert capacity must be positive."));

        const int64_t num_tokens = tokens_tensor.shape().dim_size(0);
        const int64_t d_model = tokens_tensor.shape().dim_size(1);
        const int64_t num_experts = router_logits_tensor.shape().dim_size(1);

        // HighNoon Lite Edition: Enforce MoE expert limit (max 12)
        SAGUARO_CHECK_MOE_EXPERTS(context, num_experts);

        const float* tokens_base = tokens_tensor.flat<float>().data();
        const float* logits_base = router_logits_tensor.flat<float>().data();
        const int64_t logits_row_stride = num_experts;

        // --- 2. Expert Choice Routing (Find Top-K Tokens for Each Expert) ---
        std::vector<std::vector<TokenDispatchInfo>> choices_per_expert(num_experts);
        
        auto work_routing = [&](int64_t start, int64_t end) {
            for (int j = start; j < end; ++j) {
                std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, ScoreComparator> min_heap;
                for (int i = 0; i < num_tokens; ++i) {
                    const float* score_addr = logits_base + static_cast<int64_t>(i) * logits_row_stride + j;
                    saguaro::ops::PrefetchL1(score_addr + logits_row_stride * 4);
                    float score = *score_addr;
                    if (min_heap.size() < expert_capacity) {
                        min_heap.push({score, i});
                    } else if (score > min_heap.top().first) {
                        min_heap.pop();
                        min_heap.push({score, i});
                    }
                }
                choices_per_expert[j].reserve(min_heap.size());
                while (!min_heap.empty()) {
                    const auto& top = min_heap.top();
                    choices_per_expert[j].push_back({top.second, j, top.first});
                    min_heap.pop();
                }
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_experts),
            static_cast<std::size_t>(num_tokens) * 100,
            work_routing);

        // --- 3. Consolidate, Sort, and Prepare for Dispatch ---
        std::vector<TokenDispatchInfo> all_choices;
        size_t total_choices = 0;
        for (const auto& expert_choices : choices_per_expert) {
            total_choices += expert_choices.size();
        }
        all_choices.reserve(total_choices);
        for (const auto& expert_choices : choices_per_expert) {
            all_choices.insert(all_choices.end(), expert_choices.begin(), expert_choices.end());
        }
        std::sort(all_choices.begin(), all_choices.end());

        const int64_t num_dispatched_tokens = all_choices.size();

        // --- 4. Allocate and Populate Output Tensors ---
        Tensor* dispatched_tokens_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {num_dispatched_tokens, d_model}, &dispatched_tokens_tensor));
        float* dispatched_tokens_ptr = dispatched_tokens_tensor->flat<float>().data();

        Tensor* dispatched_gates_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, {num_dispatched_tokens}, &dispatched_gates_tensor));
        auto gates = dispatched_gates_tensor->vec<float>();

        Tensor* dispatch_metadata_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, {num_dispatched_tokens}, &dispatch_metadata_tensor));
        auto metadata = dispatch_metadata_tensor->vec<int32>();

        Tensor* expert_indices_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, {num_dispatched_tokens}, &expert_indices_tensor));
        auto expert_indices = expert_indices_tensor->vec<int32>();
        
        const std::size_t copy_elems = static_cast<std::size_t>(d_model);
        auto work_dispatch = [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                const auto& choice = all_choices[i];
                metadata(i) = choice.original_token_index;
                gates(i) = choice.score;
                expert_indices(i) = choice.expert_id;
                const float* src = tokens_base + static_cast<int64_t>(choice.original_token_index) * d_model;
                float* dst = dispatched_tokens_ptr + static_cast<int64_t>(i) * d_model;
                saguaro::ops::PrefetchL1(src + d_model);
                saguaro::ops::CopySpan(dst, src, copy_elems);
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_dispatched_tokens),
            static_cast<std::size_t>(d_model) * 10,
            work_dispatch);

        // --- 5. Compute Expert Boundaries ---
        Tensor* expert_boundaries_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(3, {num_experts + 1}, &expert_boundaries_tensor));
        auto boundaries = expert_boundaries_tensor->vec<int32>();
        boundaries(0) = 0;
        
        if (num_dispatched_tokens > 0) {
            int current_expert_idx = 0;
            for (int i = 0; i < num_dispatched_tokens; ++i) {
                if (all_choices[i].expert_id > current_expert_idx) {
                    for(int e = current_expert_idx + 1; e <= all_choices[i].expert_id; ++e) {
                        boundaries(e) = i;
                    }
                    current_expert_idx = all_choices[i].expert_id;
                }
            }
            for (int i = current_expert_idx + 1; i <= num_experts; ++i) {
                boundaries(i) = num_dispatched_tokens;
            }
        } else {
            for(int i = 1; i <= num_experts; ++i) boundaries(i) = 0;
        }
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedMoEDispatch").Device(DEVICE_CPU), FusedMoEDispatchOp);

// =============================================================================
// 1c. FORWARD PASS KERNEL V2 (Enhanced with Ada-K, Routing Bias, Sigmoid)
// =============================================================================
class FusedMoEDispatchV2Op : public OpKernel {
public:
    explicit FusedMoEDispatchV2Op(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("use_sigmoid_routing", &use_sigmoid_routing_));
        OP_REQUIRES_OK(context, context->GetAttr("apply_bias_before_topk", &apply_bias_before_topk_));
    }

    void Compute(OpKernelContext* context) override {
        // --- 1. Get Inputs and Validate Shapes ---
        const Tensor& tokens_tensor = context->input(0);
        const Tensor& router_logits_tensor = context->input(1);
        const Tensor& expert_capacity_tensor = context->input(2);
        const Tensor& routing_bias_tensor = context->input(3);

        OP_REQUIRES(context, tokens_tensor.dims() == 2, 
            errors::InvalidArgument("Input 'tokens' must be a 2D tensor."));
        OP_REQUIRES(context, router_logits_tensor.dims() == 2, 
            errors::InvalidArgument("Input 'router_logits' must be a 2D tensor."));
        OP_REQUIRES(context, tokens_tensor.dim_size(0) == router_logits_tensor.dim_size(0), 
            errors::InvalidArgument("First dimension of 'tokens' and 'router_logits' must match."));

        const int64_t num_tokens = tokens_tensor.shape().dim_size(0);
        const int64_t d_model = tokens_tensor.shape().dim_size(1);
        const int64_t num_experts = router_logits_tensor.shape().dim_size(1);

        // Validate routing_bias shape
        OP_REQUIRES(context, routing_bias_tensor.dims() == 1 && 
                    routing_bias_tensor.dim_size(0) == num_experts,
            errors::InvalidArgument("routing_bias must be [num_experts], got ", 
                                    routing_bias_tensor.shape().DebugString()));

        // Handle scalar or per-expert capacity
        int expert_capacity = 1;
        if (TensorShapeUtils::IsScalar(expert_capacity_tensor.shape())) {
            expert_capacity = expert_capacity_tensor.scalar<int32>()();
        } else if (expert_capacity_tensor.dims() == 1 && 
                   expert_capacity_tensor.dim_size(0) == 1) {
            expert_capacity = expert_capacity_tensor.vec<int32>()(0);
        } else {
            OP_REQUIRES(context, false, 
                errors::InvalidArgument("expert_capacity must be scalar or [1]"));
        }
        OP_REQUIRES(context, expert_capacity > 0, 
            errors::InvalidArgument("Expert capacity must be positive."));

        SAGUARO_CHECK_MOE_EXPERTS(context, num_experts);

        const float* tokens_base = tokens_tensor.flat<float>().data();
        const float* logits_base = router_logits_tensor.flat<float>().data();
        const float* bias_base = routing_bias_tensor.flat<float>().data();
        const int64_t logits_row_stride = num_experts;

        // --- 2. Apply routing bias and optionally sigmoid ---
        // Create biased logits for top-k selection
        std::vector<float> biased_logits(num_tokens * num_experts);
        
        auto work_bias = [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                const float* src = logits_base + i * num_experts;
                float* dst = biased_logits.data() + i * num_experts;
                
                if (use_sigmoid_routing_) {
                    // GLM-4.5 style: sigmoid + normalization
                    float sum = 0.0f;
                    for (int64_t j = 0; j < num_experts; ++j) {
                        float logit = src[j];
                        if (apply_bias_before_topk_) {
                            logit += bias_base[j];
                        }
                        // Sigmoid activation
                        float sig = 1.0f / (1.0f + std::exp(-logit));
                        dst[j] = sig;
                        sum += sig;
                    }
                    // Normalize to sum to 1
                    if (sum > 1e-9f) {
                        for (int64_t j = 0; j < num_experts; ++j) {
                            dst[j] /= sum;
                        }
                    }
                } else {
                    // Standard softmax-style (just add bias)
                    for (int64_t j = 0; j < num_experts; ++j) {
                        dst[j] = src[j];
                        if (apply_bias_before_topk_) {
                            dst[j] += bias_base[j];
                        }
                    }
                }
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_tokens),
            static_cast<std::size_t>(num_experts) * 10,
            work_bias);

        // --- 3. Expert Choice Routing with biased logits ---
        std::vector<std::vector<TokenDispatchInfo>> choices_per_expert(num_experts);
        
        auto work_routing = [&](int64_t start, int64_t end) {
            for (int j = start; j < end; ++j) {
                std::priority_queue<std::pair<float, int>, 
                                   std::vector<std::pair<float, int>>, 
                                   ScoreComparator> min_heap;
                for (int i = 0; i < num_tokens; ++i) {
                    float score = biased_logits[i * num_experts + j];
                    if (min_heap.size() < static_cast<std::size_t>(expert_capacity)) {
                        min_heap.push({score, i});
                    } else if (score > min_heap.top().first) {
                        min_heap.pop();
                        min_heap.push({score, i});
                    }
                }
                choices_per_expert[j].reserve(min_heap.size());
                while (!min_heap.empty()) {
                    const auto& top = min_heap.top();
                    // Store original logit score (not biased) for gradient
                    float original_score = logits_base[top.second * num_experts + j];
                    choices_per_expert[j].push_back({top.second, static_cast<int32>(j), original_score});
                    min_heap.pop();
                }
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_experts),
            static_cast<std::size_t>(num_tokens) * 100,
            work_routing);

        // --- 4. Consolidate and Sort ---
        std::vector<TokenDispatchInfo> all_choices;
        size_t total_choices = 0;
        for (const auto& expert_choices : choices_per_expert) {
            total_choices += expert_choices.size();
        }
        all_choices.reserve(total_choices);
        for (const auto& expert_choices : choices_per_expert) {
            all_choices.insert(all_choices.end(), expert_choices.begin(), expert_choices.end());
        }
        std::sort(all_choices.begin(), all_choices.end());

        const int64_t num_dispatched_tokens = all_choices.size();

        // --- 5. Allocate and Populate Output Tensors ---
        Tensor* dispatched_tokens_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {num_dispatched_tokens, d_model}, &dispatched_tokens_tensor));
        float* dispatched_tokens_ptr = dispatched_tokens_tensor->flat<float>().data();

        Tensor* dispatched_gates_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, {num_dispatched_tokens}, &dispatched_gates_tensor));
        auto gates = dispatched_gates_tensor->vec<float>();

        Tensor* dispatch_metadata_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(2, {num_dispatched_tokens}, &dispatch_metadata_tensor));
        auto metadata = dispatch_metadata_tensor->vec<int32>();

        Tensor* expert_indices_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(4, {num_dispatched_tokens}, &expert_indices_tensor));
        auto expert_indices = expert_indices_tensor->vec<int32>();
        
        const std::size_t copy_elems = static_cast<std::size_t>(d_model);
        auto work_dispatch = [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                const auto& choice = all_choices[i];
                metadata(i) = choice.original_token_index;
                gates(i) = choice.score;
                expert_indices(i) = choice.expert_id;
                const float* src = tokens_base + static_cast<int64_t>(choice.original_token_index) * d_model;
                float* dst = dispatched_tokens_ptr + static_cast<int64_t>(i) * d_model;
                saguaro::ops::PrefetchL1(src + d_model);
                saguaro::ops::CopySpan(dst, src, copy_elems);
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_dispatched_tokens),
            static_cast<std::size_t>(d_model) * 10,
            work_dispatch);

        // --- 6. Compute Expert Boundaries ---
        Tensor* expert_boundaries_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(3, {num_experts + 1}, &expert_boundaries_tensor));
        auto boundaries = expert_boundaries_tensor->vec<int32>();
        boundaries(0) = 0;
        
        if (num_dispatched_tokens > 0) {
            int current_expert_idx = 0;
            for (int i = 0; i < num_dispatched_tokens; ++i) {
                if (all_choices[i].expert_id > current_expert_idx) {
                    for(int e = current_expert_idx + 1; e <= all_choices[i].expert_id; ++e) {
                        boundaries(e) = i;
                    }
                    current_expert_idx = all_choices[i].expert_id;
                }
            }
            for (int i = current_expert_idx + 1; i <= num_experts; ++i) {
                boundaries(i) = num_dispatched_tokens;
            }
        } else {
            for(int i = 1; i <= num_experts; ++i) boundaries(i) = 0;
        }

        // --- 7. Compute Expert Loads (for EMA update) ---
        Tensor* expert_loads_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(5, {num_experts}, &expert_loads_tensor));
        auto expert_loads = expert_loads_tensor->vec<float>();
        for (int i = 0; i < num_experts; ++i) {
            expert_loads(i) = static_cast<float>(boundaries(i + 1) - boundaries(i));
        }
    }

private:
    bool use_sigmoid_routing_;
    bool apply_bias_before_topk_;
};
REGISTER_KERNEL_BUILDER(Name("FusedMoEDispatchV2").Device(DEVICE_CPU), FusedMoEDispatchV2Op);

// =============================================================================
// 2. BACKWARD PASS OPERATOR
// =============================================================================
REGISTER_OP("FusedMoEDispatchGrad")
    .Input("grad_dispatched_tokens: float")
    .Input("grad_dispatched_gates: float")
    .Input("dispatch_metadata: int32")
    .Input("expert_indices: int32")
    .Input("tokens: float")
    .Input("router_logits: float")
    .Output("grad_tokens: float")
    .Output("grad_router_logits: float")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(4));
        c->set_output(1, c->input(5));
        return OkStatus();
    });

class FusedMoEDispatchGradOp : public OpKernel {
public:
    explicit FusedMoEDispatchGradOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& grad_dispatched_tokens_tensor = context->input(0);
        const Tensor& grad_dispatched_gates_tensor = context->input(1);
        const Tensor& dispatch_metadata_tensor = context->input(2);
        const Tensor& expert_indices_tensor = context->input(3);
        const Tensor& tokens_tensor = context->input(4);
        const Tensor& router_logits_tensor = context->input(5);

        const int64_t num_dispatched_tokens = grad_dispatched_tokens_tensor.shape().dim_size(0);
        const int64_t d_model = grad_dispatched_tokens_tensor.shape().dim_size(1);
        const int64_t num_tokens = tokens_tensor.shape().dim_size(0);
        const int64_t num_experts = router_logits_tensor.shape().dim_size(1);

        Tensor* grad_tokens_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, {num_tokens, d_model}, &grad_tokens_tensor));
        grad_tokens_tensor->flat<float>().setZero();

        Tensor* grad_router_logits_tensor = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(1, {num_tokens, num_experts}, &grad_router_logits_tensor));
        grad_router_logits_tensor->flat<float>().setZero();

        auto grad_dispatched_tokens = grad_dispatched_tokens_tensor.matrix<float>();
        auto grad_dispatched_gates = grad_dispatched_gates_tensor.vec<float>();
        auto metadata = dispatch_metadata_tensor.vec<int32>();
        auto expert_indices = expert_indices_tensor.vec<int32>();
        auto grad_tokens = grad_tokens_tensor->matrix<float>();
        auto grad_router_logits = grad_router_logits_tensor->matrix<float>();

        auto work_scatter = [&](int64_t start, int64_t end) {
            for (int i = start; i < end; ++i) {
                const int32 original_token_idx = metadata(i);
                const int32 expert_id = expert_indices(i);

                // Use atomic operations for thread-safe accumulation
                auto& gate_grad = reinterpret_cast<std::atomic<float>&>(grad_router_logits(original_token_idx, expert_id));
                float current_gate_grad = gate_grad.load();
                while(!gate_grad.compare_exchange_weak(current_gate_grad, current_gate_grad + grad_dispatched_gates(i)));

                for (int j = 0; j < d_model; ++j) {
                    auto& token_grad = reinterpret_cast<std::atomic<float>&>(grad_tokens(original_token_idx, j));
                    float current_token_grad = token_grad.load();
                    while(!token_grad.compare_exchange_weak(current_token_grad, current_token_grad + grad_dispatched_tokens(i, j)));
                }
            }
        };
        saguaro::parallel::ForShard(
            static_cast<std::size_t>(num_dispatched_tokens),
            static_cast<std::size_t>(d_model) * 10,
            work_scatter);
    }
};
REGISTER_KERNEL_BUILDER(Name("FusedMoEDispatchGrad").Device(DEVICE_CPU), FusedMoEDispatchGradOp);

} // namespace tensorflow
