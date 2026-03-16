// highnoon/_native/ops/hnsw_moe_router_op.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// HNSW-Based Graph MoE Router for O(log N) Expert Selection
//
// This implements Sprint 3 of the Technical Roadmap: Graph-MoE Integration
// using Hierarchical Navigable Small World (HNSW) graphs for efficient
// expert routing at scale.

#ifndef HIGHNOON_NATIVE_OPS_HNSW_MOE_ROUTER_OP_H_
#define HIGHNOON_NATIVE_OPS_HNSW_MOE_ROUTER_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "hnn_simd_common.h"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

#ifdef HIGHNOON_USE_HNSWLIB
#include <hnswlib/hnswlib.h>
#endif

namespace highnoon {
namespace moe {

// Configuration for HNSW MoE routing
struct HNSWMoEConfig {
    int num_experts = 10;
    int d_model = 512;
    int top_k = 2;
    int M = 16;              // HNSW connectivity parameter
    int ef_construction = 200;  // Construction-time search parameter
    int ef_search = 64;      // Query-time search parameter
    float temperature = 1.0f;
    bool use_hnsw = true;    // Fallback to linear if false
};

#ifdef HIGHNOON_USE_HNSWLIB

// HNSW-based MoE Router with O(log N) expert selection
class HNSWMoERouter {
public:
    explicit HNSWMoERouter(const HNSWMoEConfig& config)
        : config_(config),
          space_(new hnswlib::L2Space(config.d_model)),
          index_(nullptr) {
        // Defer index creation until expert patterns are provided
    }

    ~HNSWMoERouter() = default;

    // Build HNSW graph from expert centroids/patterns
    // expert_patterns: [num_experts, d_model]
    void BuildGraph(const float* expert_patterns, int num_experts) {
        config_.num_experts = num_experts;

        // Create new index
        index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            space_.get(), num_experts, config_.M, config_.ef_construction);

        // Add expert patterns to index
        for (int e = 0; e < num_experts; ++e) {
            index_->addPoint(expert_patterns + e * config_.d_model, e);
        }
    }

    // Route tokens to top-K experts using HNSW search
    // token_states: [batch_size, d_model]
    // expert_indices: [batch_size, top_k] - output
    // expert_weights: [batch_size, top_k] - output (softmax scores)
    void RouteTokens(
        const float* token_states,
        int batch_size,
        int32_t* expert_indices,
        float* expert_weights
    ) {
        if (index_ == nullptr) {
            // Fallback: use linear search
            RouteTokensLinear(token_states, batch_size, expert_indices, expert_weights);
            return;
        }

        index_->setEf(config_.ef_search);

        #pragma omp parallel for if(batch_size > 32)
        for (int b = 0; b < batch_size; ++b) {
            const float* query = token_states + b * config_.d_model;

            // Search for top_k nearest experts
            auto result = index_->searchKnn(query, config_.top_k);

            // Extract results (result is max-heap, so reverse order)
            std::vector<std::pair<float, int>> sorted_results;
            while (!result.empty()) {
                sorted_results.push_back({result.top().first, result.top().second});
                result.pop();
            }
            std::reverse(sorted_results.begin(), sorted_results.end());

            // Convert distances to softmax weights
            float max_neg_dist = -sorted_results[0].first;
            float sum_exp = 0.0f;
            std::vector<float> exp_scores(config_.top_k);

            for (int k = 0; k < config_.top_k; ++k) {
                float neg_dist = -sorted_results[k].first;
                exp_scores[k] = std::exp((neg_dist - max_neg_dist) / config_.temperature);
                sum_exp += exp_scores[k];
            }

            // Write outputs
            for (int k = 0; k < config_.top_k; ++k) {
                expert_indices[b * config_.top_k + k] = sorted_results[k].second;
                expert_weights[b * config_.top_k + k] = exp_scores[k] / sum_exp;
            }
        }
    }

    // Rebuild graph periodically with updated expert patterns
    void RebuildGraph(const float* expert_patterns) {
        BuildGraph(expert_patterns, config_.num_experts);
    }

private:
    HNSWMoEConfig config_;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;

    // Fallback linear routing for when HNSW is not built
    void RouteTokensLinear(
        const float* token_states,
        int batch_size,
        int32_t* expert_indices,
        float* expert_weights
    ) {
        // This would require expert_patterns to be stored
        // For now, just return sequential experts
        for (int b = 0; b < batch_size; ++b) {
            for (int k = 0; k < config_.top_k; ++k) {
                expert_indices[b * config_.top_k + k] = k % config_.num_experts;
                expert_weights[b * config_.top_k + k] = 1.0f / config_.top_k;
            }
        }
    }
};

#else // !HIGHNOON_USE_HNSWLIB

// Fallback implementation using linear search when hnswlib not available
class HNSWMoERouter {
public:
    explicit HNSWMoERouter(const HNSWMoEConfig& config)
        : config_(config) {}

    void BuildGraph(const float* expert_patterns, int num_experts) {
        config_.num_experts = num_experts;
        expert_patterns_.assign(
            expert_patterns,
            expert_patterns + num_experts * config_.d_model);
    }

    // O(N) linear search fallback
    void RouteTokens(
        const float* token_states,
        int batch_size,
        int32_t* expert_indices,
        float* expert_weights
    ) {
        if (expert_patterns_.empty()) {
            // No patterns, return uniform routing
            for (int b = 0; b < batch_size; ++b) {
                for (int k = 0; k < config_.top_k; ++k) {
                    expert_indices[b * config_.top_k + k] = k % config_.num_experts;
                    expert_weights[b * config_.top_k + k] = 1.0f / config_.top_k;
                }
            }
            return;
        }

        #pragma omp parallel for if(batch_size > 32)
        for (int b = 0; b < batch_size; ++b) {
            const float* query = token_states + b * config_.d_model;

            // Compute all distances
            std::vector<std::pair<float, int>> distances(config_.num_experts);
            for (int e = 0; e < config_.num_experts; ++e) {
                float dist = 0.0f;
                const float* pattern = expert_patterns_.data() + e * config_.d_model;

                // SIMD-optimized L2 distance
                #if defined(__AVX2__)
                dist = simd_l2_distance_avx2(query, pattern, config_.d_model);
                #elif defined(__ARM_NEON)
                dist = simd_l2_distance_neon(query, pattern, config_.d_model);
                #else
                for (int d = 0; d < config_.d_model; ++d) {
                    float diff = query[d] - pattern[d];
                    dist += diff * diff;
                }
                #endif
                distances[e] = {dist, e};
            }

            // Partial sort to get top-k
            std::partial_sort(
                distances.begin(),
                distances.begin() + config_.top_k,
                distances.end());

            // Convert to softmax weights
            float max_neg_dist = -distances[0].first;
            float sum_exp = 0.0f;
            std::vector<float> exp_scores(config_.top_k);

            for (int k = 0; k < config_.top_k; ++k) {
                float neg_dist = -distances[k].first;
                exp_scores[k] = std::exp((neg_dist - max_neg_dist) / config_.temperature);
                sum_exp += exp_scores[k];
            }

            for (int k = 0; k < config_.top_k; ++k) {
                expert_indices[b * config_.top_k + k] = distances[k].second;
                expert_weights[b * config_.top_k + k] = exp_scores[k] / sum_exp;
            }
        }
    }

    void RebuildGraph(const float* expert_patterns) {
        BuildGraph(expert_patterns, config_.num_experts);
    }

private:
    HNSWMoEConfig config_;
    std::vector<float> expert_patterns_;

    // SIMD L2 distance helpers
    #if defined(__AVX2__)
    static float simd_l2_distance_avx2(const float* a, const float* b, int dim) {
        __m256 sum = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= dim; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(sum, 1);
        __m128 lo = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);
        // Handle remainder
        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
        return result;
    }
    #endif

    #if defined(__ARM_NEON)
    static float simd_l2_distance_neon(const float* a, const float* b, int dim) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        int i = 0;
        for (; i + 4 <= dim; i += 4) {
            float32x4_t va = vld1q_f32(a + i);
            float32x4_t vb = vld1q_f32(b + i);
            float32x4_t diff = vsubq_f32(va, vb);
            sum = vmlaq_f32(sum, diff, diff);
        }
        float result = vaddvq_f32(sum);
        for (; i < dim; ++i) {
            float diff = a[i] - b[i];
            result += diff * diff;
        }
        return result;
    }
    #endif
};

#endif // HIGHNOON_USE_HNSWLIB

} // namespace moe
} // namespace highnoon

namespace tensorflow {

// TensorFlow Op Kernel for HNSW MoE Build
class HNSWMoEBuildOpCpu : public OpKernel {
public:
    explicit HNSWMoEBuildOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_experts", &num_experts_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("d_model", &d_model_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &M_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ef_construction", &ef_construction_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& expert_patterns = ctx->input(0);

        OP_REQUIRES(ctx, expert_patterns.dims() == 2,
            errors::InvalidArgument("expert_patterns must be 2D [num_experts, d_model]"));

        // Store patterns for later use by route op
        // In practice, this would update a shared resource
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({1}), &output));
        output->flat<int32>()(0) = expert_patterns.dim_size(0);
    }

private:
    int num_experts_;
    int d_model_;
    int M_;
    int ef_construction_;
};

// TensorFlow Op Kernel for HNSW MoE Route
class HNSWMoERouteOpCpu : public OpKernel {
public:
    explicit HNSWMoERouteOpCpu(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("top_k", &top_k_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("ef_search", &ef_search_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("temperature", &temperature_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("num_experts", &num_experts_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("d_model", &d_model_));
    }

    void Compute(OpKernelContext* ctx) override {
        const Tensor& token_states = ctx->input(0);
        const Tensor& expert_patterns = ctx->input(1);

        const int64 batch_size = token_states.dim_size(0);
        const int64 d_model = token_states.dim_size(1);
        const int64 num_experts = expert_patterns.dim_size(0);

        // Allocate outputs
        Tensor* expert_indices = nullptr;
        Tensor* expert_weights = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0,
            TensorShape({batch_size, top_k_}), &expert_indices));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1,
            TensorShape({batch_size, top_k_}), &expert_weights));

        // Configure router
        highnoon::moe::HNSWMoEConfig config;
        config.num_experts = static_cast<int>(num_experts);
        config.d_model = static_cast<int>(d_model);
        config.top_k = top_k_;
        config.ef_search = ef_search_;
        config.temperature = temperature_;

        // Create router and build graph
        highnoon::moe::HNSWMoERouter router(config);
        router.BuildGraph(expert_patterns.flat<float>().data(),
                         static_cast<int>(num_experts));

        // Route tokens
        router.RouteTokens(
            token_states.flat<float>().data(),
            static_cast<int>(batch_size),
            expert_indices->flat<int32>().data(),
            expert_weights->flat<float>().data()
        );
    }

private:
    int top_k_;
    int ef_search_;
    float temperature_;
    int num_experts_;
    int d_model_;
};

} // namespace tensorflow

#endif // HIGHNOON_NATIVE_OPS_HNSW_MOE_ROUTER_OP_H_
