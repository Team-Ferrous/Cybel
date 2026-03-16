// highnoon/_native/ops/dijkstra_moe_router_op.h
// Copyright 2025 Verso Industries
//
// Topology-Aware MoE Routing with Dijkstra multi-hop search.
// Enables expert knowledge graphs for semantic expert routing.
//
// Architecture: Models experts as nodes in a semantic knowledge graph:
//   - Nodes: Experts (e.g., "Syntax", "Math", "Entity Extraction")
//   - Edges: Semantic relationships (e.g., "Math → Physics", "Syntax → Grammar")
//   - Routing: Dijkstra finds optimal multi-hop path through expert graph
//
// Complexity: O(E + V log V) where V = num_experts, E = topology edges
//             For typical V=12, E=30: O(73) per token (effectively O(1))
// SIMD: AVX2/AVX512/NEON optimized

#ifndef HIGHNOON_DIJKSTRA_MOE_ROUTER_OP_H_
#define HIGHNOON_DIJKSTRA_MOE_ROUTER_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

// SIMD intrinsics
#if defined(__AVX512F__)
    #include <immintrin.h>
    #define SIMD_WIDTH 16
#elif defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define SIMD_WIDTH 4
#else
    #define SIMD_WIDTH 1
#endif

namespace tensorflow {
namespace highnoon {

// =============================================================================
// CSR Expert Topology Graph
// =============================================================================
struct ExpertTopologyGraph {
    std::vector<int> row_offsets;      // [num_experts + 1]
    std::vector<int> col_indices;      // [num_edges]
    std::vector<float> edge_costs;     // [num_edges] - semantic distances
    int num_experts;
    int num_edges;

    ExpertTopologyGraph() : num_experts(0), num_edges(0) {}

    ExpertTopologyGraph(int experts, int edges)
        : num_experts(experts), num_edges(edges) {
        row_offsets.resize(experts + 1, 0);
        col_indices.reserve(edges);
        edge_costs.reserve(edges);
    }

    // Build from edge list: [(from, to, cost), ...]
    void build_from_edges(
        const int* edge_from,
        const int* edge_to,
        const float* edge_cost,
        int num_edges_input
    ) {
        num_edges = num_edges_input;
        col_indices.clear();
        edge_costs.clear();

        // Count out-edges per expert
        std::vector<int> out_degree(num_experts, 0);
        for (int i = 0; i < num_edges; ++i) {
            out_degree[edge_from[i]]++;
        }

        // Build row offsets
        row_offsets[0] = 0;
        for (int i = 0; i < num_experts; ++i) {
            row_offsets[i + 1] = row_offsets[i] + out_degree[i];
        }

        // Allocate edge arrays
        col_indices.resize(num_edges);
        edge_costs.resize(num_edges);

        // Fill edges (using temporary counters)
        std::vector<int> counter(num_experts, 0);
        for (int i = 0; i < num_edges; ++i) {
            int from = edge_from[i];
            int idx = row_offsets[from] + counter[from];
            col_indices[idx] = edge_to[i];
            edge_costs[idx] = edge_cost[i];
            counter[from]++;
        }
    }

    // Get neighbors for an expert (cache-friendly, no copying)
    inline void get_neighbor_ptrs(
        int expert,
        const int** out_neighbors,
        const float** out_costs,
        int* out_count
    ) const {
        if (expert < 0 || expert >= num_experts) {
            *out_count = 0;
            return;
        }
        int start = row_offsets[expert];
        int end = row_offsets[expert + 1];
        *out_count = end - start;
        if (*out_count > 0) {
            *out_neighbors = &col_indices[start];
            *out_costs = &edge_costs[start];
        }
    }
};

// =============================================================================
// SIMD-Optimized Vector Operations
// =============================================================================
class SIMDMoEOps {
public:
    // Compute semantic distance (1 - cosine similarity)
    static inline float semantic_distance(
        const float* token_state,
        const float* expert_embedding,
        int dim
    ) {
        // Compute cosine similarity via dot product (assuming normalized)
        float dot = 0.0f;

#if defined(__AVX2__) || defined(__AVX512F__)
        __m256 sum = _mm256_setzero_ps();
        int i = 0;

        for (; i + 7 < dim; i += 8) {
            __m256 va = _mm256_loadu_ps(token_state + i);
            __m256 vb = _mm256_loadu_ps(expert_embedding + i);
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        float result[8];
        _mm256_storeu_ps(result, sum);
        dot = result[0] + result[1] + result[2] + result[3] +
              result[4] + result[5] + result[6] + result[7];

        for (; i < dim; ++i) {
            dot += token_state[i] * expert_embedding[i];
        }

#elif defined(__ARM_NEON)
        float32x4_t sum = vdupq_n_f32(0.0f);
        int i = 0;

        for (; i + 3 < dim; i += 4) {
            float32x4_t va = vld1q_f32(token_state + i);
            float32x4_t vb = vld1q_f32(expert_embedding + i);
            sum = vmlaq_f32(sum, va, vb);
        }

        dot = vaddvq_f32(sum);

        for (; i < dim; ++i) {
            dot += token_state[i] * expert_embedding[i];
        }

#else
        for (int i = 0; i < dim; ++i) {
            dot += token_state[i] * expert_embedding[i];
        }
#endif

        // Convert to distance: d = 1 - similarity
        // Clamp similarity to [0, 1] for numerical stability
        float similarity = std::max(0.0f, std::min(1.0f, dot));
        return 1.0f - similarity;
    }
};

// =============================================================================
// Dijkstra Multi-Hop Expert Routing
// =============================================================================
// =============================================================================
// Dijkstra Multi-Hop Expert Routing
// =============================================================================
struct ExpertRouteNode {
    int expert_id;
    int hop;
    float cost;

    bool operator>(const ExpertRouteNode& other) const {
        return cost > other.cost;
    }
};

class DijkstraMoERouter {
public:
    DijkstraMoERouter(int num_experts, int max_hops)
        : num_experts_(num_experts), max_hops_(max_hops) {}

    // Find optimal expert routing path via Dijkstra
    // Supports Parallel Universes (superposition) via flattening in caller
    // Returns aggregated logits directly
    void route(
        const float* token_state,
        const float* expert_embeddings,
        const ExpertTopologyGraph& graph,
        int dim,
        int top_k,
        int* out_expert_path,      // [max_hops, top_k]
        float* out_path_costs,     // [max_hops, top_k]
        int* out_path_lengths      // [top_k]
    );

private:
    int num_experts_;
    int max_hops_;

    float compute_routing_cost(
        const float* token_state,
        const float* expert_embedding,
        float edge_cost,
        int dim
    ) {
        // Total cost = semantic distance + edge cost
        float semantic_dist = SIMDMoEOps::semantic_distance(
            token_state, expert_embedding, dim
        );
        return semantic_dist + edge_cost;
    }
};

// =============================================================================
// TensorFlow Op Kernel
// =============================================================================
class __attribute__((visibility("default"))) DijkstraMoERouterOp : public OpKernel {
public:
    explicit DijkstraMoERouterOp(OpKernelConstruction* context);
    void Compute(OpKernelContext* context) override;

private:
    int max_hops_;
    int top_k_;
    float temperature_; // Added for logit scaling
};

}  // namespace highnoon
}  // namespace tensorflow

#endif  // HIGHNOON_DIJKSTRA_MOE_ROUTER_OP_H_
