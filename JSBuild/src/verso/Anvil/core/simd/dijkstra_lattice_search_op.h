// highnoon/_native/ops/dijkstra_lattice_search_op.h
// Copyright 2025 Verso Industries
//
// Token Lattice Rescoring with Dijkstra global optimization.
// Finds optimal token sequence through candidate lattice for QSG coherence.
//
// Architecture: Token generation as lattice pathfinding:
//   - Nodes: (position, token_id, score) candidates
//   - Edges: Transition probabilities between positions
//   - Goal: Find globally optimal path via Dijkstra
//
// Complexity: O(E + V log V) where V = seq_len × candidates, E = transitions
//             For seq_len=128, K=16: V=2048, E≈32K → O(55K) per sequence
// SIMD: AVX2/AVX512/NEON optimized

#ifndef HIGHNOON_DIJKSTRA_LATTICE_SEARCH_OP_H_
#define HIGHNOON_DIJKSTRA_LATTICE_SEARCH_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <limits>

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
// Lattice Node Representation
// =============================================================================
struct LatticeNode {
    int position;     // Sequence position
    int token_id;     // Token ID at this position
    float local_score; // Local candidate score (from QSG)
    int node_id;      // Unique node ID for graph indexing

    LatticeNode() : position(0), token_id(0), local_score(0.0f), node_id(0) {}
    LatticeNode(int pos, int tok, float score, int id)
        : position(pos), token_id(tok), local_score(score), node_id(id) {}
};

// =============================================================================
// Lattice Edge (CSR Format)
// =============================================================================
struct LatticeGraph {
    std::vector<int> row_offsets;      // [num_nodes + 1]
    std::vector<int> col_indices;      // [num_edges] - target node IDs
    std::vector<float> edge_weights;   // [num_edges] - transition costs
    std::vector<LatticeNode> nodes;    // Node metadata
    int num_nodes;
    int num_edges;
    int seq_len;
    int num_candidates;

    LatticeGraph() : num_nodes(0), num_edges(0), seq_len(0), num_candidates(0) {}

    void build_from_candidates(
        const float* candidate_scores,  // [seq_len, num_candidates]
        const int* candidate_tokens,    // [seq_len, num_candidates]
        const float* transition_costs,  // [seq_len-1, num_candidates, num_candidates] or null
        int seq_len_in,
        int num_candidates_in
    );
};

// =============================================================================
// Dijkstra Search State
// =============================================================================
struct DijkstraState {
    int node_id;
    float cost;
    int parent;

    bool operator>(const DijkstraState& other) const {
        return cost > other.cost;
    }
};

// =============================================================================
// Memory Arena (Zero Hot-Path Allocations)
// =============================================================================
class LatticeMemoryArena {
public:
    LatticeMemoryArena(int max_nodes)
        : distances_(max_nodes, std::numeric_limits<float>::infinity()),
          visited_(max_nodes, false),
          parents_(max_nodes, -1) {}

    void reset() {
        std::fill(distances_.begin(), distances_.end(),
                  std::numeric_limits<float>::infinity());
        std::fill(visited_.begin(), visited_.end(), false);
        std::fill(parents_.begin(), parents_.end(), -1);
    }

    std::vector<float>& distances() { return distances_; }
    std::vector<bool>& visited() { return visited_; }
    std::vector<int>& parents() { return parents_; }

private:
    std::vector<float> distances_;
    std::vector<bool> visited_;
    std::vector<int> parents_;
};

// =============================================================================
// Dijkstra Lattice Search
// =============================================================================
class DijkstraLatticeSearch {
public:
    DijkstraLatticeSearch(int seq_len, int num_candidates)
        : seq_len_(seq_len), num_candidates_(num_candidates),
          arena_(seq_len * num_candidates + 2) {}  // +2 for START/END

    // Find optimal path through token lattice
    void search(
        const LatticeGraph& graph,
        int* out_path,           // [seq_len] optimal token sequence
        float* out_path_score    // Scalar: total path score
    );

private:
    int seq_len_;
    int num_candidates_;
    LatticeMemoryArena arena_;

    // SIMD-optimized edge relaxation
    void relax_edges_simd(
        const LatticeGraph& graph,
        int node_id,
        float current_cost,
        std::priority_queue<DijkstraState,
                           std::vector<DijkstraState>,
                           std::greater<DijkstraState>>& pq
    );
};

// =============================================================================
// TensorFlow Op Kernel
// =============================================================================
class __attribute__((visibility("default"))) DijkstraLatticeSearchOp : public OpKernel {
public:
    explicit DijkstraLatticeSearchOp(OpKernelConstruction* context);
    void Compute(OpKernelContext* context) override;

private:
    int beam_width_;
    bool use_transition_costs_;
};

}  // namespace highnoon
}  // namespace tensorflow

#endif  // HIGHNOON_DIJKSTRA_LATTICE_SEARCH_OP_H_
