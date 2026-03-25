// highnoon/_native/ops/fused_qhd_spatial_mega_op.h
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
 * @file fused_qhd_spatial_mega_op.h
 * @brief PHASE V2.0-P0.1: Fused QHD Spatial Mega Operator
 *
 * This kernel fuses the entire QHD Spatial Block forward pass into a single
 * operator, eliminating intermediate memory allocations and increasing
 * cache locality. Expected speedup: 2.0-2.5×.
 *
 * FUSED STAGES:
 *   1. Input projection via TTLayer contraction
 *   2. FFT forward (in-place, reuse workspace)
 *   3. Parallel SSM scan (Blelloch algorithm, O(L) work, O(log L) depth)
 *   4. Path mixing via Cayley unitary evolution
 *   5. Born rule collapse with coherence metric
 *   6. FFT inverse
 *   7. Output projection
 *
 * MEMORY OPTIMIZATIONS:
 *   - Single pre-allocated workspace for all stages
 *   - Uses PathScratchPool from hnn_simd_common.h
 *   - In-place FFT operations
 *   - Fused normalization and projection
 *
 * SIMD: AVX2 primary, AVX-512 secondary, ARM NEON tertiary
 *
 * Reference: HIGHNOON_V2_PERFORMANCE_ANALYSIS.md Section 6.1
 */

#ifndef HIGHNOON_NATIVE_OPS_FUSED_QHD_SPATIAL_MEGA_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_QHD_SPATIAL_MEGA_OP_H_

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

#include "qhd_spatial_block_op.h"
#include "hnn_simd_common.h"
#include "fft_utils.h"

namespace highnoon {
namespace ops {
namespace mega {

// =============================================================================
// MEGA OP CONFIGURATION
// =============================================================================

struct QHDMegaConfig {
    // Core dimensions
    int batch_size;
    int chunk_size;    // Phase 1: Fixed chunk size (replaces seq_len for memory sizing)
    int hd_dim;
    int state_dim;
    int num_paths;         // K superposition paths
    
    // FFT workspace size (computed from chunk_size and hd_dim, NOT seq_len)
    size_t fft_workspace_size;
    
    // Scratch buffer requirements
    size_t total_scratch_size;  // Single allocation for all stages
    
    // Features
    bool use_frequency_stratification;
    int entanglement_topology;  // 0=adjacent, 1=hierarchical, 2=walk
    float walk_evolution_time;
    
    QHDMegaConfig()
        : batch_size(1)
        , chunk_size(128)  // Phase 1: Default streaming chunk size (QAHPO: 64-256)
        , hd_dim(4096)
        , state_dim(64)
        , num_paths(2)
        , fft_workspace_size(0)
        , total_scratch_size(0)
        , use_frequency_stratification(true)
        , entanglement_topology(2)
        , walk_evolution_time(1.0f) {
        compute_workspace_sizes();
    }
    
    void compute_workspace_sizes() {
        // Phase 1.2: FFT workspace uses chunk_size, not seq_len
        // FFT operates on CHUNK_SIZE tokens at a time, enabling O(1) memory
        // Memory: 2 × chunk_size × hd_dim × 4 bytes = ~4MB @ chunk_size=128, hd_dim=4096
        fft_workspace_size = 2 * chunk_size * hd_dim * sizeof(float);
        
        // Total scratch: covers all intermediate buffers
        // - h_re, h_im: [K, state_dim, hd_dim] × 2
        // - y_re, y_im: [K, hd_dim] × 2
        // - mixed_re, mixed_im: [K, state_size] × 2 (for path mixing)
        // - probs: [K]
        // - exp_workspace: [K × state_dim × hd_dim]
        size_t state_size = num_paths * state_dim * hd_dim;
        size_t output_size = num_paths * hd_dim;
        total_scratch_size = (
            2 * state_size +      // h_re, h_im
            2 * output_size +     // y_re, y_im
            2 * state_size +      // mixed_re, mixed_im
            num_paths +           // probs
            state_size            // exp_workspace
        ) * sizeof(float);
    }
};

// =============================================================================
// WORKSPACE LAYOUT
// =============================================================================

/**
 * @brief Pre-allocated workspace for fused mega op.
 * 
 * Single contiguous allocation divided into logical regions.
 * Regions are reused across stages where possible.
 */
struct QHDMegaWorkspace {
    float* buffer;       // Owning pointer
    size_t size;         // Total size in bytes
    
    // Offset markers for each region (computed by layout())
    size_t h_re_offset;
    size_t h_im_offset;
    size_t y_re_offset;
    size_t y_im_offset;
    size_t mixed_re_offset;
    size_t mixed_im_offset;
    size_t probs_offset;
    size_t exp_workspace_offset;
    
    QHDMegaWorkspace() : buffer(nullptr), size(0) {}
    
    ~QHDMegaWorkspace() {
        if (buffer) {
            aligned_free(buffer);
        }
    }
    
    bool allocate(const QHDMegaConfig& config) {
        if (buffer && size >= config.total_scratch_size) {
            return true;  // Already allocated and large enough
        }
        if (buffer) {
            aligned_free(buffer);
        }
        buffer = static_cast<float*>(aligned_alloc(HNN_SIMD_ALIGNMENT, config.total_scratch_size));
        size = config.total_scratch_size;
        layout(config);
        return buffer != nullptr;
    }
    
    void layout(const QHDMegaConfig& config) {
        size_t state_size = config.num_paths * config.state_dim * config.hd_dim;
        size_t output_size = config.num_paths * config.hd_dim;
        
        size_t offset = 0;
        h_re_offset = offset; offset += state_size * sizeof(float);
        h_im_offset = offset; offset += state_size * sizeof(float);
        y_re_offset = offset; offset += output_size * sizeof(float);
        y_im_offset = offset; offset += output_size * sizeof(float);
        mixed_re_offset = offset; offset += state_size * sizeof(float);
        mixed_im_offset = offset; offset += state_size * sizeof(float);
        probs_offset = offset; offset += config.num_paths * sizeof(float);
        exp_workspace_offset = offset; offset += state_size * sizeof(float);
    }
    
    // Accessor helpers
    float* h_re() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + h_re_offset); }
    float* h_im() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + h_im_offset); }
    float* y_re() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + y_re_offset); }
    float* y_im() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + y_im_offset); }
    float* mixed_re() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + mixed_re_offset); }
    float* mixed_im() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + mixed_im_offset); }
    float* probs() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + probs_offset); }
    float* exp_workspace() { return reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + exp_workspace_offset); }
};

// =============================================================================
// FUSED MEGA FORWARD KERNEL (PLACEHOLDER)
// =============================================================================

/**
 * @brief Fused QHD Spatial Block forward pass.
 *
 * Combines all 7 stages into a single kernel call:
 *   1. Input projection (TTLayer)
 *   2. FFT forward
 *   3. Parallel SSM scan
 *   4. Path mixing (Cayley unitary)
 *   5. Born rule collapse
 *   6. FFT inverse
 *   7. Output projection
 *
 * @param input Input tensor [batch, seq_len, hd_dim]
 * @param output Output tensor [batch, seq_len, hd_dim]
 * @param coherence Output coherence metric [batch]
 * @param amplitudes_real Path amplitude real parts [K]
 * @param amplitudes_imag Path amplitude imag parts [K]
 * @param a_log SSM log-decay parameters [state_dim]
 * @param b_proj SSM input projection [hd_dim, state_dim]
 * @param c_proj SSM output projection [hd_dim, state_dim]
 * @param dt SSM discretization step [hd_dim]
 * @param hamiltonian Entanglement Hamiltonian [K, K]
 * @param workspace Pre-allocated workspace
 * @param config Mega op configuration
 */
inline void qhd_spatial_mega_forward(
    const float* input,
    float* output,
    float* coherence,
    const float* amplitudes_real,
    const float* amplitudes_imag,
    const float* a_log,
    const float* b_proj,
    const float* c_proj,
    const float* dt,
    const float* hamiltonian,
    QHDMegaWorkspace& workspace,
    const QHDMegaConfig& config
) {
    // TODO(V2.0): Implement fused kernel
    // For now, delegate to existing functions from qhd_spatial_block_op.h
    
    // This placeholder documents the intended API.
    // Full implementation requires:
    // 1. Fused FFT + SSM pipeline
    // 2. Single-pass path mixing and collapse
    // 3. Optimized memory reuse from workspace
    
    // The existing qhd_spatial_block_op.h functions can be called in sequence
    // as a baseline, but the fused version should combine them into single
    // loop nests for better cache locality.
}

// =============================================================================
// THREAD COUNT HELPER
// =============================================================================

/**
 * @brief Get optimal thread count for mega op based on workload.
 */
inline int get_mega_thread_count(const QHDMegaConfig& config) {
    // Use the global function from hnn_simd_common.h, but cap based on workload
    int base_threads = get_optimal_path_thread_count();
    
    // Phase 1.4: Use chunk_size for parallelism hints (not memory sizing)
    // Work per iteration is chunk_size × hd_dim, independent of seq_len
    int min_work_per_thread = config.chunk_size * config.hd_dim;
    int total_work = config.batch_size * config.chunk_size * config.hd_dim;
    int work_threads = std::max(1, total_work / min_work_per_thread);
    
    return std::min(base_threads, work_threads);
}

}  // namespace mega
}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_FUSED_QHD_SPATIAL_MEGA_OP_H_
