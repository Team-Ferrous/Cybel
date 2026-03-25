// saguaro/native/ops/qhd_spatial_common.h
#ifndef SAGUARO_NATIVE_OPS_QHD_SPATIAL_COMMON_H_
#define SAGUARO_NATIVE_OPS_QHD_SPATIAL_COMMON_H_

#include <cmath>
#include <algorithm>

namespace hsmn {
namespace qhd_spatial {

/**
 * QHD Spatial Block Configuration.
 */
struct QHDSpatialConfig {
    int hd_dim = 4096;              // Hyperdimensional embedding dimension
    int hidden_dim = 512;           // Internal hidden dimension
    int state_dim = 16;             // SSM state dimension (Mamba N)
    int num_paths = 2;              // K superposition paths (QAHPO: 2-16)
    int entanglement_depth = 2;     // VQC entanglement layers (QAHPO: 1-4)
    float entanglement_strength = 0.3f;  // CNOT mixing strength
    float gumbel_temperature = 1.0f;     // Born rule temperature
    float dt_min = 0.001f;          // Minimum discretization step
    float dt_max = 0.1f;            // Maximum discretization step
    
    // F1 Optimization Phase 1.2: FFT Loop Tiling
    int fft_tile_size = 64;         // Cache-friendly tile size for FFT ops (L2: 256KB)
    
    // F1 Optimization Phase 4.2: Sparse Entanglement Updates
    float sparse_entanglement_threshold = 0.1f;  // Threshold for sparse updates (QAHPO: 0.05-0.2)
    
    // UQHA Phase 850: Frequency Stratification
    bool use_frequency_stratification = true;
    int freq_mask_mode = 0;
    float freq_overlap = 0.25f;
    
    // UQHA Phase 860: Quantum Walk Entanglement
    int entanglement_topology = 2;
    float walk_evolution_time = 1.0f;
    bool walk_learn_time = false;
    
    // UQHA v3.1 P1: Diagonal Skip Connection
    int skip_connection_type = 1;
    float skip_diagonal_init = 1.0f;
};


/**
 * Compute frequency masks for K superposition paths.
 */
inline void compute_frequency_masks(
    float* masks,
    const QHDSpatialConfig& config
) {
    const int K = config.num_paths;
    const int D = config.hd_dim;
    const float overlap = config.freq_overlap;
    
    for (int k = 0; k < K; ++k) {
        int base_cutoff = D >> k;
        if (base_cutoff < 1) base_cutoff = 1;
        int soft_cutoff = static_cast<int>(base_cutoff * (1.0f + overlap));
        soft_cutoff = std::min(soft_cutoff, D);
        
        for (int d = 0; d < D; ++d) {
            if (d < base_cutoff) {
                masks[k * D + d] = 1.0f;
            } else if (d < soft_cutoff) {
                float t = static_cast<float>(d - base_cutoff) / 
                         static_cast<float>(soft_cutoff - base_cutoff + 1);
                masks[k * D + d] = 0.5f * (1.0f + std::cos(M_PI * t));
            } else {
                masks[k * D + d] = 0.0f;
            }
        }
    }
}

} // namespace qhd_spatial
} // namespace hsmn

#endif // SAGUARO_NATIVE_OPS_QHD_SPATIAL_COMMON_H_
