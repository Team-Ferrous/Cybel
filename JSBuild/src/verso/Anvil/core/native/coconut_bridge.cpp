#include <cstdint>
#include <vector>

// Define necessary macros for the headers
#define HIGHNOON_NATIVE_OPS_COMMON_H_

// Include the COCONUT header from Saguaro
// We need to provide the path to the include directory
#include "fused_coconut_bfs_op.h"

extern "C" {

void coconut_expand_paths_c(
    const float* hidden_state,
    float* paths,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    float noise_scale) {
    highnoon::ops::coconut::coconut_expand_paths(hidden_state, paths, batch_size, num_paths, dim, noise_scale);
}

void coconut_evolve_paths_c(
    float* paths,
    const float* norm_gamma,
    const float* norm_beta,
    const float* dense1_weight,
    const float* dense1_bias,
    const float* dense2_weight,
    const float* dense2_bias,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim,
    int64_t hidden_dim,
    float* work_buffer) {
    // Note: g_path_scratch is initialized in hnn_simd_common.h
    highnoon::ops::coconut::coconut_evolve_paths(
        paths, norm_gamma, norm_beta, dense1_weight, dense1_bias, dense2_weight, dense2_bias,
        batch_size, num_paths, dim, hidden_dim, work_buffer, false);
}

void coconut_amplitude_score_c(
    const float* paths,
    const float* context,
    float* amplitudes,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim) {
    highnoon::ops::coconut::coconut_amplitude_score(paths, context, amplitudes, batch_size, num_paths, dim);
}

void coconut_aggregate_paths_c(
    const float* paths,
    const float* amplitudes,
    float* output,
    int64_t batch_size,
    int64_t num_paths,
    int64_t dim) {
    highnoon::ops::coconut::coconut_aggregate_paths(paths, amplitudes, output, batch_size, num_paths, dim);
}

}
