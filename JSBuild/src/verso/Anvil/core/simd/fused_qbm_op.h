// highnoon/_native/ops/fused_qbm_op.h
// Quantum Boltzmann Machine helpers
#ifndef HIGHNOON_NATIVE_OPS_FUSED_QBM_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_QBM_OP_H_
#include <cstdint>
#include <cmath>
namespace highnoon { namespace ops {
inline float qbm_energy(const float* visible, const float* hidden, const float* weights,
                        int64_t v_size, int64_t h_size) {
    float E = 0.0f;
    for (int64_t v = 0; v < v_size; ++v) {
        for (int64_t h = 0; h < h_size; ++h) {
            E -= weights[v * h_size + h] * visible[v] * hidden[h];
        }
    }
    return E;
}
inline float qbm_partition_approx(const float* weights, int64_t v_size, int64_t h_size, float beta) {
    return std::exp(-beta * 0.5f * v_size * h_size);  // Simplified
}
}} // namespace
#endif
