// saguaro.native/ops/fused_quantum_layers_op.h
// Quantum layer SIMD helpers - VQC expectation value helpers

#ifndef SAGUARO_NATIVE_OPS_FUSED_QUANTUM_LAYERS_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_QUANTUM_LAYERS_OP_H_

#include <cstdint>
#include <cmath>

namespace saguaro { namespace ops {

// Rotation gate helpers for VQC simulation
inline void quantum_rotation_y(float angle, float* matrix) {
    float c = std::cos(angle / 2.0f);
    float s = std::sin(angle / 2.0f);
    matrix[0] = c; matrix[1] = -s;
    matrix[2] = s; matrix[3] = c;
}

inline void quantum_rotation_z(float angle, float* matrix_real, float* matrix_imag) {
    float c = std::cos(angle / 2.0f);
    float s = std::sin(angle / 2.0f);
    matrix_real[0] = c; matrix_imag[0] = -s;
    matrix_real[1] = 0; matrix_imag[1] = 0;
    matrix_real[2] = 0; matrix_imag[2] = 0;
    matrix_real[3] = c; matrix_imag[3] = s;
}

}} // namespace
#endif
