// highnoon/_native/ops/mps_evolution_op.h
#ifndef HIGHNOON_OPS_MPS_EVOLUTION_OP_H_
#define HIGHNOON_OPS_MPS_EVOLUTION_OP_H_

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "Eigen/Dense"
#include <vector>

namespace highnoon {
namespace ops {

/**
 * @brief Apply a two-site gate to an MPS bond.
 * 
 * @param core_i Left core tensor [chi_L, d, chi_M]
 * @param core_next Right core tensor [chi_M, d, chi_R]
 * @param gate_real Real part of 2-site gate [d*d, d*d]
 * @param gate_imag Imaginary part of 2-site gate [d*d, d*d]
 * @param max_bond_dim Maximum bond dimension for truncation
 * @param threshold Truncation error threshold
 * @param out_core_i Output left core
 * @param out_core_next Output right core
 */
void ApplyTwoSiteGate(
    const Eigen::Tensor<float, 3, Eigen::RowMajor>& core_i,
    const Eigen::Tensor<float, 3, Eigen::RowMajor>& core_next,
    const Eigen::MatrixXf& gate_real,
    const Eigen::MatrixXf& gate_imag,
    int max_bond_dim,
    float threshold,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& out_core_i,
    Eigen::Tensor<float, 3, Eigen::RowMajor>& out_core_next);

} // namespace ops
} // namespace highnoon

#endif // HIGHNOON_OPS_MPS_EVOLUTION_OP_H_
