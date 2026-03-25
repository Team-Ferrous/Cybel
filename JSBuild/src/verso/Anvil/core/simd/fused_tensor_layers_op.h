// highnoon/_native/ops/fused_tensor_layers_op.h
// Tensor network kernels for Tucker and Tensor-Ring decompositions.
//
// Implements:
// 1. FusedTuckerForward: y = ((x @ U_in) @ G^T) @ U_out^T + b
// 2. FusedTensorRingForward: y_k = Trace(B_1 @ ... @ B_{k,i} @ ... @ B_N)
//
// Optimized for CPU with SIMD and OpenMP parallelization.

#ifndef HIGHNOON_NATIVE_OPS_FUSED_TENSOR_LAYERS_OP_H_
#define HIGHNOON_NATIVE_OPS_FUSED_TENSOR_LAYERS_OP_H_

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include "hnn_simd_common.h"

namespace highnoon { namespace ops {

using Eigen::MatrixXf;
using Eigen::Map;
using Eigen::RowMajor;

/**
 * @brief Tucker Decomposition Forward Pass Kernel.
 * y = ((x @ U_in) @ G^T) @ U_out^T + b
 * 
 * @param x Input tensor [batch, D_in]
 * @param U_in Input factor [D_in, R_in]
 * @param G Core tensor [R_out, R_in]
 * @param U_out Output factor [D_out, R_out]
 * @param bias Optional bias [D_out]
 * @param y Output tensor [batch, D_out]
 */
inline void fused_tucker_forward(
    const float* x, const float* U_in, const float* G, const float* U_out, const float* bias,
    float* y, int64_t batch, int64_t D_in, int64_t D_out, int64_t R_in, int64_t R_out) {
    
    // Efficient projection sequence:
    // 1. x_proj = x @ U_in -> [batch, R_in]
    // 2. core_proj = x_proj @ G^T -> [batch, R_out]
    // 3. y = core_proj @ U_out^T + b -> [batch, D_out]

    Map<const MatrixXf> m_x(x, batch, D_in);
    Map<const MatrixXf> m_U_in(U_in, D_in, R_in);
    Map<const MatrixXf> m_G(G, R_out, R_in);
    Map<const MatrixXf> m_U_out(U_out, D_out, R_out);
    Map<MatrixXf> m_y(y, batch, D_out);

    // Intermediate buffers
    MatrixXf x_proj = m_x * m_U_in;                     // [batch, R_in]
    MatrixXf core_proj = x_proj * m_G.transpose();     // [batch, R_out]
    m_y = core_proj * m_U_out.transpose();             // [batch, D_out]

    if (bias) {
        #pragma omp parallel for
        for (int64_t i = 0; i < batch; ++i) {
            simd_add(m_y.row(i).data(), bias, m_y.row(i).data(), D_out);
        }
    }
}

/**
 * @brief Tensor-Ring Forward Pass Kernel with proper trace contraction.
 * 
 * Implements: y_k = Trace(B_1 @ ... @ B_{site, i} @ ... @ B_N)
 * where B_m = sum_j A[m]_{j} * x_m[j]
 * 
 * @param inputs Pointer to input splits
 * @param cores Pointer to ring cores
 * @param bias Optional bias
 * @param y Output tensor [batch, D_out]
 * @param batch Total batch size
 * @param num_cores N
 * @param ring_rank R
 * @param local_dims_in List of input dims per core
 * @param local_dims_out List of output dims per core
 * @param total_D_out Sum of local_dims_out
 */
inline void fused_tensor_ring_forward(
    const float** inputs, const float** cores, const float* bias, float* y,
    int64_t batch, int num_cores, int ring_rank,
    const int* local_dims_in, const int* local_dims_out, int64_t total_D_out) {

    // Pre-calculate dimension offsets for each core to avoid recomputing in the loop
    std::vector<int> out_offsets(num_cores);
    int current_offset = 0;
    for (int m = 0; m < num_cores; ++m) {
        out_offsets[m] = current_offset;
        current_offset += local_dims_out[m];
    }

    #pragma omp parallel
    {
        // Thread-local buffers to avoid allocations per-batch-element
        // B_sum[m] stores the contraction of site m with its input
        std::vector<MatrixXf> B_sum(num_cores, MatrixXf::Zero(ring_rank, ring_rank));
        
        // B_indexed[m][i] stores the contraction of site m with input at index i
        std::vector<std::vector<MatrixXf>> B_indexed(num_cores);
        for (int m = 0; m < num_cores; ++m) {
            B_indexed[m].resize(local_dims_out[m], MatrixXf::Zero(ring_rank, ring_rank));
        }

        // Prefix and suffix product matrices
        std::vector<MatrixXf> prefix(num_cores + 1, MatrixXf::Identity(ring_rank, ring_rank));
        std::vector<MatrixXf> suffix(num_cores + 1, MatrixXf::Identity(ring_rank, ring_rank));

        #pragma omp for
        for (int64_t b = 0; b < batch; ++b) {
            // 1. Contract each site with its input
            for (int m = 0; m < num_cores; ++m) {
                int d_in = local_dims_in[m];
                int d_out = local_dims_out[m];
                B_sum[m].setZero();
                
                const float* core_ptr = cores[m]; // [R, d_out, d_in, R]
                const float* input_ptr = inputs[m] + b * d_in;

                for (int i = 0; i < d_out; ++i) {
                    B_indexed[m][i].setZero();
                    for (int j = 0; j < d_in; ++j) {
                        float x_val = input_ptr[j];
                        if (std::abs(x_val) < 1e-12f) continue;

                        // core(r1, i, j, r2) = data[r1 * (d_out*d_in*R) + i*(d_in*R) + j*R + r2]
                        for (int r1 = 0; r1 < ring_rank; ++r1) {
                            const float* r1_row = core_ptr + r1 * (d_out * d_in * ring_rank) + (i * d_in + j) * ring_rank;
                            for (int r2 = 0; r2 < ring_rank; ++r2) {
                                B_indexed[m][i](r1, r2) += r1_row[r2] * x_val;
                            }
                        }
                    }
                    B_sum[m] += B_indexed[m][i];
                }
            }

            // 2. Compute prefix and suffix products
            for (int m = 0; m < num_cores; ++m) {
                prefix[m + 1] = prefix[m] * B_sum[m];
            }
            for (int m = num_cores - 1; m >= 0; --m) {
                suffix[m] = B_sum[m] * suffix[m + 1];
            }

            // 3. Compute output y_k = Trace(prefix[m] * B_indexed[m][i] * suffix[m+1])
            float* y_batch = y + b * total_D_out;
            for (int m = 0; m < num_cores; ++m) {
                int d_out = local_dims_out[m];
                int offset = out_offsets[m];
                MatrixXf left = prefix[m];
                MatrixXf right = suffix[m + 1];
                
                for (int i = 0; i < d_out; ++i) {
                    // Trace(L * Bi * R) is faster as a dot product of flattened matrices
                    // Trace(A * B) = sum(A_ij * B_ji) or sum(A_ik * B_kj * C_ji)
                    // We'll just do the multiplication and trace for now
                    y_batch[offset + i] = (left * B_indexed[m][i] * right).trace();
                }
            }

            // 4. Apply bias
            if (bias) {
                simd_add(y_batch, bias, y_batch, total_D_out);
            }
        }
    }
}

// Legacy helper for backward compatibility
inline void tensor_mps_contract(
    const float* left, const float* right, float* out,
    int64_t left_dim, int64_t bond_dim, int64_t right_dim) {
    Map<const MatrixXf> m_left(left, left_dim, bond_dim);
    Map<const MatrixXf> m_right(right, bond_dim, right_dim);
    Map<MatrixXf> m_out(out, left_dim, right_dim);
    m_out = m_left * m_right;
}

}} // namespace
#endif
