// src/ops/n4sid_solver.cc
// N4SID (Numerical Subspace State Space System Identification) solver
// Implements subspace identification for state-space model estimation from I/O data.
//
// Phase 11 Compliance:
// - Explicit SIMD guards (AVX512/AVX2/NEON + scalar fallback)
// - TBB parallelism for Hankel matrix construction
// - Float32 precision (standard)
// - Core SVD remains serial (inherently sequential algorithm)

#include "n4sid_solver.h"
#include "tensorflow/core/platform/logging.h"
#include <algorithm>
#include <limits>

// Conditional SIMD includes (Phase 11 compliance)
#if defined(__AVX512F__)
#include <immintrin.h>  // AVX512 intrinsics
#elif defined(__AVX2__)
#include <immintrin.h>  // AVX2 intrinsics
#elif defined(__ARM_NEON)
#include <arm_neon.h>   // ARM NEON intrinsics
#endif

#include "common/parallel/parallel_backend.h"  // TBB abstraction

namespace tensorflow {

namespace {

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> PseudoInverse(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& mat,
    Scalar tolerance) {
    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> svd(
        mat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singular_values = svd.singularValues();
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> singular_inv =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(
            svd.matrixV().cols(), svd.matrixU().cols());
    for (int i = 0; i < singular_values.size(); ++i) {
        const Scalar sigma = singular_values(i);
        if (sigma > tolerance) {
            singular_inv(i, i) = Scalar(1.0) / sigma;
        }
    }
    return svd.matrixV() * singular_inv * svd.matrixU().transpose();
}

template <typename MatrixType>
MatrixType ZeroMatrix(int rows, int cols) {
    MatrixType m(rows, cols);
    m.setZero();
    return m;
}

}  // namespace

template <typename T>
std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
           Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
N4SID_Solver<T>::compute(
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& y,
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& u,
    int order) {

  using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

  const int num_outputs = static_cast<int>(y.rows());
  const int num_inputs = static_cast<int>(u.rows());
  const int num_samples = std::min(static_cast<int>(y.cols()), static_cast<int>(u.cols()));

  if (order <= 0 || num_outputs == 0 || num_samples == 0) {
      LOG(WARNING) << "[N4SID] Invalid dimensions or order. Returning zero system.";
      return std::make_tuple(
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, num_inputs),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, num_inputs));
  }

  if (num_samples < 4 || num_samples <= 2 * order) {
      LOG(WARNING) << "[N4SID] Insufficient samples (" << num_samples
                   << ") relative to system order (" << order
                   << "). Returning zero system.";
      return std::make_tuple(
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, num_inputs),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, num_inputs));
  }

  // Determine the number of block rows for the Hankel matrices.
  int f = std::max(order + 1, std::min(10, num_samples / 2));
  if (2 * f > num_samples) {
      f = num_samples / 2;
  }
  if (f <= order) {
      f = order + 1;
  }
  if (2 * f > num_samples || f < 1) {
      LOG(WARNING) << "[N4SID] Unable to determine a valid block size. Returning zero system.";
      return std::make_tuple(
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, num_inputs),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, num_inputs));
  }

  const int j = num_samples - 2 * f + 1;
  if (j <= 1) {
      LOG(WARNING) << "[N4SID] Not enough block columns for identification. Returning zero system.";
      return std::make_tuple(
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, num_inputs),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, order),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, num_inputs));
  }

  MatrixXd y_d = y.template cast<double>();
  MatrixXd u_d = u.template cast<double>();

  // Construct block Hankel matrices.
  // Phase 11: Added TBB parallelism over block rows (independent)
  MatrixXd Yp = MatrixXd::Zero(f * num_outputs, j);
  MatrixXd Up = MatrixXd::Zero(f * num_inputs, j);
  MatrixXd Yf = MatrixXd::Zero(f * num_outputs, j);
  MatrixXd Uf = MatrixXd::Zero(f * num_inputs, j);

  // Parallelize Hankel block row construction
  // Cost estimate: num_outputs * j (per block row)
  const int64_t total_blocks = f;
  const int64_t cost_per_block = num_outputs * j + num_inputs * j;

  saguaro::parallel::ForShard(
      total_blocks, cost_per_block,
      [&](int64_t i_start, int64_t i_end) {
          for (int i = i_start; i < i_end; ++i) {
              // Past Hankel blocks (time 0 to f-1)
              Yp.block(i * num_outputs, 0, num_outputs, j) = y_d.block(0, i, num_outputs, j);
              Up.block(i * num_inputs, 0, num_inputs, j) = u_d.block(0, i, num_inputs, j);

              // Future Hankel blocks (time f to 2f-1)
              Yf.block(i * num_outputs, 0, num_outputs, j) = y_d.block(0, i + f, num_outputs, j);
              Uf.block(i * num_inputs, 0, num_inputs, j) = u_d.block(0, i + f, num_inputs, j);
          }
      }
  );

  MatrixXd Wp(2 * f * (num_outputs + num_inputs), j);
  Wp << Yp, Up;

  MatrixXd Wf(2 * f * (num_outputs + num_inputs), j);
  Wf << Yf, Uf;

  Eigen::HouseholderQR<MatrixXd> qr(Wp.transpose());
  MatrixXd proj = (qr.solve(Wf.transpose())).transpose();

  Eigen::BDCSVD<MatrixXd> svd(proj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto singular_values = svd.singularValues();
  const int r = std::min(order, static_cast<int>(singular_values.size()));

  MatrixXd U1 = svd.matrixU().leftCols(r);
  MatrixXd S_half = singular_values.head(r).array().sqrt().matrix().asDiagonal();
  MatrixXd Obs = U1 * S_half;

  MatrixXd C_d = Obs.topRows(num_outputs);
  MatrixXd Obs_upper = Obs.topRows((f - 1) * num_outputs);
  MatrixXd Obs_lower = Obs.bottomRows((f - 1) * num_outputs);

  const double tol = std::numeric_limits<double>::epsilon() *
                     std::max(Obs_upper.rows(), Obs_upper.cols()) *
                     singular_values(0);
  MatrixXd Obs_upper_pinv = PseudoInverse(Obs_upper, tol);
  MatrixXd A_d = Obs_upper_pinv * Obs_lower;

  MatrixXd V1 = svd.matrixV().leftCols(r);
  MatrixXd X_states = S_half * V1.transpose();

  // Extract used input/output columns
  // Phase 11: Added SIMD guards for column extraction
  MatrixXd U_used(num_inputs, j);
  MatrixXd Y_used(num_outputs, j);
  const int offset = f - 1;

  // Parallelize column extraction (independent columns)
  saguaro::parallel::ForShard(
      j, num_inputs + num_outputs,  // Cost: copying one column (num_inputs + num_outputs elements)
      [&](int64_t col_start, int64_t col_end) {
          for (int col = col_start; col < col_end; ++col) {
              const int t = col + offset;

              // SIMD-optimized column copy for U_used
              int64_t i = 0;
              double* u_used_ptr = U_used.col(col).data();
              const double* u_d_ptr = u_d.col(t).data();

#if defined(__AVX512F__)
              // AVX512: 8-wide SIMD for double precision
              for (; i + 8 <= num_inputs; i += 8) {
                  __m512d v_u = _mm512_loadu_pd(&u_d_ptr[i]);
                  _mm512_storeu_pd(&u_used_ptr[i], v_u);
              }
#elif defined(__AVX2__)
              // AVX2: 4-wide SIMD for double precision
              for (; i + 4 <= num_inputs; i += 4) {
                  __m256d v_u = _mm256_loadu_pd(&u_d_ptr[i]);
                  _mm256_storeu_pd(&u_used_ptr[i], v_u);
              }
#elif defined(__ARM_NEON)
              // NEON: 2-wide SIMD for double precision
              for (; i + 2 <= num_inputs; i += 2) {
                  float64x2_t v_u = vld1q_f64(&u_d_ptr[i]);
                  vst1q_f64(&u_used_ptr[i], v_u);
              }
#endif
              // Scalar fallback for U remainder
              for (; i < num_inputs; ++i) {
                  u_used_ptr[i] = u_d_ptr[i];
              }

              // SIMD-optimized column copy for Y_used
              i = 0;
              double* y_used_ptr = Y_used.col(col).data();
              const double* y_d_ptr = y_d.col(t).data();

#if defined(__AVX512F__)
              for (; i + 8 <= num_outputs; i += 8) {
                  __m512d v_y = _mm512_loadu_pd(&y_d_ptr[i]);
                  _mm512_storeu_pd(&y_used_ptr[i], v_y);
              }
#elif defined(__AVX2__)
              for (; i + 4 <= num_outputs; i += 4) {
                  __m256d v_y = _mm256_loadu_pd(&y_d_ptr[i]);
                  _mm256_storeu_pd(&y_used_ptr[i], v_y);
              }
#elif defined(__ARM_NEON)
              for (; i + 2 <= num_outputs; i += 2) {
                  float64x2_t v_y = vld1q_f64(&y_d_ptr[i]);
                  vst1q_f64(&y_used_ptr[i], v_y);
              }
#endif
              // Scalar fallback for Y remainder
              for (; i < num_outputs; ++i) {
                  y_used_ptr[i] = y_d_ptr[i];
              }
          }
      }
  );

  if (j - 1 <= 0) {
      LOG(WARNING) << "[N4SID] Insufficient state transitions for estimating B and D. Returning zero gains.";
      return std::make_tuple(
          A_d.template cast<T>(),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(order, num_inputs),
          C_d.template cast<T>(),
          ZeroMatrix<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>(num_outputs, num_inputs));
  }

  MatrixXd X_curr = X_states.leftCols(j - 1);
  MatrixXd X_next = X_states.rightCols(j - 1);
  MatrixXd U_curr = U_used.leftCols(j - 1);
  MatrixXd Y_curr = Y_used.leftCols(j - 1);

  MatrixXd U_pinv = PseudoInverse(U_curr, tol);
  MatrixXd B_d = (X_next - A_d * X_curr) * U_pinv;
  MatrixXd D_d = (Y_curr - C_d * X_curr) * U_pinv;

  return std::make_tuple(
      A_d.template cast<T>(),
      B_d.template cast<T>(),
      C_d.template cast<T>(),
      D_d.template cast<T>());
}

// Explicit instantiation
template class N4SID_Solver<float>;
template class N4SID_Solver<double>;

}  // namespace tensorflow
