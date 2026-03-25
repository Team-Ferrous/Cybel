#ifndef N4SID_SOLVER_H_
#define N4SID_SOLVER_H_

#include "tensorflow/core/framework/tensor.h"
#include "Eigen/Core"
#include "Eigen/QR"
#include "Eigen/SVD"
#include <tuple>

namespace tensorflow {

template <typename T>
class N4SID_Solver {
 public:
  // Computes the system matrices A, B, C, D from input/output data.
  // y: Output data (matrix of size num_outputs x num_samples)
  // u: Input data (matrix of size num_inputs x num_samples)
  // order: The order of the system to identify.
  // Returns a tuple of matrices (A, B, C, D).
  std::tuple<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
             Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
  compute(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& y,
          const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& u,
          int order);
};

}  // namespace tensorflow

#endif  // N4SID_SOLVER_H_
