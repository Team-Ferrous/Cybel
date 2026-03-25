#ifndef SRC_CONTROLLERS_UTILS_MATRIX_UTILS_H_
#define SRC_CONTROLLERS_UTILS_MATRIX_UTILS_H_

#include <vector>
#include <Eigen/Dense>

namespace MatrixUtils {

// Converts a ragged std::vector<std::vector<float>> into an Eigen dense matrix.
Eigen::MatrixXf ToEigenMatrix(const std::vector<std::vector<float>>& matrix);

// Multiplies a row-major std::vector<std::vector<float>> with a column vector.
std::vector<float> multiply(const std::vector<std::vector<float>>& matrix,
                            const std::vector<float>& vector);

// Multiplies an Eigen dense matrix with a column vector.
std::vector<float> multiply(const Eigen::MatrixXf& matrix,
                            const std::vector<float>& vector);

}  // namespace MatrixUtils

#endif  // SRC_CONTROLLERS_UTILS_MATRIX_UTILS_H_
