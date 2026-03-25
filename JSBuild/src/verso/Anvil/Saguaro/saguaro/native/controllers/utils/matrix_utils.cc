#include "controllers/utils/matrix_utils.h"

#include <algorithm>
#include <stdexcept>

namespace MatrixUtils {

Eigen::MatrixXf ToEigenMatrix(const std::vector<std::vector<float>>& matrix) {
    if (matrix.empty()) {
        return Eigen::MatrixXf();
    }

    const Eigen::Index rows = static_cast<Eigen::Index>(matrix.size());
    Eigen::Index cols = 0;
    for (const auto& row : matrix) {
        cols = std::max(cols, static_cast<Eigen::Index>(row.size()));
    }

    if (cols == 0) {
        return Eigen::MatrixXf();
    }

    Eigen::MatrixXf eigen_matrix(rows, cols);
    eigen_matrix.setZero();
    for (Eigen::Index i = 0; i < rows; ++i) {
        const auto& row = matrix[static_cast<std::size_t>(i)];
        const Eigen::Index limit = std::min(cols, static_cast<Eigen::Index>(row.size()));
        for (Eigen::Index j = 0; j < limit; ++j) {
            eigen_matrix(i, j) = row[static_cast<std::size_t>(j)];
        }
    }
    return eigen_matrix;
}

std::vector<float> multiply(const std::vector<std::vector<float>>& matrix,
                            const std::vector<float>& vector) {
    if (matrix.empty()) {
        return {};
    }

    const std::size_t cols = vector.size();
    std::vector<float> result(matrix.size(), 0.0f);

    if (cols == 0) {
        return result;
    }

    for (std::size_t i = 0; i < matrix.size(); ++i) {
        const auto& row = matrix[i];
        if (row.size() != cols) {
            throw std::invalid_argument("MatrixUtils::multiply dimension mismatch");
        }
        float acc = 0.0f;
        for (std::size_t j = 0; j < cols; ++j) {
            acc += row[j] * vector[j];
        }
        result[i] = acc;
    }
    return result;
}

std::vector<float> multiply(const Eigen::MatrixXf& matrix,
                            const std::vector<float>& vector) {
    if (matrix.size() == 0) {
        return {};
    }
    if (static_cast<std::size_t>(matrix.cols()) != vector.size()) {
        throw std::invalid_argument("MatrixUtils::multiply dimension mismatch");
    }
    if (vector.empty()) {
        return std::vector<float>(static_cast<std::size_t>(matrix.rows()), 0.0f);
    }
    Eigen::Map<const Eigen::VectorXf> vec_map(vector.data(), static_cast<Eigen::Index>(vector.size()));
    Eigen::VectorXf product = matrix * vec_map;
    return std::vector<float>(product.data(), product.data() + product.size());
}

}  // namespace MatrixUtils
