#ifndef VERSO_LORENTZ_EXP_H_
#define VERSO_LORENTZ_EXP_H_

#include <cmath>
#include <algorithm>
#include "Eigen/Core"

namespace verso {
namespace lorentz {

using Eigen::MatrixXf;
using Eigen::VectorXf;

// Numerical constant for the smooth transition (Parabolic case)
constexpr float kEpsilon = 1e-6f;

/**
 * @brief Computes M = exp(X) for the Lorentz Lie Algebra X = [[0, a^T], [a, S]].
 * * Uses the analytic Rodrigues-like formula for the Lorentz group SO(1, D-1),
 * which ensures O(D^3) stability and speed by avoiding generic matrix exponential
 * algorithms. The computation splits into Elliptic, Hyperbolic, and Parabolic
 * cases based on the magnitude squared (rho_sq).
 * * @param X The D x D Lie Algebra matrix.
 * @param D_hyp The hyperbolic dimension D = D_spatial + 1.
 * @param D_spatial The spatial dimension D_s = D - 1.
 * @return Eigen::MatrixXf The D x D Lorentz transformation M = exp(X).
 */
inline MatrixXf matrix_exp_lorentz(const MatrixXf& X, int D_hyp, int D_spatial) {
    if (D_hyp == 1) return MatrixXf::Identity(1, 1);
    
    // Extract boost vector 'a' and rotation matrix 'S'
    const VectorXf a = X.block(1, 0, D_spatial, 1);
    const MatrixXf S = X.block(1, 1, D_spatial, D_spatial);

    // Compute fundamental invariants:
    const float theta_a_sq = a.squaredNorm();
    const MatrixXf S_sq = S * S;
    // theta_s_sq = -0.5 * Tr(S^T S). For skew-symmetric S, S^T S = -S^2.
    const float theta_s_sq = -0.5f * S_sq.trace(); 
    
    // The squared magnitude of the Lie algebra element (Lorentz norm)
    const float rho_sq = theta_a_sq + theta_s_sq; 

    // Intermediate matrix products needed for the analytic formula:
    const MatrixXf P = a * a.transpose(); // a * a^T
    const MatrixXf Z = P + S_sq;         // P + S^2
    const MatrixXf V = S * P + P * S;    // S * P + P * S (This term ensures correctness when a and S do not commute)
    
    const VectorXf S_a = S * a;          // S * a
    const VectorXf S_sq_a = S * S_a;     // S^2 * a

    MatrixXf M = MatrixXf::Zero(D_hyp, D_hyp);

    // --- Coefficients for Taylor series expansion for Parabolic case ---
    float c1, c2, c3; 
    
    if (std::abs(rho_sq) < kEpsilon) {
        // --- Case 3: Parabolic/Taylor Approximation (rho_sq ~ 0) ---
        // Use high-order Taylor coefficients for numerical stability near zero.
        c1 = 1.0f - rho_sq / 6.0f; // 1/1! - rho_sq/3!
        c2 = 0.5f - rho_sq / 24.0f; // 1/2! - rho_sq/4!
        c3 = 1.0f / 6.0f - rho_sq / 120.0f; // 1/3! - rho_sq/5!
        
    } else if (rho_sq > 0) {
        // --- Case 1: Elliptic (Rotation-like) ---
        const float rho = std::sqrt(rho_sq);
        // Use robust sin/cos functions (or explicit implementation of sinc/cosc)
        c1 = std::sin(rho) / rho;
        c2 = (1.0f - std::cos(rho)) / rho_sq;
        c3 = (rho - std::sin(rho)) / (rho_sq * rho);

    } else { // rho_sq < 0
        // --- Case 2: Hyperbolic (Boost-like) ---
        const float rho = std::sqrt(-rho_sq);
        // Use robust sinh/cosh functions (or explicit implementation of sinhc/coshc)
        c1 = std::sinh(rho) / rho;
        c2 = (std::cosh(rho) - 1.0f) / (-rho_sq);
        c3 = (std::sinh(rho) - rho) / (-rho_sq * rho);
    }
    
    // -------------------------------------------------------------------------
    // --- Assemble Matrix M = exp(X) using coefficients c1, c2, c3 (Closed Form) ---
    // -------------------------------------------------------------------------
    
    // 1. M_{0,0} (Time-Time Block A: scalar)
    M(0, 0) = 1.0f + c2 * theta_a_sq + c3 * a.dot(S_a);
    
    // 2. M_{0, 1:D_s} (Time-Spatial Block b^T: 1 x D_s row vector)
    // b^T = c1 * a^T + c2 * (S * a)^T + c3 * (S^2 * a)^T
    VectorXf b = c1 * a + c2 * S_a + c3 * S_sq_a;
    M.block(0, 1, 1, D_spatial) = b.transpose();
    
    // 3. M_{1:D_s, 0} (Spatial-Time Block b: D_s x 1 column vector)
    M.block(1, 0, D_spatial, 1) = b;
    
    // 4. M_{1:D_s, 1:D_s} (Spatial-Spatial Block R: D_s x D_s matrix)
    // R = I + c1 * S + c2 * Z + c3 * V
    MatrixXf R = MatrixXf::Identity(D_spatial, D_spatial) + c1 * S + c2 * Z + c3 * V;
    M.block(1, 1, D_spatial, D_spatial) = R;
    
    return M;
}
} // namespace lorentz
} // namespace verso

#endif // VERSO_LORENTZ_EXP_H_
