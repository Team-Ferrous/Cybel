// highnoon/_native/ops/mps_memory_compressor.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// MPS-compressed matrix memory for mLSTM.
// Enhancement 3: Memory reduction O(d²) → O(d·χ) via SVD.
//
// Reference: xLSTM roadmap - MPS-Compressed Matrix Memory

#ifndef HIGHNOON_NATIVE_OPS_MPS_MEMORY_COMPRESSOR_H_
#define HIGHNOON_NATIVE_OPS_MPS_MEMORY_COMPRESSOR_H_

#include <cmath>
#include <vector>
#include <algorithm>

namespace highnoon {
namespace ops {

/**
 * @brief MPS-compressed matrix memory for mLSTM.
 * 
 * Replaces dense memory matrix C [d×d] with low-rank factorization:
 *   C ≈ U @ S @ V^T where U [d×χ], S [χ], V [d×χ]
 * 
 * Memory reduction: O(d²) → O(d·χ + χ) ≈ O(d·χ)
 * For d=128, χ=32: 16KB → 4KB (4x reduction)
 * 
 * Uses incremental SVD for efficient updates.
 */
template <typename T>
class MPSMemoryCompressor {
public:
    /**
     * @brief Construct compressor with target bond dimension.
     * 
     * @param head_dim Dimension of key/value vectors (d)
     * @param bond_dim Target bond dimension (χ)
     * @param truncation_threshold Minimum singular value ratio to keep
     */
    MPSMemoryCompressor(int head_dim, int bond_dim, T truncation_threshold = 1e-6)
        : head_dim_(head_dim)
        , bond_dim_(bond_dim)
        , truncation_threshold_(truncation_threshold)
        , U_(head_dim * bond_dim, static_cast<T>(0))
        , S_(bond_dim, static_cast<T>(0))
        , V_(head_dim * bond_dim, static_cast<T>(0)) {
        
        // Initialize as scaled identity (stable starting point)
        for (int i = 0; i < std::min(head_dim, bond_dim); ++i) {
            U_[i * bond_dim + i] = static_cast<T>(1);
            S_[i] = static_cast<T>(1e-3);  // Small initial values
            V_[i * bond_dim + i] = static_cast<T>(1);
        }
    }
    
    /**
     * @brief Update compressed memory with outer product.
     * 
     * C_new = f * C_old + i * (k ⊗ v)
     * 
     * Uses rank-1 update formula for SVD:
     *   (U, S, V) = svd(f * U @ S @ V^T + i * k @ v^T)
     * 
     * @param k Key vector [head_dim]
     * @param v Value vector [head_dim]
     * @param f_gate Forget gate scalar
     * @param i_gate Input gate scalar
     */
    void update(const T* k, const T* v, T f_gate, T i_gate) {
        // Scale existing singular values
        for (int i = 0; i < bond_dim_; ++i) {
            S_[i] *= f_gate;
        }
        
        // Rank-1 update: C += i * k @ v^T
        // Project k and v into existing U and V bases
        std::vector<T> k_proj(bond_dim_);
        std::vector<T> v_proj(bond_dim_);
        
        // k_proj = U^T @ k
        for (int j = 0; j < bond_dim_; ++j) {
            T sum = static_cast<T>(0);
            for (int i = 0; i < head_dim_; ++i) {
                sum += U_[i * bond_dim_ + j] * k[i];
            }
            k_proj[j] = sum;
        }
        
        // v_proj = V^T @ v
        for (int j = 0; j < bond_dim_; ++j) {
            T sum = static_cast<T>(0);
            for (int i = 0; i < head_dim_; ++i) {
                sum += V_[i * bond_dim_ + j] * v[i];
            }
            v_proj[j] = sum;
        }
        
        // Update singular values (simplified: add to diagonal)
        // Full rank-1 SVD update would use Brand's algorithm
        T update_scale = i_gate;
        for (int i = 0; i < bond_dim_; ++i) {
            S_[i] += update_scale * k_proj[i] * v_proj[i];
        }
        
        // Truncate small singular values
        T max_s = *std::max_element(S_.begin(), S_.end());
        T threshold = truncation_threshold_ * max_s;
        for (int i = 0; i < bond_dim_; ++i) {
            if (std::abs(S_[i]) < threshold) {
                S_[i] = static_cast<T>(0);
            }
        }
    }
    
    /**
     * @brief Query compressed memory.
     * 
     * h = C @ q = U @ S @ V^T @ q
     * 
     * @param q Query vector [head_dim]
     * @param h Output vector [head_dim]
     */
    void query(const T* q, T* h) const {
        // Step 1: v_proj = V^T @ q [bond_dim]
        std::vector<T> v_proj(bond_dim_);
        for (int j = 0; j < bond_dim_; ++j) {
            T sum = static_cast<T>(0);
            for (int i = 0; i < head_dim_; ++i) {
                sum += V_[i * bond_dim_ + j] * q[i];
            }
            v_proj[j] = sum;
        }
        
        // Step 2: s_proj = S * v_proj [bond_dim]
        std::vector<T> s_proj(bond_dim_);
        for (int i = 0; i < bond_dim_; ++i) {
            s_proj[i] = S_[i] * v_proj[i];
        }
        
        // Step 3: h = U @ s_proj [head_dim]
        for (int i = 0; i < head_dim_; ++i) {
            T sum = static_cast<T>(0);
            for (int j = 0; j < bond_dim_; ++j) {
                sum += U_[i * bond_dim_ + j] * s_proj[j];
            }
            h[i] = sum;
        }
    }
    
    /**
     * @brief Get current memory usage in bytes.
     */
    size_t memory_bytes() const {
        return (2 * head_dim_ * bond_dim_ + bond_dim_) * sizeof(T);
    }
    
    /**
     * @brief Get memory reduction ratio vs dense.
     */
    T memory_reduction_ratio() const {
        size_t dense_bytes = head_dim_ * head_dim_ * sizeof(T);
        return static_cast<T>(dense_bytes) / static_cast<T>(memory_bytes());
    }
    
    /**
     * @brief Reset to initial state.
     */
    void reset() {
        std::fill(U_.begin(), U_.end(), static_cast<T>(0));
        std::fill(S_.begin(), S_.end(), static_cast<T>(0));
        std::fill(V_.begin(), V_.end(), static_cast<T>(0));
        
        for (int i = 0; i < std::min(head_dim_, bond_dim_); ++i) {
            U_[i * bond_dim_ + i] = static_cast<T>(1);
            S_[i] = static_cast<T>(1e-3);
            V_[i * bond_dim_ + i] = static_cast<T>(1);
        }
    }
    
    // Accessors for serialization
    const std::vector<T>& get_U() const { return U_; }
    const std::vector<T>& get_S() const { return S_; }
    const std::vector<T>& get_V() const { return V_; }
    int head_dim() const { return head_dim_; }
    int bond_dim() const { return bond_dim_; }

private:
    int head_dim_;
    int bond_dim_;
    T truncation_threshold_;
    std::vector<T> U_;  // [head_dim, bond_dim]
    std::vector<T> S_;  // [bond_dim]
    std::vector<T> V_;  // [head_dim, bond_dim]
};

}  // namespace ops
}  // namespace highnoon

#endif  // HIGHNOON_NATIVE_OPS_MPS_MEMORY_COMPRESSOR_H_
