// saguaro.native/ops/dynamic_sparse_gate.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Dynamic sparse updates for xLSTM.
// Enhancement 4: Top-K selective neuron updates for 30-50% compute reduction.
//
// Reference: xLSTM roadmap - Dynamic Gated Sparse Updates

#ifndef SAGUARO_NATIVE_OPS_DYNAMIC_SPARSE_GATE_H_
#define SAGUARO_NATIVE_OPS_DYNAMIC_SPARSE_GATE_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace saguaro {
namespace ops {

/**
 * @brief Dynamic sparse gate selection for mLSTM.
 * 
 * Selects top-K neurons for full update based on gate magnitude.
 * Other neurons use cached values or skip updates.
 * 
 * Expected compute reduction: 30-50% with minimal accuracy loss.
 */
template <typename T>
class DynamicSparseGate {
public:
    /**
     * @brief Construct sparse gate with target sparsity.
     * 
     * @param size Number of neurons
     * @param topk_ratio Ratio of neurons to update (0.5 = 50%)
     */
    DynamicSparseGate(int size, T topk_ratio = 0.5)
        : size_(size)
        , topk_count_(static_cast<int>(size * topk_ratio))
        , indices_(size)
        , magnitudes_(size)
        , selected_mask_(size, false)
        , cached_values_(size, static_cast<T>(0)) {
        
        // Ensure at least 10% are updated
        topk_count_ = std::max(topk_count_, size_ / 10);
        topk_count_ = std::min(topk_count_, size_);
        
        // Initialize indices
        std::iota(indices_.begin(), indices_.end(), 0);
    }
    
    /**
     * @brief Select top-K neurons based on gate magnitude.
     * 
     * @param gates Gate values [size]
     * @return Reference to selection mask
     */
    const std::vector<bool>& select(const T* gates) {
        // Compute magnitudes
        for (int i = 0; i < size_; ++i) {
            magnitudes_[i] = std::abs(gates[i]);
        }
        
        // Partial sort to find top-K
        std::iota(indices_.begin(), indices_.end(), 0);
        std::nth_element(
            indices_.begin(),
            indices_.begin() + topk_count_,
            indices_.end(),
            [this](int a, int b) {
                return magnitudes_[a] > magnitudes_[b];
            }
        );
        
        // Build mask
        std::fill(selected_mask_.begin(), selected_mask_.end(), false);
        for (int i = 0; i < topk_count_; ++i) {
            selected_mask_[indices_[i]] = true;
        }
        
        return selected_mask_;
    }
    
    /**
     * @brief Apply sparse update: full update for selected, cached for others.
     * 
     * @param new_values New computed values [size]
     * @param output Output values [size]
     */
    void apply(const T* new_values, T* output) {
        for (int i = 0; i < size_; ++i) {
            if (selected_mask_[i]) {
                output[i] = new_values[i];
                cached_values_[i] = new_values[i];  // Update cache
            } else {
                output[i] = cached_values_[i];  // Use cached
            }
        }
    }
    
    /**
     * @brief Get sparsity statistics.
     */
    T get_actual_sparsity() const {
        int active_count = std::count(selected_mask_.begin(), selected_mask_.end(), true);
        return static_cast<T>(size_ - active_count) / static_cast<T>(size_);
    }
    
    /**
     * @brief Reset cached values.
     */
    void reset() {
        std::fill(cached_values_.begin(), cached_values_.end(), static_cast<T>(0));
        std::fill(selected_mask_.begin(), selected_mask_.end(), true);
    }
    
    /**
     * @brief Set new top-K ratio.
     */
    void set_topk_ratio(T ratio) {
        topk_count_ = static_cast<int>(size_ * ratio);
        topk_count_ = std::max(topk_count_, size_ / 10);
        topk_count_ = std::min(topk_count_, size_);
    }
    
    int size() const { return size_; }
    int topk_count() const { return topk_count_; }

private:
    int size_;
    int topk_count_;
    std::vector<int> indices_;
    std::vector<T> magnitudes_;
    std::vector<bool> selected_mask_;
    std::vector<T> cached_values_;
};

/**
 * @brief SIMD-friendly sparse pattern for vectorized updates.
 * 
 * Groups neurons into blocks of 8 (AVX2) or 16 (AVX512) for efficient
 * vectorized sparse updates.
 */
template <typename T>
class SIMDSparsePattern {
public:
    /**
     * @brief Construct SIMD-aligned sparse pattern.
     * 
     * @param size Number of neurons (should be multiple of block_size)
     * @param block_size SIMD block size (8 for AVX2, 16 for AVX512)
     * @param block_topk_ratio Ratio of blocks to update
     */
    SIMDSparsePattern(int size, int block_size = 8, T block_topk_ratio = 0.5)
        : size_(size)
        , block_size_(block_size)
        , num_blocks_((size + block_size - 1) / block_size)
        , topk_blocks_(static_cast<int>(num_blocks_ * block_topk_ratio))
        , block_mask_(num_blocks_, false)
        , block_scores_(num_blocks_, static_cast<T>(0)) {
        
        topk_blocks_ = std::max(topk_blocks_, 1);
        topk_blocks_ = std::min(topk_blocks_, num_blocks_);
    }
    
    /**
     * @brief Select top-K blocks based on aggregate gate magnitude.
     * 
     * @param gates Gate values [size]
     */
    void select_blocks(const T* gates) {
        // Compute block scores (sum of magnitudes)
        for (int b = 0; b < num_blocks_; ++b) {
            T score = static_cast<T>(0);
            int start = b * block_size_;
            int end = std::min(start + block_size_, size_);
            for (int i = start; i < end; ++i) {
                score += std::abs(gates[i]);
            }
            block_scores_[b] = score;
        }
        
        // Find top-K blocks
        std::vector<int> indices(num_blocks_);
        std::iota(indices.begin(), indices.end(), 0);
        std::nth_element(
            indices.begin(),
            indices.begin() + topk_blocks_,
            indices.end(),
            [this](int a, int b) {
                return block_scores_[a] > block_scores_[b];
            }
        );
        
        // Build block mask
        std::fill(block_mask_.begin(), block_mask_.end(), false);
        for (int i = 0; i < topk_blocks_; ++i) {
            block_mask_[indices[i]] = true;
        }
    }
    
    /**
     * @brief Check if a block should be updated.
     */
    bool is_block_active(int block_idx) const {
        return block_mask_[block_idx];
    }
    
    /**
     * @brief Get block boundaries.
     */
    void get_block_range(int block_idx, int& start, int& end) const {
        start = block_idx * block_size_;
        end = std::min(start + block_size_, size_);
    }
    
    int num_blocks() const { return num_blocks_; }
    int block_size() const { return block_size_; }

private:
    int size_;
    int block_size_;
    int num_blocks_;
    int topk_blocks_;
    std::vector<bool> block_mask_;
    std::vector<T> block_scores_;
};

}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_DYNAMIC_SPARSE_GATE_H_
