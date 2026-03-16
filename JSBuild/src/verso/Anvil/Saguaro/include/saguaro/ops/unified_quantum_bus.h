// saguaro/native/ops/unified_quantum_bus.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file unified_quantum_bus.h
 * @brief UQHA v3.1 Unified Quantum Bus (C++ side).
 *
 * Implements centralized frequency mask caching and quantum state sharing
 * between native ops.
 */

#ifndef SAGUARO_NATIVE_OPS_UNIFIED_QUANTUM_BUS_H_
#define SAGUARO_NATIVE_OPS_UNIFIED_QUANTUM_BUS_H_

#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <utility>
#include <tuple>
#include <cstring>

#include "qhd_spatial_common.h"

namespace saguaro {
namespace ops {

/**
 * Singleton cache for frequency masks to avoid redundant computation across blocks.
 * Part of Phase 3.3: Frequency Mask Caching (S3).
 */
class FrequencyMaskCache {
public:
    static FrequencyMaskCache& instance() {
        static FrequencyMaskCache cache;
        return cache;
    }
    
    // Non-copyable
    FrequencyMaskCache(const FrequencyMaskCache&) = delete;
    FrequencyMaskCache& operator=(const FrequencyMaskCache&) = delete;

    const float* get_masks(const hsmn::qhd_spatial::QHDSpatialConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Cache key based on relevant config params: (K, D, overlap, mode)
        auto key = std::make_tuple(config.num_paths, config.hd_dim, config.freq_overlap, config.freq_mask_mode);
        auto it = masks_.find(key);
        if (it == masks_.end()) {
            int K = config.num_paths;
            int D = config.hd_dim;
            std::vector<float> new_masks(K * D);
            hsmn::qhd_spatial::compute_frequency_masks(new_masks.data(), config);
            masks_[key] = std::move(new_masks);
            return masks_[key].data();
        }
        return it->second.data();
    }
    
private:
    FrequencyMaskCache() = default;
    
    struct tuple_hash {
        template <typename... T>
        std::size_t operator()(const std::tuple<T...>& t) const {
            return hash_tuple_impl(t, std::make_index_sequence<sizeof...(T)>{});
        }
    private:
        template <typename Tuple, std::size_t... Is>
        std::size_t hash_tuple_impl(const Tuple& t, std::index_sequence<Is...>) const {
            std::size_t res = 0;
            auto hash_all = { (hash_combine(res, std::get<Is>(t)), 0)... };
            (void)hash_all;
            return res;
        }

        template <typename T>
        void hash_combine(std::size_t& seed, const T& v) const {
            seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };
    
    std::mutex mutex_;
    std::unordered_map<std::tuple<int, int, float, int>, std::vector<float>, tuple_hash> masks_;
};

/**
 * Global hub for quantum state sharing between components.
 * Part of Phase 3.2: Amplitude Warm Start (S2).
 */
class UnifiedQuantumBus {
public:
    static UnifiedQuantumBus& instance() {
        static UnifiedQuantumBus bus;
        return bus;
    }

    void set_born_amplitudes(const float* amplitudes, int B, int K) {
        std::lock_guard<std::mutex> lock(mutex_);
        born_amplitudes_.assign(amplitudes, amplitudes + B * K);
        batch_size_ = B;
        num_paths_ = K;
    }

    void get_born_amplitudes(float* amplitudes, int B, int K) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (born_amplitudes_.size() == static_cast<size_t>(B * K)) {
            std::memcpy(amplitudes, born_amplitudes_.data(), B * K * sizeof(float));
        } else {
            // Uniform fallback if sizes don't match or not set
            float val = 1.0f / K;
            for (int i = 0; i < B * K; ++i) amplitudes[i] = val;
        }
    }

private:
    UnifiedQuantumBus() : batch_size_(0), num_paths_(0) {}
    std::mutex mutex_;
    std::vector<float> born_amplitudes_;
    int batch_size_;
    int num_paths_;
};

} // namespace ops
} // namespace saguaro

#endif // SAGUARO_NATIVE_OPS_UNIFIED_QUANTUM_BUS_H_
