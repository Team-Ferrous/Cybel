// highnoon/_native/ops/hnsw_holographic_index.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// HNSW-Accelerated Holographic Retrieval Index
// Enables O(log D) search over holographic memory bundles.

#ifndef HIGHNOON_NATIVE_OPS_HNSW_HOLOGRAPHIC_INDEX_H_
#define HIGHNOON_NATIVE_OPS_HNSW_HOLOGRAPHIC_INDEX_H_

#include "tensorflow/core/framework/op_kernel.h"
#include <hnswlib/hnswlib.h>
#include <vector>
#include <memory>
#include <mutex>

namespace highnoon {
namespace retrieval {

// Configuration for HNSW Holographic Index
struct HNSWIndexConfig {
    int d_model = 512;
    int max_elements = 5000000; // 5M context limit for Lite edition
    int M = 16;
    int ef_construction = 200;
    int ef_search = 128;
};

class HNSWHolographicIndex {
public:
    explicit HNSWHolographicIndex(const HNSWIndexConfig& config)
        : config_(config),
          space_(new hnswlib::L2Space(config.d_model)),
          index_(new hnswlib::HierarchicalNSW<float>(
              space_.get(), config.max_elements, config.M, config.ef_construction)) {
    }

    // Insert a new bundle into the index
    void Insert(int64_t bundle_id, const float* key_vector) {
        std::lock_guard<std::mutex> lock(mutex_);
        index_->addPoint(key_vector, static_cast<size_t>(bundle_id));
    }

    // Bulk insert for initialization
    void BulkInsert(const std::vector<int64_t>& ids, const float* keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < ids.size(); ++i) {
            index_->addPoint(keys + i * config_.d_model, static_cast<size_t>(ids[i]));
        }
    }

    // Search for top-K nearest bundles
    std::vector<std::pair<float, int64_t>> Search(
        const float* query_vector, 
        int k, 
        int ef_search = -1
    ) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        int current_ef = (ef_search > 0) ? ef_search : config_.ef_search;
        index_->setEf(current_ef);

        auto result = index_->searchKnn(query_vector, k);
        std::vector<std::pair<float, int64_t>> sorted_results;
        
        while (!result.empty()) {
            sorted_results.push_back({result.top().first, static_cast<int64_t>(result.top().second)});
            result.pop();
        }
        std::reverse(sorted_results.begin(), sorted_results.end());
        return sorted_results;
    }

    // Memory usage estimation
    size_t EstimateMemoryBytes() const {
        return 0; // hnswlib doesn't expose total memory easily
    }

private:
    HNSWIndexConfig config_;
    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::mutex mutex_;
};

// Resource Implementation for persistent HNSW Index
class HNSWIndexResource : public tensorflow::ResourceBase {
public:
    explicit HNSWIndexResource(const HNSWIndexConfig& config)
        : index_(new HNSWHolographicIndex(config)) {}

    HNSWHolographicIndex* index() { return index_.get(); }

    std::string DebugString() const override { return "HNSWIndexResource"; }

private:
    std::unique_ptr<HNSWHolographicIndex> index_;
};

} // namespace retrieval
} // namespace highnoon

#endif // HIGHNOON_NATIVE_OPS_HNSW_HOLOGRAPHIC_INDEX_H_
