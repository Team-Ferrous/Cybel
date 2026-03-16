// saguaro/_native/ops/common/tensor_stream_pool.cc
// Copyright 2025-2026 Verso Industries (Author: Michael B. Zimmerman)
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
 * @file tensor_stream_pool.cc
 * @brief Implementation of TensorStreamPool zero-copy inter-kernel streaming.
 *
 * This is part of the TensorStreamPool C++ Enhancement Roadmap Phase 0.
 */

#include "tensor_stream_pool.h"
#include <stdexcept>
#include <cmath>
#include <iostream>  // Phase 4.3: Warning logging

namespace saguaro {
namespace ops {

// =============================================================================
// BUCKET SIZE COMPUTATION
// =============================================================================

size_t TensorStreamPool::ComputeBucket(size_t size_bytes) {
    // Minimum bucket size
    if (size_bytes <= kMinBufferSize) {
        return kMinBufferSize;
    }
    
    // Round up to next power of 2 for efficient reuse
    // This reduces fragmentation by allowing slightly larger buffers to serve
    // smaller requests within the same bucket
    size_t bucket = kMinBufferSize;
    size_t max_pool = GetMaxPoolSizeBytes();
    while (bucket < size_bytes && bucket < max_pool) {
        bucket *= 2;
    }
    
    return bucket;
}

// =============================================================================
// BUFFER FINDING
// =============================================================================

TensorStreamPool::BufferEntry* TensorStreamPool::FindReusableBuffer(size_t bucket_size) {
    auto it = buffers_.find(bucket_size);
    if (it == buffers_.end()) {
        return nullptr;
    }
    
    // Find first available buffer in this bucket
    for (auto& entry : it->second) {
        if (!entry.in_use) {
            return &entry;
        }
    }
    
    return nullptr;
}

TensorStreamPool::BufferEntry* TensorStreamPool::FindByPointer(float* ptr) {
    if (ptr == nullptr) return nullptr;
    
    for (auto& [bucket, entries] : buffers_) {
        for (auto& entry : entries) {
            if (entry.aligned_ptr == ptr) {
                return &entry;
            }
        }
    }
    
    return nullptr;
}

// =============================================================================
// ACQUIRE
// =============================================================================

float* TensorStreamPool::Acquire(size_t size_bytes, const char* producer_hint) {
    if (size_bytes == 0) {
        return nullptr;
    }
    
    stats_.acquire_count++;
    
    // Align size and compute bucket
    size_t aligned_size = AlignToSIMD(size_bytes);
    size_t bucket_size = ComputeBucket(aligned_size);
    
    // Phase 4.2: Check configurable pool size limit
    size_t max_pool = GetMaxPoolSizeBytes();
    if (stats_.current_usage_bytes + bucket_size > max_pool) {
        // Pool would exceed limit - refuse allocation
        // Future: implement LRU eviction for graceful degradation
        std::cerr << "[TensorStreamPool] ERROR: Pool limit exceeded. "
                  << "Current: " << (stats_.current_usage_bytes / (1024*1024)) << " MB, "
                  << "Requested: " << (bucket_size / (1024*1024)) << " MB, "
                  << "Limit: " << (max_pool / (1024*1024*1024)) << " GB" << std::endl;
        return nullptr;
    }
    
    // Phase 4.3: Log warning when approaching limit
    float usage_ratio = static_cast<float>(stats_.current_usage_bytes + bucket_size) / max_pool;
    if (usage_ratio > GetWarningThreshold()) {
        static int warning_count = 0;
        if (warning_count++ % 100 == 0) {  // Throttle warnings
            std::cerr << "[TensorStreamPool] WARNING: Pool at "
                      << static_cast<int>(usage_ratio * 100) << "% capacity ("
                      << ((stats_.current_usage_bytes + bucket_size) / (1024*1024)) << " / "
                      << (max_pool / (1024*1024)) << " MB)" << std::endl;
        }
    }
    
    // Try to reuse existing buffer
    BufferEntry* entry = FindReusableBuffer(bucket_size);
    if (entry != nullptr) {
        stats_.reuse_count++;
        entry->in_use = true;
        entry->handed_off = false;
        entry->current_owner = producer_hint;
        entry->acquired_at = std::chrono::steady_clock::now();
        
        stats_.current_usage_bytes += entry->size_bytes;
        if (stats_.current_usage_bytes > stats_.peak_usage_bytes) {
            stats_.peak_usage_bytes = stats_.current_usage_bytes;
        }
        
        // Prefetch for write access
        PrefetchWrite(entry->aligned_ptr);
        
        return entry->aligned_ptr;
    }
    
    // Allocate new buffer
    void* raw_ptr = AlignedAllocStream(bucket_size, kTensorStreamAlignment);
    if (raw_ptr == nullptr) {
        return nullptr;
    }
    
    // Create new entry
    BufferEntry new_entry;
    new_entry.storage.reset(static_cast<char*>(raw_ptr));
    new_entry.aligned_ptr = static_cast<float*>(raw_ptr);
    new_entry.size_bytes = bucket_size;
    new_entry.bucket_size = bucket_size;
    new_entry.in_use = true;
    new_entry.handed_off = false;
    new_entry.current_owner = producer_hint;
    new_entry.acquired_at = std::chrono::steady_clock::now();
    
    // Update stats
    stats_.total_allocated_bytes += bucket_size;
    stats_.num_buffers++;
    stats_.current_usage_bytes += bucket_size;
    if (stats_.current_usage_bytes > stats_.peak_usage_bytes) {
        stats_.peak_usage_bytes = stats_.current_usage_bytes;
    }
    
    // Store in bucket map
    float* result = new_entry.aligned_ptr;
    buffers_[bucket_size].push_back(std::move(new_entry));
    
    // Prefetch for write access
    PrefetchWrite(result);
    
    return result;
}

// =============================================================================
// HANDOFF
// =============================================================================

void TensorStreamPool::Handoff(float* ptr, const char* consumer_hint) {
    if (ptr == nullptr) return;
    
    BufferEntry* entry = FindByPointer(ptr);
    if (entry == nullptr) {
#ifdef SAGUARO_DEBUG_STREAMING
        throw std::runtime_error("TensorStreamPool::Handoff() called with unknown pointer");
#endif
        return;
    }
    
#ifdef SAGUARO_DEBUG_STREAMING
    if (!entry->in_use) {
        throw std::runtime_error("TensorStreamPool::Handoff() called on buffer not in use");
    }
    if (entry->handed_off) {
        throw std::runtime_error("TensorStreamPool::Handoff() called on already handed-off buffer");
    }
#endif
    
    // Mark as handed off
    entry->handed_off = true;
    entry->current_owner = consumer_hint;
    
    // Update stats
    stats_.zero_copy_handoffs++;
    
    // Prefetch for consumer read access
    PrefetchRead(ptr);
    
    // Prefetch next cache lines (for sequential access patterns)
    // Prefetch 4 cache lines ahead (256 bytes on most systems)
    for (size_t offset = 64; offset < 256 && offset < entry->size_bytes; offset += 64) {
        PrefetchRead(reinterpret_cast<const char*>(ptr) + offset);
    }
}

// =============================================================================
// RELEASE
// =============================================================================

void TensorStreamPool::Release(float* ptr) {
    if (ptr == nullptr) return;
    
    BufferEntry* entry = FindByPointer(ptr);
    if (entry == nullptr) {
#ifdef SAGUARO_DEBUG_STREAMING
        throw std::runtime_error("TensorStreamPool::Release() called with unknown pointer");
#endif
        return;
    }
    
#ifdef SAGUARO_DEBUG_STREAMING
    if (!entry->in_use) {
        throw std::runtime_error("TensorStreamPool::Release() called on buffer not in use");
    }
#endif
    
    // Update lifetime stats
    UpdateLifetimeStats(*entry);
    
    // Mark as available for reuse
    entry->in_use = false;
    entry->handed_off = false;
    entry->current_owner = nullptr;
    
    // Update stats
    stats_.release_count++;
    stats_.current_usage_bytes -= entry->size_bytes;
}

// =============================================================================
// CLEAR
// =============================================================================

void TensorStreamPool::Clear() {
    // Clear all buffers (unique_ptr destructor will free memory)
    buffers_.clear();
    
    // Reset allocation stats but preserve cumulative operation counts
    size_t acquire_count = stats_.acquire_count;
    size_t reuse_count = stats_.reuse_count;
    size_t handoff_count = stats_.zero_copy_handoffs;
    size_t release_count = stats_.release_count;
    
    stats_ = Stats{};
    stats_.acquire_count = acquire_count;
    stats_.reuse_count = reuse_count;
    stats_.zero_copy_handoffs = handoff_count;
    stats_.release_count = release_count;
    // num_buffers, total_allocated_bytes, current_usage_bytes all reset to 0
    
    lifetime_samples_ = 0;
    lifetime_sum_ms_ = 0;
}

// =============================================================================
// RESET STATS
// =============================================================================

void TensorStreamPool::ResetStats() {
    // Keep allocation info, reset counts
    stats_.acquire_count = 0;
    stats_.reuse_count = 0;
    stats_.zero_copy_handoffs = 0;
    stats_.fallback_copies = 0;
    stats_.release_count = 0;
    stats_.peak_usage_bytes = stats_.current_usage_bytes;
    stats_.avg_buffer_lifetime_ms = 0;
    
    lifetime_samples_ = 0;
    lifetime_sum_ms_ = 0;
}

// =============================================================================
// LIFETIME STATS UPDATE
// =============================================================================

void TensorStreamPool::UpdateLifetimeStats(const BufferEntry& entry) {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - entry.acquired_at
    ).count();
    
    double lifetime_ms = duration / 1000.0;
    
    lifetime_samples_++;
    lifetime_sum_ms_ += lifetime_ms;
    stats_.avg_buffer_lifetime_ms = lifetime_sum_ms_ / lifetime_samples_;
}

// =============================================================================
// DEBUG VALIDATION
// =============================================================================

#ifdef SAGUARO_DEBUG_STREAMING

void TensorStreamPool::ValidateAccess(float* ptr) {
    if (ptr == nullptr) {
        throw std::runtime_error("TensorStreamPool: null pointer access");
    }
    
    BufferEntry* entry = FindByPointer(ptr);
    if (entry == nullptr) {
        throw std::runtime_error("TensorStreamPool: access to unknown buffer");
    }
    
    if (!entry->in_use) {
        throw std::runtime_error("TensorStreamPool: access to released buffer");
    }
}

bool TensorStreamPool::WasHandedOff(float* ptr) const {
    // Non-modifying version - can't use FindByPointer
    if (ptr == nullptr) return false;
    
    for (const auto& [bucket, entries] : buffers_) {
        for (const auto& entry : entries) {
            if (entry.aligned_ptr == ptr) {
                return entry.handed_off;
            }
        }
    }
    
    return false;
}

#endif  // SAGUARO_DEBUG_STREAMING

// =============================================================================
// GLOBAL SINGLETON INSTANCE
// =============================================================================

// Use visibility attribute to ensure symbol is exported from shared library
// This is critical because -fvisibility=hidden may be used during compilation
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
TensorStreamPool& GetTensorStreamPool() {
    static TensorStreamPool instance;
    return instance;
}

}  // namespace ops
}  // namespace saguaro
