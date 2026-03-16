// saguaro/native/ops/common/scratch_arena.h
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
 * @file scratch_arena.h
 * @brief Phase 96: Reusable aligned memory pool for intermediate buffers.
 *
 * Provides thread-local scratch memory arenas to avoid repeated allocations
 * in hot loop operations. Inspired by GGML's memory management.
 *
 * Benefits:
 *   - Eliminates malloc overhead in inner loops
 *   - Guarantees cache-line alignment (64 bytes)
 *   - Tracks memory usage statistics
 *   - Thread-safe via thread-local storage
 */

#ifndef OPS_COMMON_SCRATCH_ARENA_H_
#define OPS_COMMON_SCRATCH_ARENA_H_

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <vector>
#include <mutex>

#include "perf_utils.h"

namespace hsmn {
namespace ops {

// =============================================================================
// Aligned Memory Allocation
// =============================================================================

/**
 * @brief Allocate aligned memory.
 *
 * @param size Size in bytes.
 * @param alignment Alignment requirement (must be power of 2).
 * @return Aligned pointer, or nullptr on failure.
 */
inline void* AlignedAlloc(std::size_t size, std::size_t alignment = 64) {
    if (size == 0) return nullptr;
    
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#elif defined(__APPLE__)
    // macOS doesn't support aligned_alloc for alignments < 16
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#else
    // Standard C11 aligned_alloc (GCC, Clang on Linux)
    // Size must be multiple of alignment
    std::size_t aligned_size = RoundUp(size, alignment);
    return std::aligned_alloc(alignment, aligned_size);
#endif
}

/**
 * @brief Free aligned memory.
 *
 * @param ptr Pointer from AlignedAlloc.
 */
inline void AlignedFree(void* ptr) {
    if (ptr == nullptr) return;
    
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// =============================================================================
// Scratch Arena
// =============================================================================

/**
 * @brief Memory block in the arena.
 */
struct ArenaBlock {
    void* ptr;              // Aligned pointer
    std::size_t size;       // Allocated size
    std::size_t used;       // Currently used bytes
    bool in_use;            // Whether currently checked out
    
    ArenaBlock(std::size_t sz)
        : ptr(AlignedAlloc(sz, kCacheLineSize))
        , size(sz)
        , used(0)
        , in_use(false) {}
    
    ~ArenaBlock() {
        AlignedFree(ptr);
    }
    
    // Non-copyable
    ArenaBlock(const ArenaBlock&) = delete;
    ArenaBlock& operator=(const ArenaBlock&) = delete;
    
    // Moveable
    ArenaBlock(ArenaBlock&& other) noexcept
        : ptr(other.ptr)
        , size(other.size)
        , used(other.used)
        , in_use(other.in_use) {
        other.ptr = nullptr;
        other.size = 0;
    }
};

/**
 * @brief Thread-local scratch memory arena.
 *
 * Provides fast reusable memory for intermediate computations.
 * Memory is cache-line aligned and reused across operations.
 *
 * Example:
 * @code
 *     ScratchArena& arena = GetThreadArena();
 *     float* buffer = arena.Get<float>(1024);
 *     // ... use buffer ...
 *     arena.Release(buffer);
 * @endcode
 */
class ScratchArena {
public:
    /**
     * @brief Default constructor.
     */
    ScratchArena()
        : total_allocated_(0)
        , peak_usage_(0)
        , current_usage_(0)
        , allocation_count_(0)
        , reuse_count_(0) {}
    
    /**
     * @brief Destructor - frees all blocks.
     */
    ~ScratchArena() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& block : blocks_) {
            // Destructor of ArenaBlock handles freeing
        }
        blocks_.clear();
    }
    
    // Non-copyable
    ScratchArena(const ScratchArena&) = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;
    
    /**
     * @brief Get aligned buffer of at least `bytes` size.
     *
     * Reuses existing blocks if available, or allocates new.
     *
     * @param bytes Required size in bytes.
     * @param alignment Alignment (default: cache line).
     * @return Aligned pointer.
     */
    void* Get(std::size_t bytes, std::size_t alignment = kCacheLineSize) {
        if (bytes == 0) return nullptr;
        
        std::size_t aligned_bytes = RoundUp(bytes, alignment);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find a suitable free block
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= aligned_bytes) {
                block.in_use = true;
                block.used = aligned_bytes;
                current_usage_ += aligned_bytes;
                peak_usage_ = std::max(peak_usage_, current_usage_);
                ++reuse_count_;
                return block.ptr;
            }
        }
        
        // Allocate new block (round up to reasonable size)
        std::size_t alloc_size = std::max(aligned_bytes, static_cast<std::size_t>(4096));
        alloc_size = RoundUp(alloc_size, kCacheLineSize);
        
        blocks_.emplace_back(alloc_size);
        ArenaBlock& new_block = blocks_.back();
        
        if (new_block.ptr == nullptr) {
            blocks_.pop_back();
            return nullptr;
        }
        
        new_block.in_use = true;
        new_block.used = aligned_bytes;
        total_allocated_ += alloc_size;
        current_usage_ += aligned_bytes;
        peak_usage_ = std::max(peak_usage_, current_usage_);
        ++allocation_count_;
        
        return new_block.ptr;
    }
    
    /**
     * @brief Get typed buffer.
     *
     * @tparam T Element type.
     * @param count Number of elements.
     * @return Typed pointer.
     */
    template <typename T>
    T* Get(std::size_t count) {
        return static_cast<T*>(Get(count * sizeof(T), alignof(T) > kCacheLineSize ? alignof(T) : kCacheLineSize));
    }
    
    /**
     * @brief Release buffer back to arena.
     *
     * Does not actually free - just marks as reusable.
     *
     * @param ptr Pointer from Get().
     */
    void Release(void* ptr) {
        if (ptr == nullptr) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            if (block.ptr == ptr && block.in_use) {
                block.in_use = false;
                current_usage_ -= block.used;
                block.used = 0;
                return;
            }
        }
    }
    
    /**
     * @brief Reset arena, marking all blocks as free.
     *
     * Does not deallocate memory.
     */
    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& block : blocks_) {
            block.in_use = false;
            block.used = 0;
        }
        current_usage_ = 0;
    }
    
    /**
     * @brief Clear arena, freeing all memory.
     */
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        blocks_.clear();
        total_allocated_ = 0;
        current_usage_ = 0;
    }
    
    // =========================================================================
    // Statistics
    // =========================================================================
    
    std::size_t total_allocated() const { return total_allocated_; }
    std::size_t peak_usage() const { return peak_usage_; }
    std::size_t current_usage() const { return current_usage_; }
    std::size_t allocation_count() const { return allocation_count_; }
    std::size_t reuse_count() const { return reuse_count_; }
    std::size_t num_blocks() const { return blocks_.size(); }
    
    /**
     * @brief Calculate reuse ratio.
     *
     * @return Ratio of reuses to total allocations (higher is better).
     */
    float reuse_ratio() const {
        std::size_t total = allocation_count_ + reuse_count_;
        if (total == 0) return 0.0f;
        return static_cast<float>(reuse_count_) / static_cast<float>(total);
    }
    
private:
    std::vector<ArenaBlock> blocks_;
    std::mutex mutex_;
    
    std::size_t total_allocated_;
    std::size_t peak_usage_;
    std::size_t current_usage_;
    std::size_t allocation_count_;
    std::size_t reuse_count_;
};

// =============================================================================
// Thread-Local Arena Access
// =============================================================================

/**
 * @brief Get thread-local scratch arena.
 *
 * Each thread gets its own arena to avoid contention.
 *
 * @return Reference to thread-local arena.
 */
inline ScratchArena& GetThreadArena() {
    thread_local ScratchArena arena;
    return arena;
}

// =============================================================================
// RAII Scratch Buffer Guard
// =============================================================================

/**
 * @brief RAII guard for scratch buffer allocation.
 *
 * Automatically releases buffer when going out of scope.
 *
 * Example:
 * @code
 *     {
 *         ScratchGuard<float> buffer(1024);
 *         // buffer.get() returns float* with 1024 elements
 *     } // Auto-released here
 * @endcode
 */
template <typename T>
class ScratchGuard {
public:
    explicit ScratchGuard(std::size_t count, ScratchArena& arena = GetThreadArena())
        : arena_(arena)
        , ptr_(arena.Get<T>(count))
        , count_(count) {}
    
    ~ScratchGuard() {
        if (ptr_) {
            arena_.Release(ptr_);
        }
    }
    
    // Non-copyable
    ScratchGuard(const ScratchGuard&) = delete;
    ScratchGuard& operator=(const ScratchGuard&) = delete;
    
    // Moveable
    ScratchGuard(ScratchGuard&& other) noexcept
        : arena_(other.arena_)
        , ptr_(other.ptr_)
        , count_(other.count_) {
        other.ptr_ = nullptr;
    }
    
    T* get() const { return ptr_; }
    T* data() const { return ptr_; }
    std::size_t size() const { return count_; }
    
    T& operator[](std::size_t idx) { return ptr_[idx]; }
    const T& operator[](std::size_t idx) const { return ptr_[idx]; }
    
    explicit operator bool() const { return ptr_ != nullptr; }
    
private:
    ScratchArena& arena_;
    T* ptr_;
    std::size_t count_;
};

}  // namespace ops
}  // namespace hsmn

#endif  // OPS_COMMON_SCRATCH_ARENA_H_
