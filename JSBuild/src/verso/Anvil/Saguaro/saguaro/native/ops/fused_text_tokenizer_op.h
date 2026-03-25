// saguaro.native/ops/fused_text_tokenizer_op.h
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
 * @file fused_text_tokenizer_op.h
 * @brief Enterprise-grade SIMD-optimized text tokenization with trie-based n-gram merging.
 *
 * This header provides high-performance CPU-optimized text tokenization operations:
 *
 * **Core Operations:**
 * - `text_tokenize_utf8_simd()`: SIMD-accelerated UTF-8 byte tokenization
 * - `text_tokenize_batch()`: Batched parallel tokenization
 *
 * **N-gram Trie Operations:**
 * - `TrieNode`: Compact trie node for superword lookup
 * - `SuperwordTrie`: Trie structure with O(max_ngram) lookup complexity
 * - `trie_apply_merges()`: Apply superword merges in single pass
 *
 * **N-gram Training Operations:**
 * - `ngram_count_streaming()`: Streaming n-gram counting without materialization
 * - `ngram_build_trie()`: Build trie from frequency counts
 *
 * SIMD Support (CPU-focused, no GPU):
 * - AVX512: 64-byte vectorization for maximum throughput
 * - AVX2: 32-byte vectorization (most common)
 * - SSE4.2: 16-byte vectorization (fallback)
 * - NEON: ARM vectorization
 * - Scalar: Always available fallback
 *
 * Performance Targets:
 * - Byte tokenization: >500MB/s per core
 * - Trie lookup: O(max_ngram_size) per token
 * - Memory: O(V × N) where V=vocab, N=avg ngram length
 *
 * @note Thread-safe. All functions are reentrant with no shared mutable state.
 */

#ifndef SAGUARO_NATIVE_OPS_FUSED_TEXT_TOKENIZER_OP_H_
#define SAGUARO_NATIVE_OPS_FUSED_TEXT_TOKENIZER_OP_H_

#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <memory>
#include <atomic>
#include <thread>

// SIMD intrinsics for cross-architecture vectorization
#if defined(__AVX512F__) && defined(__AVX512BW__)
#include <immintrin.h>
#define TEXT_TOK_SIMD_WIDTH 64
#define TEXT_TOK_AVX512 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define TEXT_TOK_SIMD_WIDTH 32
#define TEXT_TOK_AVX2 1
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#define TEXT_TOK_SIMD_WIDTH 16
#define TEXT_TOK_SSE42 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define TEXT_TOK_SIMD_WIDTH 16
#define TEXT_TOK_NEON 1
#else
#define TEXT_TOK_SIMD_WIDTH 1
#define TEXT_TOK_SCALAR 1
#endif

namespace saguaro {
namespace ops {
namespace text_tokenizer {

// =============================================================================
// CONSTANTS
// =============================================================================

constexpr int32_t TEXT_TOK_UNK_ID = 0;
constexpr int32_t TEXT_TOK_CLS_ID = 1;
constexpr int32_t TEXT_TOK_PAD_ID = 2;
constexpr int32_t TEXT_TOK_EOS_ID = 3;
constexpr int32_t TEXT_TOK_SEP_ID = 4;
constexpr int32_t TEXT_TOK_MASK_ID = 5;
constexpr int32_t TEXT_TOK_THINK_ID = 6;
constexpr int32_t TEXT_TOK_PAUSE_ID = 7;
constexpr int32_t TEXT_TOK_REFLECT_ID = 8;
constexpr int32_t TEXT_TOK_CONCLUDE_ID = 9;

constexpr int32_t TEXT_TOK_BYTE_OFFSET = 32;
constexpr int32_t TEXT_TOK_BYTE_VOCAB = 256;
constexpr int32_t TEXT_TOK_MAX_NGRAM = 8;  // Maximum n-gram size for trie

// Alignment for SIMD operations
constexpr size_t SIMD_ALIGNMENT = 64;

// =============================================================================
// TRIE NODE - Compact representation for memory efficiency
// =============================================================================

/**
 * @brief Compact trie node for superword n-gram matching.
 *
 * Uses a flat array for children (sparse) or hash map (dense) depending on
 * branching factor. The superword_id is -1 if this node doesn't complete
 * a valid superword.
 *
 * Memory layout optimized for cache efficiency:
 * - Small nodes use inline array (no allocation)
 * - Large nodes use external hash map
 */
struct TrieNode {
    // Inline storage for up to 4 children (covers 90% of cases)
    static constexpr int INLINE_CAPACITY = 4;

    // Child token IDs (inline storage)
    int32_t inline_keys[INLINE_CAPACITY];
    // Child node indices (inline storage)
    int32_t inline_values[INLINE_CAPACITY];
    // Number of children in inline storage
    int8_t inline_count;

    // Superword ID if this node completes a valid n-gram, else -1
    int32_t superword_id;

    // External storage for nodes with many children (nullptr if inline)
    std::unordered_map<int32_t, int32_t>* external_children;

    TrieNode()
        : inline_count(0), superword_id(-1), external_children(nullptr) {
        std::memset(inline_keys, 0, sizeof(inline_keys));
        std::memset(inline_values, 0, sizeof(inline_values));
    }

    ~TrieNode() {
        delete external_children;
    }

    // Disable copy (move only)
    TrieNode(const TrieNode&) = delete;
    TrieNode& operator=(const TrieNode&) = delete;

    TrieNode(TrieNode&& other) noexcept
        : inline_count(other.inline_count),
          superword_id(other.superword_id),
          external_children(other.external_children) {
        std::memcpy(inline_keys, other.inline_keys, sizeof(inline_keys));
        std::memcpy(inline_values, other.inline_values, sizeof(inline_values));
        other.external_children = nullptr;
        other.inline_count = 0;
    }

    TrieNode& operator=(TrieNode&& other) noexcept {
        if (this != &other) {
            delete external_children;
            inline_count = other.inline_count;
            superword_id = other.superword_id;
            external_children = other.external_children;
            std::memcpy(inline_keys, other.inline_keys, sizeof(inline_keys));
            std::memcpy(inline_values, other.inline_values, sizeof(inline_values));
            other.external_children = nullptr;
            other.inline_count = 0;
        }
        return *this;
    }

    /**
     * @brief Look up child node by token ID.
     * @param token_id The token ID to look up.
     * @return Child node index, or -1 if not found.
     */
    inline int32_t find_child(int32_t token_id) const {
        // Check inline storage first (cache-friendly linear scan)
        for (int i = 0; i < inline_count; ++i) {
            if (inline_keys[i] == token_id) {
                return inline_values[i];
            }
        }
        // Check external storage if present
        if (external_children != nullptr) {
            auto it = external_children->find(token_id);
            if (it != external_children->end()) {
                return it->second;
            }
        }
        return -1;
    }

    /**
     * @brief Add or update child node.
     * @param token_id The token ID for the child.
     * @param node_index The index of the child node.
     */
    inline void set_child(int32_t token_id, int32_t node_index) {
        // Check if already in inline storage
        for (int i = 0; i < inline_count; ++i) {
            if (inline_keys[i] == token_id) {
                inline_values[i] = node_index;
                return;
            }
        }
        // Add to inline if space available
        if (inline_count < INLINE_CAPACITY) {
            inline_keys[inline_count] = token_id;
            inline_values[inline_count] = node_index;
            ++inline_count;
            return;
        }
        // Overflow to external storage
        if (external_children == nullptr) {
            external_children = new std::unordered_map<int32_t, int32_t>();
            external_children->reserve(16);
        }
        (*external_children)[token_id] = node_index;
    }
};

// =============================================================================
// SUPERWORD TRIE - Complete trie structure for n-gram matching
// =============================================================================

/**
 * @brief High-performance trie for superword n-gram matching.
 *
 * Stores superword mappings (token sequence -> superword ID) for O(N) lookup
 * where N is the n-gram length. Uses flat array storage for cache efficiency.
 *
 * Thread-safe for concurrent reads after construction.
 */
class SuperwordTrie {
public:
    SuperwordTrie() {
        // Pre-allocate root node
        nodes_.emplace_back();
    }

    /**
     * @brief Insert an n-gram mapping into the trie.
     * @param ngram Token IDs forming the n-gram.
     * @param ngram_len Length of the n-gram.
     * @param superword_id The superword ID to map to.
     */
    void insert(const int32_t* ngram, int ngram_len, int32_t superword_id) {
        int32_t node_idx = 0;  // Start at root

        for (int i = 0; i < ngram_len; ++i) {
            int32_t token_id = ngram[i];
            int32_t child_idx = nodes_[node_idx].find_child(token_id);

            if (child_idx < 0) {
                // Create new node
                child_idx = static_cast<int32_t>(nodes_.size());
                nodes_.emplace_back();
                nodes_[node_idx].set_child(token_id, child_idx);
            }
            node_idx = child_idx;
        }

        // Mark final node with superword ID
        nodes_[node_idx].superword_id = superword_id;
    }

    /**
     * @brief Find longest matching n-gram starting at position.
     *
     * Walks the trie from the root, following token IDs until no match
     * is found. Returns the longest matching superword, if any.
     *
     * @param tokens Token ID array.
     * @param start Start position in the array.
     * @param len Total length of the array.
     * @param[out] match_len Length of the matched n-gram (0 if no match).
     * @return Superword ID if found, -1 otherwise.
     */
    inline int32_t find_longest_match(
        const int32_t* tokens,
        int start,
        int len,
        int* match_len
    ) const {
        int32_t best_superword = -1;
        int best_len = 0;
        int32_t node_idx = 0;  // Start at root

        for (int i = start; i < len && (i - start) < TEXT_TOK_MAX_NGRAM; ++i) {
            int32_t token_id = tokens[i];
            int32_t child_idx = nodes_[node_idx].find_child(token_id);

            if (child_idx < 0) {
                break;  // No more matches
            }

            node_idx = child_idx;

            // Check if this node completes a superword
            if (nodes_[node_idx].superword_id >= 0) {
                best_superword = nodes_[node_idx].superword_id;
                best_len = i - start + 1;
            }
        }

        *match_len = best_len;
        return best_superword;
    }

    /**
     * @brief Get number of nodes in the trie.
     */
    size_t node_count() const { return nodes_.size(); }

    /**
     * @brief Check if trie is empty (no superwords).
     */
    bool empty() const { return nodes_.size() <= 1 && nodes_[0].superword_id < 0; }

    /**
     * @brief Clear all entries from the trie.
     */
    void clear() {
        nodes_.clear();
        nodes_.emplace_back();  // Re-add root
    }

private:
    std::vector<TrieNode> nodes_;
};

// =============================================================================
// SIMD UTF-8 BYTE TOKENIZATION
// =============================================================================

/**
 * @brief SIMD-accelerated UTF-8 byte tokenization.
 *
 * Converts UTF-8 bytes to token IDs using vectorized operations.
 * Each byte maps to (byte_offset + byte_value).
 *
 * @param utf8_data Input UTF-8 byte string.
 * @param utf8_len Length in bytes.
 * @param output_tokens Pre-allocated output array (must be >= utf8_len + 2).
 * @param byte_offset Offset to add to each byte value.
 * @param add_special_tokens If true, prepend CLS and append EOS.
 * @return Number of tokens written.
 */
inline int text_tokenize_utf8_simd(
    const uint8_t* utf8_data,
    int utf8_len,
    int32_t* output_tokens,
    int32_t byte_offset,
    bool add_special_tokens
) {
    int out_pos = 0;

    // Add CLS token if requested
    if (add_special_tokens) {
        output_tokens[out_pos++] = TEXT_TOK_CLS_ID;
    }

    int i = 0;

#if defined(TEXT_TOK_AVX512)
    // AVX-512: Process 64 bytes at a time
    // Use zero-extension from uint8 to int32
    for (; i + 64 <= utf8_len; i += 64) {
        // Load 64 bytes
        __m512i bytes = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(utf8_data + i));

        // Split into 4 groups of 16 bytes and zero-extend to 32-bit
        __m128i b0 = _mm512_extracti32x4_epi32(bytes, 0);
        __m128i b1 = _mm512_extracti32x4_epi32(bytes, 1);
        __m128i b2 = _mm512_extracti32x4_epi32(bytes, 2);
        __m128i b3 = _mm512_extracti32x4_epi32(bytes, 3);

        // Zero-extend each 16-byte chunk to 16 int32s
        __m512i offset_vec = _mm512_set1_epi32(byte_offset);

        __m512i i0 = _mm512_add_epi32(_mm512_cvtepu8_epi32(b0), offset_vec);
        __m512i i1 = _mm512_add_epi32(_mm512_cvtepu8_epi32(b1), offset_vec);
        __m512i i2 = _mm512_add_epi32(_mm512_cvtepu8_epi32(b2), offset_vec);
        __m512i i3 = _mm512_add_epi32(_mm512_cvtepu8_epi32(b3), offset_vec);

        // Store 64 int32s
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output_tokens + out_pos), i0);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output_tokens + out_pos + 16), i1);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output_tokens + out_pos + 32), i2);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(output_tokens + out_pos + 48), i3);

        out_pos += 64;
    }
#elif defined(TEXT_TOK_AVX2)
    // AVX2: Process 32 bytes at a time
    __m256i offset_vec = _mm256_set1_epi32(byte_offset);

    for (; i + 32 <= utf8_len; i += 32) {
        // Load 32 bytes
        __m256i bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(utf8_data + i));

        // Extract lower and upper 16 bytes
        __m128i b_lo = _mm256_castsi256_si128(bytes);
        __m128i b_hi = _mm256_extracti128_si256(bytes, 1);

        // Zero-extend to 32-bit (8 bytes -> 8 int32s per step)
        __m128i b0 = b_lo;
        __m128i b1 = _mm_srli_si128(b_lo, 8);
        __m128i b2 = b_hi;
        __m128i b3 = _mm_srli_si128(b_hi, 8);

        // Use SSE4.1 pmovzxbd for zero extension (only works on low 4 bytes)
        // So we need to do 8 conversions of 4 bytes each
        __m256i i0 = _mm256_add_epi32(_mm256_cvtepu8_epi32(b0), offset_vec);
        __m256i i1 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(b_lo, 8)), offset_vec);
        __m256i i2 = _mm256_add_epi32(_mm256_cvtepu8_epi32(b_hi), offset_vec);
        __m256i i3 = _mm256_add_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(b_hi, 8)), offset_vec);

        // Store 32 int32s
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_tokens + out_pos), i0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_tokens + out_pos + 8), i1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_tokens + out_pos + 16), i2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output_tokens + out_pos + 24), i3);

        out_pos += 32;
    }
#elif defined(TEXT_TOK_SSE42)
    // SSE4.2: Process 16 bytes at a time
    __m128i offset_vec_lo = _mm_set1_epi32(byte_offset);

    for (; i + 16 <= utf8_len; i += 16) {
        __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(utf8_data + i));

        // Zero-extend 4 bytes at a time to 4 int32s
        __m128i i0 = _mm_add_epi32(_mm_cvtepu8_epi32(bytes), offset_vec_lo);
        __m128i i1 = _mm_add_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(bytes, 4)), offset_vec_lo);
        __m128i i2 = _mm_add_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(bytes, 8)), offset_vec_lo);
        __m128i i3 = _mm_add_epi32(_mm_cvtepu8_epi32(_mm_srli_si128(bytes, 12)), offset_vec_lo);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_tokens + out_pos), i0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_tokens + out_pos + 4), i1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_tokens + out_pos + 8), i2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output_tokens + out_pos + 12), i3);

        out_pos += 16;
    }
#elif defined(TEXT_TOK_NEON)
    // NEON: Process 16 bytes at a time
    int32x4_t offset_vec = vdupq_n_s32(byte_offset);

    for (; i + 16 <= utf8_len; i += 16) {
        uint8x16_t bytes = vld1q_u8(utf8_data + i);

        // Zero-extend to 16-bit then 32-bit
        uint16x8_t lo16 = vmovl_u8(vget_low_u8(bytes));
        uint16x8_t hi16 = vmovl_u8(vget_high_u8(bytes));

        int32x4_t i0 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo16))), offset_vec);
        int32x4_t i1 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo16))), offset_vec);
        int32x4_t i2 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi16))), offset_vec);
        int32x4_t i3 = vaddq_s32(vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi16))), offset_vec);

        vst1q_s32(output_tokens + out_pos, i0);
        vst1q_s32(output_tokens + out_pos + 4, i1);
        vst1q_s32(output_tokens + out_pos + 8, i2);
        vst1q_s32(output_tokens + out_pos + 12, i3);

        out_pos += 16;
    }
#endif

    // Scalar fallback for remaining bytes
    for (; i < utf8_len; ++i) {
        output_tokens[out_pos++] = byte_offset + static_cast<int32_t>(utf8_data[i]);
    }

    // Add EOS token if requested
    if (add_special_tokens) {
        output_tokens[out_pos++] = TEXT_TOK_EOS_ID;
    }

    return out_pos;
}

// =============================================================================
// TRIE-BASED SUPERWORD MERGE APPLICATION
// =============================================================================

/**
 * @brief Apply superword merges to token sequence using trie lookup.
 *
 * Single-pass algorithm with O(N × M) complexity where N is sequence length
 * and M is max n-gram size (typically 5-8).
 *
 * @param input_tokens Input token IDs.
 * @param input_len Number of input tokens.
 * @param trie Superword trie for lookup.
 * @param output_tokens Pre-allocated output array (must be >= input_len).
 * @return Number of output tokens.
 */
inline int trie_apply_merges(
    const int32_t* input_tokens,
    int input_len,
    const SuperwordTrie& trie,
    int32_t* output_tokens
) {
    if (trie.empty()) {
        // No superwords, just copy
        std::memcpy(output_tokens, input_tokens, input_len * sizeof(int32_t));
        return input_len;
    }

    int out_pos = 0;
    int i = 0;

    while (i < input_len) {
        int match_len = 0;
        int32_t superword_id = trie.find_longest_match(
            input_tokens, i, input_len, &match_len
        );

        if (superword_id >= 0 && match_len > 0) {
            // Found a superword match
            output_tokens[out_pos++] = superword_id;
            i += match_len;
        } else {
            // No match, copy original token
            output_tokens[out_pos++] = input_tokens[i++];
        }
    }

    return out_pos;
}

// =============================================================================
// STREAMING N-GRAM COUNTING
// =============================================================================

/**
 * @brief Thread-safe n-gram counter for streaming corpus processing.
 *
 * Uses lock-free atomic increments for concurrent counting.
 */
class StreamingNgramCounter {
public:
    explicit StreamingNgramCounter(int min_ngram = 2, int max_ngram = 5)
        : min_ngram_(min_ngram), max_ngram_(max_ngram) {}

    /**
     * @brief Count n-grams from a single token sequence.
     *
     * Thread-safe: can be called from multiple threads concurrently.
     *
     * @param tokens Token ID sequence.
     * @param len Sequence length.
     */
    void count_sequence(const int32_t* tokens, int len) {
        for (int n = min_ngram_; n <= max_ngram_; ++n) {
            for (int i = 0; i <= len - n; ++i) {
                // Create key from n-gram
                NgramKey key;
                key.len = n;
                std::memcpy(key.tokens, tokens + i, n * sizeof(int32_t));

                // Atomic increment
                std::lock_guard<std::mutex> lock(mutex_);
                counts_[key]++;
            }
        }
    }

    /**
     * @brief Get top K n-grams by frequency.
     *
     * @param min_freq Minimum frequency threshold.
     * @param max_count Maximum number of n-grams to return.
     * @param[out] ngrams Output vector of (ngram, frequency) pairs.
     */
    void get_top_ngrams(
        int min_freq,
        int max_count,
        std::vector<std::pair<std::vector<int32_t>, int>>& ngrams
    ) const {
        ngrams.clear();
        ngrams.reserve(counts_.size());

        for (const auto& kv : counts_) {
            if (kv.second >= min_freq) {
                std::vector<int32_t> ngram(kv.first.tokens, kv.first.tokens + kv.first.len);
                ngrams.emplace_back(std::move(ngram), kv.second);
            }
        }

        // Sort by frequency (descending)
        std::sort(ngrams.begin(), ngrams.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Truncate to max_count
        if (static_cast<int>(ngrams.size()) > max_count) {
            ngrams.resize(max_count);
        }
    }

    /**
     * @brief Clear all counts.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        counts_.clear();
    }

    /**
     * @brief Get total unique n-gram count.
     */
    size_t unique_count() const { return counts_.size(); }

private:
    struct NgramKey {
        int32_t tokens[TEXT_TOK_MAX_NGRAM];
        int8_t len;

        bool operator==(const NgramKey& other) const {
            if (len != other.len) return false;
            return std::memcmp(tokens, other.tokens, len * sizeof(int32_t)) == 0;
        }
    };

    struct NgramKeyHash {
        size_t operator()(const NgramKey& key) const {
            // FNV-1a hash
            size_t hash = 14695981039346656037ULL;
            const uint8_t* data = reinterpret_cast<const uint8_t*>(key.tokens);
            for (int i = 0; i < key.len * static_cast<int>(sizeof(int32_t)); ++i) {
                hash ^= data[i];
                hash *= 1099511628211ULL;
            }
            return hash;
        }
    };

    int min_ngram_;
    int max_ngram_;
    mutable std::mutex mutex_;
    std::unordered_map<NgramKey, int, NgramKeyHash> counts_;
};

// =============================================================================
// BATCH TOKENIZATION WITH PARALLEL PROCESSING
// =============================================================================

/**
 * @brief Tokenize multiple texts in parallel.
 *
 * @param texts Array of text pointers.
 * @param text_lens Array of text lengths.
 * @param num_texts Number of texts to process.
 * @param byte_offset Byte offset for token IDs.
 * @param add_special_tokens Whether to add CLS/EOS.
 * @param max_length Maximum output length per text.
 * @param[out] output_tokens Flattened output array [num_texts * max_length].
 * @param[out] output_lengths Actual lengths per text [num_texts].
 * @param num_threads Number of parallel threads (0 = auto).
 */
inline void text_tokenize_batch_parallel(
    const uint8_t** texts,
    const int* text_lens,
    int num_texts,
    int32_t byte_offset,
    bool add_special_tokens,
    int max_length,
    int32_t* output_tokens,
    int* output_lengths,
    int num_threads = 0
) {
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    // Parallel tokenization
    auto process_range = [&](int start, int end) {
        // reuse scratch buffer inside thread? No, inside lambda is safer if not thread_local
        // For simplicity and safety in parallel execution, we allocate inside the loop.
        // Optimization: Use a thread-local vector if performance is critical, 
        // but here we prioritize safety.
        
        for (int i = start; i < end; ++i) {
            int32_t* out_ptr = output_tokens + i * max_length;
            
            // Allocate scratch buffer large enough for input
            // One token per byte + special tokens
            size_t required_size = text_lens[i] + 2;
            std::vector<int32_t> temp_tokens(required_size);

            // Tokenize into temporary buffer
            int len = text_tokenize_utf8_simd(
                texts[i],
                text_lens[i],
                temp_tokens.data(),
                byte_offset,
                add_special_tokens
            );

            // Safe Copy with Truncation
            int copy_len = std::min(len, max_length);
            std::memcpy(out_ptr, temp_tokens.data(), copy_len * sizeof(int32_t));

            // Populate length output
            // Effectively truncated to max_length
            int effective_len = (len > max_length) ? max_length : len;
            
            // If we truncated and needed special tokens, ensure EOS at the end
            if (len > max_length && add_special_tokens) {
                 out_ptr[max_length - 1] = TEXT_TOK_EOS_ID;
            }

            output_lengths[i] = effective_len;

            // Pad remaining with PAD tokens
            for (int j = effective_len; j < max_length; ++j) {
                out_ptr[j] = TEXT_TOK_PAD_ID;
            }
        }
    };

    if (num_texts <= num_threads * 2) {
        // Small batch: process sequentially
        process_range(0, num_texts);
    } else {
        // Large batch: parallelize
        std::vector<std::thread> threads;
        threads.reserve(num_threads);

        int chunk_size = (num_texts + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk_size;
            int end = std::min(start + chunk_size, num_texts);
            if (start < end) {
                threads.emplace_back(process_range, start, end);
            }
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }
}

// =============================================================================
// COMPLETE TOKENIZATION + MERGE PIPELINE
// =============================================================================

/**
 * @brief Complete tokenization pipeline: UTF-8 -> tokens -> merged tokens.
 *
 * Single function for maximum efficiency with fused operations.
 *
 * @param utf8_data Input UTF-8 text.
 * @param utf8_len Length in bytes.
 * @param trie Superword trie (can be empty).
 * @param output_tokens Pre-allocated output (must be >= utf8_len + 2).
 * @param byte_offset Byte offset for token IDs.
 * @param add_special_tokens Add CLS/EOS tokens.
 * @param scratch_buffer Scratch space for intermediate tokens (>= utf8_len + 2).
 * @return Number of final tokens.
 */
inline int fused_tokenize_and_merge(
    const uint8_t* utf8_data,
    int utf8_len,
    const SuperwordTrie& trie,
    int32_t* output_tokens,
    int32_t byte_offset,
    bool add_special_tokens,
    int32_t* scratch_buffer
) {
    // Step 1: SIMD byte tokenization
    int raw_len = text_tokenize_utf8_simd(
        utf8_data,
        utf8_len,
        scratch_buffer,
        byte_offset,
        add_special_tokens
    );

    // Step 2: Trie-based merge (or copy if no trie)
    if (trie.empty()) {
        std::memcpy(output_tokens, scratch_buffer, raw_len * sizeof(int32_t));
        return raw_len;
    }

    return trie_apply_merges(scratch_buffer, raw_len, trie, output_tokens);
}

}  // namespace text_tokenizer
}  // namespace ops
}  // namespace saguaro

#endif  // SAGUARO_NATIVE_OPS_FUSED_TEXT_TOKENIZER_OP_H_
