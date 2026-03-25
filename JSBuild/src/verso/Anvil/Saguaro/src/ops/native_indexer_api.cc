// saguaro/native/ops/native_indexer_api.cc
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Implementation of native C API for indexer operations.
// This file compiles into _saguaro_core.so alongside TensorFlow ops.

#include "native_indexer_api.h"
// #include <cstdio> // Removed debug include
#include "fused_text_tokenizer_op.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace saguaro::ops::text_tokenizer;

// =============================================================================
// VERSION INFO
// =============================================================================

extern "C" {

namespace {

inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

inline float splitmix64_unit_float(uint64_t& state) {
    constexpr double kInv24Bit = 1.0 / static_cast<double>(1u << 24);
    const uint64_t bits = splitmix64_next(state) >> 40;
    return static_cast<float>(static_cast<double>(bits) * kInv24Bit);
}

inline float sorted_unique_jaccard(
    const int32_t* left,
    int left_len,
    const int32_t* right,
    int right_len
) {
    if (!left || !right || left_len <= 0 || right_len <= 0) return 0.0f;
    int li = 0;
    int ri = 0;
    int intersection = 0;
    while (li < left_len && ri < right_len) {
        const int32_t lhs = left[li];
        const int32_t rhs = right[ri];
        if (lhs == rhs) {
            ++intersection;
            ++li;
            ++ri;
        } else if (lhs < rhs) {
            ++li;
        } else {
            ++ri;
        }
    }
    const int union_count = left_len + right_len - intersection;
    if (union_count <= 0) return 0.0f;
    return static_cast<float>(intersection) / static_cast<float>(union_count);
}

}  // namespace

const char* saguaro_native_version(void) {
    return "2.0.0-native";
}

int saguaro_native_available(void) {
    return 1;
}

const char* saguaro_native_build_signature(void) {
    static const std::string signature =
        std::string("compiler=") +
#if defined(__clang__)
        "clang"
#elif defined(__GNUC__)
        "gcc"
#else
        "unknown"
#endif
        + ";openmp=" +
#if defined(_OPENMP)
        "1"
#else
        "0"
#endif
        + ";avx2=" +
#if defined(__AVX2__)
        "1"
#else
        "0"
#endif
        + ";fma=" +
#if defined(__FMA__)
        "1";
#else
        "0";
#endif
    return signature.c_str();
}

const char* saguaro_native_isa_baseline(void) {
#if defined(__AVX2__)
    return "avx2";
#else
    return "scalar";
#endif
}

int saguaro_native_openmp_enabled(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

int saguaro_native_avx2_enabled(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int saguaro_native_fma_enabled(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int saguaro_native_max_threads(void) {
#if defined(_OPENMP)
    return std::max(1, omp_get_max_threads());
#else
    const auto hardware_threads = std::thread::hardware_concurrency();
    return hardware_threads == 0 ? 1 : static_cast<int>(hardware_threads);
#endif
}

void saguaro_native_init_projection(
    float* projection,
    int vocab_size,
    int dim,
    uint64_t seed
) {
    if (!projection || vocab_size <= 0 || dim <= 0) return;

    const float init_range = 1.0f / std::sqrt(static_cast<float>(std::max(dim, 1)));
    const uint64_t base_seed = seed == 0 ? 42ULL : seed;

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int token = 0; token < vocab_size; ++token) {
        uint64_t state = base_seed ^ (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(token + 1));
        float* row = projection + (static_cast<size_t>(token) * dim);
        for (int d = 0; d < dim; ++d) {
            const float unit = splitmix64_unit_float(state);
            row[d] = ((unit * 2.0f) - 1.0f) * init_range;
        }
    }
}

// =============================================================================
// TRIE MANAGEMENT
// =============================================================================

saguaro_trie_handle_t saguaro_native_trie_create(void) {
    return static_cast<saguaro_trie_handle_t>(new SuperwordTrie());
}

void saguaro_native_trie_destroy(saguaro_trie_handle_t trie) {
    if (trie) {
        delete static_cast<SuperwordTrie*>(trie);
    }
}

void saguaro_native_trie_insert(
    saguaro_trie_handle_t trie,
    const int32_t* ngram,
    int ngram_len,
    int32_t superword_id
) {
    if (!trie || !ngram || ngram_len <= 0) return;
    static_cast<SuperwordTrie*>(trie)->insert(ngram, ngram_len, superword_id);
}

void saguaro_native_trie_build_from_table(
    saguaro_trie_handle_t trie,
    const int32_t* offsets,
    const int32_t* tokens,
    const int32_t* superword_ids,
    int num_ngrams
) {
    if (!trie || !offsets || !tokens || !superword_ids || num_ngrams <= 0) return;
    
    SuperwordTrie* t = static_cast<SuperwordTrie*>(trie);
    t->clear();
    
    for (int i = 0; i < num_ngrams; ++i) {
        int start = offsets[i];
        int end = offsets[i + 1];
        int len = end - start;
        if (len > 0) {
            t->insert(tokens + start, len, superword_ids[i]);
        }
    }
}

// =============================================================================
// TOKENIZATION
// =============================================================================

int saguaro_native_tokenize_batch(
    const char* const* texts,
    const int* text_lengths,
    int batch_size,
    int32_t* output_tokens,
    int32_t* output_lengths,
    int max_length,
    int byte_offset,
    int add_special_tokens,
    saguaro_trie_handle_t trie,
    int num_threads
) {
    if (!texts || !text_lengths || !output_tokens || !output_lengths) return -1;
    
    const SuperwordTrie* trie_ptr = static_cast<const SuperwordTrie*>(trie);

    
    // Auto-detect and clamp thread count to native capacity.
    const int max_threads = saguaro_native_max_threads();
    if (num_threads <= 0) {
        num_threads = max_threads;
    }
    num_threads = std::max(1, std::min(num_threads, max_threads));
    
    auto process_range = [&](int start, int end) {
        // Reusable scratch buffer per thread to avoid allocations
        std::vector<int32_t> scratch;

        for (int i = start; i < end; ++i) {
            const uint8_t* text_data = reinterpret_cast<const uint8_t*>(texts[i]);
            int text_len = text_lengths[i];
            int32_t* out_ptr = output_tokens + (i * max_length);
            
            // Ensure scratch buffer is large enough
            size_t needed = static_cast<size_t>(text_len) + 2;
            if (scratch.size() < needed) {
                scratch.resize(needed);
            }
            
            int final_len;
            if (trie_ptr && !trie_ptr->empty()) {
                // With trie: tokenize to scratch, merge to output (already respects max_length)
                final_len = fused_tokenize_and_merge(
                    text_data,
                    text_len,
                    *trie_ptr,
                    out_ptr,
                    max_length,
                    byte_offset,
                    add_special_tokens != 0,
                    scratch.data(),
                    false  // inject_thinking
                );
            } else {
                // Without trie: tokenize to SCRATCH first to prevent buffer overflow
                // text_tokenize_utf8_complex doesn't know about max_length
                int raw_len = text_tokenize_utf8_complex(
                    text_data,
                    text_len,
                    scratch.data(),  // Write to scratch, not directly to output!
                    byte_offset,
                    add_special_tokens != 0,
                    false  // inject_thinking
                );
                // Copy to output with truncation
                final_len = std::min(raw_len, max_length);
                std::memcpy(out_ptr, scratch.data(), final_len * sizeof(int32_t));
            }
            
            // Truncate if needed
            if (final_len > max_length) {
                final_len = max_length;
                if (add_special_tokens) {
                    out_ptr[max_length - 1] = TEXT_TOK_EOS_ID;
                }
            }
            
            // Pad remaining
            for (int j = final_len; j < max_length; ++j) {
                out_ptr[j] = TEXT_TOK_PAD_ID;
            }
            
            output_lengths[i] = final_len;
        }
    };
    
    // Parallel processing
#if defined(_OPENMP)
    if (batch_size < num_threads * 2 || num_threads == 1) {
        process_range(0, batch_size);
    } else {
        omp_set_dynamic(0);
        omp_set_num_threads(num_threads);
        #pragma omp parallel
        {
            std::vector<int32_t> scratch;
            #pragma omp for schedule(static)
            for (int i = 0; i < batch_size; ++i) {
                const uint8_t* text_data = reinterpret_cast<const uint8_t*>(texts[i]);
                int text_len = text_lengths[i];
                int32_t* out_ptr = output_tokens + (i * max_length);

                size_t needed = static_cast<size_t>(text_len) + 2;
                if (scratch.size() < needed) {
                    scratch.resize(needed);
                }

                int final_len;
                if (trie_ptr && !trie_ptr->empty()) {
                    final_len = fused_tokenize_and_merge(
                        text_data,
                        text_len,
                        *trie_ptr,
                        out_ptr,
                        max_length,
                        byte_offset,
                        add_special_tokens != 0,
                        scratch.data(),
                        false  // inject_thinking
                    );
                } else {
                    int raw_len = text_tokenize_utf8_complex(
                        text_data,
                        text_len,
                        scratch.data(),
                        byte_offset,
                        add_special_tokens != 0,
                        false  // inject_thinking
                    );
                    final_len = std::min(raw_len, max_length);
                    std::memcpy(out_ptr, scratch.data(), final_len * sizeof(int32_t));
                }

                if (final_len > max_length) {
                    final_len = max_length;
                    if (add_special_tokens) {
                        out_ptr[max_length - 1] = TEXT_TOK_EOS_ID;
                    }
                }

                for (int j = final_len; j < max_length; ++j) {
                    out_ptr[j] = TEXT_TOK_PAD_ID;
                }
                output_lengths[i] = final_len;
            }
        }
    }
#else
    if (batch_size < num_threads * 2 || num_threads == 1) {
        process_range(0, batch_size);
    } else {
        std::vector<std::thread> threads;
        int chunk = (batch_size + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int start = t * chunk;
            int end = std::min(start + chunk, batch_size);
            if (start < end) {
                threads.emplace_back(process_range, start, end);
            }
        }
        for (auto& th : threads) th.join();
    }
#endif
    
    return 0;
}

// =============================================================================
// EMBEDDING LOOKUP - SIMD OPTIMIZED
// =============================================================================

void saguaro_native_embed_lookup(
    const int32_t* tokens,
    int batch_size,
    int seq_len,
    const float* projection,
    int vocab_size,
    int dim,
    float* output
) {
    if (!tokens || !projection || !output) return;
    
    // Process in parallel over batch × sequence
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2)
    #endif
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int32_t token_id = tokens[b * seq_len + s];
            
            // Clamp to valid range
            if (token_id < 0) token_id = 0;
            if (token_id >= vocab_size) token_id = vocab_size - 1;
            
            const float* src = projection + (static_cast<size_t>(token_id) * dim);
            float* dst = output + ((static_cast<size_t>(b) * seq_len + s) * dim);
            
            int d = 0;
            
#if defined(__AVX512F__)
            // AVX-512: Process 16 floats at a time
            for (; d + 16 <= dim; d += 16) {
                __m512 v = _mm512_loadu_ps(src + d);
                _mm512_storeu_ps(dst + d, v);
            }
#endif
#if defined(__AVX2__)
            // AVX2: Process 8 floats at a time
            for (; d + 8 <= dim; d += 8) {
                __m256 v = _mm256_loadu_ps(src + d);
                _mm256_storeu_ps(dst + d, v);
            }
#elif defined(__SSE4_2__) || defined(__SSE2__)
            // SSE: Process 4 floats at a time
            for (; d + 4 <= dim; d += 4) {
                __m128 v = _mm_loadu_ps(src + d);
                _mm_storeu_ps(dst + d, v);
            }
#endif
            // Scalar fallback for remaining
            for (; d < dim; ++d) {
                dst[d] = src[d];
            }
        }
    }
}

// =============================================================================
// DOCUMENT VECTOR COMPUTATION - SIMD OPTIMIZED
// =============================================================================

void saguaro_native_compute_doc_vectors(
    const float* embeddings,
    const int32_t* lengths,
    int batch_size,
    int seq_len,
    int dim,
    float* output
) {
    if (!embeddings || !lengths || !output) return;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int b = 0; b < batch_size; ++b) {
        int len = lengths[b];
        if (len <= 0) len = 1;  // Avoid division by zero
        if (len > seq_len) len = seq_len;
        
        float* out_ptr = output + (static_cast<size_t>(b) * dim);
        
        // Initialize to zero using SIMD
        int d = 0;
#if defined(__AVX512F__)
        __m512 zero512 = _mm512_setzero_ps();
        for (; d + 16 <= dim; d += 16) {
            _mm512_storeu_ps(out_ptr + d, zero512);
        }
#endif
#if defined(__AVX2__)
        __m256 zero256 = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            _mm256_storeu_ps(out_ptr + d, zero256);
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        __m128 zero128 = _mm_setzero_ps();
        for (; d + 4 <= dim; d += 4) {
            _mm_storeu_ps(out_ptr + d, zero128);
        }
#endif
        for (; d < dim; ++d) {
            out_ptr[d] = 0.0f;
        }
        
        // Sum with positional encoding
        for (int s = 0; s < len; ++s) {
            const float* emb = embeddings + ((static_cast<size_t>(b) * seq_len + s) * dim);
            float pos_weight = 1.0f + 0.1f * (static_cast<float>(s) / static_cast<float>(len));
            
            d = 0;
#if defined(__AVX512F__)
            __m512 vpos512 = _mm512_set1_ps(pos_weight);
            for (; d + 16 <= dim; d += 16) {
                __m512 ve = _mm512_loadu_ps(emb + d);
                __m512 vo = _mm512_loadu_ps(out_ptr + d);
                __m512 vr = _mm512_fmadd_ps(ve, vpos512, vo);
                _mm512_storeu_ps(out_ptr + d, vr);
            }
#endif
#if defined(__AVX2__)
            __m256 vpos256 = _mm256_set1_ps(pos_weight);
            for (; d + 8 <= dim; d += 8) {
                __m256 ve = _mm256_loadu_ps(emb + d);
                __m256 vo = _mm256_loadu_ps(out_ptr + d);
                __m256 vr = _mm256_fmadd_ps(ve, vpos256, vo);
                _mm256_storeu_ps(out_ptr + d, vr);
            }
#elif defined(__SSE4_2__) || defined(__SSE2__)
            __m128 vpos128 = _mm_set1_ps(pos_weight);
            for (; d + 4 <= dim; d += 4) {
                __m128 ve = _mm_loadu_ps(emb + d);
                __m128 vo = _mm_loadu_ps(out_ptr + d);
                __m128 vr = _mm_add_ps(vo, _mm_mul_ps(ve, vpos128));
                _mm_storeu_ps(out_ptr + d, vr);
            }
#endif
            for (; d < dim; ++d) {
                out_ptr[d] += emb[d] * pos_weight;
            }
        }
        
        // Mean scaling with SIMD
        float scale = 1.0f / static_cast<float>(len);
        d = 0;
#if defined(__AVX512F__)
        __m512 vscale512 = _mm512_set1_ps(scale);
        for (; d + 16 <= dim; d += 16) {
            __m512 v = _mm512_loadu_ps(out_ptr + d);
            _mm512_storeu_ps(out_ptr + d, _mm512_mul_ps(v, vscale512));
        }
#endif
#if defined(__AVX2__)
        __m256 vscale256 = _mm256_set1_ps(scale);
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(out_ptr + d);
            _mm256_storeu_ps(out_ptr + d, _mm256_mul_ps(v, vscale256));
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        __m128 vscale128 = _mm_set1_ps(scale);
        for (; d + 4 <= dim; d += 4) {
            __m128 v = _mm_loadu_ps(out_ptr + d);
            _mm_storeu_ps(out_ptr + d, _mm_mul_ps(v, vscale128));
        }
#endif
        for (; d < dim; ++d) {
            out_ptr[d] *= scale;
        }
    }
}

// =============================================================================
// HOLOGRAPHIC BUNDLING - SIMD OPTIMIZED
// =============================================================================

void saguaro_native_holographic_bundle(
    const float* vectors,
    int num_vectors,
    int dim,
    float* output
) {
    if (!vectors || !output || num_vectors <= 0 || dim <= 0) return;
    
    // Initialize output with first vector using SIMD
    const float* first = vectors;
    int d = 0;
#if defined(__AVX512F__)
    for (; d + 16 <= dim; d += 16) {
        _mm512_storeu_ps(output + d, _mm512_loadu_ps(first + d));
    }
#endif
#if defined(__AVX2__)
    for (; d + 8 <= dim; d += 8) {
        _mm256_storeu_ps(output + d, _mm256_loadu_ps(first + d));
    }
#elif defined(__SSE4_2__) || defined(__SSE2__)
    for (; d + 4 <= dim; d += 4) {
        _mm_storeu_ps(output + d, _mm_loadu_ps(first + d));
    }
#endif
    for (; d < dim; ++d) {
        output[d] = first[d];
    }
    
    // Bundle remaining vectors using SIMD element-wise sum
    for (int v = 1; v < num_vectors; ++v) {
        const float* vec = vectors + (static_cast<size_t>(v) * dim);
        d = 0;
#if defined(__AVX512F__)
        for (; d + 16 <= dim; d += 16) {
            __m512 vo = _mm512_loadu_ps(output + d);
            __m512 vi = _mm512_loadu_ps(vec + d);
            _mm512_storeu_ps(output + d, _mm512_add_ps(vo, vi));
        }
#endif
#if defined(__AVX2__)
        for (; d + 8 <= dim; d += 8) {
            __m256 vo = _mm256_loadu_ps(output + d);
            __m256 vi = _mm256_loadu_ps(vec + d);
            _mm256_storeu_ps(output + d, _mm256_add_ps(vo, vi));
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        for (; d + 4 <= dim; d += 4) {
            __m128 vo = _mm_loadu_ps(output + d);
            __m128 vi = _mm_loadu_ps(vec + d);
            _mm_storeu_ps(output + d, _mm_add_ps(vo, vi));
        }
#endif
        for (; d < dim; ++d) {
            output[d] += vec[d];
        }
    }
    
    // Compute norm with SIMD
    float norm = 0.0f;
    d = 0;
#if defined(__AVX512F__)
    __m512 vnorm512 = _mm512_setzero_ps();
    for (; d + 16 <= dim; d += 16) {
        __m512 v = _mm512_loadu_ps(output + d);
        vnorm512 = _mm512_add_ps(vnorm512, _mm512_mul_ps(v, v));
    }
    norm = _mm512_reduce_add_ps(vnorm512);
#elif defined(__AVX2__)
    __m256 vnorm256 = _mm256_setzero_ps();
    for (; d + 8 <= dim; d += 8) {
        __m256 v = _mm256_loadu_ps(output + d);
        vnorm256 = _mm256_add_ps(vnorm256, _mm256_mul_ps(v, v));
    }
    // Horizontal sum
    __m128 vlow = _mm256_castps256_ps128(vnorm256);
    __m128 vhigh = _mm256_extractf128_ps(vnorm256, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    norm = _mm_cvtss_f32(sums);
#elif defined(__SSE4_2__) || defined(__SSE2__)
    __m128 vnorm128 = _mm_setzero_ps();
    for (; d + 4 <= dim; d += 4) {
        __m128 v = _mm_loadu_ps(output + d);
        vnorm128 = _mm_add_ps(vnorm128, _mm_mul_ps(v, v));
    }
    // Horizontal sum
    __m128 shuf = _mm_movehdup_ps(vnorm128);
    __m128 sums = _mm_add_ps(vnorm128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    norm = _mm_cvtss_f32(sums);
#endif
    for (; d < dim; ++d) {
        norm += output[d] * output[d];
    }
    norm = std::sqrt(norm);
    
    // Normalize with SIMD
    if (norm > 1e-9f) {
        float scale = 1.0f / norm;
        d = 0;
#if defined(__AVX512F__)
        __m512 vscale512 = _mm512_set1_ps(scale);
        for (; d + 16 <= dim; d += 16) {
            _mm512_storeu_ps(output + d, _mm512_mul_ps(_mm512_loadu_ps(output + d), vscale512));
        }
#endif
#if defined(__AVX2__)
        __m256 vscale256 = _mm256_set1_ps(scale);
        for (; d + 8 <= dim; d += 8) {
            __m256 v = _mm256_loadu_ps(output + d);
            v = _mm256_mul_ps(v, vscale256);
            _mm256_storeu_ps(output + d, v);
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        __m128 vscale128 = _mm_set1_ps(scale);
        for (; d + 4 <= dim; d += 4) {
            __m128 v = _mm_loadu_ps(output + d);
            v = _mm_mul_ps(v, vscale128);
            _mm_storeu_ps(output + d, v);
        }
#endif
        for (; d < dim; ++d) {
            output[d] *= scale;
        }
    }
}

void saguaro_native_crystallize(
    const float* knowledge,
    const float* importance,
    int num_vectors,
    int dim,
    float threshold,
    float* output
) {
    if (!knowledge || !importance || !output) return;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int v = 0; v < num_vectors; ++v) {
        const float* k_ptr = knowledge + (static_cast<size_t>(v) * dim);
        const float* imp_ptr = importance + (static_cast<size_t>(v) * dim);
        float* out_ptr = output + (static_cast<size_t>(v) * dim);
        
        int d = 0;
#if defined(__AVX512F__)
        __m512 vthresh512 = _mm512_set1_ps(threshold);
        for (; d + 16 <= dim; d += 16) {
            __m512 vk = _mm512_loadu_ps(k_ptr + d);
            __m512 vi = _mm512_loadu_ps(imp_ptr + d);
            __mmask16 mask = _mm512_cmp_ps_mask(vi, vthresh512, _CMP_GE_OQ);
            __m512 res = _mm512_maskz_mul_ps(mask, vk, vi);
            _mm512_storeu_ps(out_ptr + d, res);
        }
#endif
#if defined(__AVX2__)
        __m256 vthresh256 = _mm256_set1_ps(threshold);
        for (; d + 8 <= dim; d += 8) {
            __m256 vk = _mm256_loadu_ps(k_ptr + d);
            __m256 vi = _mm256_loadu_ps(imp_ptr + d);
            __m256 mask = _mm256_cmp_ps(vi, vthresh256, _CMP_GE_OQ);
            __m256 res = _mm256_and_ps(mask, _mm256_mul_ps(vk, vi));
            _mm256_storeu_ps(out_ptr + d, res);
        }
#endif
        for (; d < dim; ++d) {
            if (imp_ptr[d] >= threshold) {
                out_ptr[d] = k_ptr[d] * imp_ptr[d];
            } else {
                out_ptr[d] = 0.0f;
            }
        }
    }
}

int saguaro_native_rank_jaccard_pairs(
    const int32_t* left_tokens,
    const int32_t* left_lengths,
    int left_count,
    const int32_t* right_tokens,
    const int32_t* right_lengths,
    int right_count,
    int token_stride,
    int top_k,
    int32_t* output_indices,
    float* output_scores,
    int num_threads
) {
    if (!left_tokens || !left_lengths || !right_tokens || !right_lengths || !output_indices || !output_scores) {
        return -1;
    }
    if (left_count < 0 || right_count < 0 || token_stride <= 0 || top_k <= 0) {
        return -1;
    }

    const int max_threads = saguaro_native_max_threads();
    if (num_threads <= 0) {
        num_threads = max_threads;
    }
    num_threads = std::max(1, std::min(num_threads, max_threads));

    const size_t output_size = static_cast<size_t>(left_count) * static_cast<size_t>(top_k);
    for (size_t idx = 0; idx < output_size; ++idx) {
        output_indices[idx] = -1;
        output_scores[idx] = 0.0f;
    }
    if (left_count == 0 || right_count == 0) {
        return 0;
    }

#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#endif
    for (int left_idx = 0; left_idx < left_count; ++left_idx) {
        const int raw_left_len = left_lengths[left_idx];
        const int left_len = std::max(0, std::min(raw_left_len, token_stride));
        const int32_t* left_row = left_tokens + (static_cast<size_t>(left_idx) * token_stride);
        int32_t* best_indices = output_indices + (static_cast<size_t>(left_idx) * top_k);
        float* best_scores = output_scores + (static_cast<size_t>(left_idx) * top_k);

        if (left_len <= 0) {
            continue;
        }

        for (int right_idx = 0; right_idx < right_count; ++right_idx) {
            const int raw_right_len = right_lengths[right_idx];
            const int right_len = std::max(0, std::min(raw_right_len, token_stride));
            if (right_len <= 0) {
                continue;
            }
            const int32_t* right_row = right_tokens + (static_cast<size_t>(right_idx) * token_stride);
            const float score = sorted_unique_jaccard(left_row, left_len, right_row, right_len);

            int insert_at = -1;
            for (int slot = 0; slot < top_k; ++slot) {
                if (score > best_scores[slot] ||
                    (score == best_scores[slot] && (best_indices[slot] < 0 || right_idx < best_indices[slot]))) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at < 0) {
                continue;
            }
            for (int shift = top_k - 1; shift > insert_at; --shift) {
                best_scores[shift] = best_scores[shift - 1];
                best_indices[shift] = best_indices[shift - 1];
            }
            best_scores[insert_at] = score;
            best_indices[insert_at] = right_idx;
        }
    }

    return 0;
}

int saguaro_native_screen_overlap_pairs(
    const int32_t* left_tokens,
    const int32_t* left_lengths,
    int left_count,
    const int32_t* right_tokens,
    const int32_t* right_lengths,
    int right_count,
    int token_stride,
    int top_k,
    int32_t* output_indices,
    float* output_scores,
    int num_threads
) {
    if (!left_tokens || !left_lengths || !right_tokens || !right_lengths || !output_indices || !output_scores) {
        return -1;
    }
    if (left_count < 0 || right_count < 0 || token_stride <= 0 || top_k <= 0) {
        return -1;
    }

    const int max_threads = saguaro_native_max_threads();
    if (num_threads <= 0) {
        num_threads = max_threads;
    }
    num_threads = std::max(1, std::min(num_threads, max_threads));

    const size_t output_size = static_cast<size_t>(left_count) * static_cast<size_t>(top_k);
    for (size_t idx = 0; idx < output_size; ++idx) {
        output_indices[idx] = -1;
        output_scores[idx] = 0.0f;
    }
    if (left_count == 0 || right_count == 0) {
        return 0;
    }

    std::unordered_map<int32_t, std::vector<int>> inverted;
    inverted.reserve(static_cast<size_t>(right_count) * 4U);
    for (int right_idx = 0; right_idx < right_count; ++right_idx) {
        const int raw_right_len = right_lengths[right_idx];
        const int right_len = std::max(0, std::min(raw_right_len, token_stride));
        const int32_t* right_row = right_tokens + (static_cast<size_t>(right_idx) * token_stride);
        for (int token_idx = 0; token_idx < right_len; ++token_idx) {
            inverted[right_row[token_idx]].push_back(right_idx);
        }
    }

#if defined(_OPENMP)
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(static)
#endif
    for (int left_idx = 0; left_idx < left_count; ++left_idx) {
        const int raw_left_len = left_lengths[left_idx];
        const int left_len = std::max(0, std::min(raw_left_len, token_stride));
        const int32_t* left_row = left_tokens + (static_cast<size_t>(left_idx) * token_stride);
        int32_t* best_indices = output_indices + (static_cast<size_t>(left_idx) * top_k);
        float* best_scores = output_scores + (static_cast<size_t>(left_idx) * top_k);

        if (left_len <= 0) {
            continue;
        }

        std::unordered_map<int, int> overlap_counts;
        overlap_counts.reserve(static_cast<size_t>(left_len) * 8U);
        for (int token_idx = 0; token_idx < left_len; ++token_idx) {
            auto postings_it = inverted.find(left_row[token_idx]);
            if (postings_it == inverted.end()) {
                continue;
            }
            for (int right_idx : postings_it->second) {
                ++overlap_counts[right_idx];
            }
        }

        for (const auto& entry : overlap_counts) {
            const int right_idx = entry.first;
            const int overlap_count = entry.second;
            const float score = static_cast<float>(overlap_count) / static_cast<float>(std::max(left_len, 1));
            int insert_at = -1;
            for (int slot = 0; slot < top_k; ++slot) {
                if (score > best_scores[slot] ||
                    (score == best_scores[slot] && (best_indices[slot] < 0 || right_idx < best_indices[slot]))) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at < 0) {
                continue;
            }
            for (int shift = top_k - 1; shift > insert_at; --shift) {
                best_scores[shift] = best_scores[shift - 1];
                best_indices[shift] = best_indices[shift - 1];
            }
            best_scores[insert_at] = score;
            best_indices[insert_at] = right_idx;
        }
    }

    return 0;
}

// =============================================================================
// FULL PIPELINE
// =============================================================================

int saguaro_native_full_pipeline(
    const char* const* texts,
    const int* text_lengths,
    int batch_size,
    const float* projection,
    int vocab_size,
    int dim,
    int max_length,
    saguaro_trie_handle_t trie,
    float* output,
    int num_threads
) {
    if (!texts || !text_lengths || !projection || !output) return -1;
    if (batch_size <= 0 || vocab_size <= 0 || dim <= 0 || max_length <= 0) return -1;
    const int max_threads = saguaro_native_max_threads();
    if (num_threads <= 0) {
        num_threads = max_threads;
    }
    num_threads = std::max(1, std::min(num_threads, max_threads));

    std::vector<int32_t> tokens(static_cast<size_t>(batch_size) * max_length);
    std::vector<int32_t> lengths(batch_size);

    int ret = saguaro_native_tokenize_batch(
        texts, text_lengths, batch_size,
        tokens.data(), lengths.data(),
        max_length, 32, 1, trie, num_threads
    );
    if (ret != 0) return ret;

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, num_threads));
    }
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batch_size; ++b) {
        int len = lengths[b];
        if (len <= 0) len = 1;
        if (len > max_length) len = max_length;

        float* out_ptr = output + (static_cast<size_t>(b) * dim);

        int d = 0;
#if defined(__AVX512F__)
        const __m512 zero512 = _mm512_setzero_ps();
        for (; d + 16 <= dim; d += 16) {
            _mm512_storeu_ps(out_ptr + d, zero512);
        }
#endif
#if defined(__AVX2__)
        const __m256 zero256 = _mm256_setzero_ps();
        for (; d + 8 <= dim; d += 8) {
            _mm256_storeu_ps(out_ptr + d, zero256);
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        const __m128 zero128 = _mm_setzero_ps();
        for (; d + 4 <= dim; d += 4) {
            _mm_storeu_ps(out_ptr + d, zero128);
        }
#endif
        for (; d < dim; ++d) {
            out_ptr[d] = 0.0f;
        }

        for (int s = 0; s < len; ++s) {
            int32_t token_id = tokens[static_cast<size_t>(b) * max_length + s];
            if (token_id < 0) token_id = 0;
            if (token_id >= vocab_size) token_id = vocab_size - 1;

            const float* src = projection + (static_cast<size_t>(token_id) * dim);
            const float pos_weight =
                1.0f + 0.1f * (static_cast<float>(s) / static_cast<float>(len));

            d = 0;
#if defined(__AVX512F__)
            const __m512 vpos512 = _mm512_set1_ps(pos_weight);
            for (; d + 16 <= dim; d += 16) {
                const __m512 vs = _mm512_loadu_ps(src + d);
                const __m512 vo = _mm512_loadu_ps(out_ptr + d);
                _mm512_storeu_ps(out_ptr + d, _mm512_fmadd_ps(vs, vpos512, vo));
            }
#endif
#if defined(__AVX2__)
            const __m256 vpos256 = _mm256_set1_ps(pos_weight);
            for (; d + 8 <= dim; d += 8) {
                const __m256 vs = _mm256_loadu_ps(src + d);
                const __m256 vo = _mm256_loadu_ps(out_ptr + d);
                _mm256_storeu_ps(out_ptr + d, _mm256_fmadd_ps(vs, vpos256, vo));
            }
#elif defined(__SSE4_2__) || defined(__SSE2__)
            const __m128 vpos128 = _mm_set1_ps(pos_weight);
            for (; d + 4 <= dim; d += 4) {
                const __m128 vs = _mm_loadu_ps(src + d);
                const __m128 vo = _mm_loadu_ps(out_ptr + d);
                _mm_storeu_ps(out_ptr + d, _mm_add_ps(vo, _mm_mul_ps(vs, vpos128)));
            }
#endif
            for (; d < dim; ++d) {
                out_ptr[d] += src[d] * pos_weight;
            }
        }

        const float scale = 1.0f / static_cast<float>(len);
        d = 0;
#if defined(__AVX512F__)
        const __m512 vscale512 = _mm512_set1_ps(scale);
        for (; d + 16 <= dim; d += 16) {
            const __m512 v = _mm512_loadu_ps(out_ptr + d);
            _mm512_storeu_ps(out_ptr + d, _mm512_mul_ps(v, vscale512));
        }
#endif
#if defined(__AVX2__)
        const __m256 vscale256 = _mm256_set1_ps(scale);
        for (; d + 8 <= dim; d += 8) {
            const __m256 v = _mm256_loadu_ps(out_ptr + d);
            _mm256_storeu_ps(out_ptr + d, _mm256_mul_ps(v, vscale256));
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        const __m128 vscale128 = _mm_set1_ps(scale);
        for (; d + 4 <= dim; d += 4) {
            const __m128 v = _mm_loadu_ps(out_ptr + d);
            _mm_storeu_ps(out_ptr + d, _mm_mul_ps(v, vscale128));
        }
#endif
        for (; d < dim; ++d) {
            out_ptr[d] *= scale;
        }
    }
    
    return 0;
}

int saguaro_native_full_pipeline_strided(
    const char* const* texts,
    const int* text_lengths,
    int batch_size,
    const float* projection,
    int vocab_size,
    int dim,
    int output_dim,
    int output_stride,
    int max_length,
    saguaro_trie_handle_t trie,
    float* output,
    int num_threads
) {
    if (!texts || !text_lengths || !projection || !output) return -1;
    if (batch_size <= 0 || vocab_size <= 0 || dim <= 0 || output_dim <= 0 || output_stride < output_dim || max_length <= 0) return -1;
    const int max_threads = saguaro_native_max_threads();
    if (num_threads <= 0) {
        num_threads = max_threads;
    }
    num_threads = std::max(1, std::min(num_threads, max_threads));

    std::vector<int32_t> tokens(static_cast<size_t>(batch_size) * max_length);
    std::vector<int32_t> lengths(batch_size);

    int ret = saguaro_native_tokenize_batch(
        texts, text_lengths, batch_size,
        tokens.data(), lengths.data(),
        max_length, 32, 1, trie, num_threads
    );
    if (ret != 0) return ret;

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_dynamic(0);
        omp_set_num_threads(std::max(1, num_threads));
    }
    #pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < batch_size; ++b) {
        int len = lengths[b];
        if (len <= 0) len = 1;
        if (len > max_length) len = max_length;

        float* out_ptr = output + (static_cast<size_t>(b) * output_stride);
        int d = 0;
#if defined(__AVX512F__)
        const __m512 zero512 = _mm512_setzero_ps();
        for (; d + 16 <= output_dim; d += 16) {
            _mm512_storeu_ps(out_ptr + d, zero512);
        }
#endif
#if defined(__AVX2__)
        const __m256 zero256 = _mm256_setzero_ps();
        for (; d + 8 <= output_dim; d += 8) {
            _mm256_storeu_ps(out_ptr + d, zero256);
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        const __m128 zero128 = _mm_setzero_ps();
        for (; d + 4 <= output_dim; d += 4) {
            _mm_storeu_ps(out_ptr + d, zero128);
        }
#endif
        for (; d < output_dim; ++d) {
            out_ptr[d] = 0.0f;
        }

        for (int s = 0; s < len; ++s) {
            int32_t token_id = tokens[static_cast<size_t>(b) * max_length + s];
            if (token_id < 0) token_id = 0;
            if (token_id >= vocab_size) token_id = vocab_size - 1;

            const float* src = projection + (static_cast<size_t>(token_id) * dim);
            const float pos_weight =
                1.0f + 0.1f * (static_cast<float>(s) / static_cast<float>(len));

            d = 0;
#if defined(__AVX512F__)
            const __m512 vpos512 = _mm512_set1_ps(pos_weight);
            for (; d + 16 <= dim; d += 16) {
                const __m512 vs = _mm512_loadu_ps(src + d);
                const __m512 vo = _mm512_loadu_ps(out_ptr + d);
                _mm512_storeu_ps(out_ptr + d, _mm512_fmadd_ps(vs, vpos512, vo));
            }
#endif
#if defined(__AVX2__)
            const __m256 vpos256 = _mm256_set1_ps(pos_weight);
            for (; d + 8 <= dim; d += 8) {
                const __m256 vs = _mm256_loadu_ps(src + d);
                const __m256 vo = _mm256_loadu_ps(out_ptr + d);
                _mm256_storeu_ps(out_ptr + d, _mm256_fmadd_ps(vs, vpos256, vo));
            }
#elif defined(__SSE4_2__) || defined(__SSE2__)
            const __m128 vpos128 = _mm_set1_ps(pos_weight);
            for (; d + 4 <= dim; d += 4) {
                const __m128 vs = _mm_loadu_ps(src + d);
                const __m128 vo = _mm_loadu_ps(out_ptr + d);
                _mm_storeu_ps(out_ptr + d, _mm_add_ps(vo, _mm_mul_ps(vs, vpos128)));
            }
#endif
            for (; d < dim; ++d) {
                out_ptr[d] += src[d] * pos_weight;
            }
        }

        const float scale = 1.0f / static_cast<float>(len);
        d = 0;
#if defined(__AVX512F__)
        const __m512 vscale512 = _mm512_set1_ps(scale);
        for (; d + 16 <= dim; d += 16) {
            const __m512 v = _mm512_loadu_ps(out_ptr + d);
            _mm512_storeu_ps(out_ptr + d, _mm512_mul_ps(v, vscale512));
        }
#endif
#if defined(__AVX2__)
        const __m256 vscale256 = _mm256_set1_ps(scale);
        for (; d + 8 <= dim; d += 8) {
            const __m256 v = _mm256_loadu_ps(out_ptr + d);
            _mm256_storeu_ps(out_ptr + d, _mm256_mul_ps(v, vscale256));
        }
#elif defined(__SSE4_2__) || defined(__SSE2__)
        const __m128 vscale128 = _mm_set1_ps(scale);
        for (; d + 4 <= dim; d += 4) {
            const __m128 v = _mm_loadu_ps(out_ptr + d);
            _mm_storeu_ps(out_ptr + d, _mm_mul_ps(v, vscale128));
        }
#endif
        for (; d < dim; ++d) {
            out_ptr[d] *= scale;
        }
    }

    return 0;
}

int saguaro_native_match_capture_names(
    const int32_t* def_starts,
    const int32_t* def_ends,
    const int32_t* def_type_ids,
    int def_count,
    const int32_t* name_starts,
    const int32_t* name_ends,
    const int32_t* name_type_ids,
    int name_count,
    int32_t* output_name_indices
) {
    if (!def_starts || !def_ends || !def_type_ids || !output_name_indices || def_count < 0) {
        return -1;
    }
    if (name_count > 0 && (!name_starts || !name_ends || !name_type_ids)) {
        return -2;
    }

    for (int i = 0; i < def_count; ++i) {
        output_name_indices[i] = -1;
    }
    if (def_count == 0 || name_count == 0) {
        return 0;
    }

    struct CaptureBucket {
        std::vector<int32_t> def_indices;
        std::vector<int32_t> name_indices;
    };

    std::vector<CaptureBucket> buckets;
    buckets.reserve(16);
    std::unordered_map<int32_t, size_t> bucket_lookup;
    bucket_lookup.reserve(16);

    auto ensure_bucket = [&](int32_t type_id) -> CaptureBucket& {
        const auto it = bucket_lookup.find(type_id);
        if (it != bucket_lookup.end()) {
            return buckets[it->second];
        }
        const size_t next_index = buckets.size();
        buckets.emplace_back();
        bucket_lookup.emplace(type_id, next_index);
        return buckets.back();
    };

    for (int32_t i = 0; i < def_count; ++i) {
        ensure_bucket(def_type_ids[i]).def_indices.push_back(i);
    }
    for (int32_t i = 0; i < name_count; ++i) {
        ensure_bucket(name_type_ids[i]).name_indices.push_back(i);
    }
    std::vector<int32_t> best_spans(
        static_cast<size_t>(def_count),
        std::numeric_limits<int32_t>::max()
    );

#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if(buckets.size() > 1)
#endif
    for (size_t bucket_index = 0; bucket_index < buckets.size(); ++bucket_index) {
        const CaptureBucket& bucket = buckets[bucket_index];
        std::vector<int32_t> active_defs;
        active_defs.reserve(bucket.def_indices.size());

        size_t def_cursor = 0;
        size_t name_cursor = 0;
        while (def_cursor < bucket.def_indices.size() || name_cursor < bucket.name_indices.size()) {
            const bool take_def =
                def_cursor < bucket.def_indices.size() &&
                (name_cursor >= bucket.name_indices.size() ||
                 def_starts[bucket.def_indices[def_cursor]] <= name_starts[bucket.name_indices[name_cursor]]);
            const int32_t current_start = take_def
                ? def_starts[bucket.def_indices[def_cursor]]
                : name_starts[bucket.name_indices[name_cursor]];

            while (!active_defs.empty() && def_ends[active_defs.back()] < current_start) {
                active_defs.pop_back();
            }

            if (take_def) {
                active_defs.push_back(bucket.def_indices[def_cursor]);
                ++def_cursor;
                continue;
            }

            const int32_t name_index = bucket.name_indices[name_cursor];
            if (!active_defs.empty()) {
                const int32_t def_index = active_defs.back();
                if (name_ends[name_index] <= def_ends[def_index]) {
                    const int32_t span = name_ends[name_index] - name_starts[name_index];
                    if (span < best_spans[def_index]) {
                        best_spans[def_index] = span;
                        output_name_indices[def_index] = name_index;
                    }
                }
            }
            ++name_cursor;
        }
    }

    return 0;
}

}  // extern "C"
