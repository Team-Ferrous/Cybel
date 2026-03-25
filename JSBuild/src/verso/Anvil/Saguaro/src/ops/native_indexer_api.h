// saguaro/native/ops/native_indexer_api.h
// Copyright 2025 Verso Industries (Author: Michael B. Zimmerman)
//
// Native C API for indexer operations - NO TensorFlow dependency.
// These functions are exported directly from _saguaro_core.so and can be
// called via Python ctypes without loading TensorFlow.

#pragma once

#include <cstdint>
#include <cstddef>

// Symbol visibility for export
#if defined(_WIN32) || defined(__CYGWIN__)
    #define SAGUARO_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
    #define SAGUARO_EXPORT __attribute__((visibility("default")))
#else
    #define SAGUARO_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// TRIE MANAGEMENT (for superword merging)
// =============================================================================

/**
 * Opaque handle to a SuperwordTrie.
 */
typedef void* saguaro_trie_handle_t;

/**
 * Create a new empty SuperwordTrie.
 * @return Handle to the trie, or NULL on failure.
 */
SAGUARO_EXPORT saguaro_trie_handle_t saguaro_native_trie_create(void);

/**
 * Destroy a SuperwordTrie and free its memory.
 * @param trie Handle to destroy.
 */
SAGUARO_EXPORT void saguaro_native_trie_destroy(saguaro_trie_handle_t trie);

/**
 * Insert an n-gram mapping into the trie.
 * @param trie Handle to the trie.
 * @param ngram Array of token IDs forming the n-gram.
 * @param ngram_len Length of the n-gram.
 * @param superword_id The superword ID to map to.
 */
SAGUARO_EXPORT void saguaro_native_trie_insert(
    saguaro_trie_handle_t trie,
    const int32_t* ngram,
    int ngram_len,
    int32_t superword_id
);

/**
 * Build trie from table format.
 * @param trie Handle to the trie (will be cleared first).
 * @param offsets Array of offsets [num_ngrams + 1].
 * @param tokens Concatenated tokens for all n-grams.
 * @param superword_ids Array of superword IDs [num_ngrams].
 * @param num_ngrams Number of n-grams.
 */
SAGUARO_EXPORT void saguaro_native_trie_build_from_table(
    saguaro_trie_handle_t trie,
    const int32_t* offsets,
    const int32_t* tokens,
    const int32_t* superword_ids,
    int num_ngrams
);

// =============================================================================
// TOKENIZATION
// =============================================================================

/**
 * Tokenize a batch of UTF-8 texts.
 * 
 * @param texts Array of pointers to UTF-8 strings.
 * @param text_lengths Array of string lengths (bytes).
 * @param batch_size Number of texts in the batch.
 * @param output_tokens Output buffer [batch_size * max_length] for token IDs.
 * @param output_lengths Output buffer [batch_size] for actual lengths.
 * @param max_length Maximum sequence length (truncate/pad to this).
 * @param byte_offset Offset added to each byte value (typically 32).
 * @param add_special_tokens Whether to add CLS/EOS tokens.
 * @param trie Optional trie for superword merging (NULL = no merging).
 * @param num_threads Number of threads (0 = auto).
 * @return 0 on success, non-zero on error.
 */
SAGUARO_EXPORT int saguaro_native_tokenize_batch(
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
);

// =============================================================================
// PROJECTION INITIALIZATION
// =============================================================================

/**
 * Deterministically initialize a projection matrix in-place.
 *
 * @param projection Writable projection buffer [vocab_size, dim].
 * @param vocab_size Vocabulary size.
 * @param dim Embedding dimension.
 * @param seed Deterministic initialization seed.
 */
SAGUARO_EXPORT void saguaro_native_init_projection(
    float* projection,
    int vocab_size,
    int dim,
    uint64_t seed
);

// =============================================================================
// EMBEDDING LOOKUP
// =============================================================================

/**
 * Perform embedding lookup from a projection matrix.
 * 
 * @param tokens Token IDs [batch_size, seq_len].
 * @param batch_size Number of sequences.
 * @param seq_len Sequence length.
 * @param projection Projection matrix [vocab_size, dim] (shared memory).
 * @param vocab_size Vocabulary size.
 * @param dim Embedding dimension.
 * @param output Output embeddings [batch_size, seq_len, dim].
 */
SAGUARO_EXPORT void saguaro_native_embed_lookup(
    const int32_t* tokens,
    int batch_size,
    int seq_len,
    const float* projection,
    int vocab_size,
    int dim,
    float* output
);

// =============================================================================
// DOCUMENT VECTOR COMPUTATION
// =============================================================================

/**
 * Compute document vectors via mean pooling with positional encoding.
 * 
 * @param embeddings Input embeddings [batch_size, seq_len, dim].
 * @param lengths Actual sequence lengths [batch_size].
 * @param batch_size Number of sequences.
 * @param seq_len Maximum sequence length.
 * @param dim Embedding dimension.
 * @param output Output document vectors [batch_size, dim].
 */
SAGUARO_EXPORT void saguaro_native_compute_doc_vectors(
    const float* embeddings,
    const int32_t* lengths,
    int batch_size,
    int seq_len,
    int dim,
    float* output
);

// =============================================================================
// HOLOGRAPHIC BUNDLING
// =============================================================================

/**
 * Bundle multiple vectors into a single holographic representation.
 * Uses circular convolution-based binding.
 * 
 * @param vectors Input vectors [num_vectors, dim].
 * @param num_vectors Number of vectors to bundle.
 * @param dim Vector dimension.
 * @param output Output bundled vector [dim].
 */
SAGUARO_EXPORT void saguaro_native_holographic_bundle(
    const float* vectors,
    int num_vectors,
    int dim,
    float* output
);

/**
 * Crystallize memory by applying importance-weighted thresholding.
 * 
 * @param knowledge Knowledge vectors [num_vectors, dim].
 * @param importance Importance weights [num_vectors, dim].
 * @param num_vectors Number of vectors.
 * @param dim Vector dimension.
 * @param threshold Crystallization threshold.
 * @param output Crystallized output [num_vectors, dim].
 */
SAGUARO_EXPORT void saguaro_native_crystallize(
    const float* knowledge,
    const float* importance,
    int num_vectors,
    int dim,
    float threshold,
    float* output
);

// =============================================================================
// COMPARATIVE RELATION SCORING
// =============================================================================

/**
 * Rank top-k Jaccard matches between left and right token sets.
 *
 * Inputs are fixed-stride, sorted, unique token-id rows. Length arrays provide
 * the active token count for each row, allowing callers to reuse padded buffers.
 *
 * @param left_tokens Left token table [left_count, token_stride].
 * @param left_lengths Active token counts [left_count].
 * @param left_count Number of left rows.
 * @param right_tokens Right token table [right_count, token_stride].
 * @param right_lengths Active token counts [right_count].
 * @param right_count Number of right rows.
 * @param token_stride Row stride in token ids.
 * @param top_k Number of matches to retain per left row.
 * @param output_indices Output best-match indices [left_count, top_k], -1 when empty.
 * @param output_scores Output best-match Jaccard scores [left_count, top_k].
 * @param num_threads Number of threads (0 = auto).
 * @return 0 on success, non-zero on error.
 */
SAGUARO_EXPORT int saguaro_native_rank_jaccard_pairs(
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
);

/**
 * Screen top-k overlap candidates between left and right token sets.
 *
 * This is a native prefilter intended to reduce the candidate matrix before
 * exact Jaccard verification. Scores are normalized overlap counts in the
 * range [0, 1], using left-row length as the denominator.
 *
 * @param left_tokens Left token table [left_count, token_stride].
 * @param left_lengths Active token counts [left_count].
 * @param left_count Number of left rows.
 * @param right_tokens Right token table [right_count, token_stride].
 * @param right_lengths Active token counts [right_count].
 * @param right_count Number of right rows.
 * @param token_stride Row stride in token ids.
 * @param top_k Number of candidates to retain per left row.
 * @param output_indices Output candidate indices [left_count, top_k], -1 when empty.
 * @param output_scores Output normalized overlap scores [left_count, top_k].
 * @param num_threads Number of threads (0 = auto).
 * @return 0 on success, non-zero on error.
 */
SAGUARO_EXPORT int saguaro_native_screen_overlap_pairs(
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
);

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
// NATIVE SENTINEL (High-Speed Governance)
// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

/**
 * Opaque handle to a NativeSentinelEngine.
 */
typedef void* saguaro_sentinel_handle_t;

/**
 * Create a new NativeSentinelEngine.
 */
SAGUARO_EXPORT saguaro_sentinel_handle_t saguaro_native_sentinel_create(void);

/**
 * Destroy a NativeSentinelEngine.
 */
SAGUARO_EXPORT void saguaro_native_sentinel_destroy(saguaro_sentinel_handle_t handle);

/**
 * Scan a file for violations (secrets, governance).
 * @param handle Handle to the engine.
 * @param file_path Path to the file to scan.
 * @param output_json Buffer for JSON results.
 * @param max_len Size of output buffer.
 * @return Length of JSON string, or negative on error.
 */
SAGUARO_EXPORT int saguaro_native_sentinel_scan(
    saguaro_sentinel_handle_t handle,
    const char* file_path,
    char* output_json,
    int max_len
);

// =============================================================================
// FULL PIPELINE (text -> document vector)
// =============================================================================

/**
 * Full indexing pipeline: text -> document vectors.
 * Combines tokenization, embedding, and pooling in one call.
 * 
 * @param texts Array of UTF-8 text pointers.
 * @param text_lengths Array of text lengths.
 * @param batch_size Number of texts.
 * @param projection Projection matrix [vocab_size, dim].
 * @param vocab_size Vocabulary size.
 * @param dim Embedding dimension.
 * @param max_length Maximum sequence length.
 * @param trie Optional superword trie (NULL = no merging).
 * @param output Output document vectors [batch_size, dim].
 * @param num_threads Number of threads (0 = auto).
 * @return 0 on success, non-zero on error.
 */
SAGUARO_EXPORT int saguaro_native_full_pipeline(
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
);

// =============================================================================
// TREE-SITTER CAPTURE MATCHING
// =============================================================================

/**
 * Match definition captures to the best enclosed name capture of the same type.
 *
 * Type identifiers are caller-defined integers that must be consistent across
 * defs and names for a single invocation.
 *
 * @param def_starts Definition start-byte offsets [def_count].
 * @param def_ends Definition end-byte offsets [def_count].
 * @param def_type_ids Definition type identifiers [def_count].
 * @param def_count Number of definition captures.
 * @param name_starts Name start-byte offsets [name_count].
 * @param name_ends Name end-byte offsets [name_count].
 * @param name_type_ids Name type identifiers [name_count].
 * @param name_count Number of name captures.
 * @param output_name_indices Output best-matching name index per def [def_count].
 * @return 0 on success, non-zero on error.
 */
SAGUARO_EXPORT int saguaro_native_match_capture_names(
    const int32_t* def_starts,
    const int32_t* def_ends,
    const int32_t* def_type_ids,
    int def_count,
    const int32_t* name_starts,
    const int32_t* name_ends,
    const int32_t* name_type_ids,
    int name_count,
    int32_t* output_name_indices
);

// =============================================================================
// VERSION / INFO
// =============================================================================

/**
 * Get library version string.
 * @return Version string (e.g., "1.0.0").
 */
SAGUARO_EXPORT const char* saguaro_native_version(void);

/**
 * Check if native API is available.
 * @return 1 if available, 0 otherwise.
 */
SAGUARO_EXPORT int saguaro_native_available(void);

/**
 * Get a compact build signature for the native compute fabric.
 * @return Signature string describing compiler and acceleration features.
 */
SAGUARO_EXPORT const char* saguaro_native_build_signature(void);

/**
 * Get the baseline SIMD ISA compiled into the native compute fabric.
 * @return ISA label such as "avx2" or "scalar".
 */
SAGUARO_EXPORT const char* saguaro_native_isa_baseline(void);

/**
 * Report whether the native compute fabric was compiled with OpenMP support.
 * @return 1 when OpenMP is enabled at build time, else 0.
 */
SAGUARO_EXPORT int saguaro_native_openmp_enabled(void);

/**
 * Report whether the native compute fabric was compiled with AVX2 support.
 * @return 1 when AVX2 is enabled at build time, else 0.
 */
SAGUARO_EXPORT int saguaro_native_avx2_enabled(void);

/**
 * Report whether the native compute fabric was compiled with FMA support.
 * @return 1 when FMA is enabled at build time, else 0.
 */
SAGUARO_EXPORT int saguaro_native_fma_enabled(void);

/**
 * Report the maximum thread count available to the native compute fabric.
 * @return Maximum available worker threads.
 */
SAGUARO_EXPORT int saguaro_native_max_threads(void);

#ifdef __cplusplus
}  // extern "C"
#endif
