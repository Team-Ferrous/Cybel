// saguaro.native/ops/hnn_simd_compat.h
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
 * @file hnn_simd_compat.h
 * @brief Compatibility layer for migrating to unified SIMD utilities.
 *
 * This header provides namespace aliases and inline wrappers to allow
 * existing code using various SIMD implementations to migrate to the
 * unified saguaro::simd utilities in hnn_simd_common.h.
 *
 * Usage:
 *   Instead of: #include "hnn_simd_common.h" and using saguaro::ops::simd_*
 *   Use:        #include "hnn_simd_compat.h" and keep using existing namespace
 *
 * Migration path:
 *   1. Replace individual SIMD helper includes with this file
 *   2. Code continues working via compatibility aliases
 *   3. Gradually update call sites to use saguaro::simd:: directly
 *   4. Remove this header once migration complete
 */

#ifndef SAGUARO_NATIVE_OPS_HNN_SIMD_COMPAT_H_
#define SAGUARO_NATIVE_OPS_HNN_SIMD_COMPAT_H_

#include "hnn_simd_common.h"

// =============================================================================
// NAMESPACE ALIASES FOR BACKWARD COMPATIBILITY
// These allow existing code to work without modification during migration
// =============================================================================

namespace saguaro {
namespace hd_hierarchical {

// Forward declarations from fused_hd_hierarchical_block_op.h
// Replace with calls to unified utilities

/**
 * @brief Compatibility wrapper for hier_dot_product
 */
inline float hier_dot_product(const float* a, const float* b, int64_t size) {
    return saguaro::simd::simd_dot_product(a, b, size);
}

/**
 * @brief Compatibility wrapper for hier_norm
 */
inline float hier_norm(const float* x, int64_t size) {
    return saguaro::simd::simd_norm(x, size);
}

/**
 * @brief Compatibility wrapper for hier_add_scaled
 */
inline void hier_add_scaled(const float* a, const float* b, float scale,
                            float* out, int64_t size) {
    saguaro::simd::simd_add_scaled(a, b, scale, out, size);
}

/**
 * @brief Compatibility wrapper for hier_ema_blend
 */
inline void hier_ema_blend(const float* old_val, const float* new_val,
                           float alpha, float* out, int64_t size) {
    saguaro::simd::simd_ema_blend(old_val, new_val, alpha, out, size);
}

}  // namespace hd_hierarchical
}  // namespace saguaro

namespace saguaro {
namespace ops {

// Aliases for memory bank operations
namespace membank {

/**
 * @brief Compatibility wrapper for membank_cosine_similarity
 */
inline void compute_cosine_similarities(
    const float* query, const float* keys, float* out,
    int64_t num_keys, int64_t dim) {
    saguaro::simd::simd_batch_cosine_similarity(query, keys, out, num_keys, dim);
}

/**
 * @brief Compatibility wrapper for single cosine similarity
 */
inline float cosine_similarity(const float* a, const float* b, int64_t size) {
    return saguaro::simd::simd_cosine_similarity(a, b, size);
}

}  // namespace membank

// Aliases for generic SIMD operations - promotes to saguaro::ops namespace
using saguaro::simd::simd_dot_product;
using saguaro::simd::simd_norm;
using saguaro::simd::simd_cosine_similarity;
using saguaro::simd::simd_batch_cosine_similarity;
using saguaro::simd::simd_add_scaled;
using saguaro::simd::simd_gated_update;
using saguaro::simd::simd_ema_blend;
using saguaro::simd::simd_rms_norm;
using saguaro::simd::simd_tanh_inplace;

}  // namespace ops
}  // namespace saguaro

// =============================================================================
// TENSORFLOW HPC CPU COMPATIBILITY
// Some ops use tensorflow::hpc::cpu namespace for SIMD helpers
// =============================================================================

namespace tensorflow {
namespace hpc {
namespace cpu {

// Custom reduce function that may exist in some ops
#if defined(__AVX512F__)
inline float _mm512_reduce_add_ps_custom(__m512 v) {
    return _mm512_reduce_add_ps(v);
}
#endif

// Softmax from unified utilities
inline void simd_softmax(float* data, int64_t size) {
    saguaro::ops::simd_softmax_inplace(data, size);
}

}  // namespace cpu
}  // namespace hpc
}  // namespace tensorflow

// =============================================================================
// ANONYMOUS NAMESPACE MIGRATION HELPERS
// Many ops define helpers in anonymous namespaces. These macros help replace them.
// =============================================================================

/**
 * @def USE_UNIFIED_DOT_PRODUCT
 * Replace anonymous namespace dot_product with unified version
 */
#define USE_UNIFIED_DOT_PRODUCT \
    inline float dot_product(const float* a, const float* b, int64_t size) { \
        return saguaro::simd::simd_dot_product(a, b, size); \
    }

/**
 * @def USE_UNIFIED_NORM
 * Replace anonymous namespace norm with unified version
 */
#define USE_UNIFIED_NORM \
    inline float norm(const float* x, int64_t size) { \
        return saguaro::simd::simd_norm(x, size); \
    }

/**
 * @def USE_UNIFIED_COSINE_SIM
 * Replace anonymous namespace cosine_similarity with unified version
 */
#define USE_UNIFIED_COSINE_SIM \
    inline float cosine_similarity(const float* a, const float* b, int64_t size) { \
        return saguaro::simd::simd_cosine_similarity(a, b, size); \
    }

/**
 * @def USE_UNIFIED_SOFTMAX
 * Replace anonymous namespace softmax with unified version
 */
#define USE_UNIFIED_SOFTMAX \
    inline void softmax(float* data, int64_t size) { \
        saguaro::ops::simd_softmax_inplace(data, size); \
    }

/**
 * @def USE_UNIFIED_GATED_UPDATE
 * Replace anonymous namespace gated_update with unified version
 */
#define USE_UNIFIED_GATED_UPDATE \
    inline void gated_update(const float* gate, const float* current, \
                             const float* update, float* out, int64_t size) { \
        saguaro::simd::simd_gated_update(gate, current, update, out, size); \
    }

/**
 * @def USE_ALL_UNIFIED_SIMD
 * Replace all common anonymous namespace SIMD helpers
 */
#define USE_ALL_UNIFIED_SIMD \
    USE_UNIFIED_DOT_PRODUCT \
    USE_UNIFIED_NORM \
    USE_UNIFIED_COSINE_SIM \
    USE_UNIFIED_SOFTMAX \
    USE_UNIFIED_GATED_UPDATE

#endif  // SAGUARO_NATIVE_OPS_HNN_SIMD_COMPAT_H_
