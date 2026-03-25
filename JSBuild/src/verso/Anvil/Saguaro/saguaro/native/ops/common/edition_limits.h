// saguaro.native/ops/common/edition_limits.h
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
//
// =============================================================================
// HIGHNOON EDITION-BASED SCALE LIMITS
// =============================================================================
//
// This header defines compile-time scale limits based on the build edition.
// The edition is set at compile time via -DSAGUARO_EDITION=X and cannot be changed
// at runtime without recompiling the binaries.
//
// EDITION TIERS:
//   - LITE (0):       Free tier with scale limits enforced
//   - PRO (1):        Paid tier with no scale limits, pre-compiled binary
//   - ENTERPRISE (2): Source code access + no limits + dedicated support
//
// IMPORTANT: Modification of this file requires recompilation of all binaries.
// The limits are embedded at compile-time and verified via chain authentication.

#ifndef SAGUARO_EDITION_LIMITS_H_
#define SAGUARO_EDITION_LIMITS_H_

#include "tensorflow/core/framework/op_kernel.h"
#include <cstdint>
#include <cstdlib>

#include "runtime_security.h"

namespace saguaro {
namespace limits {

// =============================================================================
// EDITION TYPES (set at compile time via -DSAGUARO_EDITION=X)
// =============================================================================

#define SAGUARO_EDITION_LITE       0
#define SAGUARO_EDITION_PRO        1
#define SAGUARO_EDITION_ENTERPRISE 2

// Default to Lite edition if not specified
#ifndef SAGUARO_EDITION
#define SAGUARO_EDITION SAGUARO_EDITION_LITE
#endif

// =============================================================================
// EDITION STRING CONSTANTS
// =============================================================================

#if SAGUARO_EDITION == SAGUARO_EDITION_ENTERPRISE
constexpr const char* kEdition = "ENTERPRISE";
constexpr const char* kEditionLong = "Enterprise Edition";
#elif SAGUARO_EDITION == SAGUARO_EDITION_PRO
constexpr const char* kEdition = "PRO";
constexpr const char* kEditionLong = "Pro Edition";
#else
constexpr const char* kEdition = "LITE";
constexpr const char* kEditionLong = "Lite Edition";
#endif

// Edition as integer for runtime queries
constexpr int32_t kEditionCode = SAGUARO_EDITION;

// =============================================================================
// SCALE LIMITS - Compile-time constants based on edition
// =============================================================================

#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO
// Pro and Enterprise: No scale limits (effectively unlimited)
constexpr int64_t kMaxTotalParams = INT64_MAX;
constexpr int64_t kMaxContextLength = INT64_MAX;
constexpr int32_t kMaxReasoningBlocks = INT32_MAX;
constexpr int32_t kMaxMoeExperts = INT32_MAX;
constexpr int32_t kMaxEmbeddingDim = INT32_MAX;
constexpr int64_t kMaxVocabSize = INT64_MAX;
constexpr int32_t kMaxSuperpositionDim = INT32_MAX;     // MoE superposition dimension
constexpr bool kUnlimitedScale = true;
constexpr bool kScaleLimitsEnforced = false;

#else  // LITE Edition
// Lite: Scale limits enforced
// Lite: Scale limits enforced via runtime calculation
// We avoid static constants to confuse hex editors.

// 20B Params: 0x4A817C800
inline int64_t GetLiteMaxParams() { 
    volatile int64_t a = 0x9502F900; 
    volatile int64_t b = 0x5;
    return (a * b) + 0x3800; // 20B
}

// 5M Context: 0x4C4B40
inline int64_t GetLiteMaxContext() { return 0x4C4B40 ^ 0x0; }

inline int32_t GetLiteMaxBlocks() { return 0x18; }
inline int32_t GetLiteMaxExperts() { return 0xC; }
inline int32_t GetLiteMaxEmbed() { return 0x1000; }
inline int64_t GetLiteMaxVocab() { return 0x3E800; }
inline int32_t GetLiteMaxSuperposition() { return 0x4; }

#define kMaxTotalParams        GetLiteMaxParams()
#define kMaxContextLength      GetLiteMaxContext()
#define kMaxReasoningBlocks    GetLiteMaxBlocks()
#define kMaxMoeExperts         GetLiteMaxExperts()
#define kMaxEmbeddingDim       GetLiteMaxEmbed()
#define kMaxVocabSize          GetLiteMaxVocab()
#define kMaxSuperpositionDim   GetLiteMaxSuperposition()

constexpr bool kUnlimitedScale = false;
constexpr bool kScaleLimitsEnforced = true;

#endif

// =============================================================================
// DOMAIN MODULE ACCESS (Enterprise-only modules)
// =============================================================================

#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO
constexpr bool kChemistryModuleUnlocked = true;
constexpr bool kPhysicsModuleUnlocked = true;
constexpr bool kInverseDesignModuleUnlocked = true;
constexpr bool kHardwareControlModuleUnlocked = true;
constexpr bool kGraphLearningModuleUnlocked = true;
#else
constexpr bool kChemistryModuleUnlocked = false;
constexpr bool kPhysicsModuleUnlocked = false;
constexpr bool kInverseDesignModuleUnlocked = false;
constexpr bool kHardwareControlModuleUnlocked = false;
constexpr bool kGraphLearningModuleUnlocked = false;
#endif

// =============================================================================
// CHAIN AUTHENTICATION - Build-time generated secrets
// =============================================================================

// These values are overridden at build time via CMake -DCHAIN_SECRET_HIGH=...
#ifndef CHAIN_SECRET_HIGH
#define CHAIN_SECRET_HIGH 0xA7B3C91D4E5F6A2BULL
#endif

#ifndef CHAIN_SECRET_LOW
#define CHAIN_SECRET_LOW 0x8C7D6E5F4A3B2C1DULL
#endif

// Build version embedded at compile time
constexpr const char* kBuildVersion = __DATE__ " " __TIME__;

// =============================================================================
// VALIDATION MACROS - Edition-aware enforcement
// =============================================================================

#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO
// Pro/Enterprise: Validation macros are no-ops (unlimited scale)
#define SAGUARO_CHECK_LIMIT(context, value, limit, name) ((void)0)
#define SAGUARO_CHECK_REASONING_BLOCKS(ctx, num_blocks) ((void)0)
#define SAGUARO_CHECK_MOE_EXPERTS(ctx, num_experts) ((void)0)
#define SAGUARO_CHECK_CONTEXT_LENGTH(ctx, seq_len) ((void)0)
#define SAGUARO_CHECK_EMBEDDING_DIM(ctx, dim) ((void)0)
#define SAGUARO_CHECK_VOCAB_SIZE(ctx, size) ((void)0)
#define SAGUARO_CHECK_TOTAL_PARAMS(ctx, params) ((void)0)
#define SAGUARO_CHECK_SUPERPOSITION_DIM(ctx, dim) ((void)0)

#else  // LITE Edition: Enforce limits

/// Check a value against a limit, raising InvalidArgument on failure.
#define SAGUARO_CHECK_LIMIT(context, value, limit, name) \
    do { \
        SAGUARO_SECURITY_HEARTBEAT(); \
        const auto _hn_val = static_cast<int64_t>(value); \
        const auto _hn_lim = static_cast<int64_t>(limit); \
        OP_REQUIRES(context, _hn_val <= _hn_lim, \
            tensorflow::errors::InvalidArgument( \
                "[Error 0x1001] System integrity check failed (", _hn_val, \
                " > ", _hn_lim, ").")); \
    } while (false)

/// Check number of reasoning blocks (max 24 for Lite)
#define SAGUARO_CHECK_REASONING_BLOCKS(ctx, num_blocks) \
    SAGUARO_CHECK_LIMIT(ctx, num_blocks, ::saguaro::limits::kMaxReasoningBlocks, \
                   "sanity_check_01")

/// Check number of MoE experts (max 12 for Lite)
#define SAGUARO_CHECK_MOE_EXPERTS(ctx, num_experts) \
    SAGUARO_CHECK_LIMIT(ctx, num_experts, ::saguaro::limits::kMaxMoeExperts, \
                   "sanity_check_02")

/// Check context/sequence length (max 5M for Lite)
#define SAGUARO_CHECK_CONTEXT_LENGTH(ctx, seq_len) \
    SAGUARO_CHECK_LIMIT(ctx, seq_len, ::saguaro::limits::kMaxContextLength, \
                   "sanity_check_03")

/// Check embedding dimension (max 4096 for Lite)
#define SAGUARO_CHECK_EMBEDDING_DIM(ctx, dim) \
    SAGUARO_CHECK_LIMIT(ctx, dim, ::saguaro::limits::kMaxEmbeddingDim, \
                   "sanity_check_04")

/// Check vocabulary size (max 256K for Lite)
#define SAGUARO_CHECK_VOCAB_SIZE(ctx, size) \
    SAGUARO_CHECK_LIMIT(ctx, size, ::saguaro::limits::kMaxVocabSize, \
                   "sanity_check_05")

/// Check total parameter count (max 20B for Lite)
#define SAGUARO_CHECK_TOTAL_PARAMS(ctx, params) \
    SAGUARO_CHECK_LIMIT(ctx, params, ::saguaro::limits::kMaxTotalParams, \
                   "sanity_check_06")

/// Check MoE superposition dimension (max 4 for Lite, upgraded from 2)
#define SAGUARO_CHECK_SUPERPOSITION_DIM(ctx, dim) \
    SAGUARO_CHECK_LIMIT(ctx, dim, ::saguaro::limits::kMaxSuperpositionDim, \
                   "sanity_check_07")

#endif  // SAGUARO_EDITION

// =============================================================================
// BINARY INTEGRITY VALIDATION
// =============================================================================

/// Validate the binary chain authentication.
inline bool ValidateBinaryChain() {
    volatile uint64_t secret_check = CHAIN_SECRET_HIGH ^ CHAIN_SECRET_LOW;
    volatile bool valid = (secret_check != 0) && 
                          ((secret_check & 0xFF) != 0 || (secret_check >> 56) != 0);
    return valid;
}

/// CRC32 calculation for code section validation
inline uint32_t CalculateCRC32(const void* data, size_t length) {
    uint32_t crc = 0xFFFFFFFF;
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    
    for (size_t i = 0; i < length; ++i) {
        crc ^= bytes[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    
    return ~crc;
}

/// Auto-validate chain authentication on binary load.
struct ChainValidator {
    ChainValidator() {
        // Check chain secrets
        if (!ValidateBinaryChain()) {
            ::saguaro::security::MarkCompromised();
        }
        // Check for debuggers (TracerPid)
        ::saguaro::security::CheckForDebuggers();
    }
};

namespace {
    static ChainValidator _chain_validator_instance;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get a human-readable description of current limits.
inline const char* GetLimitsDescription() {
#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO
    return "HighNoon Pro/Enterprise Edition:\n"
           "  - Unlimited Parameters\n"
           "  - Unlimited Context Length\n"
           "  - Unlimited Reasoning Blocks\n"
           "  - Unlimited MoE Experts\n"
           "  - Unlimited Embedding Dimension\n"
           "  - All Domain Modules Unlocked\n";
#else
    return "HighNoon Lite Edition Limits:\n"
           "  - Max Parameters: 20B\n"
           "  - Max Context Length: 5M tokens\n"
           "  - Max Reasoning Blocks: 24\n"
           "  - Max MoE Experts: 12\n"
           "  - Max Embedding Dim: 4096\n"
           "\n"
           "Upgrade to Pro or Enterprise for unlimited scale:\n"
           "https://versoindustries.com/upgrade";
#endif
}

/// Check if running with unlimited scale (Pro or Enterprise)
inline bool IsUnlimitedEdition() {
    return kUnlimitedScale;
}

/// Check if Lite edition (scale limits enforced)
inline bool IsLiteEdition() {
    return SAGUARO_EDITION == SAGUARO_EDITION_LITE;
}

/// Check if Pro edition
inline bool IsProEdition() {
    return SAGUARO_EDITION == SAGUARO_EDITION_PRO;
}

/// Check if Enterprise edition
inline bool IsEnterpriseEdition() {
    return SAGUARO_EDITION == SAGUARO_EDITION_ENTERPRISE;
}

/// Get edition string
inline const char* GetEdition() {
    return kEdition;
}

/// Get edition code (0=Lite, 1=Pro, 2=Enterprise)
inline int32_t GetEditionCode() {
    return kEditionCode;
}

}  // namespace limits
}  // namespace saguaro

#endif  // SAGUARO_EDITION_LIMITS_H_
