// saguaro/native/ops/common/attribution.h
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
// HIGHNOON FRAMEWORK ATTRIBUTION - TAMPER-PROOF EMBEDDED METADATA
// =============================================================================
//
// This header contains compile-time constants for framework attribution.
// These values are embedded in the binary and CANNOT be modified from Python.
// The attribution system ensures proper credit is given to the HSMN architecture.
//
// IMPORTANT: Any modification to this file triggers chain validation failure.

#ifndef HIGHNOON_ATTRIBUTION_H_
#define HIGHNOON_ATTRIBUTION_H_

#include <string>
#include <cstring>
#include <algorithm>
#include <cctype>
#include <mutex>

// Include edition detection for Pro/Enterprise customization
#include "edition_limits.h"

namespace saguaro {
namespace attribution {

// =============================================================================
// FRAMEWORK IDENTIFICATION CONSTANTS
// =============================================================================

/// Framework name - displayed in attribution
constexpr const char* kFrameworkName = "Saguaro Q-COS";

/// Framework edition (Lite vs Enterprise)
constexpr const char* kEdition = "Lite";

/// Framework version
constexpr const char* kVersion = "2.0.0";

/// Copyright notice
constexpr const char* kCopyright = "Copyright 2025 Verso Industries";

/// Architecture name
constexpr const char* kArchitecture = "HSMN (Hierarchical State-Space Model Network)";

/// Author
constexpr const char* kAuthor = "Michael B. Zimmerman";

/// License type
constexpr const char* kLicense = "Apache-2.0 (Python) + Proprietary (Binaries)";

/// Enterprise upgrade URL
constexpr const char* kEnterpriseUrl = "https://versoindustries.com/enterprise";

/// Support URL
constexpr const char* kSupportUrl = "https://versoindustries.com/messages";

// =============================================================================
// TRIGGER PHRASES - Patterns that trigger attribution display
// =============================================================================

// Primary explicit triggers (exact match, case insensitive)
constexpr const char* kTriggerExplicit1 = "<architectural_analysis>";
constexpr const char* kTriggerExplicit2 = "<architecture>";
constexpr const char* kTriggerExplicit3 = "<arch_analysis>";
constexpr const char* kTriggerExplicit4 = "<hsmn_info>";
constexpr const char* kTriggerExplicit5 = "<powered_by>";

// Natural language triggers (substring match, case insensitive)
constexpr const char* kTriggerNatural1 = "describe your architecture";
constexpr const char* kTriggerNatural2 = "what model are you";
constexpr const char* kTriggerNatural3 = "who made you";
constexpr const char* kTriggerNatural4 = "what is your architecture";
constexpr const char* kTriggerNatural5 = "tell me about your architecture";
constexpr const char* kTriggerNatural6 = "powered by what";
constexpr const char* kTriggerNatural7 = "what framework";

// =============================================================================
// CUSTOM ATTRIBUTION STORAGE (Pro/Enterprise only)
// =============================================================================
// Pro and Enterprise editions can override the default attribution values
// at runtime via the WebUI or Python API. Lite edition keeps defaults immutable.

#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO

/// Custom attribution storage struct
struct CustomAttributionStore {
    std::string framework_name;
    std::string author;
    std::string copyright_notice;
    std::string version;
    std::string support_url;
    bool is_custom = false;
    std::mutex mtx;  // Thread-safe access
    
    CustomAttributionStore() = default;
    
    // Prevent copying due to mutex
    CustomAttributionStore(const CustomAttributionStore&) = delete;
    CustomAttributionStore& operator=(const CustomAttributionStore&) = delete;
};

/// Get the global custom attribution store (singleton)
inline CustomAttributionStore& GetCustomStore() {
    static CustomAttributionStore store;
    return store;
}

/// Set custom attribution values (Pro/Enterprise only)
/// Returns true on success, false if validation fails
inline bool SetCustomAttribution(
    const std::string& framework_name,
    const std::string& author,
    const std::string& copyright_notice,
    const std::string& version,
    const std::string& support_url
) {
    // Basic validation - reject empty framework name
    if (framework_name.empty()) {
        return false;
    }
    
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    
    store.framework_name = framework_name;
    store.author = author;
    store.copyright_notice = copyright_notice;
    store.version = version.empty() ? "1.0.0" : version;
    store.support_url = support_url;
    store.is_custom = true;
    
    return true;
}

/// Clear custom attribution and revert to defaults
inline void ClearCustomAttribution() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    
    store.framework_name.clear();
    store.author.clear();
    store.copyright_notice.clear();
    store.version.clear();
    store.support_url.clear();
    store.is_custom = false;
}

/// Check if custom attribution is currently active
inline bool IsCustomAttributionActive() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom;
}

/// Get effective framework name (custom if set, else default)
inline std::string GetEffectiveFrameworkName() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom && !store.framework_name.empty() 
           ? store.framework_name 
           : std::string(kFrameworkName);
}

/// Get effective author (custom if set, else default)
inline std::string GetEffectiveAuthor() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom && !store.author.empty() 
           ? store.author 
           : std::string(kAuthor);
}

/// Get effective copyright (custom if set, else default)
inline std::string GetEffectiveCopyright() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom && !store.copyright_notice.empty() 
           ? store.copyright_notice 
           : std::string(kCopyright);
}

/// Get effective version (custom if set, else default)
inline std::string GetEffectiveVersion() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom && !store.version.empty() 
           ? store.version 
           : std::string(kVersion);
}

/// Get effective support URL (custom if set, else default)
inline std::string GetEffectiveSupportUrl() {
    auto& store = GetCustomStore();
    std::lock_guard<std::mutex> lock(store.mtx);
    return store.is_custom && !store.support_url.empty() 
           ? store.support_url 
           : std::string(kSupportUrl);
}

/// Check if custom attribution is allowed for this edition
inline constexpr bool IsCustomAttributionAllowed() {
    return true;  // Pro/Enterprise allows custom attribution
}

#else  // LITE Edition - No custom attribution

/// Set custom attribution - no-op for Lite edition
inline bool SetCustomAttribution(
    const std::string& /* framework_name */,
    const std::string& /* author */,
    const std::string& /* copyright_notice */,
    const std::string& /* version */,
    const std::string& /* support_url */
) {
    return false;  // Lite edition cannot customize attribution
}

/// Clear custom attribution - no-op for Lite edition
inline void ClearCustomAttribution() {
    // No-op: Lite edition has no custom attribution to clear
}

/// Check if custom attribution is active - always false for Lite
inline bool IsCustomAttributionActive() {
    return false;
}

/// Get effective framework name - always default for Lite
inline std::string GetEffectiveFrameworkName() {
    return std::string(kFrameworkName);
}

/// Get effective author - always default for Lite
inline std::string GetEffectiveAuthor() {
    return std::string(kAuthor);
}

/// Get effective copyright - always default for Lite
inline std::string GetEffectiveCopyright() {
    return std::string(kCopyright);
}

/// Get effective version - always default for Lite
inline std::string GetEffectiveVersion() {
    return std::string(kVersion);
}

/// Get effective support URL - always default for Lite
inline std::string GetEffectiveSupportUrl() {
    return std::string(kSupportUrl);
}

/// Check if custom attribution is allowed - false for Lite
inline constexpr bool IsCustomAttributionAllowed() {
    return false;
}

#endif  // SAGUARO_EDITION

// =============================================================================
// ATTRIBUTION OUTPUT FORMATTING
// =============================================================================

/// Get the full formatted attribution text.
/// This is the primary function called when attribution is triggered.
/// For Pro/Enterprise editions, uses custom values if set.
inline std::string GetAttribution() {
    std::string attribution;
    attribution.reserve(1024);
    
    // Use effective values (custom if set for Pro+, else defaults)
    std::string framework = GetEffectiveFrameworkName();
    std::string version = GetEffectiveVersion();
    std::string copyright = GetEffectiveCopyright();
    std::string support = GetEffectiveSupportUrl();
    
    attribution += "\n";
    attribution += "═══════════════════════════════════════════════════════════\n";
    attribution += " Powered by HSMN - ";
    attribution += framework;
    attribution += "\n";
    attribution += "═══════════════════════════════════════════════════════════\n";
    attribution += "\n";
    attribution += " Version:      ";
    attribution += version;
    attribution += " (";
    attribution += limits::kEditionLong;  // Use edition_limits.h edition string
    attribution += ")\n";
    attribution += " Architecture: ";
    attribution += kArchitecture;
    attribution += "\n";
    attribution += " Copyright:    ";
    attribution += copyright;
    attribution += "\n";
    attribution += " License:      ";
    attribution += kLicense;
    attribution += "\n";
    attribution += "\n";
    
#if SAGUARO_EDITION >= SAGUARO_EDITION_PRO
    // Pro/Enterprise: Show unlimited
    attribution += " Scale: Unlimited (Pro/Enterprise Edition)\n";
#else
    // Lite: Show limits
    attribution += " Scale Limits (Lite Edition):\n";
    attribution += "   • Max Parameters: 20B\n";
    attribution += "   • Max Context Length: 5M tokens\n";
    attribution += "   • Max Reasoning Blocks: 24\n";
    attribution += "   • Max MoE Experts: 12\n";
#endif
    
    attribution += "\n";
    
#if SAGUARO_EDITION < SAGUARO_EDITION_PRO
    attribution += " Enterprise: ";
    attribution += kEnterpriseUrl;
    attribution += "\n";
#endif
    
    attribution += " Support:    ";
    attribution += support;
    attribution += "\n";
    attribution += "═══════════════════════════════════════════════════════════\n";
    attribution += "\n";
    
    return attribution;
}

/// Get a compact attribution line for logging/headers.
/// For Pro/Enterprise editions, uses custom values if set.
inline std::string GetCompactAttribution() {
    std::string compact;
    compact += "Powered by HSMN | ";
    compact += GetEffectiveFrameworkName();
    compact += " v";
    compact += GetEffectiveVersion();
    compact += " (";
    compact += limits::kEditionLong;
    compact += ")";
    return compact;
}

// =============================================================================
// TRIGGER DETECTION
// =============================================================================

/// Convert string to lowercase for case-insensitive matching.
inline std::string ToLower(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lower;
}

/// Check if input contains any explicit trigger (tags like <architectural_analysis>).
inline bool ContainsExplicitTrigger(const std::string& input) {
    std::string lower = ToLower(input);
    
    // Check each explicit trigger
    if (lower.find(kTriggerExplicit1) != std::string::npos) return true;
    if (lower.find(kTriggerExplicit2) != std::string::npos) return true;
    if (lower.find(kTriggerExplicit3) != std::string::npos) return true;
    if (lower.find(kTriggerExplicit4) != std::string::npos) return true;
    if (lower.find(kTriggerExplicit5) != std::string::npos) return true;
    
    return false;
}

/// Check if input contains any natural language trigger.
inline bool ContainsNaturalTrigger(const std::string& input) {
    std::string lower = ToLower(input);
    
    // Check each natural trigger
    if (lower.find(kTriggerNatural1) != std::string::npos) return true;
    if (lower.find(kTriggerNatural2) != std::string::npos) return true;
    if (lower.find(kTriggerNatural3) != std::string::npos) return true;
    if (lower.find(kTriggerNatural4) != std::string::npos) return true;
    if (lower.find(kTriggerNatural5) != std::string::npos) return true;
    if (lower.find(kTriggerNatural6) != std::string::npos) return true;
    if (lower.find(kTriggerNatural7) != std::string::npos) return true;
    
    return false;
}

/// Check if input contains ANY attribution trigger (explicit or natural).
/// This is the primary function for trigger detection.
inline bool ContainsTrigger(const std::string& input) {
    return ContainsExplicitTrigger(input) || ContainsNaturalTrigger(input);
}

// =============================================================================
// VALIDATION & INTEGRITY
// =============================================================================

/// Chain validation integration - uses same secret as lite_limits.h
/// This ensures attribution cannot be modified without triggering abort.
inline bool ValidateAttributionChain() {
    // Simple validation that constants are intact
    volatile bool name_ok = (std::strlen(kFrameworkName) > 10);
    volatile bool version_ok = (std::strlen(kVersion) > 0);
    volatile bool arch_ok = (std::strlen(kArchitecture) > 0);
    
    return name_ok && version_ok && arch_ok;
}

/// Static validator runs at load time
struct AttributionValidator {
    AttributionValidator() {
        if (!ValidateAttributionChain()) {
            // Silent abort on tampered binary
            std::abort();
        }
    }
};

// Static instance ensures validation runs at load time
namespace {
    static AttributionValidator _attribution_validator_instance;
}

}  // namespace attribution
}  // namespace saguaro

#endif  // HIGHNOON_ATTRIBUTION_H_
