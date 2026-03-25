// saguaro.native/ops/common/encrypted_strings.h
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
// COMPILE-TIME STRING ENCRYPTION
// =============================================================================
//
// This header provides compile-time string encryption to prevent sensitive
// strings (error messages, limit values, URLs) from appearing in plaintext
// in the binary. Strings are XOR-encrypted at compile time and decrypted
// at runtime only when needed.
//
// Usage:
//   std::string msg = SAGUARO_DECRYPT("Secret message");
//   throw std::runtime_error(SAGUARO_DECRYPT("Error: limit exceeded"));

#ifndef SAGUARO_ENCRYPTED_STRINGS_H_
#define SAGUARO_ENCRYPTED_STRINGS_H_

#include <array>
#include <string>
#include <cstdint>

namespace saguaro {
namespace crypto {

// =============================================================================
// ENCRYPTION KEY
// =============================================================================

// XOR encryption key - rotated per release build
// This key is embedded at compile time and can be changed via CMake
#ifndef SAGUARO_CRYPTO_KEY
#define SAGUARO_CRYPTO_KEY "V3RS0_1NDUSTR13S_H1GHN00N_2025_K3Y"
#endif

constexpr char kEncryptionKey[] = SAGUARO_CRYPTO_KEY;
constexpr size_t kKeyLength = sizeof(kEncryptionKey) - 1;

// =============================================================================
// COMPILE-TIME ENCRYPTION
// =============================================================================

/// Compile-time encrypted string container.
/// The string is XOR-encrypted at compile time and stored in the binary.
template<size_t N>
struct EncryptedString {
    std::array<char, N> data;
    
    /// Compile-time constructor - encrypts the string.
    constexpr EncryptedString(const char (&str)[N]) : data{} {
        for (size_t i = 0; i < N; ++i) {
            data[i] = str[i] ^ kEncryptionKey[i % kKeyLength];
        }
    }
    
    /// Runtime decryption - returns the original string.
    std::string decrypt() const {
        std::string result;
        result.reserve(N - 1);  // Exclude null terminator
        
        for (size_t i = 0; i < N - 1; ++i) {
            result.push_back(data[i] ^ kEncryptionKey[i % kKeyLength]);
        }
        
        return result;
    }
    
    /// Get encrypted data pointer (for advanced usage).
    const char* encrypted_data() const {
        return data.data();
    }
    
    /// Get size of encrypted data.
    constexpr size_t size() const {
        return N;
    }
};

/// Helper to create encrypted string at compile time.
template<size_t N>
constexpr EncryptedString<N> MakeEncrypted(const char (&str)[N]) {
    return EncryptedString<N>(str);
}

// =============================================================================
// DECRYPTION MACRO
// =============================================================================

/// Decrypt an encrypted string literal at runtime.
/// This macro creates a compile-time encrypted string and decrypts it.
/// 
/// Example:
///   std::string msg = SAGUARO_DECRYPT("This is encrypted in the binary");
///
#define SAGUARO_DECRYPT(str) \
    ([]() -> std::string { \
        constexpr auto encrypted = ::saguaro::crypto::MakeEncrypted(str); \
        return encrypted.decrypt(); \
    }())

/// Decrypt to C-style string (caller must manage lifetime).
/// Returns a temporary - use immediately or copy.
#define SAGUARO_DECRYPT_CSTR(str) (SAGUARO_DECRYPT(str).c_str())

// =============================================================================
// PRE-ENCRYPTED COMMON MESSAGES
// =============================================================================

/// Common error messages - pre-encrypted for consistency.
namespace messages {

inline std::string LimitExceeded() {
    return SAGUARO_DECRYPT("HighNoon Lite: Scale limit exceeded");
}

inline std::string UpgradeToEnterprise() {
    return SAGUARO_DECRYPT("Upgrade to Enterprise Edition for unlimited scale");
}

inline std::string EnterpriseUrl() {
    return SAGUARO_DECRYPT("https://versoindustries.com/enterprise");
}

inline std::string ContactSales() {
    return SAGUARO_DECRYPT("Contact: sales@versoindustries.com");
}

inline std::string BinaryIntegrityFailed() {
    return SAGUARO_DECRYPT("Binary integrity check failed");
}

inline std::string LicenseRequired() {
    return SAGUARO_DECRYPT("Enterprise license required for this feature");
}

inline std::string MaxReasoningBlocks() {
    return SAGUARO_DECRYPT("Maximum reasoning blocks exceeded (limit: 24)");
}

inline std::string MaxMoeExperts() {
    return SAGUARO_DECRYPT("Maximum MoE experts exceeded (limit: 12)");
}

inline std::string MaxContextLength() {
    return SAGUARO_DECRYPT("Maximum context length exceeded (limit: 5M tokens)");
}

inline std::string MaxEmbeddingDim() {
    return SAGUARO_DECRYPT("Maximum embedding dimension exceeded (limit: 4096)");
}

}  // namespace messages

// =============================================================================
// OBFUSCATED NUMBER ENCODING
// =============================================================================

/// Obfuscate a compile-time constant number.
/// Makes numeric limits harder to find via binary search.
template<int64_t Value>
struct ObfuscatedNumber {
    // Store the number XOR'd with a magic constant
    static constexpr int64_t kMagic = 0x5A5A5A5A5A5A5A5AULL;
    static constexpr int64_t kObfuscated = Value ^ kMagic;
    
    /// Get the original value at runtime.
    static int64_t Get() {
        volatile int64_t obf = kObfuscated;
        volatile int64_t magic = kMagic;
        return obf ^ magic;
    }
};

/// Macro for obfuscated limit access.
#define SAGUARO_OBFUSCATED_LIMIT(value) \
    (::saguaro::crypto::ObfuscatedNumber<value>::Get())

}  // namespace crypto
}  // namespace saguaro

#endif  // SAGUARO_ENCRYPTED_STRINGS_H_
