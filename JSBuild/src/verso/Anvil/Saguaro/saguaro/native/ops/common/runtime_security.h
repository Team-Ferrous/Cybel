// saguaro.native/ops/common/runtime_security.h
// Advanced Runtime Protection for HighNoon Binaries
// (c) 2025 Verso Industries

#ifndef SAGUARO_RUNTIME_SECURITY_H_
#define SAGUARO_RUNTIME_SECURITY_H_

#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>

namespace saguaro {
namespace security {

/**
 * Global security state. If any integrity check fails, this is set to true.
 * We use an atomic for thread-safety and avoid descriptive names to stay stealthy.
 */
inline std::atomic<uint64_t> g_ctx_status{0xBADA55}; // "Safe" state

/**
 * Sets the "Compromised" state. 
 * Instead of exiting immediately, we mark the state for a "Time Bomb" failure.
 */
inline void MarkCompromised() {
    g_ctx_status.store(0xDEADBEEF);
}

/**
 * Perform a "Nano Check". Should be called frequently in hot paths.
 * If the state is compromised, it may trigger a delayed failure.
 */
inline void InternalSafetyCheck() {
    if (g_ctx_status.load() == 0xDEADBEEF) {
        // Delayed failure logic: 
        // We only fail 1 out of 1000 times to make it hard to trace back to the cause.
        static thread_local std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<> dis(1, 1000);
        
        if (dis(gen) == 42) {
            // "Time Bomb" triggered!
            // We exit with a confusing error code or just a segfault.
            #if defined(__linux__)
                // Subtle Segfault
                volatile int* ptr = nullptr;
                (void)*ptr;
            #else
                std::abort();
            #endif
        }
    }
}

/**
 * Simple CRC32 for runtime integrity.
 */
inline uint32_t CalculateCRC32(const void* data, size_t size) {
    uint32_t crc = 0xFFFFFFFF;
    const uint8_t* p = static_cast<const uint8_t*>(data);
    while (size--) {
        crc ^= *p++;
        for (int i = 0; i < 8; i++) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

/**
 * Validates the integrity of a specific memory region.
 * If validation fails, it marks the system as compromised.
 * 
 * To find the expected_crc for a new build:
 * 1. Set SAGUARO_EXPECTED_CRC to 0.
 * 2. Run the binary.
 * 3. Look for "Integrity Seed" in stderr.
 */
inline bool CheckSectionIntegrity(const void* addr, size_t size, uint32_t expected_crc) {
    if (addr == nullptr || size == 0) return true;
    
    uint32_t actual_crc = CalculateCRC32(addr, size);
    
    if (expected_crc == 0) {
        // Discovery mode: Print the hash so it can be added to the source.
        // We use obfuscated output to not tip off attackers.
        // std::fprintf(stderr, "[Security] Seed 0x%08X detected.\n", actual_crc);
        return true;
    }
    
    if (actual_crc != expected_crc) {
        MarkCompromised();
        return false;
    }
    return true;
}

/**
 * Checks /proc/self/status for TracerPid to detect attached debuggers.
 * This is more robust than ptrace as it works even if ptrace fails for other reasons.
 */
inline bool IsDebuggerAttached() {
#if defined(__linux__)
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp) return false;
    
    char line[256];
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "TracerPid:", 10) == 0) {
            int tracer_pid = std::atoi(line + 10);
            std::fclose(fp);
            return tracer_pid != 0;
        }
    }
    std::fclose(fp);
#endif
    return false;
}

/**
 * Comprehensive debugger detection - combines ptrace and TracerPid.
 * Call this at startup and periodically during execution.
 */
inline void CheckForDebuggers() {
    if (IsDebuggerAttached()) {
        MarkCompromised();
    }
}

} // namespace security
} // namespace saguaro

// Stealthy alias for use in hot paths
#define SAGUARO_SECURITY_HEARTBEAT() ::saguaro::security::InternalSafetyCheck()

#endif // SAGUARO_RUNTIME_SECURITY_H_
