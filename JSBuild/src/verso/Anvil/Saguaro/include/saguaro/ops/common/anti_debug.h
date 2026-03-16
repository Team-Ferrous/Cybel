// saguaro/native/ops/common/anti_debug.h
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
// ANTI-DEBUGGING MEASURES - Production Builds Only
// =============================================================================
//
// This header implements anti-debugging and anti-tampering measures for
// production builds of HighNoon. These measures are designed to make
// reverse engineering and runtime analysis significantly more difficult.
//
// NOTE: These measures are ONLY enabled when PRODUCTION_BUILD is defined.
// Development builds have full debugging capability.

#ifndef HIGHNOON_ANTI_DEBUG_H_
#define HIGHNOON_ANTI_DEBUG_H_

#ifdef PRODUCTION_BUILD

#include <cstdlib>
#include <cstdint>
#include <cstring>

// Platform-specific includes
#if defined(__linux__)
#include <sys/ptrace.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <fstream>
#include <string>
#elif defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>
#endif

namespace saguaro {
namespace security {

// =============================================================================
// DEBUGGER DETECTION
// =============================================================================

#if defined(__linux__)

/// Detect debugger attachment via ptrace.
/// PTRACE_TRACEME can only succeed once per process - if it fails,
/// another process (debugger) has already attached.
inline bool IsDebuggerAttached_Ptrace() {
    return ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) == -1;
}

/// Check /proc/self/status for TracerPid.
/// A non-zero TracerPid indicates a debugger is attached.
inline bool IsDebuggerAttached_TracerPid() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) return false;
    
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 10, "TracerPid:") == 0) {
            // Extract the PID value
            size_t pos = line.find_first_not_of(" \t", 10);
            if (pos != std::string::npos) {
                int tracer_pid = std::atoi(line.c_str() + pos);
                return tracer_pid != 0;
            }
        }
    }
    return false;
}

#elif defined(__APPLE__)

/// macOS debugger detection via sysctl.
inline bool IsDebuggerAttached_Sysctl() {
    int mib[4] = {CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()};
    struct kinfo_proc info;
    size_t size = sizeof(info);
    
    if (sysctl(mib, 4, &info, &size, nullptr, 0) == -1) {
        return false;
    }
    
    return (info.kp_proc.p_flag & P_TRACED) != 0;
}

#endif

/// Combined debugger detection.
inline bool IsDebuggerPresent() {
#if defined(__linux__)
    return IsDebuggerAttached_Ptrace() || IsDebuggerAttached_TracerPid();
#elif defined(__APPLE__)
    return IsDebuggerAttached_Sysctl();
#else
    return false;  // Unknown platform - assume no debugger
#endif
}

// =============================================================================
// TIMING-BASED DETECTION
// =============================================================================

/// Read CPU timestamp counter for timing measurements.
inline uint64_t ReadTSC() {
#if defined(__x86_64__) || defined(_M_X64)
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return (static_cast<uint64_t>(hi) << 32) | lo;
#elif defined(__aarch64__)
    uint64_t val;
    __asm__ volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#else
    return 0;  // Fallback for unknown architectures
#endif
}

/// Detect breakpoints via timing anomalies.
/// A significant delay in a simple operation suggests a breakpoint was hit.
inline bool DetectBreakpointViaTimimg() {
    constexpr uint64_t kExpectedCycles = 1000;    // Expected cycles for test
    constexpr uint64_t kThresholdMultiplier = 50; // Suspicious if 50x slower
    
    volatile uint64_t start = ReadTSC();
    
    // Simple computation that should be fast
    volatile int result = 0;
    for (int i = 0; i < 100; ++i) {
        result += i * i;
    }
    (void)result;  // Prevent optimization
    
    volatile uint64_t end = ReadTSC();
    uint64_t elapsed = end - start;
    
    return elapsed > (kExpectedCycles * kThresholdMultiplier);
}

// =============================================================================
// ENVIRONMENT DETECTION
// =============================================================================

/// Check for environment variables commonly used by debugging tools.
inline bool DetectDebugEnvironment() {
    const char* suspicious_vars[] = {
        "LD_PRELOAD",               // Library injection
        "DYLD_INSERT_LIBRARIES",    // macOS library injection
        "GDB_PYTHON",               // GDB Python scripting
        "STRACE_GROUPS",            // strace
        "_JAVA_OPTIONS",            // Java debugging
        "MALLOC_CHECK_",            // Memory debugging
        "MALLOC_PERTURB_",          // Memory debugging
        nullptr
    };
    
    for (const char** var = suspicious_vars; *var != nullptr; ++var) {
        if (getenv(*var) != nullptr) {
            return true;
        }
    }
    return false;
}

// =============================================================================
// ANTI-DEBUG INITIALIZATION
// =============================================================================

/// Anti-debug initialization - runs before main().
/// Uses constructor attribute to ensure early execution.
__attribute__((constructor, used))
static void AntiDebugInit() {
    // Method 1: ptrace-based detection
    if (IsDebuggerPresent()) {
        _exit(1);  // Silent exit - no error message
    }
    
    // Method 2: Environment variable detection
    if (DetectDebugEnvironment()) {
        _exit(1);
    }
    
    // Method 3: Timing-based detection (called periodically during execution)
    // This is not called at init to avoid false positives during load
}

/// Periodic anti-debug check - call from critical code paths.
inline void PeriodicAntiDebugCheck() {
#if defined(__linux__)
    // Re-check TracerPid (more reliable than ptrace for ongoing detection)
    if (IsDebuggerAttached_TracerPid()) {
        _exit(1);
    }
#elif defined(__APPLE__)
    if (IsDebuggerAttached_Sysctl()) {
        _exit(1);
    }
#endif
    
    // Timing-based check
    if (DetectBreakpointViaTimimg()) {
        // Don't immediately exit on timing anomaly - could be system load
        // Instead, set a flag for multiple failures
        static int timing_failures = 0;
        if (++timing_failures > 3) {
            _exit(1);
        }
    }
}

// =============================================================================
// SIGNAL HANDLER PROTECTION
// =============================================================================

/// Install signal handlers to detect debugging signals.
__attribute__((constructor))
static void InstallSignalHandlers() {
    // SIGTRAP is sent when a breakpoint is hit
    signal(SIGTRAP, [](int) { _exit(1); });
}

}  // namespace security
}  // namespace saguaro

#else  // !PRODUCTION_BUILD

// Development build - provide no-op implementations
namespace saguaro {
namespace security {

inline bool IsDebuggerPresent() { return false; }
inline bool DetectBreakpointViaTimimg() { return false; }
inline bool DetectDebugEnvironment() { return false; }
inline void PeriodicAntiDebugCheck() {}

}  // namespace security
}  // namespace saguaro

#endif  // PRODUCTION_BUILD

#endif  // HIGHNOON_ANTI_DEBUG_H_
