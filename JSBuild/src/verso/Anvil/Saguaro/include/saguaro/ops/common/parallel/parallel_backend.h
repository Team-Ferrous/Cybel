#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(SAGUARO_WITH_TBB) && SAGUARO_WITH_TBB
#include <tbb/blocked_range.h>
#include <tbb/blocked_range2d.h>
#include <tbb/parallel_for.h>
#endif

#if defined(SAGUARO_WITH_OPENMP) && SAGUARO_WITH_OPENMP
#include <omp.h>
#endif

namespace hsmn::parallel {

enum class Backend { kTBB, kOpenMP, kStd };

inline const char* BackendName(Backend backend) {
    switch (backend) {
        case Backend::kTBB:
            return "tbb";
        case Backend::kOpenMP:
            return "openmp";
        case Backend::kStd:
            return "serial";
    }
    return "unknown";
}

namespace detail {

inline std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

inline std::string_view Trim(std::string_view view) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    while (!view.empty() && !not_space(static_cast<unsigned char>(view.front()))) {
        view.remove_prefix(1);
    }
    while (!view.empty() && !not_space(static_cast<unsigned char>(view.back()))) {
        view.remove_suffix(1);
    }
    return view;
}

inline Backend BackendFromString(std::string_view raw_value, bool* matched = nullptr) {
    auto lower = ToLower(std::string(Trim(raw_value)));
    if (matched) {
        *matched = true;
    }
    if (lower == "tbb") {
        return Backend::kTBB;
    }
    if (lower == "openmp" || lower == "omp") {
        return Backend::kOpenMP;
    }
    if (lower == "std" || lower == "serial" || lower == "single") {
        return Backend::kStd;
    }
    if (matched) {
        *matched = false;
    }
    return Backend::kStd;
}

inline std::string DetectCpuVendorFromEnv() {
    if (const char* env = std::getenv("SAGUARO_CPU_VENDOR")) {
        return ToLower(env);
    }
    return {};
}

#if defined(__x86_64__) || defined(__i386__)
inline std::string DetectCpuVendorFromCpuid() {
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) == 0) {
        return {};
    }
    char vendor[13] = {};
    std::memcpy(vendor + 0, &ebx, sizeof(ebx));
    std::memcpy(vendor + 4, &edx, sizeof(edx));
    std::memcpy(vendor + 8, &ecx, sizeof(ecx));
    vendor[12] = '\0';
    return ToLower(vendor);
}
#endif

inline std::string DetectCpuVendor() {
    auto env_vendor = DetectCpuVendorFromEnv();
    if (!env_vendor.empty()) {
        return env_vendor;
    }
#if defined(__x86_64__) || defined(__i386__)
    auto cpuid_vendor = DetectCpuVendorFromCpuid();
    if (!cpuid_vendor.empty()) {
        return cpuid_vendor;
    }
#elif defined(__aarch64__) || defined(__arm__)
    return "arm";
#endif
    return {};
}

inline Backend AutoBackendSelection() {
    const auto vendor = DetectCpuVendor();
    if (vendor.find("authenticamd") != std::string::npos ||
        vendor.find("amd") != std::string::npos ||
        vendor.find("arm") != std::string::npos ||
        vendor.find("apple") != std::string::npos) {
        // Default to OpenMP for AMD/ARM/Apple targets to avoid the regressions
        // called out in uxlfoundation/oneTBB#1871 (AMD Ryzen CPU saturation)
        // and uxlfoundation/oneTBB#1772 (Apple Silicon builds broken by -mrtm/-mwaitpkg).
        return Backend::kOpenMP;
    }
    if (vendor.find("intel") != std::string::npos ||
        vendor.find("genuineintel") != std::string::npos) {
        return Backend::kTBB;
    }
#if defined(__aarch64__) || defined(__arm__)
    return Backend::kOpenMP;
#endif
    return Backend::kStd;
}

constexpr std::size_t CeilDiv(std::size_t value, std::size_t div) {
    return div == 0 ? 0 : (value + div - 1) / div;
}

inline std::size_t SafeMultiply(std::size_t lhs, std::size_t rhs) {
    if (lhs == 0 || rhs == 0) {
        return 0;
    }
    if (lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        return std::numeric_limits<std::size_t>::max();
    }
    return lhs * rhs;
}

inline std::size_t ReadMaxParallelismOverride() {
    if (const char* env = std::getenv("SAGUARO_PAR_MAX_THREADS")) {
        char* end = nullptr;
        long value = std::strtol(env, &end, 10);
        if (end != env && value > 0) {
            return static_cast<std::size_t>(value);
        }
    }
    return 0;
}

inline std::size_t EstimateMaxParallelism() {
    if (const auto override_value = ReadMaxParallelismOverride(); override_value > 0) {
        return override_value;
    }
    auto hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) {
        hw_threads = 8;
    }
    return static_cast<std::size_t>(hw_threads);
}

inline std::size_t ComputeShardBlockSize(std::size_t total, std::size_t cost_per_unit) {
    if (total == 0) {
        return 0;
    }
    constexpr std::size_t kMinCostPerShard = 10000;
    const std::size_t cost = std::max<std::size_t>(1, cost_per_unit);
    const std::size_t max_parallelism = std::max<std::size_t>(1, EstimateMaxParallelism());
    const std::size_t total_cost = SafeMultiply(total, cost);
    const std::size_t shard_target = std::max<std::size_t>(
        1, std::min<std::size_t>(max_parallelism, total_cost / kMinCostPerShard));
    const std::size_t block_size = CeilDiv(total, shard_target);
    return std::max<std::size_t>(1, block_size);
}

#if defined(SAGUARO_WITH_OPENMP) && SAGUARO_WITH_OPENMP && defined(_OPENMP)
#define SAGUARO_PARALLEL_HAS_OPENMP 1
#else
#define SAGUARO_PARALLEL_HAS_OPENMP 0
#endif

#if defined(SAGUARO_WITH_TBB) && SAGUARO_WITH_TBB
#define SAGUARO_PARALLEL_HAS_TBB 1
#else
#define SAGUARO_PARALLEL_HAS_TBB 0
#endif

inline bool BackendAvailable(Backend backend) {
    switch (backend) {
        case Backend::kTBB:
#if SAGUARO_PARALLEL_HAS_TBB
            return true;
#else
            return false;
#endif
        case Backend::kOpenMP:
#if SAGUARO_PARALLEL_HAS_OPENMP
            return true;
#else
            return false;
#endif
        case Backend::kStd:
        default:
            return true;
    }
}

inline Backend FallbackBackend(Backend requested) {
#if SAGUARO_PARALLEL_HAS_OPENMP
    if (requested != Backend::kOpenMP) {
        return Backend::kOpenMP;
    }
#endif
#if SAGUARO_PARALLEL_HAS_TBB
    if (requested != Backend::kTBB) {
        return Backend::kTBB;
    }
#endif
    return Backend::kStd;
}

inline void WarnUnknownBackendValue(std::string_view backend_value) {
    if (backend_value.empty()) {
        return;
    }
    std::fprintf(stderr,
                 "[HSMN][parallel] Ignoring unknown backend value '%.*s'; falling back to auto detection.\n",
                 static_cast<int>(backend_value.size()), backend_value.data());
}

inline void WarnBackendFallback(std::string_view requested_value, Backend fallback) {
    std::fprintf(stderr,
                 "[HSMN][parallel] Backend '%.*s' unavailable in this build. Falling back to %s.\n",
                 static_cast<int>(requested_value.size()), requested_value.data(), BackendName(fallback));
}

inline Backend NormalizeBackend(Backend requested, bool warn_if_missing,
                                std::string_view requested_value = {}) {
    if (BackendAvailable(requested)) {
        return requested;
    }
    const Backend fallback = FallbackBackend(requested);
    if (warn_if_missing) {
        WarnBackendFallback(requested_value.empty() ? std::string_view(BackendName(requested))
                                                    : requested_value,
                            fallback);
    }
    return fallback;
}

inline int DesiredOpenMPThreadCount() {
#if SAGUARO_PARALLEL_HAS_OPENMP
    static const int thread_count = []() {
        const auto limit = std::max<std::size_t>(std::size_t{1}, EstimateMaxParallelism());
        const auto clamped = std::min<std::size_t>(
            limit, static_cast<std::size_t>(std::numeric_limits<int>::max()));
        return static_cast<int>(clamped);
    }();
    return thread_count;
#else
    return 1;
#endif
}

template <typename Func>
inline void ExecuteSerial1D(std::size_t begin, std::size_t end, std::size_t grain, Func&& fn) {
    const std::size_t chunk = grain == 0 ? 1 : grain;
    for (std::size_t block_begin = begin; block_begin < end; block_begin += chunk) {
        const std::size_t block_end = std::min(end, block_begin + chunk);
        fn(block_begin, block_end);
    }
}

#if SAGUARO_PARALLEL_HAS_OPENMP
template <typename Func>
inline void ExecuteOpenMP1D(std::size_t begin, std::size_t end, std::size_t grain, Func&& fn) {
    if (begin >= end) {
        return;
    }
    const std::size_t chunk = grain == 0 ? 1 : grain;
    const std::size_t total_blocks = CeilDiv(end - begin, chunk);
    const int thread_count = DesiredOpenMPThreadCount();
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::int64_t block_idx = 0; block_idx < static_cast<std::int64_t>(total_blocks); ++block_idx) {
        const std::size_t block_begin = begin + static_cast<std::size_t>(block_idx) * chunk;
        const std::size_t block_end = std::min(end, block_begin + chunk);
        fn(block_begin, block_end);
    }
}

template <typename Func>
inline void ExecuteOpenMP2D(std::size_t r_begin, std::size_t r_end, std::size_t r_grain,
                            std::size_t c_begin, std::size_t c_end, std::size_t c_grain,
                            Func&& fn) {
    if (r_begin >= r_end || c_begin >= c_end) {
        return;
    }
    const std::size_t row_chunk = r_grain == 0 ? 1 : r_grain;
    const std::size_t col_chunk = c_grain == 0 ? 1 : c_grain;
    const std::size_t row_blocks = CeilDiv(r_end - r_begin, row_chunk);
    const std::size_t col_blocks = CeilDiv(c_end - c_begin, col_chunk);
    const int thread_count = DesiredOpenMPThreadCount();
#pragma omp parallel for collapse(2) schedule(static) num_threads(thread_count)
    for (std::int64_t rb = 0; rb < static_cast<std::int64_t>(row_blocks); ++rb) {
        for (std::int64_t cb = 0; cb < static_cast<std::int64_t>(col_blocks); ++cb) {
            const std::size_t block_r_begin = r_begin + static_cast<std::size_t>(rb) * row_chunk;
            const std::size_t block_r_end = std::min(r_end, block_r_begin + row_chunk);
            const std::size_t block_c_begin = c_begin + static_cast<std::size_t>(cb) * col_chunk;
            const std::size_t block_c_end = std::min(c_end, block_c_begin + col_chunk);
            fn(block_r_begin, block_r_end, block_c_begin, block_c_end);
        }
    }
}
#endif

template <typename Func>
inline void ExecuteSerial2D(std::size_t r_begin, std::size_t r_end, std::size_t r_grain,
                            std::size_t c_begin, std::size_t c_end, std::size_t c_grain,
                            Func&& fn) {
    const std::size_t row_chunk = r_grain == 0 ? 1 : r_grain;
    const std::size_t col_chunk = c_grain == 0 ? 1 : c_grain;
    for (std::size_t row = r_begin; row < r_end; row += row_chunk) {
        const auto row_end = std::min(row + row_chunk, r_end);
        for (std::size_t col = c_begin; col < c_end; col += col_chunk) {
            const auto col_end = std::min(col + col_chunk, c_end);
            fn(row, row_end, col, col_end);
        }
    }
}

}  // namespace detail

inline Backend DetermineBackend() {
    if (const char* env = std::getenv("SAGUARO_PAR_BACKEND")) {
        bool matched = false;
        const auto parsed = detail::BackendFromString(env, &matched);
        if (matched) {
            return detail::NormalizeBackend(parsed, /*warn_if_missing=*/true, env);
        }
        detail::WarnUnknownBackendValue(env);
    }
    return detail::NormalizeBackend(detail::AutoBackendSelection(), /*warn_if_missing=*/false);
}

inline Backend GetBackend() {
    static Backend backend = DetermineBackend();
    return backend;
}

class SpinMutex {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#else
            std::this_thread::yield();
#endif
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

class SpinLockGuard {
public:
    explicit SpinLockGuard(SpinMutex& mutex) : mutex_(mutex) {
        mutex_.lock();
    }

    ~SpinLockGuard() {
        mutex_.unlock();
    }

    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;

private:
    SpinMutex& mutex_;
};

template <typename Func>
inline void ForRange(std::size_t begin, std::size_t end, std::size_t grain, Func&& fn) {
    if (begin >= end) {
        return;
    }
    switch (GetBackend()) {
        case Backend::kTBB:
#if SAGUARO_PARALLEL_HAS_TBB
            tbb::parallel_for(
                tbb::blocked_range<std::size_t>(begin, end, grain == 0 ? 1 : grain),
                [&](const tbb::blocked_range<std::size_t>& range) {
                    fn(range.begin(), range.end());
                });
            break;
#else
            detail::ExecuteSerial1D(begin, end, grain, std::forward<Func>(fn));
            break;
#endif
        case Backend::kOpenMP:
#if SAGUARO_PARALLEL_HAS_OPENMP
            detail::ExecuteOpenMP1D(begin, end, grain, std::forward<Func>(fn));
            break;
#else
            detail::ExecuteSerial1D(begin, end, grain, std::forward<Func>(fn));
            break;
#endif
        case Backend::kStd:
        default:
            detail::ExecuteSerial1D(begin, end, grain, std::forward<Func>(fn));
            break;
    }
}

template <typename Func>
inline void ForShard(std::size_t total, std::size_t cost_per_unit, Func&& fn) {
    if (total == 0) {
        return;
    }
    const std::size_t block_size = detail::ComputeShardBlockSize(total, cost_per_unit);
    ForRange(0, total, block_size, [&](std::size_t begin, std::size_t end) {
        if constexpr (std::is_invocable_v<Func&, std::size_t, std::size_t>) {
            fn(begin, end);
        } else if constexpr (std::is_invocable_v<Func&, std::int64_t, std::int64_t>) {
            fn(static_cast<std::int64_t>(begin), static_cast<std::int64_t>(end));
        } else {
            fn(begin, end);
        }
    });
}

template <typename Func>
inline void ForRange2D(std::size_t r_begin, std::size_t r_end, std::size_t r_grain,
                       std::size_t c_begin, std::size_t c_end, std::size_t c_grain,
                       Func&& fn) {
    if (r_begin >= r_end || c_begin >= c_end) {
        return;
    }
    switch (GetBackend()) {
        case Backend::kTBB:
#if SAGUARO_PARALLEL_HAS_TBB
            tbb::parallel_for(
                tbb::blocked_range2d<std::size_t>(r_begin, r_end, r_grain == 0 ? 1 : r_grain,
                                                  c_begin, c_end, c_grain == 0 ? 1 : c_grain),
                [&](const tbb::blocked_range2d<std::size_t>& range) {
                    fn(range.rows().begin(), range.rows().end(),
                       range.cols().begin(), range.cols().end());
                });
            break;
#else
            detail::ExecuteSerial2D(r_begin, r_end, r_grain, c_begin, c_end, c_grain,
                                    std::forward<Func>(fn));
            break;
#endif
        case Backend::kOpenMP:
#if SAGUARO_PARALLEL_HAS_OPENMP
            detail::ExecuteOpenMP2D(r_begin, r_end, r_grain, c_begin, c_end, c_grain,
                                    std::forward<Func>(fn));
            break;
#else
            detail::ExecuteSerial2D(r_begin, r_end, r_grain, c_begin, c_end, c_grain,
                                    std::forward<Func>(fn));
            break;
#endif
        case Backend::kStd:
        default:
            detail::ExecuteSerial2D(r_begin, r_end, r_grain, c_begin, c_end, c_grain,
                                    std::forward<Func>(fn));
            break;
    }
}

template <typename Func>
inline void ForEachIndex(std::size_t count, std::size_t grain, Func&& fn) {
    if (count == 0) {
        return;
    }
    ForRange(0, count, grain, [&](std::size_t begin, std::size_t end) {
        for (std::size_t idx = begin; idx < end; ++idx) {
            fn(idx);
        }
    });
}

}  // namespace hsmn::parallel
