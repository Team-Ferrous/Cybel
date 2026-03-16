#include "amx_kernels.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__)
#include <cpuid.h>
#include <immintrin.h>
#endif

#if defined(__linux__) && defined(__x86_64__)
#include <asm/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace anvil::native {

namespace {

#if defined(__x86_64__)

constexpr int kAmxTileRows = 16;
constexpr int kAmxTileCols = 16;
constexpr int kAmxTileDepth = 16;

struct alignas(64) AmxTileConfig {
    std::uint8_t palette_id = 1;
    std::uint8_t start_row = 0;
    std::uint8_t reserved[14] = {};
    std::uint16_t colsb[8] = {};
    std::uint8_t rows[8] = {};
};

inline std::uint64_t xgetbv(std::uint32_t register_id) {
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(register_id));
    return static_cast<std::uint64_t>(eax)
        | (static_cast<std::uint64_t>(edx) << 32U);
}

inline bool cpu_supports_amx_bf16() {
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    constexpr unsigned int kAmxBf16Bit = 1U << 22;
    constexpr unsigned int kAmxTileBit = 1U << 24;
    return (edx & kAmxBf16Bit) != 0U && (edx & kAmxTileBit) != 0U;
}

inline bool xstate_supports_amx() {
    constexpr std::uint64_t kTileCfgBit = 1ULL << 17;
    constexpr std::uint64_t kTileDataBit = 1ULL << 18;
    return (xgetbv(0) & (kTileCfgBit | kTileDataBit))
        == (kTileCfgBit | kTileDataBit);
}

inline bool request_amx_permission_linux() {
    static std::atomic<int> cached{-1};
    const int current = cached.load(std::memory_order_acquire);
    if (current >= 0) {
        return current == 1;
    }
#if defined(__linux__)
    const long rc = syscall(
        SYS_arch_prctl,
        ARCH_REQ_XCOMP_PERM,
        ARCH_XCOMP_TILEDATA
    );
    cached.store(rc == 0 ? 1 : 0, std::memory_order_release);
    return rc == 0;
#else
    cached.store(0, std::memory_order_release);
    return false;
#endif
}

inline std::uint16_t fp32_to_bf16(float value) {
    std::uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    const std::uint32_t lsb = (raw >> 16U) & 1U;
    raw += 0x7FFFU + lsb;
    return static_cast<std::uint16_t>(raw >> 16U);
}

inline void pack_a_block_bf16(
    const float* matrix,
    int row_stride,
    int row_start,
    int k_start,
    int rows,
    int cols,
    std::uint16_t* out
) {
    std::fill_n(out, kAmxTileRows * kAmxTileDepth, static_cast<std::uint16_t>(0));
    for (int row = 0; row < kAmxTileRows; ++row) {
        const int src_row = row_start + row;
        if (src_row >= rows) {
            continue;
        }
        for (int depth = 0; depth < kAmxTileDepth; ++depth) {
            const int src_col = k_start + depth;
            if (src_col >= cols) {
                continue;
            }
            out[row * kAmxTileDepth + depth] = fp32_to_bf16(
                matrix[static_cast<std::size_t>(src_row) * row_stride + src_col]
            );
        }
    }
}

inline void pack_b_block_bf16(
    const float* matrix,
    int row_stride,
    int k_start,
    int col_start,
    int rows,
    int cols,
    std::uint16_t* out
) {
    std::fill_n(out, kAmxTileDepth * kAmxTileCols, static_cast<std::uint16_t>(0));
    for (int depth = 0; depth < kAmxTileDepth; ++depth) {
        const int src_row = k_start + depth;
        if (src_row >= rows) {
            continue;
        }
        for (int col = 0; col < kAmxTileCols; ++col) {
            const int src_col = col_start + col;
            if (src_col >= cols) {
                continue;
            }
            out[depth * kAmxTileCols + col] = fp32_to_bf16(
                matrix[static_cast<std::size_t>(src_row) * row_stride + src_col]
            );
        }
    }
}

inline AmxTileConfig make_tile_config() {
    AmxTileConfig config{};
    config.colsb[0] = static_cast<std::uint16_t>(kAmxTileDepth * sizeof(std::uint16_t));
    config.rows[0] = static_cast<std::uint8_t>(kAmxTileRows);
    config.colsb[1] = static_cast<std::uint16_t>(kAmxTileCols * sizeof(std::uint16_t));
    config.rows[1] = static_cast<std::uint8_t>(kAmxTileDepth);
    config.colsb[2] = static_cast<std::uint16_t>(kAmxTileCols * sizeof(float));
    config.rows[2] = static_cast<std::uint8_t>(kAmxTileRows);
    return config;
}

#endif

}  // namespace

bool amx_compiled() {
#if defined(__x86_64__)
    return true;
#else
    return false;
#endif
}

bool amx_runtime_available() {
#if defined(__x86_64__)
    static std::atomic<int> cached{-1};
    const int current = cached.load(std::memory_order_acquire);
    if (current >= 0) {
        return current == 1;
    }
    const bool available = cpu_supports_amx_bf16()
        && request_amx_permission_linux()
        && xstate_supports_amx();
    cached.store(available ? 1 : 0, std::memory_order_release);
    return available;
#else
    return false;
#endif
}

bool amx_matmul_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int k,
    int n
) {
#if defined(__x86_64__)
    if (
        a == nullptr
        || b == nullptr
        || c == nullptr
        || m <= 0
        || k <= 0
        || n <= 0
        || k < kAmxTileDepth
        || n < kAmxTileCols
        || (static_cast<std::int64_t>(m) * k * n) < 4096
        || !amx_runtime_available()
    ) {
        return false;
    }

    alignas(64) std::array<std::uint16_t, kAmxTileRows * kAmxTileDepth> a_block{};
    alignas(64) std::array<std::uint16_t, kAmxTileDepth * kAmxTileCols> b_block{};
    alignas(64) std::array<float, kAmxTileRows * kAmxTileCols> c_block{};
    const AmxTileConfig config = make_tile_config();
    _tile_loadconfig(&config);

    for (int row_start = 0; row_start < m; row_start += kAmxTileRows) {
        for (int col_start = 0; col_start < n; col_start += kAmxTileCols) {
            _tile_zero(2);
            for (int k_start = 0; k_start < k; k_start += kAmxTileDepth) {
                pack_a_block_bf16(a, k, row_start, k_start, m, k, a_block.data());
                pack_b_block_bf16(b, n, k_start, col_start, k, n, b_block.data());
                _tile_loadd(0, a_block.data(), kAmxTileDepth * sizeof(std::uint16_t));
                _tile_loadd(1, b_block.data(), kAmxTileCols * sizeof(std::uint16_t));
                _tile_dpbf16ps(2, 0, 1);
            }
            _tile_stored(2, c_block.data(), kAmxTileCols * sizeof(float));
            for (int row = 0; row < kAmxTileRows; ++row) {
                const int dst_row = row_start + row;
                if (dst_row >= m) {
                    continue;
                }
                for (int col = 0; col < kAmxTileCols; ++col) {
                    const int dst_col = col_start + col;
                    if (dst_col >= n) {
                        continue;
                    }
                    c[static_cast<std::size_t>(dst_row) * n + dst_col] =
                        c_block[static_cast<std::size_t>(row) * kAmxTileCols + col];
                }
            }
        }
    }

    _tile_release();
    return true;
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)k;
    (void)n;
    return false;
#endif
}

}  // namespace anvil::native

extern "C" {

int anvil_compiled_with_amx() {
    return anvil::native::amx_compiled() ? 1 : 0;
}

int anvil_runtime_amx_available() {
    return anvil::native::amx_runtime_available() ? 1 : 0;
}

}
