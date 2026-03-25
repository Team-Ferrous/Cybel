/**
 * GGUF-accurate quantized matrix-vector multiply kernels.
 *
 * Supported:
 * - Q4_K
 * - Q6_K
 * - Q8_0
 *
 * AVX2 SIMD inner loops + OpenMP row/batch parallelism.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "fast_math.h"
#include "numa_allocator.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

extern "C" {
int anvil_get_num_threads_for_path(int decode_path);
int anvil_bind_worker_thread(int worker_tid, int role_decode);
}

namespace {

constexpr int QTYPE_Q4_K = 12;
constexpr int QTYPE_Q6_K = 14;
constexpr int QTYPE_Q8_0 = 8;
constexpr int QTYPE_Q4_K_R4 = 112;  // Anvil custom: Q4_K with R4 block-major row groups
constexpr int QTYPE_Q6_K_R4 = 114;  // Anvil custom: Q6_K with R4 block-major row groups
constexpr int QTYPE_Q6_K_LM = 214;  // Anvil custom: Q6_K expanded for LM-head decode

inline float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    const uint32_t mant = static_cast<uint32_t>(h) & 0x03FFu;
    const int exp = static_cast<int>((h >> 10) & 0x1Fu);

    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t norm_mant = mant;
            int norm_exp = -14;
            while ((norm_mant & 0x0400u) == 0) {
                norm_mant <<= 1;
                --norm_exp;
            }
            norm_mant &= 0x03FFu;
            bits = sign
                | (static_cast<uint32_t>(norm_exp + 127) << 23)
                | (norm_mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign
            | (static_cast<uint32_t>(exp - 15 + 127) << 23)
            | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline int read_env_threads(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return 0;
    }
    const int n = std::atoi(env);
    return n > 0 ? n : 0;
}

inline int get_num_threads(bool decode_path) {
    const int native = anvil_get_num_threads_for_path(decode_path ? 1 : 0);
    if (native > 0) {
        return native;
    }
    const int env_mode_threads = read_env_threads(
        decode_path ? "ANVIL_NUM_THREADS_DECODE" : "ANVIL_NUM_THREADS_BATCH"
    );
    const int env_threads = env_mode_threads > 0
        ? env_mode_threads
        : read_env_threads("ANVIL_NUM_THREADS");
    return std::max(1, env_threads);
}

inline int schedule_chunk_rows(int rows, int n_threads) {
    if (rows <= 0) {
        return 1;
    }
    const int denom = std::max(1, n_threads * 4);
    return std::max(4, rows / denom);
}

inline bool use_row_parallelism(int rows, int blocks_per_row, int n_threads) {
    if (n_threads <= 1) {
        return false;
    }
    const std::size_t work_units =
        static_cast<std::size_t>(std::max(rows, 0))
        * static_cast<std::size_t>(std::max(blocks_per_row, 1));
    return work_units >= static_cast<std::size_t>(n_threads * 64);
}

inline bool use_batch_row_parallelism(
    int batch,
    int rows,
    int blocks_per_row,
    int n_threads
) {
    if (n_threads <= 1) {
        return false;
    }
    const std::size_t work_units =
        static_cast<std::size_t>(std::max(batch, 0))
        * static_cast<std::size_t>(std::max(rows, 0))
        * static_cast<std::size_t>(std::max(blocks_per_row, 1));
    return work_units >= static_cast<std::size_t>(n_threads * 128);
}

inline bool use_dynamic_schedule(int rows, int n_threads) {
    return n_threads > 1 && rows > 0 && (rows / std::max(1, n_threads)) < 4;
}

inline bool strict_numa_enabled() {
    const char* env = std::getenv("ANVIL_NUMA_STRICT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return !(std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0);
}

inline bool decode_force_static_schedule() {
    if (!strict_numa_enabled()) {
        return false;
    }
    const char* mode = std::getenv("ANVIL_NUMA_AFFINITY_MODE");
    if (mode == nullptr || mode[0] == '\0') {
        return true;
    }
    return std::strcmp(mode, "off") != 0;
}

inline void maybe_bind_worker_thread(bool decode_path) {
#ifdef _OPENMP
    if (!strict_numa_enabled()) {
        return;
    }
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    thread_local int bound_decode = -1;
    const int decode_flag = decode_path ? 1 : 0;
    if (bound_decode != decode_flag) {
        (void)anvil_bind_worker_thread(tid, decode_flag);
        bound_decode = decode_flag;
    }
#else
    (void)decode_path;
#endif
}

inline void prefetch_read(const void* ptr, int locality = 3) {
#if defined(__GNUC__) || defined(__clang__)
    if (ptr != nullptr) {
        if (locality >= 3) {
            __builtin_prefetch(ptr, 0, 3);
        } else if (locality == 2) {
            __builtin_prefetch(ptr, 0, 2);
        } else {
            __builtin_prefetch(ptr, 0, 1);
        }
    }
#else
    (void)ptr;
    (void)locality;
#endif
}

inline int q6k_prefetch_lookahead(int blocks_per_row) {
    if (blocks_per_row >= 96) {
        return 4;
    }
    if (blocks_per_row >= 48) {
        return 3;
    }
    return 2;
}

inline int q6k_prefetch_locality(int blocks_per_row) {
    return (blocks_per_row >= 48) ? 2 : 3;
}

struct AlignedFloatBuffer {
    float* ptr = nullptr;
    std::size_t size = 0;

    ~AlignedFloatBuffer() {
        const auto opt = anvil::native::anvil_alloc_options_from_env();
        anvil::native::anvil_free_local_cpp(
            ptr,
            size * sizeof(float),
            opt
        );
    }

    void resize(std::size_t new_size) {
        if (new_size == size) {
            return;
        }
        const auto opt = anvil::native::anvil_alloc_options_from_env();
        anvil::native::anvil_free_local_cpp(
            ptr,
            size * sizeof(float),
            opt
        );
        ptr = nullptr;
        size = 0;
        if (new_size == 0) {
            return;
        }
        const std::size_t bytes = new_size * sizeof(float);
        const std::size_t rounded = (bytes + 63u) & ~std::size_t(63u);
        auto alloc_opt = anvil::native::anvil_alloc_options_from_env();
        alloc_opt.alignment = std::max<std::size_t>(alloc_opt.alignment, 64u);
        ptr = static_cast<float*>(anvil::native::anvil_alloc_local_cpp(rounded, alloc_opt));
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
        size = new_size;
    }

    float* data() { return ptr; }
};

template <int kBlockBytes>
inline int repack_rows_r4_generic(
    const void* src_rows,
    void* dst_rows,
    int k,
    int n
) {
    if (src_rows == nullptr || dst_rows == nullptr || k <= 0 || n <= 0) {
        return 0;
    }
    constexpr int kBlock = 256;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return 0;
    }

    const int blocks_per_row = k / kBlock;
    const std::size_t row_bytes =
        static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
    const auto* src = reinterpret_cast<const uint8_t*>(src_rows);
    auto* dst = reinterpret_cast<uint8_t*>(dst_rows);

    std::size_t dst_off = 0;
    int row = 0;
    for (; row + kRowsPerGroup <= n; row += kRowsPerGroup) {
        for (int b = 0; b < blocks_per_row; ++b) {
            for (int r = 0; r < kRowsPerGroup; ++r) {
                const std::size_t src_off =
                    static_cast<std::size_t>(row + r) * row_bytes
                    + static_cast<std::size_t>(b) * kBlockBytes;
                std::memcpy(
                    dst + dst_off,
                    src + src_off,
                    static_cast<std::size_t>(kBlockBytes)
                );
                dst_off += static_cast<std::size_t>(kBlockBytes);
            }
        }
    }

    if (row < n) {
        const std::size_t tail_rows = static_cast<std::size_t>(n - row);
        const std::size_t tail_src_off = static_cast<std::size_t>(row) * row_bytes;
        std::memcpy(
            dst + dst_off,
            src + tail_src_off,
            tail_rows * row_bytes
        );
    }
    return 1;
}

inline int repack_rows_q6k_lm(
    const void* src_rows,
    void* dst_rows,
    int k,
    int n
) {
    constexpr int kBlock = 256;
    constexpr int kSrcBlockBytes = 210;
    constexpr int kDstBlockBytes = 276;  // 256 int8 q + 16 int8 scales + 1 float d
    if (src_rows == nullptr || dst_rows == nullptr || k <= 0 || n <= 0 || (k % kBlock) != 0) {
        return 0;
    }

    const int blocks_per_row = k / kBlock;
    const auto* src = reinterpret_cast<const uint8_t*>(src_rows);
    auto* dst = reinterpret_cast<uint8_t*>(dst_rows);
    const std::size_t src_row_bytes = static_cast<std::size_t>(blocks_per_row) * kSrcBlockBytes;
    const std::size_t dst_row_bytes = static_cast<std::size_t>(blocks_per_row) * kDstBlockBytes;

    for (int row = 0; row < n; ++row) {
        const auto* src_row = src + static_cast<std::size_t>(row) * src_row_bytes;
        auto* dst_row = dst + static_cast<std::size_t>(row) * dst_row_bytes;
        for (int b = 0; b < blocks_per_row; ++b) {
            const auto* blk = src_row + static_cast<std::size_t>(b) * kSrcBlockBytes;
            auto* out_blk = dst_row + static_cast<std::size_t>(b) * kDstBlockBytes;
            auto* q_out = reinterpret_cast<int8_t*>(out_blk);
            auto* scales_out = reinterpret_cast<int8_t*>(out_blk + 256);

            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
            std::memcpy(scales_out, scales, 16);

            uint16_t d16 = 0;
            std::memcpy(&d16, blk + 208, sizeof(uint16_t));
            const float d = fp16_to_fp32(d16);
            std::memcpy(out_blk + 272, &d, sizeof(float));

            for (int g = 0; g < 8; ++g) {
                const int ql_seg = g / 4;
                const int ql_rem = g % 4;
                const int ql_shift = (ql_rem >= 2) ? 4 : 0;
                const int ql_start = (ql_rem % 2) * 32;

                const int qh_seg = g / 4;
                const int qh_shift = (g % 4) * 2;
                int8_t q32[32];
                for (int i = 0; i < 32; ++i) {
                    const uint8_t ql_b = ql[ql_seg * 64 + ql_start + i];
                    const uint8_t qh_b = qh[qh_seg * 32 + i];
                    const uint8_t lo = static_cast<uint8_t>((ql_b >> ql_shift) & 0x0F);
                    const uint8_t hi = static_cast<uint8_t>((qh_b >> qh_shift) & 0x03);
                    q32[i] = static_cast<int8_t>((lo | (hi << 4)) - 32);
                }
                for (int half = 0; half < 2; ++half) {
                    const int scale_idx = g * 2 + half;
                    const int q_off = half * 16;
                    std::memcpy(q_out + scale_idx * 16, q32 + q_off, 16);
                }
            }
        }
    }
    return 1;
}

#ifdef __AVX2__
inline float hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline __m256i unpack_q6k_32_avx2(
    const uint8_t* ql_ptr,
    int ql_shift,
    const uint8_t* qh_ptr,
    int qh_shift
) {
    const __m256i ql_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(ql_ptr)
    );
    const __m256i qh_bytes = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(qh_ptr)
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i hi_mask = _mm256_set1_epi8(0x03);

    const __m256i ql_nibs = _mm256_and_si256(
        (ql_shift == 0) ? ql_bytes : _mm256_srli_epi16(ql_bytes, 4),
        low_mask
    );

    __m256i qh_bits = qh_bytes;
    switch (qh_shift) {
        case 0: break;
        case 2: qh_bits = _mm256_srli_epi16(qh_bits, 2); break;
        case 4: qh_bits = _mm256_srli_epi16(qh_bits, 4); break;
        default: qh_bits = _mm256_srli_epi16(qh_bits, 6); break;
    }
    qh_bits = _mm256_and_si256(qh_bits, hi_mask);

    const __m256i q_u6 = _mm256_or_si256(ql_nibs, _mm256_slli_epi16(qh_bits, 4));
    return _mm256_sub_epi8(q_u6, _mm256_set1_epi8(32));
}
#endif

inline float dot_row_q4k(
    const float* x,
    const uint8_t* row_ptr,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kSub = 32;
    constexpr int kBlockBytes = 144;  // 2 d + 2 dmin + 12 scales + 128 qs
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + 2;
        if (pf_block < blocks_per_row) {
            prefetch_read(row_ptr + static_cast<std::size_t>(pf_block) * kBlockBytes);
            prefetch_read(x + static_cast<std::size_t>(pf_block) * kBlock);
        }
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;

        uint16_t d16 = 0;
        uint16_t dmin16 = 0;
        std::memcpy(&d16, blk, sizeof(uint16_t));
        std::memcpy(&dmin16, blk + 2, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);
        const float dmin = fp16_to_fp32(dmin16);

        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;

        float d_sc[8];
        float d_min[8];
        for (int j = 0; j < 8; ++j) {
            uint8_t sc = 0;
            uint8_t mn = 0;
            if (j < 4) {
                sc = static_cast<uint8_t>(scales[j] & 0x3F);
                mn = static_cast<uint8_t>(scales[4 + j] & 0x3F);
            } else {
                const int idx = j - 4;
                sc = static_cast<uint8_t>((scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30));
                mn = static_cast<uint8_t>((scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30));
            }
            d_sc[j] = d * static_cast<float>(sc);
            d_min[j] = dmin * static_cast<float>(mn);
        }

        const int k_base = b * kBlock;
        for (int j = 0; j < 8; ++j) {
            const int pair = j / 2;
            const bool high = (j & 1) != 0;
            const int byte_base = pair * 32;
            const int x_off = k_base + j * kSub;
            const float dsc = d_sc[j];
            const float dmn = d_min[j];

#ifdef __AVX2__
            __m256i raw_bytes = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(qs + byte_base)
            );
            __m256i nibbles = high ? _mm256_srli_epi16(raw_bytes, 4) : raw_bytes;
            nibbles = _mm256_and_si256(nibbles, _mm256_set1_epi8(0x0F));

            const __m128i lo_nibs = _mm256_castsi256_si128(nibbles);
            const __m128i hi_nibs = _mm256_extracti128_si256(nibbles, 1);
            const __m256 dsc_v = _mm256_set1_ps(dsc);
            const __m256 dmn_v = _mm256_set1_ps(dmn);
            __m256 acc_v = _mm256_setzero_ps();

            const __m256i q32_a = _mm256_cvtepu8_epi32(lo_nibs);
            const __m256i q32_b = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_nibs, 8));
            const __m256i q32_c = _mm256_cvtepu8_epi32(hi_nibs);
            const __m256i q32_d = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_nibs, 8));

            const __m256 qf_a = _mm256_cvtepi32_ps(q32_a);
            const __m256 qf_b = _mm256_cvtepi32_ps(q32_b);
            const __m256 qf_c = _mm256_cvtepi32_ps(q32_c);
            const __m256 qf_d = _mm256_cvtepi32_ps(q32_d);

            const __m256 xv_a = _mm256_loadu_ps(x + x_off + 0);
            const __m256 xv_b = _mm256_loadu_ps(x + x_off + 8);
            const __m256 xv_c = _mm256_loadu_ps(x + x_off + 16);
            const __m256 xv_d = _mm256_loadu_ps(x + x_off + 24);

            const __m256 val_a = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_a), dmn_v);
            const __m256 val_b = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_b), dmn_v);
            const __m256 val_c = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_c), dmn_v);
            const __m256 val_d = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_d), dmn_v);

            acc_v = _mm256_fmadd_ps(val_a, xv_a, acc_v);
            acc_v = _mm256_fmadd_ps(val_b, xv_b, acc_v);
            acc_v = _mm256_fmadd_ps(val_c, xv_c, acc_v);
            acc_v = _mm256_fmadd_ps(val_d, xv_d, acc_v);
            acc += hsum256_ps(acc_v);
#else
            for (int i = 0; i < kSub; ++i) {
                const uint8_t byte = qs[byte_base + i];
                const uint8_t q = high
                    ? static_cast<uint8_t>(byte >> 4)
                    : static_cast<uint8_t>(byte & 0x0F);
                acc += (dsc * static_cast<float>(q) - dmn) * x[x_off + i];
            }
#endif
        }
    }
    return acc;
}

inline void dot_rows_q4k_r4(
    const float* x,
    const uint8_t* group_ptr,
    int blocks_per_row,
    float* y4
) {
    constexpr int kBlock = 256;
    constexpr int kSub = 32;
    constexpr int kBlockBytes = 144;

#ifdef __AVX2__
    __m256 acc_v[4] = {
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    };

    float d_sc[4][8];
    float d_min[4][8];
    const uint8_t* qs_ptr[4] = {nullptr, nullptr, nullptr, nullptr};

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + 1;
        if (pf_block < blocks_per_row) {
            prefetch_read(group_ptr + static_cast<std::size_t>(pf_block) * 4 * kBlockBytes);
            prefetch_read(x + static_cast<std::size_t>(pf_block) * kBlock);
        }
        const uint8_t* blk_base = group_ptr + static_cast<std::size_t>(b) * 4 * kBlockBytes;
        for (int r = 0; r < 4; ++r) {
            const uint8_t* blk = blk_base + static_cast<std::size_t>(r) * kBlockBytes;

            uint16_t d16 = 0;
            uint16_t dmin16 = 0;
            std::memcpy(&d16, blk, sizeof(uint16_t));
            std::memcpy(&dmin16, blk + 2, sizeof(uint16_t));
            const float d = fp16_to_fp32(d16);
            const float dmin = fp16_to_fp32(dmin16);

            const uint8_t* scales = blk + 4;
            qs_ptr[r] = blk + 16;

            for (int j = 0; j < 8; ++j) {
                uint8_t sc = 0;
                uint8_t mn = 0;
                if (j < 4) {
                    sc = static_cast<uint8_t>(scales[j] & 0x3F);
                    mn = static_cast<uint8_t>(scales[4 + j] & 0x3F);
                } else {
                    const int idx = j - 4;
                    sc = static_cast<uint8_t>((scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30));
                    mn = static_cast<uint8_t>((scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30));
                }
                d_sc[r][j] = d * static_cast<float>(sc);
                d_min[r][j] = dmin * static_cast<float>(mn);
            }
        }

        const int k_base = b * kBlock;
        for (int j = 0; j < 8; ++j) {
            const int pair = j / 2;
            const bool high = (j & 1) != 0;
            const int byte_base = pair * 32;
            const int x_off = k_base + j * kSub;

            const __m256 xv_a = _mm256_loadu_ps(x + x_off + 0);
            const __m256 xv_b = _mm256_loadu_ps(x + x_off + 8);
            const __m256 xv_c = _mm256_loadu_ps(x + x_off + 16);
            const __m256 xv_d = _mm256_loadu_ps(x + x_off + 24);

            for (int r = 0; r < 4; ++r) {
                __m256i raw_bytes = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(qs_ptr[r] + byte_base)
                );
                __m256i nibbles = high ? _mm256_srli_epi16(raw_bytes, 4) : raw_bytes;
                nibbles = _mm256_and_si256(nibbles, _mm256_set1_epi8(0x0F));

                const __m128i lo_nibs = _mm256_castsi256_si128(nibbles);
                const __m128i hi_nibs = _mm256_extracti128_si256(nibbles, 1);
                const __m256i q32_a = _mm256_cvtepu8_epi32(lo_nibs);
                const __m256i q32_b = _mm256_cvtepu8_epi32(_mm_srli_si128(lo_nibs, 8));
                const __m256i q32_c = _mm256_cvtepu8_epi32(hi_nibs);
                const __m256i q32_d = _mm256_cvtepu8_epi32(_mm_srli_si128(hi_nibs, 8));

                const __m256 qf_a = _mm256_cvtepi32_ps(q32_a);
                const __m256 qf_b = _mm256_cvtepi32_ps(q32_b);
                const __m256 qf_c = _mm256_cvtepi32_ps(q32_c);
                const __m256 qf_d = _mm256_cvtepi32_ps(q32_d);

                const __m256 dsc_v = _mm256_set1_ps(d_sc[r][j]);
                const __m256 dmn_v = _mm256_set1_ps(d_min[r][j]);
                const __m256 val_a = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_a), dmn_v);
                const __m256 val_b = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_b), dmn_v);
                const __m256 val_c = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_c), dmn_v);
                const __m256 val_d = _mm256_sub_ps(_mm256_mul_ps(dsc_v, qf_d), dmn_v);

                acc_v[r] = _mm256_fmadd_ps(val_a, xv_a, acc_v[r]);
                acc_v[r] = _mm256_fmadd_ps(val_b, xv_b, acc_v[r]);
                acc_v[r] = _mm256_fmadd_ps(val_c, xv_c, acc_v[r]);
                acc_v[r] = _mm256_fmadd_ps(val_d, xv_d, acc_v[r]);
            }
        }
    }

    y4[0] = hsum256_ps(acc_v[0]);
    y4[1] = hsum256_ps(acc_v[1]);
    y4[2] = hsum256_ps(acc_v[2]);
    y4[3] = hsum256_ps(acc_v[3]);
#else
    constexpr int kSubBlocks = 8;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* blk_base = group_ptr + static_cast<std::size_t>(b) * 4 * kBlockBytes;
        float d_sc[4][8];
        float d_min[4][8];
        const uint8_t* qs_ptr[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int r = 0; r < 4; ++r) {
            const uint8_t* blk = blk_base + static_cast<std::size_t>(r) * kBlockBytes;
            uint16_t d16 = 0;
            uint16_t dmin16 = 0;
            std::memcpy(&d16, blk, sizeof(uint16_t));
            std::memcpy(&dmin16, blk + 2, sizeof(uint16_t));
            const float d = fp16_to_fp32(d16);
            const float dmin = fp16_to_fp32(dmin16);
            const uint8_t* scales = blk + 4;
            qs_ptr[r] = blk + 16;
            for (int j = 0; j < kSubBlocks; ++j) {
                uint8_t sc = 0;
                uint8_t mn = 0;
                if (j < 4) {
                    sc = static_cast<uint8_t>(scales[j] & 0x3F);
                    mn = static_cast<uint8_t>(scales[4 + j] & 0x3F);
                } else {
                    const int idx = j - 4;
                    sc = static_cast<uint8_t>((scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30));
                    mn = static_cast<uint8_t>((scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30));
                }
                d_sc[r][j] = d * static_cast<float>(sc);
                d_min[r][j] = dmin * static_cast<float>(mn);
            }
        }

        const int k_base = b * kBlock;
        for (int j = 0; j < 8; ++j) {
            const int pair = j / 2;
            const bool high = (j & 1) != 0;
            const int byte_base = pair * 32;
            const int x_off = k_base + j * kSub;
            for (int r = 0; r < 4; ++r) {
                const float dsc = d_sc[r][j];
                const float dmn = d_min[r][j];
                for (int i = 0; i < 32; ++i) {
                    const uint8_t byte = qs_ptr[r][byte_base + i];
                    const uint8_t q = high ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
                    acc[r] += (dsc * static_cast<float>(q) - dmn) * x[x_off + i];
                }
            }
        }
    }
    y4[0] = acc[0];
    y4[1] = acc[1];
    y4[2] = acc[2];
    y4[3] = acc[3];
#endif
}

inline float dot_row_q6k(
    const float* x,
    const uint8_t* row_ptr,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;  // 128 ql + 64 qh + 16 scales + 2 d
    const int prefetch_lookahead = q6k_prefetch_lookahead(blocks_per_row);
    const int prefetch_locality = q6k_prefetch_locality(blocks_per_row);
    float acc = 0.0f;
#ifdef __AVX2__
    __m256 acc_v = _mm256_setzero_ps();
#endif

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + prefetch_lookahead;
        if (pf_block < blocks_per_row) {
            prefetch_read(
                row_ptr + static_cast<std::size_t>(pf_block) * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                x + static_cast<std::size_t>(pf_block) * kBlock,
                prefetch_locality
            );
        }
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);

        uint16_t d16 = 0;
        std::memcpy(&d16, blk + 208, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);

        const int k_base = b * kBlock;
        for (int g = 0; g < 8; ++g) {
            const int ql_seg = g / 4;
            const int ql_rem = g % 4;
            const int ql_shift = (ql_rem >= 2) ? 4 : 0;
            const int ql_start = (ql_rem % 2) * 32;

            const int qh_seg = g / 4;
            const int qh_shift = (g % 4) * 2;
#ifdef __AVX2__
            const __m256i q32 = unpack_q6k_32_avx2(
                ql + ql_seg * 64 + ql_start,
                ql_shift,
                qh + qh_seg * 32,
                qh_shift
            );
            const __m128i q16_lo = _mm256_castsi256_si128(q32);
            const __m128i q16_hi = _mm256_extracti128_si256(q32, 1);
#else
            int8_t q32[32];
            for (int i = 0; i < 32; ++i) {
                const uint8_t ql_b = ql[ql_seg * 64 + ql_start + i];
                const uint8_t qh_b = qh[qh_seg * 32 + i];
                const uint8_t lo = static_cast<uint8_t>((ql_b >> ql_shift) & 0x0F);
                const uint8_t hi = static_cast<uint8_t>((qh_b >> qh_shift) & 0x03);
                q32[i] = static_cast<int8_t>((lo | (hi << 4)) - 32);
            }
#endif

            for (int half = 0; half < 2; ++half) {
                const int scale_idx = g * 2 + half;
                const float scale = d * static_cast<float>(scales[scale_idx]);
                const int out_off = k_base + scale_idx * 16;
#ifdef __AVX2__
                const __m128i q16 = (half == 0) ? q16_lo : q16_hi;
                const __m128i q8_a = q16;
                const __m128i q8_b = _mm_srli_si128(q16, 8);
                const __m256i q32_a = _mm256_cvtepi8_epi32(q8_a);
                const __m256i q32_b = _mm256_cvtepi8_epi32(q8_b);
                const __m256 qf_a = _mm256_cvtepi32_ps(q32_a);
                const __m256 qf_b = _mm256_cvtepi32_ps(q32_b);
                const __m256 scale_v = _mm256_set1_ps(scale);
                const __m256 xv_a = _mm256_loadu_ps(x + out_off + 0);
                const __m256 xv_b = _mm256_loadu_ps(x + out_off + 8);
                acc_v = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_a), xv_a, acc_v);
                acc_v = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_b), xv_b, acc_v);
#else
                const int q_off = half * 16;
                for (int i = 0; i < 16; ++i) {
                    acc += scale * static_cast<float>(q32[q_off + i]) * x[out_off + i];
                }
#endif
            }
        }
    }
#ifdef __AVX2__
    acc += hsum256_ps(acc_v);
#endif
    return acc;
}

inline void dot_rows_q6k(
    const float* x,
    const uint8_t* row_ptr0,
    const uint8_t* row_ptr1,
    const uint8_t* row_ptr2,
    const uint8_t* row_ptr3,
    int blocks_per_row,
    float* y4
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;  // 128 ql + 64 qh + 16 scales + 2 d
    const uint8_t* row_ptrs[4] = {row_ptr0, row_ptr1, row_ptr2, row_ptr3};
    const int prefetch_lookahead = q6k_prefetch_lookahead(blocks_per_row);
    const int prefetch_locality = q6k_prefetch_locality(blocks_per_row);
#ifdef __AVX2__
    __m256 acc_v[4] = {
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    };
#else
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + prefetch_lookahead;
        if (pf_block < blocks_per_row) {
            prefetch_read(
                row_ptr0 + static_cast<std::size_t>(pf_block) * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                row_ptr1 + static_cast<std::size_t>(pf_block) * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                row_ptr2 + static_cast<std::size_t>(pf_block) * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                row_ptr3 + static_cast<std::size_t>(pf_block) * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                x + static_cast<std::size_t>(pf_block) * kBlock,
                prefetch_locality
            );
        }
        const uint8_t* ql[4] = {nullptr, nullptr, nullptr, nullptr};
        const uint8_t* qh[4] = {nullptr, nullptr, nullptr, nullptr};
        const int8_t* scales[4] = {nullptr, nullptr, nullptr, nullptr};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int r = 0; r < 4; ++r) {
            const auto* blk = row_ptrs[r] + static_cast<std::size_t>(b) * kBlockBytes;
            ql[r] = blk;
            qh[r] = blk + 128;
            scales[r] = reinterpret_cast<const int8_t*>(blk + 192);
            uint16_t d16 = 0;
            std::memcpy(&d16, blk + 208, sizeof(uint16_t));
            d[r] = fp16_to_fp32(d16);
        }

        const int k_base = b * kBlock;
        for (int g = 0; g < 8; ++g) {
            const int ql_seg = g / 4;
            const int ql_rem = g % 4;
            const int ql_shift = (ql_rem >= 2) ? 4 : 0;
            const int ql_start = (ql_rem % 2) * 32;

            const int qh_seg = g / 4;
            const int qh_shift = (g % 4) * 2;
#ifdef __AVX2__
            __m128i q16_lo[4];
            __m128i q16_hi[4];
            for (int r = 0; r < 4; ++r) {
                const __m256i q32 = unpack_q6k_32_avx2(
                    ql[r] + ql_seg * 64 + ql_start,
                    ql_shift,
                    qh[r] + qh_seg * 32,
                    qh_shift
                );
                q16_lo[r] = _mm256_castsi256_si128(q32);
                q16_hi[r] = _mm256_extracti128_si256(q32, 1);
            }
#else
            int8_t q32[4][32];
            for (int r = 0; r < 4; ++r) {
                for (int i = 0; i < 32; ++i) {
                    const uint8_t ql_b = ql[r][ql_seg * 64 + ql_start + i];
                    const uint8_t qh_b = qh[r][qh_seg * 32 + i];
                    const uint8_t lo = static_cast<uint8_t>((ql_b >> ql_shift) & 0x0F);
                    const uint8_t hi = static_cast<uint8_t>((qh_b >> qh_shift) & 0x03);
                    q32[r][i] = static_cast<int8_t>((lo | (hi << 4)) - 32);
                }
            }
#endif

            for (int half = 0; half < 2; ++half) {
                const int scale_idx = g * 2 + half;
                const int out_off = k_base + scale_idx * 16;
#ifdef __AVX2__
                const __m256 xv_a = _mm256_loadu_ps(x + out_off + 0);
                const __m256 xv_b = _mm256_loadu_ps(x + out_off + 8);
                for (int r = 0; r < 4; ++r) {
                    const float scale = d[r] * static_cast<float>(scales[r][scale_idx]);
                    const __m256 scale_v = _mm256_set1_ps(scale);
                    const __m128i q16 = (half == 0) ? q16_lo[r] : q16_hi[r];
                    const __m128i q8_a = q16;
                    const __m128i q8_b = _mm_srli_si128(q16, 8);
                    const __m256i q32_a = _mm256_cvtepi8_epi32(q8_a);
                    const __m256i q32_b = _mm256_cvtepi8_epi32(q8_b);
                    const __m256 qf_a = _mm256_cvtepi32_ps(q32_a);
                    const __m256 qf_b = _mm256_cvtepi32_ps(q32_b);
                    acc_v[r] = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_a), xv_a, acc_v[r]);
                    acc_v[r] = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_b), xv_b, acc_v[r]);
                }
#else
                const int q_off = half * 16;
                for (int r = 0; r < 4; ++r) {
                    const float scale = d[r] * static_cast<float>(scales[r][scale_idx]);
                    for (int i = 0; i < 16; ++i) {
                        acc[r] += scale * static_cast<float>(q32[r][q_off + i]) * x[out_off + i];
                    }
                }
#endif
            }
        }
    }

#ifdef __AVX2__
    y4[0] = hsum256_ps(acc_v[0]);
    y4[1] = hsum256_ps(acc_v[1]);
    y4[2] = hsum256_ps(acc_v[2]);
    y4[3] = hsum256_ps(acc_v[3]);
#else
    y4[0] = acc[0];
    y4[1] = acc[1];
    y4[2] = acc[2];
    y4[3] = acc[3];
#endif
}

inline void dot_rows_q6k_r4(
    const float* x,
    const uint8_t* group_ptr,
    int blocks_per_row,
    float* y4
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;  // 128 ql + 64 qh + 16 scales + 2 d
    constexpr int kRowsPerGroup = 4;
    const int prefetch_lookahead = q6k_prefetch_lookahead(blocks_per_row);
    const int prefetch_locality = q6k_prefetch_locality(blocks_per_row);
#ifdef __AVX2__
    __m256 acc_v[4] = {
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    };
#else
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + prefetch_lookahead;
        if (pf_block < blocks_per_row) {
            prefetch_read(
                group_ptr + static_cast<std::size_t>(pf_block) * kRowsPerGroup * kBlockBytes,
                prefetch_locality
            );
            prefetch_read(
                x + static_cast<std::size_t>(pf_block) * kBlock,
                prefetch_locality
            );
        }
        const uint8_t* ql[4] = {nullptr, nullptr, nullptr, nullptr};
        const uint8_t* qh[4] = {nullptr, nullptr, nullptr, nullptr};
        const int8_t* scales[4] = {nullptr, nullptr, nullptr, nullptr};
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        const uint8_t* blk_base =
            group_ptr + static_cast<std::size_t>(b) * kRowsPerGroup * kBlockBytes;

        for (int r = 0; r < 4; ++r) {
            const auto* blk = blk_base + static_cast<std::size_t>(r) * kBlockBytes;
            ql[r] = blk;
            qh[r] = blk + 128;
            scales[r] = reinterpret_cast<const int8_t*>(blk + 192);
            uint16_t d16 = 0;
            std::memcpy(&d16, blk + 208, sizeof(uint16_t));
            d[r] = fp16_to_fp32(d16);
        }

        const int k_base = b * kBlock;
        for (int g = 0; g < 8; ++g) {
            const int ql_seg = g / 4;
            const int ql_rem = g % 4;
            const int ql_shift = (ql_rem >= 2) ? 4 : 0;
            const int ql_start = (ql_rem % 2) * 32;

            const int qh_seg = g / 4;
            const int qh_shift = (g % 4) * 2;
            __m128i q16_lo[4];
            __m128i q16_hi[4];
            for (int r = 0; r < 4; ++r) {
                const __m256i q32 = unpack_q6k_32_avx2(
                    ql[r] + ql_seg * 64 + ql_start,
                    ql_shift,
                    qh[r] + qh_seg * 32,
                    qh_shift
                );
                q16_lo[r] = _mm256_castsi256_si128(q32);
                q16_hi[r] = _mm256_extracti128_si256(q32, 1);
            }

            for (int half = 0; half < 2; ++half) {
                const int scale_idx = g * 2 + half;
                const int out_off = k_base + scale_idx * 16;
#ifdef __AVX2__
                const __m256 xv_a = _mm256_loadu_ps(x + out_off + 0);
                const __m256 xv_b = _mm256_loadu_ps(x + out_off + 8);
                for (int r = 0; r < 4; ++r) {
                    const float scale = d[r] * static_cast<float>(scales[r][scale_idx]);
                    const __m256 scale_v = _mm256_set1_ps(scale);
                    const __m128i q16 = (half == 0) ? q16_lo[r] : q16_hi[r];
                    const __m128i q8_a = q16;
                    const __m128i q8_b = _mm_srli_si128(q16, 8);
                    const __m256i q32_a = _mm256_cvtepi8_epi32(q8_a);
                    const __m256i q32_b = _mm256_cvtepi8_epi32(q8_b);
                    const __m256 qf_a = _mm256_cvtepi32_ps(q32_a);
                    const __m256 qf_b = _mm256_cvtepi32_ps(q32_b);
                    acc_v[r] = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_a), xv_a, acc_v[r]);
                    acc_v[r] = _mm256_fmadd_ps(_mm256_mul_ps(scale_v, qf_b), xv_b, acc_v[r]);
                }
#else
                const int q_off = half * 16;
                for (int r = 0; r < 4; ++r) {
                    const float scale = d[r] * static_cast<float>(scales[r][scale_idx]);
                    for (int i = 0; i < 16; ++i) {
                        acc[r] += scale * static_cast<float>(q32[r][q_off + i]) * x[out_off + i];
                    }
                }
#endif
            }
        }
    }

#ifdef __AVX2__
    y4[0] = hsum256_ps(acc_v[0]);
    y4[1] = hsum256_ps(acc_v[1]);
    y4[2] = hsum256_ps(acc_v[2]);
    y4[3] = hsum256_ps(acc_v[3]);
#else
    y4[0] = acc[0];
    y4[1] = acc[1];
    y4[2] = acc[2];
    y4[3] = acc[3];
#endif
}

inline float dot_row_q8_0(
    const float* x,
    const uint8_t* row_ptr,
    int blocks_per_row
) {
    constexpr int kBlock = 32;
    constexpr int kBlockBytes = 34;  // 2 d(fp16) + 32 qs(int8)
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + 1;
        if (pf_block < blocks_per_row) {
            prefetch_read(row_ptr + static_cast<std::size_t>(pf_block) * kBlockBytes);
            prefetch_read(x + static_cast<std::size_t>(pf_block) * kBlock);
        }
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        uint16_t d16 = 0;
        std::memcpy(&d16, blk, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);
        const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const int x_off = b * kBlock;

#ifdef __AVX2__
        __m256 d_vec = _mm256_set1_ps(d);
        __m256 acc_v = _mm256_setzero_ps();
        for (int i = 0; i < 32; i += 8) {
            __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(qs + i));
            __m256i q32 = _mm256_cvtepi8_epi32(q8);
            __m256 qf = _mm256_cvtepi32_ps(q32);
            __m256 xv = _mm256_loadu_ps(x + x_off + i);
            acc_v = _mm256_fmadd_ps(_mm256_mul_ps(d_vec, qf), xv, acc_v);
        }
        acc += hsum256_ps(acc_v);
#else
        for (int i = 0; i < kBlock; ++i) {
            acc += d * static_cast<float>(qs[i]) * x[x_off + i];
        }
#endif
    }
    return acc;
}

inline float dot_row_q6k_lm(
    const float* x,
    const uint8_t* row_ptr,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 276;
    float acc = 0.0f;

    for (int b = 0; b < blocks_per_row; ++b) {
        const int pf_block = b + 1;
        if (pf_block < blocks_per_row) {
            prefetch_read(row_ptr + static_cast<std::size_t>(pf_block) * kBlockBytes);
            prefetch_read(x + static_cast<std::size_t>(pf_block) * kBlock);
        }
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        const auto* q = reinterpret_cast<const int8_t*>(blk);
        const auto* scales = reinterpret_cast<const int8_t*>(blk + 256);
        float d = 0.0f;
        std::memcpy(&d, blk + 272, sizeof(float));
        const int k_base = b * kBlock;
#ifdef __AVX2__
        __m256 acc_v_even = _mm256_setzero_ps();
        __m256 acc_v_odd = _mm256_setzero_ps();

        for (int scale_idx = 0; scale_idx < 16; scale_idx += 2) {
            const int out_off_even = k_base + scale_idx * 16;
            const float scale_even = d * static_cast<float>(scales[scale_idx]);
            const auto* q16_even = q + scale_idx * 16;
            const __m256 scale_v_even = _mm256_set1_ps(scale_even);
            const __m128i q8_even_a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q16_even));
            const __m128i q8_even_b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q16_even + 8));
            const __m256i q32_even_a = _mm256_cvtepi8_epi32(q8_even_a);
            const __m256i q32_even_b = _mm256_cvtepi8_epi32(q8_even_b);
            const __m256 qf_even_a = _mm256_cvtepi32_ps(q32_even_a);
            const __m256 qf_even_b = _mm256_cvtepi32_ps(q32_even_b);
            const __m256 xv_even_a = _mm256_loadu_ps(x + out_off_even + 0);
            const __m256 xv_even_b = _mm256_loadu_ps(x + out_off_even + 8);
            acc_v_even = _mm256_fmadd_ps(_mm256_mul_ps(scale_v_even, qf_even_a), xv_even_a, acc_v_even);
            acc_v_even = _mm256_fmadd_ps(_mm256_mul_ps(scale_v_even, qf_even_b), xv_even_b, acc_v_even);

            const int odd_idx = scale_idx + 1;
            const int out_off_odd = k_base + odd_idx * 16;
            const float scale_odd = d * static_cast<float>(scales[odd_idx]);
            const auto* q16_odd = q + odd_idx * 16;
            const __m256 scale_v_odd = _mm256_set1_ps(scale_odd);
            const __m128i q8_odd_a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q16_odd));
            const __m128i q8_odd_b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q16_odd + 8));
            const __m256i q32_odd_a = _mm256_cvtepi8_epi32(q8_odd_a);
            const __m256i q32_odd_b = _mm256_cvtepi8_epi32(q8_odd_b);
            const __m256 qf_odd_a = _mm256_cvtepi32_ps(q32_odd_a);
            const __m256 qf_odd_b = _mm256_cvtepi32_ps(q32_odd_b);
            const __m256 xv_odd_a = _mm256_loadu_ps(x + out_off_odd + 0);
            const __m256 xv_odd_b = _mm256_loadu_ps(x + out_off_odd + 8);
            acc_v_odd = _mm256_fmadd_ps(_mm256_mul_ps(scale_v_odd, qf_odd_a), xv_odd_a, acc_v_odd);
            acc_v_odd = _mm256_fmadd_ps(_mm256_mul_ps(scale_v_odd, qf_odd_b), xv_odd_b, acc_v_odd);
        }
        acc += hsum256_ps(_mm256_add_ps(acc_v_even, acc_v_odd));
#else
        for (int scale_idx = 0; scale_idx < 16; ++scale_idx) {
            const int out_off = k_base + scale_idx * 16;
            const float scale = d * static_cast<float>(scales[scale_idx]);
            const auto* q16 = q + scale_idx * 16;
            for (int i = 0; i < 16; ++i) {
                acc += scale * static_cast<float>(q16[i]) * x[out_off + i];
            }
        }
#endif
    }

    return acc;
}

inline void compute_quant_matvec_chunk(
    const float* x,
    const void* quant_data,
    int qtype,
    float* y,
    int k,
    int n,
    int tid,
    int n_threads
) {
    if (x == nullptr || quant_data == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    const auto* data = reinterpret_cast<const uint8_t*>(quant_data);
    switch (qtype) {
        case QTYPE_Q4_K: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 144;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int begin = (n * tid) / std::max(1, n_threads);
            const int end = (n * (tid + 1)) / std::max(1, n_threads);
            for (int row = begin; row < end; ++row) {
                const auto* row_ptr =
                    data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
                y[row] = dot_row_q4k(x, row_ptr, blocks_per_row);
            }
            return;
        }
        case QTYPE_Q4_K_R4: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 144;
            constexpr int kRowsPerGroup = 4;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int full_groups = n / kRowsPerGroup;
            const int tail_rows = n - full_groups * kRowsPerGroup;
            const int g_begin = (full_groups * tid) / std::max(1, n_threads);
            const int g_end = (full_groups * (tid + 1)) / std::max(1, n_threads);
            for (int g = g_begin; g < g_end; ++g) {
                const auto* group_ptr =
                    data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                dot_rows_q4k_r4(x, group_ptr, blocks_per_row, out4);
                const int row_base = g * kRowsPerGroup;
                y[row_base + 0] = out4[0];
                y[row_base + 1] = out4[1];
                y[row_base + 2] = out4[2];
                y[row_base + 3] = out4[3];
            }
            if (tail_rows > 0 && tid == n_threads - 1) {
                const std::size_t tail_base =
                    static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
                const auto* tail_ptr = data + tail_base;
                for (int t = 0; t < tail_rows; ++t) {
                    const auto* row_ptr =
                        tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
                    y[full_groups * kRowsPerGroup + t] = dot_row_q4k(x, row_ptr, blocks_per_row);
                }
            }
            return;
        }
        case QTYPE_Q6_K: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 210;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int begin = (n * tid) / std::max(1, n_threads);
            const int end = (n * (tid + 1)) / std::max(1, n_threads);
            int row = begin;
            for (; row + 3 < end; row += 4) {
                const auto* row_ptr0 =
                    data + static_cast<std::size_t>(row + 0) * blocks_per_row * kBlockBytes;
                const auto* row_ptr1 =
                    data + static_cast<std::size_t>(row + 1) * blocks_per_row * kBlockBytes;
                const auto* row_ptr2 =
                    data + static_cast<std::size_t>(row + 2) * blocks_per_row * kBlockBytes;
                const auto* row_ptr3 =
                    data + static_cast<std::size_t>(row + 3) * blocks_per_row * kBlockBytes;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                dot_rows_q6k(
                    x,
                    row_ptr0,
                    row_ptr1,
                    row_ptr2,
                    row_ptr3,
                    blocks_per_row,
                    out4
                );
                y[row + 0] = out4[0];
                y[row + 1] = out4[1];
                y[row + 2] = out4[2];
                y[row + 3] = out4[3];
            }
            for (; row < end; ++row) {
                const auto* row_ptr =
                    data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
                y[row] = dot_row_q6k(x, row_ptr, blocks_per_row);
            }
            return;
        }
        case QTYPE_Q6_K_R4: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 210;
            constexpr int kRowsPerGroup = 4;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int full_groups = n / kRowsPerGroup;
            const int tail_rows = n - full_groups * kRowsPerGroup;
            const int g_begin = (full_groups * tid) / std::max(1, n_threads);
            const int g_end = (full_groups * (tid + 1)) / std::max(1, n_threads);
            for (int g = g_begin; g < g_end; ++g) {
                const auto* group_ptr =
                    data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
                float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                dot_rows_q6k_r4(x, group_ptr, blocks_per_row, out4);
                const int row_base = g * kRowsPerGroup;
                y[row_base + 0] = out4[0];
                y[row_base + 1] = out4[1];
                y[row_base + 2] = out4[2];
                y[row_base + 3] = out4[3];
            }
            if (tail_rows > 0 && tid == n_threads - 1) {
                const std::size_t tail_base =
                    static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
                const auto* tail_ptr = data + tail_base;
                for (int t = 0; t < tail_rows; ++t) {
                    const auto* row_ptr =
                        tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
                    y[full_groups * kRowsPerGroup + t] = dot_row_q6k(x, row_ptr, blocks_per_row);
                }
            }
            return;
        }
        case QTYPE_Q6_K_LM: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 276;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int begin = (n * tid) / std::max(1, n_threads);
            const int end = (n * (tid + 1)) / std::max(1, n_threads);
            for (int row = begin; row < end; ++row) {
                const auto* row_ptr =
                    data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
                y[row] = dot_row_q6k_lm(x, row_ptr, blocks_per_row);
            }
            return;
        }
        case QTYPE_Q8_0: {
            constexpr int kBlock = 32;
            constexpr int kBlockBytes = 34;
            if ((k % kBlock) != 0) return;
            const int blocks_per_row = k / kBlock;
            const int begin = (n * tid) / std::max(1, n_threads);
            const int end = (n * (tid + 1)) / std::max(1, n_threads);
            for (int row = begin; row < end; ++row) {
                const auto* row_ptr =
                    data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
                y[row] = dot_row_q8_0(x, row_ptr, blocks_per_row);
            }
            return;
        }
        default:
            return;
    }
}

inline void dequantize_q4k_block(
    const uint8_t* blk,
    float* out_block
) {
    constexpr int kSub = 32;

    uint16_t d16 = 0;
    uint16_t dmin16 = 0;
    std::memcpy(&d16, blk, sizeof(uint16_t));
    std::memcpy(&dmin16, blk + 2, sizeof(uint16_t));
    const float d = fp16_to_fp32(d16);
    const float dmin = fp16_to_fp32(dmin16);

    const uint8_t* scales = blk + 4;
    const uint8_t* qs = blk + 16;

    float d_sc[8];
    float d_min[8];
    for (int j = 0; j < 8; ++j) {
        uint8_t sc = 0;
        uint8_t mn = 0;
        if (j < 4) {
            sc = static_cast<uint8_t>(scales[j] & 0x3F);
            mn = static_cast<uint8_t>(scales[4 + j] & 0x3F);
        } else {
            const int idx = j - 4;
            sc = static_cast<uint8_t>(
                (scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30)
            );
            mn = static_cast<uint8_t>(
                (scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30)
            );
        }
        d_sc[j] = d * static_cast<float>(sc);
        d_min[j] = dmin * static_cast<float>(mn);
    }

    for (int j = 0; j < 8; ++j) {
        const int pair = j / 2;
        const bool high = (j & 1) != 0;
        const int byte_base = pair * 32;
        const int out_off = j * kSub;
        const float dsc = d_sc[j];
        const float dmn = d_min[j];
        for (int i = 0; i < kSub; ++i) {
            const uint8_t byte = qs[byte_base + i];
            const uint8_t q = high
                ? static_cast<uint8_t>(byte >> 4)
                : static_cast<uint8_t>(byte & 0x0F);
            out_block[out_off + i] = dsc * static_cast<float>(q) - dmn;
        }
    }
}

inline void dequantize_row_q4k(
    const uint8_t* row_ptr,
    float* out,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    for (int b = 0; b < blocks_per_row; ++b) {
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        dequantize_q4k_block(blk, out + static_cast<std::size_t>(b) * kBlock);
    }
}

inline void dequantize_row_q4k_r4(
    const uint8_t* group_ptr,
    int row_in_group,
    float* out,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    constexpr int kRowsPerGroup = 4;
    for (int b = 0; b < blocks_per_row; ++b) {
        const auto* blk =
            group_ptr
            + static_cast<std::size_t>(b * kRowsPerGroup + row_in_group) * kBlockBytes;
        dequantize_q4k_block(blk, out + static_cast<std::size_t>(b) * kBlock);
    }
}

inline void dequantize_row_q6k(
    const uint8_t* row_ptr,
    float* out,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;

    for (int b = 0; b < blocks_per_row; ++b) {
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);

        uint16_t d16 = 0;
        std::memcpy(&d16, blk + 208, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);

        const int k_base = b * kBlock;
        for (int g = 0; g < 8; ++g) {
            const int ql_seg = g / 4;
            const int ql_rem = g % 4;
            const int ql_shift = (ql_rem >= 2) ? 4 : 0;
            const int ql_start = (ql_rem % 2) * 32;

            const int qh_seg = g / 4;
            const int qh_shift = (g % 4) * 2;

            int8_t q32[32];
            for (int i = 0; i < 32; ++i) {
                const uint8_t ql_b = ql[ql_seg * 64 + ql_start + i];
                const uint8_t qh_b = qh[qh_seg * 32 + i];
                const uint8_t lo = static_cast<uint8_t>((ql_b >> ql_shift) & 0x0F);
                const uint8_t hi = static_cast<uint8_t>((qh_b >> qh_shift) & 0x03);
                q32[i] = static_cast<int8_t>((lo | (hi << 4)) - 32);
            }

            for (int half = 0; half < 2; ++half) {
                const int scale_idx = g * 2 + half;
                const float scale = d * static_cast<float>(scales[scale_idx]);
                const int out_off = k_base + scale_idx * 16;
                const int q_off = half * 16;
                for (int i = 0; i < 16; ++i) {
                    out[out_off + i] = scale * static_cast<float>(q32[q_off + i]);
                }
            }
        }
    }
}

inline void dequantize_row_q6k_r4(
    const uint8_t* group_ptr,
    int row_in_group,
    float* out,
    int blocks_per_row
) {
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;
    constexpr int kRowsPerGroup = 4;

    for (int b = 0; b < blocks_per_row; ++b) {
        const auto* blk =
            group_ptr
            + static_cast<std::size_t>(b * kRowsPerGroup + row_in_group) * kBlockBytes;
        const uint8_t* ql = blk;
        const uint8_t* qh = blk + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);

        uint16_t d16 = 0;
        std::memcpy(&d16, blk + 208, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);

        const int k_base = b * kBlock;
        for (int g = 0; g < 8; ++g) {
            const int ql_seg = g / 4;
            const int ql_rem = g % 4;
            const int ql_shift = (ql_rem >= 2) ? 4 : 0;
            const int ql_start = (ql_rem % 2) * 32;

            const int qh_seg = g / 4;
            const int qh_shift = (g % 4) * 2;

            int8_t q32[32];
            for (int i = 0; i < 32; ++i) {
                const uint8_t ql_b = ql[ql_seg * 64 + ql_start + i];
                const uint8_t qh_b = qh[qh_seg * 32 + i];
                const uint8_t lo = static_cast<uint8_t>((ql_b >> ql_shift) & 0x0F);
                const uint8_t hi = static_cast<uint8_t>((qh_b >> qh_shift) & 0x03);
                q32[i] = static_cast<int8_t>((lo | (hi << 4)) - 32);
            }

            for (int half = 0; half < 2; ++half) {
                const int scale_idx = g * 2 + half;
                const float scale = d * static_cast<float>(scales[scale_idx]);
                const int out_off = k_base + scale_idx * 16;
                const int q_off = half * 16;
                for (int i = 0; i < 16; ++i) {
                    out[out_off + i] = scale * static_cast<float>(q32[q_off + i]);
                }
            }
        }
    }
}

inline void dequantize_row_q8_0(
    const uint8_t* row_ptr,
    float* out,
    int blocks_per_row
) {
    constexpr int kBlock = 32;
    constexpr int kBlockBytes = 34;

    for (int b = 0; b < blocks_per_row; ++b) {
        const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
        uint16_t d16 = 0;
        std::memcpy(&d16, blk, sizeof(uint16_t));
        const float d = fp16_to_fp32(d16);
        const auto* qs = reinterpret_cast<const int8_t*>(blk + 2);
        const int out_off = b * kBlock;
        for (int i = 0; i < kBlock; ++i) {
            out[out_off + i] = d * static_cast<float>(qs[i]);
        }
    }
}

}  // namespace

extern "C" {

void simd_matvec_q4k(const float* x, const void* a_quant, float* y, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(true);

    const bool dynamic_schedule =
        !decode_force_static_schedule() && use_dynamic_schedule(n, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q4k(x, row_ptr, blocks_per_row);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q4k(x, row_ptr, blocks_per_row);
        }
    }
}

int simd_repack_q4k_r4(const void* src_q4k, void* dst_q4k_r4, int k, int n) {
    return repack_rows_r4_generic<144>(src_q4k, dst_q4k_r4, k, n);
}

int simd_repack_q6k_r4(const void* src_q6k, void* dst_q6k_r4, int k, int n) {
    return repack_rows_r4_generic<210>(src_q6k, dst_q6k_r4, k, n);
}

int simd_repack_q6k_lm(const void* src_q6k, void* dst_q6k_lm, int k, int n) {
    return repack_rows_q6k_lm(src_q6k, dst_q6k_lm, k, n);
}

void simd_matvec_q4k_r4(const float* x, const void* a_quant_r4, float* y, int k, int n) {
    if (x == nullptr || a_quant_r4 == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const int full_groups = n / kRowsPerGroup;
    const int tail_rows = n - full_groups * kRowsPerGroup;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant_r4);
    const int n_threads = get_num_threads(true);

    const bool dynamic_schedule =
        !decode_force_static_schedule()
        && use_dynamic_schedule(full_groups, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const auto* group_ptr = data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q4k_r4(x, group_ptr, blocks_per_row, out4);
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const auto* group_ptr = data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q4k_r4(x, group_ptr, blocks_per_row, out4);
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    }

    if (tail_rows > 0) {
        const std::size_t tail_base =
            static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
        const auto* tail_ptr = data + tail_base;
        for (int t = 0; t < tail_rows; ++t) {
            const auto* row_ptr =
                tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
            y[full_groups * kRowsPerGroup + t] = dot_row_q4k(x, row_ptr, blocks_per_row);
        }
    }
}

void simd_matvec_q6k(const float* x, const void* a_quant, float* y, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(true);
    const int full_groups = n / kRowsPerGroup;
    const int tail_rows = n - full_groups * kRowsPerGroup;

    const bool dynamic_schedule =
        !decode_force_static_schedule()
        && use_dynamic_schedule(full_groups, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
            const int row_base = g * kRowsPerGroup;
            const auto* row_ptr0 = data + static_cast<std::size_t>(row_base + 0) * row_bytes;
            const auto* row_ptr1 = data + static_cast<std::size_t>(row_base + 1) * row_bytes;
            const auto* row_ptr2 = data + static_cast<std::size_t>(row_base + 2) * row_bytes;
            const auto* row_ptr3 = data + static_cast<std::size_t>(row_base + 3) * row_bytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q6k(x, row_ptr0, row_ptr1, row_ptr2, row_ptr3, blocks_per_row, out4);
            y[row_base + 0] = out4[0];
            y[row_base + 1] = out4[1];
            y[row_base + 2] = out4[2];
            y[row_base + 3] = out4[3];
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
            const int row_base = g * kRowsPerGroup;
            const auto* row_ptr0 = data + static_cast<std::size_t>(row_base + 0) * row_bytes;
            const auto* row_ptr1 = data + static_cast<std::size_t>(row_base + 1) * row_bytes;
            const auto* row_ptr2 = data + static_cast<std::size_t>(row_base + 2) * row_bytes;
            const auto* row_ptr3 = data + static_cast<std::size_t>(row_base + 3) * row_bytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q6k(x, row_ptr0, row_ptr1, row_ptr2, row_ptr3, blocks_per_row, out4);
            y[row_base + 0] = out4[0];
            y[row_base + 1] = out4[1];
            y[row_base + 2] = out4[2];
            y[row_base + 3] = out4[3];
        }
    }

    if (tail_rows > 0) {
        const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
        const int row_base = full_groups * kRowsPerGroup;
        for (int t = 0; t < tail_rows; ++t) {
            const auto* row_ptr = data + static_cast<std::size_t>(row_base + t) * row_bytes;
            y[row_base + t] = dot_row_q6k(x, row_ptr, blocks_per_row);
        }
    }
}

void simd_matvec_q6k_r4(const float* x, const void* a_quant_r4, float* y, int k, int n) {
    if (x == nullptr || a_quant_r4 == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const int full_groups = n / kRowsPerGroup;
    const int tail_rows = n - full_groups * kRowsPerGroup;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant_r4);
    const int n_threads = get_num_threads(true);

    const bool dynamic_schedule =
        !decode_force_static_schedule()
        && use_dynamic_schedule(full_groups, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const auto* group_ptr =
                data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q6k_r4(x, group_ptr, blocks_per_row, out4);
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(true);
            const auto* group_ptr =
                data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q6k_r4(x, group_ptr, blocks_per_row, out4);
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    }

    if (tail_rows > 0) {
        const std::size_t tail_base =
            static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
        const auto* tail_ptr = data + tail_base;
        for (int t = 0; t < tail_rows; ++t) {
            const auto* row_ptr =
                tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
            y[full_groups * kRowsPerGroup + t] = dot_row_q6k(x, row_ptr, blocks_per_row);
        }
    }
}

void simd_matvec_q6k_lm(const float* x, const void* a_quant_lm, float* y, int k, int n) {
    if (x == nullptr || a_quant_lm == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 276;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant_lm);
    const int n_threads = get_num_threads(true);

    const bool dynamic_schedule =
        !decode_force_static_schedule() && use_dynamic_schedule(n, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q6k_lm(x, row_ptr, blocks_per_row);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q6k_lm(x, row_ptr, blocks_per_row);
        }
    }
}

void simd_matvec_q8_0(const float* x, const void* a_quant, float* y, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 32;
    constexpr int kBlockBytes = 34;
    if ((k % kBlock) != 0) {
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(true);

    const bool dynamic_schedule =
        !decode_force_static_schedule() && use_dynamic_schedule(n, n_threads);
    if (dynamic_schedule) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, 1) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q8_0(x, row_ptr, blocks_per_row);
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(use_row_parallelism(n, blocks_per_row, n_threads))
#endif
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(true);
            const auto* row_ptr = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[row] = dot_row_q8_0(x, row_ptr, blocks_per_row);
        }
    }
}

int simd_fused_qkv_matvec_quant(
    const float* x,
    const void* q_data, int q_qtype, float* q_out, int q_rows,
    const void* k_data, int k_qtype, float* k_out, int k_rows,
    const void* v_data, int v_qtype, float* v_out, int v_rows,
    int cols
) {
    if (
        x == nullptr || q_data == nullptr || k_data == nullptr || v_data == nullptr
        || q_out == nullptr || k_out == nullptr || v_out == nullptr
        || q_rows <= 0 || k_rows <= 0 || v_rows <= 0 || cols <= 0
    ) {
        return 0;
    }
    auto supported = [](int qtype) {
        return qtype == QTYPE_Q4_K
            || qtype == QTYPE_Q4_K_R4
            || qtype == QTYPE_Q6_K
            || qtype == QTYPE_Q6_K_R4;
    };
    if (!(supported(q_qtype) && supported(k_qtype) && supported(v_qtype))) {
        return 0;
    }

    const int n_threads = get_num_threads(true);
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const int nth = std::max(1, omp_get_num_threads());
        maybe_bind_worker_thread(true);
        compute_quant_matvec_chunk(x, q_data, q_qtype, q_out, cols, q_rows, tid, nth);
        compute_quant_matvec_chunk(x, k_data, k_qtype, k_out, cols, k_rows, tid, nth);
        compute_quant_matvec_chunk(x, v_data, v_qtype, v_out, cols, v_rows, tid, nth);
    }
#else
    compute_quant_matvec_chunk(x, q_data, q_qtype, q_out, cols, q_rows, 0, 1);
    compute_quant_matvec_chunk(x, k_data, k_qtype, k_out, cols, k_rows, 0, 1);
    compute_quant_matvec_chunk(x, v_data, v_qtype, v_out, cols, v_rows, 0, 1);
#endif
    return 1;
}

int simd_fused_quad_matvec_quant(
    const float* x,
    const void* a_data, int a_qtype, float* a_out, int a_rows,
    const void* b_data, int b_qtype, float* b_out, int b_rows,
    const void* c_data, int c_qtype, float* c_out, int c_rows,
    const void* d_data, int d_qtype, float* d_out, int d_rows,
    int cols
) {
    if (
        x == nullptr || a_data == nullptr || b_data == nullptr || c_data == nullptr || d_data == nullptr
        || a_out == nullptr || b_out == nullptr || c_out == nullptr || d_out == nullptr
        || a_rows <= 0 || b_rows <= 0 || c_rows <= 0 || d_rows <= 0 || cols <= 0
    ) {
        return 0;
    }
    auto supported = [](int qtype) {
        return qtype == QTYPE_Q4_K
            || qtype == QTYPE_Q4_K_R4
            || qtype == QTYPE_Q6_K
            || qtype == QTYPE_Q6_K_R4;
    };
    if (!(supported(a_qtype) && supported(b_qtype) && supported(c_qtype) && supported(d_qtype))) {
        return 0;
    }

    const int n_threads = get_num_threads(true);
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
    {
        const int tid = omp_get_thread_num();
        const int nth = std::max(1, omp_get_num_threads());
        maybe_bind_worker_thread(true);
        compute_quant_matvec_chunk(x, a_data, a_qtype, a_out, cols, a_rows, tid, nth);
        compute_quant_matvec_chunk(x, b_data, b_qtype, b_out, cols, b_rows, tid, nth);
        compute_quant_matvec_chunk(x, c_data, c_qtype, c_out, cols, c_rows, tid, nth);
        compute_quant_matvec_chunk(x, d_data, d_qtype, d_out, cols, d_rows, tid, nth);
    }
#else
    compute_quant_matvec_chunk(x, a_data, a_qtype, a_out, cols, a_rows, 0, 1);
    compute_quant_matvec_chunk(x, b_data, b_qtype, b_out, cols, b_rows, 0, 1);
    compute_quant_matvec_chunk(x, c_data, c_qtype, c_out, cols, c_rows, 0, 1);
    compute_quant_matvec_chunk(x, d_data, d_qtype, d_out, cols, d_rows, 0, 1);
#endif
    return 1;
}

void simd_matmul_q4k(const float* x, const void* a_quant, float* y, int m, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    if ((k % kBlock) != 0) {
        return;
    }
    if (m == 1) {
        simd_matvec_q4k(x, a_quant, y, k, n);
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(false);

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, n, blocks_per_row, n_threads))
#endif
    for (int i = 0; i < m; ++i) {
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(false);
            const float* x_row = x + static_cast<std::size_t>(i) * k;
            const auto* w_row = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[static_cast<std::size_t>(i) * n + row] = dot_row_q4k(x_row, w_row, blocks_per_row);
        }
    }
}

void simd_matmul_q4k_r4(const float* x, const void* a_quant_r4, float* y, int m, int k, int n) {
    if (x == nullptr || a_quant_r4 == nullptr || y == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 144;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return;
    }
    if (m == 1) {
        simd_matvec_q4k_r4(x, a_quant_r4, y, k, n);
        return;
    }

    const int blocks_per_row = k / kBlock;
    const int full_groups = n / kRowsPerGroup;
    const int tail_rows = n - full_groups * kRowsPerGroup;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant_r4);
    const int n_threads = get_num_threads(false);

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
    for (int i = 0; i < m; ++i) {
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(false);
            const float* x_row = x + static_cast<std::size_t>(i) * k;
            const auto* group_ptr = data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q4k_r4(x_row, group_ptr, blocks_per_row, out4);
            float* y_row = y + static_cast<std::size_t>(i) * n;
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    }

    if (tail_rows > 0) {
        const std::size_t tail_base =
            static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
        const auto* tail_ptr = data + tail_base;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, tail_rows, blocks_per_row, n_threads))
#endif
        for (int i = 0; i < m; ++i) {
            for (int t = 0; t < tail_rows; ++t) {
                maybe_bind_worker_thread(false);
                const float* x_row = x + static_cast<std::size_t>(i) * k;
                const auto* w_row =
                    tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
                y[static_cast<std::size_t>(i) * n + full_groups * kRowsPerGroup + t] =
                    dot_row_q4k(x_row, w_row, blocks_per_row);
            }
        }
    }
}

void simd_matmul_q6k(const float* x, const void* a_quant, float* y, int m, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;
    if ((k % kBlock) != 0) {
        return;
    }
    if (m == 1) {
        simd_matvec_q6k(x, a_quant, y, k, n);
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(false);

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, n, blocks_per_row, n_threads))
#endif
    for (int i = 0; i < m; ++i) {
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(false);
            const float* x_row = x + static_cast<std::size_t>(i) * k;
            const auto* w_row = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[static_cast<std::size_t>(i) * n + row] = dot_row_q6k(x_row, w_row, blocks_per_row);
        }
    }
}

void simd_matmul_q6k_r4(const float* x, const void* a_quant_r4, float* y, int m, int k, int n) {
    if (x == nullptr || a_quant_r4 == nullptr || y == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 256;
    constexpr int kBlockBytes = 210;
    constexpr int kRowsPerGroup = 4;
    if ((k % kBlock) != 0) {
        return;
    }
    if (m == 1) {
        simd_matvec_q6k_r4(x, a_quant_r4, y, k, n);
        return;
    }

    const int blocks_per_row = k / kBlock;
    const int full_groups = n / kRowsPerGroup;
    const int tail_rows = n - full_groups * kRowsPerGroup;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant_r4);
    const int n_threads = get_num_threads(false);

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, full_groups, blocks_per_row * kRowsPerGroup, n_threads))
#endif
    for (int i = 0; i < m; ++i) {
        for (int g = 0; g < full_groups; ++g) {
            maybe_bind_worker_thread(false);
            const float* x_row = x + static_cast<std::size_t>(i) * k;
            const auto* group_ptr =
                data + static_cast<std::size_t>(g) * blocks_per_row * kRowsPerGroup * kBlockBytes;
            float out4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            dot_rows_q6k_r4(x_row, group_ptr, blocks_per_row, out4);
            float* y_row = y + static_cast<std::size_t>(i) * n;
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 0] = out4[0];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 1] = out4[1];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 2] = out4[2];
            y_row[static_cast<std::size_t>(g) * kRowsPerGroup + 3] = out4[3];
        }
    }

    if (tail_rows > 0) {
        const std::size_t tail_base =
            static_cast<std::size_t>(full_groups) * blocks_per_row * kRowsPerGroup * kBlockBytes;
        const auto* tail_ptr = data + tail_base;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, tail_rows, blocks_per_row, n_threads))
#endif
        for (int i = 0; i < m; ++i) {
            for (int t = 0; t < tail_rows; ++t) {
                maybe_bind_worker_thread(false);
                const float* x_row = x + static_cast<std::size_t>(i) * k;
                const auto* w_row =
                    tail_ptr + static_cast<std::size_t>(t) * blocks_per_row * kBlockBytes;
                y[static_cast<std::size_t>(i) * n + full_groups * kRowsPerGroup + t] =
                    dot_row_q6k(x_row, w_row, blocks_per_row);
            }
        }
    }
}

void simd_matmul_q8_0(const float* x, const void* a_quant, float* y, int m, int k, int n) {
    if (x == nullptr || a_quant == nullptr || y == nullptr || m <= 0 || k <= 0 || n <= 0) {
        return;
    }
    constexpr int kBlock = 32;
    constexpr int kBlockBytes = 34;
    if ((k % kBlock) != 0) {
        return;
    }
    if (m == 1) {
        simd_matvec_q8_0(x, a_quant, y, k, n);
        return;
    }

    const int blocks_per_row = k / kBlock;
    const auto* data = reinterpret_cast<const uint8_t*>(a_quant);
    const int n_threads = get_num_threads(false);

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2) if(use_batch_row_parallelism(m, n, blocks_per_row, n_threads))
#endif
    for (int i = 0; i < m; ++i) {
        for (int row = 0; row < n; ++row) {
            maybe_bind_worker_thread(false);
            const float* x_row = x + static_cast<std::size_t>(i) * k;
            const auto* w_row = data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            y[static_cast<std::size_t>(i) * n + row] = dot_row_q8_0(x_row, w_row, blocks_per_row);
        }
    }
}

// ============================================================
// Fused SwiGLU expert FFN: gate_proj + up_proj + silu(gate)*up + down_proj
// Eliminates Python round-trips for the hot MoE expert path.
// ============================================================

/**
 * Dispatch quantized matvec by qtype. Returns false if unknown qtype.
 */
static bool dispatch_matvec(
    const float* x, const void* quant_data, float* y,
    int k, int n, int qtype
) {
    switch (qtype) {
        case 12: simd_matvec_q4k(x, quant_data, y, k, n); return true;
        case QTYPE_Q4_K_R4: simd_matvec_q4k_r4(x, quant_data, y, k, n); return true;
        case 14: simd_matvec_q6k(x, quant_data, y, k, n); return true;
        case QTYPE_Q6_K_R4: simd_matvec_q6k_r4(x, quant_data, y, k, n); return true;
        case QTYPE_Q6_K_LM: simd_matvec_q6k_lm(x, quant_data, y, k, n); return true;
        case 8:  simd_matvec_q8_0(x, quant_data, y, k, n); return true;
        default: return false;
    }
}

int simd_dequantize_row(
    const void* quant_data,
    int qtype,
    int row,
    float* out,
    int k
) {
    if (quant_data == nullptr || out == nullptr || row < 0 || k <= 0) {
        return 0;
    }

    const auto* data = reinterpret_cast<const uint8_t*>(quant_data);
    switch (qtype) {
        case QTYPE_Q4_K: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 144;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const auto* row_ptr =
                data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            dequantize_row_q4k(row_ptr, out, blocks_per_row);
            return 1;
        }
        case QTYPE_Q4_K_R4: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 144;
            constexpr int kRowsPerGroup = 4;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const int group = row / kRowsPerGroup;
            const int row_in_group = row % kRowsPerGroup;
            const auto* group_ptr =
                data
                + static_cast<std::size_t>(group)
                    * blocks_per_row
                    * kRowsPerGroup
                    * kBlockBytes;
            dequantize_row_q4k_r4(group_ptr, row_in_group, out, blocks_per_row);
            return 1;
        }
        case QTYPE_Q6_K: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 210;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const auto* row_ptr =
                data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            dequantize_row_q6k(row_ptr, out, blocks_per_row);
            return 1;
        }
        case QTYPE_Q6_K_R4: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 210;
            constexpr int kRowsPerGroup = 4;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const int group = row / kRowsPerGroup;
            const int row_in_group = row % kRowsPerGroup;
            const auto* group_ptr =
                data
                + static_cast<std::size_t>(group)
                    * blocks_per_row
                    * kRowsPerGroup
                    * kBlockBytes;
            dequantize_row_q6k_r4(group_ptr, row_in_group, out, blocks_per_row);
            return 1;
        }
        case QTYPE_Q6_K_LM: {
            constexpr int kBlock = 256;
            constexpr int kBlockBytes = 276;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const auto* row_ptr =
                data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            for (int b = 0; b < blocks_per_row; ++b) {
                const auto* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;
                const auto* q = reinterpret_cast<const int8_t*>(blk);
                const auto* scales = reinterpret_cast<const int8_t*>(blk + 256);
                float d = 0.0f;
                std::memcpy(&d, blk + 272, sizeof(float));
                const int out_off = b * kBlock;
                for (int scale_idx = 0; scale_idx < 16; ++scale_idx) {
                    const float scale = d * static_cast<float>(scales[scale_idx]);
                    const auto* q16 = q + scale_idx * 16;
                    for (int i = 0; i < 16; ++i) {
                        out[out_off + scale_idx * 16 + i] =
                            scale * static_cast<float>(q16[i]);
                    }
                }
            }
            return 1;
        }
        case QTYPE_Q8_0: {
            constexpr int kBlock = 32;
            constexpr int kBlockBytes = 34;
            if ((k % kBlock) != 0) {
                return 0;
            }
            const int blocks_per_row = k / kBlock;
            const auto* row_ptr =
                data + static_cast<std::size_t>(row) * blocks_per_row * kBlockBytes;
            dequantize_row_q8_0(row_ptr, out, blocks_per_row);
            return 1;
        }
        default:
            return 0;
    }
}

/**
 * Fused single-expert SwiGLU FFN:
 *   gate = gate_w @ x       [hidden_dim]
 *   up   = up_w   @ x       [hidden_dim]
 *   hidden = SiLU(gate) * up [hidden_dim]
 *   out  = down_w @ hidden   [out_dim]
 *
 * All weight matrices are quantized. Eliminates 5 Python ctypes calls.
 */
void simd_fused_expert_swiglu(
    const float* x, int in_dim,
    const void* gate_data, int gate_qtype, int hidden_dim,
    const void* up_data, int up_qtype,
    const void* down_data, int down_qtype, int out_dim,
    float* output
) {
    if (x == nullptr || output == nullptr || in_dim <= 0 || hidden_dim <= 0 || out_dim <= 0) return;

    thread_local AlignedFloatBuffer gate_tls;
    thread_local AlignedFloatBuffer up_tls;
    gate_tls.resize(static_cast<std::size_t>(hidden_dim));
    up_tls.resize(static_cast<std::size_t>(hidden_dim));
    float* gate_buf = gate_tls.data();
    float* up_buf = up_tls.data();

    auto supports_fused_projection_qtype = [](int qtype) {
        return qtype == QTYPE_Q4_K
            || qtype == QTYPE_Q4_K_R4
            || qtype == QTYPE_Q6_K
            || qtype == QTYPE_Q6_K_R4
            || qtype == QTYPE_Q6_K_LM
            || qtype == QTYPE_Q8_0;
    };

    if (
        gate_data != nullptr
        && up_data != nullptr
        && supports_fused_projection_qtype(gate_qtype)
        && supports_fused_projection_qtype(up_qtype)
    ) {
        const int n_threads = get_num_threads(true);
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
        {
            const int tid = omp_get_thread_num();
            const int nth = std::max(1, omp_get_num_threads());
            compute_quant_matvec_chunk(
                x,
                gate_data,
                gate_qtype,
                gate_buf,
                in_dim,
                hidden_dim,
                tid,
                nth
            );
            compute_quant_matvec_chunk(
                x,
                up_data,
                up_qtype,
                up_buf,
                in_dim,
                hidden_dim,
                tid,
                nth
            );
        }
#else
        compute_quant_matvec_chunk(x, gate_data, gate_qtype, gate_buf, in_dim, hidden_dim, 0, 1);
        compute_quant_matvec_chunk(x, up_data, up_qtype, up_buf, in_dim, hidden_dim, 0, 1);
#endif
    } else {
        // Fallback keeps support for float or unsupported quant layouts.
        if (!dispatch_matvec(x, gate_data, gate_buf, in_dim, hidden_dim, gate_qtype)) {
            return;
        }
        if (!dispatch_matvec(x, up_data, up_buf, in_dim, hidden_dim, up_qtype)) {
            return;
        }
    }

    // 3. SwiGLU: SiLU(gate) * up — fused in-place into gate_buf
#ifdef __AVX2__
    {
        int i = 0;
        const int vec_end = hidden_dim - (hidden_dim % 8);
        for (; i < vec_end; i += 8) {
            __m256 gv = _mm256_loadu_ps(gate_buf + i);
            __m256 uv = _mm256_loadu_ps(up_buf + i);
            __m256 silu = anvil_fast_math::v_silu(gv);
            __m256 result = _mm256_mul_ps(silu, uv);
            _mm256_storeu_ps(gate_buf + i, result);
        }
        if (i < hidden_dim) {
            int mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const int tail = hidden_dim - i;
            for (int j = 0; j < tail; ++j) {
                mask[j] = -1;
            }
            const __m256i tail_mask = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(mask)
            );
            __m256 gv = _mm256_maskload_ps(gate_buf + i, tail_mask);
            __m256 uv = _mm256_maskload_ps(up_buf + i, tail_mask);
            __m256 silu = anvil_fast_math::v_silu(gv);
            __m256 result = _mm256_mul_ps(silu, uv);
            _mm256_maskstore_ps(gate_buf + i, tail_mask, result);
        }
    }
#else
    for (int i = 0; i < hidden_dim; ++i) {
        const float g = gate_buf[i];
        gate_buf[i] = (g / (1.0f + anvil_fast_math::fast_exp_scalar(-g))) * up_buf[i];
    }
#endif

    // 4. Down projection
    dispatch_matvec(gate_buf, down_data, output, hidden_dim, out_dim, down_qtype);
}

/**
 * Fused multi-expert MoE FFN.
 * Evaluates top_k experts and accumulates weighted results.
 *
 * expert_indices:  [top_k] indices into the expert arrays
 * expert_weights:  [top_k] softmax weights
 * gate/up/down_data_ptrs: [n_experts] pointers to quantized data for each expert
 */
void simd_fused_moe_ffn(
    const float* x, int in_dim,
    const int* expert_indices, const float* expert_weights, int top_k,
    const void* const* gate_data_ptrs, int gate_qtype, int hidden_dim,
    const void* const* up_data_ptrs, int up_qtype,
    const void* const* down_data_ptrs, int down_qtype, int out_dim,
    float* output,
    int write_mode
) {
    if (x == nullptr || output == nullptr || top_k <= 0) return;

    thread_local AlignedFloatBuffer expert_out_tls;
    expert_out_tls.resize(static_cast<std::size_t>(out_dim));
    float* expert_out = expert_out_tls.data();
    bool have_output = write_mode != 0;
    if (!have_output) {
        std::memset(output, 0, static_cast<std::size_t>(out_dim) * sizeof(float));
    }

    for (int e = 0; e < top_k; ++e) {
        const int idx = expert_indices[e];
        const float w = expert_weights[e];
        if (w <= 0.0f) continue;
        if (e + 1 < top_k) {
            const int next_idx = expert_indices[e + 1];
            if (next_idx >= 0) {
                prefetch_read(gate_data_ptrs[next_idx]);
                prefetch_read(up_data_ptrs[next_idx]);
                prefetch_read(down_data_ptrs[next_idx]);
            }
        }

        const void* g_data = gate_data_ptrs[idx];
        const void* u_data = up_data_ptrs[idx];
        const void* d_data = down_data_ptrs[idx];
        if (g_data == nullptr || u_data == nullptr || d_data == nullptr) continue;

        if (!have_output) {
            simd_fused_expert_swiglu(
                x, in_dim,
                g_data, gate_qtype, hidden_dim,
                u_data, up_qtype,
                d_data, down_qtype, out_dim,
                output
            );
            if (w != 1.0f) {
#ifdef __AVX2__
                __m256 wv = _mm256_set1_ps(w);
                int i = 0;
                for (; i + 8 <= out_dim; i += 8) {
                    __m256 ov = _mm256_loadu_ps(output + i);
                    ov = _mm256_mul_ps(ov, wv);
                    _mm256_storeu_ps(output + i, ov);
                }
                for (; i < out_dim; ++i) {
                    output[i] *= w;
                }
#else
                for (int i = 0; i < out_dim; ++i) {
                    output[i] *= w;
                }
#endif
            }
            have_output = true;
            continue;
        }

        simd_fused_expert_swiglu(
            x, in_dim,
            g_data, gate_qtype, hidden_dim,
            u_data, up_qtype,
            d_data, down_qtype, out_dim,
            expert_out
        );

        // Accumulate: output += w * expert_out
#ifdef __AVX2__
        {
            __m256 wv = _mm256_set1_ps(w);
            int i = 0;
            for (; i + 8 <= out_dim; i += 8) {
                __m256 ov = _mm256_loadu_ps(output + i);
                __m256 ev = _mm256_loadu_ps(expert_out + i);
                ov = _mm256_fmadd_ps(wv, ev, ov);
                _mm256_storeu_ps(output + i, ov);
            }
            for (; i < out_dim; ++i) {
                output[i] += w * expert_out[i];
            }
        }
#else
        for (int i = 0; i < out_dim; ++i) {
            output[i] += w * expert_out[i];
        }
#endif
    }
    if (!have_output && write_mode == 0) {
        std::memset(output, 0, static_cast<std::size_t>(out_dim) * sizeof(float));
    }
}

}  // extern "C"
