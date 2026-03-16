/**
 * tinyBLAS-inspired kernels for LLM decode matvec hot paths.
 *
 * Key techniques from llamafile (Justine Tunney):
 *  1. Outer-loop unrolling: compute 4 output rows simultaneously
 *  2. Register blocking: keep intermediates in AVX2 registers
 *  3. L2-resident: optimized for LLM decode shapes that fit in cache
 *  4. Zero-copy from quantized: dequant + FMA in same instruction stream
 *
 * For LLM decode (M=1, batch-1 matvec):
 *  - Attention projection: 1×2048 × 2048×2048 — fits L2
 *  - FFN up/gate: 1×2048 × 2048×8192 — borderline L2
 *  - FFN down: 1×8192 × 8192×2048 — streaming
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline int get_num_threads() {
    const char* env = std::getenv("ANVIL_NUM_THREADS");
    if (env) {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

inline float fp16_to_fp32(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h) & 0x8000u) << 16;
    const std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x03FFu;
    const int exp = static_cast<int>((h >> 10) & 0x1Fu);

    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::uint32_t norm_mant = mant;
            int norm_exp = -14;
            while ((norm_mant & 0x0400u) == 0) {
                norm_mant <<= 1;
                --norm_exp;
            }
            norm_mant &= 0x03FFu;
            bits = sign
                | (static_cast<std::uint32_t>(norm_exp + 127) << 23)
                | (norm_mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign
            | (static_cast<std::uint32_t>(exp - 15 + 127) << 23)
            | (mant << 13);
    }

    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(float));
    return out;
}

inline float dot_q4k_row(
    const std::uint8_t* row_ptr,
    const float* x,
    int blocks_per_row
) {
    constexpr int kBlockBytes = 144;
    constexpr int kSub = 32;

    float acc = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const std::uint8_t* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;

        std::uint16_t d16 = 0;
        std::uint16_t dmin16 = 0;
        std::memcpy(&d16, blk, sizeof(std::uint16_t));
        std::memcpy(&dmin16, blk + 2, sizeof(std::uint16_t));

        const float d = fp16_to_fp32(d16);
        const float dmin = fp16_to_fp32(dmin16);

        const std::uint8_t* scales = blk + 4;
        const std::uint8_t* qs = blk + 16;

        float d_sc[8];
        float d_min[8];
        for (int j = 0; j < 8; ++j) {
            std::uint8_t sc = 0;
            std::uint8_t mn = 0;
            if (j < 4) {
                sc = static_cast<std::uint8_t>(scales[j] & 0x3F);
                mn = static_cast<std::uint8_t>(scales[4 + j] & 0x3F);
            } else {
                const int idx = j - 4;
                sc = static_cast<std::uint8_t>((scales[8 + idx] & 0x0F) | ((scales[idx] >> 2) & 0x30));
                mn = static_cast<std::uint8_t>((scales[8 + idx] >> 4) | ((scales[4 + idx] >> 2) & 0x30));
            }
            d_sc[j] = d * static_cast<float>(sc);
            d_min[j] = dmin * static_cast<float>(mn);
        }

        for (int j = 0; j < 8; ++j) {
            const int pair = j / 2;
            const bool high = (j & 1) != 0;
            const int byte_base = pair * 32;
            const int x_off = b * 256 + j * kSub;
            const float scale_v = d_sc[j];
            const float min_v = d_min[j];

            for (int i = 0; i < kSub; ++i) {
                const std::uint8_t packed = qs[byte_base + i];
                const std::uint8_t q = high
                    ? static_cast<std::uint8_t>(packed >> 4)
                    : static_cast<std::uint8_t>(packed & 0x0F);
                acc += (scale_v * static_cast<float>(q) - min_v) * x[x_off + i];
            }
        }
    }

    return acc;
}

#ifdef __AVX2__
// AVX2-optimized dot product for float32
inline float avx2_dot_f32(const float* a, const float* b, int len) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    // Reduce
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float result = _mm_cvtss_f32(sum);
    for (; i < len; ++i)
        result += a[i] * b[i];
    return result;
}
#endif

}  // namespace

extern "C" {

// ===================================================================
// Float32 matvec with 4-row outer unrolling + AVX2 + OpenMP
// ===================================================================
void tinyblas_matvec_f32(const float* mat, const float* x, float* y, int rows, int cols) {
    if (mat == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
        return;
    }

    int n_threads = get_num_threads();

#ifdef __AVX2__
    // Outer-loop unrolled by 4: process 4 output rows simultaneously
    // This keeps x in registers while computing 4 dot products
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows - 3; r += 4) {
        const float* row0 = mat + static_cast<std::size_t>(r) * cols;
        const float* row1 = row0 + cols;
        const float* row2 = row1 + cols;
        const float* row3 = row2 + cols;

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 xv = _mm256_loadu_ps(x + c);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row0 + c), xv, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row1 + c), xv, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row2 + c), xv, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row3 + c), xv, acc3);
        }

        // Horizontal sums
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        {
            __m128 lo, hi, s;
            lo = _mm256_castps256_ps128(acc0); hi = _mm256_extractf128_ps(acc0, 1);
            s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            s0 = _mm_cvtss_f32(s);

            lo = _mm256_castps256_ps128(acc1); hi = _mm256_extractf128_ps(acc1, 1);
            s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            s1 = _mm_cvtss_f32(s);

            lo = _mm256_castps256_ps128(acc2); hi = _mm256_extractf128_ps(acc2, 1);
            s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            s2 = _mm_cvtss_f32(s);

            lo = _mm256_castps256_ps128(acc3); hi = _mm256_extractf128_ps(acc3, 1);
            s = _mm_add_ps(lo, hi); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
            s3 = _mm_cvtss_f32(s);
        }

        // Tail elements
        for (; c < cols; ++c) {
            float xv = x[c];
            s0 += row0[c] * xv;
            s1 += row1[c] * xv;
            s2 += row2[c] * xv;
            s3 += row3[c] * xv;
        }

        y[r] = s0;
        y[r + 1] = s1;
        y[r + 2] = s2;
        y[r + 3] = s3;
    }

    // Handle remaining rows (< 4)
    int rem_start = rows - (rows % 4);
    for (int r = rem_start; r < rows; ++r) {
        y[r] = avx2_dot_f32(mat + static_cast<std::size_t>(r) * cols, x, cols);
    }
#else
    // Scalar fallback with outer-loop unrolling
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows - 3; r += 4) {
        const float* row0 = mat + static_cast<std::size_t>(r) * cols;
        float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        for (int c = 0; c < cols; ++c) {
            float xv = x[c];
            acc0 += row0[c] * xv;
            acc1 += row0[cols + c] * xv;
            acc2 += row0[2 * cols + c] * xv;
            acc3 += row0[3 * cols + c] * xv;
        }
        y[r] = acc0;
        y[r + 1] = acc1;
        y[r + 2] = acc2;
        y[r + 3] = acc3;
    }
    for (int r = rows - (rows % 4); r < rows; ++r) {
        const float* row = mat + static_cast<std::size_t>(r) * cols;
        float acc = 0;
        for (int c = 0; c < cols; ++c)
            acc += row[c] * x[c];
        y[r] = acc;
    }
#endif
}

// ===================================================================
// Q4_K quantized matvec with 4-row outer unrolling + OpenMP
// ===================================================================
void tinyblas_matvec_q4k(const std::uint8_t* mat, const float* x, float* y, int rows, int cols) {
    if (mat == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    if ((cols % 256) != 0) {
        return;
    }

    constexpr int kBlockBytes = 144;
    const int blocks_per_row = cols / 256;
    const std::size_t row_stride = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;

    int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows - 3; r += 4) {
        const std::uint8_t* row0 = mat + static_cast<std::size_t>(r) * row_stride;
        const std::uint8_t* row1 = row0 + row_stride;
        const std::uint8_t* row2 = row1 + row_stride;
        const std::uint8_t* row3 = row2 + row_stride;

        y[r]     = dot_q4k_row(row0, x, blocks_per_row);
        y[r + 1] = dot_q4k_row(row1, x, blocks_per_row);
        y[r + 2] = dot_q4k_row(row2, x, blocks_per_row);
        y[r + 3] = dot_q4k_row(row3, x, blocks_per_row);
    }

    int rem_start = rows - (rows % 4);
    for (int r = rem_start; r < rows; ++r) {
        const std::uint8_t* row = mat + static_cast<std::size_t>(r) * row_stride;
        y[r] = dot_q4k_row(row, x, blocks_per_row);
    }
}

// ===================================================================
// Q8_0 quantized matvec with AVX2
// ===================================================================
void tinyblas_matvec_q8_0(
    const std::uint8_t* mat,
    const float* x,
    float* y,
    int rows,
    int cols
) {
    if (mat == nullptr || x == nullptr || y == nullptr || rows <= 0 || cols <= 0)
        return;
    if ((cols % 32) != 0)
        return;

    // Q8_0 block: 2 bytes scale (fp16) + 32 bytes int8 quants = 34 bytes
    constexpr int kBlockSize = 32;
    constexpr int kBlockBytes = 34;
    const int blocks_per_row = cols / kBlockSize;
    const std::size_t row_stride = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;

    int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        const std::uint8_t* row_ptr = mat + static_cast<std::size_t>(r) * row_stride;
        float acc = 0.0f;

        for (int b = 0; b < blocks_per_row; ++b) {
            const std::uint8_t* blk = row_ptr + static_cast<std::size_t>(b) * kBlockBytes;

            std::uint16_t d16 = 0;
            std::memcpy(&d16, blk, sizeof(std::uint16_t));
            float d = fp16_to_fp32(d16);

            const std::int8_t* quants = reinterpret_cast<const std::int8_t*>(blk + 2);
            int x_off = b * kBlockSize;

#ifdef __AVX2__
            // Load 32 int8 quants, convert to float, multiply by x, accumulate
            __m256i q_lo = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants)));
            __m256i q_hi = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants + 8)));
            __m256i q_lo2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants + 16)));
            __m256i q_hi2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(quants + 24)));

            __m256 d_vec = _mm256_set1_ps(d);
            __m256 a0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_lo), _mm256_loadu_ps(x + x_off));
            __m256 a1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_hi), _mm256_loadu_ps(x + x_off + 8));
            __m256 a2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_lo2), _mm256_loadu_ps(x + x_off + 16));
            __m256 a3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q_hi2), _mm256_loadu_ps(x + x_off + 24));

            __m256 sum = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
            sum = _mm256_mul_ps(sum, d_vec);

            // Horizontal sum
            __m128 lo = _mm256_castps256_ps128(sum);
            __m128 hi = _mm256_extractf128_ps(sum, 1);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_hadd_ps(s, s);
            s = _mm_hadd_ps(s, s);
            acc += _mm_cvtss_f32(s);
#else
            for (int i = 0; i < kBlockSize; ++i) {
                acc += d * static_cast<float>(quants[i]) * x[x_off + i];
            }
#endif
        }
        y[r] = acc;
    }
}

// ===================================================================
// Batched RoPE for all heads in one call
// ===================================================================
void tinyblas_batch_rope_f32(
    float* q, float* k,
    int n_heads, int n_kv_heads,
    int head_dim, int pos, float theta
) {
    if (q == nullptr || k == nullptr || n_heads <= 0 || head_dim <= 1)
        return;

    int rope_dim = head_dim - (head_dim % 2);

    auto rotate = [&](float* vec, int heads) {
#ifdef _OPENMP
        int n_threads = get_num_threads();
#pragma omp parallel for num_threads(n_threads) if(heads > 4) schedule(static)
#endif
        for (int h = 0; h < heads; ++h) {
            float* ptr = vec + static_cast<std::size_t>(h) * head_dim;
            for (int i = 0; i < rope_dim; i += 2) {
                float freq = 1.0f / std::pow(theta,
                    static_cast<float>(i) / static_cast<float>(std::max(head_dim, 1)));
                float angle = static_cast<float>(pos) * freq;
                float c = std::cos(angle);
                float s = std::sin(angle);
                float x0 = ptr[i], x1 = ptr[i + 1];
                ptr[i] = x0 * c - x1 * s;
                ptr[i + 1] = x0 * s + x1 * c;
            }
        }
    };

    rotate(q, n_heads);
    if (n_kv_heads > 0)
        rotate(k, n_kv_heads);
}

}  // extern "C"
