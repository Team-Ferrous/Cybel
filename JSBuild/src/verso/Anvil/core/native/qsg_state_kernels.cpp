#include "qsg_state_kernels.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

inline std::size_t row_offset(
    const QSGRowRef& ref,
    std::int32_t rows_per_page,
    std::int32_t dim) {
    return static_cast<std::size_t>(ref.row_idx % rows_per_page) *
           static_cast<std::size_t>(dim);
}

inline std::uint16_t float_to_half_bits(float value) {
    const std::uint32_t bits = *reinterpret_cast<const std::uint32_t*>(&value);
    const std::uint32_t sign = (bits >> 16U) & 0x8000U;
    std::uint32_t mantissa = bits & 0x007fffffU;
    int exponent = static_cast<int>((bits >> 23U) & 0xffU) - 127;

    if (exponent > 15) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    if (exponent >= -14) {
        exponent += 15;
        mantissa += 0x00001000U;
        if (mantissa & 0x00800000U) {
            mantissa = 0;
            ++exponent;
        }
        if (exponent > 30) {
            return static_cast<std::uint16_t>(sign | 0x7c00U);
        }
        return static_cast<std::uint16_t>(
            sign | (static_cast<std::uint32_t>(exponent) << 10U) | (mantissa >> 13U));
    }
    if (exponent < -24) {
        return static_cast<std::uint16_t>(sign);
    }

    mantissa |= 0x00800000U;
    const std::uint32_t shift = static_cast<std::uint32_t>(-exponent - 14);
    mantissa = (mantissa + (1U << (shift + 12U))) >> (shift + 13U);
    return static_cast<std::uint16_t>(sign | mantissa);
}

inline float half_bits_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1fU;
    std::uint32_t mantissa = value & 0x03ffU;
    std::uint32_t bits = 0U;

    if (exponent == 0U) {
        if (mantissa != 0U) {
            exponent = 1U;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03ffU;
            bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
        } else {
            bits = sign;
        }
    } else if (exponent == 0x1fU) {
        bits = sign | 0x7f800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }

    return *reinterpret_cast<float*>(&bits);
}

}  // namespace

extern "C" {

void qsg_state_gather_rows(
    float* dst,
    const float* const* pages,
    const QSGRowRef* refs,
    std::int32_t n_refs,
    std::int32_t rows_per_page,
    std::int32_t dim) {
    if (!dst || !pages || !refs || n_refs <= 0 || dim <= 0 || rows_per_page <= 0) {
        return;
    }
    for (std::int32_t i = 0; i < n_refs; ++i) {
        const QSGRowRef ref = refs[i];
        const float* page = pages[ref.page_id];
        if (!page) {
            continue;
        }
        const std::size_t src_idx = row_offset(ref, rows_per_page, dim);
        const std::size_t dst_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);
        std::memcpy(dst + dst_idx, page + src_idx, static_cast<std::size_t>(dim) * sizeof(float));
    }
}

void qsg_state_scatter_rows(
    const float* src,
    float* const* pages,
    const QSGRowRef* refs,
    std::int32_t n_refs,
    std::int32_t rows_per_page,
    std::int32_t dim) {
    if (!src || !pages || !refs || n_refs <= 0 || dim <= 0 || rows_per_page <= 0) {
        return;
    }
    for (std::int32_t i = 0; i < n_refs; ++i) {
        const QSGRowRef ref = refs[i];
        float* page = pages[ref.page_id];
        if (!page) {
            continue;
        }
        const std::size_t src_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(dim);
        const std::size_t dst_idx = row_offset(ref, rows_per_page, dim);
        std::memcpy(page + dst_idx, src + src_idx, static_cast<std::size_t>(dim) * sizeof(float));
    }
}

void qsg_state_clone_cow(
    float* const* pages,
    const QSGRowRef* src_refs,
    const QSGRowRef* dst_refs,
    std::int32_t n_rows,
    std::int32_t rows_per_page,
    std::int32_t dim) {
    if (!pages || !src_refs || !dst_refs || n_rows <= 0) {
        return;
    }
    for (std::int32_t i = 0; i < n_rows; ++i) {
        const QSGRowRef src_ref = src_refs[i];
        const QSGRowRef dst_ref = dst_refs[i];
        const float* src_page = pages[src_ref.page_id];
        float* dst_page = pages[dst_ref.page_id];
        if (!src_page || !dst_page) {
            continue;
        }
        const std::size_t src_idx = row_offset(src_ref, rows_per_page, dim);
        const std::size_t dst_idx = row_offset(dst_ref, rows_per_page, dim);
        std::memcpy(dst_page + dst_idx, src_page + src_idx, static_cast<std::size_t>(dim) * sizeof(float));
    }
}

void qsg_state_compact(
    float* const* pages,
    const QSGCompactMove* moves,
    std::int32_t n_moves,
    std::int32_t rows_per_page,
    std::int32_t dim) {
    if (!pages || !moves || n_moves <= 0) {
        return;
    }
    for (std::int32_t i = 0; i < n_moves; ++i) {
        const QSGCompactMove move = moves[i];
        const float* src_page = pages[move.src.page_id];
        float* dst_page = pages[move.dst.page_id];
        if (!src_page || !dst_page) {
            continue;
        }
        const std::size_t src_idx = row_offset(move.src, rows_per_page, dim);
        const std::size_t dst_idx = row_offset(move.dst, rows_per_page, dim);
        std::memmove(dst_page + dst_idx, src_page + src_idx, static_cast<std::size_t>(dim) * sizeof(float));
    }
}

void qsg_state_weighted_merge(
    float* dst,
    const float* src,
    const float* weights,
    std::int32_t n_rows,
    std::int32_t dim) {
    if (!dst || !src || !weights || n_rows <= 0 || dim <= 0) {
        return;
    }

    std::fill(dst, dst + dim, 0.0f);
    float weight_sum = 0.0f;
    constexpr std::int32_t kVecWidth = 8;

    for (std::int32_t row = 0; row < n_rows; ++row) {
        const float weight = weights[row];
        if (weight == 0.0f) {
            continue;
        }
        weight_sum += weight;
        const float* src_row = src + static_cast<std::size_t>(row) * static_cast<std::size_t>(dim);
        const __m256 weight_vec = _mm256_set1_ps(weight);
        std::int32_t col = 0;
        for (; col + kVecWidth <= dim; col += kVecWidth) {
            __m256 acc = _mm256_loadu_ps(dst + col);
            const __m256 values = _mm256_loadu_ps(src_row + col);
            acc = _mm256_fmadd_ps(values, weight_vec, acc);
            _mm256_storeu_ps(dst + col, acc);
        }
        for (; col < dim; ++col) {
            dst[col] += src_row[col] * weight;
        }
    }

    if (weight_sum == 0.0f) {
        return;
    }

    const __m256 inv_weight = _mm256_set1_ps(1.0f / weight_sum);
    std::int32_t col = 0;
    for (; col + kVecWidth <= dim; col += kVecWidth) {
        __m256 acc = _mm256_loadu_ps(dst + col);
        acc = _mm256_mul_ps(acc, inv_weight);
        _mm256_storeu_ps(dst + col, acc);
    }
    for (; col < dim; ++col) {
        dst[col] /= weight_sum;
    }
}

void qsg_latent_encode_f16(
    const float* src,
    std::uint16_t* dst,
    std::int32_t count) {
    if (!src || !dst || count <= 0) {
        return;
    }
    for (std::int32_t idx = 0; idx < count; ++idx) {
        dst[idx] = float_to_half_bits(src[idx]);
    }
}

void qsg_latent_decode_f16(
    const std::uint16_t* src,
    float* dst,
    std::int32_t count) {
    if (!src || !dst || count <= 0) {
        return;
    }
    for (std::int32_t idx = 0; idx < count; ++idx) {
        dst[idx] = half_bits_to_float(src[idx]);
    }
}

}  // extern "C"
