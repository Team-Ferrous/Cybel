#pragma once

#include <cstddef>
#include <cstdint>

struct QSGRowRef {
    std::int32_t page_id;
    std::int32_t row_idx;
};

struct QSGCompactMove {
    QSGRowRef src;
    QSGRowRef dst;
};

extern "C" {

void qsg_state_gather_rows(
    float* dst,
    const float* const* pages,
    const QSGRowRef* refs,
    std::int32_t n_refs,
    std::int32_t rows_per_page,
    std::int32_t dim);

void qsg_state_scatter_rows(
    const float* src,
    float* const* pages,
    const QSGRowRef* refs,
    std::int32_t n_refs,
    std::int32_t rows_per_page,
    std::int32_t dim);

void qsg_state_clone_cow(
    float* const* pages,
    const QSGRowRef* src_refs,
    const QSGRowRef* dst_refs,
    std::int32_t n_rows,
    std::int32_t rows_per_page,
    std::int32_t dim);

void qsg_state_compact(
    float* const* pages,
    const QSGCompactMove* moves,
    std::int32_t n_moves,
    std::int32_t rows_per_page,
    std::int32_t dim);

void qsg_state_weighted_merge(
    float* dst,
    const float* src,
    const float* weights,
    std::int32_t n_rows,
    std::int32_t dim);

void qsg_latent_encode_f16(
    const float* src,
    std::uint16_t* dst,
    std::int32_t count);

void qsg_latent_decode_f16(
    const std::uint16_t* src,
    float* dst,
    std::int32_t count);

}
