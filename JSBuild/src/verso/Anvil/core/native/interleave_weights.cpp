/**
 * Row-block interleaving for Q4_K and Q8_0 weight layouts.
 *
 * Reorders quantized weight blocks so that consecutive output rows'
 * blocks for the same input position are adjacent in memory.
 * This enables L1 cache hits when using outer-loop-unrolled matvec.
 *
 * Standard layout: [row0_blk0][row0_blk1]...[row1_blk0][row1_blk1]...
 * Interleaved:     [row0_blk0][row1_blk0]...[row0_blk1][row1_blk1]...
 *
 * This is a pure storage transform — output is bit-identical.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

// Interleave Q4_K rows for outer-loop-unrolled matvec
int interleave_q4k_rows(
    const std::uint8_t* src,
    std::uint8_t* dst,
    int out_rows,
    int in_dim,
    int unroll
) {
    if (src == nullptr || dst == nullptr || out_rows <= 0 || in_dim <= 0 || unroll <= 0) {
        return 0;
    }
    if ((in_dim % 256) != 0) {
        return 0;
    }

    constexpr int kBlockBytes = 144;
    const int blocks_per_row = in_dim / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
    if (row_bytes == 0) {
        return 0;
    }

    std::size_t dst_off = 0;
    for (int rb = 0; rb < out_rows; rb += unroll) {
        const int group = std::min(unroll, out_rows - rb);
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            for (int r = 0; r < group; ++r) {
                const std::size_t src_row_off = static_cast<std::size_t>(rb + r) * row_bytes;
                const std::size_t src_off = src_row_off + static_cast<std::size_t>(blk) * kBlockBytes;
                std::memcpy(dst + dst_off, src + src_off, static_cast<std::size_t>(kBlockBytes));
                dst_off += static_cast<std::size_t>(kBlockBytes);
            }
        }
    }

    return 1;
}

// Interleave Q8_0 rows for outer-loop-unrolled matvec
int interleave_q8_0_rows(
    const std::uint8_t* src,
    std::uint8_t* dst,
    int out_rows,
    int in_dim,
    int unroll
) {
    if (src == nullptr || dst == nullptr || out_rows <= 0 || in_dim <= 0 || unroll <= 0) {
        return 0;
    }
    if ((in_dim % 32) != 0) {
        return 0;
    }

    constexpr int kBlockBytes = 34;  // 2 bytes scale + 32 bytes int8
    const int blocks_per_row = in_dim / 32;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;
    if (row_bytes == 0) {
        return 0;
    }

    std::size_t dst_off = 0;
    for (int rb = 0; rb < out_rows; rb += unroll) {
        const int group = std::min(unroll, out_rows - rb);
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            for (int r = 0; r < group; ++r) {
                const std::size_t src_row_off = static_cast<std::size_t>(rb + r) * row_bytes;
                const std::size_t src_off = src_row_off + static_cast<std::size_t>(blk) * kBlockBytes;
                std::memcpy(dst + dst_off, src + src_off, static_cast<std::size_t>(kBlockBytes));
                dst_off += static_cast<std::size_t>(kBlockBytes);
            }
        }
    }

    return 1;
}

// De-interleave (restore original layout) for verification
int deinterleave_q4k_rows(
    const std::uint8_t* src,
    std::uint8_t* dst,
    int out_rows,
    int in_dim,
    int unroll
) {
    if (src == nullptr || dst == nullptr || out_rows <= 0 || in_dim <= 0 || unroll <= 0) {
        return 0;
    }
    if ((in_dim % 256) != 0) {
        return 0;
    }

    constexpr int kBlockBytes = 144;
    const int blocks_per_row = in_dim / 256;
    const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * kBlockBytes;

    std::size_t src_off = 0;
    for (int rb = 0; rb < out_rows; rb += unroll) {
        const int group = std::min(unroll, out_rows - rb);
        for (int blk = 0; blk < blocks_per_row; ++blk) {
            for (int r = 0; r < group; ++r) {
                const std::size_t dst_row_off = static_cast<std::size_t>(rb + r) * row_bytes;
                const std::size_t dst_off_pos = dst_row_off + static_cast<std::size_t>(blk) * kBlockBytes;
                std::memcpy(dst + dst_off_pos, src + src_off, static_cast<std::size_t>(kBlockBytes));
                src_off += static_cast<std::size_t>(kBlockBytes);
            }
        }
    }

    return 1;
}

}  // extern "C"
