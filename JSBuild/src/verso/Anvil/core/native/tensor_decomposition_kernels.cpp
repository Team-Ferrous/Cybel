/**
 * Tensor decomposition kernels.
 */

#include <cstddef>
#include <vector>

namespace {

inline bool checked_mul_size(std::size_t a, std::size_t b, std::size_t* out) {
    if (out == nullptr) {
        return false;
    }
    if (a == 0 || b == 0) {
        *out = 0;
        return true;
    }
    if (a > (static_cast<std::size_t>(-1) / b)) {
        return false;
    }
    *out = a * b;
    return true;
}

}  // namespace

extern "C" {

int tensor_mpo_matvec_f32(
    const float* x,
    const float* const* factors,
    const int* rows,
    const int* cols,
    int num_factors,
    float* y
) {
    if (x == nullptr || factors == nullptr || rows == nullptr || cols == nullptr || y == nullptr) {
        return 0;
    }
    if (num_factors <= 0) {
        return 0;
    }

    std::vector<std::size_t> left_prefix(static_cast<std::size_t>(num_factors), 1);
    std::size_t total_in = 1;
    std::size_t total_out = 1;

    for (int i = 0; i < num_factors; ++i) {
        if (rows[i] <= 0 || cols[i] <= 0 || factors[i] == nullptr) {
            return 0;
        }
        left_prefix[static_cast<std::size_t>(i)] = total_in;
        if (!checked_mul_size(total_in, static_cast<std::size_t>(cols[i]), &total_in)) {
            return 0;
        }
        if (!checked_mul_size(total_out, static_cast<std::size_t>(rows[i]), &total_out)) {
            return 0;
        }
    }

    std::vector<float> current(total_in, 0.0f);
    for (std::size_t i = 0; i < total_in; ++i) {
        current[i] = x[i];
    }

    std::size_t right = 1;
    for (int k = num_factors - 1; k >= 0; --k) {
        const int r = rows[k];
        const int c = cols[k];
        const float* factor = factors[k];
        const std::size_t left = left_prefix[static_cast<std::size_t>(k)];

        const std::size_t next_size = left * static_cast<std::size_t>(r) * right;
        std::vector<float> next(next_size, 0.0f);

        for (std::size_t l = 0; l < left; ++l) {
            for (int out_idx = 0; out_idx < r; ++out_idx) {
                float* dst = next.data() + (l * static_cast<std::size_t>(r) + static_cast<std::size_t>(out_idx)) * right;
                for (int in_idx = 0; in_idx < c; ++in_idx) {
                    const float w = factor[static_cast<std::size_t>(out_idx) * static_cast<std::size_t>(c) + static_cast<std::size_t>(in_idx)];
                    const float* src = current.data() + (l * static_cast<std::size_t>(c) + static_cast<std::size_t>(in_idx)) * right;
                    for (std::size_t rr = 0; rr < right; ++rr) {
                        dst[rr] += w * src[rr];
                    }
                }
            }
        }

        current.swap(next);
        if (!checked_mul_size(right, static_cast<std::size_t>(r), &right)) {
            return 0;
        }
    }

    if (current.size() != total_out) {
        return 0;
    }

    for (std::size_t i = 0; i < total_out; ++i) {
        y[i] = current[i];
    }
    return 1;
}

}  // extern "C"
