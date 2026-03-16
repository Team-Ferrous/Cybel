/**
 * CPU-optimized attention kernels.
 *
 * AVX2 vectorized kernels with OpenMP outer-loop parallelization.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "fast_math.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

extern "C" int anvil_get_num_threads_for_path(int decode_path);

inline int read_env_threads(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr) {
        return 0;
    }
    const int n = std::atoi(env);
    return n > 0 ? n : 0;
}

inline int get_num_threads(bool decode_path = true) {
    const int mode_threads = anvil_get_num_threads_for_path(decode_path ? 1 : 0);
    if (mode_threads > 0) {
        return mode_threads;
    }
    const int env_mode_threads = read_env_threads(
        decode_path ? "ANVIL_NUM_THREADS_DECODE" : "ANVIL_NUM_THREADS_BATCH"
    );
    const int env_threads = env_mode_threads > 0
        ? env_mode_threads
        : read_env_threads("ANVIL_NUM_THREADS");
#ifdef _OPENMP
    const int procs = std::max(1, omp_get_num_procs());
    if (env_threads > 0) return std::min(env_threads, procs);
    const int max_threads = std::max(1, omp_get_max_threads());
    return std::min(max_threads, procs);
#else
    if (env_threads > 0) return std::max(1, env_threads);
    return 1;
#endif
}

#ifdef __AVX2__
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

inline float dot_f32(const float* a, const float* b, int len) {
#ifdef __AVX2__
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 bv = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    float sum = hsum256_ps(acc);
    for (; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

inline void softmax_scores_inplace(float* scores, int seq_k) {
    float max_score = scores[0];
    for (int i = 1; i < seq_k; ++i) {
        max_score = std::max(max_score, scores[static_cast<std::size_t>(i)]);
    }

    float sum_exp = 0.0f;
    int i = 0;
#ifdef __AVX2__
    {
        __m256 max_vec = _mm256_set1_ps(max_score);
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 8 <= seq_k; i += 8) {
            __m256 sv = _mm256_loadu_ps(scores + i);
            sv = anvil_fast_math::v_expf(_mm256_sub_ps(sv, max_vec));
            sum_vec = _mm256_add_ps(sum_vec, sv);
            _mm256_storeu_ps(scores + i, sv);
        }
        sum_exp = hsum256_ps(sum_vec);
    }
    for (; i < seq_k; ++i) {
        const float v = anvil_fast_math::fast_exp_scalar(
            scores[static_cast<std::size_t>(i)] - max_score
        );
        scores[static_cast<std::size_t>(i)] = v;
        sum_exp += v;
    }
#else
    for (int i = 0; i < seq_k; ++i) {
        const float v = anvil_fast_math::fast_exp_scalar(
            scores[static_cast<std::size_t>(i)] - max_score
        );
        scores[static_cast<std::size_t>(i)] = v;
        sum_exp += v;
    }
#endif
    const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

#ifdef __AVX2__
    const int vec_end = seq_k - (seq_k % 8);
    __m256 inv_vec = _mm256_set1_ps(inv_sum);
    int j = 0;
    for (; j + 8 <= vec_end; j += 8) {
        __m256 sv = _mm256_loadu_ps(scores + j);
        sv = _mm256_mul_ps(sv, inv_vec);
        _mm256_storeu_ps(scores + j, sv);
    }
    for (; j < seq_k; ++j) {
        scores[static_cast<std::size_t>(j)] *= inv_sum;
    }
#else
    for (int i = 0; i < seq_k; ++i) {
        scores[static_cast<std::size_t>(i)] *= inv_sum;
    }
#endif
}

inline void accumulate_attention_output(
    const float* V,
    const float* scores,
    float* out,
    int seq_k_begin,
    int seq_k_end,
    int head_dim
) {
    for (int k_idx = seq_k_begin; k_idx < seq_k_end; ++k_idx) {
        const float* v_vec = V + static_cast<std::size_t>(k_idx) * head_dim;
        const float attn_weight = scores[static_cast<std::size_t>(k_idx)];
#ifdef __AVX2__
        __m256 weight_vec = _mm256_set1_ps(attn_weight);
        int d = 0;
        for (; d + 8 <= head_dim; d += 8) {
            __m256 o = _mm256_loadu_ps(out + d);
            __m256 v = _mm256_loadu_ps(v_vec + d);
            o = _mm256_fmadd_ps(weight_vec, v, o);
            _mm256_storeu_ps(out + d, o);
        }
        for (; d < head_dim; ++d) {
            out[d] += attn_weight * v_vec[d];
        }
#else
        for (int d = 0; d < head_dim; ++d) {
            out[d] += attn_weight * v_vec[d];
        }
#endif
    }
}

inline bool use_tiled_mqa_kernel(int head_dim) {
    switch (head_dim) {
        case 64:
        case 80:
        case 96:
        case 128:
            return true;
        default:
            return false;
    }
}

}  // namespace

extern "C" {

void fused_attention_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float scale
) {
    if (
        Q == nullptr || K == nullptr || V == nullptr || out == nullptr ||
        batch_heads <= 0 || seq_q <= 0 || seq_k <= 0 || head_dim <= 0
    ) {
        return;
    }

    const int n_threads = get_num_threads(true);
#ifdef _OPENMP
    const bool allow_parallel = (batch_heads > 1) && (omp_in_parallel() == 0);
#endif
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(allow_parallel)
#endif
    for (int bh = 0; bh < batch_heads; ++bh) {
        const float* Q_bh = Q + static_cast<std::size_t>(bh) * seq_q * head_dim;
        const float* K_bh = K + static_cast<std::size_t>(bh) * seq_k * head_dim;
        const float* V_bh = V + static_cast<std::size_t>(bh) * seq_k * head_dim;
        float* out_bh = out + static_cast<std::size_t>(bh) * seq_q * head_dim;
        thread_local std::vector<float> score_buf;
        if (static_cast<int>(score_buf.size()) < seq_k) {
            score_buf.resize(static_cast<std::size_t>(seq_k));
        }

        for (int q_idx = 0; q_idx < seq_q; ++q_idx) {
            float* scores = score_buf.data();
            const float* q_vec = Q_bh + static_cast<std::size_t>(q_idx) * head_dim;

            for (int k_idx = 0; k_idx < seq_k; ++k_idx) {
                const float* k_vec = K_bh + static_cast<std::size_t>(k_idx) * head_dim;
                scores[static_cast<std::size_t>(k_idx)] = dot_f32(q_vec, k_vec, head_dim) * scale;
            }
            softmax_scores_inplace(scores, seq_k);

            float* out_vec = out_bh + static_cast<std::size_t>(q_idx) * head_dim;
            std::memset(out_vec, 0, static_cast<std::size_t>(head_dim) * sizeof(float));
            accumulate_attention_output(V_bh, scores, out_vec, 0, seq_k, head_dim);
        }
    }
}

void fused_attention_int8(
    const int8_t* Q,
    const int8_t* K,
    const int8_t* V,
    float* out,
    const float* Q_scale,
    const float* K_scale,
    const float* V_scale,
    int batch_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float attn_scale
) {
    (void)Q;
    (void)K;
    (void)V;
    (void)out;
    (void)Q_scale;
    (void)K_scale;
    (void)V_scale;
    (void)batch_heads;
    (void)seq_q;
    (void)seq_k;
    (void)head_dim;
    (void)attn_scale;
    // TODO: Implement fully quantized INT8 attention path.
}

void fused_attention_mqa_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch,
    int q_heads,
    int kv_heads,
    int seq_q,
    int seq_k,
    int head_dim,
    float scale
) {
    if (
        Q == nullptr || K == nullptr || V == nullptr || out == nullptr ||
        batch <= 0 || q_heads <= 0 || kv_heads <= 0 || seq_q <= 0 || seq_k <= 0 ||
        head_dim <= 0
    ) {
        return;
    }

    const int heads_per_kv = std::max(1, q_heads / kv_heads);
    constexpr int kMqaSeqTile = 32;
    const bool use_tiled_mqa = use_tiled_mqa_kernel(head_dim);
    const int n_threads = get_num_threads(true);
#ifdef _OPENMP
    const bool allow_parallel = (batch * q_heads > 1) && (omp_in_parallel() == 0);
#endif

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static) if(allow_parallel)
#endif
    for (int b = 0; b < batch; ++b) {
        for (int kv_h = 0; kv_h < kv_heads; ++kv_h) {
            const float* k_ptr = K + (static_cast<std::size_t>(b) * kv_heads + kv_h) * seq_k * head_dim;
            const float* v_ptr = V + (static_cast<std::size_t>(b) * kv_heads + kv_h) * seq_k * head_dim;
            const int q_begin = kv_h * heads_per_kv;
            const int q_end = std::min(q_heads, q_begin + heads_per_kv);
            thread_local std::vector<float> score_buf;
            if (use_tiled_mqa && static_cast<int>(score_buf.size()) < seq_k) {
                score_buf.resize(static_cast<std::size_t>(seq_k));
            }
            for (int q_h = q_begin; q_h < q_end; ++q_h) {
                const float* q_ptr = Q + (static_cast<std::size_t>(b) * q_heads + q_h) * seq_q * head_dim;
                float* out_ptr = out + (static_cast<std::size_t>(b) * q_heads + q_h) * seq_q * head_dim;
                if (!use_tiled_mqa) {
                    fused_attention_f32(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        out_ptr,
                        1,
                        seq_q,
                        seq_k,
                        head_dim,
                        scale
                    );
                    continue;
                }
                for (int q_idx = 0; q_idx < seq_q; ++q_idx) {
                    float* scores = score_buf.data();
                    const float* q_vec = q_ptr + static_cast<std::size_t>(q_idx) * head_dim;
                    for (int k_block = 0; k_block < seq_k; k_block += kMqaSeqTile) {
                        const int k_end = std::min(seq_k, k_block + kMqaSeqTile);
                        for (int k_idx = k_block; k_idx < k_end; ++k_idx) {
                            const float* k_vec = k_ptr + static_cast<std::size_t>(k_idx) * head_dim;
                            scores[static_cast<std::size_t>(k_idx)] = dot_f32(
                                q_vec,
                                k_vec,
                                head_dim
                            ) * scale;
                        }
                    }
                    softmax_scores_inplace(scores, seq_k);
                    float* out_vec = out_ptr + static_cast<std::size_t>(q_idx) * head_dim;
                    std::memset(
                        out_vec,
                        0,
                        static_cast<std::size_t>(head_dim) * sizeof(float)
                    );
                    for (int k_block = 0; k_block < seq_k; k_block += kMqaSeqTile) {
                        const int k_end = std::min(seq_k, k_block + kMqaSeqTile);
                        accumulate_attention_output(
                            v_ptr,
                            scores,
                            out_vec,
                            k_block,
                            k_end,
                            head_dim
                        );
                    }
                }
            }
        }
    }
}

}  // extern "C"
