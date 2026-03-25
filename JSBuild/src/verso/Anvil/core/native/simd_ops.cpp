/**
 * SIMD-accelerated operations for the native QSG inference engine.
 *
 * AVX2 vectorized kernels with OpenMP row/chunk parallelism.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <random>
#include <unordered_map>
#include <vector>

#include "amx_kernels.h"
#include "fast_math.h"

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

extern "C" int anvil_get_num_threads_for_path(int decode_path);
extern "C" int anvil_get_thread_mode();

inline int read_env_threads() {
    const char* env = std::getenv("ANVIL_NUM_THREADS");
    if (env == nullptr) {
        return 0;
    }
    const int n = std::atoi(env);
    return n > 0 ? n : 0;
}

inline int get_num_threads() {
    const int mode = anvil_get_thread_mode();
    if (mode == 0 || mode == 1) {
        const int mode_threads = anvil_get_num_threads_for_path(mode);
        if (mode_threads > 0) {
            return mode_threads;
        }
    }
    const int env_threads = read_env_threads();
    if (env_threads > 0) {
        return env_threads;
    }
#ifdef _OPENMP
    return std::max(1, omp_get_max_threads());
#else
    return 1;
#endif
}

inline std::mt19937& sampler_rng() {
    thread_local std::mt19937 rng{std::random_device{}()};
    return rng;
}

struct QSGSamplingStats {
    std::uint64_t grover_calls = 0;
    double grover_seconds = 0.0;
    std::uint64_t grover_candidate_count = 0;
    double grover_rescore_delta_sum = 0.0;
    std::uint64_t grover_rescore_delta_samples = 0;
    std::uint64_t grover_timeout_events = 0;
    std::uint64_t coconut_calls = 0;
    double coconut_seconds = 0.0;
    std::uint64_t coconut_candidate_count = 0;
    double coconut_entropy_sum = 0.0;
    std::uint64_t coconut_entropy_samples = 0;
    double coconut_amplitude_sum = 0.0;
    std::uint64_t coconut_amplitude_samples = 0;
    std::uint64_t coconut_consistency_rejects = 0;
    std::uint64_t grammar_fastlane_calls = 0;
};

thread_local QSGSamplingStats g_qsg_sampling_stats;

inline double monotonic_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
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

inline int argmax_index(const float* values, int len) {
    if (values == nullptr || len <= 0) {
        return 0;
    }
    int best_idx = 0;
    float best = values[0];
    for (int i = 1; i < len; ++i) {
        if (values[i] > best) {
            best = values[i];
            best_idx = i;
        }
    }
    return best_idx;
}

inline bool sanitize_logits_inplace(float* logits, int len) {
    if (logits == nullptr || len <= 0) {
        return false;
    }
    bool has_finite = false;
    float min_finite = std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; ++i) {
        const float v = logits[i];
        if (std::isfinite(v)) {
            has_finite = true;
            min_finite = std::min(min_finite, v);
        }
    }
    if (!has_finite) {
        std::memset(logits, 0, static_cast<std::size_t>(len) * sizeof(float));
        return false;
    }
    const float safe_floor = min_finite - 1.0e4f;
    for (int i = 0; i < len; ++i) {
        if (!std::isfinite(logits[i])) {
            logits[i] = safe_floor;
        }
    }
    return true;
}

inline void suppress_tokens_inplace(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count
) {
    if (logits == nullptr || len <= 0 || suppressed_ids == nullptr || suppressed_count <= 0) {
        return;
    }
    const float suppressed_value = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < suppressed_count; ++i) {
        const int token_id = suppressed_ids[i];
        if (token_id >= 0 && token_id < len) {
            logits[token_id] = suppressed_value;
        }
    }
}

inline void apply_allowed_token_mask_inplace(
    float* logits,
    int len,
    const int* allowed_ids,
    int allowed_count
) {
    if (logits == nullptr || len <= 0 || allowed_ids == nullptr || allowed_count <= 0) {
        return;
    }
    std::vector<unsigned char> allowed(static_cast<std::size_t>(len), 0);
    for (int i = 0; i < allowed_count; ++i) {
        const int token_id = allowed_ids[i];
        if (token_id >= 0 && token_id < len) {
            allowed[static_cast<std::size_t>(token_id)] = 1;
        }
    }
    const float suppressed_value = -std::numeric_limits<float>::infinity();
    for (int idx = 0; idx < len; ++idx) {
        if (allowed[static_cast<std::size_t>(idx)] == 0) {
            logits[idx] = suppressed_value;
        }
    }
    g_qsg_sampling_stats.grammar_fastlane_calls += 1;
}

inline void apply_token_penalties_inplace(
    float* logits,
    int len,
    const int* token_history,
    int history_len,
    float presence_penalty,
    float repetition_penalty
) {
    if (logits == nullptr || len <= 0 || token_history == nullptr || history_len <= 0) {
        return;
    }

    const bool apply_presence = std::fabs(presence_penalty) > 1.0e-12f;
    const bool apply_repetition =
        repetition_penalty > 0.0f && std::fabs(repetition_penalty - 1.0f) > 1.0e-6f;
    if (!apply_presence && !apply_repetition) {
        return;
    }

    std::unordered_map<int, int> counts;
    counts.reserve(static_cast<std::size_t>(history_len));
    for (int i = 0; i < history_len; ++i) {
        const int token_id = token_history[i];
        if (token_id < 0 || token_id >= len) {
            continue;
        }
        counts[token_id] += 1;
    }

    for (const auto& entry : counts) {
        const int token_id = entry.first;
        const int count = entry.second;
        if (apply_presence) {
            logits[token_id] -= presence_penalty;
        }
        if (apply_repetition) {
            const float scale = std::pow(repetition_penalty, static_cast<float>(std::max(1, count)));
            if (logits[token_id] > 0.0f) {
                logits[token_id] /= scale;
            } else {
                logits[token_id] *= scale;
            }
        }
    }
}

inline void suppress_repeated_ngrams_inplace(
    float* logits,
    int len,
    const int* token_history,
    int history_len,
    int ngram_size
) {
    if (
        logits == nullptr
        || len <= 0
        || token_history == nullptr
        || history_len <= 0
        || ngram_size <= 1
    ) {
        return;
    }

    const int prefix_len = ngram_size - 1;
    if (history_len < prefix_len) {
        return;
    }

    const int suffix_start = history_len - prefix_len;
    std::vector<int> banned;
    banned.reserve(16);
    for (int i = 0; i + prefix_len < history_len; ++i) {
        bool matches = true;
        for (int j = 0; j < prefix_len; ++j) {
            if (token_history[i + j] != token_history[suffix_start + j]) {
                matches = false;
                break;
            }
        }
        if (!matches) {
            continue;
        }
        const int banned_token = token_history[i + prefix_len];
        if (banned_token < 0 || banned_token >= len) {
            continue;
        }
        if (std::find(banned.begin(), banned.end(), banned_token) == banned.end()) {
            banned.push_back(banned_token);
        }
    }
    if (!banned.empty()) {
        suppress_tokens_inplace(
            logits,
            len,
            banned.data(),
            static_cast<int>(banned.size())
        );
    }
}

inline void apply_qsg_logits_transform_inplace(
    float* logits,
    int len,
    int use_coconut,
    int coconut_paths,
    float coconut_alpha,
    int use_grover,
    int grover_top_k,
    float grover_damping
) {
    if (logits == nullptr || len <= 0) {
        return;
    }
    if (!use_coconut && !use_grover) {
        return;
    }
    (void) sanitize_logits_inplace(logits, len);

    const int capped_top_k = std::max(1, std::min(len, grover_top_k > 0 ? grover_top_k : 8));
    std::vector<int> order(static_cast<std::size_t>(len));
    std::iota(order.begin(), order.end(), 0);
    std::nth_element(
        order.begin(),
        order.begin() + capped_top_k,
        order.end(),
        [&](int a, int b) {
            return logits[a] > logits[b];
        }
    );
    order.resize(static_cast<std::size_t>(capped_top_k));
    std::sort(
        order.begin(),
        order.end(),
        [&](int a, int b) {
            return logits[a] > logits[b];
        }
    );

    float anchor = logits[order.front()];
    float tail = logits[order.back()];
    float top2 = order.size() > 1 ? logits[order[1]] : tail;
    float confidence = std::max(0.0f, std::min(1.0f, (anchor - top2) / 8.0f));
    float path_scale = std::max(0.25f, std::min(2.0f, static_cast<float>(std::max(1, coconut_paths)) / 8.0f));
    float alpha = std::max(0.0f, coconut_alpha) * path_scale;

    if (use_grover && grover_damping > 0.0f) {
        const double started = monotonic_seconds();
        const float damping = std::max(0.0f, std::min(2.0f, grover_damping));
        double rescore_delta_sum = 0.0;
        for (int idx : order) {
            const float delta = std::max(0.0f, anchor - logits[idx]);
            logits[idx] += delta * damping;
            rescore_delta_sum += static_cast<double>(delta);
        }
        g_qsg_sampling_stats.grover_calls += 1;
        g_qsg_sampling_stats.grover_seconds += std::max(0.0, monotonic_seconds() - started);
        g_qsg_sampling_stats.grover_candidate_count += static_cast<std::uint64_t>(order.size());
        g_qsg_sampling_stats.grover_rescore_delta_sum += rescore_delta_sum;
        g_qsg_sampling_stats.grover_rescore_delta_samples += static_cast<std::uint64_t>(order.size());
    }

    if (use_coconut && alpha > 0.0f) {
        const double started = monotonic_seconds();
        const float mean = std::accumulate(order.begin(), order.end(), 0.0f,
            [&](float acc, int idx) {
                return acc + logits[idx];
            }
        ) / static_cast<float>(order.size());
        const float blend = std::max(0.0f, std::min(1.5f, alpha * (0.5f + 0.5f * confidence)));
        double entropy = 0.0;
        double amplitude = 0.0;
        double exp_sum = 0.0;
        for (int idx : order) {
            exp_sum += std::exp(static_cast<double>(logits[idx] - anchor));
        }
        if (confidence < 0.05f) {
            g_qsg_sampling_stats.coconut_consistency_rejects += 1;
        }
        for (int idx : order) {
            const float centered = logits[idx] - mean;
            logits[idx] = mean + centered * (1.0f + blend);
            if (exp_sum > 0.0) {
                const double prob = std::exp(static_cast<double>(logits[idx] - anchor)) / exp_sum;
                if (prob > 0.0) {
                    entropy -= prob * std::log(prob);
                }
                amplitude += std::sqrt(prob);
            }
        }
        g_qsg_sampling_stats.coconut_calls += 1;
        g_qsg_sampling_stats.coconut_seconds += std::max(0.0, monotonic_seconds() - started);
        g_qsg_sampling_stats.coconut_candidate_count += static_cast<std::uint64_t>(order.size());
        g_qsg_sampling_stats.coconut_entropy_sum += entropy;
        g_qsg_sampling_stats.coconut_entropy_samples += 1;
        g_qsg_sampling_stats.coconut_amplitude_sum += amplitude;
        g_qsg_sampling_stats.coconut_amplitude_samples += static_cast<std::uint64_t>(order.size());
    }

    (void) sanitize_logits_inplace(logits, len);
}

#ifdef __linux__
inline long read_long_file(const std::string& path) {
    std::ifstream ifs(path);
    long v = 0;
    if (!ifs.is_open()) {
        return 0;
    }
    ifs >> v;
    return v;
}

inline std::vector<int> detect_p_cores() {
    std::vector<std::pair<int, long>> cpu_freqs;
    DIR* dir = opendir("/sys/devices/system/cpu");
    if (dir == nullptr) {
        return {};
    }
    while (const dirent* ent = readdir(dir)) {
        const std::string name(ent->d_name);
        if (name.rfind("cpu", 0) != 0 || name.size() <= 3) {
            continue;
        }
        bool numeric = true;
        for (std::size_t i = 3; i < name.size(); ++i) {
            if (name[i] < '0' || name[i] > '9') {
                numeric = false;
                break;
            }
        }
        if (!numeric) {
            continue;
        }
        const int cpu = std::atoi(name.c_str() + 3);
        const std::string freq_path =
            "/sys/devices/system/cpu/" + name + "/cpufreq/cpuinfo_max_freq";
        const long freq = read_long_file(freq_path);
        if (cpu >= 0 && freq > 0) {
            cpu_freqs.emplace_back(cpu, freq);
        }
    }
    closedir(dir);
    if (cpu_freqs.empty()) {
        return {};
    }

    long max_freq = 0;
    for (const auto& cf : cpu_freqs) {
        max_freq = std::max(max_freq, cf.second);
    }
    if (max_freq <= 0) {
        return {};
    }
    const long threshold = static_cast<long>(static_cast<double>(max_freq) * 0.90);
    std::vector<int> p_cores;
    for (const auto& cf : cpu_freqs) {
        if (cf.second >= threshold) {
            p_cores.push_back(cf.first);
        }
    }
    std::sort(p_cores.begin(), p_cores.end());
    p_cores.erase(std::unique(p_cores.begin(), p_cores.end()), p_cores.end());
    return p_cores;
}
#endif

}  // namespace

extern "C" {

// ============================================================
// MATRIX MULTIPLY — The hot path (~90% of inference time)
// ============================================================

void simd_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    if (A == nullptr || B == nullptr || C == nullptr || M <= 0 || K <= 0 || N <= 0) {
        return;
    }

    if (anvil::native::amx_matmul_f32(A, B, C, M, K, N)) {
        return;
    }

    std::memset(C, 0, static_cast<std::size_t>(M) * N * sizeof(float));
    const int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        float* c_row = C + static_cast<std::size_t>(m) * N;
        const float* a_row = A + static_cast<std::size_t>(m) * K;

        for (int k = 0; k < K; ++k) {
            const float a_val = a_row[k];
            const float* b_row = B + static_cast<std::size_t>(k) * N;
#ifdef __AVX2__
            __m256 a_vec = _mm256_set1_ps(a_val);
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                __m256 c_vec = _mm256_loadu_ps(c_row + n);
                __m256 b_vec = _mm256_loadu_ps(b_row + n);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(c_row + n, c_vec);
            }
            for (; n < N; ++n) {
                c_row[n] += a_val * b_row[n];
            }
#else
            for (int n = 0; n < N; ++n) {
                c_row[n] += a_val * b_row[n];
            }
#endif
        }
    }
}

// ============================================================
// MATRIX-VECTOR MULTIPLY — M=1 specialization
// ============================================================

void simd_matvec_f32(const float* x, const float* A, float* y, int K, int N) {
    if (x == nullptr || A == nullptr || y == nullptr || K <= 0 || N <= 0) {
        return;
    }

    std::memset(y, 0, static_cast<std::size_t>(N) * sizeof(float));
    const int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
    {
        int tid = 0;
        int nthreads = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#endif
        const int chunk = (N + nthreads - 1) / nthreads;
        const int n_start = tid * chunk;
        const int n_end = std::min(n_start + chunk, N);

        for (int k = 0; k < K; ++k) {
            const float x_val = x[k];
            const float* a_row = A + static_cast<std::size_t>(k) * N;
#ifdef __AVX2__
            __m256 x_vec = _mm256_set1_ps(x_val);
            int n = n_start;
            for (; n + 8 <= n_end; n += 8) {
                __m256 y_vec = _mm256_loadu_ps(y + n);
                __m256 a_vec = _mm256_loadu_ps(a_row + n);
                y_vec = _mm256_fmadd_ps(x_vec, a_vec, y_vec);
                _mm256_storeu_ps(y + n, y_vec);
            }
            for (; n < n_end; ++n) {
                y[n] += x_val * a_row[n];
            }
#else
            for (int n = n_start; n < n_end; ++n) {
                y[n] += x_val * a_row[n];
            }
#endif
        }
    }
}

// ============================================================
// RMS NORMALIZATION
// ============================================================

void simd_rmsnorm_f32(float* x, const float* gamma, int dim, float eps) {
    if (x == nullptr || gamma == nullptr || dim <= 0) {
        return;
    }

    float mean_sq = 0.0f;
    const int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(+:mean_sq) if(dim > 2048) schedule(static)
#endif
    for (int i = 0; i < dim; ++i) {
        mean_sq += x[i] * x[i];
    }

    mean_sq /= static_cast<float>(dim);
    const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

#ifdef __AVX2__
    const int vec_end = dim - (dim % 8);
    __m256 scale = _mm256_set1_ps(inv_rms);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 2048) schedule(static)
#endif
    for (int i = 0; i < vec_end; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 gv = _mm256_loadu_ps(gamma + i);
        __m256 result = _mm256_mul_ps(_mm256_mul_ps(xv, scale), gv);
        _mm256_storeu_ps(x + i, result);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 2048) schedule(static)
#endif
    for (int i = vec_end; i < dim; ++i) {
        x[i] = x[i] * inv_rms * gamma[i];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 2048) schedule(static)
#endif
    for (int i = 0; i < dim; ++i) {
        x[i] = x[i] * inv_rms * gamma[i];
    }
#endif
}

// ============================================================
// SwiGLU ACTIVATION — silu(gate) * up
// ============================================================

void simd_swiglu_f32(const float* gate, const float* up, float* out, int dim) {
    if (gate == nullptr || up == nullptr || out == nullptr || dim <= 0) {
        return;
    }

    const int n_threads = get_num_threads();
#ifdef __AVX2__
    const int vec_end = dim - (dim % 8);
    // Fully vectorized: silu(g) * up = g * sigmoid(g) * up
    // sigmoid(g) = 1/(1+exp(-g))
    const __m256 one = _mm256_set1_ps(1.0f);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 1024) schedule(static)
#endif
    for (int i = 0; i < vec_end; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        __m256 uv = _mm256_loadu_ps(up + i);
        // sigmoid(g) = 1 / (1 + exp(-g))  — fully in AVX2 registers
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), gv);
        __m256 exp_neg = anvil_fast_math::v_expf(neg_g);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        // silu(g) = g * sigmoid(g)
        __m256 silu = _mm256_mul_ps(gv, sig);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(silu, uv));
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 1024) schedule(static)
#endif
    for (int i = vec_end; i < dim; ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + anvil_fast_math::fast_exp_scalar(-g));
        out[i] = (g * sig) * up[i];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(dim > 1024) schedule(static)
#endif
    for (int i = 0; i < dim; ++i) {
        const float g = gate[i];
        const float sig = 1.0f / (1.0f + anvil_fast_math::fast_exp_scalar(-g));
        out[i] = (g * sig) * up[i];
    }
#endif
}

// ============================================================
// SOFTMAX — numerically stable
// ============================================================

void simd_softmax_f32(float* scores, int len) {
    if (scores == nullptr || len <= 0) {
        return;
    }

    const int n_threads = get_num_threads();
    float max_val = -std::numeric_limits<float>::infinity();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(max:max_val) if(len > 4096) schedule(static)
#endif
    for (int i = 0; i < len; ++i) {
        max_val = std::max(max_val, scores[i]);
    }

    float sum = 0.0f;
#ifdef __AVX2__
    const int vec_end = len - (len % 8);
    const __m256 max_vec = _mm256_set1_ps(max_val);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(+:sum) if(len > 4096) schedule(static)
#endif
    for (int i = 0; i < vec_end; i += 8) {
        __m256 sv = _mm256_loadu_ps(scores + i);
        sv = _mm256_sub_ps(sv, max_vec);
        sv = anvil_fast_math::v_expf(sv);
        _mm256_storeu_ps(scores + i, sv);
        sum += hsum256_ps(sv);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(+:sum) if(len > 4096) schedule(static)
#endif
    for (int i = vec_end; i < len; ++i) {
        scores[i] = anvil_fast_math::fast_exp_scalar(scores[i] - max_val);
        sum += scores[i];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) reduction(+:sum) if(len > 4096) schedule(static)
#endif
    for (int i = 0; i < len; ++i) {
        scores[i] = anvil_fast_math::fast_exp_scalar(scores[i] - max_val);
        sum += scores[i];
    }
#endif

    if (sum <= 0.0f) {
        const float uniform = 1.0f / static_cast<float>(len);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(len > 4096) schedule(static)
#endif
        for (int i = 0; i < len; ++i) {
            scores[i] = uniform;
        }
        return;
    }

    const float inv_sum = 1.0f / sum;
#ifdef __AVX2__
    const __m256 inv_vec = _mm256_set1_ps(inv_sum);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(len > 4096) schedule(static)
#endif
    for (int i = 0; i < vec_end; i += 8) {
        __m256 sv = _mm256_loadu_ps(scores + i);
        _mm256_storeu_ps(scores + i, _mm256_mul_ps(sv, inv_vec));
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(len > 4096) schedule(static)
#endif
    for (int i = vec_end; i < len; ++i) {
        scores[i] *= inv_sum;
    }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(len > 4096) schedule(static)
#endif
    for (int i = 0; i < len; ++i) {
        scores[i] *= inv_sum;
    }
#endif
}

void simd_sanitize_logits_f32(float* logits, int len) {
    (void) sanitize_logits_inplace(logits, len);
}

int simd_argmax_f32(const float* values, int len) {
    return argmax_index(values, len);
}

static bool apply_sampling_filters_inplace(
    float* scores,
    int len,
    float temperature,
    int top_k,
    float top_p,
    float min_p
) {
    if (temperature <= 0.0f) {
        return false;
    }

    const float inv_temp = 1.0f / std::max(temperature, 1.0e-6f);
    float max_scaled = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < len; ++i) {
        scores[i] *= inv_temp;
        max_scaled = std::max(max_scaled, scores[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float shifted = std::max(-80.0f, std::min(0.0f, scores[i] - max_scaled));
        const float p = anvil_fast_math::fast_exp_scalar(shifted);
        scores[i] = p;
        sum += p;
    }

    if (!std::isfinite(sum) || sum <= 0.0f) {
        return false;
    }

    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; ++i) {
        scores[i] *= inv_sum;
    }

    // Optional top-k filtering in probability space.
    if (top_k > 0 && top_k < len) {
        std::vector<int> order(static_cast<std::size_t>(len));
        std::iota(order.begin(), order.end(), 0);
        std::nth_element(
            order.begin(),
            order.begin() + top_k,
            order.end(),
            [&](int a, int b) {
                return scores[a] > scores[b];
            }
        );
        std::vector<uint8_t> keep(static_cast<std::size_t>(len), 0);
        for (int i = 0; i < top_k; ++i) {
            keep[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])] = 1;
        }
        float kept_sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            if (!keep[static_cast<std::size_t>(i)]) {
                scores[i] = 0.0f;
            } else {
                kept_sum += scores[i];
            }
        }
        if (kept_sum > 0.0f && std::isfinite(kept_sum)) {
            const float inv_kept = 1.0f / kept_sum;
            for (int i = 0; i < len; ++i) {
                scores[i] *= inv_kept;
            }
        } else {
            return false;
        }
    }

    // Optional min-p filter (relative to max probability).
    if (min_p > 0.0f) {
        float p_max = 0.0f;
        for (int i = 0; i < len; ++i) {
            p_max = std::max(p_max, scores[i]);
        }
        const float threshold = p_max * std::max(0.0f, min_p);
        float kept_sum = 0.0f;
        for (int i = 0; i < len; ++i) {
            if (scores[i] < threshold) {
                scores[i] = 0.0f;
            } else {
                kept_sum += scores[i];
            }
        }
        if (kept_sum > 0.0f && std::isfinite(kept_sum)) {
            const float inv_kept = 1.0f / kept_sum;
            for (int i = 0; i < len; ++i) {
                scores[i] *= inv_kept;
            }
        } else {
            return false;
        }
    }

    // Optional nucleus sampling (top-p).
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> order(static_cast<std::size_t>(len));
        std::iota(order.begin(), order.end(), 0);
        std::sort(
            order.begin(),
            order.end(),
            [&](int a, int b) {
                return scores[a] > scores[b];
            }
        );

        float cumulative = 0.0f;
        float kept_sum = 0.0f;
        bool kept_any = false;
        for (int rank = 0; rank < len; ++rank) {
            const int idx = order[static_cast<std::size_t>(rank)];
            const float p = scores[idx];
            cumulative += p;
            if (cumulative <= top_p || !kept_any) {
                kept_any = true;
                kept_sum += p;
                continue;
            }
            scores[idx] = 0.0f;
        }

        if (kept_sum > 0.0f && std::isfinite(kept_sum)) {
            const float inv_kept = 1.0f / kept_sum;
            for (int i = 0; i < len; ++i) {
                scores[i] *= inv_kept;
            }
        } else {
            return false;
        }
    }

    float normalized_sum = 0.0f;
    for (int i = 0; i < len; ++i) {
        normalized_sum += scores[i];
    }
    if (!std::isfinite(normalized_sum) || normalized_sum <= 0.0f) {
        return false;
    }
    return true;
}

int simd_sample_token_f32(
    float* logits,
    int len,
    float temperature,
    int eos_token,
    float top_p,
    int top_k,
    float min_p
) {
    if (logits == nullptr || len <= 0) {
        return eos_token >= 0 ? eos_token : 0;
    }

    float* scores = logits;
    const bool has_finite = sanitize_logits_inplace(scores, len);
    if (!has_finite) {
        return eos_token >= 0 ? eos_token : 0;
    }

    if (temperature <= 0.0f) {
        return argmax_index(scores, len);
    }

    if (!apply_sampling_filters_inplace(
            scores,
            len,
            temperature,
            top_k,
            top_p,
            min_p
        )) {
        return argmax_index(scores, len);
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float target = dist(sampler_rng());
    float cdf = 0.0f;
    for (int i = 0; i < len; ++i) {
        cdf += scores[i];
        if (target <= cdf) {
            return i;
        }
    }
    return len - 1;
}

int simd_score_token_f32(
    float* logits,
    int len,
    float temperature,
    int token_id,
    int* greedy_token_out,
    float* token_prob_out
) {
    if (greedy_token_out != nullptr) {
        *greedy_token_out = 0;
    }
    if (token_prob_out != nullptr) {
        *token_prob_out = 0.0f;
    }
    if (logits == nullptr || len <= 0) {
        return 0;
    }

    float* scores = logits;
    const bool has_finite = sanitize_logits_inplace(scores, len);
    if (!has_finite) {
        return 0;
    }

    const int greedy_token = argmax_index(scores, len);
    if (greedy_token_out != nullptr) {
        *greedy_token_out = greedy_token;
    }

    if (token_id < 0 || token_id >= len) {
        return 1;
    }

    if (temperature <= 0.0f) {
        if (token_prob_out != nullptr) {
            *token_prob_out = token_id == greedy_token ? 1.0f : 0.0f;
        }
        return 1;
    }

    if (!apply_sampling_filters_inplace(scores, len, temperature, 0, 1.0f, 0.0f)) {
        if (token_prob_out != nullptr) {
            *token_prob_out = token_id == greedy_token ? 1.0f : 0.0f;
        }
        return 1;
    }

    if (token_prob_out != nullptr) {
        *token_prob_out = scores[token_id];
    }
    return 1;
}

int simd_postprocess_score_token_f32(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count,
    const int* token_history,
    int history_len,
    float presence_penalty,
    float repetition_penalty,
    int no_repeat_ngram_size,
    float temperature,
    int eos_token,
    float top_p,
    int top_k,
    float min_p,
    int token_id,
    int* greedy_token_out,
    float* token_prob_out
) {
    (void) eos_token;
    if (greedy_token_out != nullptr) {
        *greedy_token_out = 0;
    }
    if (token_prob_out != nullptr) {
        *token_prob_out = 0.0f;
    }
    if (logits == nullptr || len <= 0) {
        return 0;
    }
    suppress_tokens_inplace(logits, len, suppressed_ids, suppressed_count);
    apply_token_penalties_inplace(
        logits,
        len,
        token_history,
        history_len,
        presence_penalty,
        repetition_penalty
    );
    suppress_repeated_ngrams_inplace(
        logits,
        len,
        token_history,
        history_len,
        no_repeat_ngram_size
    );
    return simd_score_token_f32(
        logits,
        len,
        temperature,
        token_id,
        greedy_token_out,
        token_prob_out
    );
}

void simd_suppress_tokens_f32(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count
) {
    suppress_tokens_inplace(logits, len, suppressed_ids, suppressed_count);
}

void simd_apply_token_penalties_f32(
    float* logits,
    int len,
    const int* token_history,
    int history_len,
    float presence_penalty,
    float repetition_penalty
) {
    apply_token_penalties_inplace(
        logits,
        len,
        token_history,
        history_len,
        presence_penalty,
        repetition_penalty
    );
    (void) sanitize_logits_inplace(logits, len);
}

int simd_postprocess_sample_f32(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count,
    const int* token_history,
    int history_len,
    float presence_penalty,
    float repetition_penalty,
    int no_repeat_ngram_size,
    float temperature,
    int eos_token,
    float top_p,
    int top_k,
    float min_p
) {
    if (logits == nullptr || len <= 0) {
        return eos_token >= 0 ? eos_token : 0;
    }
    suppress_tokens_inplace(logits, len, suppressed_ids, suppressed_count);
    apply_token_penalties_inplace(
        logits,
        len,
        token_history,
        history_len,
        presence_penalty,
        repetition_penalty
    );
    suppress_repeated_ngrams_inplace(
        logits,
        len,
        token_history,
        history_len,
        no_repeat_ngram_size
    );
    return simd_sample_token_f32(
        logits,
        len,
        temperature,
        eos_token,
        top_p,
        top_k,
        min_p
    );
}

int simd_qsg_postprocess_score_token_f32(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count,
    const int* token_history,
    int history_len,
    int use_coconut,
    int coconut_paths,
    float coconut_alpha,
    int use_grover,
    int grover_top_k,
    float grover_damping,
    float presence_penalty,
    float repetition_penalty,
    int no_repeat_ngram_size,
    float temperature,
    int eos_token,
    float top_p,
    int top_k,
    float min_p,
    int token_id,
    int* greedy_token_out,
    float* token_prob_out
) {
    (void) eos_token;
    if (greedy_token_out != nullptr) {
        *greedy_token_out = 0;
    }
    if (token_prob_out != nullptr) {
        *token_prob_out = 0.0f;
    }
    if (logits == nullptr || len <= 0) {
        return 0;
    }
    apply_qsg_logits_transform_inplace(
        logits,
        len,
        use_coconut,
        coconut_paths,
        coconut_alpha,
        use_grover,
        grover_top_k,
        grover_damping
    );
    suppress_tokens_inplace(logits, len, suppressed_ids, suppressed_count);
    apply_token_penalties_inplace(
        logits,
        len,
        token_history,
        history_len,
        presence_penalty,
        repetition_penalty
    );
    suppress_repeated_ngrams_inplace(
        logits,
        len,
        token_history,
        history_len,
        no_repeat_ngram_size
    );
    return simd_score_token_f32(
        logits,
        len,
        temperature,
        token_id,
        greedy_token_out,
        token_prob_out
    );
}

int simd_qsg_postprocess_sample_f32(
    float* logits,
    int len,
    const int* suppressed_ids,
    int suppressed_count,
    const int* token_history,
    int history_len,
    const int* grammar_allowed_ids,
    int grammar_allowed_count,
    int use_coconut,
    int coconut_paths,
    float coconut_alpha,
    int use_grover,
    int grover_top_k,
    float grover_damping,
    float presence_penalty,
    float repetition_penalty,
    int no_repeat_ngram_size,
    float temperature,
    int eos_token,
    float top_p,
    int top_k,
    float min_p
) {
    if (logits == nullptr || len <= 0) {
        return eos_token >= 0 ? eos_token : 0;
    }
    apply_qsg_logits_transform_inplace(
        logits,
        len,
        use_coconut,
        coconut_paths,
        coconut_alpha,
        use_grover,
        grover_top_k,
        grover_damping
    );
    apply_allowed_token_mask_inplace(
        logits,
        len,
        grammar_allowed_ids,
        grammar_allowed_count
    );
    suppress_tokens_inplace(logits, len, suppressed_ids, suppressed_count);
    apply_token_penalties_inplace(
        logits,
        len,
        token_history,
        history_len,
        presence_penalty,
        repetition_penalty
    );
    suppress_repeated_ngrams_inplace(
        logits,
        len,
        token_history,
        history_len,
        no_repeat_ngram_size
    );
    return simd_sample_token_f32(
        logits,
        len,
        temperature,
        eos_token,
        top_p,
        top_k,
        min_p
    );
}

void anvil_reset_qsg_sampling_stats() {
    g_qsg_sampling_stats = QSGSamplingStats{};
}

int anvil_qsg_sampling_stats_json(char* out, int out_len) {
    if (out == nullptr || out_len <= 0) {
        return 0;
    }
    const double grover_rescore_delta_mean =
        g_qsg_sampling_stats.grover_rescore_delta_samples > 0
            ? (g_qsg_sampling_stats.grover_rescore_delta_sum
               / static_cast<double>(g_qsg_sampling_stats.grover_rescore_delta_samples))
            : 0.0;
    const double coconut_entropy_mean =
        g_qsg_sampling_stats.coconut_entropy_samples > 0
            ? (g_qsg_sampling_stats.coconut_entropy_sum
               / static_cast<double>(g_qsg_sampling_stats.coconut_entropy_samples))
            : 0.0;
    const double coconut_amplitude_mean =
        g_qsg_sampling_stats.coconut_amplitude_samples > 0
            ? (g_qsg_sampling_stats.coconut_amplitude_sum
               / static_cast<double>(g_qsg_sampling_stats.coconut_amplitude_samples))
            : 0.0;
    std::ostringstream payload;
    payload
        << '{'
        << "\"grover_calls\":" << g_qsg_sampling_stats.grover_calls << ','
        << "\"grover_seconds\":" << g_qsg_sampling_stats.grover_seconds << ','
        << "\"grover_candidate_count\":" << g_qsg_sampling_stats.grover_candidate_count << ','
        << "\"grover_rescore_delta_mean\":" << grover_rescore_delta_mean << ','
        << "\"grover_timeout_events\":" << g_qsg_sampling_stats.grover_timeout_events << ','
        << "\"coconut_calls\":" << g_qsg_sampling_stats.coconut_calls << ','
        << "\"coconut_seconds\":" << g_qsg_sampling_stats.coconut_seconds << ','
        << "\"coconut_candidate_count\":" << g_qsg_sampling_stats.coconut_candidate_count << ','
        << "\"coconut_entropy_mean\":" << coconut_entropy_mean << ','
        << "\"coconut_amplitude_mean\":" << coconut_amplitude_mean << ','
        << "\"coconut_consistency_rejects\":" << g_qsg_sampling_stats.coconut_consistency_rejects << ','
        << "\"grammar_fastlane_calls\":" << g_qsg_sampling_stats.grammar_fastlane_calls
        << '}';
    const std::string text = payload.str();
    const int written = std::min(out_len - 1, static_cast<int>(text.size()));
    std::memcpy(out, text.data(), static_cast<std::size_t>(written));
    out[written] = '\0';
    return written;
}

void simd_seed_rng_f32(int seed) {
    sampler_rng().seed(static_cast<std::mt19937::result_type>(seed));
}

// ============================================================
// ROTARY POSITION EMBEDDING (RoPE)
// ============================================================

void simd_rope_f32(
    float* q,
    float* k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta_base
) {
    if (q == nullptr || k == nullptr || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 1) {
        return;
    }

    const int rope_dim = head_dim - (head_dim % 2);
    const int n_threads = get_num_threads();

    auto apply_rope = [&](float* vec, int heads) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(heads > 1) schedule(static)
#endif
        for (int h = 0; h < heads; ++h) {
            float* head_ptr = vec + static_cast<std::size_t>(h) * head_dim;
            for (int i = 0; i < rope_dim; i += 2) {
                const float freq = 1.0f / std::pow(
                    theta_base,
                    static_cast<float>(i) / static_cast<float>(std::max(head_dim, 1))
                );
                const float angle = static_cast<float>(pos) * freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);
                const float x0 = head_ptr[i];
                const float x1 = head_ptr[i + 1];
                head_ptr[i] = x0 * c - x1 * s;
                head_ptr[i + 1] = x0 * s + x1 * c;
            }
        }
    };

    apply_rope(q, n_heads);
    apply_rope(k, n_kv_heads);
}

void batch_rope_f32(
    float* q,
    float* k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int pos,
    float theta_base
) {
    simd_rope_f32(q, k, n_heads, n_kv_heads, head_dim, pos, theta_base);
}

void simd_fast_exp_f32(float* data, int len) {
    if (data == nullptr || len <= 0) {
        return;
    }

#ifdef __AVX2__
    const int vec_end = len - (len % 8);
    int i = 0;
    for (; i < vec_end; i += 8) {
        __m256 x = _mm256_loadu_ps(data + i);
        _mm256_storeu_ps(data + i, anvil_fast_math::v_expf(x));
    }
    for (; i < len; ++i) {
        data[i] = anvil_fast_math::fast_exp_scalar(data[i]);
    }
#else
    for (int i = 0; i < len; ++i) {
        data[i] = anvil_fast_math::fast_exp_scalar(data[i]);
    }
#endif
}

int simd_get_p_core_count(void) {
#ifdef __linux__
    const std::vector<int> p_cores = detect_p_cores();
    return static_cast<int>(p_cores.size());
#else
    return 0;
#endif
}

int simd_pin_to_p_cores(int n_p_cores) {
#ifdef __linux__
    std::vector<int> p_cores = detect_p_cores();
    if (n_p_cores <= 0) {
        n_p_cores = static_cast<int>(p_cores.size());
    }
    if (n_p_cores <= 0) {
        return 0;
    }
    if (p_cores.empty()) {
        const int cpu_count = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
        if (cpu_count <= 0) {
            return 0;
        }
        for (int i = 0; i < std::min(n_p_cores, cpu_count); ++i) {
            p_cores.push_back(i);
        }
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    const int count = std::min(n_p_cores, static_cast<int>(p_cores.size()));
    for (int i = 0; i < count; ++i) {
        CPU_SET(p_cores[static_cast<std::size_t>(i)], &cpuset);
    }
    const int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    return rc == 0 ? 1 : 0;
#else
    (void)n_p_cores;
    return 0;
#endif
}

// ============================================================
// SSM — Gated Delta Networks (linear recurrence, NO tanh)
// ============================================================

void simd_ssm_step_f32(
    float* h,
    const float* a,
    const float* x_proj,
    int state_dim
) {
    if (h == nullptr || a == nullptr || x_proj == nullptr || state_dim <= 0) {
        return;
    }

    const int n_threads = get_num_threads();
#ifdef __AVX2__
    const int vec_end = state_dim - (state_dim % 8);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
    for (int i = 0; i < vec_end; i += 8) {
        __m256 hv = _mm256_loadu_ps(h + i);
        __m256 av = _mm256_loadu_ps(a + i);
        __m256 xv = _mm256_loadu_ps(x_proj + i);
        hv = _mm256_fmadd_ps(av, hv, xv);
        _mm256_storeu_ps(h + i, hv);
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
    for (int i = vec_end; i < state_dim; ++i) {
        h[i] = a[i] * h[i] + x_proj[i];
    }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
    for (int i = 0; i < state_dim; ++i) {
        h[i] = a[i] * h[i] + x_proj[i];
    }
#endif
}

void simd_ssm_parallel_scan_f32(
    float* H_out,
    const float* a,
    const float* alphas,
    const float* h_init,
    int seq_len,
    int state_dim
) {
    if (H_out == nullptr || a == nullptr || alphas == nullptr || seq_len <= 0 || state_dim <= 0) {
        return;
    }

    std::vector<float> state(static_cast<std::size_t>(state_dim), 0.0f);
    if (h_init != nullptr) {
        std::memcpy(state.data(), h_init, static_cast<std::size_t>(state_dim) * sizeof(float));
    }

    const int n_threads = get_num_threads();
    for (int t = 0; t < seq_len; ++t) {
        const float* alpha_row = alphas + static_cast<std::size_t>(t) * state_dim;
        float* out_row = H_out + static_cast<std::size_t>(t) * state_dim;
#ifdef __AVX2__
        const int vec_end = state_dim - (state_dim % 8);
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
        for (int i = 0; i < vec_end; i += 8) {
            __m256 sv = _mm256_loadu_ps(state.data() + i);
            __m256 av = _mm256_loadu_ps(a + i);
            __m256 xv = _mm256_loadu_ps(alpha_row + i);
            sv = _mm256_fmadd_ps(av, sv, xv);
            _mm256_storeu_ps(state.data() + i, sv);
            _mm256_storeu_ps(out_row + i, sv);
        }
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
        for (int i = vec_end; i < state_dim; ++i) {
            state[i] = a[i] * state[i] + alpha_row[i];
            out_row[i] = state[i];
        }
#else
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) if(state_dim > 2048) schedule(static)
#endif
        for (int i = 0; i < state_dim; ++i) {
            state[i] = a[i] * state[i] + alpha_row[i];
            out_row[i] = state[i];
        }
#endif
    }
}

}  // extern "C"
