/**
 * Full C++ forward-pass execution graph for native QSG inference.
 *
 * Executes the entire transformer decode step in C++ with zero Python
 * in the hot loop.  Eliminates ~3440 ctypes calls per decode step down to 1.
 *
 * Key features:
 *  - Paged KV cache: allocates 4096-token pages lazily (supports 400K+ context)
 *  - Quantized weight dispatch: calls simd_matvec_q4k/q6k/q8_0 directly
 *  - RMSNorm, RoPE, GQA/MQA fused attention, SwiGLU FFN
 *  - Pre-allocated scratch buffers (zero allocation in hot path)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "fast_math.h"
#include "numa_allocator.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declarations for quantized kernels (defined in quantized_matmul.cpp)
extern "C" {

struct GraphPerfStatsSnapshot {
    double embedding_lookup_seconds;
    double attention_proj_seconds;
    double attention_rope_kv_seconds;
    double attention_decode_seconds;
    double attention_out_proj_seconds;
    double ffn_norm_seconds;
    double ffn_gate_up_seconds;
    double ffn_down_seconds;
    double ssm_projection_seconds;
    double ssm_conv_seconds;
    double ssm_recurrent_seconds;
    double ssm_output_seconds;
    double ssm_seconds;
    double moe_seconds;
    double final_norm_seconds;
    double lm_head_seconds;
    double sanitize_seconds;
    int forward_token_calls;
    int forward_token_id_calls;
    int forward_token_ids_calls;
    int forward_token_ids_token_count;
    int attention_calls;
    int ffn_calls;
    int ssm_calls;
    int moe_calls;
    int packed_lm_head_calls;
    int64_t attention_proj_bytes;
    int64_t attention_proj_flops;
    int64_t attention_out_proj_bytes;
    int64_t attention_out_proj_flops;
    int64_t ffn_gate_up_bytes;
    int64_t ffn_gate_up_flops;
    int64_t ffn_down_bytes;
    int64_t ffn_down_flops;
    int64_t ssm_projection_bytes;
    int64_t ssm_projection_flops;
    int64_t ssm_output_bytes;
    int64_t ssm_output_flops;
    int64_t moe_bytes;
    int64_t moe_flops;
    int64_t lm_head_bytes;
    int64_t lm_head_flops;
};

struct GraphDriftConfig {
    int enabled;
    int mode;
    int block_size_tokens;
    int update_interval_tokens;
    int prune_interval_tokens;
    int preserve_head_tokens;
    int preserve_recent_tokens;
    int min_active_tokens;
    float damp_threshold;
    float prune_threshold;
    float damping_strength;
    float hysteresis;
};

struct GraphDriftSnapshot {
    float latest_drift;
    float mean_drift;
    float max_drift;
    float decay_ratio;
    int active_token_count;
    int damped_block_count;
    int pruned_block_count;
    double stabilizer_seconds;
    int stabilizer_calls;
    int mode;
};
void simd_matvec_q4k(const float* x, const void* a_quant, float* y, int k, int n);
void simd_matvec_q4k_r4(const float* x, const void* a_quant, float* y, int k, int n);
void simd_matvec_q6k(const float* x, const void* a_quant, float* y, int k, int n);
void simd_matvec_q6k_r4(const float* x, const void* a_quant, float* y, int k, int n);
void simd_matvec_q6k_lm(const float* x, const void* a_quant, float* y, int k, int n);
void simd_matvec_q8_0(const float* x, const void* a_quant, float* y, int k, int n);
int simd_dequantize_row(
    const void* quant_data,
    int qtype,
    int row,
    float* out,
    int k
);
void simd_fused_expert_swiglu(
    const float* x, int in_dim,
    const void* gate_data, int gate_qtype, int hidden_dim,
    const void* up_data, int up_qtype,
    const void* down_data, int down_qtype, int out_dim,
    float* output
);
void simd_fused_moe_ffn(
    const float* x, int in_dim,
    const int* expert_indices, const float* expert_weights, int top_k,
    const void* const* gate_data_ptrs, int gate_qtype, int hidden_dim,
    const void* const* up_data_ptrs, int up_qtype,
    const void* const* down_data_ptrs, int down_qtype, int out_dim,
    float* output,
    int write_mode
);
int simd_fused_qkv_matvec_quant(
    const float* x,
    const void* q_data, int q_qtype, float* q_out, int q_rows,
    const void* k_data, int k_qtype, float* k_out, int k_rows,
    const void* v_data, int v_qtype, float* v_out, int v_rows,
    int cols
);
int simd_fused_quad_matvec_quant(
    const float* x,
    const void* a_data, int a_qtype, float* a_out, int a_rows,
    const void* b_data, int b_qtype, float* b_out, int b_rows,
    const void* c_data, int c_qtype, float* c_out, int c_rows,
    const void* d_data, int d_qtype, float* d_out, int d_rows,
    int cols
);
int anvil_get_num_threads_for_path(int decode_path);
int anvil_detect_physical_cores();
int anvil_set_thread_affinity(int use_p_cores_only);
int anvil_bind_worker_thread(int worker_tid, int role_decode);
int anvil_get_l3_domain_count();
}

namespace {

inline double perf_now_seconds() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
#endif
}

inline bool strict_numa_enabled() {
    const char* env = std::getenv("ANVIL_NUMA_STRICT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return !(std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0);
}

inline void maybe_bind_worker_thread(bool decode_path) {
#ifdef _OPENMP
    if (!strict_numa_enabled()) {
        return;
    }
    const int tid = omp_in_parallel() ? omp_get_thread_num() : 0;
    thread_local int bound_mode = -1;
    const int mode = decode_path ? 1 : 0;
    if (bound_mode != mode) {
        (void)anvil_bind_worker_thread(tid, mode);
        bound_mode = mode;
    }
#else
    (void)decode_path;
#endif
}

struct GraphPerfStats {
    double embedding_lookup_seconds = 0.0;
    double attention_proj_seconds = 0.0;
    double attention_rope_kv_seconds = 0.0;
    double attention_decode_seconds = 0.0;
    double attention_out_proj_seconds = 0.0;
    double ffn_norm_seconds = 0.0;
    double ffn_gate_up_seconds = 0.0;
    double ffn_down_seconds = 0.0;
    double ssm_projection_seconds = 0.0;
    double ssm_conv_seconds = 0.0;
    double ssm_recurrent_seconds = 0.0;
    double ssm_output_seconds = 0.0;
    double ssm_seconds = 0.0;
    double moe_seconds = 0.0;
    double final_norm_seconds = 0.0;
    double lm_head_seconds = 0.0;
    double sanitize_seconds = 0.0;
    int forward_token_calls = 0;
    int forward_token_id_calls = 0;
    int forward_token_ids_calls = 0;
    int forward_token_ids_token_count = 0;
    int attention_calls = 0;
    int ffn_calls = 0;
    int ssm_calls = 0;
    int moe_calls = 0;
    int packed_lm_head_calls = 0;
    int64_t attention_proj_bytes = 0;
    int64_t attention_proj_flops = 0;
    int64_t attention_out_proj_bytes = 0;
    int64_t attention_out_proj_flops = 0;
    int64_t ffn_gate_up_bytes = 0;
    int64_t ffn_gate_up_flops = 0;
    int64_t ffn_down_bytes = 0;
    int64_t ffn_down_flops = 0;
    int64_t ssm_projection_bytes = 0;
    int64_t ssm_projection_flops = 0;
    int64_t ssm_output_bytes = 0;
    int64_t ssm_output_flops = 0;
    int64_t moe_bytes = 0;
    int64_t moe_flops = 0;
    int64_t lm_head_bytes = 0;
    int64_t lm_head_flops = 0;

    void reset() {
        *this = GraphPerfStats{};
    }
};

constexpr int GRAPH_DRIFT_MODE_TELEMETRY = 0;
constexpr int GRAPH_DRIFT_MODE_CONSERVATIVE = 1;
constexpr int GRAPH_DRIFT_MODE_AGGRESSIVE = 2;
constexpr int GRAPH_DRIFT_OVERHEAD_WINDOW = 128;
constexpr int GRAPH_DRIFT_RECOVERY_STEPS = 256;
constexpr float GRAPH_DRIFT_OVERHEAD_TARGET = 0.15f;
constexpr float GRAPH_DRIFT_OVERHEAD_MAX = 0.20f;

inline int clamp_graph_drift_mode(int mode) {
    if (mode <= GRAPH_DRIFT_MODE_TELEMETRY) {
        return GRAPH_DRIFT_MODE_TELEMETRY;
    }
    if (mode >= GRAPH_DRIFT_MODE_AGGRESSIVE) {
        return GRAPH_DRIFT_MODE_AGGRESSIVE;
    }
    return GRAPH_DRIFT_MODE_CONSERVATIVE;
}

inline GraphDriftConfig default_graph_drift_config() {
    GraphDriftConfig cfg{};
    cfg.enabled = 1;
    cfg.mode = GRAPH_DRIFT_MODE_AGGRESSIVE;
    cfg.block_size_tokens = 128;
    cfg.update_interval_tokens = 64;
    cfg.prune_interval_tokens = 128;
    cfg.preserve_head_tokens = 256;
    cfg.preserve_recent_tokens = 8192;
    cfg.min_active_tokens = 16384;
    cfg.damp_threshold = 0.35f;
    cfg.prune_threshold = 0.72f;
    cfg.damping_strength = 1.2f;
    cfg.hysteresis = 0.05f;
    return cfg;
}

inline GraphDriftSnapshot default_graph_drift_snapshot(int mode) {
    GraphDriftSnapshot snapshot{};
    snapshot.latest_drift = 0.0f;
    snapshot.mean_drift = 0.0f;
    snapshot.max_drift = 0.0f;
    snapshot.decay_ratio = 1.0f;
    snapshot.active_token_count = 0;
    snapshot.damped_block_count = 0;
    snapshot.pruned_block_count = 0;
    snapshot.stabilizer_seconds = 0.0;
    snapshot.stabilizer_calls = 0;
    snapshot.mode = clamp_graph_drift_mode(mode);
    return snapshot;
}

// GGML quantization type IDs (matching GGMLQuantizationType enum values)
constexpr int QTYPE_Q8_0 = 8;
constexpr int QTYPE_Q4_K = 12;
constexpr int QTYPE_Q6_K = 14;
constexpr int QTYPE_Q4_K_R4 = 112;
constexpr int QTYPE_Q6_K_R4 = 114;
constexpr int QTYPE_Q6_K_LM = 214;

inline bool uses_packed_r4_layout(int qtype) {
    return qtype == QTYPE_Q4_K_R4 || qtype == QTYPE_Q6_K_R4 || qtype == QTYPE_Q6_K_LM;
}

constexpr int LAYER_KIND_STANDARD = 0;
constexpr int LAYER_KIND_GRANITE_ATTN = 1;
constexpr int LAYER_KIND_GRANITE_SSM = 2;
constexpr int LAYER_KIND_QWEN_HYBRID = 3;
constexpr int LAYER_KIND_QWEN_FULL_ATTN = 4;

constexpr int KV_PAGE_SIZE = 4096;  // tokens per KV cache page

inline bool is_supported_quant_qtype(int qtype) {
    switch (qtype) {
        case QTYPE_Q4_K:
        case QTYPE_Q4_K_R4:
        case QTYPE_Q6_K:
        case QTYPE_Q6_K_R4:
        case QTYPE_Q6_K_LM:
        case QTYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

inline bool quant_block_layout(int qtype, int* block_bytes, int* block_elems) {
    if (block_bytes == nullptr || block_elems == nullptr) {
        return false;
    }
    switch (qtype) {
        case QTYPE_Q4_K:
        case QTYPE_Q4_K_R4:
            *block_bytes = 144;
            *block_elems = 256;
            return true;
        case QTYPE_Q6_K:
        case QTYPE_Q6_K_R4:
            *block_bytes = 210;
            *block_elems = 256;
            return true;
        case QTYPE_Q6_K_LM:
            *block_bytes = 276;
            *block_elems = 256;
            return true;
        case QTYPE_Q8_0:
            *block_bytes = 34;
            *block_elems = 32;
            return true;
        default:
            return false;
    }
}

inline int64_t estimate_matvec_weight_bytes(int rows, int cols, bool quantized, int qtype) {
    if (rows <= 0 || cols <= 0) {
        return 0;
    }
    if (!quantized) {
        return static_cast<int64_t>(rows) * cols * static_cast<int64_t>(sizeof(float));
    }
    int block_bytes = 0;
    int block_elems = 0;
    if (!quant_block_layout(qtype, &block_bytes, &block_elems)) {
        return static_cast<int64_t>(rows) * cols * static_cast<int64_t>(sizeof(float));
    }
    const int blocks_per_row = (cols + block_elems - 1) / block_elems;
    return static_cast<int64_t>(rows) * blocks_per_row * block_bytes;
}

inline int64_t estimate_matvec_bytes(int rows, int cols, bool quantized, int qtype) {
    if (rows <= 0 || cols <= 0) {
        return 0;
    }
    const int64_t weight_bytes = estimate_matvec_weight_bytes(rows, cols, quantized, qtype);
    const int64_t input_bytes = static_cast<int64_t>(cols) * static_cast<int64_t>(sizeof(float));
    const int64_t output_bytes = static_cast<int64_t>(rows) * static_cast<int64_t>(sizeof(float));
    return weight_bytes + input_bytes + output_bytes;
}

inline int64_t estimate_matvec_flops(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        return 0;
    }
    return static_cast<int64_t>(2) * rows * cols;
}

inline void account_matvec_dispatch(
    int64_t* byte_counter,
    int64_t* flop_counter,
    const float* w_f32,
    const void* w_quant,
    int qtype,
    int rows,
    int cols
) {
    if (byte_counter == nullptr || flop_counter == nullptr || rows <= 0 || cols <= 0) {
        return;
    }
    const bool use_f32 = w_f32 != nullptr;
    const bool use_quant = !use_f32 && w_quant != nullptr && qtype > 0;
    if (!use_f32 && !use_quant) {
        return;
    }
    *byte_counter += estimate_matvec_bytes(rows, cols, use_quant, qtype);
    *flop_counter += estimate_matvec_flops(rows, cols);
}

inline void account_matvec_quant(
    int64_t* byte_counter,
    int64_t* flop_counter,
    int qtype,
    int rows,
    int cols,
    int repeat = 1
) {
    if (byte_counter == nullptr || flop_counter == nullptr || rows <= 0 || cols <= 0 || repeat <= 0) {
        return;
    }
    const int64_t reps = static_cast<int64_t>(repeat);
    *byte_counter += estimate_matvec_bytes(rows, cols, true, qtype) * reps;
    *flop_counter += estimate_matvec_flops(rows, cols) * reps;
}

inline void account_matvec_f32(
    int64_t* byte_counter,
    int64_t* flop_counter,
    int rows,
    int cols,
    int repeat = 1
) {
    if (byte_counter == nullptr || flop_counter == nullptr || rows <= 0 || cols <= 0 || repeat <= 0) {
        return;
    }
    const int64_t reps = static_cast<int64_t>(repeat);
    *byte_counter += estimate_matvec_bytes(rows, cols, false, 0) * reps;
    *flop_counter += estimate_matvec_flops(rows, cols) * reps;
}

// ===== Thread management =====
inline int read_env_threads(const char* name) {
    const char* env = std::getenv(name);
    if (env == nullptr) return 0;
    int n = std::atoi(env);
    return n > 0 ? n : 0;
}

inline int get_num_threads(bool decode_path = true) {
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

inline bool env_flag_enabled(const char* name, const char* expected_value = nullptr) {
    const char* env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    if (expected_value != nullptr) {
        return std::strcmp(env, expected_value) == 0;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "true") == 0 || std::strcmp(env, "on") == 0) {
        return true;
    }
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0 || std::strcmp(env, "off") == 0) {
        return false;
    }
    return true;
}

inline void prefetch_read(const void* ptr, int locality = 3) {
#if defined(__GNUC__) || defined(__clang__)
    if (ptr != nullptr) {
        __builtin_prefetch(ptr, 0, locality);
    }
#else
    (void)ptr;
    (void)locality;
#endif
}

inline void prefetch_write(const void* ptr, int locality = 3) {
#if defined(__GNUC__) || defined(__clang__)
    if (ptr != nullptr) {
        __builtin_prefetch(ptr, 1, locality);
    }
#else
    (void)ptr;
    (void)locality;
#endif
}

inline bool use_q8_kv_cache() {
    const char* env = std::getenv("ANVIL_KV_QUANT");
    if (env == nullptr) {
        return false;
    }
    return std::strcmp(env, "q8") == 0 || env_flag_enabled("ANVIL_KV_QUANT");
}

inline float fast_sigmoid_scalar(float x) {
    const float clamped = std::max(-60.0f, std::min(60.0f, x));
    return 1.0f / (1.0f + anvil_fast_math::fast_exp_scalar(-clamped));
}

template <typename T, std::size_t Alignment = 64>
class AlignedBuffer {
public:
    AlignedBuffer() = default;

    ~AlignedBuffer() {
        const auto opt = anvil::native::anvil_alloc_options_from_env();
        anvil::native::anvil_free_local_cpp(
            data_,
            size_ * sizeof(T),
            opt
        );
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        const auto opt = anvil::native::anvil_alloc_options_from_env();
        anvil::native::anvil_free_local_cpp(
            data_,
            size_ * sizeof(T),
            opt
        );
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    void resize(std::size_t new_size) {
        if (new_size == size_) {
            return;
        }
        const auto opt = anvil::native::anvil_alloc_options_from_env();
        anvil::native::anvil_free_local_cpp(
            data_,
            size_ * sizeof(T),
            opt
        );
        data_ = nullptr;
        size_ = 0;
        if (new_size == 0) {
            return;
        }
        const std::size_t bytes = new_size * sizeof(T);
        const std::size_t rounded = (bytes + (Alignment - 1)) & ~(Alignment - 1);
        auto alloc_opt = anvil::native::anvil_alloc_options_from_env();
        alloc_opt.alignment = std::max<std::size_t>(alloc_opt.alignment, Alignment);
        data_ = static_cast<T*>(
            anvil::native::anvil_alloc_local_cpp(rounded, alloc_opt)
        );
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
        size_ = new_size;
    }

    [[nodiscard]] T* data() { return data_; }
    [[nodiscard]] const T* data() const { return data_; }
    [[nodiscard]] std::size_t size() const { return size_; }
    T& operator[](std::size_t index) { return data_[index]; }
    const T& operator[](std::size_t index) const { return data_[index]; }

private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};

inline int detect_l1d_cache_kb() {
#ifdef __linux__
    std::ifstream input("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (input.is_open()) {
        std::string raw;
        input >> raw;
        if (!raw.empty()) {
            const int value = std::atoi(raw.c_str());
            if (value > 0) {
                if (raw.back() == 'M' || raw.back() == 'm') {
                    return value * 1024;
                }
                return value;
            }
        }
    }
#endif
    return 32;
}

inline int compute_attention_tile_size(int head_dim) {
    const int l1d_kb = std::max(16, detect_l1d_cache_kb());
    const int bytes_per_tile = std::max(1, head_dim * static_cast<int>(sizeof(float)) * 3);
    int tile = (l1d_kb * 1024) / bytes_per_tile;
    tile = std::max(16, std::min(256, tile));
    tile -= tile % 8;
    return std::max(16, tile);
}

inline float fast_softplus_scalar(float x) {
    const float clamped = std::max(-60.0f, std::min(60.0f, x));
    return std::log1p(anvil_fast_math::fast_exp_scalar(-std::fabs(clamped)))
        + std::max(clamped, 0.0f);
}

// ===== AVX2 helpers =====
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

inline __attribute__((always_inline)) void mul_silu_gate_inplace(
    float* x,
    const float* gate,
    int len
) {
    if (x == nullptr || gate == nullptr || len <= 0) {
        return;
    }
#ifdef __AVX2__
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 gv = _mm256_loadu_ps(gate + i);
        xv = _mm256_mul_ps(xv, anvil_fast_math::v_silu(gv));
        _mm256_storeu_ps(x + i, xv);
    }
    for (; i < len; ++i) {
        x[i] *= gate[i] * fast_sigmoid_scalar(gate[i]);
    }
#else
    for (int i = 0; i < len; ++i) {
        x[i] *= gate[i] * fast_sigmoid_scalar(gate[i]);
    }
#endif
}

// ===== Core compute primitives =====

void rmsnorm_inplace(float* x, const float* gamma, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i)
        sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);
#ifdef __AVX2__
    __m256 scale = _mm256_set1_ps(inv_rms);
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m256 gv = _mm256_loadu_ps(gamma + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_mul_ps(xv, scale), gv));
    }
    for (; i < dim; ++i)
        x[i] = x[i] * inv_rms * gamma[i];
#else
    for (int i = 0; i < dim; ++i)
        x[i] = x[i] * inv_rms * gamma[i];
#endif
}

void rmsnorm_copy(const float* src, const float* gamma, float* dst, int dim, float eps) {
    std::memcpy(dst, src, static_cast<std::size_t>(dim) * sizeof(float));
    rmsnorm_inplace(dst, gamma, dim, eps);
}

// Float32 matvec: y = W @ x, W is [rows, cols]
void matvec_f32(const float* W, const float* x, float* y, int rows, int cols) {
    int n_threads = get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        maybe_bind_worker_thread(true);
        const float* w_row = W + static_cast<std::size_t>(r) * cols;
        float acc = 0.0f;
#ifdef __AVX2__
        __m256 vacc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 8 <= cols; c += 8) {
            __m256 wv = _mm256_loadu_ps(w_row + c);
            __m256 xv = _mm256_loadu_ps(x + c);
            vacc = _mm256_fmadd_ps(wv, xv, vacc);
        }
        acc = hsum256_ps(vacc);
        for (; c < cols; ++c)
            acc += w_row[c] * x[c];
#else
        for (int c = 0; c < cols; ++c)
            acc += w_row[c] * x[c];
#endif
        y[r] = acc;
    }
}

// Quantized matvec dispatcher: calls the right kernel based on qtype
// y = W_quant @ x, where W_quant is [rows, cols] in packed format
inline void matvec_quant(const void* W_quant, int qtype,
                         const float* x, float* y, int cols, int rows) {
    // The simd_matvec functions use (x, W, y, in_dim, out_dim) convention
    switch (qtype) {
        case QTYPE_Q4_K: simd_matvec_q4k(x, W_quant, y, cols, rows); break;
        case QTYPE_Q4_K_R4: simd_matvec_q4k_r4(x, W_quant, y, cols, rows); break;
        case QTYPE_Q6_K: simd_matvec_q6k(x, W_quant, y, cols, rows); break;
        case QTYPE_Q6_K_R4: simd_matvec_q6k_r4(x, W_quant, y, cols, rows); break;
        case QTYPE_Q6_K_LM: simd_matvec_q6k_lm(x, W_quant, y, cols, rows); break;
        case QTYPE_Q8_0: simd_matvec_q8_0(x, W_quant, y, cols, rows); break;
        default:
            // Unsupported qtype — zero output
            std::memset(y, 0, static_cast<std::size_t>(rows) * sizeof(float));
            break;
    }
}

// Unified matvec: dispatches to float32 or quantized based on what's available
inline void matvec_dispatch(const float* W_f32, const void* W_quant, int qtype,
                            const float* x, float* y, int rows, int cols) {
    if (W_f32 != nullptr) {
        matvec_f32(W_f32, x, y, rows, cols);
    } else if (W_quant != nullptr && qtype > 0) {
        matvec_quant(W_quant, qtype, x, y, cols, rows);
    } else {
        std::memset(y, 0, static_cast<std::size_t>(rows) * sizeof(float));
    }
}

inline bool fused_qkv_matvec_dispatch(
    const float* x,
    const void* q_data, int q_qtype, float* q, int q_rows,
    const void* k_data, int k_qtype, float* k, int k_rows,
    const void* v_data, int v_qtype, float* v, int v_rows,
    int dim
) {
    return simd_fused_qkv_matvec_quant(
        x,
        q_data, q_qtype, q, q_rows,
        k_data, k_qtype, k, k_rows,
        v_data, v_qtype, v, v_rows,
        dim
    ) == 1;
}

inline bool fused_quad_matvec_dispatch(
    const float* x,
    const void* a_data, int a_qtype, float* a_out, int a_rows,
    const void* b_data, int b_qtype, float* b_out, int b_rows,
    const void* c_data, int c_qtype, float* c_out, int c_rows,
    const void* d_data, int d_qtype, float* d_out, int d_rows,
    int cols
) {
    return simd_fused_quad_matvec_quant(
        x,
        a_data, a_qtype, a_out, a_rows,
        b_data, b_qtype, b_out, b_rows,
        c_data, c_qtype, c_out, c_rows,
        d_data, d_qtype, d_out, d_rows,
        cols
    ) == 1;
}

inline void scale_inplace(float* x, int len, float scale) {
    if (x == nullptr || len <= 0) {
        return;
    }
#ifdef __AVX2__
    const __m256 sv = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(xv, sv));
    }
    for (; i < len; ++i) {
        x[i] *= scale;
    }
#else
    for (int i = 0; i < len; ++i) {
        x[i] *= scale;
    }
#endif
}

inline void add_residual_inplace(float* state, const float* branch, int len, float residual_scale) {
    if (state == nullptr || branch == nullptr || len <= 0) {
        return;
    }
#ifdef __AVX2__
    const __m256 rv = _mm256_set1_ps(residual_scale == 0.0f ? 1.0f : residual_scale);
    int i = 0;
    if (residual_scale != 0.0f) {
        for (; i + 8 <= len; i += 8) {
            __m256 sv = _mm256_loadu_ps(state + i);
            __m256 bv = _mm256_loadu_ps(branch + i);
            sv = _mm256_fmadd_ps(bv, rv, sv);
            _mm256_storeu_ps(state + i, sv);
        }
        for (; i < len; ++i) {
            state[i] += branch[i] * residual_scale;
        }
    } else {
        for (; i + 8 <= len; i += 8) {
            __m256 sv = _mm256_loadu_ps(state + i);
            __m256 bv = _mm256_loadu_ps(branch + i);
            _mm256_storeu_ps(state + i, _mm256_add_ps(sv, bv));
        }
        for (; i < len; ++i) {
            state[i] += branch[i];
        }
    }
#else
    if (residual_scale != 0.0f) {
        for (int i = 0; i < len; ++i) {
            state[i] += branch[i] * residual_scale;
        }
    } else {
        for (int i = 0; i < len; ++i) {
            state[i] += branch[i];
        }
    }
#endif
}

inline float dot_f32_avx2(const float* a, const float* b, int len) {
    float acc = 0.0f;
#ifdef __AVX2__
    __m256 vacc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        vacc = _mm256_fmadd_ps(av, bv, vacc);
    }
    acc += hsum256_ps(vacc);
    for (; i < len; ++i) {
        acc += a[i] * b[i];
    }
#else
    for (int i = 0; i < len; ++i) {
        acc += a[i] * b[i];
    }
#endif
    return acc;
}

inline int granite_router_top_k_streaming(
    const float* router,
    const float* x,
    int expert_count,
    int dim,
    int top_k,
    int* expert_indices,
    float* expert_scores
) {
    int selected = 0;
    for (int i = 0; i < top_k; ++i) {
        expert_indices[i] = 0;
        expert_scores[i] = -INFINITY;
    }
    for (int expert = 0; expert < expert_count; ++expert) {
        const float score = dot_f32_avx2(
            router + static_cast<std::size_t>(expert) * dim,
            x,
            dim
        );
        if (top_k == 1) {
            if (score > expert_scores[0]) {
                expert_scores[0] = score;
                expert_indices[0] = expert;
                selected = 1;
            }
            continue;
        }
        if (top_k == 2) {
            if (score > expert_scores[0]) {
                expert_scores[1] = expert_scores[0];
                expert_indices[1] = expert_indices[0];
                expert_scores[0] = score;
                expert_indices[0] = expert;
                selected = std::min(2, selected + 1);
            } else if (score > expert_scores[1]) {
                expert_scores[1] = score;
                expert_indices[1] = expert;
                selected = std::min(2, std::max(1, selected + 1));
            }
            continue;
        }
        if (selected >= top_k && score <= expert_scores[top_k - 1]) {
            continue;
        }
        int insert_at = std::min(selected, top_k - 1);
        while (insert_at > 0 && expert_scores[insert_at - 1] < score) {
            insert_at -= 1;
        }
        const int upper = std::min(selected, top_k - 1);
        for (int j = upper; j > insert_at; --j) {
            expert_scores[j] = expert_scores[j - 1];
            expert_indices[j] = expert_indices[j - 1];
        }
        expert_scores[insert_at] = score;
        expert_indices[insert_at] = expert;
        if (selected < top_k) {
            selected += 1;
        }
    }
    return selected;
}

inline void normalize_pair_inplace(float* q, float* k, int len) {
    if (q == nullptr || k == nullptr || len <= 0) {
        return;
    }
    float q_norm = 1.0e-6f;
    float k_norm = 1.0e-6f;
#ifdef __AVX2__
    __m256 q_acc = _mm256_setzero_ps();
    __m256 k_acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q + i);
        const __m256 kv = _mm256_loadu_ps(k + i);
        q_acc = _mm256_fmadd_ps(qv, qv, q_acc);
        k_acc = _mm256_fmadd_ps(kv, kv, k_acc);
    }
    q_norm += hsum256_ps(q_acc);
    k_norm += hsum256_ps(k_acc);
    for (; i < len; ++i) {
        q_norm += q[i] * q[i];
        k_norm += k[i] * k[i];
    }
    const float q_inv = 1.0f / std::sqrt(q_norm);
    const float k_inv = 1.0f / std::sqrt(k_norm);
    const __m256 q_scale = _mm256_set1_ps(q_inv);
    const __m256 k_scale = _mm256_set1_ps(k_inv);
    i = 0;
    for (; i + 8 <= len; i += 8) {
        __m256 qv = _mm256_loadu_ps(q + i);
        __m256 kv = _mm256_loadu_ps(k + i);
        _mm256_storeu_ps(q + i, _mm256_mul_ps(qv, q_scale));
        _mm256_storeu_ps(k + i, _mm256_mul_ps(kv, k_scale));
    }
    for (; i < len; ++i) {
        q[i] *= q_inv;
        k[i] *= k_inv;
    }
#else
    for (int i = 0; i < len; ++i) {
        q_norm += q[i] * q[i];
        k_norm += k[i] * k[i];
    }
    q_norm = std::sqrt(q_norm);
    k_norm = std::sqrt(k_norm);
    for (int i = 0; i < len; ++i) {
        q[i] /= q_norm;
        k[i] /= k_norm;
    }
#endif
}

// SwiGLU activation: out = silu(gate) * up
void swiglu_f32(const float* gate, const float* up, float* out, int dim) {
#ifdef __AVX2__
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= dim; i += 8) {
        __m256 gv = _mm256_loadu_ps(gate + i);
        __m256 uv = _mm256_loadu_ps(up + i);
        __m256 neg_g = _mm256_sub_ps(_mm256_setzero_ps(), gv);
        __m256 exp_neg = anvil_fast_math::v_expf(neg_g);
        __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        __m256 silu = _mm256_mul_ps(gv, sig);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(silu, uv));
    }
    for (; i < dim; ++i) {
        float g = gate[i];
        out[i] = (g / (1.0f + anvil_fast_math::fast_exp_scalar(-g))) * up[i];
    }
#else
    for (int i = 0; i < dim; ++i) {
        float g = gate[i];
        out[i] = (g / (1.0f + anvil_fast_math::fast_exp_scalar(-g))) * up[i];
    }
#endif
}

void sigmoid_inplace(float* x, int dim) {
    if (x == nullptr || dim <= 0) return;
#ifdef __AVX2__
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= dim; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 exp_neg = anvil_fast_math::v_expf(
            _mm256_sub_ps(_mm256_setzero_ps(), xv)
        );
        const __m256 out = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        _mm256_storeu_ps(x + i, out);
    }
    for (; i < dim; ++i) {
        x[i] = fast_sigmoid_scalar(x[i]);
    }
#else
    for (int i = 0; i < dim; ++i) {
        x[i] = fast_sigmoid_scalar(x[i]);
    }
#endif
}

void silu_inplace(float* x, int dim) {
    if (x == nullptr || dim <= 0) return;
#ifdef __AVX2__
    int i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= dim; i += 8) {
        const __m256 xv = _mm256_loadu_ps(x + i);
        const __m256 exp_neg = anvil_fast_math::v_expf(
            _mm256_sub_ps(_mm256_setzero_ps(), xv)
        );
        const __m256 sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        _mm256_storeu_ps(x + i, _mm256_mul_ps(xv, sig));
    }
    for (; i < dim; ++i) {
        x[i] = x[i] * fast_sigmoid_scalar(x[i]);
    }
#else
    for (int i = 0; i < dim; ++i) {
        x[i] = x[i] * fast_sigmoid_scalar(x[i]);
    }
#endif
}

void softplus_inplace(float* x, int dim) {
    if (x == nullptr || dim <= 0) return;
    for (int i = 0; i < dim; ++i) {
        x[i] = fast_softplus_scalar(x[i]);
    }
}

void rmsnorm_rows_inplace(float* x, const float* gamma, int rows, int dim, float eps) {
    if (x == nullptr || gamma == nullptr || rows <= 0 || dim <= 0) return;
#ifdef _OPENMP
    int n_threads = get_num_threads();
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int r = 0; r < rows; ++r) {
        rmsnorm_inplace(x + static_cast<std::size_t>(r) * dim, gamma, dim, eps);
    }
}

void sanitize_tensor_inplace(float* x, int len, float clamp_abs = 1.0e6f) {
    if (x == nullptr || len <= 0) return;
#ifdef __AVX2__
    const __m256 clamp_v = _mm256_set1_ps(clamp_abs);
    const __m256 neg_clamp_v = _mm256_set1_ps(-clamp_abs);
    const __m256 zero_v = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 abs_v = _mm256_andnot_ps(sign_mask, v);
        const __m256 unordered = _mm256_cmp_ps(v, v, _CMP_UNORD_Q);
        const __m256 out_of_range = _mm256_cmp_ps(abs_v, clamp_v, _CMP_GT_OQ);
        if (_mm256_movemask_ps(_mm256_or_ps(unordered, out_of_range)) == 0) {
            continue;
        }
        const __m256 clamped =
            _mm256_min_ps(_mm256_max_ps(v, neg_clamp_v), clamp_v);
        const __m256 cleaned = _mm256_blendv_ps(clamped, zero_v, unordered);
        _mm256_storeu_ps(x + i, cleaned);
    }
    for (; i < len; ++i) {
        float v = x[i];
        if (!std::isfinite(v)) {
            x[i] = 0.0f;
        } else if (v > clamp_abs) {
            x[i] = clamp_abs;
        } else if (v < -clamp_abs) {
            x[i] = -clamp_abs;
        }
    }
    return;
#endif
    for (int i = 0; i < len; ++i) {
        float v = x[i];
        if (!std::isfinite(v)) {
            x[i] = 0.0f;
        } else if (v > clamp_abs) {
            x[i] = clamp_abs;
        } else if (v < -clamp_abs) {
            x[i] = -clamp_abs;
        }
    }
}

void scale_and_sanitize_inplace(float* x, int len, float scale, float clamp_abs = 1.0e6f) {
    if (x == nullptr || len <= 0) return;
    if (scale == 1.0f) {
        sanitize_tensor_inplace(x, len, clamp_abs);
        return;
    }
#ifdef __AVX2__
    const __m256 scale_v = _mm256_set1_ps(scale);
    const __m256 clamp_v = _mm256_set1_ps(clamp_abs);
    const __m256 neg_clamp_v = _mm256_set1_ps(-clamp_abs);
    const __m256 zero_v = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= len; i += 8) {
        const __m256 scaled = _mm256_mul_ps(_mm256_loadu_ps(x + i), scale_v);
        const __m256 unordered = _mm256_cmp_ps(scaled, scaled, _CMP_UNORD_Q);
        const __m256 clamped = _mm256_min_ps(
            _mm256_max_ps(scaled, neg_clamp_v),
            clamp_v
        );
        const __m256 cleaned = _mm256_blendv_ps(clamped, zero_v, unordered);
        _mm256_storeu_ps(x + i, cleaned);
    }
    for (; i < len; ++i) {
        float v = x[i] * scale;
        if (!std::isfinite(v)) {
            x[i] = 0.0f;
        } else if (v > clamp_abs) {
            x[i] = clamp_abs;
        } else if (v < -clamp_abs) {
            x[i] = -clamp_abs;
        } else {
            x[i] = v;
        }
    }
    return;
#endif
    for (int i = 0; i < len; ++i) {
        float v = x[i] * scale;
        if (!std::isfinite(v)) {
            x[i] = 0.0f;
        } else if (v > clamp_abs) {
            x[i] = clamp_abs;
        } else if (v < -clamp_abs) {
            x[i] = -clamp_abs;
        } else {
            x[i] = v;
        }
    }
}

// Softmax with online max tracking (numerically stable) — AVX2 vectorized
void softmax_inplace(float* scores, int len) {
    if (len <= 0) return;
    float max_val = scores[0];
    for (int i = 1; i < len; ++i)
        max_val = std::max(max_val, scores[i]);
    float sum = 0.0f;
    int i = 0;
#ifdef __AVX2__
    {
        __m256 max_vec = _mm256_set1_ps(max_val);
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 8 <= len; i += 8) {
            __m256 sv = _mm256_loadu_ps(scores + i);
            sv = anvil_fast_math::v_expf(_mm256_sub_ps(sv, max_vec));
            sum_vec = _mm256_add_ps(sum_vec, sv);
            _mm256_storeu_ps(scores + i, sv);
        }
        sum = hsum256_ps(sum_vec);
    }
#endif
    for (; i < len; ++i) {
        scores[i] = anvil_fast_math::fast_exp_scalar(scores[i] - max_val);
        sum += scores[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
#ifdef __AVX2__
        {
            __m256 inv_vec = _mm256_set1_ps(inv);
            int j = 0;
            for (; j + 8 <= len; j += 8) {
                __m256 sv = _mm256_loadu_ps(scores + j);
                _mm256_storeu_ps(scores + j, _mm256_mul_ps(sv, inv_vec));
            }
            for (; j < len; ++j)
                scores[j] *= inv;
        }
#else
        for (int j = 0; j < len; ++j)
            scores[j] *= inv;
#endif
    }
}

// RoPE for single token position
void apply_rope_with_dim(float* q, float* k, int n_heads, int n_kv_heads,
                         int head_dim, int rope_dim, int pos, float theta) {
    rope_dim = std::max(0, std::min(head_dim, rope_dim));
    rope_dim -= (rope_dim % 2);
    if (rope_dim <= 0) {
        return;
    }

    const int safe_head_dim = std::max(1, head_dim);
    const int pair_count = rope_dim / 2;
    const float safe_theta = (std::isfinite(theta) && theta > 0.0f) ? theta : 10000.0f;

    struct RopeRotationCache {
        int head_dim = -1;
        int rope_dim = -1;
        int pos = std::numeric_limits<int>::min();
        float theta = 0.0f;
        std::vector<float> inv_freq;
        std::vector<float> cos_pair;
        std::vector<float> sin_pair;
        std::vector<float> cos_lanes;
        std::vector<float> sin_lanes;
    };
    static thread_local RopeRotationCache cache;

    const bool shape_changed =
        cache.head_dim != safe_head_dim
        || cache.rope_dim != rope_dim
        || cache.theta != safe_theta;
    if (shape_changed) {
        cache.head_dim = safe_head_dim;
        cache.rope_dim = rope_dim;
        cache.theta = safe_theta;
        cache.pos = std::numeric_limits<int>::min();
        cache.inv_freq.assign(static_cast<std::size_t>(pair_count), 1.0f);
        cache.cos_pair.assign(static_cast<std::size_t>(pair_count), 1.0f);
        cache.sin_pair.assign(static_cast<std::size_t>(pair_count), 0.0f);
        cache.cos_lanes.assign(static_cast<std::size_t>(rope_dim), 1.0f);
        cache.sin_lanes.assign(static_cast<std::size_t>(rope_dim), 0.0f);

        if (pair_count > 1) {
            const float freq_step = std::pow(
                safe_theta,
                -2.0f / static_cast<float>(safe_head_dim)
            );
            for (int p = 1; p < pair_count; ++p) {
                cache.inv_freq[static_cast<std::size_t>(p)] =
                    cache.inv_freq[static_cast<std::size_t>(p - 1)] * freq_step;
            }
        }
    }

    if (shape_changed || cache.pos != pos) {
        for (int p = 0; p < pair_count; ++p) {
            const float angle = static_cast<float>(pos) * cache.inv_freq[static_cast<std::size_t>(p)];
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            cache.cos_pair[static_cast<std::size_t>(p)] = c;
            cache.sin_pair[static_cast<std::size_t>(p)] = s;
            const std::size_t lane = static_cast<std::size_t>(p) * 2;
            cache.cos_lanes[lane] = c;
            cache.cos_lanes[lane + 1] = c;
            cache.sin_lanes[lane] = s;
            cache.sin_lanes[lane + 1] = s;
        }
        cache.pos = pos;
    }

#ifdef __AVX2__
    const __m256 rope_alt_sign = _mm256_setr_ps(
        -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f
    );
#endif
    auto rotate = [&](float* vec, int heads) {
        for (int h = 0; h < heads; ++h) {
            float* ptr = vec + static_cast<std::size_t>(h) * head_dim;
            int i = 0;
#ifdef __AVX2__
            for (; i + 8 <= rope_dim; i += 8) {
                const __m256 xv = _mm256_loadu_ps(ptr + i);
                const __m256 xsw = _mm256_permute_ps(xv, 0xB1);
                const __m256 cv = _mm256_loadu_ps(cache.cos_lanes.data() + i);
                const __m256 sv = _mm256_loadu_ps(cache.sin_lanes.data() + i);
                const __m256 signed_xsw =
                    _mm256_mul_ps(xsw, _mm256_mul_ps(sv, rope_alt_sign));
                const __m256 outv = _mm256_fmadd_ps(xv, cv, signed_xsw);
                _mm256_storeu_ps(ptr + i, outv);
            }
#endif
            for (; i < rope_dim; i += 2) {
                const int p = i / 2;
                const float c = cache.cos_pair[static_cast<std::size_t>(p)];
                const float s = cache.sin_pair[static_cast<std::size_t>(p)];
                float x0 = ptr[i], x1 = ptr[i + 1];
                ptr[i] = x0 * c - x1 * s;
                ptr[i + 1] = x0 * s + x1 * c;
            }
        }
    };
    rotate(q, n_heads);
    rotate(k, n_kv_heads);
}

void apply_rope(float* q, float* k, int n_heads, int n_kv_heads,
                int head_dim, int pos, float theta) {
    apply_rope_with_dim(q, k, n_heads, n_kv_heads, head_dim, head_dim, pos, theta);
}

// Fused GQA attention for single query token (decode hot path)
// q: [n_heads, head_dim], k_cache: [kv_len, n_kv_heads, head_dim],
// v_cache: [kv_len, n_kv_heads, head_dim], out: [n_heads, head_dim]
void fused_gqa_attention_decode(
    const float* q, const float* k_cache, const float* v_cache,
    float* out, int n_heads, int n_kv_heads, int kv_len,
    int head_dim, float scale) {
    if (kv_len <= 0 || head_dim <= 0) return;
    int heads_per_kv = std::max(1, n_heads / std::max(1, n_kv_heads));

    int n_threads = get_num_threads();
    const int scratch_threads = std::max(1, n_threads);
    std::vector<float> score_scratch(
        static_cast<std::size_t>(scratch_threads) * static_cast<std::size_t>(kv_len),
        0.0f
    );
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int h = 0; h < n_heads; ++h) {
        maybe_bind_worker_thread(true);
        int kv_h = std::min(n_kv_heads - 1, h / heads_per_kv);
        const float* q_h = q + static_cast<std::size_t>(h) * head_dim;

        int thread_idx = 0;
#ifdef _OPENMP
        thread_idx = omp_get_thread_num();
#endif
        float* scores = score_scratch.data()
            + static_cast<std::size_t>(thread_idx) * static_cast<std::size_t>(kv_len);

        for (int j = 0; j < kv_len; ++j) {
            const float* k_j = k_cache + (static_cast<std::size_t>(j) * n_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
#ifdef __AVX2__
            __m256 vacc = _mm256_setzero_ps();
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 qv = _mm256_loadu_ps(q_h + d);
                __m256 kv = _mm256_loadu_ps(k_j + d);
                vacc = _mm256_fmadd_ps(qv, kv, vacc);
            }
            dot = hsum256_ps(vacc);
            for (; d < head_dim; ++d)
                dot += q_h[d] * k_j[d];
#else
            for (int d = 0; d < head_dim; ++d)
                dot += q_h[d] * k_j[d];
#endif
            scores[j] = dot * scale;
        }

        softmax_inplace(scores, kv_len);

        float* out_h = out + static_cast<std::size_t>(h) * head_dim;
        std::memset(out_h, 0, static_cast<std::size_t>(head_dim) * sizeof(float));
        for (int j = 0; j < kv_len; ++j) {
            const float* v_j = v_cache + (static_cast<std::size_t>(j) * n_kv_heads + kv_h) * head_dim;
            float w = scores[j];
#ifdef __AVX2__
            __m256 wv = _mm256_set1_ps(w);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(out_h + d);
                __m256 vv = _mm256_loadu_ps(v_j + d);
                ov = _mm256_fmadd_ps(wv, vv, ov);
                _mm256_storeu_ps(out_h + d, ov);
            }
            for (; d < head_dim; ++d)
                out_h[d] += w * v_j[d];
#else
            for (int d = 0; d < head_dim; ++d)
                out_h[d] += w * v_j[d];
#endif
        }
    }
}

// ===== Paged KV Cache =====

// Each page stores KV_PAGE_SIZE tokens of KV data for ONE layer.
// Float mode layout: [KV_PAGE_SIZE * kv_dim] float
// Q8 mode layout:    [KV_PAGE_SIZE * kv_dim] int8 + [KV_PAGE_SIZE * n_kv_heads] scales
struct PagedKVStore {
    int max_seq;
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int kv_dim;         // n_kv_heads * head_dim
    int page_size;      // tokens per page (KV_PAGE_SIZE)
    int total_pages;    // max_seq / page_size (rounded up)
    bool q8_enabled;

    // Pages indexed as [layer_idx * total_pages + page_idx]
    // nullptr = not yet allocated (lazy)
    std::vector<float*> k_pages;
    std::vector<float*> v_pages;
    std::vector<std::int8_t*> k_pages_q8;
    std::vector<std::int8_t*> v_pages_q8;
    std::vector<float*> k_scales_pages;
    std::vector<float*> v_scales_pages;
    // Backing storage for allocated pages (owns the memory)
    std::vector<std::vector<float>> k_storage;
    std::vector<std::vector<float>> v_storage;
    std::vector<std::vector<std::int8_t>> k_storage_q8;
    std::vector<std::vector<std::int8_t>> v_storage_q8;
    std::vector<std::vector<float>> k_scales_storage;
    std::vector<std::vector<float>> v_scales_storage;
    std::vector<int> owner_l3_domain;
    std::vector<int> owner_numa_node;

    struct Snapshot {
        std::vector<std::vector<float>> k_storage;
        std::vector<std::vector<float>> v_storage;
        std::vector<std::vector<std::int8_t>> k_storage_q8;
        std::vector<std::vector<std::int8_t>> v_storage_q8;
        std::vector<std::vector<float>> k_scales_storage;
        std::vector<std::vector<float>> v_scales_storage;
        std::vector<int> owner_l3_domain;
        std::vector<int> owner_numa_node;
    };

    PagedKVStore(int seq, int layers, int kv_heads, int h_dim, bool use_q8)
        : max_seq(seq), n_layers(layers), n_kv_heads(kv_heads), head_dim(h_dim),
          kv_dim(kv_heads * h_dim), page_size(KV_PAGE_SIZE), q8_enabled(use_q8) {
        total_pages = (max_seq + page_size - 1) / page_size;
        std::size_t n_slots = static_cast<std::size_t>(n_layers) * total_pages;
        k_pages.resize(n_slots, nullptr);
        v_pages.resize(n_slots, nullptr);
        k_pages_q8.resize(n_slots, nullptr);
        v_pages_q8.resize(n_slots, nullptr);
        k_scales_pages.resize(n_slots, nullptr);
        v_scales_pages.resize(n_slots, nullptr);
        k_storage.resize(n_slots);
        v_storage.resize(n_slots);
        k_storage_q8.resize(n_slots);
        v_storage_q8.resize(n_slots);
        k_scales_storage.resize(n_slots);
        v_scales_storage.resize(n_slots);
        owner_l3_domain.resize(n_slots, -1);
        owner_numa_node.resize(n_slots, -1);
    }

    // Ensure a page is allocated for (layer, page_idx)
    void ensure_page(int layer, int page_idx) {
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (!q8_enabled) {
            if (k_pages[slot] == nullptr) {
                std::size_t page_floats = static_cast<std::size_t>(page_size) * kv_dim;
                k_storage[slot].resize(page_floats, 0.0f);
                v_storage[slot].resize(page_floats, 0.0f);
                k_pages[slot] = k_storage[slot].data();
                v_pages[slot] = v_storage[slot].data();
                anvil::native::anvil_numa_advise_region(
                    k_pages[slot],
                    page_floats * sizeof(float)
                );
                anvil::native::anvil_numa_advise_region(
                    v_pages[slot],
                    page_floats * sizeof(float)
                );
                const int domain_count = std::max(1, anvil_get_l3_domain_count());
                owner_l3_domain[slot] = page_idx % domain_count;
                owner_numa_node[slot] = 0;
            }
            return;
        }
        if (k_pages_q8[slot] == nullptr) {
            std::size_t page_values = static_cast<std::size_t>(page_size) * kv_dim;
            std::size_t page_scales = static_cast<std::size_t>(page_size) * n_kv_heads;
            k_storage_q8[slot].resize(page_values, 0);
            v_storage_q8[slot].resize(page_values, 0);
            k_scales_storage[slot].resize(page_scales, 1.0f);
            v_scales_storage[slot].resize(page_scales, 1.0f);
            k_pages_q8[slot] = k_storage_q8[slot].data();
            v_pages_q8[slot] = v_storage_q8[slot].data();
            k_scales_pages[slot] = k_scales_storage[slot].data();
            v_scales_pages[slot] = v_scales_storage[slot].data();
            anvil::native::anvil_numa_advise_region(
                k_pages_q8[slot],
                page_values * sizeof(std::int8_t)
            );
            anvil::native::anvil_numa_advise_region(
                v_pages_q8[slot],
                page_values * sizeof(std::int8_t)
            );
            const int domain_count = std::max(1, anvil_get_l3_domain_count());
            owner_l3_domain[slot] = page_idx % domain_count;
            owner_numa_node[slot] = 0;
        }
    }

    // Write KV for a single token at position pos in layer
    void write(int layer, int pos, const float* k, const float* v, int local_kv_heads) {
        int page_idx = pos / page_size;
        int offset = pos % page_size;
        ensure_page(layer, page_idx);
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        std::size_t off = static_cast<std::size_t>(offset) * kv_dim;
        const int kv_heads = std::max(1, std::min(local_kv_heads, n_kv_heads));
        const int local_kv_dim = kv_heads * head_dim;
        if (!q8_enabled) {
            std::memset(k_pages[slot] + off, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));
            std::memset(v_pages[slot] + off, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));
            std::memcpy(k_pages[slot] + off, k, static_cast<std::size_t>(local_kv_dim) * sizeof(float));
            std::memcpy(v_pages[slot] + off, v, static_cast<std::size_t>(local_kv_dim) * sizeof(float));
            return;
        }

        const std::size_t scale_off = static_cast<std::size_t>(offset) * n_kv_heads;
        std::int8_t* k_dst = k_pages_q8[slot] + off;
        std::int8_t* v_dst = v_pages_q8[slot] + off;
        float* k_scales = k_scales_pages[slot] + scale_off;
        float* v_scales = v_scales_pages[slot] + scale_off;
        std::memset(k_dst, 0, static_cast<std::size_t>(kv_dim) * sizeof(std::int8_t));
        std::memset(v_dst, 0, static_cast<std::size_t>(kv_dim) * sizeof(std::int8_t));
        std::fill_n(k_scales, n_kv_heads, 0.0f);
        std::fill_n(v_scales, n_kv_heads, 0.0f);
        for (int h = 0; h < kv_heads; ++h) {
            const float* k_src_h = k + static_cast<std::size_t>(h) * head_dim;
            const float* v_src_h = v + static_cast<std::size_t>(h) * head_dim;
            float k_abs_max = 0.0f;
            float v_abs_max = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                k_abs_max = std::max(k_abs_max, std::fabs(k_src_h[d]));
                v_abs_max = std::max(v_abs_max, std::fabs(v_src_h[d]));
            }
            const float k_scale = (k_abs_max > 0.0f) ? (k_abs_max / 127.0f) : (1.0f / 127.0f);
            const float v_scale = (v_abs_max > 0.0f) ? (v_abs_max / 127.0f) : (1.0f / 127.0f);
            k_scales[h] = k_scale;
            v_scales[h] = v_scale;
            const float k_inv = 1.0f / k_scale;
            const float v_inv = 1.0f / v_scale;
            std::int8_t* k_dst_h = k_dst + static_cast<std::size_t>(h) * head_dim;
            std::int8_t* v_dst_h = v_dst + static_cast<std::size_t>(h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                const float kq = std::round(k_src_h[d] * k_inv);
                const float vq = std::round(v_src_h[d] * v_inv);
                const int kqi = static_cast<int>(kq);
                const int vqi = static_cast<int>(vq);
                const int kclamped = std::max(-127, std::min(127, kqi));
                const int vclamped = std::max(-127, std::min(127, vqi));
                k_dst_h[d] = static_cast<std::int8_t>(kclamped);
                v_dst_h[d] = static_cast<std::int8_t>(vclamped);
            }
        }
    }

    // Get K pointer for a specific position in a layer
    const float* get_k(int layer, int pos) const {
        if (q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (k_pages[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return k_pages[slot] + static_cast<std::size_t>(offset) * kv_dim;
    }

    const float* get_v(int layer, int pos) const {
        if (q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (v_pages[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return v_pages[slot] + static_cast<std::size_t>(offset) * kv_dim;
    }

    const std::int8_t* get_k_q8(int layer, int pos) const {
        if (!q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (k_pages_q8[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return k_pages_q8[slot] + static_cast<std::size_t>(offset) * kv_dim;
    }

    const std::int8_t* get_v_q8(int layer, int pos) const {
        if (!q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (v_pages_q8[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return v_pages_q8[slot] + static_cast<std::size_t>(offset) * kv_dim;
    }

    const float* get_k_scales(int layer, int pos) const {
        if (!q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (k_scales_pages[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return k_scales_pages[slot] + static_cast<std::size_t>(offset) * n_kv_heads;
    }

    const float* get_v_scales(int layer, int pos) const {
        if (!q8_enabled) return nullptr;
        int page_idx = pos / page_size;
        std::size_t slot = static_cast<std::size_t>(layer) * total_pages + page_idx;
        if (v_scales_pages[slot] == nullptr) return nullptr;
        int offset = pos % page_size;
        return v_scales_pages[slot] + static_cast<std::size_t>(offset) * n_kv_heads;
    }

    void reset() {
        for (auto& p : k_pages) p = nullptr;
        for (auto& p : v_pages) p = nullptr;
        for (auto& p : k_pages_q8) p = nullptr;
        for (auto& p : v_pages_q8) p = nullptr;
        for (auto& p : k_scales_pages) p = nullptr;
        for (auto& p : v_scales_pages) p = nullptr;
        for (auto& s : k_storage) s.clear();
        for (auto& s : v_storage) s.clear();
        for (auto& s : k_storage_q8) s.clear();
        for (auto& s : v_storage_q8) s.clear();
        for (auto& s : k_scales_storage) s.clear();
        for (auto& s : v_scales_storage) s.clear();
    }

    void rebuild_views() {
        const std::size_t slots = k_pages.size();
        for (std::size_t slot = 0; slot < slots; ++slot) {
            k_pages[slot] = k_storage[slot].empty() ? nullptr : k_storage[slot].data();
            v_pages[slot] = v_storage[slot].empty() ? nullptr : v_storage[slot].data();
            k_pages_q8[slot] = k_storage_q8[slot].empty() ? nullptr : k_storage_q8[slot].data();
            v_pages_q8[slot] = v_storage_q8[slot].empty() ? nullptr : v_storage_q8[slot].data();
            k_scales_pages[slot] = k_scales_storage[slot].empty() ? nullptr : k_scales_storage[slot].data();
            v_scales_pages[slot] = v_scales_storage[slot].empty() ? nullptr : v_scales_storage[slot].data();
        }
    }

    Snapshot snapshot() const {
        Snapshot out{};
        out.k_storage = k_storage;
        out.v_storage = v_storage;
        out.k_storage_q8 = k_storage_q8;
        out.v_storage_q8 = v_storage_q8;
        out.k_scales_storage = k_scales_storage;
        out.v_scales_storage = v_scales_storage;
        out.owner_l3_domain = owner_l3_domain;
        out.owner_numa_node = owner_numa_node;
        return out;
    }

    void restore(const Snapshot& snapshot) {
        k_storage = snapshot.k_storage;
        v_storage = snapshot.v_storage;
        k_storage_q8 = snapshot.k_storage_q8;
        v_storage_q8 = snapshot.v_storage_q8;
        k_scales_storage = snapshot.k_scales_storage;
        v_scales_storage = snapshot.v_scales_storage;
        owner_l3_domain = snapshot.owner_l3_domain;
        owner_numa_node = snapshot.owner_numa_node;
        rebuild_views();
    }
};

void fused_gqa_attention_decode_paged(
    const float* q,
    const PagedKVStore& kv_store,
    int layer_idx,
    float* out,
    int n_heads,
    int n_kv_heads,
    int kv_len,
    int head_dim,
    float scale,
    int tile_size,
    const float* drift_block_decay_gain,
    const std::uint8_t* drift_block_pruned,
    int drift_block_count,
    int drift_block_size,
    int preserve_head_tokens,
    int preserve_recent_tokens
) {
    if (kv_len <= 0 || head_dim <= 0) return;
    int heads_per_kv = std::max(1, n_heads / std::max(1, n_kv_heads));

    // Tiled online softmax (Flash Attention algorithm):
    // Process K positions in tiles, maintaining running max M, sum S,
    // and accumulator acc. When max changes, rescale by exp(M_old - M_new).
    // This eliminates the separate score buffer allocation and reduces
    // from 4 passes (QK, max, exp, V·attn) to 1 pass.
    const int runtime_tile = std::max(16, std::min(256, tile_size > 0 ? tile_size : 64));
    const int safe_drift_block_size = std::max(1, drift_block_size);
    int drift_blocks_in_kv = 0;
    if (
        drift_block_decay_gain != nullptr
        && drift_block_count > 0
        && drift_block_size > 0
    ) {
        drift_blocks_in_kv = std::min(
            drift_block_count,
            (kv_len + safe_drift_block_size - 1) / safe_drift_block_size
        );
    }
    std::vector<float> drift_log_gain;
    std::vector<std::uint8_t> drift_skip;
    if (drift_blocks_in_kv > 0) {
        drift_log_gain.assign(static_cast<std::size_t>(drift_blocks_in_kv), 0.0f);
        drift_skip.assign(static_cast<std::size_t>(drift_blocks_in_kv), static_cast<std::uint8_t>(0));
        const int safe_preserve_head = std::max(0, preserve_head_tokens);
        const int safe_preserve_recent = std::max(0, preserve_recent_tokens);
        for (int b = 0; b < drift_blocks_in_kv; ++b) {
            const bool pruned = drift_block_pruned != nullptr && drift_block_pruned[b] != 0;
            bool protected_block = false;
            if (pruned) {
                const int block_start = b * safe_drift_block_size;
                const int block_end = std::min(kv_len, block_start + safe_drift_block_size);
                protected_block = block_end <= safe_preserve_head;
                if (!protected_block && safe_preserve_recent > 0) {
                    const int recent_start = std::max(0, kv_len - safe_preserve_recent);
                    protected_block = block_start >= recent_start;
                }
                if (!protected_block) {
                    drift_skip[static_cast<std::size_t>(b)] = static_cast<std::uint8_t>(1);
                    continue;
                }
            }
            float gain = protected_block ? 1.0f : drift_block_decay_gain[b];
            if (!std::isfinite(gain)) gain = 1.0f;
            gain = std::clamp(gain, 1.0e-6f, 1.0f);
            drift_log_gain[static_cast<std::size_t>(b)] = std::log(gain);
        }
    }

    int n_threads = get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int h = 0; h < n_heads; ++h) {
        maybe_bind_worker_thread(true);
        int kv_h = std::min(n_kv_heads - 1, h / heads_per_kv);
        const float* q_h = q + static_cast<std::size_t>(h) * head_dim;

        float* out_h = out + static_cast<std::size_t>(h) * head_dim;
        std::memset(out_h, 0, static_cast<std::size_t>(head_dim) * sizeof(float));

        float M = -INFINITY;  // running max
        float S = 0.0f;       // running sum of exp(score - M)

        for (int k0 = 0; k0 < kv_len; k0 += runtime_tile) {
            const int next_tile = k0 + runtime_tile;
            if (next_tile < kv_len) {
                prefetch_read(kv_store.q8_enabled
                    ? static_cast<const void*>(kv_store.get_k_q8(layer_idx, next_tile))
                    : static_cast<const void*>(kv_store.get_k(layer_idx, next_tile)));
                prefetch_read(kv_store.q8_enabled
                    ? static_cast<const void*>(kv_store.get_v_q8(layer_idx, next_tile))
                    : static_cast<const void*>(kv_store.get_v(layer_idx, next_tile)));
            }
            int tile_end = std::min(k0 + runtime_tile, kv_len);
            int tile_k = tile_end - k0;

            // 1. Compute QK scores for this tile and find tile max
            float tile_scores[256];
            float tile_max = -INFINITY;
            for (int t = 0; t < tile_k; ++t) {
                int j = k0 + t;
                float score;

                if (kv_store.q8_enabled) {
                    const std::int8_t* k_j_full = kv_store.get_k_q8(layer_idx, j);
                    const float* k_scales = kv_store.get_k_scales(layer_idx, j);
                    if (k_j_full == nullptr || k_scales == nullptr) {
                        tile_scores[t] = -1e9f;
                        continue;
                    }
                    const std::int8_t* k_j = k_j_full + static_cast<std::size_t>(kv_h) * head_dim;
                    float dot = 0.0f;
#ifdef __AVX2__
                    __m256 vacc = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m128i k8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(k_j + d));
                        __m256i k32 = _mm256_cvtepi8_epi32(k8);
                        __m256 kf = _mm256_cvtepi32_ps(k32);
                        __m256 qv = _mm256_loadu_ps(q_h + d);
                        vacc = _mm256_fmadd_ps(qv, kf, vacc);
                    }
                    dot = hsum256_ps(vacc);
                    for (; d < head_dim; ++d)
                        dot += q_h[d] * static_cast<float>(k_j[d]);
#else
                    for (int d = 0; d < head_dim; ++d)
                        dot += q_h[d] * static_cast<float>(k_j[d]);
#endif
                    score = (dot * k_scales[kv_h]) * scale;
                } else {
                    const float* k_j_full = kv_store.get_k(layer_idx, j);
                    if (k_j_full == nullptr) {
                        tile_scores[t] = -1e9f;
                        continue;
                    }
                    const float* k_j = k_j_full + static_cast<std::size_t>(kv_h) * head_dim;
                    float dot = 0.0f;
#ifdef __AVX2__
                    __m256 vacc = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 qv = _mm256_loadu_ps(q_h + d);
                        __m256 kv = _mm256_loadu_ps(k_j + d);
                        vacc = _mm256_fmadd_ps(qv, kv, vacc);
                    }
                    dot = hsum256_ps(vacc);
                    for (; d < head_dim; ++d)
                        dot += q_h[d] * k_j[d];
#else
                    for (int d = 0; d < head_dim; ++d)
                        dot += q_h[d] * k_j[d];
#endif
                    score = dot * scale;
                }
                if (!std::isfinite(score)) {
                    score = -1e9f;
                }
                if (drift_blocks_in_kv > 0) {
                    const int block_idx = j / safe_drift_block_size;
                    if (block_idx >= 0 && block_idx < drift_blocks_in_kv) {
                        if (drift_skip[static_cast<std::size_t>(block_idx)] != 0) {
                            tile_scores[t] = -INFINITY;
                            continue;
                        }
                        score += drift_log_gain[static_cast<std::size_t>(block_idx)];
                    }
                }
                tile_scores[t] = score;
                tile_max = std::max(tile_max, score);
            }

            // 2. Online softmax update: rescale accumulator if new max > old max
            if (tile_max > M) {
                float rescale = anvil_fast_math::fast_exp_scalar(M - tile_max);
                S *= rescale;
                // Rescale existing accumulator
#ifdef __AVX2__
                {
                    __m256 rv = _mm256_set1_ps(rescale);
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 ov = _mm256_loadu_ps(out_h + d);
                        _mm256_storeu_ps(out_h + d, _mm256_mul_ps(ov, rv));
                    }
                    for (; d < head_dim; ++d)
                        out_h[d] *= rescale;
                }
#else
                for (int d = 0; d < head_dim; ++d)
                    out_h[d] *= rescale;
#endif
                M = tile_max;
            }

            // 3. Compute exp(score - M) and accumulate V
            for (int t = 0; t < tile_k; ++t) {
                if (!std::isfinite(tile_scores[t])) {
                    continue;
                }
                float p = anvil_fast_math::fast_exp_scalar(tile_scores[t] - M);
                if (!std::isfinite(p) || p < 0.0f) {
                    p = 0.0f;
                }
                S += p;

                int j = k0 + t;
                if (kv_store.q8_enabled) {
                    const std::int8_t* v_j_full = kv_store.get_v_q8(layer_idx, j);
                    const float* v_scales = kv_store.get_v_scales(layer_idx, j);
                    if (v_j_full == nullptr || v_scales == nullptr) continue;
                    const std::int8_t* v_j = v_j_full + static_cast<std::size_t>(kv_h) * head_dim;
                    float w = p * v_scales[kv_h];
#ifdef __AVX2__
                    __m256 wv = _mm256_set1_ps(w);
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m128i v8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(v_j + d));
                        __m256i v32 = _mm256_cvtepi8_epi32(v8);
                        __m256 vf = _mm256_cvtepi32_ps(v32);
                        __m256 ov = _mm256_loadu_ps(out_h + d);
                        ov = _mm256_fmadd_ps(wv, vf, ov);
                        _mm256_storeu_ps(out_h + d, ov);
                    }
                    for (; d < head_dim; ++d)
                        out_h[d] += w * static_cast<float>(v_j[d]);
#else
                    for (int d = 0; d < head_dim; ++d)
                        out_h[d] += w * static_cast<float>(v_j[d]);
#endif
                } else {
                    const float* v_j_full = kv_store.get_v(layer_idx, j);
                    if (v_j_full == nullptr) continue;
                    const float* v_j = v_j_full + static_cast<std::size_t>(kv_h) * head_dim;
#ifdef __AVX2__
                    __m256 wv = _mm256_set1_ps(p);
                    int d = 0;
                    for (; d + 8 <= head_dim; d += 8) {
                        __m256 ov = _mm256_loadu_ps(out_h + d);
                        __m256 vv = _mm256_loadu_ps(v_j + d);
                        ov = _mm256_fmadd_ps(wv, vv, ov);
                        _mm256_storeu_ps(out_h + d, ov);
                    }
                    for (; d < head_dim; ++d)
                        out_h[d] += p * v_j[d];
#else
                    for (int d = 0; d < head_dim; ++d)
                        out_h[d] += p * v_j[d];
#endif
                }
            }
        }

        // 4. Final normalize: output = accumulator / S
        if (S > 0.0f && std::isfinite(S)) {
            float inv_S = 1.0f / S;
#ifdef __AVX2__
            {
                __m256 inv_vec = _mm256_set1_ps(inv_S);
                int d = 0;
                for (; d + 8 <= head_dim; d += 8) {
                    __m256 ov = _mm256_loadu_ps(out_h + d);
                    _mm256_storeu_ps(out_h + d, _mm256_mul_ps(ov, inv_vec));
                }
                for (; d < head_dim; ++d)
                    out_h[d] *= inv_S;
            }
#else
            for (int d = 0; d < head_dim; ++d)
                out_h[d] *= inv_S;
#endif
        } else {
            std::memset(out_h, 0, static_cast<std::size_t>(head_dim) * sizeof(float));
        }
        sanitize_tensor_inplace(out_h, head_dim);
    }
}

void causal_conv1d_step(
    std::vector<float>& buffer,
    int& head_row,
    const float* kernel,
    int kernel_rows,
    int channels,
    const float* input,
    const float* bias,
    float* output,
    bool apply_silu = false
) {
    if (kernel == nullptr || input == nullptr || output == nullptr ||
        kernel_rows <= 0 || channels <= 0) {
        return;
    }
    const int hist_rows = std::max(0, kernel_rows - 1);
    if (static_cast<int>(buffer.size()) != hist_rows * channels) {
        buffer.assign(static_cast<std::size_t>(hist_rows) * channels, 0.0f);
        head_row = 0;
    }
    if (hist_rows > 0 && (head_row < 0 || head_row >= hist_rows)) {
        head_row = 0;
    }
    const int read_head = hist_rows > 0 ? head_row : 0;
    const int conv_threads = get_num_threads(false);
    auto accumulate_row = [&](const float* kernel_row, const float* source_row) {
#ifdef _OPENMP
#pragma omp parallel num_threads(conv_threads) if(channels >= 1024)
#endif
        {
            int c_begin = 0;
            int c_end = channels;
#ifdef _OPENMP
            if (channels >= 1024) {
                const int tid = omp_get_thread_num();
                const int nth = std::max(1, omp_get_num_threads());
                c_begin = (channels * tid) / nth;
                c_end = (channels * (tid + 1)) / nth;
            }
#endif
#ifdef __AVX2__
            int c = c_begin;
            for (; c + 8 <= c_end; c += 8) {
                __m256 out_v = _mm256_loadu_ps(output + c);
                __m256 k_v = _mm256_loadu_ps(kernel_row + c);
                __m256 src_v = _mm256_loadu_ps(source_row + c);
                out_v = _mm256_fmadd_ps(k_v, src_v, out_v);
                _mm256_storeu_ps(output + c, out_v);
            }
            for (; c < c_end; ++c) {
                output[c] += kernel_row[c] * source_row[c];
            }
#else
            for (int c = c_begin; c < c_end; ++c) {
                output[c] += kernel_row[c] * source_row[c];
            }
#endif
        }
    };

    if (bias != nullptr) {
        std::memcpy(output, bias, static_cast<std::size_t>(channels) * sizeof(float));
    } else {
        std::memset(output, 0, static_cast<std::size_t>(channels) * sizeof(float));
    }
    for (int k = 0; k < hist_rows; ++k) {
        const int row = read_head + k < hist_rows ? (read_head + k) : (read_head + k - hist_rows);
        const float* kernel_row = kernel + static_cast<std::size_t>(k) * channels;
        const float* buffer_row = buffer.data() + static_cast<std::size_t>(row) * channels;
        accumulate_row(kernel_row, buffer_row);
    }
    accumulate_row(
        kernel + static_cast<std::size_t>(hist_rows) * channels,
        input
    );
    if (apply_silu) {
#ifdef __AVX2__
        int c = 0;
        for (; c + 8 <= channels; c += 8) {
            __m256 out_v = _mm256_loadu_ps(output + c);
            _mm256_storeu_ps(output + c, anvil_fast_math::v_silu(out_v));
        }
        for (; c < channels; ++c) {
            const float x = output[c];
            output[c] = x * fast_sigmoid_scalar(x);
        }
#else
        for (int c = 0; c < channels; ++c) {
            const float x = output[c];
            output[c] = x * fast_sigmoid_scalar(x);
        }
#endif
    }
    if (hist_rows <= 0) {
        return;
    }
    const int write_row = head_row;
    std::memcpy(
        buffer.data() + static_cast<std::size_t>(write_row) * channels,
        input,
        static_cast<std::size_t>(channels) * sizeof(float)
    );
    head_row += 1;
    if (head_row >= hist_rows) {
        head_row = 0;
    }
}

void apply_qwen_mrope(
    float* q,
    float* k,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int rope_dim,
    int pos,
    float theta,
    bool interleaved,
    int sec_t,
    int sec_h,
    int sec_w,
    int sec_e
) {
    const int half = rope_dim / 2;
    if (half <= 0) {
        return;
    }

    const int sect_dims = std::max(0, sec_t) + std::max(0, sec_h)
        + std::max(0, sec_w) + std::max(0, sec_e);
    const float safe_theta = (std::isfinite(theta) && theta > 0.0f) ? theta : 10000.0f;
    const int safe_rope_dim = std::max(1, rope_dim);

    struct QwenMropeCache {
        int half = -1;
        int rope_dim = -1;
        int pos = std::numeric_limits<int>::min();
        int sec_t = -1;
        int sec_h = -1;
        int sec_w = -1;
        int sec_e = -1;
        int sect_dims = -1;
        bool interleaved = false;
        float theta = 0.0f;
        std::vector<float> inv_freq;
        std::vector<float> cos;
        std::vector<float> sin;
    };
    static thread_local QwenMropeCache cache;

    const bool shape_changed = (
        cache.half != half
        || cache.rope_dim != safe_rope_dim
        || cache.theta != safe_theta
        || cache.sec_t != sec_t
        || cache.sec_h != sec_h
        || cache.sec_w != sec_w
        || cache.sec_e != sec_e
        || cache.sect_dims != sect_dims
        || cache.interleaved != interleaved
    );
    if (shape_changed) {
        cache.half = half;
        cache.rope_dim = safe_rope_dim;
        cache.theta = safe_theta;
        cache.sec_t = sec_t;
        cache.sec_h = sec_h;
        cache.sec_w = sec_w;
        cache.sec_e = sec_e;
        cache.sect_dims = sect_dims;
        cache.interleaved = interleaved;
        cache.pos = std::numeric_limits<int>::min();
        cache.inv_freq.assign(static_cast<std::size_t>(half), 1.0f);
        cache.cos.assign(static_cast<std::size_t>(half), 1.0f);
        cache.sin.assign(static_cast<std::size_t>(half), 0.0f);
        if (half > 1) {
            const float freq_step = std::pow(
                safe_theta,
                -2.0f / static_cast<float>(safe_rope_dim)
            );
            for (int i = 1; i < half; ++i) {
                cache.inv_freq[static_cast<std::size_t>(i)] =
                    cache.inv_freq[static_cast<std::size_t>(i - 1)] * freq_step;
            }
        }
    }

    if (shape_changed || cache.pos != pos) {
        if (sect_dims <= 0) {
            for (int i = 0; i < half; ++i) {
                const float angle = static_cast<float>(pos) * cache.inv_freq[static_cast<std::size_t>(i)];
                cache.cos[static_cast<std::size_t>(i)] = std::cos(angle);
                cache.sin[static_cast<std::size_t>(i)] = std::sin(angle);
            }
        } else {
            const float p_t = static_cast<float>(pos);
            const float p_h = static_cast<float>(pos);
            const float p_w = static_cast<float>(pos);
            const float p_e = 0.0f;
            const int sec_w_off = sec_t + sec_h;
            const int sec_e_off = sec_w_off + sec_w;

            for (int i = 0; i < half; ++i) {
                const int sector = i % sect_dims;
                float position = p_t;
                if (interleaved) {
                    if ((sector % 3 == 1) && (sector < 3 * sec_h)) {
                        position = p_h;
                    } else if ((sector % 3 == 2) && (sector < 3 * sec_w)) {
                        position = p_w;
                    } else if ((sector % 3 == 0) && (sector < 3 * sec_t)) {
                        position = p_t;
                    } else {
                        position = p_e;
                    }
                } else {
                    if (sec_t <= sector && sector < sec_w_off) {
                        position = p_h;
                    } else if (sec_w_off <= sector && sector < sec_e_off) {
                        position = p_w;
                    } else if (sector >= sec_e_off) {
                        position = p_e;
                    }
                }
                const float angle = position * cache.inv_freq[static_cast<std::size_t>(i)];
                cache.cos[static_cast<std::size_t>(i)] = std::cos(angle);
                cache.sin[static_cast<std::size_t>(i)] = std::sin(angle);
            }
        }
        cache.pos = pos;
    }

    auto rotate = [&](float* vec, int heads) {
        for (int h = 0; h < heads; ++h) {
            float* ptr = vec + static_cast<std::size_t>(h) * head_dim;
            int i = 0;
#ifdef __AVX2__
            for (; i + 8 <= half; i += 8) {
                const __m256 cv = _mm256_loadu_ps(cache.cos.data() + i);
                const __m256 sv = _mm256_loadu_ps(cache.sin.data() + i);
                const __m256 x0v = _mm256_loadu_ps(ptr + i);
                const __m256 x1v = _mm256_loadu_ps(ptr + half + i);
                const __m256 out0 = _mm256_sub_ps(_mm256_mul_ps(x0v, cv), _mm256_mul_ps(x1v, sv));
                const __m256 out1 = _mm256_add_ps(_mm256_mul_ps(x0v, sv), _mm256_mul_ps(x1v, cv));
                _mm256_storeu_ps(ptr + i, out0);
                _mm256_storeu_ps(ptr + half + i, out1);
            }
#endif
            for (; i < half; ++i) {
                const float c = cache.cos[static_cast<std::size_t>(i)];
                const float s = cache.sin[static_cast<std::size_t>(i)];
                const float x0 = ptr[i];
                const float x1 = ptr[half + i];
                ptr[i] = x0 * c - x1 * s;
                ptr[half + i] = x0 * s + x1 * c;
            }
        }
    };

    rotate(q, n_heads);
    rotate(k, n_kv_heads);
}

// ===== Layer configuration =====

struct LayerConfig {
    // Float32 weight pointers (nullptr if quantized)
    const float* attn_norm;
    const float* ffn_norm;
    const float* wq;
    const float* wk;
    const float* wv;
    const float* wo;
    const float* w_gate;
    const float* w_up;
    const float* w_down;
    const float* attn_q_norm;
    const float* attn_k_norm;
    const float* ssm_a;
    const float* ssm_d;
    const float* ssm_dt;
    const float* ssm_conv;
    const float* ssm_conv_bias;
    const float* ssm_norm;
    const float* router;

    // Quantized weight pointers (nullptr if float32)
    const void* wq_quant;    int wq_qtype;
    const void* wk_quant;    int wk_qtype;
    const void* wv_quant;    int wv_qtype;
    const void* wo_quant;    int wo_qtype;
    const void* wgate_quant; int wgate_qtype;
    const void* wup_quant;   int wup_qtype;
    const void* wdown_quant; int wdown_qtype;
    const void* attn_gate_quant; int attn_gate_qtype;
    const void* ssm_in_quant; int ssm_in_qtype;
    const void* ssm_out_quant; int ssm_out_qtype;
    const void* ssm_alpha_quant; int ssm_alpha_qtype;
    const void* ssm_beta_quant; int ssm_beta_qtype;
    const void* shared_gate_quant; int shared_gate_qtype;
    const void* shared_up_quant; int shared_up_qtype;
    const void* shared_down_quant; int shared_down_qtype;
    const void* const* expert_gate_ptrs; int expert_gate_qtype;
    const void* const* expert_up_ptrs; int expert_up_qtype;
    const void* const* expert_down_ptrs; int expert_down_qtype;

    int q_out_dim;
    int kv_out_dim;
    int ffn_dim;
    int attn_gate_dim;
    int ssm_in_dim;
    int ssm_out_dim;
    int ssm_alpha_dim;
    int ssm_beta_dim;
    int shared_hidden_dim;
    int expert_hidden_dim;
    int expert_count;
    int moe_top_k;
    int ssm_conv_rows;
    int ssm_conv_cols;
    int layer_kind;
    bool is_attention;
    int layer_idx;

    LayerConfig()
        : attn_norm(nullptr), ffn_norm(nullptr),
          wq(nullptr), wk(nullptr), wv(nullptr), wo(nullptr),
          w_gate(nullptr), w_up(nullptr), w_down(nullptr),
          attn_q_norm(nullptr), attn_k_norm(nullptr),
          ssm_a(nullptr), ssm_d(nullptr), ssm_dt(nullptr),
          ssm_conv(nullptr), ssm_conv_bias(nullptr),
          ssm_norm(nullptr), router(nullptr),
          wq_quant(nullptr), wq_qtype(0),
          wk_quant(nullptr), wk_qtype(0),
          wv_quant(nullptr), wv_qtype(0),
          wo_quant(nullptr), wo_qtype(0),
          wgate_quant(nullptr), wgate_qtype(0),
          wup_quant(nullptr), wup_qtype(0),
          wdown_quant(nullptr), wdown_qtype(0),
          attn_gate_quant(nullptr), attn_gate_qtype(0),
          ssm_in_quant(nullptr), ssm_in_qtype(0),
          ssm_out_quant(nullptr), ssm_out_qtype(0),
          ssm_alpha_quant(nullptr), ssm_alpha_qtype(0),
          ssm_beta_quant(nullptr), ssm_beta_qtype(0),
          shared_gate_quant(nullptr), shared_gate_qtype(0),
          shared_up_quant(nullptr), shared_up_qtype(0),
          shared_down_quant(nullptr), shared_down_qtype(0),
          expert_gate_ptrs(nullptr), expert_gate_qtype(0),
          expert_up_ptrs(nullptr), expert_up_qtype(0),
          expert_down_ptrs(nullptr), expert_down_qtype(0),
          q_out_dim(0), kv_out_dim(0), ffn_dim(0),
          attn_gate_dim(0), ssm_in_dim(0), ssm_out_dim(0),
          ssm_alpha_dim(0), ssm_beta_dim(0),
          shared_hidden_dim(0), expert_hidden_dim(0),
          expert_count(0), moe_top_k(0),
          ssm_conv_rows(0), ssm_conv_cols(0), layer_kind(LAYER_KIND_STANDARD),
          is_attention(true), layer_idx(0) {}

    // Convenience: check if a projection has any weight (float32 or quantized)
    bool has_wq() const { return wq != nullptr || wq_quant != nullptr; }
    bool has_wk() const { return wk != nullptr || wk_quant != nullptr; }
    bool has_wv() const { return wv != nullptr || wv_quant != nullptr; }
    bool has_wo() const { return wo != nullptr || wo_quant != nullptr; }
    bool has_gate() const { return w_gate != nullptr || wgate_quant != nullptr; }
    bool has_up() const { return w_up != nullptr || wup_quant != nullptr; }
    bool has_down() const { return w_down != nullptr || wdown_quant != nullptr; }
    bool has_attn_gate() const { return attn_gate_quant != nullptr; }
    bool has_ssm_in() const { return ssm_in_quant != nullptr; }
    bool has_ssm_out() const { return ssm_out_quant != nullptr; }
    bool has_ssm_alpha() const { return ssm_alpha_quant != nullptr; }
    bool has_ssm_beta() const { return ssm_beta_quant != nullptr; }
    bool has_shared_experts() const {
        return shared_gate_quant != nullptr
            && is_supported_quant_qtype(shared_gate_qtype)
            && shared_up_quant != nullptr
            && is_supported_quant_qtype(shared_up_qtype)
            && shared_down_quant != nullptr
            && is_supported_quant_qtype(shared_down_qtype)
            && shared_hidden_dim > 0;
    }
    bool has_routed_experts() const {
        return router != nullptr
            && expert_gate_ptrs != nullptr
            && is_supported_quant_qtype(expert_gate_qtype)
            && expert_up_ptrs != nullptr
            && is_supported_quant_qtype(expert_up_qtype)
            && expert_down_ptrs != nullptr
            && is_supported_quant_qtype(expert_down_qtype)
            && expert_hidden_dim > 0
            && expert_count > 0
            && moe_top_k > 0;
    }
};

// ===== Model graph handle =====

struct ModelGraphHandle {
    int n_layers;
    int dim;
    int vocab_size;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int max_seq;
    float rms_eps;
    float rope_theta;
    float embedding_scale;
    float residual_scale;
    float logit_scale;
    float attention_scale;

    // Per-layer configurations
    std::vector<LayerConfig> layers;

    // Paged KV cache (lazy allocation)
    PagedKVStore kv_store;

    // Final norm weights
    const float* final_norm;
    const float* token_embeddings;
    const void* token_embeddings_quant;
    int token_embeddings_qtype;
    int token_embeddings_input_dim;
    int token_embeddings_output_dim;
    bool token_embeddings_transposed;

    // LM head: [vocab_size, dim]
    const float* lm_head;
    const void* lm_head_quant;
    int lm_head_qtype;
    bool lm_head_transposed;

    // Scratch buffers (pre-allocated, reused per layer)
    AlignedBuffer<float> scratch_normed;
    AlignedBuffer<float> scratch_q;
    AlignedBuffer<float> scratch_k;
    AlignedBuffer<float> scratch_v;
    AlignedBuffer<float> scratch_attn_out;
    AlignedBuffer<float> scratch_proj;
    AlignedBuffer<float> scratch_gate;
    AlignedBuffer<float> scratch_up;
    AlignedBuffer<float> scratch_ffn;
    AlignedBuffer<float> scratch_state;
    AlignedBuffer<float> scratch_last_hidden;
    AlignedBuffer<float> scratch_aux0;
    AlignedBuffer<float> scratch_aux1;
    AlignedBuffer<float> scratch_aux2;
    AlignedBuffer<float> scratch_token_embeddings;
    std::vector<int> scratch_expert_indices;
    std::vector<float> scratch_expert_scores;
    std::vector<float> scratch_expert_weights;
    std::vector<std::vector<float>> ssm_state_storage;
    std::vector<std::vector<float>> conv_state_storage;
    std::vector<int> conv_state_head_storage;
    GraphPerfStats perf_stats;
    GraphDriftConfig drift_config;
    GraphDriftSnapshot last_drift_snapshot;
    std::vector<float> drift_block_energy_ref;
    std::vector<float> drift_block_phase_ref;
    std::vector<float> drift_block_latest;
    std::vector<float> drift_block_decay_gain;
    std::vector<std::uint8_t> drift_block_pruned;
    std::vector<std::uint8_t> drift_block_prune_streak;
    int drift_state_block_size;
    std::vector<float> drift_overhead_window;
    double drift_overhead_sum;
    int drift_overhead_index;
    int drift_overhead_count;
    int drift_overhead_below_target_streak;
    int drift_auto_downgrade_events;
    int drift_auto_upgrade_events;
    bool drift_force_conservative;
    int drift_tokens_since_update;
    int drift_tokens_since_prune;
    double drift_stabilizer_seconds_total;
    int drift_stabilizer_calls_total;
    // Current sequence position
    int current_pos;
    bool last_hidden_valid;
    bool qwen_rope_finetuned;
    bool qwen_mrope_interleaved;
    int qwen_rope_dim;
    int qwen_sec_t;
    int qwen_sec_h;
    int qwen_sec_w;
    int qwen_sec_e;
    int qwen_ssm_state_size;
    int qwen_ssm_group_count;
    int qwen_ssm_n_v_heads;
    int attention_tile_size;

    ModelGraphHandle(int n_lay, int d, int vocab, int heads, int kv_heads,
                     int h_dim, int max_s, float eps, float theta)
        : n_layers(n_lay), dim(d), vocab_size(vocab),
          n_heads(heads), n_kv_heads(kv_heads), head_dim(h_dim),
          max_seq(max_s), rms_eps(eps), rope_theta(theta),
          embedding_scale(0.0f), residual_scale(0.0f),
          logit_scale(0.0f), attention_scale(0.0f),
          kv_store(max_s, n_lay, kv_heads, h_dim, use_q8_kv_cache()),
          final_norm(nullptr), token_embeddings(nullptr),
          token_embeddings_quant(nullptr), token_embeddings_qtype(0),
          token_embeddings_input_dim(0), token_embeddings_output_dim(0),
          token_embeddings_transposed(false), lm_head(nullptr),
          lm_head_quant(nullptr), lm_head_qtype(0),
          lm_head_transposed(false), current_pos(0),
          qwen_rope_finetuned(true),
          qwen_mrope_interleaved(false),
          qwen_rope_dim(0),
          qwen_sec_t(0), qwen_sec_h(0), qwen_sec_w(0), qwen_sec_e(0),
          qwen_ssm_state_size(0), qwen_ssm_group_count(0),
          qwen_ssm_n_v_heads(0),
          attention_tile_size(compute_attention_tile_size(h_dim)) {

        layers.resize(static_cast<std::size_t>(n_layers));

        int max_ffn = 4 * dim;
        int kv_dim = n_kv_heads * head_dim;
        scratch_normed.resize(static_cast<std::size_t>(dim));
        scratch_q.resize(static_cast<std::size_t>(n_heads * head_dim));
        scratch_k.resize(static_cast<std::size_t>(kv_dim));
        scratch_v.resize(static_cast<std::size_t>(kv_dim));
        scratch_attn_out.resize(static_cast<std::size_t>(n_heads * head_dim));
        scratch_proj.resize(static_cast<std::size_t>(dim));
        scratch_gate.resize(static_cast<std::size_t>(max_ffn));
        scratch_up.resize(static_cast<std::size_t>(max_ffn));
        scratch_ffn.resize(static_cast<std::size_t>(max_ffn));
        scratch_state.resize(static_cast<std::size_t>(dim));
        scratch_last_hidden.resize(static_cast<std::size_t>(dim));
        scratch_aux0.resize(static_cast<std::size_t>(std::max(max_ffn, dim)));
        scratch_aux1.resize(static_cast<std::size_t>(std::max(max_ffn, dim)));
        scratch_aux2.resize(static_cast<std::size_t>(std::max(max_ffn, dim)));
        scratch_expert_indices.resize(8);
        scratch_expert_scores.resize(8);
        scratch_expert_weights.resize(8);
        ssm_state_storage.resize(static_cast<std::size_t>(n_layers));
        conv_state_storage.resize(static_cast<std::size_t>(n_layers));
        conv_state_head_storage.assign(static_cast<std::size_t>(n_layers), 0);
        drift_config = default_graph_drift_config();
        last_drift_snapshot = default_graph_drift_snapshot(drift_config.mode);
        drift_state_block_size = 0;
        drift_overhead_window.assign(static_cast<std::size_t>(GRAPH_DRIFT_OVERHEAD_WINDOW), 0.0f);
        drift_overhead_sum = 0.0;
        drift_overhead_index = 0;
        drift_overhead_count = 0;
        drift_overhead_below_target_streak = 0;
        drift_auto_downgrade_events = 0;
        drift_auto_upgrade_events = 0;
        drift_force_conservative = false;
        drift_tokens_since_update = 0;
        drift_tokens_since_prune = 0;
        drift_stabilizer_seconds_total = 0.0;
        drift_stabilizer_calls_total = 0;
        last_hidden_valid = false;
    }

    void ensure_scratch(int ffn_dim) {
        std::size_t needed = static_cast<std::size_t>(std::max(ffn_dim, dim));
        if (scratch_gate.size() < needed) {
            scratch_gate.resize(needed);
            scratch_up.resize(needed);
            scratch_ffn.resize(needed);
            scratch_aux0.resize(needed);
            scratch_aux1.resize(needed);
            scratch_aux2.resize(needed);
        }
    }

    void ensure_attention_scratch(int q_dim, int kv_dim) {
        const std::size_t q_needed = static_cast<std::size_t>(std::max(1, q_dim));
        const std::size_t kv_needed = static_cast<std::size_t>(std::max(1, kv_dim));
        if (scratch_q.size() < q_needed) {
            scratch_q.resize(q_needed);
        }
        if (scratch_attn_out.size() < q_needed) {
            scratch_attn_out.resize(q_needed);
        }
        if (scratch_k.size() < kv_needed) {
            scratch_k.resize(kv_needed);
        }
        if (scratch_v.size() < kv_needed) {
            scratch_v.resize(kv_needed);
        }
    }

    void ensure_expert_scratch(int top_k) {
        const std::size_t needed = static_cast<std::size_t>(std::max(top_k, 0));
        if (scratch_expert_indices.size() < needed) {
            scratch_expert_indices.resize(needed);
            scratch_expert_scores.resize(needed);
            scratch_expert_weights.resize(needed);
        }
    }
};

struct GraphExecutionCheckpoint {
    PagedKVStore::Snapshot kv_snapshot;
    std::vector<std::vector<float>> ssm_state_storage;
    std::vector<std::vector<float>> conv_state_storage;
    std::vector<int> conv_state_head_storage;
    GraphDriftSnapshot last_drift_snapshot;
    std::vector<float> drift_block_energy_ref;
    std::vector<float> drift_block_phase_ref;
    std::vector<float> drift_block_latest;
    std::vector<float> drift_block_decay_gain;
    std::vector<std::uint8_t> drift_block_pruned;
    std::vector<std::uint8_t> drift_block_prune_streak;
    std::vector<float> drift_overhead_window;
    std::vector<float> last_hidden;
    int drift_state_block_size = 0;
    double drift_overhead_sum = 0.0;
    int drift_overhead_index = 0;
    int drift_overhead_count = 0;
    int drift_overhead_below_target_streak = 0;
    int drift_auto_downgrade_events = 0;
    int drift_auto_upgrade_events = 0;
    bool drift_force_conservative = false;
    int drift_tokens_since_update = 0;
    int drift_tokens_since_prune = 0;
    double drift_stabilizer_seconds_total = 0.0;
    int drift_stabilizer_calls_total = 0;
    int current_pos = 0;
    bool last_hidden_valid = false;
};

GraphDriftConfig sanitize_graph_drift_config(const GraphDriftConfig& requested, int max_seq) {
    const GraphDriftConfig defaults = default_graph_drift_config();
    GraphDriftConfig cfg = requested;
    cfg.enabled = cfg.enabled != 0 ? 1 : 0;
    cfg.mode = clamp_graph_drift_mode(cfg.mode);
    cfg.block_size_tokens = std::max(1, cfg.block_size_tokens);
    cfg.block_size_tokens = std::min(cfg.block_size_tokens, std::max(1, max_seq));
    cfg.update_interval_tokens = std::max(1, cfg.update_interval_tokens);
    cfg.prune_interval_tokens = std::max(1, cfg.prune_interval_tokens);
    cfg.preserve_head_tokens = std::max(0, cfg.preserve_head_tokens);
    cfg.preserve_recent_tokens = std::max(0, cfg.preserve_recent_tokens);
    cfg.min_active_tokens = std::max(0, cfg.min_active_tokens);
    if (!std::isfinite(cfg.damp_threshold)) cfg.damp_threshold = defaults.damp_threshold;
    if (!std::isfinite(cfg.prune_threshold)) cfg.prune_threshold = defaults.prune_threshold;
    if (!std::isfinite(cfg.damping_strength) || cfg.damping_strength <= 0.0f) {
        cfg.damping_strength = defaults.damping_strength;
    }
    if (!std::isfinite(cfg.hysteresis) || cfg.hysteresis < 0.0f) cfg.hysteresis = defaults.hysteresis;
    cfg.damp_threshold = std::clamp(cfg.damp_threshold, 0.0f, 2.0f);
    cfg.prune_threshold = std::clamp(cfg.prune_threshold, cfg.damp_threshold, 2.0f);
    return cfg;
}

inline int graph_drift_effective_mode(const ModelGraphHandle* g) {
    if (g == nullptr) {
        return GRAPH_DRIFT_MODE_TELEMETRY;
    }
    if (g->drift_force_conservative && g->drift_config.mode == GRAPH_DRIFT_MODE_AGGRESSIVE) {
        return GRAPH_DRIFT_MODE_CONSERVATIVE;
    }
    return clamp_graph_drift_mode(g->drift_config.mode);
}

struct DriftDecodeContext {
    const float* block_decay_gain = nullptr;
    const std::uint8_t* block_pruned = nullptr;
    int block_count = 0;
    int block_size = 0;
    int preserve_head_tokens = 0;
    int preserve_recent_tokens = 0;
};

inline DriftDecodeContext make_drift_decode_context(const ModelGraphHandle* g) {
    DriftDecodeContext ctx{};
    if (g == nullptr || g->drift_config.enabled == 0) {
        return ctx;
    }
    if (g->drift_state_block_size <= 0 || g->drift_block_decay_gain.empty()) {
        return ctx;
    }
    std::size_t count = g->drift_block_decay_gain.size();
    if (!g->drift_block_pruned.empty()) {
        count = std::min(count, g->drift_block_pruned.size());
        ctx.block_pruned = g->drift_block_pruned.data();
    }
    if (count == 0) {
        return ctx;
    }
    ctx.block_decay_gain = g->drift_block_decay_gain.data();
    ctx.block_count = static_cast<int>(
        std::min<std::size_t>(count, static_cast<std::size_t>(std::numeric_limits<int>::max()))
    );
    ctx.block_size = g->drift_state_block_size;
    ctx.preserve_head_tokens = g->drift_config.preserve_head_tokens;
    ctx.preserve_recent_tokens = g->drift_config.preserve_recent_tokens;
    return ctx;
}

void graph_drift_ensure_state(ModelGraphHandle* g) {
    if (g == nullptr) return;
    const int block_size = std::max(1, g->drift_config.block_size_tokens);
    const std::size_t block_count =
        static_cast<std::size_t>((g->max_seq + block_size - 1) / block_size);
    if (
        g->drift_state_block_size == block_size
        && g->drift_block_latest.size() == block_count
    ) {
        return;
    }
    g->drift_state_block_size = block_size;
    g->drift_block_energy_ref.assign(
        block_count, std::numeric_limits<float>::quiet_NaN()
    );
    g->drift_block_phase_ref.assign(block_count, 0.0f);
    g->drift_block_latest.assign(block_count, 0.0f);
    g->drift_block_decay_gain.assign(block_count, 1.0f);
    g->drift_block_pruned.assign(block_count, static_cast<std::uint8_t>(0));
    g->drift_block_prune_streak.assign(block_count, static_cast<std::uint8_t>(0));
}

void graph_drift_reset_runtime(ModelGraphHandle* g) {
    if (g == nullptr) return;
    graph_drift_ensure_state(g);
    std::fill(
        g->drift_block_energy_ref.begin(),
        g->drift_block_energy_ref.end(),
        std::numeric_limits<float>::quiet_NaN()
    );
    std::fill(g->drift_block_phase_ref.begin(), g->drift_block_phase_ref.end(), 0.0f);
    std::fill(g->drift_block_latest.begin(), g->drift_block_latest.end(), 0.0f);
    std::fill(g->drift_block_decay_gain.begin(), g->drift_block_decay_gain.end(), 1.0f);
    std::fill(g->drift_block_pruned.begin(), g->drift_block_pruned.end(), static_cast<std::uint8_t>(0));
    std::fill(g->drift_block_prune_streak.begin(), g->drift_block_prune_streak.end(), static_cast<std::uint8_t>(0));
    std::fill(g->drift_overhead_window.begin(), g->drift_overhead_window.end(), 0.0f);
    g->drift_overhead_sum = 0.0;
    g->drift_overhead_index = 0;
    g->drift_overhead_count = 0;
    g->drift_overhead_below_target_streak = 0;
    g->drift_auto_downgrade_events = 0;
    g->drift_auto_upgrade_events = 0;
    g->drift_force_conservative = false;
    g->drift_tokens_since_update = 0;
    g->drift_tokens_since_prune = 0;
    g->drift_stabilizer_seconds_total = 0.0;
    g->drift_stabilizer_calls_total = 0;
    g->last_drift_snapshot = default_graph_drift_snapshot(g->drift_config.mode);
}

inline int graph_drift_block_span(int block_index, int block_size, int token_count) {
    const int block_start = block_index * block_size;
    const int block_end = std::min(token_count, block_start + block_size);
    return std::max(0, block_end - block_start);
}

bool graph_drift_block_is_protected(
    const GraphDriftConfig& cfg,
    int block_start,
    int block_end,
    int token_count
) {
    if (block_end <= cfg.preserve_head_tokens) {
        return true;
    }
    if (cfg.preserve_recent_tokens > 0) {
        const int recent_start = std::max(0, token_count - cfg.preserve_recent_tokens);
        if (block_start >= recent_start) {
            return true;
        }
    }
    return false;
}

int graph_drift_count_active_tokens(const ModelGraphHandle* g, int token_count) {
    if (g == nullptr || token_count <= 0) return 0;
    const int block_size = std::max(1, g->drift_config.block_size_tokens);
    const int block_count = (token_count + block_size - 1) / block_size;
    int active = token_count;
    for (int b = 0; b < block_count; ++b) {
        if (static_cast<std::size_t>(b) >= g->drift_block_pruned.size()) break;
        if (g->drift_block_pruned[static_cast<std::size_t>(b)] == 0) continue;
        active -= graph_drift_block_span(b, block_size, token_count);
    }
    return std::max(0, active);
}

bool graph_drift_compute_block_summary(
    const ModelGraphHandle* g,
    int block_start,
    int block_end,
    float* energy_out,
    float* phase_out
) {
    if (g == nullptr || energy_out == nullptr || phase_out == nullptr) return false;
    if (g->n_layers <= 0 || block_end <= block_start) return false;
    const int layer_idx = 0;
    double energy_acc = 0.0;
    double phase_real = 0.0;
    double phase_imag = 0.0;
    int samples = 0;
    const int head_dim = std::max(1, g->head_dim);

    for (int pos = block_start; pos < block_end; ++pos) {
        if (g->kv_store.q8_enabled) {
            const std::int8_t* kq = g->kv_store.get_k_q8(layer_idx, pos);
            const std::int8_t* vq = g->kv_store.get_v_q8(layer_idx, pos);
            const float* k_scales = g->kv_store.get_k_scales(layer_idx, pos);
            const float* v_scales = g->kv_store.get_v_scales(layer_idx, pos);
            if (kq == nullptr || vq == nullptr || k_scales == nullptr || v_scales == nullptr) {
                continue;
            }
            const float k_scale = k_scales[0];
            const float v_scale = v_scales[0];
            const float k0 = static_cast<float>(kq[0]) * k_scale;
            const float v0 = static_cast<float>(vq[0]) * v_scale;
            const float k1 = head_dim > 1 ? static_cast<float>(kq[1]) * k_scale : 0.0f;
            const float v1 = head_dim > 1 ? static_cast<float>(vq[1]) * v_scale : 0.0f;
            energy_acc += std::fabs(k0) + std::fabs(v0) + std::fabs(k1) + std::fabs(v1);
            phase_real += static_cast<double>(k0 + v0);
            phase_imag += static_cast<double>(k1 + v1);
            samples += 1;
            continue;
        }

        const float* k = g->kv_store.get_k(layer_idx, pos);
        const float* v = g->kv_store.get_v(layer_idx, pos);
        if (k == nullptr || v == nullptr) {
            continue;
        }
        const float k0 = k[0];
        const float v0 = v[0];
        const float k1 = head_dim > 1 ? k[1] : 0.0f;
        const float v1 = head_dim > 1 ? v[1] : 0.0f;
        energy_acc += std::fabs(k0) + std::fabs(v0) + std::fabs(k1) + std::fabs(v1);
        phase_real += static_cast<double>(k0 + v0);
        phase_imag += static_cast<double>(k1 + v1);
        samples += 1;
    }

    if (samples <= 0) {
        return false;
    }
    *energy_out = static_cast<float>(energy_acc / static_cast<double>(samples));
    *phase_out = static_cast<float>(std::atan2(phase_imag, phase_real + 1e-6));
    return true;
}

void graph_drift_evaluate_policy(
    ModelGraphHandle* g,
    int token_count,
    bool allow_prune_step
) {
    if (g == nullptr) return;
    token_count = std::clamp(token_count, 0, g->max_seq);
    const int effective_mode = graph_drift_effective_mode(g);
    if (token_count <= 0) {
        g->last_drift_snapshot = default_graph_drift_snapshot(effective_mode);
        return;
    }
    graph_drift_ensure_state(g);
    const int block_size = std::max(1, g->drift_config.block_size_tokens);
    const int block_count = (token_count + block_size - 1) / block_size;
    const int min_active_floor = std::min(std::max(0, g->drift_config.min_active_tokens), token_count);

    int active_token_count = graph_drift_count_active_tokens(g, token_count);
    int damped_block_count = 0;
    int pruned_block_count = 0;
    float latest_drift = 0.0f;
    float max_drift = 0.0f;
    double drift_sum = 0.0;
    int drift_count = 0;
    double decay_sum = 0.0;
    int decay_count = 0;

    for (int b = 0; b < block_count; ++b) {
        const std::size_t idx = static_cast<std::size_t>(b);
        const int block_start = b * block_size;
        const int block_end = std::min(token_count, block_start + block_size);
        const int span_tokens = std::max(0, block_end - block_start);
        bool protected_block = graph_drift_block_is_protected(
            g->drift_config, block_start, block_end, token_count
        );
        bool is_pruned = g->drift_block_pruned[idx] != 0;
        if (is_pruned && protected_block) {
            g->drift_block_pruned[idx] = 0;
            is_pruned = false;
            active_token_count = std::min(token_count, active_token_count + span_tokens);
        }

        float energy = 0.0f;
        float phase = 0.0f;
        if (!graph_drift_compute_block_summary(g, block_start, block_end, &energy, &phase)) {
            const float center = 0.5f * static_cast<float>(block_start + block_end);
            energy = 1.0f + 1e-4f * center;
            phase = 0.01f * static_cast<float>((b * 17) % 314);
        }

        float drift = 0.0f;
        const float ref_energy = g->drift_block_energy_ref[idx];
        if (!std::isfinite(ref_energy) || ref_energy <= 0.0f) {
            g->drift_block_energy_ref[idx] = std::max(1e-6f, energy);
            g->drift_block_phase_ref[idx] = phase;
        } else {
            const float ref_phase = g->drift_block_phase_ref[idx];
            const float energy_delta = std::fabs(energy - ref_energy) / (std::fabs(ref_energy) + 1e-6f);
            const float phase_error = 1.0f - std::cos(phase - ref_phase);
            drift = std::clamp(0.7f * energy_delta + 0.3f * phase_error, 0.0f, 2.0f);
            const float blend = 0.99f;
            g->drift_block_energy_ref[idx] = blend * ref_energy + (1.0f - blend) * energy;
            g->drift_block_phase_ref[idx] = blend * ref_phase + (1.0f - blend) * phase;
        }

        g->drift_block_latest[idx] = drift;
        latest_drift = drift;
        max_drift = std::max(max_drift, drift);
        drift_sum += drift;
        drift_count += 1;

        const bool can_apply_damp = effective_mode != GRAPH_DRIFT_MODE_TELEMETRY;
        float gain = is_pruned ? 0.0f : 1.0f;
        if (can_apply_damp && !is_pruned && drift >= g->drift_config.damp_threshold) {
            gain = std::exp(
                -g->drift_config.damping_strength
                * std::max(0.0f, drift - g->drift_config.damp_threshold)
            );
            gain = std::clamp(gain, 0.25f, 1.0f);
        }
        g->drift_block_decay_gain[idx] = gain;

        if (!is_pruned && gain < 0.999f) {
            damped_block_count += 1;
        }

        if (drift >= g->drift_config.prune_threshold) {
            const int next_streak = static_cast<int>(g->drift_block_prune_streak[idx]) + 1;
            g->drift_block_prune_streak[idx] = static_cast<std::uint8_t>(std::min(255, next_streak));
        } else if (drift < (g->drift_config.prune_threshold - g->drift_config.hysteresis)) {
            g->drift_block_prune_streak[idx] = 0;
        }

        if (
            effective_mode == GRAPH_DRIFT_MODE_AGGRESSIVE
            && allow_prune_step
            && !is_pruned
            && !protected_block
            && g->drift_block_prune_streak[idx] >= 2
            && (active_token_count - span_tokens) >= min_active_floor
        ) {
            g->drift_block_pruned[idx] = 1;
            g->drift_block_decay_gain[idx] = 0.0f;
            active_token_count -= span_tokens;
            is_pruned = true;
        }

        if (is_pruned) {
            pruned_block_count += 1;
        } else {
            decay_sum += static_cast<double>(g->drift_block_decay_gain[idx]);
            decay_count += 1;
        }
    }

    GraphDriftSnapshot snapshot = g->last_drift_snapshot;
    snapshot.latest_drift = latest_drift;
    snapshot.mean_drift = drift_count > 0 ? static_cast<float>(drift_sum / drift_count) : 0.0f;
    snapshot.max_drift = max_drift;
    snapshot.decay_ratio = decay_count > 0 ? static_cast<float>(decay_sum / decay_count) : 1.0f;
    snapshot.active_token_count = std::max(0, std::min(token_count, active_token_count));
    snapshot.damped_block_count = damped_block_count;
    snapshot.pruned_block_count = pruned_block_count;
    snapshot.mode = effective_mode;
    g->last_drift_snapshot = snapshot;
}

float graph_drift_record_overhead(
    ModelGraphHandle* g,
    double stabilizer_seconds,
    double decode_seconds
) {
    if (g == nullptr) return 0.0f;
    float overhead = 0.0f;
    if (decode_seconds > 0.0 && stabilizer_seconds > 0.0) {
        overhead = static_cast<float>(stabilizer_seconds / decode_seconds);
    }
    overhead = std::clamp(overhead, 0.0f, 4.0f);

    const std::size_t window = g->drift_overhead_window.size();
    if (window == 0) {
        return overhead;
    }
    const std::size_t idx = static_cast<std::size_t>(
        std::clamp(g->drift_overhead_index, 0, static_cast<int>(window - 1))
    );
    if (g->drift_overhead_count >= static_cast<int>(window)) {
        g->drift_overhead_sum -= g->drift_overhead_window[idx];
    } else {
        g->drift_overhead_count += 1;
    }
    g->drift_overhead_window[idx] = overhead;
    g->drift_overhead_sum += overhead;
    g->drift_overhead_index = (static_cast<int>(idx) + 1) % static_cast<int>(window);
    if (g->drift_overhead_count <= 0) {
        return overhead;
    }
    return static_cast<float>(g->drift_overhead_sum / static_cast<double>(g->drift_overhead_count));
}

void graph_drift_maybe_adjust_mode_from_overhead(ModelGraphHandle* g, float overhead_avg) {
    if (g == nullptr) return;
    if (g->drift_config.mode != GRAPH_DRIFT_MODE_AGGRESSIVE || g->drift_config.enabled == 0) {
        g->drift_force_conservative = false;
        g->drift_overhead_below_target_streak = 0;
        return;
    }
    if (!g->drift_force_conservative && overhead_avg > GRAPH_DRIFT_OVERHEAD_MAX) {
        g->drift_force_conservative = true;
        g->drift_auto_downgrade_events += 1;
        g->drift_overhead_below_target_streak = 0;
        return;
    }
    if (!g->drift_force_conservative) {
        return;
    }
    if (overhead_avg < GRAPH_DRIFT_OVERHEAD_TARGET) {
        g->drift_overhead_below_target_streak += 1;
    } else {
        g->drift_overhead_below_target_streak = 0;
    }
    if (g->drift_overhead_below_target_streak >= GRAPH_DRIFT_RECOVERY_STEPS) {
        g->drift_force_conservative = false;
        g->drift_auto_upgrade_events += 1;
        g->drift_overhead_below_target_streak = 0;
    }
}

void graph_drift_update_after_decode(
    ModelGraphHandle* g,
    int token_count,
    int processed_tokens,
    double decode_seconds,
    GraphDriftSnapshot* drift_out
) {
    if (g == nullptr) return;
    token_count = std::clamp(token_count, 0, g->max_seq);
    processed_tokens = std::max(1, processed_tokens);
    g->drift_config = sanitize_graph_drift_config(g->drift_config, g->max_seq);
    graph_drift_ensure_state(g);

    double stabilizer_seconds = 0.0;
    if (g->drift_config.enabled != 0) {
        g->drift_tokens_since_update += processed_tokens;
        g->drift_tokens_since_prune += processed_tokens;
        const bool allow_update = (
            g->drift_stabilizer_calls_total == 0
            || g->drift_tokens_since_update >= g->drift_config.update_interval_tokens
        );
        if (allow_update) {
            const double started = perf_now_seconds();
            const bool allow_prune_step =
                g->drift_tokens_since_prune >= g->drift_config.prune_interval_tokens;
            graph_drift_evaluate_policy(g, token_count, allow_prune_step);
            if (allow_prune_step) {
                g->drift_tokens_since_prune = 0;
            }
            g->drift_tokens_since_update = 0;
            stabilizer_seconds = perf_now_seconds() - started;
            g->drift_stabilizer_seconds_total += stabilizer_seconds;
            g->drift_stabilizer_calls_total += 1;
        } else {
            g->last_drift_snapshot.active_token_count = graph_drift_count_active_tokens(g, token_count);
        }
    } else {
        g->drift_tokens_since_update = 0;
        g->drift_tokens_since_prune = 0;
        g->drift_force_conservative = false;
        GraphDriftSnapshot snapshot = default_graph_drift_snapshot(g->drift_config.mode);
        snapshot.active_token_count = token_count;
        g->last_drift_snapshot = snapshot;
    }

    const float overhead_avg = graph_drift_record_overhead(g, stabilizer_seconds, decode_seconds);
    graph_drift_maybe_adjust_mode_from_overhead(g, overhead_avg);
    g->last_drift_snapshot.stabilizer_seconds = g->drift_stabilizer_seconds_total;
    g->last_drift_snapshot.stabilizer_calls = g->drift_stabilizer_calls_total;
    g->last_drift_snapshot.mode = graph_drift_effective_mode(g);
    if (drift_out != nullptr) {
        *drift_out = g->last_drift_snapshot;
    }
}

void granite_moe_ffn(
    ModelGraphHandle* g,
    const LayerConfig& lc,
    const float* state,
    float* output
) {
    std::memset(output, 0, static_cast<std::size_t>(g->dim) * sizeof(float));
    if (lc.ffn_norm == nullptr) {
        return;
    }

    float* normed = g->scratch_normed.data();
    rmsnorm_copy(state, lc.ffn_norm, normed, g->dim, g->rms_eps);

    if (lc.has_shared_experts() && lc.shared_hidden_dim > 0) {
        simd_fused_expert_swiglu(
            normed,
            g->dim,
            lc.shared_gate_quant,
            lc.shared_gate_qtype,
            lc.shared_hidden_dim,
            lc.shared_up_quant,
            lc.shared_up_qtype,
            lc.shared_down_quant,
            lc.shared_down_qtype,
            g->dim,
            output
        );
        account_matvec_quant(
            &g->perf_stats.moe_bytes,
            &g->perf_stats.moe_flops,
            lc.shared_gate_qtype,
            lc.shared_hidden_dim,
            g->dim
        );
        account_matvec_quant(
            &g->perf_stats.moe_bytes,
            &g->perf_stats.moe_flops,
            lc.shared_up_qtype,
            lc.shared_hidden_dim,
            g->dim
        );
        account_matvec_quant(
            &g->perf_stats.moe_bytes,
            &g->perf_stats.moe_flops,
            lc.shared_down_qtype,
            g->dim,
            lc.shared_hidden_dim
        );
    }

    if (!lc.has_routed_experts() || lc.expert_hidden_dim <= 0) {
        sanitize_tensor_inplace(output, g->dim);
        return;
    }

    const int top_k = std::max(1, std::min(lc.moe_top_k > 0 ? lc.moe_top_k : 1, lc.expert_count));
    g->ensure_expert_scratch(top_k);
    int* expert_indices = g->scratch_expert_indices.data();
    float* expert_scores = g->scratch_expert_scores.data();
    float* expert_weights = g->scratch_expert_weights.data();
    int selected = 0;
    if (lc.router != nullptr && top_k <= 2) {
        selected = granite_router_top_k_streaming(
            lc.router,
            normed,
            lc.expert_count,
            g->dim,
            top_k,
            expert_indices,
            expert_scores
        );
    } else {
        float* router_logits = g->scratch_gate.data();
        matvec_f32(lc.router, normed, router_logits, lc.expert_count, g->dim);
        account_matvec_f32(
            &g->perf_stats.moe_bytes,
            &g->perf_stats.moe_flops,
            lc.expert_count,
            g->dim
        );
        for (int i = 0; i < top_k; ++i) {
            expert_indices[i] = 0;
            expert_scores[i] = -INFINITY;
            expert_weights[i] = 0.0f;
        }
        selected = 0;
        for (int i = 0; i < lc.expert_count; ++i) {
            const float score = router_logits[i];
            if (selected >= top_k && score <= expert_scores[top_k - 1]) {
                continue;
            }
            int insert_at = std::min(selected, top_k - 1);
            while (insert_at > 0 && expert_scores[insert_at - 1] < score) {
                insert_at -= 1;
            }
            const int upper = std::min(selected, top_k - 1);
            for (int j = upper; j > insert_at; --j) {
                expert_scores[j] = expert_scores[j - 1];
                expert_indices[j] = expert_indices[j - 1];
            }
            expert_scores[insert_at] = score;
            expert_indices[insert_at] = i;
            if (selected < top_k) {
                selected += 1;
            }
        }
    }
    if (selected <= 0 || !std::isfinite(expert_scores[0])) {
        sanitize_tensor_inplace(output, g->dim);
        return;
    }
    float score_max = expert_scores[0];
    float denom = 0.0f;
    for (int i = 0; i < selected; ++i) {
        const float w = anvil_fast_math::fast_exp_scalar(expert_scores[i] - score_max);
        expert_weights[i] = w;
        denom += w;
    }
    if (denom <= 0.0f) {
        return;
    }
    for (int i = 0; i < selected; ++i) {
        expert_weights[i] /= denom;
    }

    simd_fused_moe_ffn(
        normed,
        g->dim,
        expert_indices,
        expert_weights,
        selected,
        lc.expert_gate_ptrs,
        lc.expert_gate_qtype,
        lc.expert_hidden_dim,
        lc.expert_up_ptrs,
        lc.expert_up_qtype,
        lc.expert_down_ptrs,
        lc.expert_down_qtype,
        g->dim,
        output,
        lc.has_shared_experts() ? 1 : 0
    );
    account_matvec_quant(
        &g->perf_stats.moe_bytes,
        &g->perf_stats.moe_flops,
        lc.expert_gate_qtype,
        lc.expert_hidden_dim,
        g->dim,
        selected
    );
    account_matvec_quant(
        &g->perf_stats.moe_bytes,
        &g->perf_stats.moe_flops,
        lc.expert_up_qtype,
        lc.expert_hidden_dim,
        g->dim,
        selected
    );
    account_matvec_quant(
        &g->perf_stats.moe_bytes,
        &g->perf_stats.moe_flops,
        lc.expert_down_qtype,
        g->dim,
        lc.expert_hidden_dim,
        selected
    );
    sanitize_tensor_inplace(output, g->dim);
}

bool lookup_token_embedding(
    ModelGraphHandle* g,
    int token_id,
    const float** embedding_out
) {
    if (g == nullptr || embedding_out == nullptr) {
        return false;
    }
    if (token_id < 0 || token_id >= g->vocab_size) {
        return false;
    }

    if (g->token_embeddings != nullptr) {
        const double started = perf_now_seconds();
        if (g->token_embeddings_transposed) {
            const float* embedding = g->token_embeddings + static_cast<std::size_t>(token_id);
            float* state = g->scratch_aux2.data();
            for (int d = 0; d < g->dim; ++d) {
                state[d] = embedding[static_cast<std::size_t>(d) * g->vocab_size];
            }
            g->perf_stats.embedding_lookup_seconds += perf_now_seconds() - started;
            *embedding_out = state;
            return true;
        }

        *embedding_out =
            g->token_embeddings + static_cast<std::size_t>(token_id) * g->dim;
        g->perf_stats.embedding_lookup_seconds += perf_now_seconds() - started;
        return true;
    }

    if (g->token_embeddings_quant == nullptr) {
        return false;
    }
    if (g->token_embeddings_input_dim != g->dim) {
        return false;
    }
    if (g->token_embeddings_output_dim != g->vocab_size) {
        return false;
    }
    if (token_id >= g->token_embeddings_output_dim) {
        return false;
    }

    const double started = perf_now_seconds();
    float* state = g->scratch_aux2.data();
    if (
        simd_dequantize_row(
            g->token_embeddings_quant,
            g->token_embeddings_qtype,
            token_id,
            state,
            g->token_embeddings_input_dim
        ) != 1
    ) {
        return false;
    }
    g->perf_stats.embedding_lookup_seconds += perf_now_seconds() - started;
    *embedding_out = state;
    return true;
}

}  // namespace

extern "C" {

void simd_mul_silu_gate_inplace(float* x, const float* gate, int len) {
    mul_silu_gate_inplace(x, gate, len);
}

// Create the model graph with explicit architecture parameters.
ModelGraphHandle* create_model_graph_v2(
    int n_layers, int dim, int vocab_size,
    int n_heads, int n_kv_heads, int head_dim,
    int max_seq, float rms_eps, float rope_theta
) {
    if (n_layers <= 0 || dim <= 0 || vocab_size <= 0 ||
        n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq <= 0) {
        return nullptr;
    }
    try {
        return new ModelGraphHandle(n_layers, dim, vocab_size, n_heads, n_kv_heads,
                                    head_dim, max_seq, rms_eps, rope_theta);
    } catch (...) {
        return nullptr;
    }
}

// Legacy ABI: keep 3-arg constructor for older ctypes callers/tests.
ModelGraphHandle* create_model_graph(int n_layers, int dim, int vocab_size) {
    if (n_layers <= 0 || dim <= 0 || vocab_size <= 0) {
        return nullptr;
    }
    constexpr int kDefaultMaxSeq = 2048;
    constexpr float kDefaultRmsEps = 1e-6f;
    constexpr float kDefaultRopeTheta = 10000.0f;
    return create_model_graph_v2(
        n_layers, dim, vocab_size,
        1, 1, dim,
        kDefaultMaxSeq, kDefaultRmsEps, kDefaultRopeTheta
    );
}

int graph_set_drift_config(ModelGraphHandle* g, const GraphDriftConfig* config) {
    if (g == nullptr || config == nullptr) return 0;
    const GraphDriftConfig normalized = sanitize_graph_drift_config(*config, g->max_seq);
    const bool shape_changed =
        normalized.block_size_tokens != g->drift_config.block_size_tokens;
    const bool mode_or_enabled_changed =
        normalized.mode != g->drift_config.mode
        || normalized.enabled != g->drift_config.enabled;
    g->drift_config = normalized;
    if (shape_changed || mode_or_enabled_changed) {
        graph_drift_reset_runtime(g);
    } else {
        g->last_drift_snapshot.mode = graph_drift_effective_mode(g);
    }
    return 1;
}

int graph_get_drift_config(const ModelGraphHandle* g, GraphDriftConfig* out) {
    if (g == nullptr || out == nullptr) return 0;
    *out = g->drift_config;
    return 1;
}

int graph_get_last_drift_snapshot(const ModelGraphHandle* g, GraphDriftSnapshot* out) {
    if (g == nullptr || out == nullptr) return 0;
    *out = g->last_drift_snapshot;
    return 1;
}

// Set float32 layer weight pointers (called once during init)
int graph_set_layer_weights(
    ModelGraphHandle* g, int layer_idx,
    const float* attn_norm, const float* ffn_norm,
    const float* wq, int q_out_dim,
    const float* wk, int kv_out_dim,
    const float* wv,
    const float* wo,
    const float* w_gate, int ffn_dim,
    const float* w_up,
    const float* w_down,
    int is_attention
) {
    if (g == nullptr || layer_idx < 0 || layer_idx >= g->n_layers) return 0;

    auto& lc = g->layers[static_cast<std::size_t>(layer_idx)];
    lc.attn_norm = attn_norm;
    lc.ffn_norm = ffn_norm;
    lc.wq = wq;
    lc.q_out_dim = q_out_dim > 0 ? q_out_dim : g->n_heads * g->head_dim;
    lc.wk = wk;
    lc.kv_out_dim = kv_out_dim > 0 ? kv_out_dim : g->n_kv_heads * g->head_dim;
    lc.wv = wv;
    lc.wo = wo;
    lc.w_gate = w_gate;
    lc.ffn_dim = ffn_dim > 0 ? ffn_dim : 4 * g->dim;
    lc.w_up = w_up;
    lc.w_down = w_down;
    lc.is_attention = (is_attention != 0);
    lc.layer_idx = layer_idx;

    g->ensure_scratch(lc.ffn_dim);
    g->ensure_attention_scratch(lc.q_out_dim, lc.kv_out_dim);
    return 1;
}

// Set quantized layer weight pointers — no dequantization, zero-copy
int graph_set_layer_weights_quantized(
    ModelGraphHandle* g, int layer_idx,
    const float* attn_norm, const float* ffn_norm,
    const void* wq, int wq_qtype, int q_out_dim,
    const void* wk, int wk_qtype, int kv_out_dim,
    const void* wv, int wv_qtype,
    const void* wo, int wo_qtype,
    const void* w_gate, int wgate_qtype, int ffn_dim,
    const void* w_up, int wup_qtype,
    const void* w_down, int wdown_qtype,
    int is_attention
) {
    if (g == nullptr || layer_idx < 0 || layer_idx >= g->n_layers) return 0;

    auto& lc = g->layers[static_cast<std::size_t>(layer_idx)];
    lc.attn_norm = attn_norm;
    lc.ffn_norm = ffn_norm;

    // Clear float32 paths
    lc.wq = nullptr; lc.wk = nullptr; lc.wv = nullptr; lc.wo = nullptr;
    lc.w_gate = nullptr; lc.w_up = nullptr; lc.w_down = nullptr;

    // Set quantized paths
    lc.wq_quant = wq;       lc.wq_qtype = wq_qtype;
    lc.wk_quant = wk;       lc.wk_qtype = wk_qtype;
    lc.wv_quant = wv;       lc.wv_qtype = wv_qtype;
    lc.wo_quant = wo;       lc.wo_qtype = wo_qtype;
    lc.wgate_quant = w_gate; lc.wgate_qtype = wgate_qtype;
    lc.wup_quant = w_up;    lc.wup_qtype = wup_qtype;
    lc.wdown_quant = w_down; lc.wdown_qtype = wdown_qtype;

    lc.q_out_dim = q_out_dim > 0 ? q_out_dim : g->n_heads * g->head_dim;
    lc.kv_out_dim = kv_out_dim > 0 ? kv_out_dim : g->n_kv_heads * g->head_dim;
    lc.ffn_dim = ffn_dim > 0 ? ffn_dim : 4 * g->dim;
    lc.is_attention = (is_attention != 0);
    lc.layer_idx = layer_idx;

    g->ensure_scratch(lc.ffn_dim);
    g->ensure_attention_scratch(lc.q_out_dim, lc.kv_out_dim);
    return 1;
}

int graph_set_layer_extras(
    ModelGraphHandle* g, int layer_idx, int layer_kind,
    const float* attn_q_norm, const float* attn_k_norm,
    const float* ssm_a, const float* ssm_d, const float* ssm_dt,
    const float* ssm_conv, int ssm_conv_rows, int ssm_conv_cols,
    const float* ssm_conv_bias,
    const float* ssm_norm,
    const float* router,
    const void* attn_gate, int attn_gate_qtype, int attn_gate_dim,
    const void* ssm_in, int ssm_in_qtype, int ssm_in_dim,
    const void* ssm_out, int ssm_out_qtype, int ssm_out_dim,
    const void* ssm_alpha, int ssm_alpha_qtype, int ssm_alpha_dim,
    const void* ssm_beta, int ssm_beta_qtype, int ssm_beta_dim,
    const void* shared_gate, int shared_gate_qtype, int shared_hidden_dim,
    const void* shared_up, int shared_up_qtype,
    const void* shared_down, int shared_down_qtype,
    const void* const* expert_gate_ptrs, int expert_gate_qtype, int expert_hidden_dim,
    const void* const* expert_up_ptrs, int expert_up_qtype,
    const void* const* expert_down_ptrs, int expert_down_qtype,
    int expert_count, int moe_top_k
) {
    if (g == nullptr || layer_idx < 0 || layer_idx >= g->n_layers) return 0;
    auto& lc = g->layers[static_cast<std::size_t>(layer_idx)];
    lc.layer_kind = layer_kind;
    lc.attn_q_norm = attn_q_norm;
    lc.attn_k_norm = attn_k_norm;
    lc.ssm_a = ssm_a;
    lc.ssm_d = ssm_d;
    lc.ssm_dt = ssm_dt;
    lc.ssm_conv = ssm_conv;
    lc.ssm_conv_rows = ssm_conv_rows;
    lc.ssm_conv_cols = ssm_conv_cols;
    lc.ssm_conv_bias = ssm_conv_bias;
    lc.ssm_norm = ssm_norm;
    lc.router = router;

    lc.attn_gate_quant = attn_gate;
    lc.attn_gate_qtype = attn_gate_qtype;
    lc.attn_gate_dim = attn_gate_dim;
    lc.ssm_in_quant = ssm_in;
    lc.ssm_in_qtype = ssm_in_qtype;
    lc.ssm_in_dim = ssm_in_dim;
    lc.ssm_out_quant = ssm_out;
    lc.ssm_out_qtype = ssm_out_qtype;
    lc.ssm_out_dim = ssm_out_dim;
    lc.ssm_alpha_quant = ssm_alpha;
    lc.ssm_alpha_qtype = ssm_alpha_qtype;
    lc.ssm_alpha_dim = ssm_alpha_dim;
    lc.ssm_beta_quant = ssm_beta;
    lc.ssm_beta_qtype = ssm_beta_qtype;
    lc.ssm_beta_dim = ssm_beta_dim;
    lc.shared_gate_quant = shared_gate;
    lc.shared_gate_qtype = shared_gate_qtype;
    lc.shared_hidden_dim = shared_hidden_dim;
    lc.shared_up_quant = shared_up;
    lc.shared_up_qtype = shared_up_qtype;
    lc.shared_down_quant = shared_down;
    lc.shared_down_qtype = shared_down_qtype;
    lc.expert_gate_ptrs = expert_gate_ptrs;
    lc.expert_gate_qtype = expert_gate_qtype;
    lc.expert_hidden_dim = expert_hidden_dim;
    lc.expert_up_ptrs = expert_up_ptrs;
    lc.expert_up_qtype = expert_up_qtype;
    lc.expert_down_ptrs = expert_down_ptrs;
    lc.expert_down_qtype = expert_down_qtype;
    lc.expert_count = expert_count;
    lc.moe_top_k = moe_top_k;

    if (!lc.has_shared_experts()) {
        lc.shared_gate_quant = nullptr;
        lc.shared_gate_qtype = 0;
        lc.shared_up_quant = nullptr;
        lc.shared_up_qtype = 0;
        lc.shared_down_quant = nullptr;
        lc.shared_down_qtype = 0;
        lc.shared_hidden_dim = 0;
    }
    if (!lc.has_routed_experts()) {
        lc.router = nullptr;
        lc.expert_gate_ptrs = nullptr;
        lc.expert_gate_qtype = 0;
        lc.expert_up_ptrs = nullptr;
        lc.expert_up_qtype = 0;
        lc.expert_down_ptrs = nullptr;
        lc.expert_down_qtype = 0;
        lc.expert_hidden_dim = 0;
        lc.expert_count = 0;
        lc.moe_top_k = 0;
    } else {
        lc.moe_top_k = std::max(1, std::min(lc.moe_top_k, lc.expert_count));
    }

    g->ensure_scratch(std::max({
        lc.ffn_dim,
        lc.ssm_in_dim,
        lc.attn_gate_dim,
        lc.ssm_out_dim,
        lc.shared_hidden_dim,
        lc.expert_hidden_dim,
        lc.ssm_conv_cols,
    }));
    const int hist = std::max(0, ssm_conv_rows - 1) * std::max(0, ssm_conv_cols);
    g->conv_state_storage[static_cast<std::size_t>(layer_idx)].assign(
        static_cast<std::size_t>(hist), 0.0f
    );
    g->conv_state_head_storage[static_cast<std::size_t>(layer_idx)] = 0;
    g->ssm_state_storage[static_cast<std::size_t>(layer_idx)].clear();
    return 1;
}

int graph_set_qwen_mrope_config(
    ModelGraphHandle* g,
    int rope_finetuned,
    int interleaved,
    int rope_dim,
    int sec_t,
    int sec_h,
    int sec_w,
    int sec_e
) {
    if (g == nullptr) return 0;
    g->qwen_rope_finetuned = (rope_finetuned != 0);
    g->qwen_mrope_interleaved = (interleaved != 0);
    g->qwen_rope_dim = std::max(0, rope_dim);
    g->qwen_sec_t = std::max(0, sec_t);
    g->qwen_sec_h = std::max(0, sec_h);
    g->qwen_sec_w = std::max(0, sec_w);
    g->qwen_sec_e = std::max(0, sec_e);
    return 1;
}

int graph_set_qwen_hybrid_config(
    ModelGraphHandle* g,
    int ssm_state_size,
    int ssm_group_count,
    int ssm_n_v_heads
) {
    if (g == nullptr) return 0;
    g->qwen_ssm_state_size = std::max(0, ssm_state_size);
    g->qwen_ssm_group_count = std::max(0, ssm_group_count);
    g->qwen_ssm_n_v_heads = std::max(0, ssm_n_v_heads);
    return 1;
}

// Set final norm and LM head pointers
int graph_set_head_weights(
    ModelGraphHandle* g,
    const float* final_norm,
    const float* lm_head,
    int lm_head_transposed,
    float embedding_scale,
    float residual_scale,
    float logit_scale,
    float attention_scale
) {
    if (g == nullptr) return 0;
    g->final_norm = final_norm;
    g->lm_head = lm_head;
    g->lm_head_transposed = (lm_head_transposed != 0);
    g->embedding_scale = embedding_scale;
    g->residual_scale = residual_scale;
    g->logit_scale = logit_scale;
    g->attention_scale = attention_scale;
    return 1;
}

int graph_set_embedding_weights(
    ModelGraphHandle* g,
    const float* token_embeddings,
    int embeddings_transposed
) {
    if (g == nullptr) return 0;
    g->token_embeddings = token_embeddings;
    g->token_embeddings_quant = nullptr;
    g->token_embeddings_qtype = 0;
    g->token_embeddings_input_dim = 0;
    g->token_embeddings_output_dim = 0;
    g->token_embeddings_transposed = (embeddings_transposed != 0);
    return 1;
}

int graph_set_embedding_weights_quantized(
    ModelGraphHandle* g,
    const void* token_embeddings_quant,
    int token_embeddings_qtype,
    int token_embeddings_input_dim,
    int token_embeddings_output_dim,
    int embeddings_transposed
) {
    if (g == nullptr || token_embeddings_quant == nullptr) return 0;
    if (!is_supported_quant_qtype(token_embeddings_qtype)) return 0;
    if (token_embeddings_input_dim <= 0 || token_embeddings_output_dim <= 0) return 0;
    g->token_embeddings = nullptr;
    g->token_embeddings_quant = token_embeddings_quant;
    g->token_embeddings_qtype = token_embeddings_qtype;
    g->token_embeddings_input_dim = token_embeddings_input_dim;
    g->token_embeddings_output_dim = token_embeddings_output_dim;
    g->token_embeddings_transposed = (embeddings_transposed != 0);
    return 1;
}

// Set quantized LM head
int graph_set_head_weights_quantized(
    ModelGraphHandle* g,
    const float* final_norm,
    const void* lm_head_quant, int lm_head_qtype,
    float embedding_scale,
    float residual_scale,
    float logit_scale,
    float attention_scale
) {
    if (g == nullptr) return 0;
    g->final_norm = final_norm;
    g->lm_head = nullptr;
    g->lm_head_quant = lm_head_quant;
    g->lm_head_qtype = lm_head_qtype;
    g->lm_head_transposed = false;
    g->embedding_scale = embedding_scale;
    g->residual_scale = residual_scale;
    g->logit_scale = logit_scale;
    g->attention_scale = attention_scale;
    return 1;
}

// Core single-token decode body shared by graph_forward_token APIs.
static int graph_forward_token_impl(
    ModelGraphHandle* g,
    const float* hidden_in,
    int hidden_len,
    float* logits_out,
    int logits_len,
    int pos,
    bool emit_logits,
    int layer_begin,
    int layer_end,
    bool apply_input_scale,
    bool capture_last_hidden,
    bool advance_position
) {
    if (g == nullptr || hidden_in == nullptr || logits_out == nullptr) return 0;
    if (hidden_len != g->dim || logits_len != g->vocab_size) return 0;
    if (pos < 0 || pos >= g->max_seq) return 0;
    if (g->drift_config.enabled != 0) {
        graph_drift_ensure_state(g);
    }

    int dim = g->dim;
    int n_heads = g->n_heads;
    int n_kv_heads = g->n_kv_heads;
    int head_dim = g->head_dim;
    float eps = g->rms_eps;

    // Initialize state from embedding
    float* state = g->scratch_state.data();
    std::memcpy(state, hidden_in, static_cast<std::size_t>(dim) * sizeof(float));

    // Apply embedding scale if set
    if (apply_input_scale && g->embedding_scale != 0.0f) {
        scale_inplace(state, dim, g->embedding_scale);
    }

    float scale = g->attention_scale > 0.0f
        ? g->attention_scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    auto add_residual = [&](const float* branch) {
        add_residual_inplace(state, branch, dim, g->residual_scale);
    };

    auto sanitize_tracked = [&](float* x, int len) {
        const double started = perf_now_seconds();
        sanitize_tensor_inplace(x, len);
        g->perf_stats.sanitize_seconds += perf_now_seconds() - started;
    };

    auto run_simple_ffn = [&](const LayerConfig& lc) {
        if (!(lc.has_gate() && lc.has_up() && lc.has_down() && lc.ffn_norm != nullptr)) {
            return;
        }
        g->perf_stats.ffn_calls += 1;
        float* normed = g->scratch_normed.data();
        {
            const double started = perf_now_seconds();
            rmsnorm_copy(state, lc.ffn_norm, normed, dim, eps);
            g->perf_stats.ffn_norm_seconds += perf_now_seconds() - started;
        }
        const bool can_fuse_quant_ffn =
            lc.w_gate == nullptr &&
            lc.w_up == nullptr &&
            lc.w_down == nullptr &&
            lc.wgate_quant != nullptr &&
            lc.wup_quant != nullptr &&
            lc.wdown_quant != nullptr &&
            is_supported_quant_qtype(lc.wgate_qtype) &&
            is_supported_quant_qtype(lc.wup_qtype) &&
            is_supported_quant_qtype(lc.wdown_qtype);
        if (can_fuse_quant_ffn) {
            const double started = perf_now_seconds();
            simd_fused_expert_swiglu(
                normed,
                dim,
                lc.wgate_quant,
                lc.wgate_qtype,
                lc.ffn_dim,
                lc.wup_quant,
                lc.wup_qtype,
                lc.wdown_quant,
                lc.wdown_qtype,
                dim,
                g->scratch_proj.data()
            );
            account_matvec_quant(
                &g->perf_stats.ffn_gate_up_bytes,
                &g->perf_stats.ffn_gate_up_flops,
                lc.wgate_qtype,
                lc.ffn_dim,
                dim
            );
            account_matvec_quant(
                &g->perf_stats.ffn_gate_up_bytes,
                &g->perf_stats.ffn_gate_up_flops,
                lc.wup_qtype,
                lc.ffn_dim,
                dim
            );
            account_matvec_quant(
                &g->perf_stats.ffn_down_bytes,
                &g->perf_stats.ffn_down_flops,
                lc.wdown_qtype,
                dim,
                lc.ffn_dim
            );
            const double elapsed = perf_now_seconds() - started;
            const double total_work = static_cast<double>(lc.ffn_dim * 2 + dim);
            const double gate_up_share =
                total_work > 0.0 ? static_cast<double>(lc.ffn_dim * 2) / total_work : 0.67;
            g->perf_stats.ffn_gate_up_seconds += elapsed * gate_up_share;
            g->perf_stats.ffn_down_seconds += elapsed * (1.0 - gate_up_share);
        } else {
            float* gate_out = g->scratch_gate.data();
            float* up_out = g->scratch_up.data();
            float* ffn_hidden = g->scratch_ffn.data();
            {
                const double started = perf_now_seconds();
                matvec_dispatch(lc.w_gate, lc.wgate_quant, lc.wgate_qtype, normed, gate_out, lc.ffn_dim, dim);
                matvec_dispatch(lc.w_up, lc.wup_quant, lc.wup_qtype, normed, up_out, lc.ffn_dim, dim);
                account_matvec_dispatch(
                    &g->perf_stats.ffn_gate_up_bytes,
                    &g->perf_stats.ffn_gate_up_flops,
                    lc.w_gate,
                    lc.wgate_quant,
                    lc.wgate_qtype,
                    lc.ffn_dim,
                    dim
                );
                account_matvec_dispatch(
                    &g->perf_stats.ffn_gate_up_bytes,
                    &g->perf_stats.ffn_gate_up_flops,
                    lc.w_up,
                    lc.wup_quant,
                    lc.wup_qtype,
                    lc.ffn_dim,
                    dim
                );
                g->perf_stats.ffn_gate_up_seconds += perf_now_seconds() - started;
            }
            sanitize_tracked(gate_out, lc.ffn_dim);
            sanitize_tracked(up_out, lc.ffn_dim);
            swiglu_f32(gate_out, up_out, ffn_hidden, lc.ffn_dim);
            sanitize_tracked(ffn_hidden, lc.ffn_dim);
            {
                const double started = perf_now_seconds();
                matvec_dispatch(
                    lc.w_down,
                    lc.wdown_quant,
                    lc.wdown_qtype,
                    ffn_hidden,
                    g->scratch_proj.data(),
                    dim,
                    lc.ffn_dim
                );
                account_matvec_dispatch(
                    &g->perf_stats.ffn_down_bytes,
                    &g->perf_stats.ffn_down_flops,
                    lc.w_down,
                    lc.wdown_quant,
                    lc.wdown_qtype,
                    dim,
                    lc.ffn_dim
                );
                g->perf_stats.ffn_down_seconds += perf_now_seconds() - started;
            }
        }
        sanitize_tracked(g->scratch_proj.data(), dim);
        add_residual(g->scratch_proj.data());
    };

    auto run_attention = [&](const LayerConfig& lc, int layer_idx, bool qwen_full) {
        if (!(lc.is_attention && lc.has_wq() && lc.has_wk() &&
              lc.has_wv() && lc.has_wo() && lc.attn_norm != nullptr)) {
            return;
        }
        g->perf_stats.attention_calls += 1;
        const int layer_n_heads = (lc.q_out_dim > 0 && lc.q_out_dim % head_dim == 0)
            ? std::max(1, lc.q_out_dim / head_dim)
            : n_heads;
        const int layer_n_kv_heads = (lc.kv_out_dim > 0 && lc.kv_out_dim % head_dim == 0)
            ? std::max(1, lc.kv_out_dim / head_dim)
            : n_kv_heads;
        const int layer_q_dim = layer_n_heads * head_dim;
        const int layer_kv_dim = layer_n_kv_heads * head_dim;
        float* normed = g->scratch_normed.data();
        float* q = g->scratch_q.data();
        float* k = g->scratch_k.data();
        float* v = g->scratch_v.data();
        {
            const double started = perf_now_seconds();
            rmsnorm_copy(state, lc.attn_norm, normed, dim, eps);
            const bool fused_qkv = (
                lc.wq == nullptr && lc.wk == nullptr && lc.wv == nullptr
                && lc.wq_quant != nullptr && lc.wk_quant != nullptr && lc.wv_quant != nullptr
                && fused_qkv_matvec_dispatch(
                    normed,
                    lc.wq_quant, lc.wq_qtype, q, lc.q_out_dim,
                    lc.wk_quant, lc.wk_qtype, k, lc.kv_out_dim,
                    lc.wv_quant, lc.wv_qtype, v, lc.kv_out_dim,
                    dim
                )
            );
            if (!fused_qkv) {
                matvec_dispatch(lc.wq, lc.wq_quant, lc.wq_qtype, normed, q, lc.q_out_dim, dim);
                matvec_dispatch(lc.wk, lc.wk_quant, lc.wk_qtype, normed, k, lc.kv_out_dim, dim);
                matvec_dispatch(lc.wv, lc.wv_quant, lc.wv_qtype, normed, v, lc.kv_out_dim, dim);
                account_matvec_dispatch(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wq,
                    lc.wq_quant,
                    lc.wq_qtype,
                    lc.q_out_dim,
                    dim
                );
                account_matvec_dispatch(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wk,
                    lc.wk_quant,
                    lc.wk_qtype,
                    lc.kv_out_dim,
                    dim
                );
                account_matvec_dispatch(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wv,
                    lc.wv_quant,
                    lc.wv_qtype,
                    lc.kv_out_dim,
                    dim
                );
            } else {
                account_matvec_quant(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wq_qtype,
                    lc.q_out_dim,
                    dim
                );
                account_matvec_quant(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wk_qtype,
                    lc.kv_out_dim,
                    dim
                );
                account_matvec_quant(
                    &g->perf_stats.attention_proj_bytes,
                    &g->perf_stats.attention_proj_flops,
                    lc.wv_qtype,
                    lc.kv_out_dim,
                    dim
                );
            }
            g->perf_stats.attention_proj_seconds += perf_now_seconds() - started;
        }
        sanitize_tracked(q, lc.q_out_dim);
        sanitize_tracked(k, lc.kv_out_dim);
        sanitize_tracked(v, lc.kv_out_dim);

        {
            const double started = perf_now_seconds();
        if (qwen_full && lc.attn_q_norm != nullptr) {
            rmsnorm_rows_inplace(q, lc.attn_q_norm, layer_n_heads, head_dim, eps);
        }
        if (qwen_full && lc.attn_k_norm != nullptr) {
            rmsnorm_rows_inplace(k, lc.attn_k_norm, layer_n_kv_heads, head_dim, eps);
        }
        if (qwen_full) {
            int qwen_rope_dim = g->qwen_rope_dim > 0 ? g->qwen_rope_dim : head_dim;
            qwen_rope_dim = std::max(2, std::min(head_dim, qwen_rope_dim));
            qwen_rope_dim -= (qwen_rope_dim % 2);
            const bool has_qwen_sections =
                g->qwen_sec_t > 0 || g->qwen_sec_h > 0 || g->qwen_sec_w > 0 || g->qwen_sec_e > 0;
            if (qwen_rope_dim > 0) {
                if (g->qwen_mrope_interleaved && has_qwen_sections) {
                    apply_qwen_mrope(
                        q,
                        k,
                        layer_n_heads,
                        layer_n_kv_heads,
                        head_dim,
                        qwen_rope_dim,
                        pos,
                        g->rope_theta,
                        g->qwen_mrope_interleaved,
                        g->qwen_sec_t,
                        g->qwen_sec_h,
                        g->qwen_sec_w,
                        g->qwen_sec_e
                    );
                } else {
                    apply_rope_with_dim(
                        q,
                        k,
                        layer_n_heads,
                        layer_n_kv_heads,
                        head_dim,
                        qwen_rope_dim,
                        pos,
                        g->rope_theta
                    );
                }
            }
        } else {
            apply_rope(q, k, layer_n_heads, layer_n_kv_heads, head_dim, pos, g->rope_theta);
        }

        g->kv_store.write(layer_idx, pos, k, v, layer_n_kv_heads);
            g->perf_stats.attention_rope_kv_seconds += perf_now_seconds() - started;
        }
        const int kv_len = pos + 1;
        float* attn_out = g->scratch_attn_out.data();
        const DriftDecodeContext drift_ctx = make_drift_decode_context(g);
        {
            const double started = perf_now_seconds();
            fused_gqa_attention_decode_paged(
                q,
                g->kv_store,
                layer_idx,
                attn_out,
                layer_n_heads,
                layer_n_kv_heads,
                kv_len,
                head_dim,
                scale,
                g->attention_tile_size,
                drift_ctx.block_decay_gain,
                drift_ctx.block_pruned,
                drift_ctx.block_count,
                drift_ctx.block_size,
                drift_ctx.preserve_head_tokens,
                drift_ctx.preserve_recent_tokens
            );
            g->perf_stats.attention_decode_seconds += perf_now_seconds() - started;
        }
        sanitize_tracked(attn_out, layer_q_dim);

        if (qwen_full && lc.has_attn_gate() && lc.attn_gate_dim > 0) {
            float* gate_proj = g->scratch_gate.data();
            matvec_quant(lc.attn_gate_quant, lc.attn_gate_qtype, normed, gate_proj, dim, lc.attn_gate_dim);
            if (lc.attn_gate_dim == layer_q_dim) {
                sigmoid_inplace(gate_proj, lc.attn_gate_dim);
                for (int i = 0; i < lc.attn_gate_dim; ++i) {
                    attn_out[i] *= gate_proj[i];
                }
            }
        }

        float* proj = g->scratch_proj.data();
        {
            const double started = perf_now_seconds();
            matvec_dispatch(lc.wo, lc.wo_quant, lc.wo_qtype, attn_out, proj, dim, layer_q_dim);
            account_matvec_dispatch(
                &g->perf_stats.attention_out_proj_bytes,
                &g->perf_stats.attention_out_proj_flops,
                lc.wo,
                lc.wo_quant,
                lc.wo_qtype,
                dim,
                layer_q_dim
            );
            g->perf_stats.attention_out_proj_seconds += perf_now_seconds() - started;
        }
        sanitize_tracked(proj, dim);
        add_residual(proj);
    };

    // Process each layer
    const int begin = std::max(0, layer_begin);
    const int end = std::max(begin, std::min(layer_end, g->n_layers));
    for (int l = begin; l < end; ++l) {
        const auto& lc = g->layers[static_cast<std::size_t>(l)];
        if (lc.layer_kind == LAYER_KIND_GRANITE_SSM) {
            if (!(lc.ssm_a != nullptr && lc.ssm_d != nullptr && lc.ssm_dt != nullptr &&
                  lc.ssm_conv != nullptr && lc.ssm_norm != nullptr &&
                  lc.has_ssm_in() && lc.has_ssm_out())) {
                return 0;
            }
            g->perf_stats.ssm_calls += 1;
            float* normed = g->scratch_normed.data();
            rmsnorm_copy(state, lc.attn_norm, normed, dim, eps);
            double ssm_projection_elapsed = 0.0;
            double ssm_conv_elapsed = 0.0;
            double ssm_recurrent_elapsed = 0.0;
            double ssm_output_elapsed = 0.0;
            const int ssm_heads = 48;
            const int d_inner = 3072;
            const int d_conv = lc.ssm_conv_cols;
            const int d_state = std::max(1, (d_conv - d_inner) / 2);
            const int ssm_head_dim = std::max(1, d_inner / ssm_heads);

            float* inner = g->scratch_aux0.data();
            {
                const double started = perf_now_seconds();
                matvec_quant(lc.ssm_in_quant, lc.ssm_in_qtype, normed, inner, dim, lc.ssm_in_dim);
                account_matvec_quant(
                    &g->perf_stats.ssm_projection_bytes,
                    &g->perf_stats.ssm_projection_flops,
                    lc.ssm_in_qtype,
                    lc.ssm_in_dim,
                    dim
                );
                ssm_projection_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_projection_seconds += ssm_projection_elapsed;
            }
            float* z = inner;
            float* xbc = inner + d_inner;
            float* dt_proj = inner + d_inner + d_conv;

            float* conv_out = g->scratch_aux1.data();
            {
                const double started = perf_now_seconds();
                causal_conv1d_step(
                    g->conv_state_storage[static_cast<std::size_t>(l)],
                    g->conv_state_head_storage[static_cast<std::size_t>(l)],
                    lc.ssm_conv,
                    lc.ssm_conv_rows,
                    d_conv,
                    xbc,
                    lc.ssm_conv_bias,
                    conv_out,
                    true
                );
                ssm_conv_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_conv_seconds += ssm_conv_elapsed;
            }

            float* x_ssm = conv_out;
            float* b_vec = conv_out + d_inner;
            float* c_vec = conv_out + d_inner + d_state;
            auto& layer_state = g->ssm_state_storage[static_cast<std::size_t>(l)];
            if (static_cast<int>(layer_state.size()) != ssm_heads * ssm_head_dim * d_state) {
                layer_state.assign(
                    static_cast<std::size_t>(ssm_heads * ssm_head_dim * d_state),
                    0.0f
                );
            }
            float* recurrent = g->scratch_aux2.data();
            const int ssm_threads = get_num_threads();
            {
                const double started = perf_now_seconds();
#ifdef _OPENMP
#pragma omp parallel for num_threads(ssm_threads) schedule(static)
#endif
                for (int h = 0; h < ssm_heads; ++h) {
                    maybe_bind_worker_thread(true);
                    const float dt_t = fast_softplus_scalar(dt_proj[h] + lc.ssm_dt[h]);
                    const float dA = anvil_fast_math::fast_exp_scalar(
                        std::max(-60.0f, std::min(60.0f, lc.ssm_a[h] * dt_t))
                    );
                    const float d_scale = lc.ssm_d[h];
                    const float* x_h = x_ssm + static_cast<std::size_t>(h) * ssm_head_dim;
                    float* y_h = recurrent + static_cast<std::size_t>(h) * ssm_head_dim;
                    float* state_h = layer_state.data() + static_cast<std::size_t>(h) * ssm_head_dim * d_state;
                    for (int d = 0; d < ssm_head_dim; ++d) {
                        prefetch_write(state_h + static_cast<std::size_t>(std::min(d + 1, ssm_head_dim - 1)) * d_state);
                        float* state_row = state_h + static_cast<std::size_t>(d) * d_state;
                        float y = d_scale * x_h[d];
#ifdef __AVX2__
                        const __m256 dA_v = _mm256_set1_ps(dA);
                        const __m256 xdt_v = _mm256_set1_ps(x_h[d] * dt_t);
                        __m256 vacc = _mm256_setzero_ps();
                        int s = 0;
                        for (; s + 8 <= d_state; s += 8) {
                            __m256 state_v = _mm256_loadu_ps(state_row + s);
                            __m256 b_v = _mm256_loadu_ps(b_vec + s);
                            __m256 c_v = _mm256_loadu_ps(c_vec + s);
                            state_v = _mm256_fmadd_ps(xdt_v, b_v, _mm256_mul_ps(dA_v, state_v));
                            _mm256_storeu_ps(state_row + s, state_v);
                            vacc = _mm256_fmadd_ps(state_v, c_v, vacc);
                        }
                        y += hsum256_ps(vacc);
                        for (; s < d_state; ++s) {
                            state_row[s] = dA * state_row[s] + x_h[d] * (dt_t * b_vec[s]);
                            y += state_row[s] * c_vec[s];
                        }
#else
                        for (int s = 0; s < d_state; ++s) {
                            state_row[s] = dA * state_row[s] + x_h[d] * (dt_t * b_vec[s]);
                            y += state_row[s] * c_vec[s];
                        }
#endif
                        y_h[d] = y;
                    }
                }
                ssm_recurrent_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_recurrent_seconds += ssm_recurrent_elapsed;
            }
            {
                const double started = perf_now_seconds();
                mul_silu_gate_inplace(recurrent, z, d_inner);
                rmsnorm_inplace(recurrent, lc.ssm_norm, d_inner, eps);
                matvec_quant(lc.ssm_out_quant, lc.ssm_out_qtype, recurrent, g->scratch_proj.data(), d_inner, dim);
                account_matvec_quant(
                    &g->perf_stats.ssm_output_bytes,
                    &g->perf_stats.ssm_output_flops,
                    lc.ssm_out_qtype,
                    dim,
                    d_inner
                );
                add_residual(g->scratch_proj.data());
                ssm_output_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_output_seconds += ssm_output_elapsed;
            }
            g->perf_stats.moe_calls += 1;
            {
                const double moe_started = perf_now_seconds();
                granite_moe_ffn(g, lc, state, g->scratch_proj.data());
                g->perf_stats.moe_seconds += perf_now_seconds() - moe_started;
            }
            add_residual(g->scratch_proj.data());
            sanitize_tracked(state, dim);
            g->perf_stats.ssm_seconds +=
                ssm_projection_elapsed
                + ssm_conv_elapsed
                + ssm_recurrent_elapsed
                + ssm_output_elapsed;
            continue;
        }

        if (lc.layer_kind == LAYER_KIND_GRANITE_ATTN) {
            run_attention(lc, l, false);
            g->perf_stats.moe_calls += 1;
            {
                const double moe_started = perf_now_seconds();
                granite_moe_ffn(g, lc, state, g->scratch_proj.data());
                g->perf_stats.moe_seconds += perf_now_seconds() - moe_started;
            }
            add_residual(g->scratch_proj.data());
            sanitize_tracked(state, dim);
            continue;
        }

        if (lc.layer_kind == LAYER_KIND_QWEN_HYBRID) {
            if (!(lc.has_ssm_in() && lc.has_attn_gate() && lc.has_ssm_alpha() &&
                  lc.has_ssm_beta() && lc.has_ssm_out() && lc.ssm_dt != nullptr &&
                  lc.ssm_a != nullptr && lc.ssm_conv != nullptr &&
                  lc.ssm_norm != nullptr && lc.attn_norm != nullptr)) {
                return 0;
            }
            g->perf_stats.ssm_calls += 1;
            float* normed = g->scratch_normed.data();
            rmsnorm_copy(state, lc.attn_norm, normed, dim, eps);
            double ssm_projection_elapsed = 0.0;
            double ssm_conv_elapsed = 0.0;
            double ssm_recurrent_elapsed = 0.0;
            double ssm_output_elapsed = 0.0;

            const int d_inner = lc.attn_gate_dim > 0 ? lc.attn_gate_dim : dim;
            const int inferred_conv_channels = std::max(0, lc.ssm_in_dim);
            const int inferred_qk_dim = std::max(0, (inferred_conv_channels - d_inner) / 2);
            const int meta_d_state = std::max(0, g->qwen_ssm_state_size);
            int n_groups = std::max(1, g->qwen_ssm_group_count);
            int d_state = meta_d_state;
            if (d_state <= 0 && inferred_qk_dim > 0) {
                d_state = std::max(1, inferred_qk_dim / std::max(1, n_groups));
            }
            d_state = std::max(1, d_state);
            int n_v_heads = std::max(0, g->qwen_ssm_n_v_heads);
            if (n_v_heads <= 0) {
                n_v_heads = std::max(
                    1,
                    std::min(
                        std::max(1, lc.ssm_alpha_dim),
                        std::max(1, lc.ssm_beta_dim)
                    )
                );
            }
            if ((d_inner % std::max(1, n_v_heads)) != 0) {
                n_v_heads = std::max(1, d_inner / d_state);
            }
            while (n_v_heads > 1 && (d_inner % n_v_heads) != 0) {
                --n_v_heads;
            }
            const int head_v_dim = std::max(1, d_inner / std::max(1, n_v_heads));
            const int qk_dim = std::max(1, n_groups * d_state);
            const int conv_channels = d_inner + 2 * qk_dim;
            const int heads_per_group = std::max(1, n_v_heads / std::max(1, n_groups));
            const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_v_dim));

            float* qkv = g->scratch_aux0.data();
            float* z = g->scratch_aux1.data();
            float* beta = g->scratch_gate.data();
            float* alpha = g->scratch_up.data();
            const int beta_rows = std::max(0, std::min(n_v_heads, lc.ssm_beta_dim));
            const int alpha_rows = std::max(0, std::min(n_v_heads, lc.ssm_alpha_dim));
            {
                const double started = perf_now_seconds();
                std::fill(beta, beta + n_v_heads, 0.0f);
                std::fill(alpha, alpha + n_v_heads, 0.0f);
                const bool fused_quad = fused_quad_matvec_dispatch(
                    normed,
                    lc.ssm_in_quant, lc.ssm_in_qtype, qkv, lc.ssm_in_dim,
                    lc.attn_gate_quant, lc.attn_gate_qtype, z, d_inner,
                    lc.ssm_beta_quant, lc.ssm_beta_qtype, beta, beta_rows,
                    lc.ssm_alpha_quant, lc.ssm_alpha_qtype, alpha, alpha_rows,
                    dim
                );
                if (!fused_quad) {
                    matvec_quant(lc.ssm_in_quant, lc.ssm_in_qtype, normed, qkv, dim, lc.ssm_in_dim);
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.ssm_in_qtype,
                        lc.ssm_in_dim,
                        dim
                    );
                    matvec_quant(lc.attn_gate_quant, lc.attn_gate_qtype, normed, z, dim, d_inner);
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.attn_gate_qtype,
                        d_inner,
                        dim
                    );
                    if (beta_rows > 0) {
                        matvec_quant(lc.ssm_beta_quant, lc.ssm_beta_qtype, normed, beta, dim, beta_rows);
                        account_matvec_quant(
                            &g->perf_stats.ssm_projection_bytes,
                            &g->perf_stats.ssm_projection_flops,
                            lc.ssm_beta_qtype,
                            beta_rows,
                            dim
                        );
                    }
                    if (alpha_rows > 0) {
                        matvec_quant(lc.ssm_alpha_quant, lc.ssm_alpha_qtype, normed, alpha, dim, alpha_rows);
                        account_matvec_quant(
                            &g->perf_stats.ssm_projection_bytes,
                            &g->perf_stats.ssm_projection_flops,
                            lc.ssm_alpha_qtype,
                            alpha_rows,
                            dim
                        );
                    }
                } else {
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.ssm_in_qtype,
                        lc.ssm_in_dim,
                        dim
                    );
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.attn_gate_qtype,
                        d_inner,
                        dim
                    );
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.ssm_beta_qtype,
                        beta_rows,
                        dim
                    );
                    account_matvec_quant(
                        &g->perf_stats.ssm_projection_bytes,
                        &g->perf_stats.ssm_projection_flops,
                        lc.ssm_alpha_qtype,
                        alpha_rows,
                        dim
                    );
                }
                if (lc.ssm_in_dim < conv_channels) {
                    std::fill(qkv + lc.ssm_in_dim, qkv + conv_channels, 0.0f);
                }
                for (int i = 0; i < n_v_heads; ++i) {
                    beta[i] = fast_sigmoid_scalar(beta[i]);
                    alpha[i] = fast_softplus_scalar(alpha[i] + lc.ssm_dt[i]) * lc.ssm_a[i];
                }
                ssm_projection_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_projection_seconds += ssm_projection_elapsed;
            }

            float* conv_out = g->scratch_ffn.data();
            {
                const double started = perf_now_seconds();
                causal_conv1d_step(
                    g->conv_state_storage[static_cast<std::size_t>(l)],
                    g->conv_state_head_storage[static_cast<std::size_t>(l)],
                    lc.ssm_conv,
                    lc.ssm_conv_rows,
                    conv_channels,
                    qkv,
                    lc.ssm_conv_bias,
                    conv_out,
                    true
                );
                const int group_threads = get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(group_threads) schedule(static)
#endif
                for (int grp = 0; grp < n_groups; ++grp) {
                    maybe_bind_worker_thread(true);
                    float* q_g = conv_out + static_cast<std::size_t>(grp) * d_state;
                    float* k_g = conv_out + static_cast<std::size_t>(qk_dim + grp * d_state);
                    normalize_pair_inplace(q_g, k_g, d_state);
                }
                ssm_conv_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_conv_seconds += ssm_conv_elapsed;
            }

            auto& layer_state = g->ssm_state_storage[static_cast<std::size_t>(l)];
            if (static_cast<int>(layer_state.size()) != n_v_heads * head_v_dim * d_state) {
                layer_state.assign(
                    static_cast<std::size_t>(n_v_heads * head_v_dim * d_state),
                    0.0f
                );
            }

            float* out_heads = g->scratch_aux2.data();
            const int head_threads = get_num_threads();
            {
                const double started = perf_now_seconds();
#ifdef _OPENMP
#pragma omp parallel for num_threads(head_threads) schedule(static)
#endif
                for (int h = 0; h < n_v_heads; ++h) {
                    maybe_bind_worker_thread(true);
                    const int group = std::min(n_groups - 1, h / heads_per_group);
                    const float* q_h = conv_out + static_cast<std::size_t>(group) * d_state;
                    const float* k_h = conv_out + static_cast<std::size_t>(qk_dim + group * d_state);
                    const float* v_h = conv_out + static_cast<std::size_t>(2 * qk_dim + h * head_v_dim);
                    float* state_h = layer_state.data() + static_cast<std::size_t>(h) * head_v_dim * d_state;
                    const float gate_scale = anvil_fast_math::fast_exp_scalar(
                        std::max(-60.0f, std::min(60.0f, alpha[h]))
                    );
                    prefetch_read(q_h);
                    prefetch_read(k_h);
                    for (int d = 0; d < head_v_dim; ++d) {
                        prefetch_write(state_h + static_cast<std::size_t>(std::min(d + 1, head_v_dim - 1)) * d_state);
                        float* row = state_h + static_cast<std::size_t>(d) * d_state;
                        float sk = 0.0f;
#ifdef __AVX2__
                        const __m256 gate_v = _mm256_set1_ps(gate_scale);
                        __m256 sk_acc = _mm256_setzero_ps();
                        int j = 0;
                        for (; j + 8 <= d_state; j += 8) {
                            __m256 row_v = _mm256_loadu_ps(row + j);
                            __m256 k_v = _mm256_loadu_ps(k_h + j);
                            row_v = _mm256_mul_ps(row_v, gate_v);
                            _mm256_storeu_ps(row + j, row_v);
                            sk_acc = _mm256_fmadd_ps(row_v, k_v, sk_acc);
                        }
                        sk += hsum256_ps(sk_acc);
                        for (; j < d_state; ++j) {
                            row[j] *= gate_scale;
                            sk += row[j] * k_h[j];
                        }
#else
                        for (int j = 0; j < d_state; ++j) {
                            row[j] *= gate_scale;
                            sk += row[j] * k_h[j];
                        }
#endif
                        const float delta = (v_h[d] - sk) * beta[h];
                        float out = 0.0f;
#ifdef __AVX2__
                        const __m256 delta_v = _mm256_set1_ps(delta);
                        __m256 out_acc = _mm256_setzero_ps();
                        int k = 0;
                        for (; k + 8 <= d_state; k += 8) {
                            __m256 row_v = _mm256_loadu_ps(row + k);
                            __m256 k_v = _mm256_loadu_ps(k_h + k);
                            __m256 q_v = _mm256_loadu_ps(q_h + k);
                            row_v = _mm256_fmadd_ps(delta_v, k_v, row_v);
                            _mm256_storeu_ps(row + k, row_v);
                            out_acc = _mm256_fmadd_ps(row_v, q_v, out_acc);
                        }
                        out += hsum256_ps(out_acc);
                        for (; k < d_state; ++k) {
                            row[k] += delta * k_h[k];
                            out += row[k] * q_h[k];
                        }
#else
                        for (int j = 0; j < d_state; ++j) {
                            row[j] += delta * k_h[j];
                            out += row[j] * q_h[j];
                        }
#endif
                        out_heads[static_cast<std::size_t>(h) * head_v_dim + d] = out * q_scale;
                    }
                }
                ssm_recurrent_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_recurrent_seconds += ssm_recurrent_elapsed;
            }
            {
                const double started = perf_now_seconds();
                rmsnorm_rows_inplace(out_heads, lc.ssm_norm, n_v_heads, head_v_dim, eps);
                mul_silu_gate_inplace(out_heads, z, d_inner);
                matvec_quant(lc.ssm_out_quant, lc.ssm_out_qtype, out_heads, g->scratch_proj.data(), d_inner, dim);
                account_matvec_quant(
                    &g->perf_stats.ssm_output_bytes,
                    &g->perf_stats.ssm_output_flops,
                    lc.ssm_out_qtype,
                    dim,
                    d_inner
                );
                add_residual(g->scratch_proj.data());
                ssm_output_elapsed = perf_now_seconds() - started;
                g->perf_stats.ssm_output_seconds += ssm_output_elapsed;
            }
            run_simple_ffn(lc);
            sanitize_tracked(state, dim);
            g->perf_stats.ssm_seconds +=
                ssm_projection_elapsed
                + ssm_conv_elapsed
                + ssm_recurrent_elapsed
                + ssm_output_elapsed;
            continue;
        }

        run_attention(lc, l, lc.layer_kind == LAYER_KIND_QWEN_FULL_ATTN);
        run_simple_ffn(lc);
        sanitize_tracked(state, dim);
    }

    if (capture_last_hidden) {
        std::memcpy(
            g->scratch_last_hidden.data(),
            state,
            static_cast<std::size_t>(dim) * sizeof(float));
        g->last_hidden_valid = true;
    }

    if (emit_logits) {
        // Final RMSNorm only affects logits, so skip it for intermediate batched-prefill tokens.
        if (g->final_norm != nullptr) {
            const double started = perf_now_seconds();
            rmsnorm_inplace(state, g->final_norm, dim, eps);
            g->perf_stats.final_norm_seconds += perf_now_seconds() - started;
            sanitize_tracked(state, dim);
        }

        // Logit scale
        float ls = g->logit_scale;

        // LM head projection: logits = lm_head @ state
        if (g->lm_head != nullptr) {
            const double started = perf_now_seconds();
            if (g->lm_head_transposed) {
                int n_threads = get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
                for (int v = 0; v < g->vocab_size; ++v) {
                    float acc = 0.0f;
                    const float* col = g->lm_head + static_cast<std::size_t>(v);
#ifdef __AVX2__
                    __m256 vacc = _mm256_setzero_ps();
                    int d = 0;
                    for (; d + 8 <= dim; d += 8) {
                        __m256 xv = _mm256_loadu_ps(state + d);
                        __m256 wv = _mm256_set_ps(
                            col[static_cast<std::size_t>(d + 7) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 6) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 5) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 4) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 3) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 2) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 1) * g->vocab_size],
                            col[static_cast<std::size_t>(d + 0) * g->vocab_size]
                        );
                        vacc = _mm256_fmadd_ps(xv, wv, vacc);
                    }
                    acc = hsum256_ps(vacc);
                    for (; d < dim; ++d) {
                        acc += state[d] * col[static_cast<std::size_t>(d) * g->vocab_size];
                    }
#else
                    for (int d = 0; d < dim; ++d) {
                        acc += state[d] * col[static_cast<std::size_t>(d) * g->vocab_size];
                    }
#endif
                    logits_out[v] = (ls != 0.0f) ? acc * (1.0f / ls) : acc;
                }
                account_matvec_f32(
                    &g->perf_stats.lm_head_bytes,
                    &g->perf_stats.lm_head_flops,
                    g->vocab_size,
                    dim
                );
            } else {
                matvec_f32(g->lm_head, state, logits_out, g->vocab_size, dim);
                account_matvec_f32(
                    &g->perf_stats.lm_head_bytes,
                    &g->perf_stats.lm_head_flops,
                    g->vocab_size,
                    dim
                );
                scale_and_sanitize_inplace(
                    logits_out,
                    g->vocab_size,
                    (ls != 0.0f) ? (1.0f / ls) : 1.0f
                );
            }
            g->perf_stats.lm_head_seconds += perf_now_seconds() - started;
        } else if (g->lm_head_quant != nullptr) {
            const double started = perf_now_seconds();
            if (uses_packed_r4_layout(g->lm_head_qtype)) {
                g->perf_stats.packed_lm_head_calls += 1;
            }
            matvec_quant(g->lm_head_quant, g->lm_head_qtype, state, logits_out, dim, g->vocab_size);
            account_matvec_quant(
                &g->perf_stats.lm_head_bytes,
                &g->perf_stats.lm_head_flops,
                g->lm_head_qtype,
                g->vocab_size,
                dim
            );
            scale_and_sanitize_inplace(
                logits_out,
                g->vocab_size,
                (ls != 0.0f) ? (1.0f / ls) : 1.0f
            );
            g->perf_stats.lm_head_seconds += perf_now_seconds() - started;
        } else {
            std::memset(logits_out, 0, static_cast<std::size_t>(g->vocab_size) * sizeof(float));
        }
        if (g->lm_head != nullptr && g->lm_head_transposed) {
            sanitize_tracked(logits_out, g->vocab_size);
        }
    }

    if (advance_position) {
        g->current_pos = pos + 1;
    }
    return 1;
}

// Execute single-token decode: embedding → all layers → logits
int graph_forward_token(
    ModelGraphHandle* g,
    const float* hidden_in,
    int hidden_len,
    float* logits_out,
    int logits_len,
    int pos
) {
    if (g == nullptr || hidden_in == nullptr || logits_out == nullptr) return 0;
    if (hidden_len != g->dim || logits_len != g->vocab_size) return 0;
    if (pos < 0 || pos >= g->max_seq) return 0;
    g->perf_stats.forward_token_calls += 1;
    return graph_forward_token_impl(
        g,
        hidden_in,
        hidden_len,
        logits_out,
        logits_len,
        pos,
        true,
        0,
        g->n_layers,
        true,
        true,
        true
    );
}

int graph_forward_token_id(
    ModelGraphHandle* g,
    int token_id,
    float* logits_out,
    int logits_len,
    int pos,
    GraphDriftSnapshot* drift_out
) {
    if (g == nullptr || logits_out == nullptr) return 0;
    if (token_id < 0 || token_id >= g->vocab_size) return 0;
    const double decode_started = perf_now_seconds();
    g->perf_stats.forward_token_id_calls += 1;
    const float* embedding = nullptr;
    if (!lookup_token_embedding(g, token_id, &embedding) || embedding == nullptr) {
        if (drift_out != nullptr) {
            *drift_out = g->last_drift_snapshot;
        }
        return 0;
    }
    g->perf_stats.forward_token_calls += 1;
    const int ok = graph_forward_token_impl(
        g,
        embedding,
        g->dim,
        logits_out,
        logits_len,
        pos,
        true,
        0,
        g->n_layers,
        true,
        true,
        true
    );
    if (ok == 1) {
        graph_drift_update_after_decode(
            g,
            pos + 1,
            1,
            perf_now_seconds() - decode_started,
            drift_out
        );
    } else if (drift_out != nullptr) {
        *drift_out = g->last_drift_snapshot;
    }
    return ok;
}

int graph_forward_token_ids(
    ModelGraphHandle* g,
    const int* token_ids,
    int token_count,
    float* logits_out,
    int logits_len,
    int start_pos,
    GraphDriftSnapshot* drift_out
) {
    if (g == nullptr || token_ids == nullptr || logits_out == nullptr) return 0;
    if (token_count <= 0) return 0;
    if (start_pos < 0) return 0;
    const double decode_started = perf_now_seconds();
    g->perf_stats.forward_token_ids_calls += 1;
    g->perf_stats.forward_token_ids_token_count += token_count;
    for (int i = 0; i < token_count; ++i) {
        const float* embedding = nullptr;
        if (!lookup_token_embedding(g, token_ids[i], &embedding) || embedding == nullptr) {
            if (drift_out != nullptr) {
                *drift_out = g->last_drift_snapshot;
            }
            return 0;
        }
        g->perf_stats.forward_token_calls += 1;
        const bool emit_logits = (i + 1) == token_count;
        if (
            graph_forward_token_impl(
                g,
                embedding,
                g->dim,
                logits_out,
                logits_len,
                start_pos + i,
                emit_logits,
                0,
                g->n_layers,
                true,
                true,
                true
            ) != 1
        ) {
            if (drift_out != nullptr) {
                *drift_out = g->last_drift_snapshot;
            }
            return 0;
        }
    }
    graph_drift_update_after_decode(
        g,
        start_pos + token_count,
        token_count,
        perf_now_seconds() - decode_started,
        drift_out
    );
    return 1;
}

GraphExecutionCheckpoint* graph_create_execution_checkpoint(
    const ModelGraphHandle* g
) {
    if (g == nullptr) {
        return nullptr;
    }
    auto* checkpoint = new (std::nothrow) GraphExecutionCheckpoint();
    if (checkpoint == nullptr) {
        return nullptr;
    }
    checkpoint->kv_snapshot = g->kv_store.snapshot();
    checkpoint->ssm_state_storage = g->ssm_state_storage;
    checkpoint->conv_state_storage = g->conv_state_storage;
    checkpoint->conv_state_head_storage = g->conv_state_head_storage;
    checkpoint->last_drift_snapshot = g->last_drift_snapshot;
    checkpoint->drift_block_energy_ref = g->drift_block_energy_ref;
    checkpoint->drift_block_phase_ref = g->drift_block_phase_ref;
    checkpoint->drift_block_latest = g->drift_block_latest;
    checkpoint->drift_block_decay_gain = g->drift_block_decay_gain;
    checkpoint->drift_block_pruned = g->drift_block_pruned;
    checkpoint->drift_block_prune_streak = g->drift_block_prune_streak;
    checkpoint->drift_overhead_window = g->drift_overhead_window;
    checkpoint->drift_state_block_size = g->drift_state_block_size;
    checkpoint->drift_overhead_sum = g->drift_overhead_sum;
    checkpoint->drift_overhead_index = g->drift_overhead_index;
    checkpoint->drift_overhead_count = g->drift_overhead_count;
    checkpoint->drift_overhead_below_target_streak = g->drift_overhead_below_target_streak;
    checkpoint->drift_auto_downgrade_events = g->drift_auto_downgrade_events;
    checkpoint->drift_auto_upgrade_events = g->drift_auto_upgrade_events;
    checkpoint->drift_force_conservative = g->drift_force_conservative;
    checkpoint->drift_tokens_since_update = g->drift_tokens_since_update;
    checkpoint->drift_tokens_since_prune = g->drift_tokens_since_prune;
    checkpoint->drift_stabilizer_seconds_total = g->drift_stabilizer_seconds_total;
    checkpoint->drift_stabilizer_calls_total = g->drift_stabilizer_calls_total;
    checkpoint->current_pos = g->current_pos;
    checkpoint->last_hidden_valid = g->last_hidden_valid;
    checkpoint->last_hidden.assign(
        g->scratch_last_hidden.data(),
        g->scratch_last_hidden.data() + g->dim
    );
    return checkpoint;
}

int graph_restore_execution_checkpoint(
    ModelGraphHandle* g,
    const GraphExecutionCheckpoint* checkpoint
) {
    if (g == nullptr || checkpoint == nullptr) {
        return 0;
    }
    g->kv_store.restore(checkpoint->kv_snapshot);
    g->ssm_state_storage = checkpoint->ssm_state_storage;
    g->conv_state_storage = checkpoint->conv_state_storage;
    g->conv_state_head_storage = checkpoint->conv_state_head_storage;
    g->last_drift_snapshot = checkpoint->last_drift_snapshot;
    g->drift_block_energy_ref = checkpoint->drift_block_energy_ref;
    g->drift_block_phase_ref = checkpoint->drift_block_phase_ref;
    g->drift_block_latest = checkpoint->drift_block_latest;
    g->drift_block_decay_gain = checkpoint->drift_block_decay_gain;
    g->drift_block_pruned = checkpoint->drift_block_pruned;
    g->drift_block_prune_streak = checkpoint->drift_block_prune_streak;
    g->drift_overhead_window = checkpoint->drift_overhead_window;
    g->drift_state_block_size = checkpoint->drift_state_block_size;
    g->drift_overhead_sum = checkpoint->drift_overhead_sum;
    g->drift_overhead_index = checkpoint->drift_overhead_index;
    g->drift_overhead_count = checkpoint->drift_overhead_count;
    g->drift_overhead_below_target_streak = checkpoint->drift_overhead_below_target_streak;
    g->drift_auto_downgrade_events = checkpoint->drift_auto_downgrade_events;
    g->drift_auto_upgrade_events = checkpoint->drift_auto_upgrade_events;
    g->drift_force_conservative = checkpoint->drift_force_conservative;
    g->drift_tokens_since_update = checkpoint->drift_tokens_since_update;
    g->drift_tokens_since_prune = checkpoint->drift_tokens_since_prune;
    g->drift_stabilizer_seconds_total = checkpoint->drift_stabilizer_seconds_total;
    g->drift_stabilizer_calls_total = checkpoint->drift_stabilizer_calls_total;
    g->current_pos = checkpoint->current_pos;
    g->last_hidden_valid = checkpoint->last_hidden_valid;
    if (
        static_cast<int>(checkpoint->last_hidden.size()) == g->dim
        && g->scratch_last_hidden.size() >= checkpoint->last_hidden.size()
    ) {
        std::memcpy(
            g->scratch_last_hidden.data(),
            checkpoint->last_hidden.data(),
            checkpoint->last_hidden.size() * sizeof(float)
        );
    }
    return 1;
}

void graph_destroy_execution_checkpoint(GraphExecutionCheckpoint* checkpoint) {
    delete checkpoint;
}

int graph_forward_token_id_to_exit(
    ModelGraphHandle* g,
    int token_id,
    int exit_layer,
    float* hidden_out,
    int hidden_len,
    int pos
) {
    if (g == nullptr || hidden_out == nullptr) return 0;
    if (hidden_len != g->dim) return 0;
    if (g->n_layers < 2) return 0;
    if (token_id < 0 || token_id >= g->vocab_size) return 0;
    const float* embedding = nullptr;
    if (!lookup_token_embedding(g, token_id, &embedding) || embedding == nullptr) {
        return 0;
    }
    const int clamped_exit = std::max(1, std::min(exit_layer, g->n_layers - 1));
    if (
        graph_forward_token_impl(
            g,
            embedding,
            g->dim,
            g->scratch_aux0.data(),
            g->vocab_size,
            pos,
            false,
            0,
            clamped_exit,
            true,
            false,
            false
        ) != 1
    ) {
        return 0;
    }
    std::memcpy(
        hidden_out,
        g->scratch_state.data(),
        static_cast<std::size_t>(g->dim) * sizeof(float)
    );
    return 1;
}

int graph_continue_from_hidden(
    ModelGraphHandle* g,
    const float* hidden_in,
    int hidden_len,
    int start_layer,
    float* logits_out,
    int logits_len,
    int pos
) {
    if (g == nullptr || hidden_in == nullptr || logits_out == nullptr) return 0;
    if (hidden_len != g->dim || logits_len != g->vocab_size) return 0;
    const int clamped_start = std::max(0, std::min(start_layer, g->n_layers));
    return graph_forward_token_impl(
        g,
        hidden_in,
        hidden_len,
        logits_out,
        logits_len,
        pos,
        true,
        clamped_start,
        g->n_layers,
        false,
        true,
        true
    );
}

// Execute a single attention+FFN layer on a hidden state vector.
// hidden_io is both input and output (updated in-place).
// Returns 1 on success.  The caller is responsible for tracking position.
int graph_forward_layer(
    ModelGraphHandle* g,
    float* hidden_io,
    int hidden_len,
    int layer_idx,
    int pos
) {
    if (g == nullptr || hidden_io == nullptr) return 0;
    if (hidden_len != g->dim) return 0;
    if (layer_idx < 0 || layer_idx >= g->n_layers) return 0;
    if (pos < 0 || pos >= g->max_seq) return 0;

    int dim = g->dim;
    int n_heads = g->n_heads;
    int n_kv_heads = g->n_kv_heads;
    int head_dim = g->head_dim;
    float eps = g->rms_eps;
    if (g->drift_config.enabled != 0) {
        graph_drift_ensure_state(g);
    }

    float* state = hidden_io;
    const auto& lc = g->layers[static_cast<std::size_t>(layer_idx)];
    float* normed = g->scratch_normed.data();

    float scale = g->attention_scale > 0.0f
        ? g->attention_scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));

    // === Attention branch ===
    if (lc.is_attention && lc.has_wq() && lc.has_wk() &&
        lc.has_wv() && lc.has_wo() && lc.attn_norm != nullptr) {

        rmsnorm_copy(state, lc.attn_norm, normed, dim, eps);
        sanitize_tensor_inplace(normed, dim);

        float* q = g->scratch_q.data();
        float* k = g->scratch_k.data();
        float* v = g->scratch_v.data();

        matvec_dispatch(lc.wq, lc.wq_quant, lc.wq_qtype, normed, q, lc.q_out_dim, dim);
        matvec_dispatch(lc.wk, lc.wk_quant, lc.wk_qtype, normed, k, lc.kv_out_dim, dim);
        matvec_dispatch(lc.wv, lc.wv_quant, lc.wv_qtype, normed, v, lc.kv_out_dim, dim);
        account_matvec_dispatch(
            &g->perf_stats.attention_proj_bytes,
            &g->perf_stats.attention_proj_flops,
            lc.wq,
            lc.wq_quant,
            lc.wq_qtype,
            lc.q_out_dim,
            dim
        );
        account_matvec_dispatch(
            &g->perf_stats.attention_proj_bytes,
            &g->perf_stats.attention_proj_flops,
            lc.wk,
            lc.wk_quant,
            lc.wk_qtype,
            lc.kv_out_dim,
            dim
        );
        account_matvec_dispatch(
            &g->perf_stats.attention_proj_bytes,
            &g->perf_stats.attention_proj_flops,
            lc.wv,
            lc.wv_quant,
            lc.wv_qtype,
            lc.kv_out_dim,
            dim
        );
        sanitize_tensor_inplace(q, lc.q_out_dim);
        sanitize_tensor_inplace(k, lc.kv_out_dim);
        sanitize_tensor_inplace(v, lc.kv_out_dim);

        const int layer_n_heads = (lc.q_out_dim > 0 && lc.q_out_dim % head_dim == 0)
            ? std::max(1, lc.q_out_dim / head_dim)
            : n_heads;
        const int layer_n_kv_heads = (lc.kv_out_dim > 0 && lc.kv_out_dim % head_dim == 0)
            ? std::max(1, lc.kv_out_dim / head_dim)
            : n_kv_heads;
        const int layer_q_dim = layer_n_heads * head_dim;
        apply_rope(q, k, layer_n_heads, layer_n_kv_heads, head_dim, pos, g->rope_theta);

        g->kv_store.write(layer_idx, pos, k, v, layer_n_kv_heads);

        int kv_len = pos + 1;

        float* attn_out = g->scratch_attn_out.data();
        const DriftDecodeContext drift_ctx = make_drift_decode_context(g);
        fused_gqa_attention_decode_paged(
            q,
            g->kv_store,
            layer_idx,
            attn_out,
            layer_n_heads,
            layer_n_kv_heads,
            kv_len,
            head_dim,
            scale,
            g->attention_tile_size,
            drift_ctx.block_decay_gain,
            drift_ctx.block_pruned,
            drift_ctx.block_count,
            drift_ctx.block_size,
            drift_ctx.preserve_head_tokens,
            drift_ctx.preserve_recent_tokens
        );
        sanitize_tensor_inplace(attn_out, layer_q_dim);

        float* proj = g->scratch_proj.data();
        matvec_dispatch(lc.wo, lc.wo_quant, lc.wo_qtype, attn_out, proj, dim, layer_q_dim);
        account_matvec_dispatch(
            &g->perf_stats.attention_out_proj_bytes,
            &g->perf_stats.attention_out_proj_flops,
            lc.wo,
            lc.wo_quant,
            lc.wo_qtype,
            dim,
            layer_q_dim
        );
        sanitize_tensor_inplace(proj, dim);

        float rs = g->residual_scale;
        if (rs != 0.0f) {
            for (int i = 0; i < dim; ++i)
                state[i] += proj[i] * rs;
        } else {
            for (int i = 0; i < dim; ++i)
                state[i] += proj[i];
        }
    }

    // === FFN branch ===
    if (lc.has_gate() && lc.has_up() && lc.has_down() && lc.ffn_norm != nullptr) {

        rmsnorm_copy(state, lc.ffn_norm, normed, dim, eps);
        sanitize_tensor_inplace(normed, dim);

        float* gate_out = g->scratch_gate.data();
        float* up_out = g->scratch_up.data();
        float* ffn_hidden = g->scratch_ffn.data();

        matvec_dispatch(lc.w_gate, lc.wgate_quant, lc.wgate_qtype, normed, gate_out, lc.ffn_dim, dim);
        matvec_dispatch(lc.w_up, lc.wup_quant, lc.wup_qtype, normed, up_out, lc.ffn_dim, dim);
        account_matvec_dispatch(
            &g->perf_stats.ffn_gate_up_bytes,
            &g->perf_stats.ffn_gate_up_flops,
            lc.w_gate,
            lc.wgate_quant,
            lc.wgate_qtype,
            lc.ffn_dim,
            dim
        );
        account_matvec_dispatch(
            &g->perf_stats.ffn_gate_up_bytes,
            &g->perf_stats.ffn_gate_up_flops,
            lc.w_up,
            lc.wup_quant,
            lc.wup_qtype,
            lc.ffn_dim,
            dim
        );
        sanitize_tensor_inplace(gate_out, lc.ffn_dim);
        sanitize_tensor_inplace(up_out, lc.ffn_dim);
        swiglu_f32(gate_out, up_out, ffn_hidden, lc.ffn_dim);
        sanitize_tensor_inplace(ffn_hidden, lc.ffn_dim);

        matvec_dispatch(lc.w_down, lc.wdown_quant, lc.wdown_qtype, ffn_hidden, g->scratch_proj.data(), dim, lc.ffn_dim);
        account_matvec_dispatch(
            &g->perf_stats.ffn_down_bytes,
            &g->perf_stats.ffn_down_flops,
            lc.w_down,
            lc.wdown_quant,
            lc.wdown_qtype,
            dim,
            lc.ffn_dim
        );
        sanitize_tensor_inplace(g->scratch_proj.data(), dim);

        float rs = g->residual_scale;
        if (rs != 0.0f) {
            for (int i = 0; i < dim; ++i)
                state[i] += g->scratch_proj[i] * rs;
        } else {
            for (int i = 0; i < dim; ++i)
                state[i] += g->scratch_proj[i];
        }
    }

    sanitize_tensor_inplace(state, dim);
    return 1;
}

// Execute final RMSNorm + LM head projection (no layer execution).
// hidden_in: [dim], logits_out: [vocab_size].
int graph_forward_head(
    ModelGraphHandle* g,
    const float* hidden_in,
    int hidden_len,
    float* logits_out,
    int logits_len
) {
    if (g == nullptr || hidden_in == nullptr || logits_out == nullptr) return 0;
    if (hidden_len != g->dim || logits_len != g->vocab_size) return 0;

    int dim = g->dim;
    float eps = g->rms_eps;

    float* state = g->scratch_state.data();
    std::memcpy(state, hidden_in, static_cast<std::size_t>(dim) * sizeof(float));
    std::memcpy(
        g->scratch_last_hidden.data(),
        hidden_in,
        static_cast<std::size_t>(dim) * sizeof(float));
    g->last_hidden_valid = true;

    if (g->final_norm != nullptr) {
        const double started = perf_now_seconds();
        rmsnorm_inplace(state, g->final_norm, dim, eps);
        g->perf_stats.final_norm_seconds += perf_now_seconds() - started;
    }

    float ls = g->logit_scale;

    if (g->lm_head != nullptr) {
        const double started = perf_now_seconds();
        if (g->lm_head_transposed) {
            int n_threads = get_num_threads();
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
            for (int v = 0; v < g->vocab_size; ++v) {
                float acc = 0.0f;
                const float* col = g->lm_head + static_cast<std::size_t>(v);
                for (int d = 0; d < dim; ++d)
                    acc += state[d] * col[static_cast<std::size_t>(d) * g->vocab_size];
                logits_out[v] = (ls != 0.0f) ? acc * (1.0f / ls) : acc;
            }
            account_matvec_f32(
                &g->perf_stats.lm_head_bytes,
                &g->perf_stats.lm_head_flops,
                g->vocab_size,
                dim
            );
        } else {
            matvec_f32(g->lm_head, state, logits_out, g->vocab_size, dim);
            account_matvec_f32(
                &g->perf_stats.lm_head_bytes,
                &g->perf_stats.lm_head_flops,
                g->vocab_size,
                dim
            );
            scale_and_sanitize_inplace(
                logits_out,
                g->vocab_size,
                (ls != 0.0f) ? (1.0f / ls) : 1.0f
            );
        }
        g->perf_stats.lm_head_seconds += perf_now_seconds() - started;
    } else if (g->lm_head_quant != nullptr) {
        const double started = perf_now_seconds();
        if (uses_packed_r4_layout(g->lm_head_qtype)) {
            g->perf_stats.packed_lm_head_calls += 1;
        }
        matvec_quant(g->lm_head_quant, g->lm_head_qtype, state, logits_out, dim, g->vocab_size);
        account_matvec_quant(
            &g->perf_stats.lm_head_bytes,
            &g->perf_stats.lm_head_flops,
            g->lm_head_qtype,
            g->vocab_size,
            dim
        );
        scale_and_sanitize_inplace(
            logits_out,
            g->vocab_size,
            (ls != 0.0f) ? (1.0f / ls) : 1.0f
        );
        g->perf_stats.lm_head_seconds += perf_now_seconds() - started;
    } else {
        std::memset(logits_out, 0, static_cast<std::size_t>(g->vocab_size) * sizeof(float));
    }
    return 1;
}

int graph_copy_last_hidden(
    const ModelGraphHandle* g,
    float* hidden_out,
    int hidden_len
) {
    if (g == nullptr || hidden_out == nullptr) return 0;
    if (!g->last_hidden_valid || hidden_len != g->dim) return 0;
    std::memcpy(
        hidden_out,
        g->scratch_last_hidden.data(),
        static_cast<std::size_t>(g->dim) * sizeof(float));
    return 1;
}

// Reset KV cache and position
int graph_reset(ModelGraphHandle* g) {
    if (g == nullptr) return 0;
    g->kv_store.reset();
    for (auto& state : g->ssm_state_storage) {
        std::fill(state.begin(), state.end(), 0.0f);
    }
    for (auto& buf : g->conv_state_storage) {
        std::fill(buf.begin(), buf.end(), 0.0f);
    }
    for (auto& head : g->conv_state_head_storage) {
        head = 0;
    }
    g->current_pos = 0;
    g->last_hidden_valid = false;
    g->perf_stats.reset();
    graph_drift_reset_runtime(g);
    return 1;
}

// Reset perf stats only (preserve KV/SSM state and position).
int graph_reset_perf_stats(ModelGraphHandle* g) {
    if (g == nullptr) return 0;
    g->perf_stats.reset();
    return 1;
}

// Get current position
int graph_get_position(const ModelGraphHandle* g) {
    return g ? g->current_pos : 0;
}

int graph_get_perf_stats(const ModelGraphHandle* g, GraphPerfStatsSnapshot* out) {
    if (g == nullptr || out == nullptr) {
        return 0;
    }
    out->embedding_lookup_seconds = g->perf_stats.embedding_lookup_seconds;
    out->attention_proj_seconds = g->perf_stats.attention_proj_seconds;
    out->attention_rope_kv_seconds = g->perf_stats.attention_rope_kv_seconds;
    out->attention_decode_seconds = g->perf_stats.attention_decode_seconds;
    out->attention_out_proj_seconds = g->perf_stats.attention_out_proj_seconds;
    out->ffn_norm_seconds = g->perf_stats.ffn_norm_seconds;
    out->ffn_gate_up_seconds = g->perf_stats.ffn_gate_up_seconds;
    out->ffn_down_seconds = g->perf_stats.ffn_down_seconds;
    out->ssm_projection_seconds = g->perf_stats.ssm_projection_seconds;
    out->ssm_conv_seconds = g->perf_stats.ssm_conv_seconds;
    out->ssm_recurrent_seconds = g->perf_stats.ssm_recurrent_seconds;
    out->ssm_output_seconds = g->perf_stats.ssm_output_seconds;
    out->ssm_seconds = g->perf_stats.ssm_seconds;
    out->moe_seconds = g->perf_stats.moe_seconds;
    out->final_norm_seconds = g->perf_stats.final_norm_seconds;
    out->lm_head_seconds = g->perf_stats.lm_head_seconds;
    out->sanitize_seconds = g->perf_stats.sanitize_seconds;
    out->forward_token_calls = g->perf_stats.forward_token_calls;
    out->forward_token_id_calls = g->perf_stats.forward_token_id_calls;
    out->forward_token_ids_calls = g->perf_stats.forward_token_ids_calls;
    out->forward_token_ids_token_count = g->perf_stats.forward_token_ids_token_count;
    out->attention_calls = g->perf_stats.attention_calls;
    out->ffn_calls = g->perf_stats.ffn_calls;
    out->ssm_calls = g->perf_stats.ssm_calls;
    out->moe_calls = g->perf_stats.moe_calls;
    out->packed_lm_head_calls = g->perf_stats.packed_lm_head_calls;
    out->attention_proj_bytes = g->perf_stats.attention_proj_bytes;
    out->attention_proj_flops = g->perf_stats.attention_proj_flops;
    out->attention_out_proj_bytes = g->perf_stats.attention_out_proj_bytes;
    out->attention_out_proj_flops = g->perf_stats.attention_out_proj_flops;
    out->ffn_gate_up_bytes = g->perf_stats.ffn_gate_up_bytes;
    out->ffn_gate_up_flops = g->perf_stats.ffn_gate_up_flops;
    out->ffn_down_bytes = g->perf_stats.ffn_down_bytes;
    out->ffn_down_flops = g->perf_stats.ffn_down_flops;
    out->ssm_projection_bytes = g->perf_stats.ssm_projection_bytes;
    out->ssm_projection_flops = g->perf_stats.ssm_projection_flops;
    out->ssm_output_bytes = g->perf_stats.ssm_output_bytes;
    out->ssm_output_flops = g->perf_stats.ssm_output_flops;
    out->moe_bytes = g->perf_stats.moe_bytes;
    out->moe_flops = g->perf_stats.moe_flops;
    out->lm_head_bytes = g->perf_stats.lm_head_bytes;
    out->lm_head_flops = g->perf_stats.lm_head_flops;
    return 1;
}

// Legacy compatibility: forward using just hidden state (no per-layer weights)
int graph_forward(
    ModelGraphHandle* g,
    const float* hidden_in,
    int hidden_len,
    float* logits_out,
    int logits_len
) {
    if (g == nullptr || hidden_in == nullptr || logits_out == nullptr) return 0;
    if (hidden_len != g->dim || logits_len != g->vocab_size) return 0;

    float* state = g->scratch_state.data();
    std::memcpy(state, hidden_in, static_cast<std::size_t>(g->dim) * sizeof(float));

    if (g->final_norm != nullptr)
        rmsnorm_inplace(state, g->final_norm, g->dim, g->rms_eps);

    if (g->lm_head != nullptr) {
        if (g->lm_head_transposed) {
            for (int v = 0; v < g->vocab_size; ++v) {
                float acc = 0.0f;
                for (int d = 0; d < g->dim; ++d)
                    acc += state[d] * g->lm_head[static_cast<std::size_t>(d) * g->vocab_size + v];
                logits_out[v] = acc;
            }
        } else {
            matvec_f32(g->lm_head, state, logits_out, g->vocab_size, g->dim);
        }
    } else if (g->lm_head_quant != nullptr) {
        matvec_quant(g->lm_head_quant, g->lm_head_qtype, state, logits_out, g->dim, g->vocab_size);
    } else {
        std::memset(logits_out, 0, static_cast<std::size_t>(g->vocab_size) * sizeof(float));
    }
    return 1;
}

void destroy_model_graph(ModelGraphHandle* g) {
    delete g;
}

}  // extern "C"
