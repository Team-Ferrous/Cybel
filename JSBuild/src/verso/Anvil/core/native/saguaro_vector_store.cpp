#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr float kMinNorm = 1.0e-9f;
std::atomic<long long> g_open_handles{0};
std::atomic<long long> g_remap_count{0};
std::atomic<long long> g_query_calls{0};

inline float normalize_score(float dot, float lhs_norm, float rhs_norm) {
    const float denom = std::max(lhs_norm * rhs_norm, kMinNorm);
    return dot / denom;
}

inline float scalar_norm(const float* values, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += values[i] * values[i];
    }
    return std::sqrt(sum);
}

inline float scalar_dot(const float* lhs, const float* rhs, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

#if defined(__AVX2__)
inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    const __m128 sum128 = _mm_add_ps(low, high);
    __m128 shuf = _mm_movehdup_ps(sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float fast_norm(const float* values, int dim) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 v = _mm256_loadu_ps(values + i);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    float sum = horizontal_sum(acc);
    for (; i < dim; ++i) {
        sum += values[i] * values[i];
    }
    return std::sqrt(sum);
}

inline float fast_dot(const float* lhs, const float* rhs, int dim) {
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        const __m256 lv = _mm256_loadu_ps(lhs + i);
        const __m256 rv = _mm256_loadu_ps(rhs + i);
        acc = _mm256_fmadd_ps(lv, rv, acc);
    }
    float sum = horizontal_sum(acc);
    for (; i < dim; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}
#endif

inline float compute_norm(const float* values, int dim) {
#if defined(__AVX2__)
    return fast_norm(values, dim);
#else
    return scalar_norm(values, dim);
#endif
}

inline float compute_dot(const float* lhs, const float* rhs, int dim) {
#if defined(__AVX2__)
    return fast_dot(lhs, rhs, dim);
#else
    return scalar_dot(lhs, rhs, dim);
#endif
}

struct Mapping {
    int fd = -1;
    void* base = MAP_FAILED;
    std::size_t bytes = 0;

    void close_mapping() {
        if (base != MAP_FAILED && bytes > 0) {
            munmap(base, bytes);
        }
        base = MAP_FAILED;
        bytes = 0;
        if (fd >= 0) {
            close(fd);
        }
        fd = -1;
    }

    ~Mapping() { close_mapping(); }
};

inline bool open_mapping(
    const char* path,
    std::size_t required_bytes,
    int open_flags,
    int protection,
    Mapping* mapping
) {
    mapping->fd = open(path, open_flags);
    if (mapping->fd < 0) {
        return false;
    }

    struct stat stat_buf {};
    if (fstat(mapping->fd, &stat_buf) != 0) {
        return false;
    }
    const std::size_t available_bytes = static_cast<std::size_t>(stat_buf.st_size);
    if (available_bytes < required_bytes) {
        errno = EINVAL;
        return false;
    }

    mapping->bytes = required_bytes;
    mapping->base = mmap(nullptr, required_bytes, protection, MAP_SHARED, mapping->fd, 0);
    if (mapping->base == MAP_FAILED) {
        return false;
    }
    g_remap_count.fetch_add(1, std::memory_order_relaxed);
    return true;
}

struct StoreHandle {
    int dim = 0;
    int count = 0;
    Mapping vectors;
    Mapping norms;

    const float* vector_base() const {
        return static_cast<const float*>(vectors.base);
    }

    const float* norm_base() const {
        return static_cast<const float*>(norms.base);
    }

    ~StoreHandle() {
        g_open_handles.fetch_sub(1, std::memory_order_relaxed);
    }
};

inline int query_store_impl(
    const StoreHandle* handle,
    const float* query,
    const std::int32_t* indices,
    int index_count,
    float* out_scores
) {
    if (handle == nullptr || query == nullptr || out_scores == nullptr) {
        errno = EINVAL;
        return -1;
    }
    const int count = handle->count;
    const int dim = handle->dim;
    if (count < 0 || dim <= 0 || index_count < 0) {
        errno = EINVAL;
        return -1;
    }

    const int active_count = index_count > 0 ? index_count : count;
    if (active_count == 0) {
        return 0;
    }

    const float* vectors = handle->vector_base();
    const float* norms = handle->norm_base();
    const float query_norm = compute_norm(query, dim);
    g_query_calls.fetch_add(1, std::memory_order_relaxed);
    if (query_norm <= kMinNorm) {
        for (int i = 0; i < active_count; ++i) {
            out_scores[i] = 0.0f;
        }
        return active_count;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int candidate = 0; candidate < active_count; ++candidate) {
        const int row_index = index_count > 0 ? static_cast<int>(indices[candidate]) : candidate;
        if (row_index < 0 || row_index >= count) {
            out_scores[candidate] = -std::numeric_limits<float>::infinity();
            continue;
        }
        const float* row = vectors + (static_cast<std::size_t>(row_index) * static_cast<std::size_t>(dim));
        const float row_norm = norms ? norms[row_index] : compute_norm(row, dim);
        if (row_norm <= kMinNorm) {
            out_scores[candidate] = 0.0f;
            continue;
        }
        const float dot = compute_dot(row, query, dim);
        out_scores[candidate] = normalize_score(dot, row_norm, query_norm);
    }

    return active_count;
}

}  // namespace

extern "C" int anvil_saguaro_query_cosine(
    const char* vectors_path,
    int dim,
    int count,
    const float* query,
    const std::int32_t* indices,
    int index_count,
    float* out_scores
) {
    if (vectors_path == nullptr || query == nullptr || out_scores == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (dim <= 0 || count < 0 || index_count < 0) {
        errno = EINVAL;
        return -1;
    }

    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    const std::size_t required_bytes = static_cast<std::size_t>(count) * row_bytes;
    Mapping mapping;
    if (!open_mapping(vectors_path, required_bytes, O_RDONLY, PROT_READ, &mapping)) {
        return -1;
    }

    StoreHandle handle;
    handle.dim = dim;
    handle.count = count;
    handle.vectors.fd = mapping.fd;
    handle.vectors.base = mapping.base;
    handle.vectors.bytes = mapping.bytes;
    mapping.fd = -1;
    mapping.base = MAP_FAILED;
    mapping.bytes = 0;
    return query_store_impl(&handle, query, indices, index_count, out_scores);
}

extern "C" void* anvil_saguaro_open_store(
    const char* vectors_path,
    const char* norms_path,
    int dim,
    int count
) {
    if (vectors_path == nullptr || norms_path == nullptr || dim <= 0 || count < 0) {
        errno = EINVAL;
        return nullptr;
    }

    const std::size_t vector_bytes =
        static_cast<std::size_t>(count) * static_cast<std::size_t>(dim) * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(count) * sizeof(float);

    auto* handle = new StoreHandle();
    handle->dim = dim;
    handle->count = count;

    if (count > 0 && !open_mapping(vectors_path, vector_bytes, O_RDONLY, PROT_READ, &handle->vectors)) {
        delete handle;
        return nullptr;
    }
    if (count > 0 && !open_mapping(norms_path, norm_bytes, O_RDONLY, PROT_READ, &handle->norms)) {
        delete handle;
        return nullptr;
    }
    g_open_handles.fetch_add(1, std::memory_order_relaxed);
    return handle;
}

extern "C" int anvil_saguaro_query_store(
    void* store_handle,
    const float* query,
    const std::int32_t* indices,
    int index_count,
    float* out_scores
) {
    return query_store_impl(
        static_cast<const StoreHandle*>(store_handle),
        query,
        indices,
        index_count,
        out_scores
    );
}

extern "C" int anvil_saguaro_close_store(void* store_handle) {
    if (store_handle == nullptr) {
        return 0;
    }
    delete static_cast<StoreHandle*>(store_handle);
    return 0;
}

extern "C" long long anvil_saguaro_perf_open_handles() {
    return g_open_handles.load(std::memory_order_relaxed);
}

extern "C" long long anvil_saguaro_perf_remap_count() {
    return g_remap_count.load(std::memory_order_relaxed);
}

extern "C" long long anvil_saguaro_perf_query_calls() {
    return g_query_calls.load(std::memory_order_relaxed);
}

extern "C" int anvil_saguaro_write_indexed_rows(
    const char* vectors_path,
    int dim,
    int capacity,
    const std::int32_t* indices,
    int index_count,
    const float* rows
) {
    if (vectors_path == nullptr || indices == nullptr || rows == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (dim <= 0 || capacity < 0 || index_count < 0) {
        errno = EINVAL;
        return -1;
    }
    if (index_count == 0) {
        return 0;
    }

    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    const std::size_t required_bytes = static_cast<std::size_t>(capacity) * row_bytes;
    for (int row = 0; row < index_count; ++row) {
        const int target_index = static_cast<int>(indices[row]);
        if (target_index < 0 || target_index >= capacity) {
            errno = ERANGE;
            return -1;
        }
    }

    Mapping mapping;
    if (!open_mapping(vectors_path, required_bytes, O_RDWR, PROT_READ | PROT_WRITE, &mapping)) {
        return -1;
    }

    auto* base = static_cast<std::uint8_t*>(mapping.base);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(index_count > 8)
#endif
    for (int row = 0; row < index_count; ++row) {
        const int target_index = static_cast<int>(indices[row]);
        const std::size_t target_offset = static_cast<std::size_t>(target_index) * row_bytes;
        const auto* src = reinterpret_cast<const std::uint8_t*>(
            rows + (static_cast<std::size_t>(row) * static_cast<std::size_t>(dim))
        );
        std::memcpy(base + target_offset, src, row_bytes);
    }

    return index_count;
}

extern "C" int anvil_saguaro_write_indexed_rows_with_norms(
    const char* vectors_path,
    int dim,
    int capacity,
    const std::int32_t* indices,
    int index_count,
    const float* rows,
    float* out_norms
) {
    if (vectors_path == nullptr || indices == nullptr || rows == nullptr || out_norms == nullptr) {
        errno = EINVAL;
        return -1;
    }
    if (dim <= 0 || capacity < 0 || index_count < 0) {
        errno = EINVAL;
        return -1;
    }
    if (index_count == 0) {
        return 0;
    }

    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    const std::size_t required_bytes = static_cast<std::size_t>(capacity) * row_bytes;
    for (int row = 0; row < index_count; ++row) {
        const int target_index = static_cast<int>(indices[row]);
        if (target_index < 0 || target_index >= capacity) {
            errno = ERANGE;
            return -1;
        }
    }

    Mapping mapping;
    if (!open_mapping(vectors_path, required_bytes, O_RDWR, PROT_READ | PROT_WRITE, &mapping)) {
        return -1;
    }

    auto* base = static_cast<std::uint8_t*>(mapping.base);

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if(index_count > 8)
#endif
    for (int row = 0; row < index_count; ++row) {
        const int target_index = static_cast<int>(indices[row]);
        const std::size_t target_offset = static_cast<std::size_t>(target_index) * row_bytes;
        const float* src = rows + (static_cast<std::size_t>(row) * static_cast<std::size_t>(dim));
        std::memcpy(base + target_offset, src, row_bytes);
        out_norms[row] = compute_norm(src, dim);
    }

    return index_count;
}
