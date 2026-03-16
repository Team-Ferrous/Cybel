/**
 * Native contiguous KV cache with integrated Flash Attention.
 *
 * Features:
 *  - Paged memory allocation: 4096-token pages allocated lazily
 *  - Supports 400K+ token context without upfront allocation
 *  - O(1) append via memcpy into page slots
 *  - Tiled online-softmax attention (CPU Flash Attention)
 *  - Direct pointer access for zero-copy from C++ graph execution
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fast_math.h"
#include "numa_allocator.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

extern "C" int anvil_get_num_threads_for_path(int decode_path);
extern "C" int anvil_get_thread_mode();
extern "C" int anvil_bind_worker_thread(int worker_tid, int role_decode);
extern "C" int anvil_get_l3_domain_count();

constexpr int KV_PAGE_TOKENS = 4096;  // Tokens per page

inline int get_num_threads() {
    const int mode = anvil_get_thread_mode();
    if (mode == 0 || mode == 1) {
        const int mode_threads = anvil_get_num_threads_for_path(mode);
        if (mode_threads > 0) {
            return mode_threads;
        }
    }
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

// ===== Paged KV Cache =====
// Pages are allocated lazily: only when tokens are first written to that range.
// Each page stores KV_PAGE_TOKENS tokens for one layer, one cache (K or V).

struct KVCacheMetricsSnapshot {
    int active_page_slots;
    int resident_page_count;
    int shared_page_slots;
    int snapshot_count;
    int active_tokens;
    int committed_token_capacity;
    int copy_on_write_events;
    int prefix_share_events;
    int page_tokens;
    float fragmentation_ratio;
};

struct KVPageStorage {
    std::vector<float> k_storage;
    std::vector<float> v_storage;
    int owner_l3_domain = -1;
    int owner_numa_node = -1;
    std::uint64_t lineage_id = 0;
};

struct KVPrefixSnapshot {
    int length_tokens = 0;
    std::vector<int> layer_lengths;
    std::vector<std::shared_ptr<KVPageStorage>> page_refs;
};

struct KVCacheHandle {
    int max_seq;
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int token_width;    // n_kv_heads * head_dim
    int page_tokens;    // KV_PAGE_TOKENS
    int total_pages;    // per layer: ceil(max_seq / page_tokens)

    // Pages: indexed as [layer * total_pages + page_idx]
    // nullptr means not yet allocated
    std::vector<std::shared_ptr<KVPageStorage>> page_refs;
    std::vector<int> layer_lengths;
    std::unordered_map<int, KVPrefixSnapshot> snapshots;
    int next_snapshot_id;
    std::uint64_t next_lineage_id;
    int prefix_share_events;
    int copy_on_write_events;
    // Contiguous gather buffer for flash attention
    std::vector<float> k_gather;
    std::vector<float> v_gather;

    KVCacheHandle(int seq, int layers, int kv_heads, int dim)
        : max_seq(seq),
          n_layers(layers),
          n_kv_heads(kv_heads),
          head_dim(dim),
          token_width(kv_heads * dim),
          page_tokens(KV_PAGE_TOKENS),
          next_snapshot_id(1),
          next_lineage_id(1),
          prefix_share_events(0),
          copy_on_write_events(0) {
        total_pages = (max_seq + page_tokens - 1) / page_tokens;
        std::size_t n_slots = static_cast<std::size_t>(n_layers) * total_pages;
        // Pre-allocate gather buffers to 1 page capacity
        std::size_t initial_gather = static_cast<std::size_t>(page_tokens) * token_width;
        k_gather.resize(initial_gather, 0.0f);
        v_gather.resize(initial_gather, 0.0f);
        page_refs.resize(n_slots);
        layer_lengths.resize(n_layers, 0);
    }

    bool valid_layer(int layer_idx) const {
        return layer_idx >= 0 && layer_idx < n_layers;
    }
    bool valid_pos(int pos) const {
        return pos >= 0 && pos < max_seq;
    }

    std::size_t slot(int layer_idx, int page_idx) const {
        return static_cast<std::size_t>(layer_idx) * total_pages + page_idx;
    }

    KVPageStorage* page_for_slot(std::size_t slot_idx) {
        const auto& page = page_refs[slot_idx];
        return page ? page.get() : nullptr;
    }

    const KVPageStorage* page_for_slot(std::size_t slot_idx) const {
        const auto& page = page_refs[slot_idx];
        return page ? page.get() : nullptr;
    }

    void allocate_page(std::size_t slot_idx, int page_idx) {
        auto page = std::make_shared<KVPageStorage>();
        std::size_t page_floats = static_cast<std::size_t>(page_tokens) * token_width;
        page->k_storage.resize(page_floats, 0.0f);
        page->v_storage.resize(page_floats, 0.0f);
        anvil::native::anvil_numa_advise_region(
            page->k_storage.data(),
            page_floats * sizeof(float)
        );
        anvil::native::anvil_numa_advise_region(
            page->v_storage.data(),
            page_floats * sizeof(float)
        );
        const int domain_count = std::max(1, anvil_get_l3_domain_count());
        page->owner_l3_domain = page_idx % domain_count;
        page->owner_numa_node = 0;
        page->lineage_id = next_lineage_id++;
        page_refs[slot_idx] = std::move(page);
    }

    void ensure_page(int layer_idx, int page_idx) {
        std::size_t slot_idx = slot(layer_idx, page_idx);
        if (!page_refs[slot_idx]) {
            allocate_page(slot_idx, page_idx);
        }
    }

    void ensure_unique_page(int layer_idx, int page_idx) {
        std::size_t slot_idx = slot(layer_idx, page_idx);
        if (!page_refs[slot_idx]) {
            allocate_page(slot_idx, page_idx);
            return;
        }
        if (page_refs[slot_idx].use_count() <= 1) {
            return;
        }
        auto clone = std::make_shared<KVPageStorage>(*page_refs[slot_idx]);
        clone->lineage_id = next_lineage_id++;
        page_refs[slot_idx] = std::move(clone);
        ++copy_on_write_events;
    }

    void ensure_gather(int kv_len) {
        std::size_t needed = static_cast<std::size_t>(kv_len) * token_width;
        if (k_gather.size() < needed) {
            k_gather.resize(needed);
            v_gather.resize(needed);
        }
    }

    // Gather contiguous K/V into buffers for attention
    void gather_kv(int layer_idx, int kv_len) {
        ensure_gather(kv_len);
        int pos = 0;
        while (pos < kv_len) {
            int pidx = pos / page_tokens;
            int off_in_page = pos % page_tokens;
            int count = std::min(page_tokens - off_in_page, kv_len - pos);
            std::size_t s = slot(layer_idx, pidx);
            std::size_t src_off = static_cast<std::size_t>(off_in_page) * token_width;
            std::size_t dst_off = static_cast<std::size_t>(pos) * token_width;
            std::size_t bytes = static_cast<std::size_t>(count) * token_width * sizeof(float);
            const auto* page = page_for_slot(s);
            if (page != nullptr) {
                std::memcpy(
                    k_gather.data() + dst_off,
                    page->k_storage.data() + src_off,
                    bytes
                );
                std::memcpy(
                    v_gather.data() + dst_off,
                    page->v_storage.data() + src_off,
                    bytes
                );
            } else {
                std::memset(k_gather.data() + dst_off, 0, bytes);
                std::memset(v_gather.data() + dst_off, 0, bytes);
            }
            pos += count;
        }
    }

    int clamp_length(int length_tokens) const {
        return std::max(0, std::min(length_tokens, max_seq));
    }

    int layer_length(int layer_idx) const {
        if (!valid_layer(layer_idx)) {
            return 0;
        }
        return layer_lengths[static_cast<std::size_t>(layer_idx)];
    }

    int snapshot_prefix(int length_tokens) {
        const int clamped = clamp_length(length_tokens);
        KVPrefixSnapshot snapshot;
        snapshot.length_tokens = clamped;
        snapshot.layer_lengths.resize(n_layers, 0);
        snapshot.page_refs.resize(page_refs.size());
        int shared_slots = 0;
        for (int layer = 0; layer < n_layers; ++layer) {
            const int layer_len = std::min(clamped, layer_lengths[static_cast<std::size_t>(layer)]);
            snapshot.layer_lengths[static_cast<std::size_t>(layer)] = layer_len;
            const int page_count = (layer_len + page_tokens - 1) / page_tokens;
            for (int page_idx = 0; page_idx < page_count; ++page_idx) {
                std::size_t slot_idx = slot(layer, page_idx);
                snapshot.page_refs[slot_idx] = page_refs[slot_idx];
                if (snapshot.page_refs[slot_idx]) {
                    ++shared_slots;
                }
            }
        }
        const int snapshot_id = next_snapshot_id++;
        snapshots.emplace(snapshot_id, std::move(snapshot));
        prefix_share_events += shared_slots;
        return snapshot_id;
    }

    bool restore_prefix(int snapshot_id) {
        const auto it = snapshots.find(snapshot_id);
        if (it == snapshots.end()) {
            return false;
        }
        page_refs = it->second.page_refs;
        layer_lengths = it->second.layer_lengths;
        int shared_slots = 0;
        for (const auto& page : page_refs) {
            if (page) {
                ++shared_slots;
            }
        }
        prefix_share_events += shared_slots;
        return true;
    }

    bool release_snapshot(int snapshot_id) {
        return snapshots.erase(snapshot_id) > 0;
    }

    void fill_metrics(KVCacheMetricsSnapshot* out) const {
        if (out == nullptr) {
            return;
        }
        std::memset(out, 0, sizeof(*out));
        std::unordered_set<const KVPageStorage*> resident_pages;
        int active_page_slots = 0;
        int shared_page_slots = 0;
        int active_tokens = 0;
        for (int layer = 0; layer < n_layers; ++layer) {
            active_tokens += layer_lengths[static_cast<std::size_t>(layer)];
        }
        for (const auto& page : page_refs) {
            if (!page) {
                continue;
            }
            ++active_page_slots;
            if (page.use_count() > 1) {
                ++shared_page_slots;
            }
            resident_pages.insert(page.get());
        }
        for (const auto& entry : snapshots) {
            for (const auto& page : entry.second.page_refs) {
                if (page) {
                    resident_pages.insert(page.get());
                }
            }
        }
        const int committed_capacity = active_page_slots * page_tokens;
        out->active_page_slots = active_page_slots;
        out->resident_page_count = static_cast<int>(resident_pages.size());
        out->shared_page_slots = shared_page_slots;
        out->snapshot_count = static_cast<int>(snapshots.size());
        out->active_tokens = active_tokens;
        out->committed_token_capacity = committed_capacity;
        out->copy_on_write_events = copy_on_write_events;
        out->prefix_share_events = prefix_share_events;
        out->page_tokens = page_tokens;
        if (committed_capacity > 0) {
            out->fragmentation_ratio = std::max(
                0.0f,
                1.0f - (static_cast<float>(active_tokens) / static_cast<float>(committed_capacity))
            );
        } else {
            out->fragmentation_ratio = 0.0f;
        }
    }
};

}  // namespace

extern "C" {

KVCacheHandle* kv_cache_create(int max_seq, int n_layers, int n_kv_heads, int head_dim) {
    if (max_seq <= 0 || n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0)
        return nullptr;
    try {
        return new KVCacheHandle(max_seq, n_layers, n_kv_heads, head_dim);
    } catch (...) {
        return nullptr;
    }
}

int kv_cache_append(
    KVCacheHandle* handle,
    int layer_idx, int pos,
    const float* k_ptr, const float* v_ptr
) {
    if (handle == nullptr || k_ptr == nullptr || v_ptr == nullptr) return 0;
    if (!handle->valid_layer(layer_idx) || !handle->valid_pos(pos)) return 0;

    int pidx = pos / handle->page_tokens;
    int off = pos % handle->page_tokens;
    handle->ensure_unique_page(layer_idx, pidx);

    std::size_t s = handle->slot(layer_idx, pidx);
    std::size_t offset = static_cast<std::size_t>(off) * handle->token_width;
    std::size_t bytes = static_cast<std::size_t>(handle->token_width) * sizeof(float);
    auto* page = handle->page_for_slot(s);
    if (page == nullptr) return 0;
    std::memcpy(page->k_storage.data() + offset, k_ptr, bytes);
    std::memcpy(page->v_storage.data() + offset, v_ptr, bytes);
    handle->layer_lengths[static_cast<std::size_t>(layer_idx)] = std::max(
        handle->layer_lengths[static_cast<std::size_t>(layer_idx)],
        pos + 1
    );
    return 1;
}

int kv_cache_append_batch(
    KVCacheHandle* handle,
    int layer_idx, int pos,
    const float* k_ptr, const float* v_ptr,
    int seq_len
) {
    if (handle == nullptr || k_ptr == nullptr || v_ptr == nullptr) return 0;
    if (!handle->valid_layer(layer_idx)) return 0;
    if (pos < 0 || pos + seq_len > handle->max_seq) return 0;

    int tw = handle->token_width;
    for (int t = 0; t < seq_len; ++t) {
        int p = pos + t;
        int pidx = p / handle->page_tokens;
        int off = p % handle->page_tokens;
        handle->ensure_unique_page(layer_idx, pidx);
        std::size_t s = handle->slot(layer_idx, pidx);
        std::size_t offset = static_cast<std::size_t>(off) * tw;
        std::size_t src_off = static_cast<std::size_t>(t) * tw;
        auto* page = handle->page_for_slot(s);
        if (page == nullptr) return 0;
        std::memcpy(page->k_storage.data() + offset, k_ptr + src_off, tw * sizeof(float));
        std::memcpy(page->v_storage.data() + offset, v_ptr + src_off, tw * sizeof(float));
    }
    handle->layer_lengths[static_cast<std::size_t>(layer_idx)] = std::max(
        handle->layer_lengths[static_cast<std::size_t>(layer_idx)],
        pos + seq_len
    );
    return 1;
}

const float* kv_cache_get_k(const KVCacheHandle* handle, int layer_idx) {
    // This returns the gather buffer after a gather call — for backward compat
    if (handle == nullptr || !handle->valid_layer(layer_idx)) return nullptr;
    const int kv_len = handle->layer_length(layer_idx);
    if (kv_len <= 0) return nullptr;
    const_cast<KVCacheHandle*>(handle)->gather_kv(layer_idx, kv_len);
    return handle->k_gather.data();
}

const float* kv_cache_get_v(const KVCacheHandle* handle, int layer_idx) {
    if (handle == nullptr || !handle->valid_layer(layer_idx)) return nullptr;
    const int kv_len = handle->layer_length(layer_idx);
    if (kv_len <= 0) return nullptr;
    const_cast<KVCacheHandle*>(handle)->gather_kv(layer_idx, kv_len);
    return handle->v_gather.data();
}

int kv_cache_snapshot_prefix(KVCacheHandle* handle, int length_tokens) {
    if (handle == nullptr) return 0;
    return handle->snapshot_prefix(length_tokens);
}

int kv_cache_restore_prefix(KVCacheHandle* handle, int snapshot_id) {
    if (handle == nullptr) return 0;
    return handle->restore_prefix(snapshot_id) ? 1 : 0;
}

int kv_cache_release_snapshot(KVCacheHandle* handle, int snapshot_id) {
    if (handle == nullptr) return 0;
    return handle->release_snapshot(snapshot_id) ? 1 : 0;
}

int kv_cache_get_metrics(
    const KVCacheHandle* handle,
    KVCacheMetricsSnapshot* out
) {
    if (handle == nullptr || out == nullptr) return 0;
    handle->fill_metrics(out);
    return 1;
}

// CPU Flash Attention with paged KV cache
int kv_cache_flash_attention(
    KVCacheHandle* handle,
    int layer_idx,
    const float* q,
    float* out,
    int n_heads,
    int kv_len,
    float scale
) {
    if (handle == nullptr || q == nullptr || out == nullptr) return 0;
    if (!handle->valid_layer(layer_idx)) return 0;
    if (kv_len <= 0 || kv_len > handle->max_seq) return 0;

    int n_kv_heads = handle->n_kv_heads;
    int head_dim = handle->head_dim;
    int heads_per_kv = std::max(1, n_heads / std::max(1, n_kv_heads));
    int tw = handle->token_width;

    // Gather contiguous KV for this layer (allows tiled access)
    handle->gather_kv(layer_idx, kv_len);
    const float* k_base = handle->k_gather.data();
    const float* v_base = handle->v_gather.data();

    constexpr int TILE_SIZE = 256;

    int n_threads = get_num_threads();

#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(static)
#endif
    for (int h = 0; h < n_heads; ++h) {
        maybe_bind_worker_thread(true);
        int kv_h = std::min(n_kv_heads - 1, h / heads_per_kv);
        const float* q_h = q + static_cast<std::size_t>(h) * head_dim;
        float* out_h = out + static_cast<std::size_t>(h) * head_dim;

        float running_max = -1e30f;
        float running_sum = 0.0f;
        std::memset(out_h, 0, static_cast<std::size_t>(head_dim) * sizeof(float));

        for (int tile_start = 0; tile_start < kv_len; tile_start += TILE_SIZE) {
            int tile_end = std::min(tile_start + TILE_SIZE, kv_len);
            int tile_len = tile_end - tile_start;

            float tile_scores[TILE_SIZE];
            float tile_max = -1e30f;

            for (int j = 0; j < tile_len; ++j) {
                int pos = tile_start + j;
                const float* k_j = k_base + static_cast<std::size_t>(pos) * tw
                                  + static_cast<std::size_t>(kv_h) * head_dim;
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
                tile_scores[j] = dot * scale;
                tile_max = std::max(tile_max, tile_scores[j]);
            }

            if (tile_max > running_max) {
                float correction = anvil_fast_math::fast_exp_scalar(running_max - tile_max);
                running_sum *= correction;
                for (int d = 0; d < head_dim; ++d)
                    out_h[d] *= correction;
                running_max = tile_max;
            }

            float tile_sum = 0.0f;
            for (int j = 0; j < tile_len; ++j) {
                float w = anvil_fast_math::fast_exp_scalar(tile_scores[j] - running_max);
                tile_sum += w;

                int pos = tile_start + j;
                const float* v_j = v_base + static_cast<std::size_t>(pos) * tw
                                  + static_cast<std::size_t>(kv_h) * head_dim;
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
            running_sum += tile_sum;
        }

        if (running_sum > 0.0f) {
            float inv = 1.0f / running_sum;
#ifdef __AVX2__
            __m256 inv_vec = _mm256_set1_ps(inv);
            int d = 0;
            for (; d + 8 <= head_dim; d += 8) {
                __m256 ov = _mm256_loadu_ps(out_h + d);
                _mm256_storeu_ps(out_h + d, _mm256_mul_ps(ov, inv_vec));
            }
            for (; d < head_dim; ++d)
                out_h[d] *= inv;
#else
            for (int d = 0; d < head_dim; ++d)
                out_h[d] *= inv;
#endif
        }
    }

    return 1;
}

int kv_cache_reset(KVCacheHandle* handle) {
    if (handle == nullptr) return 0;
    // Free all pages — resets to zero memory usage
    for (auto& page : handle->page_refs) page.reset();
    std::fill(handle->layer_lengths.begin(), handle->layer_lengths.end(), 0);
    handle->snapshots.clear();
    handle->next_snapshot_id = 1;
    handle->next_lineage_id = 1;
    handle->prefix_share_events = 0;
    handle->copy_on_write_events = 0;
    handle->k_gather.clear();
    handle->v_gather.clear();
    return 1;
}

void kv_cache_destroy(KVCacheHandle* handle) {
    delete handle;
}

}  // extern "C"
