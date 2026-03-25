#include "qsg_parallel_kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fused_speculative_op.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif

extern "C" {
int graph_forward_token_id(
    void* g,
    int token_id,
    float* logits_out,
    int logits_len,
    int pos,
    void* drift_out);
int graph_forward_token_ids(
    void* g,
    const int* token_ids,
    int token_count,
    float* logits_out,
    int logits_len,
    int start_pos,
    void* drift_out);
void* graph_create_execution_checkpoint(const void* g);
int graph_restore_execution_checkpoint(
    void* g,
    const void* checkpoint);
void graph_destroy_execution_checkpoint(void* checkpoint);
int graph_reset(void* g);
void simd_seed_rng_f32(int seed);
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
    float min_p);
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
    float* token_prob_out);
}

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kLatencyWindow = 4096;
constexpr std::size_t kQueueWaitWindow = 8192;

struct RequestEntry {
    std::int32_t priority = 0;
    std::int64_t arrival_ts_ns = 0;
    std::int64_t first_scheduled_ns = 0;
    std::int32_t prompt_token_count = 0;
    std::int32_t max_new_tokens = 0;
    std::int32_t prefill_chunk_size = 0;
    std::int32_t prefill_chunk_count = 0;
    std::int64_t generated_tokens = 0;
    bool in_pending = false;
    bool in_active = false;
    bool cancelled = false;
    bool completed = false;
    bool prefill_accounted = false;
    bool decode_started = false;
    bool latent = false;
    bool suspended = false;
};

struct NativeSchedulerState {
    std::int32_t max_active_requests = 1;
    std::int32_t max_pending_requests = 1;
    bool priority_policy = false;
    bool interleaved_streams = false;

    std::unordered_map<std::string, RequestEntry> requests;
    std::deque<std::string> pending;
    std::deque<std::string> active;

    std::vector<double> iteration_latencies_ms;
    std::vector<double> queue_wait_ms;

    std::int64_t admitted_requests = 0;
    std::int64_t completed_requests = 0;
    std::int64_t cancelled_requests = 0;
    std::int64_t evicted_requests = 0;
    std::int64_t iterations = 0;
    std::int64_t prefill_request_count = 0;
    std::int64_t prefill_tokens_scheduled = 0;
    std::int64_t decode_tokens_emitted = 0;
    std::int64_t chunked_prefill_requests = 0;
    std::int64_t chunked_prefill_chunks = 0;

    double iteration_last_ms = 0.0;
    double iteration_sum_ms = 0.0;
};

struct RuntimeEvent {
    std::int32_t token_id = -1;
    bool has_token = false;
    bool done = false;
    std::string error;
};

struct RuntimeRequest {
    std::string request_id;
    std::vector<std::int32_t> prompt_tokens;
    std::vector<std::int32_t> token_history;
    std::vector<float> logits;
    std::deque<RuntimeEvent> events;
    void* checkpoint = nullptr;
    std::int32_t max_new_tokens = 0;
    std::int32_t generated_tokens = 0;
    std::int32_t position = 0;
    std::int32_t no_repeat_ngram_size = 0;
    std::int32_t min_new_tokens_before_eos = 0;
    std::int64_t seed = 0;
    float temperature = 0.8f;
    float top_p = 1.0f;
    std::int32_t top_k = 0;
    float min_p = 0.0f;
    float presence_penalty = 0.0f;
    float repetition_penalty = 1.0f;
    bool prefilled = false;
    bool completed = false;
    bool cancelled = false;
    bool latent = false;
    bool suspended = false;
    bool seed_applied = false;
};

struct NativeRuntimeState {
    void* graph = nullptr;
    std::int32_t vocab_size = 0;
    std::int32_t eos_token = -1;
    std::int32_t ubatch = 1;
    NativeSchedulerState scheduler;
    std::unordered_map<std::string, RuntimeRequest> requests;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread worker;
    bool shutdown_requested = false;
    bool worker_running = false;
    std::int64_t worker_iterations = 0;
    std::int64_t emitted_events = 0;
    std::int64_t prefill_batches = 0;
    std::int64_t runtime_prefill_tokens = 0;
    std::int64_t runtime_decode_steps = 0;
};

double percentile(const std::vector<double>& values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::vector<double> ordered(values);
    std::sort(ordered.begin(), ordered.end());
    const double clamped_q = std::max(0.0, std::min(1.0, q));
    const std::size_t idx = static_cast<std::size_t>(
        std::floor(clamped_q * static_cast<double>(ordered.size() - 1)));
    return ordered[idx];
}

void erase_from_deque(std::deque<std::string>& values, const std::string& item) {
    const auto it = std::find(values.begin(), values.end(), item);
    if (it != values.end()) {
        values.erase(it);
    }
}

void append_sample(std::vector<double>& values, double sample, std::size_t max_size) {
    if (!std::isfinite(sample)) {
        return;
    }
    values.push_back(std::max(0.0, sample));
    if (values.size() > max_size) {
        values.erase(values.begin());
    }
}

void promote_pending_locked(NativeSchedulerState* state) {
    if (state == nullptr) {
        return;
    }
    std::size_t inspected = 0;
    const std::size_t pending_window = state->pending.size();
    while (!state->pending.empty() &&
           static_cast<std::int32_t>(state->active.size()) < state->max_active_requests &&
           inspected < pending_window) {
        const std::string request_id = state->pending.front();
        state->pending.pop_front();
        inspected += 1;
        auto it = state->requests.find(request_id);
        if (it == state->requests.end()) {
            continue;
        }
        RequestEntry& entry = it->second;
        if (entry.completed || entry.cancelled) {
            continue;
        }
        if (entry.suspended) {
            entry.in_pending = true;
            state->pending.push_back(request_id);
            continue;
        }
        entry.in_pending = false;
        entry.in_active = true;
        if (entry.first_scheduled_ns <= 0) {
            entry.first_scheduled_ns =
                std::chrono::time_point_cast<std::chrono::nanoseconds>(
                    Clock::now())
                    .time_since_epoch()
                    .count();
            if (entry.arrival_ts_ns > 0) {
                const double wait_ms = static_cast<double>(
                    std::max<std::int64_t>(0, entry.first_scheduled_ns - entry.arrival_ts_ns)) /
                    1'000'000.0;
                append_sample(state->queue_wait_ms, wait_ms, kQueueWaitWindow);
            }
        }
        if (!entry.prefill_accounted && entry.prompt_token_count > 0) {
            entry.prefill_accounted = true;
            state->prefill_request_count += 1;
            state->prefill_tokens_scheduled +=
                static_cast<std::int64_t>(entry.prompt_token_count);
            if (entry.prefill_chunk_count > 1) {
                state->chunked_prefill_requests += 1;
                state->chunked_prefill_chunks +=
                    static_cast<std::int64_t>(entry.prefill_chunk_count);
            }
        }
        state->active.push_back(request_id);
    }
}

std::int32_t completion_code_for(bool cancelled) {
    return cancelled ? 2 : 1;
}

std::int32_t finish_request(
    NativeSchedulerState* state,
    const std::string& request_id,
    bool cancelled) {
    if (state == nullptr) {
        return -1;
    }
    const auto it = state->requests.find(request_id);
    if (it == state->requests.end()) {
        return 0;
    }
    RequestEntry& entry = it->second;
    if (entry.completed) {
        return completion_code_for(entry.cancelled);
    }
    entry.completed = true;
    if (cancelled) {
        entry.cancelled = true;
    }
    entry.in_active = false;
    entry.in_pending = false;
    erase_from_deque(state->pending, request_id);
    erase_from_deque(state->active, request_id);
    if (cancelled) {
        state->cancelled_requests += 1;
        state->evicted_requests += 1;
    } else {
        state->completed_requests += 1;
    }
    return completion_code_for(cancelled);
}

bool logits_buffer_valid(const float* logits, std::int32_t vocab_size) {
    return logits != nullptr && vocab_size > 1;
}

void apply_temperature_and_topk(
    std::vector<float>& logits,
    float temperature,
    std::int32_t top_k) {
    highnoon::ops::speculative_temperature_scale(
        logits.data(),
        static_cast<std::int64_t>(logits.size()),
        std::max(temperature, 1.0e-6f));
    if (top_k > 0 && top_k < static_cast<std::int32_t>(logits.size())) {
        highnoon::ops::speculative_top_k_filter(
            logits.data(),
            static_cast<std::int64_t>(logits.size()),
            static_cast<std::int64_t>(top_k));
    }
}

float dot_product_hidden_row(
    const float* hidden,
    const float* row,
    std::int32_t hidden_dim) {
    if (hidden == nullptr || row == nullptr || hidden_dim <= 0) {
        return 0.0f;
    }
#ifdef __AVX2__
    __m256 vacc = _mm256_setzero_ps();
    std::int32_t d = 0;
    for (; d + 8 <= hidden_dim; d += 8) {
        const __m256 hv = _mm256_loadu_ps(hidden + d);
        const __m256 wv = _mm256_loadu_ps(row + d);
        vacc = _mm256_fmadd_ps(hv, wv, vacc);
    }
    alignas(32) float partial[8];
    _mm256_store_ps(partial, vacc);
    float acc = partial[0] + partial[1] + partial[2] + partial[3] +
                partial[4] + partial[5] + partial[6] + partial[7];
    for (; d < hidden_dim; ++d) {
        acc += hidden[d] * row[d];
    }
    return acc;
#else
    float acc = 0.0f;
    for (std::int32_t d = 0; d < hidden_dim; ++d) {
        acc += hidden[d] * row[d];
    }
    return acc;
#endif
}

void project_head_logits(
    const float* hidden,
    const float* weights,
    const float* bias,
    std::int32_t hidden_dim,
    std::int32_t vocab_size,
    std::vector<float>& out_logits) {
    out_logits.assign(static_cast<std::size_t>(vocab_size), 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int32_t vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx) {
        const float* row =
            weights + static_cast<std::size_t>(vocab_idx) * hidden_dim;
        float acc = dot_product_hidden_row(hidden, row, hidden_dim);
        if (bias != nullptr) {
            acc += bias[vocab_idx];
        }
        out_logits[static_cast<std::size_t>(vocab_idx)] = acc;
    }
}

void destroy_runtime_checkpoint(RuntimeRequest& request) {
    if (request.checkpoint != nullptr) {
        graph_destroy_execution_checkpoint(request.checkpoint);
        request.checkpoint = nullptr;
    }
}

void enqueue_runtime_event(
    NativeRuntimeState* state,
    RuntimeRequest& request,
    RuntimeEvent event) {
    request.events.push_back(std::move(event));
    if (state != nullptr) {
        state->emitted_events += 1;
    }
}

bool runtime_has_unfinished_requests(const NativeRuntimeState* state) {
    if (state == nullptr) {
        return false;
    }
    for (const auto& pair : state->requests) {
        const RuntimeRequest& request = pair.second;
        if (!request.completed && !request.cancelled) {
            return true;
        }
    }
    return false;
}

void finish_runtime_request(
    NativeRuntimeState* state,
    RuntimeRequest& request,
    bool cancelled,
    const std::string& error = std::string()) {
    if (state == nullptr || request.completed) {
        return;
    }
    request.completed = true;
    request.cancelled = cancelled;
    destroy_runtime_checkpoint(request);
    finish_request(&state->scheduler, request.request_id, cancelled);
    RuntimeEvent event;
    event.done = true;
    event.error = error;
    enqueue_runtime_event(state, request, std::move(event));
}

bool prefill_runtime_request(
    NativeRuntimeState* state,
    RuntimeRequest& request,
    std::string& error_out) {
    if (state == nullptr || state->graph == nullptr || state->vocab_size <= 0) {
        error_out = "native runtime graph unavailable";
        return false;
    }
    if (graph_reset(state->graph) != 1) {
        error_out = "graph reset failed";
        return false;
    }
    request.position = 0;
    request.logits.assign(static_cast<std::size_t>(state->vocab_size), 0.0f);
    if (request.prompt_tokens.empty()) {
        request.prefilled = true;
        return true;
    }
    const std::int32_t chunk_size = std::max<std::int32_t>(1, state->ubatch);
    for (std::size_t offset = 0; offset < request.prompt_tokens.size();
         offset += static_cast<std::size_t>(chunk_size)) {
        const std::size_t remaining = request.prompt_tokens.size() - offset;
        const std::int32_t count = static_cast<std::int32_t>(
            std::min<std::size_t>(remaining, static_cast<std::size_t>(chunk_size)));
        if (graph_forward_token_ids(
                state->graph,
                request.prompt_tokens.data() + offset,
                count,
                request.logits.data(),
                state->vocab_size,
                request.position,
                nullptr) != 1) {
            error_out = "batched prefill failed";
            return false;
        }
        request.position += count;
        state->prefill_batches += 1;
        state->runtime_prefill_tokens += count;
    }
    destroy_runtime_checkpoint(request);
    request.checkpoint = graph_create_execution_checkpoint(state->graph);
    request.prefilled = true;
    return true;
}

bool restore_runtime_request(
    NativeRuntimeState* state,
    RuntimeRequest& request,
    std::string& error_out) {
    if (state == nullptr || state->graph == nullptr) {
        error_out = "native runtime graph unavailable";
        return false;
    }
    if (!request.prefilled) {
        return prefill_runtime_request(state, request, error_out);
    }
    if (request.checkpoint == nullptr) {
        error_out = "missing execution checkpoint";
        return false;
    }
    if (graph_restore_execution_checkpoint(state->graph, request.checkpoint) != 1) {
        error_out = "restore execution checkpoint failed";
        return false;
    }
    return true;
}

bool advance_runtime_request(
    NativeRuntimeState* state,
    RuntimeRequest& request,
    std::int32_t token,
    std::string& error_out) {
    if (state == nullptr || state->graph == nullptr || state->vocab_size <= 0) {
        error_out = "native runtime graph unavailable";
        return false;
    }
    request.logits.assign(static_cast<std::size_t>(state->vocab_size), 0.0f);
    if (graph_forward_token_id(
            state->graph,
            token,
            request.logits.data(),
            state->vocab_size,
            request.position,
            nullptr) != 1) {
        error_out = "graph decode step failed";
        return false;
    }
    request.position += 1;
    destroy_runtime_checkpoint(request);
    request.checkpoint = graph_create_execution_checkpoint(state->graph);
    return true;
}

void build_runtime_suppressed_ids(
    const NativeRuntimeState& state,
    const RuntimeRequest& request,
    std::vector<int>& suppressed_ids) {
    suppressed_ids.clear();
    if (request.generated_tokens < request.min_new_tokens_before_eos &&
        state.eos_token >= 0) {
        suppressed_ids.push_back(state.eos_token);
    }
}

bool sample_runtime_recovery_token(
    const NativeRuntimeState& state,
    const RuntimeRequest& request,
    std::int32_t& token_out) {
    token_out = -1;
    if (request.logits.empty()) {
        return false;
    }
    std::vector<int> suppressed_ids;
    build_runtime_suppressed_ids(state, request, suppressed_ids);
    token_out = static_cast<std::int32_t>(simd_qsg_postprocess_sample_f32(
        const_cast<float*>(request.logits.data()),
        static_cast<int>(request.logits.size()),
        suppressed_ids.empty() ? nullptr : suppressed_ids.data(),
        static_cast<int>(suppressed_ids.size()),
        request.token_history.empty() ? nullptr : request.token_history.data(),
        static_cast<int>(request.token_history.size()),
        nullptr,
        0,
        0,
        0,
        0.0f,
        0,
        0,
        0.0f,
        request.presence_penalty,
        request.repetition_penalty,
        request.no_repeat_ngram_size,
        std::max(1.0e-6f, request.temperature),
        state.eos_token,
        request.top_p,
        request.top_k,
        request.min_p));
    return token_out >= 0;
}

void process_runtime_request(
    NativeRuntimeState* state,
    RuntimeRequest& request) {
    if (state == nullptr || request.completed || request.cancelled || request.suspended) {
        return;
    }
    std::string error;
    if (!restore_runtime_request(state, request, error)) {
        finish_runtime_request(state, request, false, error);
        return;
    }
    if (request.generated_tokens >= std::max<std::int32_t>(0, request.max_new_tokens)) {
        finish_runtime_request(state, request, false);
        return;
    }
    if (!request.seed_applied) {
        if (request.seed != 0) {
            simd_seed_rng_f32(static_cast<int>(request.seed));
        }
        request.seed_applied = true;
    }
    std::vector<int> suppressed_ids;
    if (request.generated_tokens < request.min_new_tokens_before_eos &&
        state->eos_token >= 0) {
        suppressed_ids.push_back(state->eos_token);
    }
    const std::int32_t token = static_cast<std::int32_t>(simd_qsg_postprocess_sample_f32(
        request.logits.empty() ? nullptr : request.logits.data(),
        static_cast<int>(request.logits.size()),
        suppressed_ids.empty() ? nullptr : suppressed_ids.data(),
        static_cast<int>(suppressed_ids.size()),
        request.token_history.empty() ? nullptr : request.token_history.data(),
        static_cast<int>(request.token_history.size()),
        nullptr,
        0,
        0,
        0,
        0.0f,
        0,
        0,
        0.0f,
        request.presence_penalty,
        request.repetition_penalty,
        request.no_repeat_ngram_size,
        std::max(1.0e-6f, request.temperature),
        state->eos_token,
        request.top_p,
        request.top_k,
        request.min_p));
    request.token_history.push_back(token);
    request.generated_tokens += 1;
    qsg_scheduler_record_decode_emit(&state->scheduler, request.request_id.c_str(), 1);
    state->runtime_decode_steps += 1;
    RuntimeEvent token_event;
    token_event.has_token = true;
    token_event.token_id = token;
    enqueue_runtime_event(state, request, std::move(token_event));
    if (token == state->eos_token ||
        request.generated_tokens >= std::max<std::int32_t>(0, request.max_new_tokens)) {
        finish_runtime_request(state, request, false);
        return;
    }
    if (!advance_runtime_request(state, request, token, error)) {
        finish_runtime_request(state, request, false, error);
    }
}

void runtime_worker_loop(NativeRuntimeState* state) {
    if (state == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> lock(state->mutex);
    state->worker_running = true;
    state->cv.notify_all();
    while (true) {
        promote_pending_locked(&state->scheduler);
        const bool has_active = !state->scheduler.active.empty();
        const bool has_unfinished = runtime_has_unfinished_requests(state);
        if (state->shutdown_requested && !has_active && !has_unfinished) {
            break;
        }
        if (!has_active) {
            state->cv.wait_for(lock, std::chrono::milliseconds(1));
            continue;
        }
        const auto iteration_started = Clock::now();
        const std::size_t active_count = state->scheduler.interleaved_streams
            ? state->scheduler.active.size()
            : std::min<std::size_t>(1, state->scheduler.active.size());
        std::vector<std::string> active_ids;
        active_ids.reserve(active_count);
        for (std::size_t idx = 0; idx < active_count; ++idx) {
            active_ids.push_back(state->scheduler.active[idx]);
        }
        for (const std::string& request_id : active_ids) {
            auto it = state->requests.find(request_id);
            if (it == state->requests.end()) {
                continue;
            }
            process_runtime_request(state, it->second);
        }
        const auto iteration_finished = Clock::now();
        const double iteration_ms =
            std::chrono::duration<double, std::milli>(iteration_finished - iteration_started)
                .count();
        qsg_scheduler_record_iteration(&state->scheduler, iteration_ms);
        qsg_scheduler_rotate_active(&state->scheduler);
        promote_pending_locked(&state->scheduler);
        state->worker_iterations += 1;
        state->cv.notify_all();
    }
    state->worker_running = false;
    lock.unlock();
    state->cv.notify_all();
}

}  // namespace

extern "C" {

void* qsg_scheduler_create(
    std::int32_t max_active_requests,
    std::int32_t max_pending_requests,
    std::int32_t scheduler_policy_priority,
    std::int32_t interleaved_streams) {
    auto* state = new NativeSchedulerState();
    state->max_active_requests = std::max<std::int32_t>(1, max_active_requests);
    state->max_pending_requests = std::max<std::int32_t>(1, max_pending_requests);
    state->priority_policy = scheduler_policy_priority != 0;
    state->interleaved_streams = interleaved_streams != 0;
    return state;
}

void qsg_scheduler_destroy(void* handle) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    delete state;
}

std::int32_t qsg_scheduler_submit(
    void* handle,
    const char* request_id,
    std::int32_t priority,
    std::int64_t arrival_ts_ns,
    std::int32_t prompt_token_count,
    std::int32_t max_new_tokens,
    std::int32_t prefill_chunk_size) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr || request_id[0] == '\0') {
        return -1;
    }
    const std::string id(request_id);
    if (state->requests.find(id) != state->requests.end()) {
        return -3;
    }
    if (static_cast<std::int32_t>(state->pending.size()) >= state->max_pending_requests) {
        return -2;
    }
    RequestEntry entry;
    entry.priority = priority;
    entry.arrival_ts_ns = arrival_ts_ns;
    entry.prompt_token_count = std::max<std::int32_t>(0, prompt_token_count);
    entry.max_new_tokens = std::max<std::int32_t>(0, max_new_tokens);
    entry.prefill_chunk_size = std::max<std::int32_t>(1, prefill_chunk_size);
    entry.prefill_chunk_count =
        entry.prompt_token_count > 0
            ? std::max<std::int32_t>(
                  1,
                  (entry.prompt_token_count + entry.prefill_chunk_size - 1) /
                      entry.prefill_chunk_size)
            : 0;
    entry.in_pending = true;
    state->requests.emplace(id, entry);
    if (!state->priority_policy || state->pending.empty()) {
        state->pending.push_back(id);
    } else {
        bool inserted = false;
        for (auto it = state->pending.begin(); it != state->pending.end(); ++it) {
            const auto existing = state->requests.find(*it);
            if (existing == state->requests.end()) {
                continue;
            }
            const RequestEntry& other = existing->second;
            if (priority > other.priority) {
                state->pending.insert(it, id);
                inserted = true;
                break;
            }
            if (priority == other.priority && arrival_ts_ns < other.arrival_ts_ns) {
                state->pending.insert(it, id);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            state->pending.push_back(id);
        }
    }
    state->admitted_requests += 1;
    return 0;
}

std::int32_t qsg_scheduler_cancel(void* handle, const char* request_id) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    return finish_request(state, std::string(request_id), true);
}

std::int32_t qsg_scheduler_complete(
    void* handle,
    const char* request_id,
    std::int32_t cancelled) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    return finish_request(state, std::string(request_id), cancelled != 0);
}

void qsg_scheduler_promote(void* handle) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    promote_pending_locked(state);
}

std::int32_t qsg_scheduler_active_count(const void* handle) {
    const auto* state = reinterpret_cast<const NativeSchedulerState*>(handle);
    if (state == nullptr) {
        return 0;
    }
    if (state->active.empty()) {
        return 0;
    }
    if (!state->interleaved_streams) {
        return 1;
    }
    return static_cast<std::int32_t>(state->active.size());
}

std::int32_t qsg_scheduler_copy_active_id(
    const void* handle,
    std::int32_t idx,
    char* out,
    std::int32_t out_cap) {
    const auto* state = reinterpret_cast<const NativeSchedulerState*>(handle);
    if (state == nullptr || out == nullptr || out_cap <= 0) {
        return -1;
    }
    if (idx < 0) {
        out[0] = '\0';
        return -1;
    }
    std::string value;
    if (!state->interleaved_streams) {
        if (idx > 0 || state->active.empty()) {
            out[0] = '\0';
            return 0;
        }
        value = state->active.front();
    } else {
        if (idx >= static_cast<std::int32_t>(state->active.size())) {
            out[0] = '\0';
            return 0;
        }
        value = state->active[static_cast<std::size_t>(idx)];
    }
    const std::size_t len = std::min<std::size_t>(
        static_cast<std::size_t>(out_cap - 1),
        value.size());
    std::memcpy(out, value.data(), len);
    out[len] = '\0';
    return static_cast<std::int32_t>(len);
}

void qsg_scheduler_rotate_active(void* handle) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || !state->interleaved_streams || state->active.size() <= 1) {
        return;
    }
    const std::string head = state->active.front();
    state->active.pop_front();
    state->active.push_back(head);
}

std::int64_t qsg_scheduler_first_scheduled_ns(
    const void* handle,
    const char* request_id) {
    const auto* state = reinterpret_cast<const NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return 0;
    }
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    return it->second.first_scheduled_ns;
}

std::int32_t qsg_scheduler_request_state(
    const void* handle,
    const char* request_id) {
    const auto* state = reinterpret_cast<const NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return 0;
    }
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    const RequestEntry& entry = it->second;
    std::int32_t bits = 1;
    if (entry.completed) {
        bits |= 2;
    }
    if (entry.cancelled) {
        bits |= 4;
    }
    if (entry.in_active) {
        bits |= 8;
    }
    if (entry.in_pending) {
        bits |= 16;
    }
    if (entry.latent) {
        bits |= 32;
    }
    if (entry.suspended) {
        bits |= 64;
    }
    return bits;
}

std::int32_t qsg_scheduler_set_request_latent(
    void* handle,
    const char* request_id,
    std::int32_t is_latent) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    it->second.latent = is_latent != 0;
    return 0;
}

std::int32_t qsg_scheduler_set_request_suspended(
    void* handle,
    const char* request_id,
    std::int32_t is_suspended) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    RequestEntry& entry = it->second;
    const bool suspended = is_suspended != 0;
    entry.suspended = suspended;
    if (suspended) {
        if (entry.in_active) {
            entry.in_active = false;
            erase_from_deque(state->active, std::string(request_id));
        }
        if (!entry.completed && !entry.cancelled && !entry.in_pending) {
            entry.in_pending = true;
            state->pending.push_back(std::string(request_id));
        }
        return 0;
    }
    if (!entry.completed && !entry.cancelled) {
        if (entry.in_pending) {
            erase_from_deque(state->pending, std::string(request_id));
        }
        if (!entry.in_active) {
            entry.in_pending = true;
            state->pending.push_front(std::string(request_id));
        }
    }
    return 0;
}

void qsg_scheduler_record_iteration(void* handle, double iteration_ms) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr) {
        return;
    }
    const double clipped = std::isfinite(iteration_ms) ? std::max(0.0, iteration_ms) : 0.0;
    state->iterations += 1;
    state->iteration_last_ms = clipped;
    state->iteration_sum_ms += clipped;
    append_sample(state->iteration_latencies_ms, clipped, kLatencyWindow);
}

void qsg_scheduler_record_decode_emit(
    void* handle,
    const char* request_id,
    std::int32_t emitted_tokens) {
    auto* state = reinterpret_cast<NativeSchedulerState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return;
    }
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return;
    }
    RequestEntry& entry = it->second;
    const std::int64_t emitted = std::max<std::int32_t>(0, emitted_tokens);
    if (emitted <= 0) {
        return;
    }
    entry.decode_started = true;
    entry.generated_tokens += emitted;
    state->decode_tokens_emitted += emitted;
}

void qsg_scheduler_get_metrics(
    const void* handle,
    QSGSchedulerMetricsNative* out_metrics) {
    if (out_metrics == nullptr) {
        return;
    }
    std::memset(out_metrics, 0, sizeof(QSGSchedulerMetricsNative));
    const auto* state = reinterpret_cast<const NativeSchedulerState*>(handle);
    if (state == nullptr) {
        return;
    }
    std::int32_t prefill_active = 0;
    std::int32_t decode_active = 0;
    for (const std::string& request_id : state->active) {
        const auto it = state->requests.find(request_id);
        if (it == state->requests.end()) {
            continue;
        }
        const RequestEntry& entry = it->second;
        if (entry.completed || entry.cancelled) {
            continue;
        }
        if (entry.decode_started) {
            decode_active += 1;
        } else {
            prefill_active += 1;
        }
    }
    out_metrics->queue_depth = static_cast<std::int32_t>(state->pending.size());
    out_metrics->active_requests = static_cast<std::int32_t>(state->active.size());
    out_metrics->inflight_requests = static_cast<std::int32_t>(state->requests.size());
    out_metrics->prefill_active_requests = prefill_active;
    out_metrics->decode_active_requests = decode_active;
    out_metrics->admitted_requests = state->admitted_requests;
    out_metrics->completed_requests = state->completed_requests;
    out_metrics->cancelled_requests = state->cancelled_requests;
    out_metrics->evicted_requests = state->evicted_requests;
    out_metrics->iterations = state->iterations;
    out_metrics->prefill_request_count = state->prefill_request_count;
    out_metrics->prefill_tokens_scheduled = state->prefill_tokens_scheduled;
    out_metrics->decode_tokens_emitted = state->decode_tokens_emitted;
    out_metrics->chunked_prefill_requests = state->chunked_prefill_requests;
    out_metrics->chunked_prefill_chunks = state->chunked_prefill_chunks;
    std::int64_t latent_requests = 0;
    std::int64_t suspended_requests = 0;
    for (const auto& pair : state->requests) {
        const RequestEntry& entry = pair.second;
        if (entry.completed || entry.cancelled) {
            continue;
        }
        if (entry.latent) {
            latent_requests += 1;
        }
        if (entry.suspended) {
            suspended_requests += 1;
        }
    }
    out_metrics->latent_requests = latent_requests;
    out_metrics->suspended_requests = suspended_requests;
    out_metrics->iteration_last_ms = state->iteration_last_ms;
    out_metrics->iteration_avg_ms =
        state->iterations > 0
            ? (state->iteration_sum_ms / static_cast<double>(state->iterations))
            : 0.0;
    out_metrics->iteration_p95_ms = percentile(state->iteration_latencies_ms, 0.95);
    out_metrics->queue_wait_p50_ms = percentile(state->queue_wait_ms, 0.50);
    out_metrics->queue_wait_p95_ms = percentile(state->queue_wait_ms, 0.95);
    out_metrics->queue_wait_p99_ms = percentile(state->queue_wait_ms, 0.99);
}

void* qsg_runtime_create(
    void* model_graph_handle,
    std::int32_t vocab_size,
    std::int32_t eos_token,
    std::int32_t ubatch,
    std::int32_t max_active_requests,
    std::int32_t max_pending_requests,
    std::int32_t scheduler_policy_priority,
    std::int32_t interleaved_streams) {
    auto* state = new NativeRuntimeState();
    state->graph = model_graph_handle;
    state->vocab_size = std::max<std::int32_t>(0, vocab_size);
    state->eos_token = eos_token;
    state->ubatch = std::max<std::int32_t>(1, ubatch);
    state->scheduler.max_active_requests = std::max<std::int32_t>(1, max_active_requests);
    state->scheduler.max_pending_requests = std::max<std::int32_t>(1, max_pending_requests);
    state->scheduler.priority_policy = scheduler_policy_priority != 0;
    state->scheduler.interleaved_streams = interleaved_streams != 0;
    state->worker = std::thread(runtime_worker_loop, state);
    return state;
}

void qsg_runtime_shutdown(void* handle) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->shutdown_requested = true;
    }
    state->cv.notify_all();
}

void qsg_runtime_destroy(void* handle) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return;
    }
    qsg_runtime_shutdown(handle);
    if (state->worker.joinable()) {
        state->worker.join();
    }
    for (auto& pair : state->requests) {
        destroy_runtime_checkpoint(pair.second);
    }
    delete state;
}

std::int32_t qsg_runtime_submit(
    void* handle,
    const char* request_id,
    std::int32_t priority,
    std::int64_t arrival_ts_ns,
    const std::int32_t* prompt_tokens,
    std::int32_t prompt_token_count,
    std::int32_t max_new_tokens,
    float temperature,
    float top_p,
    std::int32_t top_k,
    float min_p,
    float presence_penalty,
    float repetition_penalty,
    std::int32_t no_repeat_ngram_size,
    std::int32_t min_new_tokens_before_eos,
    std::int32_t seed_enabled,
    std::int64_t seed,
    std::int32_t latent,
    std::int32_t suspended) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr || request_id == nullptr || request_id[0] == '\0') {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const std::string id(request_id);
    if (state->requests.find(id) != state->requests.end()) {
        return -3;
    }
    const std::int32_t status = qsg_scheduler_submit(
        &state->scheduler,
        request_id,
        priority,
        arrival_ts_ns,
        prompt_token_count,
        max_new_tokens,
        state->ubatch);
    if (status != 0) {
        return status;
    }
    RuntimeRequest request;
    request.request_id = id;
    if (prompt_tokens != nullptr && prompt_token_count > 0) {
        request.prompt_tokens.assign(
            prompt_tokens,
            prompt_tokens + static_cast<std::size_t>(prompt_token_count));
        request.token_history = request.prompt_tokens;
    }
    request.max_new_tokens = std::max<std::int32_t>(0, max_new_tokens);
    request.temperature = std::isfinite(temperature) ? temperature : 0.8f;
    request.top_p = std::isfinite(top_p) ? top_p : 1.0f;
    request.top_k = std::max<std::int32_t>(0, top_k);
    request.min_p = std::isfinite(min_p) ? min_p : 0.0f;
    request.presence_penalty =
        std::isfinite(presence_penalty) ? presence_penalty : 0.0f;
    request.repetition_penalty =
        std::isfinite(repetition_penalty) ? repetition_penalty : 1.0f;
    request.no_repeat_ngram_size = std::max<std::int32_t>(0, no_repeat_ngram_size);
    request.min_new_tokens_before_eos =
        std::max<std::int32_t>(0, min_new_tokens_before_eos);
    request.seed = seed_enabled != 0 ? seed : 0;
    request.latent = latent != 0;
    request.suspended = suspended != 0;
    state->requests.emplace(id, std::move(request));
    qsg_scheduler_set_request_latent(&state->scheduler, request_id, latent);
    qsg_scheduler_set_request_suspended(&state->scheduler, request_id, suspended);
    state->cv.notify_all();
    return 0;
}

std::int32_t qsg_runtime_cancel(void* handle, const char* request_id) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    finish_runtime_request(state, it->second, true, "cancelled");
    state->cv.notify_all();
    return 0;
}

std::int32_t qsg_runtime_set_request_latent(
    void* handle,
    const char* request_id,
    std::int32_t is_latent) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    it->second.latent = is_latent != 0;
    return qsg_scheduler_set_request_latent(&state->scheduler, request_id, is_latent);
}

std::int32_t qsg_runtime_set_request_suspended(
    void* handle,
    const char* request_id,
    std::int32_t is_suspended) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const auto it = state->requests.find(std::string(request_id));
    if (it == state->requests.end()) {
        return 0;
    }
    it->second.suspended = is_suspended != 0;
    const std::int32_t status =
        qsg_scheduler_set_request_suspended(&state->scheduler, request_id, is_suspended);
    state->cv.notify_all();
    return status;
}

std::int64_t qsg_runtime_first_scheduled_ns(
    const void* handle,
    const char* request_id) {
    const auto* state = reinterpret_cast<const NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return 0;
    }
    return qsg_scheduler_first_scheduled_ns(&state->scheduler, request_id);
}

std::int32_t qsg_runtime_request_state(
    const void* handle,
    const char* request_id) {
    const auto* state = reinterpret_cast<const NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return 0;
    }
    return qsg_scheduler_request_state(&state->scheduler, request_id);
}

std::int32_t qsg_runtime_poll_event(
    void* handle,
    const char* request_id,
    std::int32_t* out_token_id,
    std::int32_t* out_has_token,
    std::int32_t* out_done,
    char* out_error,
    std::int32_t out_error_cap) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (out_token_id != nullptr) {
        *out_token_id = -1;
    }
    if (out_has_token != nullptr) {
        *out_has_token = 0;
    }
    if (out_done != nullptr) {
        *out_done = 0;
    }
    if (out_error != nullptr && out_error_cap > 0) {
        out_error[0] = '\0';
    }
    if (state == nullptr || request_id == nullptr) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    const std::string id(request_id);
    auto it = state->requests.find(id);
    if (it == state->requests.end() || it->second.events.empty()) {
        return 0;
    }
    RuntimeRequest& request = it->second;
    RuntimeEvent event = std::move(request.events.front());
    request.events.pop_front();
    if (out_token_id != nullptr) {
        *out_token_id = event.token_id;
    }
    if (out_has_token != nullptr) {
        *out_has_token = event.has_token ? 1 : 0;
    }
    if (out_done != nullptr) {
        *out_done = event.done ? 1 : 0;
    }
    if (out_error != nullptr && out_error_cap > 0 && !event.error.empty()) {
        const std::size_t length = std::min<std::size_t>(
            static_cast<std::size_t>(out_error_cap - 1),
            event.error.size());
        std::memcpy(out_error, event.error.data(), length);
        out_error[length] = '\0';
    }
    if (event.done && request.events.empty()) {
        destroy_runtime_checkpoint(request);
        state->requests.erase(id);
    }
    return 1;
}

void qsg_runtime_get_metrics(
    const void* handle,
    QSGRuntimeMetricsNative* out_metrics) {
    if (out_metrics == nullptr) {
        return;
    }
    std::memset(out_metrics, 0, sizeof(QSGRuntimeMetricsNative));
    const auto* state = reinterpret_cast<const NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return;
    }
    qsg_scheduler_get_metrics(&state->scheduler, &out_metrics->scheduler);
    out_metrics->worker_iterations = state->worker_iterations;
    out_metrics->emitted_events = state->emitted_events;
    out_metrics->prefill_batches = state->prefill_batches;
    out_metrics->runtime_prefill_tokens = state->runtime_prefill_tokens;
    out_metrics->runtime_decode_steps = state->runtime_decode_steps;
    out_metrics->worker_running = state->worker_running ? 1 : 0;
    out_metrics->native_runtime_abi_ready = 1;
}

void qsg_runtime_run_forever(void* handle) {
    auto* state = reinterpret_cast<NativeRuntimeState*>(handle);
    if (state == nullptr) {
        return;
    }
    if (state->worker.joinable()) {
        state->worker.join();
    }
}

std::int32_t qsg_autoregressive_generate(
    void* model_graph_handle,
    const std::int32_t* prompt_tokens,
    std::int32_t prompt_token_count,
    std::int32_t max_new_tokens,
    std::int32_t vocab_size,
    std::int32_t eos_token,
    float temperature,
    float top_p,
    std::int32_t top_k,
    float min_p,
    float presence_penalty,
    float repetition_penalty,
    std::int32_t no_repeat_ngram_size,
    std::int32_t min_new_tokens_before_eos,
    std::int32_t seed_enabled,
    std::int64_t seed,
    std::int32_t* out_tokens,
    std::int32_t out_capacity,
    std::int32_t* out_token_count,
    std::int32_t* out_stop_reason) {
    if (out_token_count != nullptr) {
        *out_token_count = 0;
    }
    if (out_stop_reason != nullptr) {
        *out_stop_reason = 0;
    }
    if (model_graph_handle == nullptr || out_tokens == nullptr || out_capacity <= 0 ||
        vocab_size <= 0) {
        return 0;
    }
    NativeRuntimeState runtime;
    runtime.graph = model_graph_handle;
    runtime.vocab_size = vocab_size;
    runtime.eos_token = eos_token;
    runtime.ubatch = std::max<std::int32_t>(1, prompt_token_count);
    RuntimeRequest request;
    request.request_id = "autoregressive";
    if (prompt_tokens != nullptr && prompt_token_count > 0) {
        request.prompt_tokens.assign(prompt_tokens, prompt_tokens + prompt_token_count);
    }
    request.max_new_tokens = std::max<std::int32_t>(0, max_new_tokens);
    request.temperature = std::max(temperature, 1.0e-6f);
    request.top_p = std::max(0.0f, std::min(1.0f, top_p));
    request.top_k = std::max<std::int32_t>(0, top_k);
    request.min_p = std::max(0.0f, std::min(1.0f, min_p));
    request.presence_penalty = presence_penalty;
    request.repetition_penalty = repetition_penalty;
    request.no_repeat_ngram_size = std::max<std::int32_t>(0, no_repeat_ngram_size);
    request.min_new_tokens_before_eos = std::max<std::int32_t>(0, min_new_tokens_before_eos);
    request.seed = seed_enabled != 0 ? seed : 0;
    if (prompt_tokens != nullptr && prompt_token_count > 0) {
        request.token_history.assign(prompt_tokens, prompt_tokens + prompt_token_count);
    }
    std::string error;
    if (!prefill_runtime_request(&runtime, request, error)) {
        destroy_runtime_checkpoint(request);
        return 0;
    }
    const std::int32_t limit =
        std::min<std::int32_t>(std::max<std::int32_t>(0, request.max_new_tokens), out_capacity);
    for (std::int32_t idx = 0; idx < limit; ++idx) {
        if (!request.seed_applied && request.seed != 0) {
            simd_seed_rng_f32(static_cast<int>(request.seed));
            request.seed_applied = true;
        }
        std::vector<int> suppressed_ids;
        if (request.generated_tokens < request.min_new_tokens_before_eos &&
            eos_token >= 0) {
            suppressed_ids.push_back(eos_token);
        }
        const std::int32_t token_id = static_cast<std::int32_t>(simd_qsg_postprocess_sample_f32(
            request.logits.empty() ? nullptr : request.logits.data(),
            static_cast<int>(request.logits.size()),
            suppressed_ids.empty() ? nullptr : suppressed_ids.data(),
            static_cast<int>(suppressed_ids.size()),
            request.token_history.empty() ? nullptr : request.token_history.data(),
            static_cast<int>(request.token_history.size()),
            nullptr,
            0,
            0,
            0,
            0.0f,
            0,
            0,
            0.0f,
            request.presence_penalty,
            request.repetition_penalty,
            request.no_repeat_ngram_size,
            std::max(1.0e-6f, request.temperature),
            eos_token,
            request.top_p,
            request.top_k,
            request.min_p));
        out_tokens[idx] = token_id;
        request.token_history.push_back(token_id);
        request.generated_tokens += 1;
        if (out_token_count != nullptr) {
            *out_token_count = idx + 1;
        }
        const bool reached_eos = eos_token >= 0 && token_id == eos_token;
        const bool reached_limit = request.generated_tokens >= request.max_new_tokens;
        if (reached_eos || reached_limit) {
            if (out_stop_reason != nullptr) {
                *out_stop_reason = reached_eos ? 1 : 2;
            }
            destroy_runtime_checkpoint(request);
            return 1;
        }
        if (!advance_runtime_request(&runtime, request, token_id, error)) {
            destroy_runtime_checkpoint(request);
            return 0;
        }
    }
    destroy_runtime_checkpoint(request);
    if (out_stop_reason != nullptr && limit > 0) {
        *out_stop_reason = 2;
    }
    return 1;
}

std::int32_t qsg_verify_draft_tokens(
    void* model_graph_handle,
    const std::int32_t* prompt_tokens,
    std::int32_t prompt_token_count,
    const std::int32_t* draft_tokens,
    std::int32_t draft_token_count,
    std::int32_t generated_prefix_count,
    std::int32_t vocab_size,
    std::int32_t eos_token,
    float temperature,
    float top_p,
    std::int32_t top_k,
    float min_p,
    float presence_penalty,
    float repetition_penalty,
    std::int32_t no_repeat_ngram_size,
    std::int32_t min_new_tokens_before_eos,
    float min_accept_probability,
    std::int32_t sample_recovery_token,
    float* out_probs,
    std::int32_t out_prob_capacity,
    std::int32_t* out_prob_count,
    std::int32_t* out_accepted_count,
    std::int32_t* out_recovery_token,
    std::int32_t* out_stop_reason) {
    if (out_prob_count != nullptr) {
        *out_prob_count = 0;
    }
    if (out_accepted_count != nullptr) {
        *out_accepted_count = 0;
    }
    if (out_recovery_token != nullptr) {
        *out_recovery_token = -1;
    }
    if (out_stop_reason != nullptr) {
        *out_stop_reason = 0;
    }
    if (model_graph_handle == nullptr || vocab_size <= 0 || draft_token_count < 0 ||
        generated_prefix_count < 0) {
        return 0;
    }
    if (draft_token_count == 0) {
        return 1;
    }
    NativeRuntimeState runtime;
    runtime.graph = model_graph_handle;
    runtime.vocab_size = vocab_size;
    runtime.eos_token = eos_token;
    runtime.ubatch = std::max<std::int32_t>(1, prompt_token_count);
    RuntimeRequest request;
    request.request_id = "draft_verify";
    if (prompt_tokens != nullptr && prompt_token_count > 0) {
        request.prompt_tokens.assign(prompt_tokens, prompt_tokens + prompt_token_count);
        request.token_history.assign(prompt_tokens, prompt_tokens + prompt_token_count);
    }
    request.generated_tokens = std::max<std::int32_t>(0, generated_prefix_count);
    request.temperature = std::max(temperature, 1.0e-6f);
    request.top_p = std::max(0.0f, std::min(1.0f, top_p));
    request.top_k = std::max<std::int32_t>(0, top_k);
    request.min_p = std::max(0.0f, std::min(1.0f, min_p));
    request.presence_penalty = presence_penalty;
    request.repetition_penalty = repetition_penalty;
    request.no_repeat_ngram_size = std::max<std::int32_t>(0, no_repeat_ngram_size);
    request.min_new_tokens_before_eos = std::max<std::int32_t>(0, min_new_tokens_before_eos);
    std::string error;
    if (!prefill_runtime_request(&runtime, request, error)) {
        destroy_runtime_checkpoint(request);
        return 0;
    }

    std::int32_t accepted = 0;
    std::int32_t prob_count = 0;
    const std::int32_t prob_capacity = std::max<std::int32_t>(0, out_prob_capacity);
    std::vector<int> suppressed_ids;
    for (std::int32_t idx = 0; idx < draft_token_count; ++idx) {
        if (request.logits.empty()) {
            destroy_runtime_checkpoint(request);
            return 0;
        }
        const std::int32_t token_id = draft_tokens[idx];
        std::vector<float> working_logits = request.logits;
        build_runtime_suppressed_ids(runtime, request, suppressed_ids);
        int greedy_token = 0;
        float token_prob = 0.0f;
        if (simd_qsg_postprocess_score_token_f32(
                working_logits.data(),
                static_cast<int>(working_logits.size()),
                suppressed_ids.empty() ? nullptr : suppressed_ids.data(),
                static_cast<int>(suppressed_ids.size()),
                request.token_history.empty() ? nullptr : request.token_history.data(),
                static_cast<int>(request.token_history.size()),
                0,
                0,
                0.0f,
                0,
                0,
                0.0f,
                request.presence_penalty,
                request.repetition_penalty,
                request.no_repeat_ngram_size,
                std::max(1.0e-6f, request.temperature),
                runtime.eos_token,
                request.top_p,
                request.top_k,
                request.min_p,
                token_id,
                &greedy_token,
                &token_prob) != 1) {
            destroy_runtime_checkpoint(request);
            return 0;
        }
        if (out_probs != nullptr && prob_count < prob_capacity) {
            out_probs[prob_count] = token_prob;
        }
        prob_count += 1;
        if (greedy_token != token_id &&
            token_prob < std::max(0.0f, std::min(1.0f, min_accept_probability))) {
            break;
        }
        accepted += 1;
        request.token_history.push_back(token_id);
        request.generated_tokens += 1;
        if (runtime.eos_token >= 0 && token_id == runtime.eos_token) {
            if (out_stop_reason != nullptr) {
                *out_stop_reason = 1;
            }
            if (out_prob_count != nullptr) {
                *out_prob_count = prob_count;
            }
            if (out_accepted_count != nullptr) {
                *out_accepted_count = accepted;
            }
            destroy_runtime_checkpoint(request);
            return 1;
        }
        if (!advance_runtime_request(&runtime, request, token_id, error)) {
            destroy_runtime_checkpoint(request);
            return 0;
        }
    }
    if (accepted >= draft_token_count && out_stop_reason != nullptr) {
        *out_stop_reason = 2;
    }
    if (sample_recovery_token != 0 &&
        accepted < draft_token_count &&
        out_recovery_token != nullptr) {
        std::int32_t sampled_token = -1;
        if (sample_runtime_recovery_token(runtime, request, sampled_token)) {
            *out_recovery_token = sampled_token;
        }
    }
    if (out_prob_count != nullptr) {
        *out_prob_count = prob_count;
    }
    if (out_accepted_count != nullptr) {
        *out_accepted_count = accepted;
    }
    destroy_runtime_checkpoint(request);
    return 1;
}

std::int32_t qsg_prompt_lookup_draft(
    const std::int32_t* prompt_tokens,
    std::int32_t n_tokens,
    std::int32_t min_ngram,
    std::int32_t max_ngram,
    std::int32_t max_draft_tokens,
    std::int32_t* out_tokens,
    std::int32_t out_capacity) {
    if (prompt_tokens == nullptr || out_tokens == nullptr || n_tokens <= 0 ||
        max_draft_tokens <= 0 || out_capacity <= 0) {
        return 0;
    }
    const std::int32_t min_window = std::max<std::int32_t>(1, min_ngram);
    if (n_tokens < (min_window * 2)) {
        return 0;
    }
    const std::int32_t max_window =
        std::min<std::int32_t>(std::max(min_window, max_ngram), n_tokens - 1);
    const std::int32_t max_out = std::min(out_capacity, max_draft_tokens);
    for (std::int32_t window = max_window; window >= min_window; --window) {
        const std::int32_t suffix_start = n_tokens - window;
        std::int32_t best_follow_start = -1;
        const std::int32_t limit = n_tokens - window;
        for (std::int32_t start = 0; start < limit; ++start) {
            bool matches = true;
            for (std::int32_t i = 0; i < window; ++i) {
                if (prompt_tokens[start + i] != prompt_tokens[suffix_start + i]) {
                    matches = false;
                    break;
                }
            }
            if (!matches) {
                continue;
            }
            const std::int32_t follow_start = start + window;
            if (follow_start >= n_tokens) {
                continue;
            }
            best_follow_start = follow_start;
        }
        if (best_follow_start >= 0) {
            const std::int32_t available = n_tokens - best_follow_start;
            const std::int32_t count = std::min<std::int32_t>(max_out, available);
            for (std::int32_t i = 0; i < count; ++i) {
                out_tokens[i] = prompt_tokens[best_follow_start + i];
            }
            return count;
        }
    }
    return 0;
}

std::int32_t qsg_masked_diffusion_draft(
    const float* logits,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    std::int32_t mask_stride,
    float temperature,
    std::int32_t top_k,
    float min_probability,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs,
    std::int32_t* out_positions) {
    if (!logits_buffer_valid(logits, vocab_size) || draft_tokens <= 0 ||
        out_tokens == nullptr || out_probs == nullptr || out_positions == nullptr) {
        return 0;
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<float> working(logits, logits + vocab_size);
    std::vector<float> probs(static_cast<std::size_t>(vocab_size), 0.0f);
    const std::int32_t max_steps = std::max<std::int32_t>(1, draft_tokens);
    const std::int32_t stride = std::max<std::int32_t>(1, mask_stride);
    const float min_prob = std::max(0.0f, std::min(1.0f, min_probability));
    std::int32_t produced = 0;
    for (std::int32_t step = 0; step < max_steps; ++step) {
        std::vector<float> step_logits = working;
        apply_temperature_and_topk(step_logits, temperature, top_k);
        highnoon::ops::speculative_softmax(
            step_logits.data(),
            probs.data(),
            static_cast<std::int64_t>(vocab_size));
        const std::int32_t token = highnoon::ops::speculative_sample_token(
            probs.data(), static_cast<std::int64_t>(vocab_size), rng);
        const float token_prob = probs[static_cast<std::size_t>(token)];
        if (token_prob < min_prob && produced > 0) {
            break;
        }
        out_tokens[produced] = token;
        out_probs[produced] = token_prob;
        out_positions[produced] = step * stride;
        produced += 1;
        working[static_cast<std::size_t>(token)] -= 0.10f * static_cast<float>(step + 1);
    }
    return produced;
}

std::int32_t qsg_block_diffusion_draft(
    const float* logits,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    float temperature,
    std::int32_t top_k,
    float min_probability,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs) {
    if (!logits_buffer_valid(logits, vocab_size) || draft_tokens <= 0 ||
        out_tokens == nullptr || out_probs == nullptr) {
        return 0;
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<float> working(logits, logits + vocab_size);
    std::vector<float> probs(static_cast<std::size_t>(vocab_size), 0.0f);
    const std::int32_t max_steps = std::max<std::int32_t>(1, draft_tokens);
    const float min_prob = std::max(0.0f, std::min(1.0f, min_probability));
    std::int32_t produced = 0;
    for (std::int32_t step = 0; step < max_steps; ++step) {
        std::vector<float> step_logits = working;
        apply_temperature_and_topk(step_logits, temperature, top_k);
        highnoon::ops::speculative_softmax(
            step_logits.data(),
            probs.data(),
            static_cast<std::int64_t>(vocab_size));
        const std::int32_t token = highnoon::ops::speculative_sample_token(
            probs.data(), static_cast<std::int64_t>(vocab_size), rng);
        const float token_prob = probs[static_cast<std::size_t>(token)];
        if (token_prob < min_prob && produced > 0) {
            break;
        }
        out_tokens[produced] = token;
        out_probs[produced] = token_prob;
        produced += 1;
        working[static_cast<std::size_t>(token)] -= 0.15f * static_cast<float>(step + 1);
    }
    return produced;
}

std::int32_t qsg_eagle_replacement_draft(
    const float* draft_logits,
    const float* target_logits,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    float temperature,
    std::int32_t max_tree_width,
    float acceptance_threshold,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs) {
    if (!logits_buffer_valid(draft_logits, vocab_size) ||
        !logits_buffer_valid(target_logits, vocab_size) ||
        draft_tokens <= 0 || out_tokens == nullptr || out_probs == nullptr) {
        return 0;
    }
    const std::int32_t depth = std::max<std::int32_t>(1, draft_tokens);
    std::vector<float> draft_matrix(static_cast<std::size_t>(depth) * vocab_size, 0.0f);
    std::vector<float> target_probs(static_cast<std::size_t>(depth) * vocab_size, 0.0f);
    std::vector<float> probs(static_cast<std::size_t>(vocab_size), 0.0f);

    for (std::int32_t d = 0; d < depth; ++d) {
        std::vector<float> draft_step(draft_logits, draft_logits + vocab_size);
        std::vector<float> target_step(target_logits, target_logits + vocab_size);
        for (std::int32_t v = 0; v < vocab_size; ++v) {
            const float decay = 0.03f * static_cast<float>(d);
            draft_step[static_cast<std::size_t>(v)] -= decay;
            target_step[static_cast<std::size_t>(v)] += decay * 0.5f;
        }
        apply_temperature_and_topk(draft_step, temperature, max_tree_width * 2);
        apply_temperature_and_topk(target_step, std::max(0.8f, temperature * 0.9f), max_tree_width * 2);
        std::memcpy(
            draft_matrix.data() + static_cast<std::size_t>(d) * vocab_size,
            draft_step.data(),
            static_cast<std::size_t>(vocab_size) * sizeof(float));
        highnoon::ops::speculative_softmax(
            target_step.data(),
            probs.data(),
            static_cast<std::int64_t>(vocab_size));
        std::memcpy(
            target_probs.data() + static_cast<std::size_t>(d) * vocab_size,
            probs.data(),
            static_cast<std::size_t>(vocab_size) * sizeof(float));
    }

    highnoon::ops::EAGLEConfig config;
    config.draft_depth = depth;
    config.max_tree_width = std::max<std::int32_t>(1, max_tree_width);
    config.temperature = std::max(1.0e-6f, temperature);
    config.acceptance_threshold = std::max(0.0f, std::min(1.0f, acceptance_threshold));
    config.use_dynamic_tree = true;

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<highnoon::ops::DraftTreeNode> tree;
    highnoon::ops::eagle_build_dynamic_tree(
        draft_matrix.data(),
        config,
        static_cast<std::int64_t>(vocab_size),
        tree,
        rng);
    const int accepted = highnoon::ops::eagle_verify_tree(
        target_probs.data(),
        tree,
        config,
        static_cast<std::int64_t>(vocab_size),
        rng);
    if (accepted <= 0 || tree.empty()) {
        return 0;
    }

    std::int32_t produced = 0;
    for (std::int32_t depth_idx = 0; depth_idx < accepted; ++depth_idx) {
        float best_prob = -std::numeric_limits<float>::infinity();
        std::int32_t best_token = -1;
        for (const auto& node : tree) {
            if (!node.accepted || node.depth != depth_idx) {
                continue;
            }
            if (node.prob > best_prob) {
                best_prob = node.prob;
                best_token = node.token;
            }
        }
        if (best_token < 0) {
            break;
        }
        out_tokens[produced] = best_token;
        out_probs[produced] = std::max(0.0f, best_prob);
        produced += 1;
    }
    return produced;
}

std::int32_t qsg_medusa_head_draft(
    const float* hidden,
    const float* head_weights,
    const float* head_bias,
    std::int32_t num_heads,
    std::int32_t hidden_dim,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    float temperature,
    std::int32_t top_k,
    float min_probability,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs) {
    if (hidden == nullptr || head_weights == nullptr || vocab_size <= 1 ||
        hidden_dim <= 0 || num_heads <= 0 || draft_tokens <= 0 ||
        out_tokens == nullptr || out_probs == nullptr) {
        return 0;
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<float> logits;
    std::vector<float> probs(static_cast<std::size_t>(vocab_size), 0.0f);
    const std::int32_t max_steps =
        std::min(std::max<std::int32_t>(1, draft_tokens), num_heads);
    const float min_prob = std::max(0.0f, std::min(1.0f, min_probability));
    std::int32_t produced = 0;
    const std::size_t head_stride =
        static_cast<std::size_t>(hidden_dim) * static_cast<std::size_t>(vocab_size);
    for (std::int32_t step = 0; step < max_steps; ++step) {
        const float* weights = head_weights + static_cast<std::size_t>(step) * head_stride;
        const float* bias = head_bias != nullptr
            ? head_bias + static_cast<std::size_t>(step) * vocab_size
            : nullptr;
        project_head_logits(hidden, weights, bias, hidden_dim, vocab_size, logits);
        apply_temperature_and_topk(logits, temperature, top_k);
        highnoon::ops::speculative_softmax(
            logits.data(),
            probs.data(),
            static_cast<std::int64_t>(vocab_size));
        const std::int32_t token = highnoon::ops::speculative_sample_token(
            probs.data(),
            static_cast<std::int64_t>(vocab_size),
            rng);
        const float token_prob = probs[static_cast<std::size_t>(token)];
        if (token_prob < min_prob && produced > 0) {
            break;
        }
        out_tokens[produced] = token;
        out_probs[produced] = token_prob;
        produced += 1;
    }
    return produced;
}

std::int32_t qsg_hydra_head_draft(
    const float* hidden,
    const float* base_logits,
    const float* head_weights,
    const float* head_bias,
    std::int32_t num_heads,
    std::int32_t hidden_dim,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    float temperature,
    std::int32_t top_k,
    float blend_alpha,
    float min_probability,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs) {
    if (hidden == nullptr || base_logits == nullptr || head_weights == nullptr ||
        vocab_size <= 1 || hidden_dim <= 0 || num_heads <= 0 || draft_tokens <= 0 ||
        out_tokens == nullptr || out_probs == nullptr) {
        return 0;
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<float> logits;
    std::vector<float> probs(static_cast<std::size_t>(vocab_size), 0.0f);
    const std::int32_t max_steps =
        std::min(std::max<std::int32_t>(1, draft_tokens), num_heads);
    const float min_prob = std::max(0.0f, std::min(1.0f, min_probability));
    const float alpha = std::max(0.0f, std::min(1.0f, blend_alpha));
    const float base_scale = 1.0f - alpha;
    const std::size_t head_stride =
        static_cast<std::size_t>(hidden_dim) * static_cast<std::size_t>(vocab_size);
    std::int32_t produced = 0;
    for (std::int32_t step = 0; step < max_steps; ++step) {
        const float* weights = head_weights + static_cast<std::size_t>(step) * head_stride;
        const float* bias = head_bias != nullptr
            ? head_bias + static_cast<std::size_t>(step) * vocab_size
            : nullptr;
        project_head_logits(hidden, weights, bias, hidden_dim, vocab_size, logits);
        const float depth_decay = 1.0f - (0.08f * static_cast<float>(step));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int32_t vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx) {
            logits[static_cast<std::size_t>(vocab_idx)] =
                (alpha * logits[static_cast<std::size_t>(vocab_idx)] * depth_decay) +
                (base_scale * base_logits[vocab_idx]);
        }
        apply_temperature_and_topk(logits, temperature, top_k);
        highnoon::ops::speculative_softmax(
            logits.data(),
            probs.data(),
            static_cast<std::int64_t>(vocab_size));
        const std::int32_t token = highnoon::ops::speculative_sample_token(
            probs.data(),
            static_cast<std::int64_t>(vocab_size),
            rng);
        const float token_prob = probs[static_cast<std::size_t>(token)];
        if (token_prob < min_prob && produced > 0) {
            break;
        }
        out_tokens[produced] = token;
        out_probs[produced] = token_prob;
        produced += 1;
    }
    return produced;
}

}  // extern "C"
