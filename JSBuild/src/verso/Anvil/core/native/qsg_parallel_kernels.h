#pragma once

#include <cstdint>

struct QSGSchedulerMetricsNative {
    std::int32_t queue_depth;
    std::int32_t active_requests;
    std::int32_t inflight_requests;
    std::int32_t prefill_active_requests;
    std::int32_t decode_active_requests;
    std::int64_t admitted_requests;
    std::int64_t completed_requests;
    std::int64_t cancelled_requests;
    std::int64_t evicted_requests;
    std::int64_t iterations;
    std::int64_t prefill_request_count;
    std::int64_t prefill_tokens_scheduled;
    std::int64_t decode_tokens_emitted;
    std::int64_t chunked_prefill_requests;
    std::int64_t chunked_prefill_chunks;
    std::int64_t latent_requests;
    std::int64_t suspended_requests;
    double iteration_last_ms;
    double iteration_avg_ms;
    double iteration_p95_ms;
    double queue_wait_p50_ms;
    double queue_wait_p95_ms;
    double queue_wait_p99_ms;
};

struct QSGRuntimeMetricsNative {
    QSGSchedulerMetricsNative scheduler;
    std::int64_t worker_iterations;
    std::int64_t emitted_events;
    std::int64_t prefill_batches;
    std::int64_t runtime_prefill_tokens;
    std::int64_t runtime_decode_steps;
    std::int32_t worker_running;
    std::int32_t native_runtime_abi_ready;
};

extern "C" {

void* qsg_scheduler_create(
    std::int32_t max_active_requests,
    std::int32_t max_pending_requests,
    std::int32_t scheduler_policy_priority,
    std::int32_t interleaved_streams);

void qsg_scheduler_destroy(void* handle);

std::int32_t qsg_scheduler_submit(
    void* handle,
    const char* request_id,
    std::int32_t priority,
    std::int64_t arrival_ts_ns,
    std::int32_t prompt_token_count,
    std::int32_t max_new_tokens,
    std::int32_t prefill_chunk_size);

std::int32_t qsg_scheduler_cancel(void* handle, const char* request_id);

std::int32_t qsg_scheduler_complete(
    void* handle,
    const char* request_id,
    std::int32_t cancelled);

void qsg_scheduler_promote(void* handle);

std::int32_t qsg_scheduler_active_count(const void* handle);

std::int32_t qsg_scheduler_copy_active_id(
    const void* handle,
    std::int32_t idx,
    char* out,
    std::int32_t out_cap);

void qsg_scheduler_rotate_active(void* handle);

std::int64_t qsg_scheduler_first_scheduled_ns(
    const void* handle,
    const char* request_id);

std::int32_t qsg_scheduler_request_state(
    const void* handle,
    const char* request_id);

std::int32_t qsg_scheduler_set_request_latent(
    void* handle,
    const char* request_id,
    std::int32_t is_latent);

std::int32_t qsg_scheduler_set_request_suspended(
    void* handle,
    const char* request_id,
    std::int32_t is_suspended);

void qsg_scheduler_record_iteration(void* handle, double iteration_ms);

void qsg_scheduler_record_decode_emit(
    void* handle,
    const char* request_id,
    std::int32_t emitted_tokens);

void qsg_scheduler_get_metrics(
    const void* handle,
    QSGSchedulerMetricsNative* out_metrics);

void* qsg_runtime_create(
    void* model_graph_handle,
    std::int32_t vocab_size,
    std::int32_t eos_token,
    std::int32_t ubatch,
    std::int32_t max_active_requests,
    std::int32_t max_pending_requests,
    std::int32_t scheduler_policy_priority,
    std::int32_t interleaved_streams);

void qsg_runtime_destroy(void* handle);

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
    std::int32_t suspended);

std::int32_t qsg_runtime_cancel(void* handle, const char* request_id);

std::int32_t qsg_runtime_set_request_latent(
    void* handle,
    const char* request_id,
    std::int32_t is_latent);

std::int32_t qsg_runtime_set_request_suspended(
    void* handle,
    const char* request_id,
    std::int32_t is_suspended);

std::int64_t qsg_runtime_first_scheduled_ns(
    const void* handle,
    const char* request_id);

std::int32_t qsg_runtime_request_state(
    const void* handle,
    const char* request_id);

std::int32_t qsg_runtime_poll_event(
    void* handle,
    const char* request_id,
    std::int32_t* out_token_id,
    std::int32_t* out_has_token,
    std::int32_t* out_done,
    char* out_error,
    std::int32_t out_error_cap);

void qsg_runtime_get_metrics(
    const void* handle,
    QSGRuntimeMetricsNative* out_metrics);

void qsg_runtime_shutdown(void* handle);

void qsg_runtime_run_forever(void* handle);

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
    std::int32_t* out_stop_reason);

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
    std::int32_t* out_stop_reason);

std::int32_t qsg_prompt_lookup_draft(
    const std::int32_t* prompt_tokens,
    std::int32_t n_tokens,
    std::int32_t min_ngram,
    std::int32_t max_ngram,
    std::int32_t max_draft_tokens,
    std::int32_t* out_tokens,
    std::int32_t out_capacity);

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
    std::int32_t* out_positions);

std::int32_t qsg_block_diffusion_draft(
    const float* logits,
    std::int32_t vocab_size,
    std::int32_t draft_tokens,
    float temperature,
    std::int32_t top_k,
    float min_probability,
    std::int64_t seed,
    std::int32_t* out_tokens,
    float* out_probs);

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
    float* out_probs);

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
    float* out_probs);

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
    float* out_probs);

}
