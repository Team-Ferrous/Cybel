from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntFlag
import threading
from typing import Any

import numpy as np

from core.native.native_ops import load_native_library

_LIB: ctypes.CDLL | None = None
_LOCK = threading.RLock()
_INT32_P = ctypes.POINTER(ctypes.c_int32)
_FLOAT_P = ctypes.POINTER(ctypes.c_float)


class _SchedulerMetrics(ctypes.Structure):
    _fields_ = [
        ("queue_depth", ctypes.c_int32),
        ("active_requests", ctypes.c_int32),
        ("inflight_requests", ctypes.c_int32),
        ("prefill_active_requests", ctypes.c_int32),
        ("decode_active_requests", ctypes.c_int32),
        ("admitted_requests", ctypes.c_int64),
        ("completed_requests", ctypes.c_int64),
        ("cancelled_requests", ctypes.c_int64),
        ("evicted_requests", ctypes.c_int64),
        ("iterations", ctypes.c_int64),
        ("prefill_request_count", ctypes.c_int64),
        ("prefill_tokens_scheduled", ctypes.c_int64),
        ("decode_tokens_emitted", ctypes.c_int64),
        ("chunked_prefill_requests", ctypes.c_int64),
        ("chunked_prefill_chunks", ctypes.c_int64),
        ("latent_requests", ctypes.c_int64),
        ("suspended_requests", ctypes.c_int64),
        ("iteration_last_ms", ctypes.c_double),
        ("iteration_avg_ms", ctypes.c_double),
        ("iteration_p95_ms", ctypes.c_double),
        ("queue_wait_p50_ms", ctypes.c_double),
        ("queue_wait_p95_ms", ctypes.c_double),
        ("queue_wait_p99_ms", ctypes.c_double),
    ]


class _RuntimeMetrics(ctypes.Structure):
    _fields_ = [
        ("scheduler", _SchedulerMetrics),
        ("worker_iterations", ctypes.c_int64),
        ("emitted_events", ctypes.c_int64),
        ("prefill_batches", ctypes.c_int64),
        ("runtime_prefill_tokens", ctypes.c_int64),
        ("runtime_decode_steps", ctypes.c_int64),
        ("worker_running", ctypes.c_int32),
        ("native_runtime_abi_ready", ctypes.c_int32),
    ]


def _lib() -> ctypes.CDLL:
    global _LIB
    with _LOCK:
        if _LIB is not None:
            return _LIB
        lib = load_native_library()
        required = (
            "qsg_scheduler_create",
            "qsg_scheduler_destroy",
            "qsg_scheduler_submit",
            "qsg_scheduler_cancel",
            "qsg_scheduler_complete",
            "qsg_scheduler_promote",
            "qsg_scheduler_active_count",
            "qsg_scheduler_copy_active_id",
            "qsg_scheduler_rotate_active",
            "qsg_scheduler_first_scheduled_ns",
            "qsg_scheduler_request_state",
            "qsg_scheduler_set_request_latent",
            "qsg_scheduler_set_request_suspended",
            "qsg_scheduler_record_iteration",
            "qsg_scheduler_record_decode_emit",
            "qsg_scheduler_get_metrics",
            "qsg_runtime_create",
            "qsg_runtime_destroy",
            "qsg_runtime_submit",
            "qsg_runtime_cancel",
            "qsg_runtime_set_request_latent",
            "qsg_runtime_set_request_suspended",
            "qsg_runtime_first_scheduled_ns",
            "qsg_runtime_request_state",
            "qsg_runtime_poll_event",
            "qsg_runtime_get_metrics",
            "qsg_runtime_shutdown",
            "qsg_runtime_run_forever",
            "qsg_autoregressive_generate",
            "qsg_verify_draft_tokens",
            "qsg_prompt_lookup_draft",
            "qsg_masked_diffusion_draft",
            "qsg_block_diffusion_draft",
            "qsg_eagle_replacement_draft",
            "qsg_medusa_head_draft",
            "qsg_hydra_head_draft",
        )
        missing = [symbol for symbol in required if not hasattr(lib, symbol)]
        if missing:
            raise RuntimeError(
                "Native-only parallel kernels require symbols: " + ", ".join(missing)
            )

        lib.qsg_scheduler_create.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_create.restype = ctypes.c_void_p

        lib.qsg_scheduler_destroy.argtypes = [ctypes.c_void_p]
        lib.qsg_scheduler_destroy.restype = None

        lib.qsg_scheduler_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_submit.restype = ctypes.c_int32

        lib.qsg_scheduler_cancel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.qsg_scheduler_cancel.restype = ctypes.c_int32

        lib.qsg_scheduler_complete.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_complete.restype = ctypes.c_int32

        lib.qsg_scheduler_promote.argtypes = [ctypes.c_void_p]
        lib.qsg_scheduler_promote.restype = None

        lib.qsg_scheduler_active_count.argtypes = [ctypes.c_void_p]
        lib.qsg_scheduler_active_count.restype = ctypes.c_int32

        lib.qsg_scheduler_copy_active_id.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_copy_active_id.restype = ctypes.c_int32

        lib.qsg_scheduler_rotate_active.argtypes = [ctypes.c_void_p]
        lib.qsg_scheduler_rotate_active.restype = None

        lib.qsg_scheduler_first_scheduled_ns.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.qsg_scheduler_first_scheduled_ns.restype = ctypes.c_int64

        lib.qsg_scheduler_request_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.qsg_scheduler_request_state.restype = ctypes.c_int32

        lib.qsg_scheduler_set_request_latent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_set_request_latent.restype = ctypes.c_int32

        lib.qsg_scheduler_set_request_suspended.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_set_request_suspended.restype = ctypes.c_int32

        lib.qsg_scheduler_record_iteration.argtypes = [ctypes.c_void_p, ctypes.c_double]
        lib.qsg_scheduler_record_iteration.restype = None

        lib.qsg_scheduler_record_decode_emit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_scheduler_record_decode_emit.restype = None

        lib.qsg_scheduler_get_metrics.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_SchedulerMetrics),
        ]
        lib.qsg_scheduler_get_metrics.restype = None

        lib.qsg_runtime_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.qsg_runtime_create.restype = ctypes.c_void_p

        lib.qsg_runtime_destroy.argtypes = [ctypes.c_void_p]
        lib.qsg_runtime_destroy.restype = None

        lib.qsg_runtime_submit.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int64,
            _INT32_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        lib.qsg_runtime_submit.restype = ctypes.c_int32

        lib.qsg_runtime_cancel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.qsg_runtime_cancel.restype = ctypes.c_int32

        lib.qsg_runtime_set_request_latent.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_runtime_set_request_latent.restype = ctypes.c_int32

        lib.qsg_runtime_set_request_suspended.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_runtime_set_request_suspended.restype = ctypes.c_int32

        lib.qsg_runtime_first_scheduled_ns.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.qsg_runtime_first_scheduled_ns.restype = ctypes.c_int64

        lib.qsg_runtime_request_state.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.qsg_runtime_request_state.restype = ctypes.c_int32

        lib.qsg_runtime_poll_event.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p,
            ctypes.c_int32,
        ]
        lib.qsg_runtime_poll_event.restype = ctypes.c_int32

        lib.qsg_runtime_get_metrics.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_RuntimeMetrics),
        ]
        lib.qsg_runtime_get_metrics.restype = None

        lib.qsg_runtime_shutdown.argtypes = [ctypes.c_void_p]
        lib.qsg_runtime_shutdown.restype = None

        lib.qsg_runtime_run_forever.argtypes = [ctypes.c_void_p]
        lib.qsg_runtime_run_forever.restype = None

        lib.qsg_autoregressive_generate.argtypes = [
            ctypes.c_void_p,
            _INT32_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int64,
            _INT32_P,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.qsg_autoregressive_generate.restype = ctypes.c_int32

        lib.qsg_verify_draft_tokens.argtypes = [
            ctypes.c_void_p,
            _INT32_P,
            ctypes.c_int32,
            _INT32_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        lib.qsg_verify_draft_tokens.restype = ctypes.c_int32

        lib.qsg_prompt_lookup_draft.argtypes = [
            _INT32_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            _INT32_P,
            ctypes.c_int32,
        ]
        lib.qsg_prompt_lookup_draft.restype = ctypes.c_int32

        lib.qsg_masked_diffusion_draft.argtypes = [
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int64,
            _INT32_P,
            _FLOAT_P,
            _INT32_P,
        ]
        lib.qsg_masked_diffusion_draft.restype = ctypes.c_int32

        lib.qsg_block_diffusion_draft.argtypes = [
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int64,
            _INT32_P,
            _FLOAT_P,
        ]
        lib.qsg_block_diffusion_draft.restype = ctypes.c_int32

        lib.qsg_eagle_replacement_draft.argtypes = [
            _FLOAT_P,
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int64,
            _INT32_P,
            _FLOAT_P,
        ]
        lib.qsg_eagle_replacement_draft.restype = ctypes.c_int32

        lib.qsg_medusa_head_draft.argtypes = [
            _FLOAT_P,
            _FLOAT_P,
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int64,
            _INT32_P,
            _FLOAT_P,
        ]
        lib.qsg_medusa_head_draft.restype = ctypes.c_int32

        lib.qsg_hydra_head_draft.argtypes = [
            _FLOAT_P,
            _FLOAT_P,
            _FLOAT_P,
            _FLOAT_P,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int64,
            _INT32_P,
            _FLOAT_P,
        ]
        lib.qsg_hydra_head_draft.restype = ctypes.c_int32
        _LIB = lib
        return lib


def native_parallel_kernels_available() -> bool:
    try:
        _lib()
    except Exception:
        return False
    return True


def _as_i32_contiguous(values: list[int] | Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("token inputs must be 1D")
    return np.ascontiguousarray(arr)


def _as_f32_contiguous(values: list[float] | Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("logits input must be 1D")
    return np.ascontiguousarray(arr)


def qsg_prompt_lookup_draft(
    prompt_tokens: list[int] | Any,
    *,
    min_ngram: int,
    max_ngram: int,
    max_draft_tokens: int,
) -> list[int]:
    tokens = _as_i32_contiguous(prompt_tokens)
    max_draft = max(0, int(max_draft_tokens))
    if tokens.size == 0 or max_draft <= 0:
        return []
    out = np.zeros((max_draft,), dtype=np.int32)
    count = int(
        _lib().qsg_prompt_lookup_draft(
            tokens.ctypes.data_as(_INT32_P),
            int(tokens.shape[0]),
            int(min_ngram),
            int(max_ngram),
            max_draft,
            out.ctypes.data_as(_INT32_P),
            int(out.shape[0]),
        )
    )
    if count <= 0:
        return []
    return [int(token) for token in out[:count].tolist()]


def qsg_block_diffusion_draft(
    logits: list[float] | Any,
    *,
    draft_tokens: int,
    temperature: float,
    top_k: int,
    min_probability: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    logits_arr = _as_f32_contiguous(logits)
    target = max(0, int(draft_tokens))
    if logits_arr.size == 0 or target <= 0:
        return [], []
    out_tokens = np.zeros((target,), dtype=np.int32)
    out_probs = np.zeros((target,), dtype=np.float32)
    count = int(
        _lib().qsg_block_diffusion_draft(
            logits_arr.ctypes.data_as(_FLOAT_P),
            int(logits_arr.shape[0]),
            target,
            float(temperature),
            int(top_k),
            float(min_probability),
            int(seed),
            out_tokens.ctypes.data_as(_INT32_P),
            out_probs.ctypes.data_as(_FLOAT_P),
        )
    )
    if count <= 0:
        return [], []
    return (
        [int(token) for token in out_tokens[:count].tolist()],
        [float(prob) for prob in out_probs[:count].tolist()],
    )


def qsg_eagle_replacement_draft(
    draft_logits: list[float] | Any,
    target_logits: list[float] | Any,
    *,
    draft_tokens: int,
    temperature: float,
    max_tree_width: int,
    acceptance_threshold: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    draft = _as_f32_contiguous(draft_logits)
    target = _as_f32_contiguous(target_logits)
    if draft.shape[0] != target.shape[0]:
        raise ValueError("draft_logits and target_logits must have same vocab length")
    max_draft = max(0, int(draft_tokens))
    if max_draft <= 0 or draft.size == 0:
        return [], []
    out_tokens = np.zeros((max_draft,), dtype=np.int32)
    out_probs = np.zeros((max_draft,), dtype=np.float32)
    count = int(
        _lib().qsg_eagle_replacement_draft(
            draft.ctypes.data_as(_FLOAT_P),
            target.ctypes.data_as(_FLOAT_P),
            int(draft.shape[0]),
            max_draft,
            float(temperature),
            int(max_tree_width),
            float(acceptance_threshold),
            int(seed),
            out_tokens.ctypes.data_as(_INT32_P),
            out_probs.ctypes.data_as(_FLOAT_P),
        )
    )
    if count <= 0:
        return [], []
    return (
        [int(token) for token in out_tokens[:count].tolist()],
        [float(prob) for prob in out_probs[:count].tolist()],
    )


def qsg_medusa_head_draft(
    hidden: list[float] | Any,
    head_weights: list[float] | Any,
    head_bias: list[float] | Any | None,
    *,
    num_heads: int,
    hidden_dim: int,
    vocab_size: int,
    draft_tokens: int,
    temperature: float,
    top_k: int,
    min_probability: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    hidden_arr = _as_f32_contiguous(hidden)
    weights_arr = _as_f32_contiguous(head_weights)
    bias_arr = None if head_bias is None else _as_f32_contiguous(head_bias)
    max_draft = max(0, int(draft_tokens))
    if hidden_arr.size == 0 or weights_arr.size == 0 or max_draft <= 0:
        return [], []
    out_tokens = np.zeros((max_draft,), dtype=np.int32)
    out_probs = np.zeros((max_draft,), dtype=np.float32)
    bias_ptr = (
        bias_arr.ctypes.data_as(_FLOAT_P)
        if bias_arr is not None
        else ctypes.POINTER(ctypes.c_float)()
    )
    count = int(
        _lib().qsg_medusa_head_draft(
            hidden_arr.ctypes.data_as(_FLOAT_P),
            weights_arr.ctypes.data_as(_FLOAT_P),
            bias_ptr,
            int(num_heads),
            int(hidden_dim),
            int(vocab_size),
            max_draft,
            float(temperature),
            int(top_k),
            float(min_probability),
            int(seed),
            out_tokens.ctypes.data_as(_INT32_P),
            out_probs.ctypes.data_as(_FLOAT_P),
        )
    )
    if count <= 0:
        return [], []
    return (
        [int(token) for token in out_tokens[:count].tolist()],
        [float(prob) for prob in out_probs[:count].tolist()],
    )


def qsg_hydra_head_draft(
    hidden: list[float] | Any,
    base_logits: list[float] | Any,
    head_weights: list[float] | Any,
    head_bias: list[float] | Any | None,
    *,
    num_heads: int,
    hidden_dim: int,
    vocab_size: int,
    draft_tokens: int,
    temperature: float,
    top_k: int,
    blend_alpha: float,
    min_probability: float,
    seed: int,
) -> tuple[list[int], list[float]]:
    hidden_arr = _as_f32_contiguous(hidden)
    base_logits_arr = _as_f32_contiguous(base_logits)
    weights_arr = _as_f32_contiguous(head_weights)
    bias_arr = None if head_bias is None else _as_f32_contiguous(head_bias)
    max_draft = max(0, int(draft_tokens))
    if (
        hidden_arr.size == 0
        or base_logits_arr.size == 0
        or weights_arr.size == 0
        or max_draft <= 0
    ):
        return [], []
    out_tokens = np.zeros((max_draft,), dtype=np.int32)
    out_probs = np.zeros((max_draft,), dtype=np.float32)
    bias_ptr = (
        bias_arr.ctypes.data_as(_FLOAT_P)
        if bias_arr is not None
        else ctypes.POINTER(ctypes.c_float)()
    )
    count = int(
        _lib().qsg_hydra_head_draft(
            hidden_arr.ctypes.data_as(_FLOAT_P),
            base_logits_arr.ctypes.data_as(_FLOAT_P),
            weights_arr.ctypes.data_as(_FLOAT_P),
            bias_ptr,
            int(num_heads),
            int(hidden_dim),
            int(vocab_size),
            max_draft,
            float(temperature),
            int(top_k),
            float(blend_alpha),
            float(min_probability),
            int(seed),
            out_tokens.ctypes.data_as(_INT32_P),
            out_probs.ctypes.data_as(_FLOAT_P),
        )
    )
    if count <= 0:
        return [], []
    return (
        [int(token) for token in out_tokens[:count].tolist()],
        [float(prob) for prob in out_probs[:count].tolist()],
    )


@dataclass(slots=True)
class NativeSchedulerMetrics:
    queue_depth: int = 0
    active_requests: int = 0
    inflight_requests: int = 0
    prefill_active_requests: int = 0
    decode_active_requests: int = 0
    admitted_requests: int = 0
    completed_requests: int = 0
    cancelled_requests: int = 0
    evicted_requests: int = 0
    iterations: int = 0
    prefill_request_count: int = 0
    prefill_tokens_scheduled: int = 0
    decode_tokens_emitted: int = 0
    chunked_prefill_requests: int = 0
    chunked_prefill_chunks: int = 0
    iteration_last_ms: float = 0.0
    iteration_avg_ms: float = 0.0
    iteration_p95_ms: float = 0.0
    queue_wait_p50_ms: float = 0.0
    queue_wait_p95_ms: float = 0.0
    queue_wait_p99_ms: float = 0.0
    latent_requests: int = 0
    suspended_requests: int = 0


class NativeQSGRequestState(IntFlag):
    KNOWN = 1
    COMPLETED = 2
    CANCELLED = 4
    ACTIVE = 8
    PENDING = 16
    LATENT = 32
    SUSPENDED = 64


@dataclass(slots=True)
class NativeRuntimeMetrics:
    scheduler: NativeSchedulerMetrics
    worker_iterations: int = 0
    emitted_events: int = 0
    prefill_batches: int = 0
    runtime_prefill_tokens: int = 0
    runtime_decode_steps: int = 0
    worker_running: bool = False
    native_runtime_abi_ready: bool = False


@dataclass(slots=True)
class NativeRuntimeEvent:
    token_id: int | None = None
    done: bool = False
    error: str | None = None


@dataclass(slots=True)
class NativeDraftVerificationResult:
    accepted_count: int = 0
    probabilities: list[float] | None = None
    recovery_token: int | None = None
    stop_reason: int = 0


def qsg_autoregressive_generate(
    *,
    model_graph_handle: int,
    prompt_tokens: list[int] | Any,
    max_new_tokens: int,
    vocab_size: int,
    eos_token: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens_before_eos: int,
    seed: int | None = None,
) -> tuple[list[int], int]:
    prompt_arr = _as_i32_contiguous(prompt_tokens)
    prompt_ptr = (
        prompt_arr.ctypes.data_as(_INT32_P)
        if prompt_arr.size > 0
        else _INT32_P()
    )
    out_capacity = int(max(0, max_new_tokens))
    if out_capacity <= 0:
        return [], 0
    out_tokens = np.empty((out_capacity,), dtype=np.int32)
    out_count = ctypes.c_int32(0)
    out_stop_reason = ctypes.c_int32(0)
    ok = int(
        _lib().qsg_autoregressive_generate(
            ctypes.c_void_p(int(model_graph_handle)),
            prompt_ptr,
            int(prompt_arr.size),
            int(max_new_tokens),
            int(vocab_size),
            int(eos_token),
            float(temperature),
            float(top_p),
            int(max(0, top_k)),
            float(min_p),
            float(presence_penalty),
            float(repetition_penalty),
            int(max(0, no_repeat_ngram_size)),
            int(max(0, min_new_tokens_before_eos)),
            int(seed is not None),
            int(seed or 0),
            out_tokens.ctypes.data_as(_INT32_P),
            int(out_capacity),
            ctypes.byref(out_count),
            ctypes.byref(out_stop_reason),
        )
    )
    if ok != 1:
        raise RuntimeError("Native autoregressive generation failed")
    count = int(max(0, out_count.value))
    return out_tokens[:count].tolist(), int(out_stop_reason.value)


def qsg_verify_draft_tokens(
    *,
    model_graph_handle: int,
    prompt_tokens: list[int] | Any,
    draft_tokens: list[int] | Any,
    generated_prefix_count: int,
    vocab_size: int,
    eos_token: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    min_new_tokens_before_eos: int,
    min_accept_probability: float,
    sample_recovery_token: bool = False,
) -> NativeDraftVerificationResult:
    prompt_arr = _as_i32_contiguous(prompt_tokens)
    draft_arr = _as_i32_contiguous(draft_tokens)
    if draft_arr.size == 0:
        return NativeDraftVerificationResult(
            accepted_count=0,
            probabilities=[],
            recovery_token=None,
            stop_reason=0,
        )
    out_probs = np.zeros((int(draft_arr.size),), dtype=np.float32)
    out_prob_count = ctypes.c_int32(0)
    out_accepted_count = ctypes.c_int32(0)
    out_recovery_token = ctypes.c_int32(-1)
    out_stop_reason = ctypes.c_int32(0)
    ok = int(
        _lib().qsg_verify_draft_tokens(
            ctypes.c_void_p(int(model_graph_handle)),
            (
                prompt_arr.ctypes.data_as(_INT32_P)
                if prompt_arr.size > 0
                else _INT32_P()
            ),
            int(prompt_arr.size),
            draft_arr.ctypes.data_as(_INT32_P),
            int(draft_arr.size),
            int(max(0, generated_prefix_count)),
            int(vocab_size),
            int(eos_token),
            float(temperature),
            float(top_p),
            int(max(0, top_k)),
            float(min_p),
            float(presence_penalty),
            float(repetition_penalty),
            int(max(0, no_repeat_ngram_size)),
            int(max(0, min_new_tokens_before_eos)),
            float(min_accept_probability),
            int(bool(sample_recovery_token)),
            out_probs.ctypes.data_as(_FLOAT_P),
            int(out_probs.shape[0]),
            ctypes.byref(out_prob_count),
            ctypes.byref(out_accepted_count),
            ctypes.byref(out_recovery_token),
            ctypes.byref(out_stop_reason),
        )
    )
    if ok != 1:
        raise RuntimeError("Native draft verification failed")
    prob_count = int(max(0, out_prob_count.value))
    recovery_token = int(out_recovery_token.value)
    return NativeDraftVerificationResult(
        accepted_count=int(max(0, out_accepted_count.value)),
        probabilities=[float(value) for value in out_probs[:prob_count].tolist()],
        recovery_token=(recovery_token if recovery_token >= 0 else None),
        stop_reason=int(out_stop_reason.value),
    )


def qsg_masked_diffusion_draft(
    logits: list[float] | Any,
    *,
    draft_tokens: int,
    mask_stride: int,
    temperature: float,
    top_k: int,
    min_probability: float,
    seed: int,
) -> tuple[list[int], list[float], list[int]]:
    logits_arr = _as_f32_contiguous(logits)
    target = max(0, int(draft_tokens))
    if logits_arr.size == 0 or target <= 0:
        return [], [], []
    out_tokens = np.zeros((target,), dtype=np.int32)
    out_probs = np.zeros((target,), dtype=np.float32)
    out_positions = np.zeros((target,), dtype=np.int32)
    count = int(
        _lib().qsg_masked_diffusion_draft(
            logits_arr.ctypes.data_as(_FLOAT_P),
            int(logits_arr.shape[0]),
            target,
            int(max(1, mask_stride)),
            float(temperature),
            int(top_k),
            float(min_probability),
            int(seed),
            out_tokens.ctypes.data_as(_INT32_P),
            out_probs.ctypes.data_as(_FLOAT_P),
            out_positions.ctypes.data_as(_INT32_P),
        )
    )
    if count <= 0:
        return [], [], []
    return (
        [int(token) for token in out_tokens[:count].tolist()],
        [float(prob) for prob in out_probs[:count].tolist()],
        [int(position) for position in out_positions[:count].tolist()],
    )


class NativeQSGScheduler:
    """Thin Python binding over the native C++ request scheduler."""

    def __init__(
        self,
        *,
        max_active_requests: int,
        max_pending_requests: int,
        priority_policy: bool,
        interleaved_streams: bool,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = _lib().qsg_scheduler_create(
            int(max(1, max_active_requests)),
            int(max(1, max_pending_requests)),
            int(bool(priority_policy)),
            int(bool(interleaved_streams)),
        )
        if not self._handle:
            raise RuntimeError("Failed to create native QSG scheduler.")

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            _lib().qsg_scheduler_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_handle(self) -> ctypes.c_void_p:
        handle = self._handle
        if not handle:
            raise RuntimeError("Native scheduler handle is closed.")
        return handle

    def submit(self, request_id: str, *, priority: int, arrival_ts_ns: int) -> None:
        self.submit_with_metadata(
            request_id,
            priority=priority,
            arrival_ts_ns=arrival_ts_ns,
            prompt_token_count=0,
            max_new_tokens=0,
            prefill_chunk_size=1,
        )

    def submit_with_metadata(
        self,
        request_id: str,
        *,
        priority: int,
        arrival_ts_ns: int,
        prompt_token_count: int,
        max_new_tokens: int,
        prefill_chunk_size: int,
    ) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            code = int(
                _lib().qsg_scheduler_submit(
                    self._require_handle(),
                    encoded,
                    int(priority),
                    int(arrival_ts_ns),
                    int(max(0, prompt_token_count)),
                    int(max(0, max_new_tokens)),
                    int(max(1, prefill_chunk_size)),
                )
            )
        if code == 0:
            return
        if code == -2:
            raise RuntimeError("QSGInferenceEngine pending queue is full")
        if code == -3:
            raise ValueError(f"Duplicate request_id '{request_id}'")
        raise RuntimeError("Native scheduler submit failed")

    def cancel(self, request_id: str) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_scheduler_cancel(self._require_handle(), encoded)

    def complete(self, request_id: str, *, cancelled: bool = False) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_scheduler_complete(
                self._require_handle(),
                encoded,
                int(bool(cancelled)),
            )

    def promote(self) -> None:
        with self._lock:
            _lib().qsg_scheduler_promote(self._require_handle())

    def active_ids(self) -> list[str]:
        with self._lock:
            count = int(_lib().qsg_scheduler_active_count(self._require_handle()))
            if count <= 0:
                return []
            values: list[str] = []
            for idx in range(count):
                buf = ctypes.create_string_buffer(192)
                written = int(
                    _lib().qsg_scheduler_copy_active_id(
                        self._require_handle(),
                        idx,
                        buf,
                        len(buf),
                    )
                )
                if written <= 0:
                    continue
                values.append(buf.value.decode("utf-8", errors="ignore"))
            return values

    def rotate_active(self) -> None:
        with self._lock:
            _lib().qsg_scheduler_rotate_active(self._require_handle())

    def first_scheduled_ns(self, request_id: str) -> int:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            return int(
                _lib().qsg_scheduler_first_scheduled_ns(
                    self._require_handle(),
                    encoded,
                )
            )

    def request_state(self, request_id: str) -> NativeQSGRequestState:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            return NativeQSGRequestState(
                _lib().qsg_scheduler_request_state(
                    self._require_handle(),
                    encoded,
                )
            )

    def mark_request_latent(self, request_id: str, is_latent: bool) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_scheduler_set_request_latent(
                self._require_handle(),
                encoded,
                int(bool(is_latent)),
            )

    def mark_request_suspended(self, request_id: str, is_suspended: bool) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_scheduler_set_request_suspended(
                self._require_handle(),
                encoded,
                int(bool(is_suspended)),
            )

    def record_iteration(self, iteration_ms: float) -> None:
        with self._lock:
            _lib().qsg_scheduler_record_iteration(
                self._require_handle(),
                float(iteration_ms),
            )

    def record_decode_emit(self, request_id: str, emitted_tokens: int) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_scheduler_record_decode_emit(
                self._require_handle(),
                encoded,
                int(max(0, emitted_tokens)),
            )

    def metrics(self) -> NativeSchedulerMetrics:
        payload = _SchedulerMetrics()
        with self._lock:
            _lib().qsg_scheduler_get_metrics(
                self._require_handle(), ctypes.byref(payload)
            )
        return NativeSchedulerMetrics(
            queue_depth=int(payload.queue_depth),
            active_requests=int(payload.active_requests),
            inflight_requests=int(payload.inflight_requests),
            prefill_active_requests=int(payload.prefill_active_requests),
            decode_active_requests=int(payload.decode_active_requests),
            admitted_requests=int(payload.admitted_requests),
            completed_requests=int(payload.completed_requests),
            cancelled_requests=int(payload.cancelled_requests),
            evicted_requests=int(payload.evicted_requests),
            iterations=int(payload.iterations),
            prefill_request_count=int(payload.prefill_request_count),
            prefill_tokens_scheduled=int(payload.prefill_tokens_scheduled),
            decode_tokens_emitted=int(payload.decode_tokens_emitted),
            chunked_prefill_requests=int(payload.chunked_prefill_requests),
            chunked_prefill_chunks=int(payload.chunked_prefill_chunks),
            iteration_last_ms=float(payload.iteration_last_ms),
            iteration_avg_ms=float(payload.iteration_avg_ms),
            iteration_p95_ms=float(payload.iteration_p95_ms),
            queue_wait_p50_ms=float(payload.queue_wait_p50_ms),
            queue_wait_p95_ms=float(payload.queue_wait_p95_ms),
            queue_wait_p99_ms=float(payload.queue_wait_p99_ms),
            latent_requests=int(payload.latent_requests),
            suspended_requests=int(payload.suspended_requests),
        )


class NativeQSGRuntime:
    """Native-owned continuous decode runtime over the shared model graph."""

    def __init__(
        self,
        *,
        model_graph_handle: int,
        vocab_size: int,
        eos_token: int,
        ubatch: int,
        max_active_requests: int,
        max_pending_requests: int,
        priority_policy: bool,
        interleaved_streams: bool,
    ) -> None:
        self._lock = threading.RLock()
        self._handle = _lib().qsg_runtime_create(
            ctypes.c_void_p(int(model_graph_handle)),
            int(max(0, vocab_size)),
            int(eos_token),
            int(max(1, ubatch)),
            int(max(1, max_active_requests)),
            int(max(1, max_pending_requests)),
            int(bool(priority_policy)),
            int(bool(interleaved_streams)),
        )
        if not self._handle:
            raise RuntimeError("Failed to create native QSG runtime.")

    def close(self) -> None:
        with self._lock:
            if not self._handle:
                return
            _lib().qsg_runtime_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require_handle(self) -> ctypes.c_void_p:
        handle = self._handle
        if not handle:
            raise RuntimeError("Native runtime handle is closed.")
        return handle

    def submit(
        self,
        request_id: str,
        *,
        priority: int,
        arrival_ts_ns: int,
        prompt_tokens: list[int] | Any,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        min_new_tokens_before_eos: int,
        seed: int | None = None,
        latent: bool = False,
        suspended: bool = False,
    ) -> None:
        encoded = str(request_id).encode("utf-8")
        prompt_arr = _as_i32_contiguous(prompt_tokens)
        prompt_ptr = (
            prompt_arr.ctypes.data_as(_INT32_P)
            if prompt_arr.size > 0
            else _INT32_P()
        )
        with self._lock:
            code = int(
                _lib().qsg_runtime_submit(
                    self._require_handle(),
                    encoded,
                    int(priority),
                    int(arrival_ts_ns),
                    prompt_ptr,
                    int(prompt_arr.size),
                    int(max(0, max_new_tokens)),
                    float(temperature),
                    float(top_p),
                    int(max(0, top_k)),
                    float(min_p),
                    float(presence_penalty),
                    float(repetition_penalty),
                    int(max(0, no_repeat_ngram_size)),
                    int(max(0, min_new_tokens_before_eos)),
                    int(seed is not None),
                    int(seed or 0),
                    int(bool(latent)),
                    int(bool(suspended)),
                )
            )
        if code == 0:
            return
        if code == -2:
            raise RuntimeError("Native runtime pending queue is full")
        if code == -3:
            raise ValueError(f"Duplicate request_id '{request_id}'")
        raise RuntimeError("Native runtime submit failed")

    def cancel(self, request_id: str) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_runtime_cancel(self._require_handle(), encoded)

    def mark_request_latent(self, request_id: str, is_latent: bool) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_runtime_set_request_latent(
                self._require_handle(), encoded, int(bool(is_latent))
            )

    def mark_request_suspended(self, request_id: str, is_suspended: bool) -> None:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            _lib().qsg_runtime_set_request_suspended(
                self._require_handle(), encoded, int(bool(is_suspended))
            )

    def first_scheduled_ns(self, request_id: str) -> int:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            return int(
                _lib().qsg_runtime_first_scheduled_ns(self._require_handle(), encoded)
            )

    def request_state(self, request_id: str) -> NativeQSGRequestState:
        encoded = str(request_id).encode("utf-8")
        with self._lock:
            return NativeQSGRequestState(
                _lib().qsg_runtime_request_state(self._require_handle(), encoded)
            )

    def poll(self, request_id: str) -> NativeRuntimeEvent | None:
        encoded = str(request_id).encode("utf-8")
        token_id = ctypes.c_int32(-1)
        has_token = ctypes.c_int32(0)
        done = ctypes.c_int32(0)
        error_buf = ctypes.create_string_buffer(512)
        with self._lock:
            status = int(
                _lib().qsg_runtime_poll_event(
                    self._require_handle(),
                    encoded,
                    ctypes.byref(token_id),
                    ctypes.byref(has_token),
                    ctypes.byref(done),
                    error_buf,
                    len(error_buf),
                )
            )
        if status <= 0:
            return None
        error = error_buf.value.decode("utf-8", errors="ignore") or None
        return NativeRuntimeEvent(
            token_id=int(token_id.value) if bool(has_token.value) else None,
            done=bool(done.value),
            error=error,
        )

    def metrics(self) -> NativeRuntimeMetrics:
        payload = _RuntimeMetrics()
        with self._lock:
            _lib().qsg_runtime_get_metrics(self._require_handle(), ctypes.byref(payload))
        scheduler = NativeSchedulerMetrics(
            queue_depth=int(payload.scheduler.queue_depth),
            active_requests=int(payload.scheduler.active_requests),
            inflight_requests=int(payload.scheduler.inflight_requests),
            prefill_active_requests=int(payload.scheduler.prefill_active_requests),
            decode_active_requests=int(payload.scheduler.decode_active_requests),
            admitted_requests=int(payload.scheduler.admitted_requests),
            completed_requests=int(payload.scheduler.completed_requests),
            cancelled_requests=int(payload.scheduler.cancelled_requests),
            evicted_requests=int(payload.scheduler.evicted_requests),
            iterations=int(payload.scheduler.iterations),
            prefill_request_count=int(payload.scheduler.prefill_request_count),
            prefill_tokens_scheduled=int(payload.scheduler.prefill_tokens_scheduled),
            decode_tokens_emitted=int(payload.scheduler.decode_tokens_emitted),
            chunked_prefill_requests=int(payload.scheduler.chunked_prefill_requests),
            chunked_prefill_chunks=int(payload.scheduler.chunked_prefill_chunks),
            iteration_last_ms=float(payload.scheduler.iteration_last_ms),
            iteration_avg_ms=float(payload.scheduler.iteration_avg_ms),
            iteration_p95_ms=float(payload.scheduler.iteration_p95_ms),
            queue_wait_p50_ms=float(payload.scheduler.queue_wait_p50_ms),
            queue_wait_p95_ms=float(payload.scheduler.queue_wait_p95_ms),
            queue_wait_p99_ms=float(payload.scheduler.queue_wait_p99_ms),
            latent_requests=int(payload.scheduler.latent_requests),
            suspended_requests=int(payload.scheduler.suspended_requests),
        )
        return NativeRuntimeMetrics(
            scheduler=scheduler,
            worker_iterations=int(payload.worker_iterations),
            emitted_events=int(payload.emitted_events),
            prefill_batches=int(payload.prefill_batches),
            runtime_prefill_tokens=int(payload.runtime_prefill_tokens),
            runtime_decode_steps=int(payload.runtime_decode_steps),
            worker_running=bool(payload.worker_running),
            native_runtime_abi_ready=bool(payload.native_runtime_abi_ready),
        )

    def shutdown(self) -> None:
        with self._lock:
            _lib().qsg_runtime_shutdown(self._require_handle())

    def run_forever(self) -> None:
        with self._lock:
            handle = self._require_handle()
        _lib().qsg_runtime_run_forever(handle)
