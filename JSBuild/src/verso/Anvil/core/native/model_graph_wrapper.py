"""ctypes wrapper for the native C++ model execution graph.

The NativeModelGraph eliminates Python from the decode hot path by
executing the entire forward pass (RMSNorm → QKV → RoPE → Attention →
FFN → LM head) in a single C++ call.  This reduces ~3440 ctypes calls
per decode step to exactly 1.

Supports:
  - Paged KV cache (lazy allocation, supports 400K+ context)
  - Direct quantized weight pass-through (no dequantization overhead)
  - Float32 weight fallback
"""

from __future__ import annotations

import ctypes
import os
from array import array
from typing import Any, Optional

import numpy as np

from core.model.model_profile import ModelProfile
from core.native.native_ops import load_native_library
from core.native.weight_store import WeightStore
from core.native import quantized_matmul_wrapper as quant_ops


class _GraphPerfStatsSnapshot(ctypes.Structure):
    _fields_ = [
        ("embedding_lookup_seconds", ctypes.c_double),
        ("attention_proj_seconds", ctypes.c_double),
        ("attention_rope_kv_seconds", ctypes.c_double),
        ("attention_decode_seconds", ctypes.c_double),
        ("attention_out_proj_seconds", ctypes.c_double),
        ("ffn_norm_seconds", ctypes.c_double),
        ("ffn_gate_up_seconds", ctypes.c_double),
        ("ffn_down_seconds", ctypes.c_double),
        ("ssm_projection_seconds", ctypes.c_double),
        ("ssm_conv_seconds", ctypes.c_double),
        ("ssm_recurrent_seconds", ctypes.c_double),
        ("ssm_output_seconds", ctypes.c_double),
        ("ssm_seconds", ctypes.c_double),
        ("moe_seconds", ctypes.c_double),
        ("final_norm_seconds", ctypes.c_double),
        ("lm_head_seconds", ctypes.c_double),
        ("sanitize_seconds", ctypes.c_double),
        ("forward_token_calls", ctypes.c_int),
        ("forward_token_id_calls", ctypes.c_int),
        ("forward_token_ids_calls", ctypes.c_int),
        ("forward_token_ids_token_count", ctypes.c_int),
        ("attention_calls", ctypes.c_int),
        ("ffn_calls", ctypes.c_int),
        ("ssm_calls", ctypes.c_int),
        ("moe_calls", ctypes.c_int),
        ("packed_lm_head_calls", ctypes.c_int),
        ("attention_proj_bytes", ctypes.c_int64),
        ("attention_proj_flops", ctypes.c_int64),
        ("attention_out_proj_bytes", ctypes.c_int64),
        ("attention_out_proj_flops", ctypes.c_int64),
        ("ffn_gate_up_bytes", ctypes.c_int64),
        ("ffn_gate_up_flops", ctypes.c_int64),
        ("ffn_down_bytes", ctypes.c_int64),
        ("ffn_down_flops", ctypes.c_int64),
        ("ssm_projection_bytes", ctypes.c_int64),
        ("ssm_projection_flops", ctypes.c_int64),
        ("ssm_output_bytes", ctypes.c_int64),
        ("ssm_output_flops", ctypes.c_int64),
        ("moe_bytes", ctypes.c_int64),
        ("moe_flops", ctypes.c_int64),
        ("lm_head_bytes", ctypes.c_int64),
        ("lm_head_flops", ctypes.c_int64),
    ]


class _GraphDriftConfig(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int),
        ("mode", ctypes.c_int),
        ("block_size_tokens", ctypes.c_int),
        ("update_interval_tokens", ctypes.c_int),
        ("prune_interval_tokens", ctypes.c_int),
        ("preserve_head_tokens", ctypes.c_int),
        ("preserve_recent_tokens", ctypes.c_int),
        ("min_active_tokens", ctypes.c_int),
        ("damp_threshold", ctypes.c_float),
        ("prune_threshold", ctypes.c_float),
        ("damping_strength", ctypes.c_float),
        ("hysteresis", ctypes.c_float),
    ]


class _GraphDriftSnapshot(ctypes.Structure):
    _fields_ = [
        ("latest_drift", ctypes.c_float),
        ("mean_drift", ctypes.c_float),
        ("max_drift", ctypes.c_float),
        ("decay_ratio", ctypes.c_float),
        ("active_token_count", ctypes.c_int),
        ("damped_block_count", ctypes.c_int),
        ("pruned_block_count", ctypes.c_int),
        ("stabilizer_seconds", ctypes.c_double),
        ("stabilizer_calls", ctypes.c_int),
        ("mode", ctypes.c_int),
    ]


def _graph_drift_snapshot_dict(snapshot: _GraphDriftSnapshot) -> dict[str, int | float]:
    return {
        "latest_drift": float(snapshot.latest_drift),
        "mean_drift": float(snapshot.mean_drift),
        "max_drift": float(snapshot.max_drift),
        "decay_ratio": float(snapshot.decay_ratio),
        "active_token_count": int(snapshot.active_token_count),
        "damped_block_count": int(snapshot.damped_block_count),
        "pruned_block_count": int(snapshot.pruned_block_count),
        "stabilizer_seconds": float(snapshot.stabilizer_seconds),
        "stabilizer_calls": int(snapshot.stabilizer_calls),
        "mode": int(snapshot.mode),
    }


def _metadata_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "off", "no"}:
            return False
        if normalized in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def _split_fused_qkv_weight(
    weight: np.ndarray | quant_ops.QuantizedMatrix,
    q_out: int,
    kv_out: int,
) -> tuple[
    np.ndarray | quant_ops.QuantizedMatrix,
    np.ndarray | quant_ops.QuantizedMatrix | None,
    np.ndarray | quant_ops.QuantizedMatrix | None,
]:
    """Split a fused QKV projection into separate Q/K/V projections.

    The graph API expects individual projections. Some models only expose
    `attn_qkv`; for those, split deterministically in logical output order.
    """
    q_dim = int(max(0, q_out))
    kv_dim = int(max(0, kv_out))
    total = q_dim + 2 * kv_dim
    if q_dim <= 0 or kv_dim <= 0:
        return weight, None, None

    if isinstance(weight, quant_ops.QuantizedMatrix):
        if weight.output_dim < total:
            return weight, None, None
        packed = weight.ensure_packed()

        def _slice_quant(start: int, count: int) -> quant_ops.QuantizedMatrix:
            logical = np.arange(start, start + count, dtype=np.int64)
            if weight._inverse_row_permutation is not None:
                physical = weight._inverse_row_permutation[logical]
            else:
                physical = logical
            data = np.ascontiguousarray(packed[physical], dtype=np.uint8)
            out = quant_ops.QuantizedMatrix(
                name=f"{weight.name}::{start}:{start + count}",
                qtype=weight.qtype,
                shape=(weight.input_dim, int(count)),
                data=data,
                interleave_factor=1,
            )
            out._validated = bool(weight._validated)
            return out

        q = _slice_quant(0, q_dim)
        k = _slice_quant(q_dim, kv_dim)
        v = _slice_quant(q_dim + kv_dim, kv_dim)
        return q, k, v

    arr = np.asarray(weight)
    if arr.ndim != 2:
        return weight, None, None

    if arr.shape[0] >= total:
        q = np.ascontiguousarray(arr[:q_dim, :], dtype=np.float32)
        k = np.ascontiguousarray(arr[q_dim:q_dim + kv_dim, :], dtype=np.float32)
        v = np.ascontiguousarray(arr[q_dim + kv_dim:q_dim + 2 * kv_dim, :], dtype=np.float32)
        return q, k, v

    if arr.shape[1] >= total:
        q = np.ascontiguousarray(arr[:, :q_dim].T, dtype=np.float32)
        k = np.ascontiguousarray(arr[:, q_dim:q_dim + kv_dim].T, dtype=np.float32)
        v = np.ascontiguousarray(arr[:, q_dim + kv_dim:q_dim + 2 * kv_dim].T, dtype=np.float32)
        return q, k, v

    return weight, None, None


def _slice_quant_rows(
    weight: quant_ops.QuantizedMatrix,
    logical_rows: np.ndarray,
    *,
    name_suffix: str,
) -> quant_ops.QuantizedMatrix:
    packed = weight.ensure_packed()
    if weight._inverse_row_permutation is not None:
        physical = weight._inverse_row_permutation[logical_rows]
    else:
        physical = logical_rows
    data = np.ascontiguousarray(packed[physical], dtype=np.uint8)
    out = quant_ops.QuantizedMatrix(
        name=f"{weight.name}::{name_suffix}",
        qtype=weight.qtype,
        shape=(weight.input_dim, int(logical_rows.shape[0])),
        data=data,
        interleave_factor=1,
    )
    out._validated = bool(weight._validated)
    return out


def _slice_dense_rows(
    weight: np.ndarray,
    logical_rows: np.ndarray,
) -> np.ndarray:
    arr = np.asarray(weight)
    if arr.ndim != 2:
        return arr
    if arr.shape[1] == logical_rows.shape[0]:
        return np.ascontiguousarray(arr[:, logical_rows], dtype=np.float32)
    if arr.shape[0] == logical_rows.shape[0]:
        return np.ascontiguousarray(arr[logical_rows, :], dtype=np.float32)
    if arr.shape[0] >= np.max(logical_rows) + 1:
        return np.ascontiguousarray(arr[logical_rows, :], dtype=np.float32)
    if arr.shape[1] >= np.max(logical_rows) + 1:
        return np.ascontiguousarray(arr[:, logical_rows].T, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _split_qwen_q_and_gate_weight(
    weight: np.ndarray | quant_ops.QuantizedMatrix,
    *,
    n_heads: int,
    head_dim: int,
) -> tuple[
    np.ndarray | quant_ops.QuantizedMatrix,
    np.ndarray | quant_ops.QuantizedMatrix | None,
]:
    pair_dim = int(2 * head_dim)
    total_dim = int(n_heads * pair_dim)
    if pair_dim <= 0 or total_dim <= 0:
        return weight, None

    q_rows = []
    gate_rows = []
    for h in range(n_heads):
        base = h * pair_dim
        q_rows.extend(range(base, base + head_dim))
        gate_rows.extend(range(base + head_dim, base + pair_dim))
    q_idx = np.asarray(q_rows, dtype=np.int64)
    gate_idx = np.asarray(gate_rows, dtype=np.int64)

    if isinstance(weight, quant_ops.QuantizedMatrix):
        if weight.output_dim < total_dim:
            return weight, None
        q = _slice_quant_rows(weight, q_idx, name_suffix="q")
        gate = _slice_quant_rows(weight, gate_idx, name_suffix="gate")
        return q, gate

    arr = np.asarray(weight)
    if arr.ndim != 2:
        return np.ascontiguousarray(arr, dtype=np.float32), None
    if max(arr.shape) < total_dim:
        return np.ascontiguousarray(arr, dtype=np.float32), None
    q = _slice_dense_rows(arr, q_idx)
    gate = _slice_dense_rows(arr, gate_idx)
    return q, gate


class NativeModelGraph:
    """Full C++ execution graph for transformer decode with zero Python overhead."""

    def __init__(
        self,
        n_layers: int,
        embedding_dim: int,
        vocab_size: int,
        n_heads: int = 0,
        n_kv_heads: int = 0,
        head_dim: int = 0,
        max_seq: int = 2048,
        rms_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        weight_store: Optional[WeightStore] = None,
        profile: Optional[ModelProfile] = None,
    ):
        self.n_layers = int(n_layers)
        self.embedding_dim = int(embedding_dim)
        self.vocab_size = int(vocab_size)
        self.n_heads = int(n_heads) if n_heads > 0 else 1
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads > 0 else self.n_heads
        self.head_dim = int(head_dim) if head_dim > 0 else (embedding_dim // self.n_heads)
        self.max_seq = int(max_seq)
        self.rms_eps = float(rms_eps)
        self.rope_theta = float(rope_theta)
        self._lib: Optional[ctypes.CDLL] = None
        self._handle: Optional[int] = None
        self._has_full_graph = False
        self._has_quantized_api = False
        self._has_layer_api = False
        self._has_extended_api = False
        self._has_batch_token_id_api = False
        self._has_perf_stats_api = False
        self._has_perf_stats_reset_api = False
        self._has_last_hidden_api = False
        self._has_execution_checkpoint_api = False
        self._has_exit_continuation_api = False
        self._has_drift_config_api = False
        self._has_drift_get_config_api = False
        self._has_drift_state_api = False
        self._has_drift_snapshot_arg_api = False
        self._lm_head_qtype: int = 0
        self._lm_head_layout: str = "none"
        # Per-layer C++ capability maps for hybrid dispatch.
        self._layer_cpp_ok: list[bool] = []
        self._layer_cpp_attention_ok: list[bool] = []
        self._layer_cpp_requires_python_ffn: list[bool] = []
        # Keep references to weight arrays to prevent GC
        self._weight_refs: list = []
        self._token_id_buf_capacity = 0
        self._token_id_buf: Optional[ctypes.Array] = None
        self._logits_out = (ctypes.c_float * self.vocab_size)()
        self._last_drift_snapshot = _GraphDriftSnapshot()
        self._last_drift_snapshot_valid = False
        self._load()

        # If weight_store is provided, initialize layer weights
        if weight_store is not None and profile is not None and self._has_full_graph:
            self._init_weights(weight_store, profile)

    def _load(self) -> None:
        try:
            lib = load_native_library()
        except Exception:
            return

        float_p = ctypes.POINTER(ctypes.c_float)
        void_p = ctypes.c_void_p
        c_int = ctypes.c_int

        if hasattr(lib, "graph_forward_token"):
            try:
                # New expanded API
                create_graph = getattr(lib, "create_model_graph_v2", None)
                if create_graph is None:
                    create_graph = getattr(lib, "create_model_graph")
                create_graph.argtypes = [
                    c_int, c_int, c_int,
                    c_int, c_int, c_int,
                    c_int, ctypes.c_float, ctypes.c_float,
                ]
                create_graph.restype = void_p

                lib.graph_set_layer_weights.argtypes = [
                    void_p, c_int,
                    float_p, float_p,
                    float_p, c_int,
                    float_p, c_int,
                    float_p, float_p,
                    float_p, c_int,
                    float_p, float_p,
                    c_int,
                ]
                lib.graph_set_layer_weights.restype = c_int

                lib.graph_set_head_weights.argtypes = [
                    void_p,
                    float_p, float_p, c_int,
                    ctypes.c_float, ctypes.c_float,
                    ctypes.c_float, ctypes.c_float,
                ]
                lib.graph_set_head_weights.restype = c_int
                if hasattr(lib, "graph_set_embedding_weights"):
                    lib.graph_set_embedding_weights.argtypes = [
                        void_p,
                        float_p,
                        c_int,
                    ]
                    lib.graph_set_embedding_weights.restype = c_int
                if hasattr(lib, "graph_set_embedding_weights_quantized"):
                    lib.graph_set_embedding_weights_quantized.argtypes = [
                        void_p,
                        void_p,
                        c_int,
                        c_int,
                        c_int,
                        c_int,
                    ]
                    lib.graph_set_embedding_weights_quantized.restype = c_int

                lib.graph_forward_token.argtypes = [
                    void_p, float_p, c_int,
                    float_p, c_int, c_int,
                ]
                lib.graph_forward_token.restype = c_int
                if hasattr(lib, "graph_forward_token_id"):
                    drift_snapshot_api = bool(
                        hasattr(lib, "graph_get_last_drift_snapshot")
                        or hasattr(lib, "graph_set_drift_config")
                        or hasattr(lib, "graph_get_drift_config")
                    )
                    if drift_snapshot_api:
                        lib.graph_forward_token_id.argtypes = [
                            void_p,
                            c_int,
                            float_p,
                            c_int,
                            c_int,
                            ctypes.POINTER(_GraphDriftSnapshot),
                        ]
                        self._has_drift_snapshot_arg_api = True
                    else:
                        lib.graph_forward_token_id.argtypes = [
                            void_p, c_int, float_p, c_int, c_int
                        ]
                    lib.graph_forward_token_id.restype = c_int
                if hasattr(lib, "graph_forward_token_ids"):
                    if self._has_drift_snapshot_arg_api:
                        lib.graph_forward_token_ids.argtypes = [
                            void_p,
                            ctypes.POINTER(c_int),
                            c_int,
                            float_p,
                            c_int,
                            c_int,
                            ctypes.POINTER(_GraphDriftSnapshot),
                        ]
                    else:
                        lib.graph_forward_token_ids.argtypes = [
                            void_p,
                            ctypes.POINTER(c_int),
                            c_int,
                            float_p,
                            c_int,
                            c_int,
                        ]
                    lib.graph_forward_token_ids.restype = c_int
                    self._has_batch_token_id_api = True
                if hasattr(lib, "graph_set_drift_config"):
                    lib.graph_set_drift_config.argtypes = [
                        void_p,
                        ctypes.POINTER(_GraphDriftConfig),
                    ]
                    lib.graph_set_drift_config.restype = c_int
                    self._has_drift_config_api = True
                if hasattr(lib, "graph_get_drift_config"):
                    lib.graph_get_drift_config.argtypes = [
                        void_p,
                        ctypes.POINTER(_GraphDriftConfig),
                    ]
                    lib.graph_get_drift_config.restype = c_int
                    self._has_drift_get_config_api = True
                if hasattr(lib, "graph_get_last_drift_snapshot"):
                    lib.graph_get_last_drift_snapshot.argtypes = [
                        void_p,
                        ctypes.POINTER(_GraphDriftSnapshot),
                    ]
                    lib.graph_get_last_drift_snapshot.restype = c_int
                    self._has_drift_state_api = True

                lib.graph_reset.argtypes = [void_p]
                lib.graph_reset.restype = c_int
                if hasattr(lib, "graph_reset_perf_stats"):
                    lib.graph_reset_perf_stats.argtypes = [void_p]
                    lib.graph_reset_perf_stats.restype = c_int
                    self._has_perf_stats_reset_api = True

                lib.graph_get_position.argtypes = [void_p]
                lib.graph_get_position.restype = c_int
                if hasattr(lib, "graph_get_perf_stats"):
                    lib.graph_get_perf_stats.argtypes = [
                        void_p,
                        ctypes.POINTER(_GraphPerfStatsSnapshot),
                    ]
                    lib.graph_get_perf_stats.restype = c_int
                    self._has_perf_stats_api = True
                if hasattr(lib, "graph_copy_last_hidden"):
                    lib.graph_copy_last_hidden.argtypes = [
                        void_p,
                        float_p,
                        c_int,
                    ]
                    lib.graph_copy_last_hidden.restype = c_int
                    self._has_last_hidden_api = True
                if all(
                    hasattr(lib, name)
                    for name in (
                        "graph_create_execution_checkpoint",
                        "graph_restore_execution_checkpoint",
                        "graph_destroy_execution_checkpoint",
                    )
                ):
                    lib.graph_create_execution_checkpoint.argtypes = [void_p]
                    lib.graph_create_execution_checkpoint.restype = void_p
                    lib.graph_restore_execution_checkpoint.argtypes = [void_p, void_p]
                    lib.graph_restore_execution_checkpoint.restype = c_int
                    lib.graph_destroy_execution_checkpoint.argtypes = [void_p]
                    lib.graph_destroy_execution_checkpoint.restype = None
                    self._has_execution_checkpoint_api = True
                if all(
                    hasattr(lib, name)
                    for name in (
                        "graph_forward_token_id_to_exit",
                        "graph_continue_from_hidden",
                    )
                ):
                    lib.graph_forward_token_id_to_exit.argtypes = [
                        void_p,
                        c_int,
                        c_int,
                        float_p,
                        c_int,
                        c_int,
                    ]
                    lib.graph_forward_token_id_to_exit.restype = c_int
                    lib.graph_continue_from_hidden.argtypes = [
                        void_p,
                        float_p,
                        c_int,
                        c_int,
                        float_p,
                        c_int,
                        c_int,
                    ]
                    lib.graph_continue_from_hidden.restype = c_int
                    self._has_exit_continuation_api = True

                lib.destroy_model_graph.argtypes = [void_p]
                lib.destroy_model_graph.restype = None

                # Per-layer execution API (for hybrid mode)
                self._has_layer_api = False
                if hasattr(lib, "graph_forward_layer"):
                    lib.graph_forward_layer.argtypes = [
                        void_p, float_p, c_int, c_int, c_int,
                    ]
                    lib.graph_forward_layer.restype = c_int
                    lib.graph_forward_head.argtypes = [
                        void_p, float_p, c_int, float_p, c_int,
                    ]
                    lib.graph_forward_head.restype = c_int
                    self._has_layer_api = True

                # Check for quantized weight API
                if hasattr(lib, "graph_set_layer_weights_quantized"):
                    lib.graph_set_layer_weights_quantized.argtypes = [
                        void_p, c_int,
                        float_p, float_p,           # attn_norm, ffn_norm
                        void_p, c_int, c_int,       # wq, wq_qtype, q_out_dim
                        void_p, c_int, c_int,       # wk, wk_qtype, kv_out_dim
                        void_p, c_int,              # wv, wv_qtype
                        void_p, c_int,              # wo, wo_qtype
                        void_p, c_int, c_int,       # w_gate, wgate_qtype, ffn_dim
                        void_p, c_int,              # w_up, wup_qtype
                        void_p, c_int,              # w_down, wdown_qtype
                        c_int,                      # is_attention
                    ]
                    lib.graph_set_layer_weights_quantized.restype = c_int
                    self._has_quantized_api = True

                if hasattr(lib, "graph_set_head_weights_quantized"):
                    lib.graph_set_head_weights_quantized.argtypes = [
                        void_p,
                        float_p,                    # final_norm
                        void_p, c_int,              # lm_head_quant, qtype
                        ctypes.c_float, ctypes.c_float,
                        ctypes.c_float, ctypes.c_float,
                    ]
                    lib.graph_set_head_weights_quantized.restype = c_int

                if hasattr(lib, "graph_set_layer_extras"):
                    lib.graph_set_layer_extras.argtypes = [
                        void_p, c_int, c_int,         # graph, layer, layer_kind
                        float_p, float_p,             # attn_q_norm, attn_k_norm
                        float_p, float_p, float_p,    # ssm_a, ssm_d, ssm_dt
                        float_p, c_int, c_int,       # ssm_conv, rows, cols
                        float_p,                     # ssm_conv_bias
                        float_p,                     # ssm_norm
                        float_p,                     # router
                        void_p, c_int, c_int,       # attn_gate, qtype, dim
                        void_p, c_int, c_int,       # ssm_in, qtype, dim
                        void_p, c_int, c_int,       # ssm_out, qtype, dim
                        void_p, c_int, c_int,       # ssm_alpha, qtype, dim
                        void_p, c_int, c_int,       # ssm_beta, qtype, dim
                        void_p, c_int, c_int,       # shared_gate, qtype, dim
                        void_p, c_int,              # shared_up, qtype
                        void_p, c_int,              # shared_down, qtype
                        ctypes.POINTER(void_p), c_int, c_int,  # expert_gate_ptrs, qtype, hidden
                        ctypes.POINTER(void_p), c_int,         # expert_up_ptrs, qtype
                        ctypes.POINTER(void_p), c_int,         # expert_down_ptrs, qtype
                        c_int, c_int,               # expert_count, moe_top_k
                    ]
                    lib.graph_set_layer_extras.restype = c_int
                    self._has_extended_api = True

                if hasattr(lib, "graph_set_qwen_mrope_config"):
                    lib.graph_set_qwen_mrope_config.argtypes = [
                        void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                    ]
                    lib.graph_set_qwen_mrope_config.restype = c_int
                if hasattr(lib, "graph_set_qwen_hybrid_config"):
                    lib.graph_set_qwen_hybrid_config.argtypes = [
                        void_p, c_int, c_int, c_int,
                    ]
                    lib.graph_set_qwen_hybrid_config.restype = c_int

                handle = create_graph(
                    self.n_layers, self.embedding_dim, self.vocab_size,
                    self.n_heads, self.n_kv_heads, self.head_dim,
                    self.max_seq, ctypes.c_float(self.rms_eps),
                    ctypes.c_float(self.rope_theta),
                )
                if not handle:
                    return
                self._lib = lib
                self._handle = int(handle)
                self._has_full_graph = True
                return
            except Exception:
                pass

        # Fallback to legacy 3-arg create_model_graph
        try:
            create = getattr(lib, "create_model_graph")
            forward = getattr(lib, "graph_forward")
            destroy = getattr(lib, "destroy_model_graph")
        except AttributeError:
            return

        create.argtypes = [c_int, c_int, c_int]
        create.restype = void_p
        forward.argtypes = [void_p, float_p, c_int, float_p, c_int]
        forward.restype = c_int
        destroy.argtypes = [void_p]
        destroy.restype = None

        handle = create(self.n_layers, self.embedding_dim, self.vocab_size)
        if not handle:
            return
        self._lib = lib
        self._handle = int(handle)

    @property
    def available(self) -> bool:
        return self._lib is not None and self._handle is not None

    @property
    def has_full_graph(self) -> bool:
        """True if the full token-by-token graph execution is available."""
        return self._has_full_graph and self.available

    def _init_weights(self, ws: WeightStore, profile: ModelProfile) -> None:
        """Walk all layers and pass weight pointers to C++.

        Quantized weights are passed directly as packed data pointers (zero-copy).
        Float32 weights are passed as float pointers.
        Also builds the per-layer C++ capability map for hybrid mode.
        """
        if not self.has_full_graph:
            return
        self._layer_cpp_ok = [False] * self.n_layers
        self._layer_cpp_attention_ok = [False] * self.n_layers
        self._layer_cpp_requires_python_ffn = [False] * self.n_layers

        float_p = ctypes.POINTER(ctypes.c_float)
        void_p = ctypes.c_void_p
        q4k_r4_env = os.getenv("ANVIL_Q4K_R4")
        if q4k_r4_env is None:
            # Granite hybrid kernels benefit from the shadow R4 layout by default.
            # Qwen keeps canonical layer weights unless explicitly opted in so the
            # LM-head path can be tuned independently.
            q4k_r4_enabled = str(getattr(profile, "architecture", "")) == "granitehybrid"
        else:
            q4k_r4_enabled = q4k_r4_env.strip().lower() not in {"0", "false", "off"}
        q4k_r4_budget_mb_raw = os.getenv("ANVIL_Q4K_R4_MAX_MB")
        q4k_r4_budget_bytes = (
            max(0, int(q4k_r4_budget_mb_raw or "0")) * 1024 * 1024
            if q4k_r4_budget_mb_raw is not None
            else 0
        )
        q4k_r4_repacked_bytes = 0
        lm_head_packed_env = os.getenv("ANVIL_LM_HEAD_PACKED")
        if lm_head_packed_env is None:
            # Native CPU decode is LM-head bound on both target models. Favor the
            # packed decode layout by default and let operators opt out explicitly.
            lm_head_packed_enabled = True
        else:
            lm_head_packed_enabled = lm_head_packed_env.strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }
        lm_head_pack_budget_mb_raw = os.getenv("ANVIL_LM_HEAD_PACKED_MAX_MB")
        lm_head_pack_budget_bytes = (
            max(0, int(lm_head_pack_budget_mb_raw or "0")) * 1024 * 1024
            if lm_head_pack_budget_mb_raw is not None
            else 0
        )
        lm_head_packed_bytes = 0
        lm_head_q4k_r4_env = os.getenv("ANVIL_LM_HEAD_Q4K_R4")
        lm_head_q6k_lm_env = os.getenv("ANVIL_LM_HEAD_Q6K_LM")
        if lm_head_q4k_r4_env is None:
            lm_head_q4k_r4_enabled = bool(q4k_r4_enabled)
        else:
            lm_head_q4k_r4_enabled = lm_head_q4k_r4_env.strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }
        if lm_head_q6k_lm_env is None:
            # Default to the shared decode layout used by both Granite and Qwen.
            # Q6_K_LM stays available as an explicit opt-in override.
            lm_head_q6k_lm_enabled = False
        else:
            lm_head_q6k_lm_enabled = lm_head_q6k_lm_env.strip().lower() not in {
                "0",
                "false",
                "off",
                "no",
            }

        def _get_float_ptr(w) -> Optional[ctypes.POINTER(ctypes.c_float)]:
            """Get a float* from a float32 weight. Returns None for quantized."""
            if w is None:
                return None
            if isinstance(w, quant_ops.QuantizedMatrix):
                return None  # Don't dequantize — use quantized API
            arr = np.asarray(w, dtype=np.float32)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self._weight_refs.append(arr)
            return arr.ctypes.data_as(float_p)

        def _get_float_projection_ptr(
            w,
            *,
            rows: int,
            cols: int,
        ) -> Optional[ctypes.POINTER(ctypes.c_float)]:
            """Get a float* for a projection matrix normalized to [rows, cols]."""
            if w is None or isinstance(w, quant_ops.QuantizedMatrix):
                return None
            arr = np.asarray(w, dtype=np.float32)
            if arr.ndim != 2:
                return _get_float_ptr(arr)
            if arr.shape == (rows, cols):
                pass
            elif arr.shape == (cols, rows):
                arr = arr.T
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self._weight_refs.append(arr)
            return arr.ctypes.data_as(float_p)

        def _get_quant_info(w, *, prefer_decode_pack: bool = False):
            """Get (void* data_ptr, int qtype) for a quantized weight."""
            nonlocal q4k_r4_repacked_bytes, lm_head_packed_bytes
            if w is None or not isinstance(w, quant_ops.QuantizedMatrix):
                return None, 0
            packed = w.ensure_packed()
            # Graph kernels consume canonical row order. If the runtime weight store
            # applied row interleaving for Python-side kernels, materialize a
            # canonical view here so graph execution remains numerically coherent.
            perm = getattr(w, "_inverse_row_permutation", None)
            if isinstance(perm, np.ndarray) and perm.size > 0:
                packed = np.ascontiguousarray(packed[perm], dtype=np.uint8)
                canonical = quant_ops.QuantizedMatrix(
                    name=f"{w.name}::canonical",
                    qtype=int(w.qtype),
                    shape=w.shape,
                    data=packed,
                    interleave_factor=1,
                )
                canonical._validated = bool(w._validated)
                w = canonical

            if prefer_decode_pack and lm_head_packed_enabled:
                repacked = w
                if int(w.qtype) == int(quant_ops.QTYPE_Q6_K):
                    if lm_head_q6k_lm_enabled:
                        repacked = quant_ops.repack_q6k_lm(w)
                        if int(repacked.qtype) == int(w.qtype):
                            repacked = quant_ops.repack_q6k_r4(w)
                    else:
                        repacked = quant_ops.repack_q6k_r4(w)
                elif (
                    int(w.qtype) == int(quant_ops.QTYPE_Q4_K)
                    and lm_head_q4k_r4_enabled
                ):
                    repacked = quant_ops.repack_q4k_r4(w)
                if isinstance(repacked, quant_ops.QuantizedMatrix) and int(repacked.qtype) != int(w.qtype):
                    repacked_packed = repacked.ensure_packed()
                    if lm_head_pack_budget_bytes > 0:
                        repacked_bytes = int(getattr(repacked_packed, "nbytes", 0) or 0)
                        if repacked_bytes > 0 and (
                            lm_head_packed_bytes + repacked_bytes > lm_head_pack_budget_bytes
                        ):
                            self._weight_refs.append(packed)
                            return void_p(packed.ctypes.data), int(w.qtype)
                        lm_head_packed_bytes += max(0, repacked_bytes)
                    self._weight_refs.append(repacked_packed)
                    return void_p(repacked_packed.ctypes.data), int(repacked.qtype)

            if (
                q4k_r4_enabled
                and int(w.qtype) == quant_ops.QTYPE_Q4_K
            ):
                repacked = quant_ops.repack_q4k_r4(w)
                if isinstance(repacked, quant_ops.QuantizedMatrix) and repacked.qtype == quant_ops.QTYPE_Q4_K_R4:
                    repacked_packed = repacked.ensure_packed()
                    if q4k_r4_budget_bytes > 0:
                        repacked_bytes = int(getattr(repacked_packed, "nbytes", 0) or 0)
                        if repacked_bytes > 0 and (
                            q4k_r4_repacked_bytes + repacked_bytes > q4k_r4_budget_bytes
                        ):
                            self._weight_refs.append(packed)
                            return void_p(packed.ctypes.data), int(w.qtype)
                        q4k_r4_repacked_bytes += max(0, repacked_bytes)
                    self._weight_refs.append(repacked_packed)
                    return void_p(repacked_packed.ctypes.data), int(repacked.qtype)

            self._weight_refs.append(packed)
            return void_p(packed.ctypes.data), int(w.qtype)

        def _projection_output_dim(
            w,
            input_dim: int,
            default: int,
        ) -> int:
            if isinstance(w, quant_ops.QuantizedMatrix):
                return int(w.output_dim)
            arr = np.asarray(w) if w is not None else None
            if arr is None or arr.ndim != 2:
                return int(default)
            rows, cols = (int(arr.shape[0]), int(arr.shape[1]))
            if rows == input_dim and cols != input_dim:
                return cols
            if cols == input_dim and rows != input_dim:
                return rows
            return max(rows, cols) if max(rows, cols) > 0 else int(default)

        def _normalize_conv_kernel(
            weight: np.ndarray | quant_ops.QuantizedMatrix | None,
        ) -> tuple[Optional[np.ndarray], int, int]:
            if weight is None or isinstance(weight, quant_ops.QuantizedMatrix):
                return None, 0, 0
            kernel = np.asarray(weight, dtype=np.float32)
            if kernel.ndim != 2:
                return None, 0, 0
            if kernel.shape[0] <= 16 and kernel.shape[1] > kernel.shape[0]:
                out = np.ascontiguousarray(kernel, dtype=np.float32)
            elif kernel.shape[1] <= 16 and kernel.shape[0] > kernel.shape[1]:
                out = np.ascontiguousarray(kernel.T, dtype=np.float32)
            else:
                out = np.ascontiguousarray(kernel, dtype=np.float32)
            self._weight_refs.append(out)
            return out, int(out.shape[0]), int(out.shape[1])

        def _get_float_matrix_ptr(w) -> Optional[ctypes.POINTER(ctypes.c_float)]:
            if w is None or isinstance(w, quant_ops.QuantizedMatrix):
                return None
            arr = np.asarray(w, dtype=np.float32)
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self._weight_refs.append(arr)
            return arr.ctypes.data_as(float_p)

        supported_granite_qtypes = {
            int(quant_ops.QTYPE_Q8_0),
            int(quant_ops.QTYPE_Q4_K),
            int(quant_ops.QTYPE_Q4_K_R4),
            int(quant_ops.QTYPE_Q6_K),
        }

        def _is_supported_granite_qtype(qtype: int) -> bool:
            return int(qtype) in supported_granite_qtypes

        def _build_shared_expert_ptrs(
            gate_w,
            up_w,
            down_w,
        ) -> tuple[ctypes.c_void_p, int, int, ctypes.c_void_p, int, ctypes.c_void_p, int, bool, bool]:
            present = any(w is not None for w in (gate_w, up_w, down_w))
            if not present:
                return void_p(0), 0, 0, void_p(0), 0, void_p(0), 0, False, False
            if not (
                isinstance(gate_w, quant_ops.QuantizedMatrix)
                and isinstance(up_w, quant_ops.QuantizedMatrix)
                and isinstance(down_w, quant_ops.QuantizedMatrix)
            ):
                return void_p(0), 0, 0, void_p(0), 0, void_p(0), 0, True, False
            if not (
                _is_supported_granite_qtype(gate_w.qtype)
                and _is_supported_granite_qtype(up_w.qtype)
                and _is_supported_granite_qtype(down_w.qtype)
            ):
                return void_p(0), 0, 0, void_p(0), 0, void_p(0), 0, True, False
            hidden_dim = int(gate_w.output_dim)
            if (
                gate_w.input_dim != self.embedding_dim
                or up_w.input_dim != self.embedding_dim
                or gate_w.output_dim != up_w.output_dim
                or hidden_dim <= 0
                or down_w.input_dim != hidden_dim
                or down_w.output_dim != self.embedding_dim
            ):
                return void_p(0), 0, 0, void_p(0), 0, void_p(0), 0, True, False

            gate_ptr, gate_qt = _get_quant_info(gate_w)
            up_ptr, up_qt = _get_quant_info(up_w)
            down_ptr, down_qt = _get_quant_info(down_w)
            if not gate_ptr or not up_ptr or not down_ptr:
                return void_p(0), 0, 0, void_p(0), 0, void_p(0), 0, True, False
            return (
                gate_ptr or void_p(0),
                int(gate_qt),
                hidden_dim,
                up_ptr or void_p(0),
                int(up_qt),
                down_ptr or void_p(0),
                int(down_qt),
                True,
                True,
            )

        def _build_expert_ptrs(
            gate_name: str | None,
            up_name: str | None,
            down_name: str | None,
        ) -> tuple[Optional[ctypes.Array], int, int, Optional[ctypes.Array], int, Optional[ctypes.Array], int, int, bool, bool]:
            present = bool(gate_name or up_name or down_name)
            if not gate_name or not up_name or not down_name:
                return None, 0, 0, None, 0, None, 0, 0, present, False
            gate_tensor = ws._tensor_index.get(gate_name)
            up_tensor = ws._tensor_index.get(up_name)
            down_tensor = ws._tensor_index.get(down_name)
            tensors = (gate_tensor, up_tensor, down_tensor)
            if any(t is None or len(getattr(t, "shape", ())) != 3 for t in tensors):
                return None, 0, 0, None, 0, None, 0, 0, True, False
            expert_count = int(gate_tensor.shape[-1])
            if int(up_tensor.shape[-1]) != expert_count or int(down_tensor.shape[-1]) != expert_count:
                return None, 0, 0, None, 0, None, 0, 0, True, False
            if expert_count <= 0:
                return None, 0, 0, None, 0, None, 0, 0, True, False
            gate_ptrs: list[int] = []
            up_ptrs: list[int] = []
            down_ptrs: list[int] = []
            gate_qtype = 0
            up_qtype = 0
            down_qtype = 0
            hidden_dim = 0
            input_dim = 0
            output_dim = 0
            for idx in range(expert_count):
                g = ws.get_expert_matrix(gate_name, idx)
                u = ws.get_expert_matrix(up_name, idx)
                d = ws.get_expert_matrix(down_name, idx)
                if not isinstance(g, quant_ops.QuantizedMatrix):
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                if not isinstance(u, quant_ops.QuantizedMatrix):
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                if not isinstance(d, quant_ops.QuantizedMatrix):
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                if not (
                    _is_supported_granite_qtype(g.qtype)
                    and _is_supported_granite_qtype(u.qtype)
                    and _is_supported_granite_qtype(d.qtype)
                ):
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                current_input_dim = int(g.input_dim)
                current_hidden_dim = int(g.output_dim)
                current_output_dim = int(d.output_dim)
                if idx == 0:
                    input_dim = current_input_dim
                    hidden_dim = current_hidden_dim
                    output_dim = current_output_dim
                if (
                    current_input_dim != input_dim
                    or int(u.input_dim) != input_dim
                    or current_hidden_dim != hidden_dim
                    or int(u.output_dim) != hidden_dim
                    or int(d.input_dim) != hidden_dim
                    or current_output_dim != output_dim
                    or input_dim != self.embedding_dim
                    or output_dim != self.embedding_dim
                    or hidden_dim <= 0
                ):
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                g_ptr, g_qt = _get_quant_info(g)
                u_ptr, u_qt = _get_quant_info(u)
                d_ptr, d_qt = _get_quant_info(d)
                if not g_ptr or not u_ptr or not d_ptr:
                    return None, 0, 0, None, 0, None, 0, 0, True, False
                current_gate_qtype = int(g_qt)
                current_up_qtype = int(u_qt)
                current_down_qtype = int(d_qt)
                if idx == 0:
                    gate_qtype = current_gate_qtype
                    up_qtype = current_up_qtype
                    down_qtype = current_down_qtype
                gate_ptrs.append(int(ctypes.cast(g_ptr, ctypes.c_void_p).value or 0))
                up_ptrs.append(int(ctypes.cast(u_ptr, ctypes.c_void_p).value or 0))
                down_ptrs.append(int(ctypes.cast(d_ptr, ctypes.c_void_p).value or 0))
                if (
                    current_gate_qtype != gate_qtype
                    or current_up_qtype != up_qtype
                    or current_down_qtype != down_qtype
                ):
                    return None, 0, 0, None, 0, None, 0, 0, True, False

            gate_arr = (ctypes.c_void_p * expert_count)(*gate_ptrs)
            up_arr = (ctypes.c_void_p * expert_count)(*up_ptrs)
            down_arr = (ctypes.c_void_p * expert_count)(*down_ptrs)
            self._weight_refs.extend([gate_arr, up_arr, down_arr])
            return (
                gate_arr,
                gate_qtype,
                hidden_dim,
                up_arr,
                up_qtype,
                down_arr,
                down_qtype,
                expert_count,
                True,
                True,
            )

        def _is_quantized(w) -> bool:
            return isinstance(w, quant_ops.QuantizedMatrix)

        has_extended_api = bool(getattr(self, "_has_extended_api", False))

        if has_extended_api and hasattr(self._lib, "graph_set_qwen_mrope_config"):
            metadata = ws.loader.get_metadata()
            rope_dim = int(
                metadata.get("qwen35.rope.dimension_count", self.head_dim)
                or self.head_dim
            )
            rope_dim = max(2, min(self.head_dim, rope_dim - (rope_dim % 2)))
            raw_sections = (
                metadata.get("qwen35.rope.dimension_sections")
                or metadata.get("qwen35.mrope_sections")
                or metadata.get("qwen35.rope.mrope_section")
                or ()
            )
            sections: list[int] = []
            if isinstance(raw_sections, np.ndarray):
                raw_sections = raw_sections.tolist()
            if isinstance(raw_sections, (list, tuple)):
                for item in raw_sections:
                    if isinstance(item, np.ndarray):
                        item = item.tolist()
                    if isinstance(item, (list, tuple)):
                        sections.append(int(item[0]) if item else 0)
                    else:
                        sections.append(int(item))
            while len(sections) < 4:
                sections.append(0)
            self._lib.graph_set_qwen_mrope_config(
                void_p(self._handle),
                1 if _metadata_bool(metadata.get("qwen35.rope.scaling.finetuned"), True) else 0,
                1 if _metadata_bool(metadata.get("qwen35.rope.mrope_interleaved", False)) else 0,
                int(rope_dim),
                int(sections[0]),
                int(sections[1]),
                int(sections[2]),
                int(sections[3]),
            )
            if hasattr(self._lib, "graph_set_qwen_hybrid_config"):
                ssm_n_v_heads = metadata.get("qwen35.ssm.n_v_heads")
                if ssm_n_v_heads is None:
                    ssm_n_v_heads = metadata.get("qwen35.ssm.time_step_rank", 0)
                self._lib.graph_set_qwen_hybrid_config(
                    void_p(self._handle),
                    int(metadata.get("qwen35.ssm.state_size", 0) or 0),
                    int(metadata.get("qwen35.ssm.group_count", 0) or 0),
                    int(ssm_n_v_heads or 0),
                )

        for layer_idx in range(self.n_layers):
            weights = ws.get_layer_weights(layer_idx)
            layer_type = ws.get_layer_type(layer_idx)
            is_attention = layer_type in {"attention", "hybrid"}

            # Norm weights (always float32)
            attn_norm = _get_float_ptr(weights.get("attn_norm"))
            ffn_norm_w = weights.get("post_attn_norm")
            if ffn_norm_w is None:
                ffn_norm_w = weights.get("ffn_norm")
            ffn_norm = _get_float_ptr(ffn_norm_w)

            # Get raw weight references
            wq_raw = weights.get("attn_q")
            wk_raw = weights.get("attn_k")
            wv_raw = weights.get("attn_v")
            qkv_raw = weights.get("attn_qkv")
            wo_raw = weights.get("attn_output")
            wgate_raw = weights.get("ffn_gate")
            wup_raw = weights.get("ffn_up")
            wdown_raw = weights.get("ffn_down")
            attn_gate_raw = weights.get("attn_gate")
            attn_q_norm = _get_float_ptr(weights.get("attn_q_norm"))
            attn_k_norm = _get_float_ptr(weights.get("attn_k_norm"))

            q_out = self.n_heads * self.head_dim
            kv_out = self.n_kv_heads * self.head_dim
            if qkv_raw is not None and (wq_raw is None or wk_raw is None or wv_raw is None):
                sq, sk, sv = _split_fused_qkv_weight(qkv_raw, q_out=q_out, kv_out=kv_out)
                if wq_raw is None:
                    wq_raw = sq
                if wk_raw is None:
                    wk_raw = sk
                if wv_raw is None:
                    wv_raw = sv
            elif (
                profile.architecture == "qwen35"
                and qkv_raw is None
                and wq_raw is not None
                and attn_gate_raw is None
                and layer_type == "attention"
            ):
                sq, sg = _split_qwen_q_and_gate_weight(
                    wq_raw,
                    n_heads=self.n_heads,
                    head_dim=self.head_dim,
                )
                wq_raw = sq
                if sg is not None:
                    attn_gate_raw = sg

            q_out = _projection_output_dim(wq_raw, self.embedding_dim, q_out)
            kv_out = _projection_output_dim(
                wk_raw if wk_raw is not None else wv_raw,
                self.embedding_dim,
                kv_out,
            )

            # Determine FFN dim
            ffn_dim = 0
            if wgate_raw is not None:
                if isinstance(wgate_raw, quant_ops.QuantizedMatrix):
                    ffn_dim = wgate_raw.output_dim
                elif isinstance(wgate_raw, np.ndarray) and wgate_raw.ndim == 2:
                    ffn_dim = max(wgate_raw.shape)
            if ffn_dim <= 0:
                ffn_dim = 4 * self.embedding_dim

            # Check if any projection weights are quantized
            has_quant = any(
                _is_quantized(w) for w in [wq_raw, wk_raw, wv_raw, wo_raw,
                                            wgate_raw, wup_raw, wdown_raw]
            )

            if has_quant and self._has_quantized_api:
                # Use quantized API — zero-copy, no dequantization
                wq_ptr, wq_qt = _get_quant_info(wq_raw)
                wk_ptr, wk_qt = _get_quant_info(wk_raw)
                wv_ptr, wv_qt = _get_quant_info(wv_raw)
                wo_ptr, wo_qt = _get_quant_info(wo_raw)
                wgate_ptr, wgate_qt = _get_quant_info(wgate_raw)
                wup_ptr, wup_qt = _get_quant_info(wup_raw)
                wdown_ptr, wdown_qt = _get_quant_info(wdown_raw)

                self._lib.graph_set_layer_weights_quantized(
                    void_p(self._handle),
                    layer_idx,
                    attn_norm, ffn_norm,
                    wq_ptr or void_p(0), wq_qt, q_out,
                    wk_ptr or void_p(0), wk_qt, kv_out,
                    wv_ptr or void_p(0), wv_qt,
                    wo_ptr or void_p(0), wo_qt,
                    wgate_ptr or void_p(0), wgate_qt, ffn_dim,
                    wup_ptr or void_p(0), wup_qt,
                    wdown_ptr or void_p(0), wdown_qt,
                    1 if is_attention else 0,
                )
            else:
                # Float32 API (legacy or all-float32 weights)
                wq = _get_float_projection_ptr(wq_raw, rows=q_out, cols=self.embedding_dim)
                wk = _get_float_projection_ptr(wk_raw, rows=kv_out, cols=self.embedding_dim)
                wv = _get_float_projection_ptr(wv_raw, rows=kv_out, cols=self.embedding_dim)
                wo = _get_float_projection_ptr(wo_raw, rows=self.embedding_dim, cols=q_out)
                w_gate = _get_float_projection_ptr(wgate_raw, rows=ffn_dim, cols=self.embedding_dim)
                w_up = _get_float_projection_ptr(wup_raw, rows=ffn_dim, cols=self.embedding_dim)
                w_down = _get_float_projection_ptr(wdown_raw, rows=self.embedding_dim, cols=ffn_dim)

                self._lib.graph_set_layer_weights(
                    void_p(self._handle),
                    layer_idx,
                    attn_norm, ffn_norm,
                    wq, q_out,
                    wk, kv_out,
                    wv, wo,
                    w_gate, ffn_dim,
                    w_up, w_down,
                    1 if is_attention else 0,
                )

            # Determine C++ capability for this layer.
            # - attention_only: attention branch can run in C++.
            # - full_layer: attention + FFN can run in C++.
            gate_name, up_name, down_name = ws.get_expert_tensor_names(layer_idx)
            has_experts = bool(gate_name or up_name or down_name)
            granite_shared_present = False
            granite_shared_valid = False
            granite_routed_requested = False
            granite_routed_valid = False
            router_ptr = None
            shared_gate_ptr = void_p(0)
            shared_gate_qt = 0
            shared_hidden_dim = 0
            shared_up_ptr = void_p(0)
            shared_up_qt = 0
            shared_down_ptr = void_p(0)
            shared_down_qt = 0
            expert_gate_arr = None
            expert_gate_qt = 0
            expert_hidden_dim = 0
            expert_up_arr = None
            expert_up_qt = 0
            expert_down_arr = None
            expert_down_qt = 0
            expert_count = 0
            if profile.architecture == "granitehybrid":
                (
                    shared_gate_ptr,
                    shared_gate_qt,
                    shared_hidden_dim,
                    shared_up_ptr,
                    shared_up_qt,
                    shared_down_ptr,
                    shared_down_qt,
                    granite_shared_present,
                    granite_shared_valid,
                ) = _build_shared_expert_ptrs(
                    weights.get("ffn_gate_shexp"),
                    weights.get("ffn_up_shexp"),
                    weights.get("ffn_down_shexp"),
                )
                (
                    expert_gate_arr,
                    expert_gate_qt,
                    expert_hidden_dim,
                    expert_up_arr,
                    expert_up_qt,
                    expert_down_arr,
                    expert_down_qt,
                    expert_count,
                    granite_routed_requested,
                    granite_routed_valid,
                ) = _build_expert_ptrs(gate_name, up_name, down_name)
                router_present = weights.get("ffn_gate_inp") is not None
                granite_routed_requested = bool(granite_routed_requested or router_present)
                if granite_routed_valid and router_present:
                    router_ptr = _get_float_projection_ptr(
                        weights.get("ffn_gate_inp"),
                        rows=max(1, int(expert_count)),
                        cols=self.embedding_dim,
                    )
                    granite_routed_valid = router_ptr is not None and expert_count > 0
                else:
                    granite_routed_valid = False
                if granite_shared_present and not granite_shared_valid:
                    raise RuntimeError(
                        f"Granite layer {layer_idx} has an invalid shared expert bundle"
                    )
                if granite_routed_requested and not granite_routed_valid:
                    raise RuntimeError(
                        f"Granite layer {layer_idx} has an invalid routed expert bundle"
                    )
            has_simple_ffn = (wgate_raw is not None and wup_raw is not None and wdown_raw is not None)
            can_attention_cpp = (
                is_attention
                and attn_norm is not None
                and wq_raw is not None
                and wk_raw is not None
                and wv_raw is not None
                and wo_raw is not None
            )
            layer_can_cpp = False
            if profile.architecture == "qwen35":
                qwen_hybrid_quantized = (
                    isinstance(qkv_raw, quant_ops.QuantizedMatrix)
                    and isinstance(attn_gate_raw, quant_ops.QuantizedMatrix)
                    and isinstance(weights.get("ssm_alpha"), quant_ops.QuantizedMatrix)
                    and isinstance(weights.get("ssm_beta"), quant_ops.QuantizedMatrix)
                    and isinstance(weights.get("ssm_out"), quant_ops.QuantizedMatrix)
                )
                if layer_type == "attention":
                    layer_can_cpp = (
                        can_attention_cpp
                        and has_simple_ffn
                        and ffn_norm is not None
                        and has_extended_api
                    )
                elif layer_type == "hybrid":
                    layer_can_cpp = (
                        has_extended_api
                        and qkv_raw is not None
                        and attn_gate_raw is not None
                        and weights.get("ssm_alpha") is not None
                        and weights.get("ssm_beta") is not None
                        and weights.get("ssm_out") is not None
                        and weights.get("ssm_conv1d") is not None
                        and weights.get("ssm_dt") is not None
                        and weights.get("ssm_a") is not None
                        and weights.get("ssm_norm") is not None
                        and qwen_hybrid_quantized
                        and has_simple_ffn
                        and ffn_norm is not None
                    )
            elif profile.architecture == "granitehybrid":
                has_granite_shared = granite_shared_valid and has_extended_api
                has_granite_routed = granite_routed_valid and has_extended_api
                has_granite_moe = has_granite_shared or has_granite_routed
                if layer_type == "ssm":
                    layer_can_cpp = (
                        weights.get("ssm_in") is not None
                        and weights.get("ssm_out") is not None
                        and weights.get("ssm_conv1d") is not None
                        and weights.get("ssm_dt") is not None
                        and weights.get("ssm_a") is not None
                        and weights.get("ssm_d") is not None
                        and weights.get("ssm_norm") is not None
                        and has_granite_moe
                    )
                elif layer_type == "attention":
                    layer_can_cpp = can_attention_cpp and has_granite_moe
            else:
                layer_can_cpp = (
                    can_attention_cpp
                    and not has_experts
                    and has_simple_ffn
                    and ffn_norm is not None
                )
            needs_python_ffn = can_attention_cpp and not layer_can_cpp

            self._layer_cpp_attention_ok[layer_idx] = can_attention_cpp
            self._layer_cpp_ok[layer_idx] = layer_can_cpp
            self._layer_cpp_requires_python_ffn[layer_idx] = needs_python_ffn

            if not has_extended_api:
                continue

            layer_kind = 0
            if profile.architecture == "granitehybrid":
                if layer_type == "ssm":
                    layer_kind = 2
                elif layer_type == "attention":
                    layer_kind = 1
            elif profile.architecture == "qwen35":
                if layer_type == "hybrid":
                    layer_kind = 3
                elif layer_type == "attention":
                    layer_kind = 4

            attn_gate_ptr, attn_gate_qt = _get_quant_info(attn_gate_raw)
            attn_gate_dim = 0
            if isinstance(attn_gate_raw, quant_ops.QuantizedMatrix):
                attn_gate_dim = int(attn_gate_raw.output_dim)
            elif isinstance(attn_gate_raw, np.ndarray) and attn_gate_raw.ndim == 2:
                attn_gate_dim = int(max(attn_gate_raw.shape))

            special_ssm_in = weights.get("ssm_in")
            if profile.architecture == "qwen35" and layer_type == "hybrid":
                special_ssm_in = qkv_raw
            ssm_in_ptr, ssm_in_qt = _get_quant_info(special_ssm_in)
            ssm_in_dim = (
                int(special_ssm_in.output_dim)
                if isinstance(special_ssm_in, quant_ops.QuantizedMatrix)
                else 0
            )
            ssm_out_ptr, ssm_out_qt = _get_quant_info(weights.get("ssm_out"))
            ssm_out_dim = (
                int(weights["ssm_out"].output_dim)
                if isinstance(weights.get("ssm_out"), quant_ops.QuantizedMatrix)
                else 0
            )
            ssm_alpha_ptr, ssm_alpha_qt = _get_quant_info(weights.get("ssm_alpha"))
            ssm_alpha_dim = (
                int(weights["ssm_alpha"].output_dim)
                if isinstance(weights.get("ssm_alpha"), quant_ops.QuantizedMatrix)
                else 0
            )
            ssm_beta_ptr, ssm_beta_qt = _get_quant_info(weights.get("ssm_beta"))
            ssm_beta_dim = (
                int(weights["ssm_beta"].output_dim)
                if isinstance(weights.get("ssm_beta"), quant_ops.QuantizedMatrix)
                else 0
            )

            conv_kernel, conv_rows, conv_cols = _normalize_conv_kernel(weights.get("ssm_conv1d"))
            conv_ptr = (
                conv_kernel.ctypes.data_as(float_p)
                if conv_kernel is not None
                else None
            )
            conv_bias = _get_float_matrix_ptr(weights.get("ssm_conv1d_bias"))

            metadata = ws.loader.get_metadata()
            meta_top_k = int(metadata.get(f"{profile.architecture}.expert_used_count", 1) or 1)
            granite_cap_raw = os.getenv("ANVIL_GRANITE_MAX_MOE_TOP_K")
            granite_cap = 2 if granite_cap_raw is None else int(granite_cap_raw or "0")
            moe_top_k = 0
            if profile.architecture == "granitehybrid" and granite_routed_valid and expert_count > 0:
                desired_top_k = max(1, meta_top_k)
                if granite_cap > 0:
                    desired_top_k = min(desired_top_k, granite_cap)
                moe_top_k = min(desired_top_k, int(expert_count))

            self._lib.graph_set_layer_extras(
                void_p(self._handle),
                layer_idx,
                int(layer_kind),
                attn_q_norm,
                attn_k_norm,
                _get_float_ptr(weights.get("ssm_a")),
                _get_float_ptr(weights.get("ssm_d")),
                _get_float_ptr(weights.get("ssm_dt")),
                conv_ptr,
                int(conv_rows),
                int(conv_cols),
                conv_bias,
                _get_float_ptr(weights.get("ssm_norm")),
                router_ptr,
                attn_gate_ptr or void_p(0),
                int(attn_gate_qt),
                int(attn_gate_dim),
                ssm_in_ptr or void_p(0),
                int(ssm_in_qt),
                int(ssm_in_dim),
                ssm_out_ptr or void_p(0),
                int(ssm_out_qt),
                int(ssm_out_dim),
                ssm_alpha_ptr or void_p(0),
                int(ssm_alpha_qt),
                int(ssm_alpha_dim),
                ssm_beta_ptr or void_p(0),
                int(ssm_beta_qt),
                int(ssm_beta_dim),
                shared_gate_ptr or void_p(0),
                int(shared_gate_qt),
                int(shared_hidden_dim),
                shared_up_ptr or void_p(0),
                int(shared_up_qt),
                shared_down_ptr or void_p(0),
                int(shared_down_qt),
                ctypes.cast(expert_gate_arr, ctypes.POINTER(void_p)) if expert_gate_arr is not None else ctypes.POINTER(void_p)(),
                int(expert_gate_qt),
                int(expert_hidden_dim),
                ctypes.cast(expert_up_arr, ctypes.POINTER(void_p)) if expert_up_arr is not None else ctypes.POINTER(void_p)(),
                int(expert_up_qt),
                ctypes.cast(expert_down_arr, ctypes.POINTER(void_p)) if expert_down_arr is not None else ctypes.POINTER(void_p)(),
                int(expert_down_qt),
                int(expert_count),
                int(moe_top_k),
            )

        # Final norm
        final_norm_w = ws.get_tensor("output_norm.weight")
        final_norm = _get_float_ptr(final_norm_w)

        # LM head
        emb_weights = ws.get_embedding_weights()
        token_embd = emb_weights.get("token_embd")
        lm_w = emb_weights.get("output")

        metadata = ws.loader.get_metadata()
        arch = profile.architecture
        embedding_scale = float(metadata.get(f"{arch}.embedding_scale", 0.0) or 0.0)
        residual_scale = float(metadata.get(f"{arch}.residual_scale", 0.0) or 0.0)
        logit_scale = float(metadata.get(f"{arch}.logit_scale", 0.0) or 0.0)
        attention_scale = float(metadata.get(f"{arch}.attention.scale", 0.0) or 0.0)

        if isinstance(token_embd, np.ndarray) and token_embd.ndim == 2 and hasattr(
            self._lib, "graph_set_embedding_weights"
        ):
            token_embd_arr = np.asarray(token_embd, dtype=np.float32)
            if not token_embd_arr.flags["C_CONTIGUOUS"]:
                token_embd_arr = np.ascontiguousarray(token_embd_arr)
            self._weight_refs.append(token_embd_arr)
            token_embd_transposed = 1 if token_embd_arr.shape[0] == self.embedding_dim else 0
            self._lib.graph_set_embedding_weights(
                void_p(self._handle),
                token_embd_arr.ctypes.data_as(float_p),
                token_embd_transposed,
            )
        elif (
            _is_quantized(token_embd)
            and self._has_quantized_api
            and hasattr(self._lib, "graph_set_embedding_weights_quantized")
        ):
            token_embd_ptr, token_embd_qt = _get_quant_info(token_embd)
            token_embd_input_dim = int(token_embd.input_dim)
            token_embd_output_dim = int(token_embd.output_dim)
            token_embd_transposed = int(
                token_embd_input_dim == self.embedding_dim
                and token_embd_output_dim == self.vocab_size
            )
            if not token_embd_transposed:
                raise RuntimeError(
                    "Strict native graph requires quantized token embeddings in "
                    "[embedding_dim, vocab_size] layout."
                )
            self._lib.graph_set_embedding_weights_quantized(
                void_p(self._handle),
                token_embd_ptr or void_p(0),
                int(token_embd_qt),
                token_embd_input_dim,
                token_embd_output_dim,
                token_embd_transposed,
            )

        if _is_quantized(lm_w) and self._has_quantized_api:
            lm_ptr, lm_qt = _get_quant_info(lm_w, prefer_decode_pack=True)
            self._lm_head_qtype = int(lm_qt)
            if int(lm_qt) == int(getattr(quant_ops, "QTYPE_Q4_K_R4", -1)):
                self._lm_head_layout = "q4_k_r4"
            elif int(lm_qt) == int(getattr(quant_ops, "QTYPE_Q6_K_LM", -1)):
                self._lm_head_layout = "q6_k_lm"
            elif int(lm_qt) == int(getattr(quant_ops, "QTYPE_Q6_K_R4", -1)):
                self._lm_head_layout = "q6_k_r4"
            else:
                self._lm_head_layout = f"qtype_{int(lm_qt)}"
            self._lib.graph_set_head_weights_quantized(
                void_p(self._handle),
                final_norm,
                lm_ptr or void_p(0), lm_qt,
                ctypes.c_float(embedding_scale),
                ctypes.c_float(residual_scale),
                ctypes.c_float(logit_scale),
                ctypes.c_float(attention_scale),
            )
        else:
            lm_ptr = _get_float_ptr(lm_w)
            lm_transposed = 0
            if isinstance(lm_w, np.ndarray) and lm_w.ndim == 2:
                if lm_w.shape[0] == self.embedding_dim:
                    lm_transposed = 1
            self._lm_head_qtype = 0
            self._lm_head_layout = "f32_transposed" if lm_transposed else "f32"

            self._lib.graph_set_head_weights(
                void_p(self._handle),
                final_norm, lm_ptr, lm_transposed,
                ctypes.c_float(embedding_scale),
                ctypes.c_float(residual_scale),
                ctypes.c_float(logit_scale),
                ctypes.c_float(attention_scale),
            )

    def forward_token(self, embedding: np.ndarray, position: int) -> Optional[np.ndarray]:
        """Execute full forward pass for a single token. Returns logits."""
        if not self.has_full_graph:
            return None

        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.embedding_dim:
            return None
        if not emb.flags["C_CONTIGUOUS"]:
            emb = np.ascontiguousarray(emb)

        out = np.empty((self.vocab_size,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_forward_token(
            ctypes.c_void_p(self._handle),
            emb.ctypes.data_as(float_p),
            self.embedding_dim,
            out.ctypes.data_as(float_p),
            self.vocab_size,
            int(position),
        )
        return out if int(ok) == 1 else None

    def forward_token_id(
        self,
        token_id: int,
        position: int,
    ) -> Optional[Any]:
        """Execute full forward pass for a single token id. Returns logits."""
        if not self.has_full_graph or not hasattr(self._lib, "graph_forward_token_id"):
            return None

        if self._has_drift_snapshot_arg_api:
            snapshot = _GraphDriftSnapshot()
            ok = self._lib.graph_forward_token_id(
                ctypes.c_void_p(self._handle),
                int(token_id),
                self._logits_out,
                self.vocab_size,
                int(position),
                ctypes.byref(snapshot),
            )
            if int(ok) == 1:
                self._last_drift_snapshot = snapshot
                self._last_drift_snapshot_valid = True
        else:
            ok = self._lib.graph_forward_token_id(
                ctypes.c_void_p(self._handle),
                int(token_id),
                self._logits_out,
                self.vocab_size,
                int(position),
            )
            if int(ok) == 1:
                self._update_last_drift_snapshot()
        if int(ok) != 1:
            return None
        return self._logits_out

    def forward_token_ids(self, token_ids: list[int], start_pos: int) -> Optional[Any]:
        """Execute a contiguous batch of token ids and return final-position logits."""
        if not self.has_full_graph or not self._has_batch_token_id_api:
            return None
        if not token_ids:
            return None

        token_count = int(len(token_ids))
        if self._token_id_buf is None or token_count > self._token_id_buf_capacity:
            self._token_id_buf_capacity = max(token_count, self._token_id_buf_capacity * 2, 8)
            self._token_id_buf = (ctypes.c_int * self._token_id_buf_capacity)()
        token_buf = self._token_id_buf
        for idx, token_id in enumerate(token_ids):
            token_buf[idx] = int(token_id)
        if self._has_drift_snapshot_arg_api:
            snapshot = _GraphDriftSnapshot()
            ok = self._lib.graph_forward_token_ids(
                ctypes.c_void_p(self._handle),
                token_buf,
                token_count,
                self._logits_out,
                self.vocab_size,
                int(start_pos),
                ctypes.byref(snapshot),
            )
            if int(ok) == 1:
                self._last_drift_snapshot = snapshot
                self._last_drift_snapshot_valid = True
        else:
            ok = self._lib.graph_forward_token_ids(
                ctypes.c_void_p(self._handle),
                token_buf,
                token_count,
                self._logits_out,
                self.vocab_size,
                int(start_pos),
            )
            if int(ok) == 1:
                self._update_last_drift_snapshot()
        if int(ok) != 1:
            return None
        return self._logits_out

    def forward_layer_inplace(
        self, hidden: np.ndarray, layer_idx: int, position: int,
    ) -> bool:
        """Execute a single attention+FFN layer in C++.

        ``hidden`` is modified in-place.
        Returns True on success.
        """
        if not self.has_full_graph or not self._has_layer_api:
            return False
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.embedding_dim:
            return False
        if not h.flags["C_CONTIGUOUS"]:
            return False
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_forward_layer(
            ctypes.c_void_p(self._handle),
            h.ctypes.data_as(float_p),
            self.embedding_dim,
            int(layer_idx),
            int(position),
        )
        return int(ok) == 1

    def forward_head(self, hidden: np.ndarray) -> Optional[np.ndarray]:
        """Execute final RMSNorm + LM head projection. Returns logits."""
        if not self.has_full_graph or not self._has_layer_api:
            return None
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.embedding_dim:
            return None
        if not h.flags["C_CONTIGUOUS"]:
            h = np.ascontiguousarray(h)
        out = np.empty((self.vocab_size,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_forward_head(
            ctypes.c_void_p(self._handle),
            h.ctypes.data_as(float_p),
            self.embedding_dim,
            out.ctypes.data_as(float_p),
            self.vocab_size,
        )
        return out if int(ok) == 1 else None

    def copy_last_hidden(self) -> Optional[np.ndarray]:
        """Copy the most recent decode hidden state from the native graph."""
        if not self.has_full_graph or not self._has_last_hidden_api:
            return None
        out = np.empty((self.embedding_dim,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_copy_last_hidden(
            ctypes.c_void_p(self._handle),
            out.ctypes.data_as(float_p),
            self.embedding_dim,
        )
        return out if int(ok) == 1 else None

    @property
    def supports_execution_checkpoint(self) -> bool:
        return self.has_full_graph and self._has_execution_checkpoint_api

    @property
    def supports_exit_continuation(self) -> bool:
        return self.has_full_graph and self._has_exit_continuation_api

    def create_execution_checkpoint(self) -> Optional[int]:
        if not self.supports_execution_checkpoint:
            return None
        handle = self._lib.graph_create_execution_checkpoint(  # type: ignore[union-attr]
            ctypes.c_void_p(self._handle)
        )
        if not handle:
            return None
        return int(handle)

    def restore_execution_checkpoint(self, checkpoint_handle: int) -> bool:
        if not self.supports_execution_checkpoint:
            return False
        ok = self._lib.graph_restore_execution_checkpoint(  # type: ignore[union-attr]
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(int(checkpoint_handle)),
        )
        return int(ok) == 1

    def destroy_execution_checkpoint(self, checkpoint_handle: int) -> None:
        if not self.supports_execution_checkpoint:
            return
        self._lib.graph_destroy_execution_checkpoint(  # type: ignore[union-attr]
            ctypes.c_void_p(int(checkpoint_handle))
        )

    def forward_token_id_to_exit(
        self,
        token_id: int,
        exit_layer: int,
        position: int,
    ) -> Optional[np.ndarray]:
        if not self.supports_exit_continuation:
            return None
        out = np.empty((self.embedding_dim,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_forward_token_id_to_exit(  # type: ignore[union-attr]
            ctypes.c_void_p(self._handle),
            int(token_id),
            int(exit_layer),
            out.ctypes.data_as(float_p),
            self.embedding_dim,
            int(position),
        )
        return out if int(ok) == 1 else None

    def continue_from_hidden(
        self,
        hidden: np.ndarray,
        *,
        start_layer: int,
        position: int,
    ) -> Optional[np.ndarray]:
        if not self.supports_exit_continuation:
            return None
        h = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if h.shape[0] != self.embedding_dim:
            return None
        if not h.flags["C_CONTIGUOUS"]:
            h = np.ascontiguousarray(h)
        out = np.empty((self.vocab_size,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_continue_from_hidden(  # type: ignore[union-attr]
            ctypes.c_void_p(self._handle),
            h.ctypes.data_as(float_p),
            self.embedding_dim,
            int(start_layer),
            out.ctypes.data_as(float_p),
            self.vocab_size,
            int(position),
        )
        if int(ok) != 1:
            return None
        return out

    def can_layer_cpp(self, layer_idx: int) -> bool:
        """True if layer_idx can be executed in C++ (attention + simple FFN)."""
        if 0 <= layer_idx < len(self._layer_cpp_ok):
            return self._layer_cpp_ok[layer_idx]
        return False

    def can_layer_cpp_attention(self, layer_idx: int) -> bool:
        """True if layer_idx attention branch can be executed in C++."""
        if 0 <= layer_idx < len(self._layer_cpp_attention_ok):
            return self._layer_cpp_attention_ok[layer_idx]
        return False

    def layer_requires_python_ffn(self, layer_idx: int) -> bool:
        """True if C++ handles attention but FFN must run in Python."""
        if 0 <= layer_idx < len(self._layer_cpp_requires_python_ffn):
            return self._layer_cpp_requires_python_ffn[layer_idx]
        return False

    @property
    def has_hybrid_mode(self) -> bool:
        """True if per-layer hybrid execution is available."""
        return self.has_full_graph and self._has_layer_api

    def reset(self) -> None:
        """Reset KV cache and position."""
        if self.has_full_graph:
            self._lib.graph_reset(ctypes.c_void_p(self._handle))
            self._last_drift_snapshot = _GraphDriftSnapshot()
            self._last_drift_snapshot_valid = False

    def reset_perf_stats(self) -> bool:
        """Reset perf counters only, preserving KV/SSM state and position."""
        if (
            not self.has_full_graph
            or not self._has_perf_stats_reset_api
            or not hasattr(self._lib, "graph_reset_perf_stats")
        ):
            return False
        ok = self._lib.graph_reset_perf_stats(ctypes.c_void_p(self._handle))
        return int(ok) == 1

    def _update_last_drift_snapshot(self) -> None:
        if (
            not self.has_full_graph
            or not self._has_drift_state_api
            or not hasattr(self._lib, "graph_get_last_drift_snapshot")
        ):
            return
        snapshot = _GraphDriftSnapshot()
        ok = self._lib.graph_get_last_drift_snapshot(
            ctypes.c_void_p(self._handle),
            ctypes.byref(snapshot),
        )
        if int(ok) == 1:
            self._last_drift_snapshot = snapshot
            self._last_drift_snapshot_valid = True

    def set_drift_config(self, config: dict[str, int | float]) -> bool:
        if (
            not self.has_full_graph
            or not self._has_drift_config_api
            or not hasattr(self._lib, "graph_set_drift_config")
        ):
            return False
        cfg = _GraphDriftConfig()
        cfg.enabled = int(config.get("enabled", 1))
        cfg.mode = int(config.get("mode", 0))
        cfg.block_size_tokens = int(config.get("block_size_tokens", 128))
        cfg.update_interval_tokens = int(config.get("update_interval_tokens", 64))
        cfg.prune_interval_tokens = int(config.get("prune_interval_tokens", 128))
        cfg.preserve_head_tokens = int(config.get("preserve_head_tokens", 256))
        cfg.preserve_recent_tokens = int(config.get("preserve_recent_tokens", 8192))
        cfg.min_active_tokens = int(config.get("min_active_tokens", 16384))
        cfg.damp_threshold = float(config.get("damp_threshold", 0.35))
        cfg.prune_threshold = float(config.get("prune_threshold", 0.72))
        cfg.damping_strength = float(config.get("damping_strength", 1.2))
        cfg.hysteresis = float(config.get("hysteresis", 0.05))
        ok = self._lib.graph_set_drift_config(
            ctypes.c_void_p(self._handle),
            ctypes.byref(cfg),
        )
        return int(ok) == 1

    def get_drift_config(self) -> dict[str, int | float]:
        if (
            not self.has_full_graph
            or not self._has_drift_get_config_api
            or not hasattr(self._lib, "graph_get_drift_config")
        ):
            return {}
        cfg = _GraphDriftConfig()
        ok = self._lib.graph_get_drift_config(
            ctypes.c_void_p(self._handle),
            ctypes.byref(cfg),
        )
        if int(ok) != 1:
            return {}
        return {
            "enabled": int(cfg.enabled),
            "mode": int(cfg.mode),
            "block_size_tokens": int(cfg.block_size_tokens),
            "update_interval_tokens": int(cfg.update_interval_tokens),
            "prune_interval_tokens": int(cfg.prune_interval_tokens),
            "preserve_head_tokens": int(cfg.preserve_head_tokens),
            "preserve_recent_tokens": int(cfg.preserve_recent_tokens),
            "min_active_tokens": int(cfg.min_active_tokens),
            "damp_threshold": float(cfg.damp_threshold),
            "prune_threshold": float(cfg.prune_threshold),
            "damping_strength": float(cfg.damping_strength),
            "hysteresis": float(cfg.hysteresis),
        }

    def get_last_drift_snapshot(self) -> dict[str, int | float]:
        if not self.has_full_graph:
            return {}
        self._update_last_drift_snapshot()
        if not self._last_drift_snapshot_valid:
            return {}
        return _graph_drift_snapshot_dict(self._last_drift_snapshot)

    def get_perf_stats(self) -> dict[str, int | float]:
        """Return accumulated native graph stage timings and call counts."""
        if not self.has_full_graph or not self._has_perf_stats_api:
            return {}
        snapshot = _GraphPerfStatsSnapshot()
        ok = self._lib.graph_get_perf_stats(
            ctypes.c_void_p(self._handle),
            ctypes.byref(snapshot),
        )
        if int(ok) != 1:
            return {}
        return {
            "embedding_lookup_seconds": float(snapshot.embedding_lookup_seconds),
            "attention_proj_seconds": float(snapshot.attention_proj_seconds),
            "attention_rope_kv_seconds": float(snapshot.attention_rope_kv_seconds),
            "attention_decode_seconds": float(snapshot.attention_decode_seconds),
            "attention_out_proj_seconds": float(snapshot.attention_out_proj_seconds),
            "ffn_norm_seconds": float(snapshot.ffn_norm_seconds),
            "ffn_gate_up_seconds": float(snapshot.ffn_gate_up_seconds),
            "ffn_down_seconds": float(snapshot.ffn_down_seconds),
            "ssm_projection_seconds": float(snapshot.ssm_projection_seconds),
            "ssm_conv_seconds": float(snapshot.ssm_conv_seconds),
            "ssm_recurrent_seconds": float(snapshot.ssm_recurrent_seconds),
            "ssm_output_seconds": float(snapshot.ssm_output_seconds),
            "ssm_seconds": float(snapshot.ssm_seconds),
            "moe_seconds": float(snapshot.moe_seconds),
            "final_norm_seconds": float(snapshot.final_norm_seconds),
            "lm_head_seconds": float(snapshot.lm_head_seconds),
            "sanitize_seconds": float(snapshot.sanitize_seconds),
            "forward_token_calls": int(snapshot.forward_token_calls),
            "forward_token_id_calls": int(snapshot.forward_token_id_calls),
            "forward_token_ids_calls": int(snapshot.forward_token_ids_calls),
            "forward_token_ids_token_count": int(snapshot.forward_token_ids_token_count),
            "attention_calls": int(snapshot.attention_calls),
            "ffn_calls": int(snapshot.ffn_calls),
            "ssm_calls": int(snapshot.ssm_calls),
            "moe_calls": int(snapshot.moe_calls),
            "packed_lm_head_calls": int(snapshot.packed_lm_head_calls),
            "attention_proj_bytes": int(snapshot.attention_proj_bytes),
            "attention_proj_flops": int(snapshot.attention_proj_flops),
            "attention_out_proj_bytes": int(snapshot.attention_out_proj_bytes),
            "attention_out_proj_flops": int(snapshot.attention_out_proj_flops),
            "ffn_gate_up_bytes": int(snapshot.ffn_gate_up_bytes),
            "ffn_gate_up_flops": int(snapshot.ffn_gate_up_flops),
            "ffn_down_bytes": int(snapshot.ffn_down_bytes),
            "ffn_down_flops": int(snapshot.ffn_down_flops),
            "ssm_projection_bytes": int(snapshot.ssm_projection_bytes),
            "ssm_projection_flops": int(snapshot.ssm_projection_flops),
            "ssm_output_bytes": int(snapshot.ssm_output_bytes),
            "ssm_output_flops": int(snapshot.ssm_output_flops),
            "moe_bytes": int(snapshot.moe_bytes),
            "moe_flops": int(snapshot.moe_flops),
            "lm_head_bytes": int(snapshot.lm_head_bytes),
            "lm_head_flops": int(snapshot.lm_head_flops),
        }

    def get_position(self) -> int:
        """Get current sequence position."""
        if self.has_full_graph:
            return int(self._lib.graph_get_position(ctypes.c_void_p(self._handle)))
        return 0

    def forward(self, hidden: np.ndarray) -> Optional[np.ndarray]:
        """Legacy: project hidden state to logits (no layer execution)."""
        if not self.available:
            return None
        hidden_f = np.asarray(hidden, dtype=np.float32).reshape(-1)
        if hidden_f.shape[0] != self.embedding_dim:
            return None
        if not hidden_f.flags["C_CONTIGUOUS"]:
            hidden_f = np.ascontiguousarray(hidden_f)

        out = np.empty((self.vocab_size,), dtype=np.float32)
        float_p = ctypes.POINTER(ctypes.c_float)
        ok = self._lib.graph_forward(
            ctypes.c_void_p(self._handle),
            hidden_f.ctypes.data_as(float_p),
            self.embedding_dim,
            out.ctypes.data_as(float_p),
            self.vocab_size,
        )
        return out if int(ok) == 1 else None

    def close(self) -> None:
        if not self.available:
            return
        self._lib.destroy_model_graph(ctypes.c_void_p(self._handle))
        self._handle = None
        self._weight_refs.clear()
        self._last_drift_snapshot = _GraphDriftSnapshot()
        self._last_drift_snapshot_valid = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
