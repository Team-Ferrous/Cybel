from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import numpy as np
import pytest

from core.native.native_ops import load_native_library


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


def _deterministic_weight(row: int, col: int, dim: int) -> float:
    x = (row * 1315423911) ^ ((col * 2654435761 + 2246822519) & 0xFFFFFFFF)
    bucket = x & 0xFFFF
    centered = (float(bucket) / 32768.0) - 1.0
    return centered * (1.0 / math.sqrt(float(dim)))


@dataclass
class _PythonFallbackModelGraph:
    n_layers: int
    embedding_dim: int
    vocab_size: int

    def __post_init__(self) -> None:
        self._layer_mix = np.asarray(
            [0.01 * float((i % 7) + 1) for i in range(self.n_layers)],
            dtype=np.float32,
        )
        self._projection = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)
        for r in range(self.vocab_size):
            for c in range(self.embedding_dim):
                self._projection[r, c] = _deterministic_weight(r, c, self.embedding_dim)
        self._bias = np.asarray(
            [0.001 * float((r % 13) - 6) for r in range(self.vocab_size)],
            dtype=np.float32,
        )

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        state = np.asarray(hidden, dtype=np.float32).reshape(-1).copy()
        if state.shape[0] != self.embedding_dim:
            raise ValueError("hidden size mismatch")
        for layer_idx in range(self.n_layers):
            shift = (layer_idx * 7 + 1) % self.embedding_dim
            rolled = np.roll(state, -shift)
            mix = self._layer_mix[layer_idx]
            state = state + mix * (0.5 * state + 0.5 * rolled)
        return (self._projection @ state + self._bias).astype(np.float32)

    def close(self) -> None:
        return


class _NativeCtypesModelGraph:
    def __init__(self, lib: ctypes.CDLL, n_layers: int, embedding_dim: int, vocab_size: int):
        self._lib = lib
        self._n_layers = int(n_layers)
        self._embedding_dim = int(embedding_dim)
        self._vocab_size = int(vocab_size)
        self._lib.create_model_graph.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.create_model_graph.restype = ctypes.c_void_p
        self._lib.graph_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self._lib.graph_forward.restype = ctypes.c_int
        self._lib.destroy_model_graph.argtypes = [ctypes.c_void_p]
        self._lib.destroy_model_graph.restype = None
        self._handle = self._lib.create_model_graph(
            self._n_layers,
            self._embedding_dim,
            self._vocab_size,
        )
        if not self._handle:
            raise RuntimeError("create_model_graph returned null")

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        vec = np.asarray(hidden, dtype=np.float32).reshape(-1)
        out = np.zeros((self._vocab_size,), dtype=np.float32)
        ok = self._lib.graph_forward(
            ctypes.c_void_p(self._handle),
            vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(vec.shape[0]),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(out.shape[0]),
        )
        if ok != 1:
            raise RuntimeError("graph_forward failed")
        return out

    def close(self) -> None:
        if self._handle:
            self._lib.destroy_model_graph(ctypes.c_void_p(self._handle))
            self._handle = None


def _load_native_graph_if_available(n_layers: int, embedding_dim: int, vocab_size: int):
    try:
        lib = load_native_library()
    except Exception:
        return None
    required = ("create_model_graph", "graph_forward", "destroy_model_graph")
    if not all(hasattr(lib, symbol) for symbol in required):
        return None
    return _NativeCtypesModelGraph(lib, n_layers, embedding_dim, vocab_size)


def _build_graph(n_layers: int, embedding_dim: int, vocab_size: int):
    native = _load_native_graph_if_available(n_layers, embedding_dim, vocab_size)
    if native is not None:
        return native, "native"
    return _PythonFallbackModelGraph(n_layers, embedding_dim, vocab_size), "fallback"


def test_model_graph_forward_shape_and_deterministic_output() -> None:
    graph, _backend = _build_graph(n_layers=3, embedding_dim=8, vocab_size=16)
    hidden = np.linspace(-0.75, 0.75, 8, dtype=np.float32)
    try:
        out1 = graph.forward(hidden)
        out2 = graph.forward(hidden)
    finally:
        graph.close()

    assert out1.shape == (16,)
    assert out1.dtype == np.float32
    assert np.isfinite(out1).all()
    np.testing.assert_allclose(out1, out2, rtol=0.0, atol=0.0)


def test_model_graph_native_ctypes_symbols_if_built() -> None:
    graph = _load_native_graph_if_available(n_layers=2, embedding_dim=6, vocab_size=10)
    if graph is None:
        pytest.skip("Native model_graph symbols are not built into the current shared library.")

    hidden = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32)
    try:
        out = graph.forward(hidden)
    finally:
        graph.close()

    assert out.shape == (10,)
    assert np.isfinite(out).all()


def test_model_graph_native_drift_abi_smoke_if_built() -> None:
    try:
        lib = load_native_library()
    except Exception:
        pytest.skip("Native model_graph symbols are not built into the current shared library.")

    required = (
        "create_model_graph",
        "destroy_model_graph",
        "graph_set_embedding_weights",
        "graph_set_head_weights",
        "graph_forward_token_id",
        "graph_forward_token_ids",
        "graph_set_drift_config",
        "graph_get_drift_config",
        "graph_get_last_drift_snapshot",
        "graph_reset",
    )
    missing = [name for name in required if not hasattr(lib, name)]
    if missing:
        pytest.skip(f"Native drift ABI symbols missing from current shared library: {', '.join(missing)}")

    lib.create_model_graph.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.create_model_graph.restype = ctypes.c_void_p
    lib.destroy_model_graph.argtypes = [ctypes.c_void_p]
    lib.destroy_model_graph.restype = None
    lib.graph_set_embedding_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.graph_set_embedding_weights.restype = ctypes.c_int
    lib.graph_set_head_weights.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.graph_set_head_weights.restype = ctypes.c_int
    lib.graph_set_drift_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GraphDriftConfig)]
    lib.graph_set_drift_config.restype = ctypes.c_int
    lib.graph_get_drift_config.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GraphDriftConfig)]
    lib.graph_get_drift_config.restype = ctypes.c_int
    lib.graph_get_last_drift_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(_GraphDriftSnapshot)]
    lib.graph_get_last_drift_snapshot.restype = ctypes.c_int
    lib.graph_forward_token_id.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_GraphDriftSnapshot),
    ]
    lib.graph_forward_token_id.restype = ctypes.c_int
    lib.graph_forward_token_ids.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_GraphDriftSnapshot),
    ]
    lib.graph_forward_token_ids.restype = ctypes.c_int
    lib.graph_reset.argtypes = [ctypes.c_void_p]
    lib.graph_reset.restype = ctypes.c_int

    n_layers = 1
    dim = 4
    vocab = 6
    handle = lib.create_model_graph(n_layers, dim, vocab)
    if not handle:
        pytest.skip("create_model_graph returned null.")

    embeddings = np.asarray(
        [[0.05 * float((r + 1) * (c + 1)) for c in range(dim)] for r in range(vocab)],
        dtype=np.float32,
    )
    final_norm = np.ones((dim,), dtype=np.float32)
    lm_head = np.asarray(
        [[_deterministic_weight(r, c, dim) for c in range(dim)] for r in range(vocab)],
        dtype=np.float32,
    )
    logits = np.zeros((vocab,), dtype=np.float32)

    cfg = _GraphDriftConfig(
        enabled=1,
        mode=2,
        block_size_tokens=1,
        update_interval_tokens=1,
        prune_interval_tokens=1,
        preserve_head_tokens=0,
        preserve_recent_tokens=0,
        min_active_tokens=0,
        damp_threshold=0.1,
        prune_threshold=0.9,
        damping_strength=1.2,
        hysteresis=0.05,
    )
    out_cfg = _GraphDriftConfig()
    drift_from_forward = _GraphDriftSnapshot()
    drift_from_batch = _GraphDriftSnapshot()
    drift_last = _GraphDriftSnapshot()
    token_ids = np.asarray([2, 3], dtype=np.int32)

    try:
        assert (
            lib.graph_set_embedding_weights(
                ctypes.c_void_p(handle),
                embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0,
            )
            == 1
        )
        assert (
            lib.graph_set_head_weights(
                ctypes.c_void_p(handle),
                final_norm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                lm_head.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0,
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
                ctypes.c_float(0.0),
            )
            == 1
        )
        assert lib.graph_set_drift_config(ctypes.c_void_p(handle), ctypes.byref(cfg)) == 1
        assert lib.graph_get_drift_config(ctypes.c_void_p(handle), ctypes.byref(out_cfg)) == 1
        assert out_cfg.enabled == 1
        assert out_cfg.mode == 2
        assert out_cfg.block_size_tokens == 1
        assert out_cfg.update_interval_tokens == 1
        assert (
            lib.graph_forward_token_id(
                ctypes.c_void_p(handle),
                1,
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(logits.shape[0]),
                0,
                ctypes.byref(drift_from_forward),
            )
            == 1
        )
        assert np.isfinite(logits).all()
        assert drift_from_forward.stabilizer_calls >= 1
        assert drift_from_forward.active_token_count >= 1
        assert drift_from_forward.mode in (0, 1, 2)
        assert (
            lib.graph_forward_token_ids(
                ctypes.c_void_p(handle),
                token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                int(token_ids.shape[0]),
                logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(logits.shape[0]),
                1,
                ctypes.byref(drift_from_batch),
            )
            == 1
        )
        assert drift_from_batch.stabilizer_calls >= drift_from_forward.stabilizer_calls
        assert drift_from_batch.active_token_count >= drift_from_forward.active_token_count
        assert (
            lib.graph_get_last_drift_snapshot(ctypes.c_void_p(handle), ctypes.byref(drift_last))
            == 1
        )
        assert drift_last.stabilizer_calls == drift_from_batch.stabilizer_calls
        assert lib.graph_reset(ctypes.c_void_p(handle)) == 1
        assert (
            lib.graph_get_last_drift_snapshot(ctypes.c_void_p(handle), ctypes.byref(drift_last))
            == 1
        )
        assert drift_last.stabilizer_calls == 0
        assert drift_last.active_token_count == 0
    finally:
        lib.destroy_model_graph(ctypes.c_void_p(handle))


def test_mul_silu_gate_inplace_matches_scalar_reference() -> None:
    try:
        lib = load_native_library()
    except Exception:
        pytest.skip("Native model_graph symbols are not built into the current shared library.")
    if not hasattr(lib, "simd_mul_silu_gate_inplace"):
        pytest.skip("simd_mul_silu_gate_inplace is unavailable in the current shared library.")

    lib.simd_mul_silu_gate_inplace.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    lib.simd_mul_silu_gate_inplace.restype = None

    rng = np.random.default_rng(23)
    x = rng.standard_normal(37).astype(np.float32)
    gate = rng.standard_normal(37).astype(np.float32)

    ref = x.copy()
    for idx in range(ref.shape[0]):
        ref[idx] *= gate[idx] / (1.0 + math.exp(-float(gate[idx])))

    got = x.copy()
    lib.simd_mul_silu_gate_inplace(
        got.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(got.shape[0]),
    )

    np.testing.assert_allclose(got, ref, rtol=5e-5, atol=5e-5)
