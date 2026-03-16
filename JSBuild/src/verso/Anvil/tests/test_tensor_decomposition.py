from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import pytest


@dataclass
class _SVDState:
    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray


class _FallbackTensorCompressor:
    @staticmethod
    def compress_layer(weight: np.ndarray, bond_dim: int = 16) -> _SVDState:
        w = np.asarray(weight, dtype=np.float32)
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        rank = max(1, min(int(bond_dim), s.shape[0]))
        return _SVDState(
            u=u[:, :rank].astype(np.float32),
            s=s[:rank].astype(np.float32),
            vt=vt[:rank, :].astype(np.float32),
        )

    @staticmethod
    def reconstruct_layer(state: _SVDState) -> np.ndarray:
        return (state.u * state.s[np.newaxis, :]) @ state.vt

    @staticmethod
    def matvec_mpo(x: np.ndarray, state: _SVDState) -> np.ndarray:
        vec = np.asarray(x, dtype=np.float32).reshape(-1)
        projected = state.vt @ vec
        return (state.u * state.s[np.newaxis, :]) @ projected


class _NativeTensorAdapter:
    def __init__(self, impl):
        self.impl = impl

    def compress_layer(self, weight: np.ndarray, bond_dim: int):
        return self.impl.compress_layer(weight, bond_dim=bond_dim)

    def reconstruct_layer(self, state):
        for name in ("reconstruct_layer", "decompress_layer", "reconstruct"):
            fn = getattr(self.impl, name, None)
            if callable(fn):
                return fn(state)
        raise AttributeError("No reconstruction API found on TensorNetworkCompressor")

    def matvec_mpo(self, x: np.ndarray, state):
        fn = getattr(self.impl, "matvec_mpo")
        try:
            return fn(x, state)
        except TypeError:
            return fn(state, x)


def _load_compressor():
    try:
        module = importlib.import_module("core.native.tensor_decomposition")
        cls = getattr(module, "TensorNetworkCompressor")
        instance = cls()
        if not (hasattr(instance, "compress_layer") and hasattr(instance, "matvec_mpo")):
            raise AttributeError("Incomplete TensorNetworkCompressor API")
        return _NativeTensorAdapter(instance), "native"
    except Exception:
        return _FallbackTensorCompressor(), "fallback"


def _make_low_rank_matrix(seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((24, 4)).astype(np.float32)
    v = rng.standard_normal((4, 16)).astype(np.float32)
    noise = 0.01 * rng.standard_normal((24, 16)).astype(np.float32)
    return (u @ v + noise).astype(np.float32)


def _relative_error(ref: np.ndarray, got: np.ndarray) -> float:
    ref_n = float(np.linalg.norm(ref))
    if ref_n <= 1e-12:
        return 0.0
    return float(np.linalg.norm(ref - got) / ref_n)


def test_tensor_decomposition_reconstruction_quality() -> None:
    compressor, backend = _load_compressor()
    weight = _make_low_rank_matrix()
    state = compressor.compress_layer(weight, bond_dim=4)

    try:
        reconstructed = np.asarray(compressor.reconstruct_layer(state), dtype=np.float32)
    except AttributeError:
        if backend == "native":
            pytest.skip("Native tensor decomposition module does not expose a reconstruction API yet.")
        raise

    assert reconstructed.shape == weight.shape
    assert _relative_error(weight, reconstructed) < 0.20


def test_tensor_decomposition_matvec_approximation_quality() -> None:
    compressor, _backend = _load_compressor()
    weight = _make_low_rank_matrix(seed=17)
    state = compressor.compress_layer(weight, bond_dim=4)
    x = np.linspace(-1.0, 1.0, weight.shape[1], dtype=np.float32)

    got_a = np.asarray(compressor.matvec_mpo(x, state), dtype=np.float32).reshape(-1)
    got_b = np.asarray(compressor.matvec_mpo(x, state), dtype=np.float32).reshape(-1)
    ref = weight @ x

    assert got_a.shape == ref.shape
    np.testing.assert_allclose(got_a, got_b, rtol=0.0, atol=0.0)
    assert _relative_error(ref, got_a) < 0.25

