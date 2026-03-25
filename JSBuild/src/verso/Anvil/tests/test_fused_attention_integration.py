from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import core.native.qsg_forward as qsg_forward_module
from core.native.qsg_forward import QSGForwardPass


class _WeightStoreStub:
    def __init__(self) -> None:
        self.loader = SimpleNamespace(get_metadata=lambda: {})

    @staticmethod
    def attention_dims() -> tuple[int, int, int] | None:
        return None

    @staticmethod
    def get_layer_type(_layer_idx: int) -> str:
        return "attention"


def _profile_stub() -> SimpleNamespace:
    return SimpleNamespace(
        architecture="generic",
        embedding_dim=4,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
    )


def _build_forward() -> QSGForwardPass:
    return QSGForwardPass(_WeightStoreStub(), _profile_stub())


def _weights(dim: int = 4) -> dict[str, np.ndarray]:
    eye = np.eye(dim, dtype=np.float32)
    return {
        "attn_q": eye.copy(),
        "attn_k": eye.copy(),
        "attn_v": eye.copy(),
        "attn_output": eye.copy(),
    }


def _numpy_softmax(scores: np.ndarray) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float32)
    s = s - np.max(s)
    exp = np.exp(s)
    return exp / np.sum(exp)


def _patch_simd_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        qsg_forward_module.simd_ops,
        "matmul",
        lambda a, b: (np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32)).astype(np.float32),
    )
    monkeypatch.setattr(qsg_forward_module.simd_ops, "softmax", _numpy_softmax)
    monkeypatch.setattr(
        qsg_forward_module.simd_ops,
        "rope",
        lambda *_args, **_kwargs: None,
    )


def test_qsg_forward_attention_branch_has_deterministic_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_simd_to_numpy(monkeypatch)
    forward = _build_forward()
    normed = np.asarray([[0.2, -0.1, 0.4, 0.0], [0.1, 0.3, -0.2, 0.5]], dtype=np.float32)

    out_a = forward._forward_attention_branch(
        normed=normed,
        layer_idx=0,
        weights=_weights(),
        start_pos=0,
    )
    forward.reset()
    out_b = forward._forward_attention_branch(
        normed=normed,
        layer_idx=0,
        weights=_weights(),
        start_pos=0,
    )

    assert out_a.shape == normed.shape
    assert out_a.dtype == np.float32
    np.testing.assert_allclose(out_a, out_b, rtol=0.0, atol=0.0)


def test_qsg_forward_uses_fused_attention_when_integration_hook_present(monkeypatch: pytest.MonkeyPatch) -> None:
    source = inspect.getsource(QSGForwardPass._forward_attention_branch)
    if not any(token in source for token in ("FastAttention", "compute_attention", "fused_attention")):
        pytest.skip("Fused-attention integration hook is not present in QSGForwardPass yet.")

    _patch_simd_to_numpy(monkeypatch)
    import core.native.fast_attention_wrapper as fast_attention_wrapper

    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    original = fast_attention_wrapper.FastAttention.compute_attention

    def _spy_compute(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, scale=None):
        calls.append((q.shape, k.shape, v.shape))
        return original(self, q, k, v, scale=scale)

    monkeypatch.setattr(
        fast_attention_wrapper.FastAttention,
        "compute_attention",
        _spy_compute,
    )

    forward = _build_forward()
    normed = np.asarray([[0.25, -0.5, 0.75, -1.0]], dtype=np.float32)
    _ = forward._forward_attention_branch(
        normed=normed,
        layer_idx=0,
        weights=_weights(),
        start_pos=0,
    )

    assert calls, "Expected fused attention wrapper to be invoked when integration is available."

