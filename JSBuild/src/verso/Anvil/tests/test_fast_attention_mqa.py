from __future__ import annotations

from array import array
import ctypes
import math

import pytest

from core.native.native_ops import load_native_library


def _configure_mqa_symbol() -> ctypes._CFuncPtr | None:
    lib = load_native_library()
    fn = getattr(lib, "fused_attention_mqa_f32", None)
    if fn is None:
        return None
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
    ]
    fn.restype = None
    return fn


def _ptr(buffer: array) -> ctypes.POINTER(ctypes.c_float):
    return ctypes.cast(buffer.buffer_info()[0], ctypes.POINTER(ctypes.c_float))


def test_fused_attention_mqa_matches_python_reference_for_tiled_head_dim():
    fn = _configure_mqa_symbol()
    if fn is None:
        pytest.skip("fused_attention_mqa_f32 is not available")

    batch = 1
    q_heads = 2
    kv_heads = 1
    seq_q = 1
    seq_k = 2
    head_dim = 64
    scale = 1.0

    q = array("f", [0.0] * (batch * q_heads * seq_q * head_dim))
    k = array("f", [0.0] * (batch * kv_heads * seq_k * head_dim))
    v = array("f", [0.0] * (batch * kv_heads * seq_k * head_dim))
    out = array("f", [0.0] * (batch * q_heads * seq_q * head_dim))

    q[0] = 1.0
    q[head_dim + 1] = 1.0
    k[0] = 1.0
    k[head_dim + 1] = 1.0
    for dim in range(head_dim):
        v[dim] = 1.0
        v[head_dim + dim] = 3.0

    fn(
        _ptr(q),
        _ptr(k),
        _ptr(v),
        _ptr(out),
        batch,
        q_heads,
        kv_heads,
        seq_q,
        seq_k,
        head_dim,
        ctypes.c_float(scale),
    )

    weight_hi = math.exp(1.0)
    expected_head0 = (weight_hi * 1.0 + 1.0 * 3.0) / (weight_hi + 1.0)
    expected_head1 = (1.0 * 1.0 + weight_hi * 3.0) / (weight_hi + 1.0)

    head0 = out[:head_dim]
    head1 = out[head_dim : 2 * head_dim]
    assert list(head0) == pytest.approx([expected_head0] * head_dim, rel=1e-5, abs=1e-5)
    assert list(head1) == pytest.approx([expected_head1] * head_dim, rel=1e-5, abs=1e-5)
