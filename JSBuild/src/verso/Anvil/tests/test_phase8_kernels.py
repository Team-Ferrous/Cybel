from __future__ import annotations

import numpy as np

from core.native import simd_ops_wrapper as simd_ops


def test_fast_exp_matches_numpy_within_tolerance() -> None:
    x = np.linspace(-10.0, 10.0, 2048, dtype=np.float32)
    got = simd_ops.fast_exp(x)
    ref = np.exp(x).astype(np.float32)
    np.testing.assert_allclose(got, ref, rtol=3e-3, atol=3e-4)


def test_batch_rope_matches_per_head_rope() -> None:
    rng = np.random.default_rng(7)
    n_heads = 4
    n_kv_heads = 2
    head_dim = 16
    pos = 23
    theta = 10000.0

    q = rng.standard_normal((n_heads, head_dim), dtype=np.float32)
    k = rng.standard_normal((n_kv_heads, head_dim), dtype=np.float32)

    q_ref = q.copy()
    k_ref = k.copy()
    simd_ops.rope(q_ref, k_ref, n_heads, n_kv_heads, head_dim, pos, theta)

    q_batch = q.copy()
    k_batch = k.copy()
    simd_ops.batch_rope(q_batch, k_batch, n_heads, n_kv_heads, head_dim, pos, theta)

    np.testing.assert_allclose(q_batch, q_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(k_batch, k_ref, rtol=1e-6, atol=1e-6)


def test_p_core_affinity_calls_are_safe() -> None:
    p_count = simd_ops.get_p_core_count()
    assert isinstance(p_count, int)
    assert p_count >= 0

    pinned_auto = simd_ops.pin_to_p_cores()
    assert isinstance(pinned_auto, bool)

    if p_count > 0:
        pinned_one = simd_ops.pin_to_p_cores(1)
        assert isinstance(pinned_one, bool)
