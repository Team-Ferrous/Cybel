from __future__ import annotations

import numpy as np

from core.native import simd_ops_wrapper as simd_ops


def test_matmul_threaded_matches_numpy() -> None:
    rng = np.random.default_rng(123)
    m, k, n = 6, 512, 384
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)

    got = simd_ops.matmul(a, b)
    expected = a @ b

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-3)


def test_matvec_threaded_matches_numpy() -> None:
    rng = np.random.default_rng(456)
    k, n = 1024, 768
    x = rng.standard_normal((k,), dtype=np.float32)
    a = rng.standard_normal((k, n), dtype=np.float32)

    got = simd_ops.matvec(x, a)
    expected = a.T @ x

    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=2e-3)


def test_openmp_thread_count_control_roundtrip() -> None:
    original = simd_ops.get_num_threads()
    assert original > 0

    simd_ops.set_num_threads(2)
    assert simd_ops.get_num_threads() >= 1
    assert simd_ops.get_num_procs() >= 1

    simd_ops.set_num_threads(original)
