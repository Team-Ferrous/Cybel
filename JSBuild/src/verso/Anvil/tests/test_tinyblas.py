from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from core.native.native_ops import load_native_library
from core.native import quantized_matmul_wrapper as quant_ops


def test_tinyblas_symbols_callable_when_native_library_is_built() -> None:
    try:
        lib = load_native_library()
    except Exception as exc:
        pytest.skip(f"Native library unavailable: {exc}")

    has_tinyblas = any(
        hasattr(lib, name)
        for name in ("tinyblas_matvec_q4k", "tinyblas_matmul_q4k")
    )
    has_compat = all(
        hasattr(lib, name)
        for name in ("simd_matvec_q4k", "simd_matmul_q4k")
    )
    if not (has_tinyblas or has_compat):
        pytest.skip("No tinyBLAS-compatible symbols exported by the native library.")

    if not hasattr(lib, "simd_matvec_f32"):
        pytest.skip("simd_matvec_f32 not exported; cannot run a safe callable smoke check.")

    fn = lib.simd_matvec_f32
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    fn.restype = None

    x = np.asarray([1.0, 2.0], dtype=np.float32)
    a = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = np.zeros((2,), dtype=np.float32)
    fn(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        2,
        2,
    )
    np.testing.assert_allclose(out, a.T @ x, rtol=1e-5, atol=1e-5)


def test_quantized_wrapper_fallback_with_stubbed_tinyblas(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"q4k": 0}

    def _simd_matvec_q4k(x_ptr, _a_ptr, y_ptr, k, n):
        k = int(k)
        n = int(n)
        calls["q4k"] += 1
        x = np.ctypeslib.as_array(x_ptr, shape=(k,))
        y = np.ctypeslib.as_array(y_ptr, shape=(n,))
        y[:] = x[:n] + 1.0

    def _noop_matmul(*_args, **_kwargs):
        return None

    fake_lib = SimpleNamespace(
        simd_matvec_q4k=_simd_matvec_q4k,
        simd_matvec_q6k=_simd_matvec_q4k,
        simd_matvec_q8_0=_simd_matvec_q4k,
        simd_matmul_q4k=_noop_matmul,
        simd_matmul_q6k=_noop_matmul,
        simd_matmul_q8_0=_noop_matmul,
    )

    monkeypatch.setattr(quant_ops, "_LIB", None)
    quant_ops._MATVEC_FN.clear()
    quant_ops._MATMUL_FN.clear()
    monkeypatch.setattr(quant_ops, "load_native_library", lambda: fake_lib)

    matrix = quant_ops.QuantizedMatrix(
        name="stub_q4k",
        qtype=12,  # Q4_K
        shape=(256, 2),
        data=np.zeros((2, 144), dtype=np.uint8),
    )
    x = np.arange(256, dtype=np.float32)
    got = quant_ops.matvec(x, matrix)

    assert calls["q4k"] == 1
    np.testing.assert_allclose(got, np.asarray([1.0, 2.0], dtype=np.float32), rtol=0.0, atol=0.0)

