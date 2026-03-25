import numpy as np
import pytest
from core.simd.simd_ops import SIMDOps


@pytest.fixture
def simd():
    return SIMDOps()


def assert_close(actual, expected, rtol=1e-5, atol=1e-5, name=""):
    diff = np.abs(actual - expected)
    max_diff = np.max(diff)
    print(f"[{name}] Max Diff: {max_diff:.6f}")
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_exp_inplace(simd):
    data = np.random.randn(100).astype(np.float32)
    expected = np.exp(data)
    simd.exp_inplace(data)
    assert_close(data, expected, rtol=1e-4, atol=1e-4, name="Exp")


def test_log_inplace(simd):
    data = np.abs(np.random.randn(100)).astype(np.float32) + 0.1
    expected = np.log(data)
    simd.log_inplace(data)
    # Log approximation using Taylor 5th order is not very precise for t near 1
    assert_close(data, expected, rtol=0.1, atol=0.1, name="Log")


def test_sigmoid_inplace(simd):
    data = np.random.randn(100).astype(np.float32)
    expected = 1 / (1 + np.exp(-data))
    simd.sigmoid_inplace(data)
    assert_close(data, expected, rtol=1e-4, atol=1e-4, name="Sigmoid")


def test_softmax(simd):
    data = np.random.randn(100).astype(np.float32)
    e_x = np.exp(data - np.max(data))
    expected = e_x / e_x.sum()

    output = simd.softmax(data)
    assert_close(output, expected, rtol=1e-4, atol=1e-4, name="Softmax")


def test_silu_inplace(simd):
    data = np.random.randn(100).astype(np.float32)
    expected = data * (1 / (1 + np.exp(-data)))
    simd.silu_inplace(data)
    # SiLU uses sigmoid approximation
    assert_close(data, expected, rtol=1e-3, atol=1e-3, name="SiLU")


def test_dot_product(simd):
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    expected = np.dot(a, b)
    result = simd.dot_product(a, b)
    assert_close(result, expected, rtol=1e-4, atol=1e-4, name="Dot")


def test_cosine_similarity(simd):
    a = np.random.randn(100).astype(np.float32)
    b = np.random.randn(100).astype(np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    expected = np.dot(a, b) / (norm_a * norm_b)
    result = simd.cosine_similarity(a, b)
    assert_close(result, expected, rtol=1e-4, atol=1e-4, name="Cosine")
