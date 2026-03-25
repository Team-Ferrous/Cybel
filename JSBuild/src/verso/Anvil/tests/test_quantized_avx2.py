from __future__ import annotations

import numpy as np
import pytest
from gguf import dequantize

from core.model.gguf_loader import GGUFModelLoader
from core.native.quantized_matmul_wrapper import QuantizedMatrix, matmul, matvec


_QTYPE_TO_MODEL = {
    12: "qwen3.5:9b",     # Q4_K
    14: "granite4:tiny-h",  # Q6_K
    8: "granite4:tiny-h",    # Q8_0 (may not be present)
}


def _load_or_skip(model_name: str) -> GGUFModelLoader:
    try:
        return GGUFModelLoader(model_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Model '{model_name}' unavailable: {exc}")


def _find_quant_tensor(loader: GGUFModelLoader, qtype: int):
    for tensor in loader.reader.tensors:
        if len(tensor.shape) != 2:
            continue
        if int(tensor.tensor_type.value) != qtype:
            continue
        return tensor
    return None


@pytest.mark.parametrize("qtype", [12, 14, 8])
def test_avx2_quantized_matvec_matches_dequantized_reference(qtype: int, monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_QUANT_MATMUL_MIN_BATCH", "1")
    loader = _load_or_skip(_QTYPE_TO_MODEL[qtype])
    tensor = _find_quant_tensor(loader, qtype)
    if tensor is None:
        pytest.skip(f"No 2D tensor with qtype={qtype} found")

    raw = np.asarray(tensor.data, dtype=np.uint8)
    matrix = QuantizedMatrix(
        name=tensor.name,
        qtype=qtype,
        shape=(int(tensor.shape[0]), int(tensor.shape[1])),
        data=raw,
    )

    rng = np.random.default_rng(99 + qtype)
    x = rng.standard_normal(matrix.input_dim).astype(np.float32)

    got = matvec(x, matrix)
    dq = np.asarray(dequantize(raw, tensor.tensor_type), dtype=np.float32)
    ref = dq @ x

    np.testing.assert_allclose(got, ref, rtol=5e-4, atol=5e-4)

    xb = rng.standard_normal((3, matrix.input_dim)).astype(np.float32)
    got_b = matmul(xb, matrix)
    ref_b = xb @ dq.T
    np.testing.assert_allclose(got_b, ref_b, rtol=6e-4, atol=6e-4)
