from __future__ import annotations

import numpy as np
import pytest
from gguf import dequantize

from core.model.gguf_loader import GGUFModelLoader
from core.model.model_profile import ModelProfile
from core.native import quantized_matmul_wrapper as quant_ops
from core.native.quantized_matmul_wrapper import (
    QuantizedMatrix,
    dequantize_rows,
    fused_expert_swiglu,
    fused_moe_ffn,
    matmul,
    matvec,
)
from core.native.weight_store import WeightStore


def _load_model_or_skip(model_name: str) -> GGUFModelLoader:
    try:
        return GGUFModelLoader(model_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Model '{model_name}' unavailable: {exc}")


def _quant_matrix(loader: GGUFModelLoader, tensor_name: str) -> QuantizedMatrix:
    tensor = next((t for t in loader.reader.tensors if t.name == tensor_name), None)
    if tensor is None:
        pytest.skip(f"Tensor '{tensor_name}' not found in model.")
    if len(tensor.shape) != 2:
        pytest.skip(f"Tensor '{tensor_name}' is not 2D.")
    raw = np.asarray(tensor.data, dtype=np.uint8)
    return QuantizedMatrix(
        name=tensor_name,
        qtype=int(tensor.tensor_type.value),
        shape=(int(tensor.shape[0]), int(tensor.shape[1])),
        data=raw,
    )


def _fake_q8_matrix(
    name: str, input_dim: int = 32, output_dim: int = 32
) -> QuantizedMatrix:
    row_bytes = (input_dim // 32) * 34
    matrix = QuantizedMatrix(
        name=name,
        qtype=8,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    matrix.ensure_packed()
    matrix._validated = True
    return matrix


def _pack_q6k_block(
    q_values: np.ndarray, scales: np.ndarray, d: np.float16
) -> np.ndarray:
    if q_values.shape != (16, 16):
        raise AssertionError(f"expected q_values shape (16, 16), got {q_values.shape}")
    if scales.shape != (16,):
        raise AssertionError(f"expected scales shape (16,), got {scales.shape}")

    blk = np.zeros((210,), dtype=np.uint8)
    ql = blk[:128]
    qh = blk[128:192]

    qv = np.asarray(q_values, dtype=np.int16)
    if np.any(qv < -32) or np.any(qv > 31):
        raise AssertionError("Q6_K values must be in [-32, 31].")

    for g in range(8):
        ql_seg = g // 4
        ql_rem = g % 4
        ql_shift = 4 if ql_rem >= 2 else 0
        ql_start = (ql_rem % 2) * 32
        qh_seg = g // 4
        qh_shift = (g % 4) * 2

        for half in range(2):
            scale_idx = g * 2 + half
            vals = qv[scale_idx] + 32
            for i in range(16):
                idx32 = half * 16 + i
                u6 = int(vals[i])
                lo = u6 & 0x0F
                hi = (u6 >> 4) & 0x03

                ql_idx = ql_seg * 64 + ql_start + idx32
                ql_prev = int(ql[ql_idx])
                if ql_shift == 0:
                    ql[ql_idx] = np.uint8((ql_prev & 0xF0) | lo)
                else:
                    ql[ql_idx] = np.uint8((ql_prev & 0x0F) | (lo << 4))

                qh_idx = qh_seg * 32 + idx32
                qh_prev = int(qh[qh_idx])
                qh_mask = ~(0x03 << qh_shift) & 0xFF
                qh[qh_idx] = np.uint8((qh_prev & qh_mask) | (hi << qh_shift))

    blk[192:208] = np.asarray(scales, dtype=np.int8).view(np.uint8)
    d16 = int(np.asarray([d], dtype=np.float16).view(np.uint16)[0])
    blk[208] = np.uint8(d16 & 0xFF)
    blk[209] = np.uint8((d16 >> 8) & 0xFF)
    return blk


def _synthetic_q6k_matrix(seed: int = 101) -> QuantizedMatrix:
    input_dim = 512
    output_dim = 7
    row_bytes = (input_dim // 256) * 210
    data = np.zeros((output_dim, row_bytes), dtype=np.uint8)
    rng = np.random.default_rng(seed)

    for row in range(output_dim):
        for blk_idx in range(input_dim // 256):
            q_values = rng.integers(-32, 32, size=(16, 16), dtype=np.int16).astype(
                np.int8
            )
            scales = rng.integers(-32, 32, size=(16,), dtype=np.int16).astype(np.int8)
            d = np.float16(rng.uniform(0.02, 0.5))
            block = _pack_q6k_block(q_values=q_values, scales=scales, d=d)
            off = blk_idx * 210
            data[row, off : off + 210] = block

    matrix = QuantizedMatrix(
        name="synthetic_q6k",
        qtype=quant_ops.QTYPE_Q6_K,
        shape=(input_dim, output_dim),
        data=data,
    )
    matrix.ensure_packed()
    matrix._validated = True
    return matrix


def _granite_moe_bundle_or_skip() -> tuple[
    list[QuantizedMatrix],
    list[QuantizedMatrix],
    list[QuantizedMatrix],
]:
    loader = _load_model_or_skip("granite4:tiny-h")
    profile = ModelProfile.from_loader("granite4:tiny-h", loader)
    store = WeightStore(loader, profile)
    for layer_idx in range(int(profile.n_layers)):
        gate_name, up_name, down_name = store.get_expert_tensor_names(layer_idx)
        if not gate_name or not up_name or not down_name:
            continue
        tensor = store._tensor_index.get(gate_name)
        if tensor is None or len(getattr(tensor, "shape", ())) != 3:
            continue
        expert_count = int(tensor.shape[2])
        if expert_count < 2:
            continue

        gates: list[QuantizedMatrix] = []
        ups: list[QuantizedMatrix] = []
        downs: list[QuantizedMatrix] = []
        for expert_idx in range(2):
            gate = store.get_expert_matrix(gate_name, expert_idx)
            up = store.get_expert_matrix(up_name, expert_idx)
            down = store.get_expert_matrix(down_name, expert_idx)
            if not isinstance(gate, QuantizedMatrix):
                break
            if not isinstance(up, QuantizedMatrix):
                break
            if not isinstance(down, QuantizedMatrix):
                break
            gates.append(gate)
            ups.append(up)
            downs.append(down)
        if len(gates) == 2 and len(ups) == 2 and len(downs) == 2:
            return gates, ups, downs
    pytest.skip("Granite routed expert tensors were not available for fused MoE validation.")


def _reference_projection_from_raw(
    loader: GGUFModelLoader, matrix: QuantizedMatrix, x: np.ndarray
) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32).reshape(-1)
    tensor = next(t for t in loader.reader.tensors if t.name == matrix.name)
    raw = np.asarray(tensor.data, dtype=np.uint8)
    dq = np.asarray(dequantize(raw, tensor.tensor_type), dtype=np.float32)
    if dq.ndim != 2:
        raise AssertionError(f"Expected 2D dequantized matrix, got shape {dq.shape}")
    if dq.shape[1] != x_f.shape[0]:
        raise AssertionError(f"Weight/input mismatch: {dq.shape} vs {x_f.shape}")
    return dq @ x_f


def _reference_projection_batch_from_raw(
    loader: GGUFModelLoader,
    matrix: QuantizedMatrix,
    x: np.ndarray,
) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    tensor = next(t for t in loader.reader.tensors if t.name == matrix.name)
    raw = np.asarray(tensor.data, dtype=np.uint8)
    dq = np.asarray(dequantize(raw, tensor.tensor_type), dtype=np.float32)
    if dq.ndim != 2:
        raise AssertionError(f"Expected 2D dequantized matrix, got shape {dq.shape}")
    if dq.shape[1] != x_f.shape[1]:
        raise AssertionError(f"Weight/input mismatch: {dq.shape} vs {x_f.shape}")
    return x_f @ dq.T


def test_q4k_matvec_matches_dequantized_reference() -> None:
    loader = _load_model_or_skip("qwen3.5:9b")
    matrix = _quant_matrix(loader, "blk.0.attn_qkv.weight")
    if matrix.qtype != 12:
        pytest.skip("Expected Q4_K tensor for qwen3.5 attn_qkv.")

    x = np.random.default_rng(7).standard_normal(matrix.input_dim).astype(np.float32)
    got = matvec(x, matrix)
    ref = _reference_projection_from_raw(loader, matrix, x)

    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-4)


def test_q4k_matmul_matches_dequantized_reference(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_QUANT_MATMUL_MIN_BATCH", "1")
    loader = _load_model_or_skip("qwen3.5:9b")
    matrix = _quant_matrix(loader, "blk.0.attn_qkv.weight")
    if matrix.qtype != 12:
        pytest.skip("Expected Q4_K tensor for qwen3.5 attn_qkv.")

    x = (
        np.random.default_rng(17)
        .standard_normal((4, matrix.input_dim))
        .astype(np.float32)
    )
    got = matmul(x, matrix)
    ref = _reference_projection_batch_from_raw(loader, matrix, x)

    np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-4)


def test_q4k_r4_repack_matches_q4k_matvec() -> None:
    loader = _load_model_or_skip("qwen3.5:9b")
    matrix = _quant_matrix(loader, "blk.0.attn_qkv.weight")
    if matrix.qtype != 12:
        pytest.skip("Expected Q4_K tensor for qwen3.5 attn_qkv.")

    repacked = quant_ops.repack_q4k_r4(matrix)
    if repacked is matrix or repacked.qtype != quant_ops.QTYPE_Q4_K_R4:
        pytest.skip("Q4_K_R4 repack kernel is not available in native library.")

    x = np.random.default_rng(23).standard_normal(matrix.input_dim).astype(np.float32)
    base = matvec(x, matrix)
    got = matvec(x, repacked)
    np.testing.assert_allclose(got, base, rtol=2e-4, atol=2e-4)


def test_q6k_matvec_matches_dequantized_reference() -> None:
    loader = _load_model_or_skip("granite4:tiny-h")
    matrix = _quant_matrix(loader, "blk.25.attn_v.weight")
    if matrix.qtype != 14:
        pytest.skip("Expected Q6_K tensor for granite4 attn_v.")

    x = np.random.default_rng(11).standard_normal(matrix.input_dim).astype(np.float32)
    got = matvec(x, matrix)
    ref = _reference_projection_from_raw(loader, matrix, x)

    np.testing.assert_allclose(got, ref, rtol=3e-4, atol=3e-4)


def test_q6k_matmul_matches_dequantized_reference(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_QUANT_MATMUL_MIN_BATCH", "1")
    loader = _load_model_or_skip("granite4:tiny-h")
    matrix = _quant_matrix(loader, "blk.25.attn_v.weight")
    if matrix.qtype != 14:
        pytest.skip("Expected Q6_K tensor for granite4 attn_v.")

    x = (
        np.random.default_rng(19)
        .standard_normal((4, matrix.input_dim))
        .astype(np.float32)
    )
    got = matmul(x, matrix)
    ref = _reference_projection_batch_from_raw(loader, matrix, x)

    np.testing.assert_allclose(got, ref, rtol=4e-4, atol=4e-4)


def test_q6k_r4_repack_matches_q6k_matvec() -> None:
    loader = _load_model_or_skip("granite4:tiny-h")
    matrix = _quant_matrix(loader, "blk.25.attn_v.weight")
    if matrix.qtype != 14:
        pytest.skip("Expected Q6_K tensor for granite4 attn_v.")

    repacked = quant_ops.repack_q6k_r4(matrix)
    if repacked is matrix or repacked.qtype != quant_ops.QTYPE_Q6_K_R4:
        pytest.skip("Q6_K_R4 repack kernel is not available in native library.")

    x = np.random.default_rng(29).standard_normal(matrix.input_dim).astype(np.float32)
    base = matvec(x, matrix)
    got = matvec(x, repacked)
    np.testing.assert_allclose(got, base, rtol=3e-4, atol=3e-4)


def test_q6k_synthetic_matvec_matches_dequantized_reference() -> None:
    matrix = _synthetic_q6k_matrix(seed=127)
    x = np.random.default_rng(131).standard_normal(matrix.input_dim).astype(np.float32)
    got = matvec(x, matrix)
    dense = dequantize_rows(matrix, range(matrix.output_dim))
    ref = dense @ x
    np.testing.assert_allclose(got, ref, rtol=4e-4, atol=4e-4)


def test_q6k_lm_repack_matches_q6k_matvec_synthetic() -> None:
    matrix = _synthetic_q6k_matrix(seed=137)
    repacked = quant_ops.repack_q6k_lm(matrix)
    if repacked is matrix or repacked.qtype != quant_ops.QTYPE_Q6_K_LM:
        pytest.skip("Q6_K_LM repack kernel is not available in native library.")

    x = np.random.default_rng(139).standard_normal(matrix.input_dim).astype(np.float32)
    base = matvec(x, matrix)
    got = matvec(x, repacked)
    np.testing.assert_allclose(got, base, rtol=3e-4, atol=3e-4)


def test_dequantize_rows_matches_gguf_reference() -> None:
    loader = _load_model_or_skip("granite4:tiny-h")
    matrix = _quant_matrix(loader, "token_embd.weight")
    row_ids = [0, 7, 123]

    tensor = next(t for t in loader.reader.tensors if t.name == "token_embd.weight")
    ref = np.asarray(
        dequantize(
            np.asarray(tensor.data[row_ids], dtype=np.uint8), tensor.tensor_type
        ),
        dtype=np.float32,
    )
    got = dequantize_rows(matrix, row_ids)

    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6)


def test_fused_expert_swiglu_falls_back_when_hidden_is_interleaved(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_STRICT_NATIVE_QSG", "0")
    gate = _fake_q8_matrix("gate")
    up = _fake_q8_matrix("up")
    down = _fake_q8_matrix("down")
    gate._inverse_row_permutation = np.arange(gate.output_dim, dtype=np.int64)
    up._inverse_row_permutation = np.arange(up.output_dim, dtype=np.int64)

    class _FakeLib:
        @staticmethod
        def simd_fused_expert_swiglu(*_args):
            raise AssertionError(
                "fused kernel must not run for interleaved hidden matrices"
            )

    monkeypatch.setattr(quant_ops, "_lib", lambda: _FakeLib())

    def _stub_matvec(x: np.ndarray, matrix: QuantizedMatrix) -> np.ndarray:
        if matrix.name == "gate":
            return np.full((matrix.output_dim,), 2.0, dtype=np.float32)
        if matrix.name == "up":
            return np.full((matrix.output_dim,), 3.0, dtype=np.float32)
        if matrix.name == "down":
            return np.full((matrix.output_dim,), float(np.mean(x)), dtype=np.float32)
        raise AssertionError(f"unexpected matrix {matrix.name}")

    monkeypatch.setattr(quant_ops, "matvec", _stub_matvec)
    from core.native import simd_ops_wrapper

    monkeypatch.setattr(simd_ops_wrapper, "swiglu", lambda g, u: g + u)

    out = fused_expert_swiglu(np.ones((32,), dtype=np.float32), gate, up, down)

    np.testing.assert_allclose(
        out, np.full((32,), 5.0, dtype=np.float32), rtol=0.0, atol=0.0
    )


def test_fused_moe_ffn_falls_back_when_any_matrix_is_interleaved(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_STRICT_NATIVE_QSG", "0")
    g0, u0, d0 = _fake_q8_matrix("g0"), _fake_q8_matrix("u0"), _fake_q8_matrix("d0")
    g1, u1, d1 = _fake_q8_matrix("g1"), _fake_q8_matrix("u1"), _fake_q8_matrix("d1")
    g0._inverse_row_permutation = np.arange(g0.output_dim, dtype=np.int64)

    class _FakeLib:
        @staticmethod
        def simd_fused_moe_ffn(*_args):
            raise AssertionError(
                "fused moe kernel must not run for interleaved matrices"
            )

    monkeypatch.setattr(quant_ops, "_lib", lambda: _FakeLib())

    def _stub_fused_expert(x, gate_matrix, up_matrix, down_matrix):  # noqa: ARG001
        val = 1.0 if gate_matrix.name == "g0" else 4.0
        return np.full((down_matrix.output_dim,), val, dtype=np.float32)

    monkeypatch.setattr(quant_ops, "fused_expert_swiglu", _stub_fused_expert)
    out = fused_moe_ffn(
        np.ones((32,), dtype=np.float32),
        expert_weights=[0.25, 0.75],
        gate_matrices=[g0, g1],
        up_matrices=[u0, u1],
        down_matrices=[d0, d1],
    )
    np.testing.assert_allclose(
        out, np.full((32,), 3.25, dtype=np.float32), rtol=0.0, atol=0.0
    )


def test_fused_expert_swiglu_strict_mode_rejects_python_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_STRICT_NATIVE_QSG", "1")
    gate = _fake_q8_matrix("gate")
    up = _fake_q8_matrix("up")
    down = _fake_q8_matrix("down")
    gate._inverse_row_permutation = np.arange(gate.output_dim, dtype=np.int64)

    class _FakeLib:
        @staticmethod
        def simd_fused_expert_swiglu(*_args):
            raise AssertionError("should not be called")

    monkeypatch.setattr(quant_ops, "_lib", lambda: _FakeLib())
    with pytest.raises(RuntimeError, match="Strict native QSG"):
        fused_expert_swiglu(np.ones((32,), dtype=np.float32), gate, up, down)


def test_fused_moe_ffn_strict_mode_rejects_python_fallback(monkeypatch) -> None:
    monkeypatch.setenv("ANVIL_STRICT_NATIVE_QSG", "1")
    g0, u0, d0 = _fake_q8_matrix("g0"), _fake_q8_matrix("u0"), _fake_q8_matrix("d0")
    g1, u1, d1 = _fake_q8_matrix("g1"), _fake_q8_matrix("u1"), _fake_q8_matrix("d1")
    g0._inverse_row_permutation = np.arange(g0.output_dim, dtype=np.int64)

    class _FakeLib:
        @staticmethod
        def simd_fused_moe_ffn(*_args):
            raise AssertionError("should not be called")

    monkeypatch.setattr(quant_ops, "_lib", lambda: _FakeLib())
    with pytest.raises(RuntimeError, match="Strict native QSG"):
        fused_moe_ffn(
            np.ones((32,), dtype=np.float32),
            expert_weights=[0.25, 0.75],
            gate_matrices=[g0, g1],
            up_matrices=[u0, u1],
            down_matrices=[d0, d1],
        )


def test_fused_moe_ffn_matches_weighted_expert_sum_on_granite() -> None:
    gate_matrices, up_matrices, down_matrices = _granite_moe_bundle_or_skip()
    x = np.random.default_rng(41).standard_normal(gate_matrices[0].input_dim).astype(np.float32)
    expert_weights = [0.625, 0.375]

    expected = np.zeros((down_matrices[0].output_dim,), dtype=np.float32)
    for weight, gate, up, down in zip(
        expert_weights, gate_matrices, up_matrices, down_matrices
    ):
        expected += float(weight) * fused_expert_swiglu(x, gate, up, down)

    got = fused_moe_ffn(
        x,
        expert_weights=expert_weights,
        gate_matrices=gate_matrices,
        up_matrices=up_matrices,
        down_matrices=down_matrices,
    )

    np.testing.assert_allclose(got, expected, rtol=5e-4, atol=5e-4)
