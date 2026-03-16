from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.native.weight_store import WeightStore
from core.native.quantized_matmul_wrapper import QuantizedMatrix


class _FakeLoader:
    def __init__(self, tensors, tensor_values):
        self.reader = SimpleNamespace(tensors=tensors)
        self._tensor_values = tensor_values
        self.calls: list[str] = []

    def get_tensor(self, name: str):
        self.calls.append(name)
        return self._tensor_values.get(name)


def _profile() -> SimpleNamespace:
    return SimpleNamespace(n_layers=1, embedding_dim=4, architecture="qwen35")


def _tensor(name: str, shape: tuple[int, ...], qtype: int, data: np.ndarray):
    return SimpleNamespace(
        name=name,
        shape=shape,
        tensor_type=SimpleNamespace(value=qtype),
        data=data,
    )


def test_weight_store_skips_quantized_3d_tensor_by_default(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_LOAD_MOE_EXPERTS", raising=False)
    tensor_name = "blk.0.ffn_down_exps.weight"
    tensors = [
        _tensor(
            name=tensor_name,
            shape=(8, 8, 2),
            qtype=14,  # Q6_K
            data=np.zeros((2, 8, 84), dtype=np.uint8),
        )
    ]
    loader = _FakeLoader(
        tensors=tensors,
        tensor_values={tensor_name: np.ones((8, 8, 2), dtype=np.float32)},
    )
    store = WeightStore(loader, _profile())
    loader.calls.clear()

    got = store.get_weight(tensor_name)

    assert got is None
    assert loader.calls == []


def test_weight_store_can_load_quantized_3d_tensor_when_enabled(monkeypatch):
    monkeypatch.setenv("ANVIL_NATIVE_LOAD_MOE_EXPERTS", "1")
    tensor_name = "blk.0.ffn_down_exps.weight"
    tensors = [
        _tensor(
            name=tensor_name,
            shape=(8, 8, 2),
            qtype=14,  # Q6_K
            data=np.zeros((2, 8, 84), dtype=np.uint8),
        )
    ]
    expected = np.ones((8, 8, 2), dtype=np.float32)
    loader = _FakeLoader(tensors=tensors, tensor_values={tensor_name: expected})
    store = WeightStore(loader, _profile())
    loader.calls.clear()

    got = store.get_weight(tensor_name)

    assert isinstance(got, np.ndarray)
    assert got.shape == expected.shape
    assert tensor_name in store._cache


def test_weight_store_preserves_float16_tensor_dtype(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_LOAD_MOE_EXPERTS", raising=False)
    tensor_name = "blk.0.attn_norm.weight"
    tensors = [
        _tensor(
            name=tensor_name,
            shape=(4,),
            qtype=1,  # F16
            data=np.zeros((4,), dtype=np.uint16),
        )
    ]
    expected = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float16)
    loader = _FakeLoader(tensors=tensors, tensor_values={tensor_name: expected})
    store = WeightStore(loader, _profile())
    loader.calls.clear()

    got = store.get_tensor(tensor_name)

    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float16
    assert np.array_equal(got, expected)


def test_weight_store_returns_quantized_expert_slice_without_dequant(monkeypatch):
    monkeypatch.setenv("ANVIL_NATIVE_LOAD_MOE_EXPERTS", "1")
    tensor_name = "blk.0.ffn_down_exps.weight"
    raw = np.zeros((2, 8, 84), dtype=np.uint8)
    tensors = [
        _tensor(
            name=tensor_name,
            shape=(8, 8, 2),
            qtype=14,  # Q6_K
            data=raw,
        )
    ]
    loader = _FakeLoader(tensors=tensors, tensor_values={})
    store = WeightStore(loader, _profile())

    got = store.get_expert_matrix(tensor_name, 1)

    assert isinstance(got, QuantizedMatrix)
    assert got.qtype == 14
    assert got.shape == (8, 8)
    assert got.data.shape == (8, 84)


def test_weight_store_keeps_expert_quant_matrices_non_interleaved(monkeypatch):
    monkeypatch.setenv("ANVIL_QUANT_ROW_INTERLEAVE", "4")
    tensor_name = "blk.0.ffn_gate_exps.weight"
    raw = np.zeros((2, 8, 84), dtype=np.uint8)
    tensors = [
        _tensor(
            name=tensor_name,
            shape=(8, 8, 2),
            qtype=14,  # Q6_K
            data=raw,
        )
    ]
    loader = _FakeLoader(tensors=tensors, tensor_values={})
    store = WeightStore(loader, _profile())

    got = store.get_expert_matrix(tensor_name, 1)

    assert isinstance(got, QuantizedMatrix)
    assert got.interleave_factor == 1
    assert got._inverse_row_permutation is None


def test_weight_store_default_interleave_prefers_granite_graph_mode(monkeypatch):
    monkeypatch.delenv("ANVIL_QUANT_ROW_INTERLEAVE", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_GRAPH_MODE", raising=False)
    loader = _FakeLoader(tensors=[], tensor_values={})
    profile = SimpleNamespace(n_layers=1, embedding_dim=4, architecture="granitehybrid")

    store = WeightStore(loader, profile)

    assert store._quant_interleave == 4


def test_weight_store_default_interleave_disables_when_graph_opted_out(monkeypatch):
    monkeypatch.delenv("ANVIL_QUANT_ROW_INTERLEAVE", raising=False)
    monkeypatch.setenv("ANVIL_NATIVE_GRAPH_MODE", "0")
    loader = _FakeLoader(tensors=[], tensor_values={})
    profile = SimpleNamespace(n_layers=1, embedding_dim=4, architecture="granitehybrid")

    store = WeightStore(loader, profile)

    assert store._quant_interleave == 1


def test_weight_store_defaults_granite_to_full_moe(monkeypatch):
    monkeypatch.delenv("ANVIL_DISABLE_MOE_ROUTED", raising=False)
    monkeypatch.delenv("ANVIL_DISABLE_MOE_SHARED", raising=False)
    loader = _FakeLoader(tensors=[], tensor_values={})
    profile = SimpleNamespace(n_layers=1, embedding_dim=4, architecture="granitehybrid")

    store = WeightStore(loader, profile)

    assert store._disable_moe_routed is False
    assert store._disable_moe_shared is False


def test_weight_store_honors_explicit_moe_route_override(monkeypatch):
    monkeypatch.setenv("ANVIL_DISABLE_MOE_ROUTED", "0")
    loader = _FakeLoader(tensors=[], tensor_values={})
    profile = SimpleNamespace(n_layers=1, embedding_dim=4, architecture="granitehybrid")

    store = WeightStore(loader, profile)

    assert store._disable_moe_routed is False
