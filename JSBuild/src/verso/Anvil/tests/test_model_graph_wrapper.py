from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np
import pytest

from core.native import quantized_matmul_wrapper as quant_ops
from core.native.model_graph_wrapper import (
    NativeModelGraph,
    _GraphDriftSnapshot,
    _GraphPerfStatsSnapshot,
    _split_fused_qkv_weight,
)


def _make_q8_matrix(name: str, input_dim: int, output_dim: int) -> quant_ops.QuantizedMatrix:
    row_bytes = 34  # q8_0 row bytes for input_dim=32
    matrix = quant_ops.QuantizedMatrix(
        name=name,
        qtype=quant_ops.QTYPE_Q8_0,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    matrix.ensure_packed()
    matrix._validated = True
    return matrix


def test_split_fused_qkv_weight_dense_transposed_layout():
    # Simulate [in_dim, out_dim] fused projection layout.
    in_dim = 6
    q_out = 4
    kv_out = 2
    total = q_out + 2 * kv_out
    fused = np.arange(in_dim * total, dtype=np.float32).reshape(in_dim, total)

    q, k, v = _split_fused_qkv_weight(fused, q_out=q_out, kv_out=kv_out)

    assert isinstance(q, np.ndarray)
    assert isinstance(k, np.ndarray)
    assert isinstance(v, np.ndarray)
    np.testing.assert_array_equal(q, fused[:, :q_out].T)
    np.testing.assert_array_equal(k, fused[:, q_out:q_out + kv_out].T)
    np.testing.assert_array_equal(v, fused[:, q_out + kv_out:q_out + 2 * kv_out].T)


def test_split_fused_qkv_weight_quantized_with_interleaving():
    input_dim = 32  # Q8_0 block size.
    q_out = 4
    kv_out = 2
    total = q_out + 2 * kv_out
    row_bytes = 34
    data = np.arange(total * row_bytes, dtype=np.uint8).reshape(total, row_bytes)

    fused = quant_ops.QuantizedMatrix(
        name="blk.0.attn_qkv.weight",
        qtype=8,  # Q8_0
        shape=(input_dim, total),
        data=np.ascontiguousarray(data),
    )
    interleaved = quant_ops.interleave_quantized_rows(fused, factor=2)
    packed = interleaved.ensure_packed()

    q, k, v = _split_fused_qkv_weight(interleaved, q_out=q_out, kv_out=kv_out)

    assert isinstance(q, quant_ops.QuantizedMatrix)
    assert isinstance(k, quant_ops.QuantizedMatrix)
    assert isinstance(v, quant_ops.QuantizedMatrix)
    assert q.output_dim == q_out
    assert k.output_dim == kv_out
    assert v.output_dim == kv_out

    inv = interleaved._inverse_row_permutation
    assert inv is not None
    np.testing.assert_array_equal(q.ensure_packed(), packed[inv[:q_out]])
    np.testing.assert_array_equal(k.ensure_packed(), packed[inv[q_out:q_out + kv_out]])
    np.testing.assert_array_equal(v.ensure_packed(), packed[inv[q_out + kv_out:q_out + 2 * kv_out]])


def test_layer_cpp_capability_helpers():
    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph._layer_cpp_ok = [False, True]
    graph._layer_cpp_attention_ok = [True, True]
    graph._layer_cpp_requires_python_ffn = [True, False]

    assert graph.can_layer_cpp(0) is False
    assert graph.can_layer_cpp(1) is True
    assert graph.can_layer_cpp_attention(0) is True
    assert graph.can_layer_cpp_attention(1) is True
    assert graph.layer_requires_python_ffn(0) is True
    assert graph.layer_requires_python_ffn(1) is False


def test_init_weights_deinterleaves_quantized_rows_for_graph_pointers():
    row_bytes = 34  # q8_0 row bytes for input_dim=32
    physical = np.zeros((4, row_bytes), dtype=np.uint8)
    physical[:, 0] = np.asarray([1, 3, 0, 2], dtype=np.uint8)
    qmat = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight",
        qtype=8,
        shape=(32, 4),
        data=physical,
    )
    qmat.ensure_packed()
    qmat._validated = True
    qmat._inverse_row_permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)

    class _FakeLib:
        def __init__(self):
            self.wq_first_bytes = None

        def graph_set_layer_weights_quantized(self, *_args):
            wq_ptr = _args[4]
            addr = int(wq_ptr.value)
            buf = (ctypes.c_uint8 * (4 * row_bytes)).from_address(addr)
            arr = np.ctypeslib.as_array(buf).reshape(4, row_bytes).copy()
            self.wq_first_bytes = arr[:, 0].tolist()
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((32,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((32,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((32,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"token_embd": qmat, "output": qmat},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = 32
    graph.vocab_size = 16
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = 4
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.wq_first_bytes == [0, 1, 2, 3]


def test_init_weights_sets_quantized_token_embeddings_for_graph():
    input_dim = 32
    vocab_size = 6
    row_bytes = 34
    token_embd = quant_ops.QuantizedMatrix(
        name="token_embd.weight",
        qtype=quant_ops.QTYPE_Q8_0,
        shape=(input_dim, vocab_size),
        data=np.arange(vocab_size * row_bytes, dtype=np.uint8).reshape(vocab_size, row_bytes),
    )
    lm_head = quant_ops.QuantizedMatrix(
        name="output.weight",
        qtype=quant_ops.QTYPE_Q8_0,
        shape=(input_dim, vocab_size),
        data=np.zeros((vocab_size, row_bytes), dtype=np.uint8),
    )
    for mat in (token_embd, lm_head):
        mat.ensure_packed()
        mat._validated = True

    class _FakeLib:
        def __init__(self):
            self.embedding_qtype = None
            self.embedding_input_dim = None
            self.embedding_output_dim = None
            self.embedding_transposed = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_embedding_weights_quantized(self, *_args):
            self.embedding_qtype = int(_args[2])
            self.embedding_input_dim = int(_args[3])
            self.embedding_output_dim = int(_args[4])
            self.embedding_transposed = int(_args[5])
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((input_dim,), dtype=np.float32),
            "attn_q": _make_q8_matrix("blk.0.attn_q.weight", input_dim, input_dim),
            "attn_k": _make_q8_matrix("blk.0.attn_k.weight", input_dim, input_dim),
            "attn_v": _make_q8_matrix("blk.0.attn_v.weight", input_dim, input_dim),
            "attn_output": _make_q8_matrix("blk.0.attn_output.weight", input_dim, input_dim),
            "ffn_norm": np.ones((input_dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((input_dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"token_embd": token_embd, "output": lm_head},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = input_dim
    graph.vocab_size = vocab_size
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = input_dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.embedding_qtype == quant_ops.QTYPE_Q8_0
    assert fake_lib.embedding_input_dim == input_dim
    assert fake_lib.embedding_output_dim == vocab_size
    assert fake_lib.embedding_transposed == 1


def test_init_weights_prefers_packed_lm_head_layout_when_available(monkeypatch):
    input_dim = 256
    vocab_size = 8
    row_bytes = (input_dim // 256) * 210
    lm_head = quant_ops.QuantizedMatrix(
        name="output.weight",
        qtype=quant_ops.QTYPE_Q6_K,
        shape=(input_dim, vocab_size),
        data=np.arange(vocab_size * row_bytes, dtype=np.uint8).reshape(vocab_size, row_bytes),
    )
    lm_head.ensure_packed()
    lm_head._validated = True
    token_embd = _make_q8_matrix("token_embd.weight", 32, vocab_size)

    class _FakeLib:
        def __init__(self):
            self.head_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_embedding_weights_quantized(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            self.head_qtype = int(_args[3])
            return 1

    fake_lib = _FakeLib()
    repacked = quant_ops.QuantizedMatrix(
        name="output.weight::q6k_r4",
        qtype=quant_ops.QTYPE_Q6_K_R4,
        shape=(input_dim, vocab_size),
        data=np.ascontiguousarray(lm_head.ensure_packed()),
    )
    repacked.ensure_packed()
    repacked._validated = True
    monkeypatch.setenv("ANVIL_LM_HEAD_PACKED", "1")
    monkeypatch.setattr(quant_ops, "repack_q6k_r4", lambda matrix: repacked)

    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((32,), dtype=np.float32),
            "attn_q": _make_q8_matrix("blk.0.attn_q.weight", 32, 32),
            "attn_k": _make_q8_matrix("blk.0.attn_k.weight", 32, 32),
            "attn_v": _make_q8_matrix("blk.0.attn_v.weight", 32, 32),
            "attn_output": _make_q8_matrix("blk.0.attn_output.weight", 32, 32),
            "ffn_norm": np.ones((32,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((32,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"token_embd": token_embd, "output": lm_head},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = 32
    graph.vocab_size = vocab_size
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = 4
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []
    graph._lm_head_qtype = 0
    graph._lm_head_layout = "none"

    graph._init_weights(ws, profile)

    assert fake_lib.head_qtype == quant_ops.QTYPE_Q6_K_R4
    assert graph._lm_head_qtype == quant_ops.QTYPE_Q6_K_R4
    assert graph._lm_head_layout == "q6_k_r4"


def test_init_weights_defaults_to_packed_lm_head_layout(monkeypatch):
    input_dim = 256
    vocab_size = 8
    row_bytes = (input_dim // 256) * 210
    lm_head = quant_ops.QuantizedMatrix(
        name="output.weight",
        qtype=quant_ops.QTYPE_Q6_K,
        shape=(input_dim, vocab_size),
        data=np.arange(vocab_size * row_bytes, dtype=np.uint8).reshape(vocab_size, row_bytes),
    )
    lm_head.ensure_packed()
    lm_head._validated = True
    token_embd = _make_q8_matrix("token_embd.weight", 32, vocab_size)

    class _FakeLib:
        def __init__(self):
            self.head_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_embedding_weights_quantized(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            self.head_qtype = int(_args[3])
            return 1

    fake_lib = _FakeLib()
    repacked = quant_ops.QuantizedMatrix(
        name="output.weight::q6k_r4",
        qtype=quant_ops.QTYPE_Q6_K_R4,
        shape=(input_dim, vocab_size),
        data=np.ascontiguousarray(lm_head.ensure_packed()),
    )
    repacked.ensure_packed()
    repacked._validated = True
    monkeypatch.delenv("ANVIL_LM_HEAD_PACKED", raising=False)
    monkeypatch.setattr(quant_ops, "repack_q6k_r4", lambda matrix: repacked)

    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((32,), dtype=np.float32),
            "attn_q": _make_q8_matrix("blk.0.attn_q.weight", 32, 32),
            "attn_k": _make_q8_matrix("blk.0.attn_k.weight", 32, 32),
            "attn_v": _make_q8_matrix("blk.0.attn_v.weight", 32, 32),
            "attn_output": _make_q8_matrix("blk.0.attn_output.weight", 32, 32),
            "ffn_norm": np.ones((32,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((32,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"token_embd": token_embd, "output": lm_head},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = 32
    graph.vocab_size = vocab_size
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = 4
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []
    graph._lm_head_qtype = 0
    graph._lm_head_layout = "none"

    graph._init_weights(ws, profile)

    assert fake_lib.head_qtype == quant_ops.QTYPE_Q6_K_R4
    assert graph._lm_head_qtype == quant_ops.QTYPE_Q6_K_R4
    assert graph._lm_head_layout == "q6_k_r4"


def test_init_weights_normalizes_false_like_qwen_mrope_metadata():
    dim = 32
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, dim)

    class _FakeLib:
        def __init__(self):
            self.rope_finetuned = None
            self.mrope_interleaved = None
            self.qwen_ssm_state_size = None
            self.qwen_ssm_group_count = None
            self.qwen_ssm_n_v_heads = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

        def graph_set_qwen_mrope_config(self, *_args):
            self.rope_finetuned = int(_args[1])
            self.mrope_interleaved = int(_args[2])
            return 1

        def graph_set_qwen_hybrid_config(self, *_args):
            self.qwen_ssm_state_size = int(_args[1])
            self.qwen_ssm_group_count = int(_args[2])
            self.qwen_ssm_n_v_heads = int(_args[3])
            return 1

        def graph_set_layer_extras(self, *_args):
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(
            get_metadata=lambda: {
                "qwen35.rope.scaling.finetuned": "false",
                "qwen35.rope.mrope_interleaved": "0",
                "qwen35.ssm.state_size": 128,
                "qwen35.ssm.group_count": 4,
                "qwen35.ssm.time_step_rank": 7,
            }
        ),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = dim
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.rope_finetuned == 0
    assert fake_lib.mrope_interleaved == 0
    assert fake_lib.qwen_ssm_state_size == 128
    assert fake_lib.qwen_ssm_group_count == 4
    assert fake_lib.qwen_ssm_n_v_heads == 7


def test_init_weights_prefers_explicit_qwen_ssm_n_v_heads_metadata():
    dim = 32
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, dim)

    class _FakeLib:
        def __init__(self):
            self.qwen_ssm_n_v_heads = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

        def graph_set_qwen_mrope_config(self, *_args):
            return 1

        def graph_set_qwen_hybrid_config(self, *_args):
            self.qwen_ssm_n_v_heads = int(_args[3])
            return 1

        def graph_set_layer_extras(self, *_args):
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(
            get_metadata=lambda: {
                "qwen35.ssm.n_v_heads": 5,
                "qwen35.ssm.time_step_rank": 7,
            }
        ),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = dim
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.qwen_ssm_n_v_heads == 5


def test_get_perf_stats_reads_native_snapshot() -> None:
    class _FakeLib:
        def graph_get_perf_stats(self, _handle, snapshot_ptr):
            snapshot = ctypes.cast(
                snapshot_ptr,
                ctypes.POINTER(_GraphPerfStatsSnapshot),
            ).contents
            snapshot.embedding_lookup_seconds = 0.25
            snapshot.lm_head_seconds = 1.5
            snapshot.forward_token_calls = 7
            snapshot.forward_token_id_calls = 6
            snapshot.forward_token_ids_calls = 2
            snapshot.attention_calls = 6
            return 1

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph._lib = _FakeLib()
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_perf_stats_api = True

    stats = graph.get_perf_stats()

    assert stats["embedding_lookup_seconds"] == pytest.approx(0.25)
    assert stats["lm_head_seconds"] == pytest.approx(1.5)
    assert stats["forward_token_calls"] == 7
    assert stats["forward_token_id_calls"] == 6
    assert stats["forward_token_ids_calls"] == 2
    assert stats["attention_calls"] == 6


def test_load_binds_drift_snapshot_argtypes_when_symbols_exist(monkeypatch) -> None:
    def _create_model_graph(*_args):
        return 1

    def _graph_set_layer_weights(*_args):
        return 1

    def _graph_set_head_weights(*_args):
        return 1

    def _graph_forward_token(*_args):
        return 1

    def _graph_forward_token_id(*_args):
        return 1

    def _graph_forward_token_ids(*_args):
        return 1

    def _graph_reset(*_args):
        return 1

    def _graph_get_position(*_args):
        return 0

    def _graph_get_last_drift_snapshot(*_args):
        return 1

    def _graph_set_drift_config(*_args):
        return 1

    def _graph_get_drift_config(*_args):
        return 1

    def _destroy_model_graph(*_args):
        return None

    fake_lib = SimpleNamespace(
        create_model_graph_v2=_create_model_graph,
        graph_set_layer_weights=_graph_set_layer_weights,
        graph_set_head_weights=_graph_set_head_weights,
        graph_forward_token=_graph_forward_token,
        graph_forward_token_id=_graph_forward_token_id,
        graph_forward_token_ids=_graph_forward_token_ids,
        graph_reset=_graph_reset,
        graph_get_position=_graph_get_position,
        graph_get_last_drift_snapshot=_graph_get_last_drift_snapshot,
        graph_set_drift_config=_graph_set_drift_config,
        graph_get_drift_config=_graph_get_drift_config,
        destroy_model_graph=_destroy_model_graph,
    )
    monkeypatch.setattr(
        "core.native.model_graph_wrapper.load_native_library",
        lambda: fake_lib,
    )

    graph = NativeModelGraph(
        n_layers=1,
        embedding_dim=8,
        vocab_size=16,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        max_seq=16,
    )

    assert graph.has_full_graph
    assert graph._has_drift_snapshot_arg_api is True
    assert graph._has_drift_config_api is True
    assert graph._has_drift_state_api is True
    assert fake_lib.graph_forward_token_id.argtypes[-1] == ctypes.POINTER(
        _GraphDriftSnapshot
    )
    assert fake_lib.graph_forward_token_ids.argtypes[-1] == ctypes.POINTER(
        _GraphDriftSnapshot
    )
    graph.close()


def test_forward_token_id_captures_last_drift_snapshot() -> None:
    class _FakeLib:
        @staticmethod
        def graph_forward_token_id(_handle, _token_id, logits_out, logits_len, _pos, snapshot_ptr):
            for i in range(int(logits_len)):
                logits_out[i] = float(i)
            snapshot = ctypes.cast(
                snapshot_ptr,
                ctypes.POINTER(_GraphDriftSnapshot),
            ).contents
            snapshot.latest_drift = 0.3
            snapshot.mean_drift = 0.2
            snapshot.max_drift = 0.6
            snapshot.decay_ratio = 0.91
            snapshot.active_token_count = 1024
            snapshot.damped_block_count = 4
            snapshot.pruned_block_count = 1
            snapshot.stabilizer_seconds = 0.15
            snapshot.stabilizer_calls = 7
            snapshot.mode = 2
            return 1

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph._lib = _FakeLib()
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_drift_snapshot_arg_api = True
    graph._has_drift_state_api = False
    graph.vocab_size = 4
    graph._logits_out = (ctypes.c_float * graph.vocab_size)()
    graph._last_drift_snapshot = _GraphDriftSnapshot()
    graph._last_drift_snapshot_valid = False

    logits = graph.forward_token_id(42, 3)

    assert logits is not None
    snapshot = graph.get_last_drift_snapshot()
    assert snapshot["latest_drift"] == pytest.approx(0.3)
    assert snapshot["mean_drift"] == pytest.approx(0.2)
    assert snapshot["max_drift"] == pytest.approx(0.6)
    assert snapshot["decay_ratio"] == pytest.approx(0.91)
    assert snapshot["active_token_count"] == 1024
    assert snapshot["damped_block_count"] == 4
    assert snapshot["pruned_block_count"] == 1
    assert snapshot["stabilizer_seconds"] == pytest.approx(0.15)
    assert snapshot["stabilizer_calls"] == 7
    assert snapshot["mode"] == 2


def test_init_weights_repack_q4k_to_r4_for_graph_quantized_api(monkeypatch):
    input_dim = 256
    output_dim = 4
    row_bytes = 144
    q4k = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    q4k.ensure_packed()
    q4k._validated = True

    repacked = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight::q4k_r4",
        qtype=quant_ops.QTYPE_Q4_K_R4,
        shape=(input_dim, output_dim),
        data=np.ones((output_dim, row_bytes), dtype=np.uint8),
    )
    repacked.ensure_packed()
    repacked._validated = True

    monkeypatch.setenv("ANVIL_Q4K_R4", "1")
    monkeypatch.setattr(quant_ops, "repack_q4k_r4", lambda _m: repacked)

    class _FakeLib:
        def __init__(self):
            self.layer_qtypes = None
            self.head_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            self.layer_qtypes = (
                int(_args[5]),   # wq_qtype
                int(_args[8]),   # wk_qtype
                int(_args[11]),  # wv_qtype
                int(_args[13]),  # wo_qtype
            )
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            self.head_qtype = int(_args[3])
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((input_dim,), dtype=np.float32),
            "attn_q": q4k,
            "attn_k": q4k,
            "attn_v": q4k,
            "attn_output": q4k,
            "ffn_norm": np.ones((input_dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((input_dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": q4k},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = input_dim
    graph.vocab_size = output_dim
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = input_dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.layer_qtypes == (
        quant_ops.QTYPE_Q4_K_R4,
        quant_ops.QTYPE_Q4_K_R4,
        quant_ops.QTYPE_Q4_K_R4,
        quant_ops.QTYPE_Q4_K_R4,
    )
    assert fake_lib.head_qtype == quant_ops.QTYPE_Q4_K_R4


def test_init_weights_qwen_defaults_to_no_q4k_r4_repack(monkeypatch):
    input_dim = 256
    output_dim = 4
    row_bytes = 144
    q4k = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    q4k.ensure_packed()
    q4k._validated = True

    repack_calls = {"count": 0}

    def _track_repack(matrix):
        repack_calls["count"] += 1
        return matrix

    monkeypatch.delenv("ANVIL_Q4K_R4", raising=False)
    monkeypatch.setattr(quant_ops, "repack_q4k_r4", _track_repack)

    class _FakeLib:
        def __init__(self):
            self.layer_qtypes = None
            self.head_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            self.layer_qtypes = (
                int(_args[5]),   # wq_qtype
                int(_args[8]),   # wk_qtype
                int(_args[11]),  # wv_qtype
                int(_args[13]),  # wo_qtype
            )
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            self.head_qtype = int(_args[3])
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((input_dim,), dtype=np.float32),
            "attn_q": q4k,
            "attn_k": q4k,
            "attn_v": q4k,
            "attn_output": q4k,
            "ffn_norm": np.ones((input_dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((input_dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": q4k},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(family="qwen", architecture="qwen35")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = input_dim
    graph.vocab_size = output_dim
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = input_dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert repack_calls["count"] == 0
    assert fake_lib.layer_qtypes == (
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
    )
    assert fake_lib.head_qtype == quant_ops.QTYPE_Q4_K


def test_init_weights_qwen_can_opt_in_lm_head_q4k_r4_without_layer_repack(monkeypatch):
    input_dim = 256
    output_dim = 4
    row_bytes = 144
    q4k = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    q4k.ensure_packed()
    q4k._validated = True
    lm_head = quant_ops.QuantizedMatrix(
        name="output.weight",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(input_dim, output_dim),
        data=np.zeros((output_dim, row_bytes), dtype=np.uint8),
    )
    lm_head.ensure_packed()
    lm_head._validated = True

    repacked = quant_ops.QuantizedMatrix(
        name="output.weight::q4k_r4",
        qtype=quant_ops.QTYPE_Q4_K_R4,
        shape=(input_dim, output_dim),
        data=np.ones((output_dim, row_bytes), dtype=np.uint8),
    )
    repacked.ensure_packed()
    repacked._validated = True

    repack_calls = {"count": 0}

    def _track_repack(matrix):
        repack_calls["count"] += 1
        return repacked if getattr(matrix, "name", "") == "output.weight" else matrix

    monkeypatch.delenv("ANVIL_Q4K_R4", raising=False)
    monkeypatch.setenv("ANVIL_LM_HEAD_PACKED", "1")
    monkeypatch.setenv("ANVIL_LM_HEAD_Q4K_R4", "1")
    monkeypatch.setattr(quant_ops, "repack_q4k_r4", _track_repack)

    class _FakeLib:
        def __init__(self):
            self.layer_qtypes = None
            self.head_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            self.layer_qtypes = (
                int(_args[5]),
                int(_args[8]),
                int(_args[11]),
                int(_args[13]),
            )
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            self.head_qtype = int(_args[3])
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((input_dim,), dtype=np.float32),
            "attn_q": q4k,
            "attn_k": q4k,
            "attn_v": q4k,
            "attn_output": q4k,
            "ffn_norm": np.ones((input_dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((input_dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": lm_head},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(family="qwen", architecture="qwen35")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = input_dim
    graph.vocab_size = output_dim
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = input_dim
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert repack_calls["count"] == 1
    assert fake_lib.layer_qtypes == (
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
        quant_ops.QTYPE_Q4_K,
    )
    assert fake_lib.head_qtype == quant_ops.QTYPE_Q4_K_R4


def test_init_weights_transposes_granite_router_for_layer_extras():
    dim = 32
    hidden = 32
    expert_count = 2
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, hidden)
    router = np.arange(dim * expert_count, dtype=np.float32).reshape(dim, expert_count)

    class _FakeLib:
        def __init__(self):
            self.router_rows = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_layer_extras(self, *_args):
            router_ptr = _args[13]
            addr = int(ctypes.cast(router_ptr, ctypes.c_void_p).value or 0)
            assert addr != 0
            buf = (ctypes.c_float * (dim * expert_count)).from_address(addr)
            self.router_rows = (
                np.ctypeslib.as_array(buf).reshape(expert_count, dim).copy()
            )
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    gate_name = "blk.0.ffn_gate_exps.weight"
    up_name = "blk.0.ffn_up_exps.weight"
    down_name = "blk.0.ffn_down_exps.weight"
    fake_tensor = SimpleNamespace(shape=(dim, hidden, expert_count))
    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        _tensor_index={
            gate_name: fake_tensor,
            up_name: fake_tensor,
            down_name: fake_tensor,
        },
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
            "ffn_gate_inp": router,
            "ffn_gate_shexp": qmat,
            "ffn_up_shexp": qmat,
            "ffn_down_shexp": qmat,
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (gate_name, up_name, down_name),
        get_expert_matrix=lambda _name, _idx: qmat,
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(
            get_metadata=lambda: {"granitehybrid.expert_used_count": expert_count}
        ),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = hidden
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = hidden
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.router_rows is not None
    np.testing.assert_array_equal(fake_lib.router_rows, router.T)


def test_init_weights_granite_caps_moe_top_k_to_valid_expert_count(monkeypatch):
    dim = 32
    hidden = 32
    expert_count = 3
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, hidden)
    router = np.arange(dim * expert_count, dtype=np.float32).reshape(dim, expert_count)

    class _FakeLib:
        def __init__(self):
            self.moe_top_k = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_layer_extras(self, *_args):
            self.moe_top_k = int(_args[-1])
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    gate_name = "blk.0.ffn_gate_exps.weight"
    up_name = "blk.0.ffn_up_exps.weight"
    down_name = "blk.0.ffn_down_exps.weight"
    fake_tensor = SimpleNamespace(shape=(dim, hidden, expert_count))
    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        _tensor_index={
            gate_name: fake_tensor,
            up_name: fake_tensor,
            down_name: fake_tensor,
        },
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
            "ffn_gate_inp": router,
            "ffn_gate_shexp": qmat,
            "ffn_up_shexp": qmat,
            "ffn_down_shexp": qmat,
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (gate_name, up_name, down_name),
        get_expert_matrix=lambda _name, _idx: qmat,
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(
            get_metadata=lambda: {"granitehybrid.expert_used_count": 7}
        ),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = hidden
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = hidden
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    monkeypatch.setenv("ANVIL_GRANITE_MAX_MOE_TOP_K", "0")
    graph._init_weights(ws, profile)

    assert fake_lib.moe_top_k == expert_count


def test_init_weights_granite_routed_bundle_accepts_q4k_r4_repack(monkeypatch):
    dim = 256
    hidden = 4
    expert_count = 2
    q4k_row_bytes = 144
    q6k_row_bytes = 210
    gate = quant_ops.QuantizedMatrix(
        name="blk.0.ffn_gate_exps.weight::expert0",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(dim, hidden),
        data=np.zeros((hidden, q4k_row_bytes), dtype=np.uint8),
    )
    up = quant_ops.QuantizedMatrix(
        name="blk.0.ffn_up_exps.weight::expert0",
        qtype=quant_ops.QTYPE_Q4_K,
        shape=(dim, hidden),
        data=np.zeros((hidden, q4k_row_bytes), dtype=np.uint8),
    )
    down = quant_ops.QuantizedMatrix(
        name="blk.0.ffn_down_exps.weight::expert0",
        qtype=quant_ops.QTYPE_Q6_K,
        shape=(hidden, dim),
        data=np.zeros((dim, q6k_row_bytes), dtype=np.uint8),
    )
    for mat in (gate, up, down):
        mat.ensure_packed()
        mat._validated = True

    repacked_gate = quant_ops.QuantizedMatrix(
        name="blk.0.ffn_gate_exps.weight::expert0::q4k_r4",
        qtype=quant_ops.QTYPE_Q4_K_R4,
        shape=(dim, hidden),
        data=np.ones((hidden, q4k_row_bytes), dtype=np.uint8),
    )
    repacked_up = quant_ops.QuantizedMatrix(
        name="blk.0.ffn_up_exps.weight::expert0::q4k_r4",
        qtype=quant_ops.QTYPE_Q4_K_R4,
        shape=(dim, hidden),
        data=np.ones((hidden, q4k_row_bytes), dtype=np.uint8),
    )
    for mat in (repacked_gate, repacked_up):
        mat.ensure_packed()
        mat._validated = True

    def _repack(matrix):
        if matrix.name.startswith("blk.0.ffn_gate_exps.weight"):
            return repacked_gate
        if matrix.name.startswith("blk.0.ffn_up_exps.weight"):
            return repacked_up
        return matrix

    monkeypatch.setenv("ANVIL_Q4K_R4", "1")
    monkeypatch.setattr(quant_ops, "repack_q4k_r4", _repack)

    class _FakeLib:
        def __init__(self):
            self.expert_gate_qtype = None
            self.expert_up_qtype = None
            self.expert_down_qtype = None

        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_layer_extras(self, *_args):
            self.expert_gate_qtype = int(_args[37])
            self.expert_up_qtype = int(_args[40])
            self.expert_down_qtype = int(_args[42])
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    gate_name = "blk.0.ffn_gate_exps.weight"
    up_name = "blk.0.ffn_up_exps.weight"
    down_name = "blk.0.ffn_down_exps.weight"
    fake_tensor = SimpleNamespace(shape=(dim, hidden, expert_count))
    router = np.arange(dim * expert_count, dtype=np.float32).reshape(dim, expert_count)
    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        _tensor_index={
            gate_name: fake_tensor,
            up_name: fake_tensor,
            down_name: fake_tensor,
        },
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": _make_q8_matrix("blk.0.attn_q.weight", dim, hidden),
            "attn_k": _make_q8_matrix("blk.0.attn_k.weight", dim, hidden),
            "attn_v": _make_q8_matrix("blk.0.attn_v.weight", dim, hidden),
            "attn_output": _make_q8_matrix("blk.0.attn_output.weight", dim, hidden),
            "ffn_norm": np.ones((dim,), dtype=np.float32),
            "ffn_gate_inp": router,
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (gate_name, up_name, down_name),
        get_expert_matrix=lambda name, _idx: (
            gate if name == gate_name else up if name == up_name else down
        ),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": _make_q8_matrix("output.weight", dim, hidden)},
        loader=SimpleNamespace(
            get_metadata=lambda: {"granitehybrid.expert_used_count": expert_count}
        ),
    )
    profile = SimpleNamespace(architecture="granitehybrid", family="granite")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = hidden
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = hidden
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.expert_gate_qtype == quant_ops.QTYPE_Q4_K_R4
    assert fake_lib.expert_up_qtype == quant_ops.QTYPE_Q4_K_R4
    assert fake_lib.expert_down_qtype == quant_ops.QTYPE_Q6_K


def test_init_weights_granite_invalid_shared_bundle_fails_closed():
    dim = 32
    hidden = 32
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, hidden)

    class _FakeLib:
        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_layer_extras(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    ws = SimpleNamespace(
        _tensor_index={},
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
            "ffn_gate_shexp": np.ones((hidden, dim), dtype=np.float32),
            "ffn_up_shexp": qmat,
            "ffn_down_shexp": qmat,
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = hidden
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = hidden
    graph._lib = _FakeLib()
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    with pytest.raises(RuntimeError, match="invalid shared expert bundle"):
        graph._init_weights(ws, profile)


def test_init_weights_granite_invalid_routed_bundle_fails_closed():
    dim = 32
    hidden = 32
    expert_count = 2
    qmat = _make_q8_matrix("blk.0.attn_q.weight", dim, hidden)
    router = np.arange(dim * expert_count, dtype=np.float32).reshape(dim, expert_count)

    class _FakeLib:
        def graph_set_layer_weights_quantized(self, *_args):
            return 1

        def graph_set_layer_extras(self, *_args):
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    gate_name = "blk.0.ffn_gate_exps.weight"
    up_name = "blk.0.ffn_up_exps.weight"
    down_name = "blk.0.ffn_down_exps.weight"
    fake_tensor = SimpleNamespace(shape=(dim, hidden, expert_count))
    ws = SimpleNamespace(
        _tensor_index={
            gate_name: fake_tensor,
            up_name: fake_tensor,
            down_name: fake_tensor,
        },
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": qmat,
            "attn_v": qmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
            "ffn_gate_inp": router,
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (gate_name, up_name, down_name),
        get_expert_matrix=lambda _name, _idx: np.ones((hidden, dim), dtype=np.float32),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="granitehybrid")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = hidden
    graph.n_heads = 1
    graph.n_kv_heads = 1
    graph.head_dim = hidden
    graph._lib = _FakeLib()
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._has_extended_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    with pytest.raises(RuntimeError, match="invalid routed expert bundle"):
        graph._init_weights(ws, profile)


def test_init_weights_uses_per_layer_qwen_kv_dims():
    dim = 32
    q_out = 64
    kv_out = 16
    row_bytes = 34  # q8_0 row bytes for input_dim=32
    qmat = quant_ops.QuantizedMatrix(
        name="blk.0.attn_q.weight",
        qtype=8,
        shape=(dim, q_out),
        data=np.zeros((q_out, row_bytes), dtype=np.uint8),
    )
    kmat = quant_ops.QuantizedMatrix(
        name="blk.0.attn_k.weight",
        qtype=8,
        shape=(dim, kv_out),
        data=np.zeros((kv_out, row_bytes), dtype=np.uint8),
    )
    vmat = quant_ops.QuantizedMatrix(
        name="blk.0.attn_v.weight",
        qtype=8,
        shape=(dim, kv_out),
        data=np.zeros((kv_out, row_bytes), dtype=np.uint8),
    )
    for mat in (qmat, kmat, vmat):
        mat.ensure_packed()
        mat._validated = True

    class _FakeLib:
        def __init__(self):
            self.q_out_dim = None
            self.kv_out_dim = None

        def graph_set_layer_weights_quantized(self, *_args):
            self.q_out_dim = int(_args[6])
            self.kv_out_dim = int(_args[9])
            return 1

        def graph_set_head_weights_quantized(self, *_args):
            return 1

    fake_lib = _FakeLib()
    ws = SimpleNamespace(
        get_layer_weights=lambda _idx: {
            "attn_norm": np.ones((dim,), dtype=np.float32),
            "attn_q": qmat,
            "attn_k": kmat,
            "attn_v": vmat,
            "attn_output": qmat,
            "ffn_norm": np.ones((dim,), dtype=np.float32),
        },
        get_layer_type=lambda _idx: "attention",
        get_expert_tensor_names=lambda _idx: (None, None, None),
        get_tensor=lambda name: np.ones((dim,), dtype=np.float32)
        if name == "output_norm.weight"
        else None,
        get_embedding_weights=lambda: {"output": qmat},
        loader=SimpleNamespace(get_metadata=lambda: {}),
    )
    profile = SimpleNamespace(architecture="qwen35", family="qwen")

    graph = NativeModelGraph.__new__(NativeModelGraph)
    graph.n_layers = 1
    graph.embedding_dim = dim
    graph.vocab_size = q_out
    graph.n_heads = 2
    graph.n_kv_heads = 2
    graph.head_dim = dim // 2
    graph._lib = fake_lib
    graph._handle = 1
    graph._has_full_graph = True
    graph._has_quantized_api = True
    graph._layer_cpp_ok = []
    graph._layer_cpp_attention_ok = []
    graph._layer_cpp_requires_python_ffn = []
    graph._weight_refs = []

    graph._init_weights(ws, profile)

    assert fake_lib.q_out_dim == graph.n_heads * graph.head_dim
    assert fake_lib.kv_out_dim == kv_out


def _run_simple_graph_decode_once() -> tuple[np.ndarray, np.ndarray]:
    graph = NativeModelGraph(
        n_layers=1,
        embedding_dim=8,
        vocab_size=8,
        n_heads=1,
        n_kv_heads=1,
        head_dim=8,
        max_seq=16,
    )
    if not graph.has_full_graph:
        raise RuntimeError("Native graph API unavailable")

    dim = 8
    float_p = ctypes.POINTER(ctypes.c_float)
    ones = np.ones((dim,), dtype=np.float32)
    zeros = np.zeros((dim, dim), dtype=np.float32)
    eye = np.eye(dim, dtype=np.float32)
    lm_head = np.eye(dim, dtype=np.float32)
    refs = [ones, zeros, eye, lm_head]
    graph._weight_refs.extend(refs)

    ok = graph._lib.graph_set_layer_weights(
        ctypes.c_void_p(graph._handle),
        0,
        ones.ctypes.data_as(float_p),  # attn_norm
        None,                           # ffn_norm (disable ffn branch)
        zeros.ctypes.data_as(float_p),  # wq
        dim,
        zeros.ctypes.data_as(float_p),  # wk
        dim,
        eye.ctypes.data_as(float_p),    # wv
        eye.ctypes.data_as(float_p),    # wo
        None, 0,                        # w_gate, ffn_dim
        None,                           # w_up
        None,                           # w_down
        1,                              # is_attention
    )
    assert int(ok) == 1

    ok = graph._lib.graph_set_head_weights(
        ctypes.c_void_p(graph._handle),
        ones.ctypes.data_as(float_p),   # final_norm
        lm_head.ctypes.data_as(float_p),
        0,                              # not transposed
        ctypes.c_float(0.0),
        ctypes.c_float(0.0),
        ctypes.c_float(0.0),
        ctypes.c_float(0.0),
    )
    assert int(ok) == 1

    token0 = np.linspace(-1.0, 1.0, dim, dtype=np.float32)
    token1 = np.linspace(1.0, -1.0, dim, dtype=np.float32)
    out0 = graph.forward_token(token0, 0)
    out1 = graph.forward_token(token1, 1)
    graph.close()

    assert out0 is not None
    assert out1 is not None
    return out0, out1


def test_graph_q8_kv_mode_matches_f32_mode(monkeypatch):
    try:
        monkeypatch.setenv("ANVIL_KV_QUANT", "off")
        ref0, ref1 = _run_simple_graph_decode_once()
        monkeypatch.setenv("ANVIL_KV_QUANT", "q8")
        got0, got1 = _run_simple_graph_decode_once()
    except FileNotFoundError:
        pytest.skip("Native library unavailable")
    except RuntimeError:
        pytest.skip("Native graph API unavailable")

    assert np.isfinite(got0).all()
    assert np.isfinite(got1).all()
    np.testing.assert_allclose(got0, ref0, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(got1, ref1, rtol=5e-2, atol=5e-2)
