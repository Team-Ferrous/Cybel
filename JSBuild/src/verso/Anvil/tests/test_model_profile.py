import builtins
import importlib
import sys
from types import SimpleNamespace

import numpy as np

from core.model.model_profile import ModelProfile


def _loader_with_metadata(
    metadata, tensor_names, *, vocab_tokens=None, vocab_size=32000, embedding_dim=4096
):
    tensors = [SimpleNamespace(name=name) for name in tensor_names]
    return SimpleNamespace(
        get_metadata=lambda: metadata,
        reader=SimpleNamespace(tensors=tensors),
        get_vocab_size=lambda: vocab_size,
        get_embedding_dim=lambda: embedding_dim,
        get_layer_count=lambda: 32,
        get_vocab_tokens=lambda: vocab_tokens or [],
    )


def test_gguf_loader_import_does_not_require_scipy(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "core.model.gguf_loader", raising=False)
    monkeypatch.delitem(sys.modules, "scipy", raising=False)
    monkeypatch.delitem(sys.modules, "scipy.linalg", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "gguf",
        SimpleNamespace(
            GGUFReader=object, dequantize=lambda data, tensor_type: data
        ),
    )
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    module = importlib.import_module("core.model.gguf_loader")
    assert module._svd is module.np.linalg.svd


def test_gguf_metadata_decodes_byte_array_architecture():
    from core.model.gguf_loader import GGUFModelLoader

    loader = GGUFModelLoader.__new__(GGUFModelLoader)
    loader._metadata = {}
    loader._reader = SimpleNamespace(
        fields={
            "general.architecture": SimpleNamespace(
                parts=[np.array([113, 119, 101, 110, 51, 53], dtype=np.uint8)]
            )
        }
    )

    metadata = loader.get_metadata()
    assert metadata["general.architecture"] == "qwen35"


def test_qwen_profile_detects_gqa_and_defaults_without_cli():
    loader = _loader_with_metadata(
        {
            "general.architecture": "qwen35",
            "qwen35.attention.head_count": 16,
            "qwen35.attention.head_count_kv": 4,
            "tokenizer.chat_template": "<|im_start|>{role}\n{content}<|im_end|>\n",
        },
        tensor_names=["token_embd.weight"],
        vocab_size=248_320,
        embedding_dim=4096,
    )
    profile = ModelProfile.from_loader("qwen3.5:9b", loader)
    assert profile.family == "qwen"
    assert profile.gqa is True
    assert profile.chat_template == "chatml"
    assert profile.propagator_strategy == "mlp"
    assert profile.tied_embeddings is True
    assert profile.vocab_size == 248_320
    assert profile.embedding_dim == 4096
    assert profile.n_layers == 32
    assert profile.n_heads == 16
    assert profile.n_kv_heads == 4
    assert profile.has_moe is False
    assert profile.speculative_enabled is False


def test_qwen_profile_infers_family_from_byte_architecture():
    loader = _loader_with_metadata(
        {"general.architecture": [113, 119, 101, 110, 51, 53]},
        tensor_names=["token_embd.weight"],
    )
    profile = ModelProfile.from_loader("custom-model-name", loader)
    assert profile.family == "qwen"
    assert profile.architecture == "qwen35"
    assert profile.propagator_strategy == "mlp"


def test_settings_override_applies_for_granite():
    loader = _loader_with_metadata(
        {
            "general.architecture": "granitehybrid",
            "granitehybrid.attention.head_count": 12,
            "granitehybrid.attention.head_count_kv": 12,
            "tokenizer.chat_template": "<|start_of_role|>{role}<|end_of_role|>",
        },
        tensor_names=["token_embd.weight", "output.weight", "blk.0.ffn_gate_exps.weight"],
        vocab_size=100_352,
        embedding_dim=1536,
    )
    profile = ModelProfile.from_loader("granite4:tiny-h", loader)
    assert profile.family == "granite"
    assert profile.chat_template == "granite"
    assert profile.spec_num_candidates == 4
    assert profile.spec_acceptance_threshold == 0.65
    assert profile.speculative_enabled is False
    assert profile.tied_embeddings is False
    assert profile.vocab_size == 100_352
    assert profile.embedding_dim == 1536
    assert profile.n_heads == 12
    assert profile.n_kv_heads == 12
    assert profile.has_moe is True


def test_template_detection_uses_vocab_tokens_before_cli():
    loader = _loader_with_metadata(
        {"general.architecture": "qwen35"},
        tensor_names=["token_embd.weight"],
        vocab_tokens=["<bos>", "<|im_start|>", "<|im_end|>"],
    )
    profile = ModelProfile.from_loader("qwen3.5:9b", loader)
    assert profile.chat_template == "chatml"


def test_qwen_template_detection_prefers_metadata_over_vocab_tokens():
    loader = _loader_with_metadata(
        {
            "general.architecture": "qwen35",
            "tokenizer.chat_template": "<|im_start|>{role}\n{content}<|im_end|>\n",
        },
        tensor_names=["token_embd.weight"],
        vocab_tokens=["<bos>", "<|start_of_role|>", "<|end_of_role|>"],
    )
    profile = ModelProfile.from_loader("qwen3.5:9b", loader)
    assert profile.chat_template == "chatml"


def test_template_detection_falls_back_to_family_default_without_template_markers():
    loader = _loader_with_metadata(
        {"general.architecture": "qwen35"},
        tensor_names=["token_embd.weight"],
        vocab_tokens=["plain", "tokens", "only"],
    )
    profile = ModelProfile.from_loader("qwen3.5:9b", loader)
    assert profile.chat_template == "chatml"


def test_generic_models_do_not_default_to_granite_template():
    loader = _loader_with_metadata(
        {"general.architecture": "gpt2"},
        tensor_names=["token_embd.weight"],
        vocab_tokens=["plain", "tokens", "only"],
    )
    profile = ModelProfile.from_loader("custom-model-name", loader)
    assert profile.chat_template == ""


def test_granite_template_detection_uses_vocab_tokens_before_name_heuristic():
    loader = _loader_with_metadata(
        {"general.architecture": "granitehybrid"},
        tensor_names=["token_embd.weight"],
        vocab_tokens=["<bos>", "<|start_of_role|>", "<|end_of_role|>"],
    )
    profile = ModelProfile.from_loader("granite4:tiny-h", loader)
    assert profile.chat_template == "granite"


def test_qwen_profile_infers_kv_heads_from_fused_qkv_shape():
    loader = SimpleNamespace(
        get_metadata=lambda: {
            "general.architecture": "qwen35",
            "qwen35.attention.head_count": 16,
            # Metadata can be malformed/packed; prefer tensor-shape inference when available.
            "qwen35.attention.head_count_kv": 4,
        },
        reader=SimpleNamespace(
            tensors=[
                SimpleNamespace(name="token_embd.weight", shape=(4096, 248320)),
                SimpleNamespace(name="blk.0.attn_qkv.weight", shape=(4096, 8192)),
            ]
        ),
        get_vocab_size=lambda: 248_320,
        get_embedding_dim=lambda: 4096,
        get_layer_count=lambda: 32,
        get_vocab_tokens=lambda: [],
    )

    profile = ModelProfile.from_loader("qwen3.5:9b", loader)
    assert profile.n_heads == 16
    assert profile.n_kv_heads == 8
    assert profile.gqa is True
