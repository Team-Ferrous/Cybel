import numpy as np
from types import SimpleNamespace

from core.native.engine import NativeInferenceEngine


class _FakeLoader:
    def __init__(self, _model_name):
        self.model_path = "/tmp/fake.gguf"
        self._emb = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._lm = self._emb.copy()
        self._tensors = {
            "token_embd.weight": self._emb,
            "output.weight": self._lm,
            "output_norm.weight": np.ones((4,), dtype=np.float32),
            "blk.0.attn_norm.weight": np.ones((4,), dtype=np.float32),
            "blk.0.attn_q.weight": np.eye(4, dtype=np.float32),
            "blk.0.attn_k.weight": np.eye(4, dtype=np.float32),
            "blk.0.attn_v.weight": np.eye(4, dtype=np.float32),
            "blk.0.attn_output.weight": np.eye(4, dtype=np.float32),
            "blk.0.ffn_norm.weight": np.ones((4,), dtype=np.float32),
            "blk.0.ffn_gate.weight": np.eye(4, dtype=np.float32),
            "blk.0.ffn_up.weight": np.eye(4, dtype=np.float32),
            "blk.0.ffn_down.weight": np.eye(4, dtype=np.float32),
        }
        self.reader = SimpleNamespace(
            tensors=[SimpleNamespace(name=name, shape=value.shape) for name, value in self._tensors.items()]
        )

    def get_metadata(self):
        return {
            "general.architecture": "granite",
            "granite.attention.head_count": 1,
            "granite.attention.head_count_kv": 1,
            "tokenizer.ggml.add_bos_token": True,
            "tokenizer.chat_template": "{role}: {content}",
        }

    def get_vocab_tokens(self):
        return ["<pad>", "<bos>", "<eos>", "hello", "world", "answer"]

    def get_special_tokens(self):
        return {"bos": 1, "eos": 2, "pad": 0}

    def get_token_embeddings(self):
        return self._emb

    def get_lm_head_weight(self):
        return self._lm

    def get_tensor(self, name):
        return self._tensors.get(name)

    def get_layer_count(self):
        return 1

    def get_embedding_dim(self):
        return 4

    def get_vocab_size(self):
        return 6

    def get_context_length(self):
        return 256

    def close(self):
        return None


class _FakeGraph:
    def __init__(self, **kwargs):
        self.available = True
        self._has_full_graph = True
        self._handle = 1
        self._vocab = int(kwargs.get("vocab_size", 6))

    @property
    def has_full_graph(self):
        return self._has_full_graph and self.available

    @property
    def has_hybrid_mode(self):
        return False

    def forward_token(self, embedding, position):
        out = np.full((self._vocab,), -10.0, dtype=np.float32)
        idx = min(3, self._vocab - 1)
        out[idx] = 10.0
        return out

    def reset(self):
        return None

    def close(self):
        self.available = False


class _FakeTokenizer:
    eos_id = 2

    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def encode(text, add_bos=True):
        base = [3] if str(text).strip() else []
        return ([1] + base) if add_bos else base

    @staticmethod
    def decode(tokens, skip_special=True):
        del skip_special
        return " ".join(str(token) for token in tokens)


class _FakeDecodeRuntime:
    def __init__(self, token_sequence=None):
        generated_tokens = list(token_sequence or [3, 3, 3])
        self._events = [
            SimpleNamespace(token_id=token, done=False, error=None)
            for token in generated_tokens
        ]
        self._events.append(SimpleNamespace(token_id=None, done=True, error=None))

    def submit(self, request_id, **kwargs):
        del request_id, kwargs

    def poll(self, request_id):
        del request_id
        if not self._events:
            return None
        return self._events.pop(0)

    @staticmethod
    def metrics():
        return SimpleNamespace(
            prefill_batches=1,
            runtime_prefill_tokens=1,
            runtime_decode_steps=1,
        )

    @staticmethod
    def close():
        return None


def _patch_native_runtime(monkeypatch):
    monkeypatch.setattr("core.native.native_qsg_engine.NativeModelGraph", _FakeGraph)
    monkeypatch.setattr("core.native.native_qsg_engine.NativeTokenizer", _FakeTokenizer)
    monkeypatch.setattr("core.native.native_qsg_engine.AnvilTokenizer", _FakeTokenizer)
    monkeypatch.setattr("core.native.native_qsg_engine.NativeKVCacheWrapper", None)
    monkeypatch.setattr("core.native.native_qsg_engine.SSMSelfSpeculativeDecoder", None)
    monkeypatch.setattr(
        "core.native.native_qsg_engine.NativeQSGEngine._build_decode_runtime",
        lambda self: _FakeDecodeRuntime(),
    )
    monkeypatch.setattr(
        "core.native.native_qsg_engine.simd_ops.get_num_threads",
        lambda: 1,
    )
    monkeypatch.setattr(
        "core.native.native_qsg_engine.simd_ops.set_num_threads",
        lambda _n: None,
    )


def test_native_qsg_engine_generate(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_ENGINE_BACKEND", raising=False)
    monkeypatch.setattr("core.native.native_qsg_engine.GGUFModelLoader", _FakeLoader)
    _patch_native_runtime(monkeypatch)

    engine = NativeInferenceEngine("granite4:tiny-h", context_length=32)

    assert engine.backend == "native_qsg"
    assert engine.supports_token_api is True

    prompt_tokens = engine.tokenize("hello")
    generated = engine.generate(prompt_tokens, max_new_tokens=3, temperature=0.0)

    assert len(generated) >= len(prompt_tokens)
    assert all(isinstance(token, int) for token in generated)


def test_native_qsg_stream_matches_generate(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_ENGINE_BACKEND", raising=False)
    monkeypatch.setattr("core.native.native_qsg_engine.GGUFModelLoader", _FakeLoader)
    _patch_native_runtime(monkeypatch)

    engine = NativeInferenceEngine("granite4:tiny-h", context_length=32)
    prompt_tokens = engine.tokenize("hello")

    generated = engine.generate(prompt_tokens, max_new_tokens=2, temperature=0.0)
    streamed = list(
        engine.generate_stream(prompt_tokens, max_new_tokens=2, temperature=0.0)
    )

    assert streamed == generated[len(prompt_tokens) :]


class _FakeLoaderNoBos(_FakeLoader):
    def get_metadata(self):
        metadata = super().get_metadata()
        metadata["tokenizer.ggml.add_bos_token"] = False
        return metadata


def test_native_qsg_tokenize_respects_add_bos_metadata(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_ENGINE_BACKEND", raising=False)
    monkeypatch.setattr("core.native.native_qsg_engine.GGUFModelLoader", _FakeLoaderNoBos)
    _patch_native_runtime(monkeypatch)

    engine = NativeInferenceEngine("granite4:tiny-h", context_length=32)
    tokens = engine.tokenize("hello")

    assert tokens
    assert tokens[0] != 1
