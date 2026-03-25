from types import SimpleNamespace
import sys

import numpy as np
import pytest

from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGChunk
from core.qsg.ollama_adapter import OllamaQSGAdapter
from config.settings import PERFORMANCE_CONFIG


class _FakeLoader:
    def __init__(
        self,
        architecture: str = "granite",
        head_count: int = 16,
        head_count_kv: int = 16,
        seed: int = 0,
    ):
        rng = np.random.RandomState(seed)
        self.model_path = "/tmp/fake.gguf"
        self._emb = rng.randn(256, 32).astype(np.float32)
        self._lm = rng.randn(256, 32).astype(np.float32)
        self.extract_calls = []
        self.reader = SimpleNamespace(
            tensors=[SimpleNamespace(name="blk.0.ffn_down.weight")]
        )
        self._architecture = architecture
        self._head_count = head_count
        self._head_count_kv = head_count_kv

    def get_vocab_tokens(self):
        return [f"tok_{i}" for i in range(256)]

    def get_special_tokens(self):
        return {"eos": 2}

    def get_token_embeddings(self):
        return self._emb

    def get_lm_head_weight(self):
        return self._lm

    def extract_propagator(self, rank=128, layers=4, strategy="auto", profile=None):
        self.extract_calls.append(
            {"rank": rank, "layers": layers, "strategy": strategy, "profile": profile}
        )
        return np.eye(self._emb.shape[1], dtype=np.float32)

    def get_metadata(self):
        prefix = "qwen" if "qwen" in self._architecture else "granite"
        chat_template = (
            "<|im_start|>{role}\n{content}<|im_end|>\n"
            if "qwen" in self._architecture
            else "<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
        )
        return {
            "general.architecture": self._architecture,
            f"{prefix}.attention.head_count": self._head_count,
            f"{prefix}.attention.head_count_kv": self._head_count_kv,
            "tokenizer.chat_template": chat_template,
        }

    def get_vocab_size(self):
        return self._emb.shape[0]

    def get_embedding_dim(self):
        return self._emb.shape[1]

    def get_layer_count(self):
        return 4

    def get_tensor(self, name):
        if name == "output.weight":
            return None
        return None


class _FakeNativeEngine:
    def __init__(self):
        self.context_length = 2048
        self.supports_token_api = True
        self.tokenize_calls = 0

    def detokenize(self, ids):
        return "ctx"

    def tokenize(self, text):
        self.tokenize_calls += 1
        return [1, 2, 3]

    def generate(
        self,
        prompt_tokens,
        max_new_tokens=20,
        temperature=0.8,
        logits_processor=None,
    ):
        return list(prompt_tokens) + [42]

    def generate_stream(
        self,
        prompt_tokens,
        max_new_tokens=20,
        temperature=0.8,
        logits_processor=None,
    ):
        yield 42


class _FakeTokenizer:
    eos_id = 2

    def __init__(self, *_args, **_kwargs):
        pass

    @staticmethod
    def encode(_text, add_bos=True):
        return [1, 2, 3] if add_bos else [2, 3]

    @staticmethod
    def decode(_tokens, skip_special=True):
        del skip_special
        return "ctx"


class _FakeCoconutBridge:
    def expand_paths(self, hidden_state, num_paths, noise_scale=0.01):
        return np.repeat(hidden_state[:, None, :], num_paths, axis=1)

    def evolve_paths(self, paths, *args):
        paths += 0.01

    def score_paths(self, paths, context):
        batch, num_paths, _ = paths.shape
        return np.ones((batch, num_paths), dtype=np.float32) / float(num_paths)

    def aggregate_paths(self, paths, amplitudes):
        weights = amplitudes[:, :, None]
        return np.sum(paths * weights, axis=1)

    def verify_paths_with_consistency(
        self, paths, amplitudes, verification_threshold=0.7
    ):
        return self.aggregate_paths(paths, amplitudes), np.array(
            [0.5], dtype=np.float32
        )


class _CaptureGroverAmplifier:
    def __init__(self):
        self.calls = []

    def amplify_with_resonance(
        self,
        logits_reshaped,
        tokens,
        context_text,
        iterations,
        model_profile,
        top_k_oracle,
        damping,
    ):
        self.calls.append(
            {
                "iterations": iterations,
                "model_profile": model_profile,
                "top_k_oracle": top_k_oracle,
                "damping": damping,
            }
        )
        probs = np.exp(logits_reshaped - np.max(logits_reshaped, axis=1, keepdims=True))
        return probs / np.sum(probs, axis=1, keepdims=True)


def _patch_loaders(monkeypatch, loaders):
    monkeypatch.setattr(
        "core.qsg.ollama_adapter.get_loader", lambda model: loaders[model]
    )
    monkeypatch.setattr(
        "core.qsg.ollama_adapter.AnvilTokenizer",
        _FakeTokenizer,
    )
    monkeypatch.setattr(
        "core.qsg.ollama_adapter.NativeTokenizer",
        _FakeTokenizer,
    )
    monkeypatch.setattr(
        "core.native.engine.NativeInferenceEngine",
        lambda *args, **kwargs: _FakeNativeEngine(),
    )
    monkeypatch.setattr(
        "core.native.coconut_bridge.CoconutNativeBridge",
        lambda *args, **kwargs: _FakeCoconutBridge(),
    )


def test_apply_sampling_profile_normalizes_token_limit_aliases() -> None:
    adapter = OllamaQSGAdapter.__new__(OllamaQSGAdapter)
    adapter.model_name = "granite4:tiny-h"
    adapter.strict_native_qsg = False
    adapter._resolve_granite_profile_name = lambda prompt, resolved: "coding_balanced"

    resolved = adapter._apply_sampling_profile({"max_tokens": 7}, prompt="prompt")
    fallback = adapter._apply_sampling_profile({"max_new_tokens": 9}, prompt="prompt")

    assert resolved["num_predict"] == 7
    assert fallback["num_predict"] == 9


def _attach_coconut_weights(adapter):
    adapter.coconut_bridge = _FakeCoconutBridge()
    adapter.evolution_weights = {
        "norm_gamma": np.ones(32, dtype=np.float32),
        "norm_beta": np.zeros(32, dtype=np.float32),
        "w1": np.eye(32, dtype=np.float32),
        "b1": np.zeros(32, dtype=np.float32),
        "w2": np.eye(32, dtype=np.float32),
        "b2": np.zeros(32, dtype=np.float32),
        "hidden_dim": 32,
    }


def test_adapter_init_fails_fast_when_native_engine_unavailable(monkeypatch):
    fake_loader = _FakeLoader()
    monkeypatch.setattr("core.qsg.ollama_adapter.get_loader", lambda model: fake_loader)
    monkeypatch.setattr("core.qsg.ollama_adapter.AnvilTokenizer", _FakeTokenizer)
    monkeypatch.setattr("core.qsg.ollama_adapter.NativeTokenizer", _FakeTokenizer)

    def _raise_engine(*args, **kwargs):
        raise RuntimeError("no engine")

    monkeypatch.setattr("core.native.engine.NativeInferenceEngine", _raise_engine)

    with pytest.raises(
        RuntimeError, match="Failed to initialize NativeInferenceEngine"
    ):
        OllamaQSGAdapter("granite4:tiny-h", config=QSGConfig())


@pytest.mark.parametrize(
    ("model_name", "loader", "expected_iters", "expected_paths"),
    [
        ("granite4:tiny-h", _FakeLoader(architecture="granite"), 2, 8),
        (
            "qwen3.5:9b",
            _FakeLoader(architecture="qwen2", head_count=32, head_count_kv=8),
            1,
            8,
        ),
    ],
)
def test_profile_defaults_apply_consistently_by_model(
    monkeypatch, model_name, loader, expected_iters, expected_paths
):
    _patch_loaders(monkeypatch, {model_name: loader})

    config = QSGConfig()
    # 0/None means "use profile defaults" in adapter normalization.
    config.grover_iterations = 0
    config.coconut_paths = 0
    config.acceptance_threshold = None
    adapter = OllamaQSGAdapter(model_name, config=config)

    assert adapter.config.grover_iterations == expected_iters
    assert adapter.config.coconut_paths == expected_paths
    assert adapter.config.acceptance_threshold == pytest.approx(
        adapter.profile.speculative_acceptance_threshold
    )
    assert loader.extract_calls[0]["strategy"] == "auto"
    assert loader.extract_calls[0]["profile"] is adapter.profile


def test_profile_defaults_respect_explicit_coconut_disable(monkeypatch):
    model_name = "qwen3.5:9b"
    loader = _FakeLoader(architecture="qwen2", head_count=32, head_count_kv=8)
    _patch_loaders(monkeypatch, {model_name: loader})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_native_qsg", False)

    config = QSGConfig()
    config.use_coconut = False
    config.use_coconut_reasoning = False
    adapter = OllamaQSGAdapter(model_name, config=config)

    assert adapter.config.use_coconut is False


@pytest.mark.parametrize(
    ("model_name", "loader"),
    [
        ("granite4:tiny-h", _FakeLoader(architecture="granite")),
        (
            "qwen3.5:9b",
            _FakeLoader(architecture="qwen2", head_count=32, head_count_kv=8),
        ),
    ],
)
def test_speculative_decoder_uses_profile_specific_defaults(
    monkeypatch, model_name, loader
):
    _patch_loaders(monkeypatch, {model_name: loader})

    captured = {}

    class _SpeculativeConfig:
        def __init__(
            self,
            num_candidates,
            max_draft_length,
            acceptance_threshold,
            use_coconut_drafts,
            fallback_to_top_k=True,
            fallback_to_standard_generation=True,
            strict_native_only=False,
        ):
            self.num_candidates = num_candidates
            self.max_draft_length = max_draft_length
            self.acceptance_threshold = acceptance_threshold
            self.use_coconut_drafts = use_coconut_drafts
            self.fallback_to_top_k = fallback_to_top_k
            self.fallback_to_standard_generation = fallback_to_standard_generation
            self.strict_native_only = strict_native_only

    class _CPUSpeculativeDecoder:
        def __init__(self, native_engine, coconut_bridge, config):
            captured["config"] = config

    fake_module = SimpleNamespace(
        CPUSpeculativeDecoder=_CPUSpeculativeDecoder,
        SpeculativeConfig=_SpeculativeConfig,
    )
    monkeypatch.setitem(sys.modules, "core.native.cpu_speculative_decode", fake_module)

    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())

    assert adapter.profile.speculative_enabled is False
    assert adapter.speculative_decoder is None
    assert "config" not in captured


def test_speculative_decoder_initializes_when_profile_explicitly_enables_it(
    monkeypatch,
):
    model_name = "granite4:tiny-h"
    loader = _FakeLoader(architecture="granite")
    _patch_loaders(monkeypatch, {model_name: loader})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "cpu_speculative_decode", True)

    captured = {}

    class _SpeculativeConfig:
        def __init__(
            self,
            num_candidates,
            max_draft_length,
            acceptance_threshold,
            use_coconut_drafts,
            fallback_to_top_k=True,
            fallback_to_standard_generation=True,
            strict_native_only=False,
        ):
            self.num_candidates = num_candidates
            self.max_draft_length = max_draft_length
            self.acceptance_threshold = acceptance_threshold
            self.use_coconut_drafts = use_coconut_drafts
            self.fallback_to_top_k = fallback_to_top_k
            self.fallback_to_standard_generation = fallback_to_standard_generation
            self.strict_native_only = strict_native_only

    class _CPUSpeculativeDecoder:
        def __init__(self, native_engine, coconut_bridge, config):
            captured["config"] = config

    fake_module = SimpleNamespace(
        CPUSpeculativeDecoder=_CPUSpeculativeDecoder,
        SpeculativeConfig=_SpeculativeConfig,
    )
    monkeypatch.setitem(sys.modules, "core.native.cpu_speculative_decode", fake_module)

    config = QSGConfig()
    adapter = OllamaQSGAdapter(model_name, config=config)
    profile_data = dict(adapter.profile.__dict__)
    profile_data["speculative_enabled"] = True
    adapter.profile = SimpleNamespace(**profile_data)
    adapter._init_extra_components()

    spec_config = captured["config"]
    assert spec_config.num_candidates == adapter.profile.spec_num_candidates
    assert spec_config.max_draft_length == adapter.profile.spec_max_draft_length
    assert spec_config.acceptance_threshold == pytest.approx(
        adapter.profile.spec_acceptance_threshold
    )
    assert spec_config.use_coconut_drafts is True
    assert spec_config.fallback_to_top_k is False
    assert spec_config.fallback_to_standard_generation is False
    assert spec_config.strict_native_only is True


def test_coconut_processor_respects_disabled_profile_mode(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    _attach_coconut_weights(adapter)

    adapter.config.use_coconut = True
    adapter.config.use_grover = False
    adapter.profile = SimpleNamespace(coconut_mode="disabled")

    processors = adapter._create_logits_processor()
    assert len(processors) == 0


def test_coconut_logits_proxy_uses_profile_scaled_alpha(monkeypatch):
    loaders = {
        "granite4:tiny-h": _FakeLoader(architecture="granite", seed=7),
        "qwen3.5:9b": _FakeLoader(architecture="qwen2", seed=7),
    }
    _patch_loaders(monkeypatch, loaders)

    adapters = {}
    for model_name in loaders:
        config = QSGConfig()
        config.use_coconut = True
        config.use_grover = False
        config.use_self_consistency = True
        adapter = OllamaQSGAdapter(model_name, config=config)
        _attach_coconut_weights(adapter)
        adapters[model_name] = adapter

    original = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    deltas = {}

    for model_name, adapter in adapters.items():
        processors = adapter._create_logits_processor()
        coconut_proc = processors[0]
        updated = coconut_proc([123], original.copy())
        deltas[model_name] = float(np.mean(np.abs(updated - original)))

    assert (
        adapters["granite4:tiny-h"].profile.coconut_alpha
        > adapters["qwen3.5:9b"].profile.coconut_alpha
    )
    assert deltas["granite4:tiny-h"] > deltas["qwen3.5:9b"]


@pytest.mark.parametrize(
    ("model_name", "loader"),
    [
        ("granite4:tiny-h", _FakeLoader(architecture="granite")),
        (
            "qwen3.5:9b",
            _FakeLoader(architecture="qwen2", head_count=32, head_count_kv=8),
        ),
    ],
)
def test_grover_processor_uses_profile_parameters(monkeypatch, model_name, loader):
    _patch_loaders(monkeypatch, {model_name: loader})
    config = QSGConfig()
    config.grover_iterations = 0
    config.coconut_paths = 0
    config.acceptance_threshold = None
    adapter = OllamaQSGAdapter(model_name, config=config)

    adapter.config.use_coconut = False
    adapter.config.use_grover = True
    adapter.grover_amplifier = _CaptureGroverAmplifier()

    processors = adapter._create_logits_processor()
    assert len(processors) == 1

    scores = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    grover_proc = processors[0]
    _ = grover_proc([1, 2, 3], scores.copy())

    call = adapter.grover_amplifier.calls[0]
    assert call["iterations"] == adapter.profile.grover_iterations
    assert call["model_profile"] is adapter.profile
    assert call["top_k_oracle"] == adapter.profile.grover_top_k
    assert call["damping"] == pytest.approx(adapter.profile.grover_damping)


def test_coconut_processor_sanitizes_non_finite_scores(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    _attach_coconut_weights(adapter)
    adapter.config.use_coconut = True
    adapter.config.use_grover = False
    adapter.steering_engine = None

    processors = adapter._create_logits_processor()
    coconut_proc = processors[0]
    scores = np.linspace(-1.0, 1.0, 256).astype(np.float32)
    scores[:3] = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)

    updated = coconut_proc([1, 2, 3], scores.copy())
    assert updated.shape == scores.shape
    assert np.all(np.isfinite(updated))


def test_grover_processor_sanitizes_non_finite_amplifier_output(monkeypatch):
    class _NonFiniteGroverAmplifier:
        def amplify_with_resonance(
            self,
            logits_reshaped,
            tokens,  # noqa: ARG002
            context_text,  # noqa: ARG002
            iterations,  # noqa: ARG002
            model_profile,  # noqa: ARG002
            top_k_oracle,  # noqa: ARG002
            damping,  # noqa: ARG002
        ):
            bad = np.full_like(logits_reshaped, np.nan, dtype=np.float32)
            bad[0, 0] = np.inf
            bad[0, 1] = -np.inf
            return bad

    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    adapter.config.use_coconut = False
    adapter.config.use_grover = True
    adapter.grover_amplifier = _NonFiniteGroverAmplifier()
    adapter.steering_engine = None

    processors = adapter._create_logits_processor()
    grover_proc = processors[0]
    scores = np.linspace(-1.0, 1.0, 256).astype(np.float32)

    updated = grover_proc([4, 5, 6], scores.copy())
    assert updated.shape == scores.shape
    assert np.all(np.isfinite(updated))


def test_grover_processor_threads_latent_resonance_context(monkeypatch):
    class _CaptureGroverAmplifierV2:
        def __init__(self):
            self.calls = []

        def amplify_with_resonance(
            self,
            logits_reshaped,
            tokens,  # noqa: ARG002
            context_text,  # noqa: ARG002
            iterations,  # noqa: ARG002
            model_profile,  # noqa: ARG002
            top_k_oracle,  # noqa: ARG002
            damping,  # noqa: ARG002
            token_embeddings=None,
            latent_prior=None,
            repo_delta=None,
            invariant_terms=None,
            telemetry_sink=None,
        ):
            self.calls.append(
                {
                    "token_embeddings_shape": np.asarray(token_embeddings).shape,
                    "latent_prior": list(latent_prior or []),
                    "repo_delta": dict(repo_delta or {}),
                    "invariant_terms": list(invariant_terms or []),
                }
            )
            if telemetry_sink is not None:
                telemetry_sink["grover_oracle_source_count"] = 3.0
            probs = np.exp(
                logits_reshaped - np.max(logits_reshaped, axis=1, keepdims=True)
            )
            return probs / np.sum(probs, axis=1, keepdims=True)

    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    adapter.config.use_coconut = False
    adapter.config.use_grover = True
    adapter.grover_amplifier = _CaptureGroverAmplifierV2()
    adapter._activate_resonance_context(
        "prompt",
        {
            "latent_prior": [0.2, 0.3, 0.4],
            "delta_watermark": {"changed_paths": ["core/engine.py"]},
            "invariant_terms": ["engine"],
        },
    )

    processors = adapter._create_logits_processor()
    grover_proc = processors[0]
    scores = np.linspace(-1.0, 1.0, 256).astype(np.float32)

    _ = grover_proc([1, 2, 3], scores.copy())
    call = adapter.grover_amplifier.calls[0]
    assert call["token_embeddings_shape"][0] == 256
    assert call["latent_prior"] == [0.2, 0.3, 0.4]
    assert call["repo_delta"]["changed_paths"] == ["core/engine.py"]
    assert call["invariant_terms"] == ["engine"]
    assert (
        adapter.get_runtime_status()["grover_resonance"]["grover_oracle_source_count"]
        == 3.0
    )


def test_adapter_strict_mode_fails_when_native_coconut_bridge_unavailable(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_native_qsg", True)
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_coconut_bridge", True)

    def _raise_bridge(*_args, **_kwargs):
        raise RuntimeError("native bridge unavailable")

    monkeypatch.setattr("core.native.coconut_bridge.CoconutNativeBridge", _raise_bridge)

    with pytest.raises(RuntimeError, match="Unified native QSG"):
        OllamaQSGAdapter(model_name, config=QSGConfig())


def test_task_hint_does_not_misclassify_capital_as_coding():
    resolved: dict[str, str] = {}

    task_hint = OllamaQSGAdapter._infer_task_hint("The capital of France is", resolved)
    qwen_profile = OllamaQSGAdapter._resolve_qwen_profile_name(
        "The capital of France is",
        resolved,
    )
    granite_profile = OllamaQSGAdapter._resolve_granite_profile_name(
        "The capital of France is",
        resolved,
    )

    assert task_hint == "general"
    assert qwen_profile == "instruct_general"
    assert granite_profile == "coding_deterministic"


def test_coconut_logits_processor_strict_mode_raises_typed_error(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_native_qsg", False)
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_logits_processor", True)

    config = QSGConfig()
    config.use_coconut = True
    config.use_grover = False
    adapter = OllamaQSGAdapter(model_name, config=config)
    _attach_coconut_weights(adapter)

    def _broken_expand(*_args, **_kwargs):
        raise ValueError("bad tensor shape")

    adapter.coconut_bridge.expand_paths = _broken_expand

    processors = adapter._create_logits_processor()
    coconut_proc = processors[0]
    scores = np.linspace(-1.0, 1.0, 256).astype(np.float32)

    with pytest.raises(
        RuntimeError, match="Unified native QSG rejected COCONUT logits"
    ):
        coconut_proc([1, 2, 3], scores.copy())


def test_generate_qsg_raises_when_speculative_rejects(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_native_qsg", False)
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_speculative_decode", False)

    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())

    class SpeculativeFallbackDisabledError(Exception):
        pass

    class _RejectingSpecDecoder:
        def generate_speculative(self, *args, **kwargs):
            raise SpeculativeFallbackDisabledError("reject all candidates")

    adapter.speculative_decoder = _RejectingSpecDecoder()
    adapter.hybrid_ssm = None

    with pytest.raises(RuntimeError, match="disallows speculative decode fallback"):
        adapter._generate_qsg("hello", {"num_predict": 1, "temperature": 0.2})


def test_generate_qsg_does_not_silently_disable_speculative_after_rejections(
    monkeypatch,
):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_native_qsg", False)
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_speculative_decode", False)
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    adapter.hybrid_ssm = None
    adapter._spec_disable_after_rejections = 2

    class SpeculativeFallbackDisabledError(Exception):
        pass

    class _RejectingSpecDecoder:
        def __init__(self):
            self.calls = 0

        def generate_speculative(self, *args, **kwargs):
            self.calls += 1
            raise SpeculativeFallbackDisabledError("reject all candidates")

    rejecting = _RejectingSpecDecoder()
    adapter.speculative_decoder = rejecting

    with pytest.raises(RuntimeError, match="disallows speculative decode fallback"):
        adapter._generate_qsg("hello", {"num_predict": 1, "temperature": 0.2})
    assert adapter.speculative_decoder is rejecting

    with pytest.raises(RuntimeError, match="disallows speculative decode fallback"):
        adapter._generate_qsg("hello", {"num_predict": 1, "temperature": 0.2})
    assert adapter.speculative_decoder is rejecting
    assert rejecting.calls == 2


def test_generate_qsg_strict_mode_raises_when_speculative_rejects(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_speculative_decode", True)
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())

    class SpeculativeFallbackDisabledError(Exception):
        pass

    class _RejectingSpecDecoder:
        def generate_speculative(self, *args, **kwargs):
            raise SpeculativeFallbackDisabledError("reject all candidates")

    adapter.speculative_decoder = _RejectingSpecDecoder()
    adapter.hybrid_ssm = None

    with pytest.raises(RuntimeError, match="disallows speculative decode fallback"):
        adapter._generate_qsg("hello", {"num_predict": 1, "temperature": 0.2})


def test_stream_generate_qsg_strict_mode_raises_when_speculative_rejects(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    monkeypatch.setitem(PERFORMANCE_CONFIG, "strict_speculative_decode", True)
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())

    class SpeculativeFallbackDisabledError(Exception):
        pass

    class _RejectingSpecDecoder:
        def generate_speculative(self, *args, **kwargs):
            raise SpeculativeFallbackDisabledError("reject all candidates")

    adapter.speculative_decoder = _RejectingSpecDecoder()
    adapter.hybrid_ssm = None

    with pytest.raises(RuntimeError, match="disallows speculative streaming fallback"):
        list(
            adapter._stream_generate_qsg(
                "hello", {"num_predict": 1, "temperature": 0.2}
            )
        )


def test_generate_qsg_requires_token_level_engine(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    adapter.native_engine.supports_token_api = False

    with pytest.raises(RuntimeError, match="token-level native generation support"):
        adapter._generate_qsg("hello", {"num_predict": 1})


def test_stream_generate_requires_token_level_engine(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())
    adapter.native_engine.supports_token_api = False

    with pytest.raises(RuntimeError, match="token-level native streaming support"):
        list(adapter._stream_generate_qsg("hello", {"num_predict": 1}))


def test_repetition_penalty_handles_numpy_input_ids():
    from core.qsg.ollama_adapter import RepetitionPenaltyProcessor

    proc = RepetitionPenaltyProcessor(penalty=1.1, window=16)
    scores = np.ones((32,), dtype=np.float32)
    ids = np.asarray([1, 1, 2, 3], dtype=np.int32)

    updated = proc.apply(ids, scores.copy())
    assert updated.shape == scores.shape
    # repeated token should be penalized
    assert updated[1] < scores[1]


def test_adapter_routes_generate_through_continuous_engine_when_enabled(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})

    class _FakeContinuousEngine:
        last_instance = None

        def __init__(self, *, config, stream_producer):
            del config, stream_producer
            self._chunks = {}
            self.submitted = []
            _FakeContinuousEngine.last_instance = self

        def submit(self, request):
            self.submitted.append(request)
            request_id = f"req-{len(self.submitted)}"
            self._chunks[request_id] = [
                QSGChunk(request_id=request_id, text="hello ", done=False),
                QSGChunk(request_id=request_id, text="world", done=False),
                QSGChunk(request_id=request_id, text="", done=True),
            ]
            return request_id

        def poll(self, request_id):
            queue = self._chunks.get(request_id, [])
            if not queue:
                return None
            return queue.pop(0)

        def cancel(self, request_id):
            del request_id

        def run_forever(self):
            return None

        def shutdown(self, graceful_timeout_s: float = 1.0):
            del graceful_timeout_s
            return None

        def metrics_snapshot(self):
            return {}

    monkeypatch.setattr(
        _FakeNativeEngine,
        "build_parallel_generation_engine",
        lambda self, **kwargs: _FakeContinuousEngine(**kwargs),
        raising=False,
    )

    config = QSGConfig(
        continuous_batching_enabled=True, semantic_resonance_timeout_ms=1
    )
    adapter = OllamaQSGAdapter(model_name, config=config)

    text = adapter.generate("prompt", options={"num_predict": 4})

    assert text == "hello world"
    assert _FakeContinuousEngine.last_instance is not None
    assert len(_FakeContinuousEngine.last_instance.submitted) == 1
    submitted = _FakeContinuousEngine.last_instance.submitted[0]
    assert submitted.prompt == "prompt"
    assert submitted.prompt_tokens == [1, 2, 3]
    assert submitted.max_new_tokens == 4
    assert submitted.sampling["num_predict"] == 4


def test_continuous_stream_producer_reuses_pretokenized_prompt(monkeypatch):
    model_name = "granite4:tiny-h"
    _patch_loaders(monkeypatch, {model_name: _FakeLoader(architecture="granite")})
    adapter = OllamaQSGAdapter(model_name, config=QSGConfig())

    request = adapter._build_continuous_request("prompt", {"num_predict": 2})
    tokenize_calls = adapter.native_engine.tokenize_calls

    output = list(adapter._continuous_stream_producer(request))

    assert output == ["ctx"]
    assert adapter.native_engine.tokenize_calls == tokenize_calls
