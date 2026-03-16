from types import SimpleNamespace

import pytest

from config.settings import (
    CONTINUOUS_BATCHING_CONFIG,
    GENERATION_PARAMS,
    GRANITE4_SAMPLING_PROFILES,
)
from core.ollama_client import DeterministicOllama


class _FakeLoader:
    def __init__(self, template: str = "granite"):
        self.profile = SimpleNamespace(chat_template=template)
        self.last_prompt = None
        self.last_params = None
        self.native_engine = SimpleNamespace(
            embed=lambda text: [float(len(text))],
            get_runtime_status=lambda: {
                "backend": "native_qsg",
                "digest": "sha256:test",
                "decode_threads": 2,
                "batch_threads": 2,
                "openmp_enabled": True,
                "avx2_enabled": True,
            },
        )

    def generate(self, prompt, params):
        self.last_prompt = prompt
        self.last_params = params
        return "loader-response"

    def stream_generate(self, prompt, params):
        self.last_prompt = prompt
        self.last_params = params
        yield "x"

    def batch_embeddings(self, prompts):
        return [[float(len(prompt))] for prompt in prompts]

    def runtime_status(self):
        return {"backend": "native_qsg", "digest": "sha256:test", "model": "granite4:tiny-h"}


def setup_function():
    DeterministicOllama._loader_cache.clear()


def test_ollama_client_init_uses_qsg_loader(monkeypatch):
    fake_loader = _FakeLoader()
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )
    client = DeterministicOllama("granite4:tiny-h")
    assert client.model_name == "granite4:tiny-h"
    assert client.loader is fake_loader


def test_ollama_client_rejects_unsupported_model_name() -> None:
    with pytest.raises(RuntimeError, match="Unsupported production model"):
        DeterministicOllama("deepseek-coder:33b")


def test_ollama_client_generate_raises_when_loader_missing(monkeypatch):
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: None,
    )
    client = DeterministicOllama("granite4:tiny-h")
    with pytest.raises(RuntimeError, match="HTTP Ollama API fallback is disabled"):
        client.generate("hello")


def test_ollama_client_generate_loader(monkeypatch):
    fake_loader = _FakeLoader()
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )
    client = DeterministicOllama("granite4:tiny-h")
    res = client.generate("hello")
    assert res == "loader-response"
    assert fake_loader.last_prompt == "hello"
    expected = GENERATION_PARAMS.copy()
    profile_name = str(expected.get("granite4_sampling_profile", "coding_balanced"))
    expected.update(GRANITE4_SAMPLING_PROFILES[profile_name])
    expected["granite4_sampling_profile"] = profile_name
    assert fake_loader.last_params == expected


def test_stream_generate_raises_when_loader_missing(monkeypatch):
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: None,
    )
    client = DeterministicOllama("granite4:tiny-h")
    with pytest.raises(RuntimeError, match="HTTP Ollama API fallback is disabled"):
        list(client.stream_generate("hello"))


def test_chat_uses_loader_and_custom_params(monkeypatch):
    fake_loader = _FakeLoader(template="granite")
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )
    client = DeterministicOllama("granite4:tiny-h")
    result = client.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=17,
        temperature=0.25,
    )
    assert result == "loader-response"
    assert "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>" in (
        fake_loader.last_prompt
    )
    assert fake_loader.last_prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")
    assert fake_loader.last_params["num_predict"] == 17
    assert fake_loader.last_params["temperature"] == 0.25


def test_chat_raises_when_loader_missing(monkeypatch):
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: None,
    )
    client = DeterministicOllama("granite4:tiny-h")
    with pytest.raises(RuntimeError, match="HTTP Ollama API fallback is disabled"):
        client.chat([{"role": "user", "content": "Hello"}])


def test_stream_chat_raises_when_loader_missing(monkeypatch):
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: None,
    )
    client = DeterministicOllama("granite4:tiny-h")
    with pytest.raises(RuntimeError, match="HTTP Ollama API fallback is disabled"):
        list(client.stream_chat([{"role": "user", "content": "Hello"}]))


def test_batch_embeddings_raises_when_loader_missing(monkeypatch):
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: None,
    )
    client = DeterministicOllama("granite4:tiny-h")
    with pytest.raises(RuntimeError, match="HTTP Ollama API fallback is disabled"):
        client.batch_embeddings(["hello"])


def test_batch_embeddings_uses_loader(monkeypatch):
    fake_loader = _FakeLoader()
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )
    client = DeterministicOllama("granite4:tiny-h")
    result = client.batch_embeddings(["hi", "hello"])
    assert result == [[2.0], [5.0]]


def test_runtime_status_delegates_to_loader(monkeypatch):
    fake_loader = _FakeLoader()
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )
    client = DeterministicOllama("granite4:tiny-h")
    status = client.runtime_status()
    assert status["backend"] == "native_qsg"
    assert status["digest"] == "sha256:test"


def test_get_loader_applies_continuous_batching_config(monkeypatch):
    captured = {}

    class _FakeAdapter:
        def __init__(self, model_name, config, parent_ollama):
            captured["model_name"] = model_name
            captured["config"] = config
            captured["parent"] = parent_ollama

    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "enabled", True)
    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "max_active_requests", 4)
    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "max_pending_requests", 17)
    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "scheduler_policy", "priority")
    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "batch_wait_timeout_ms", 7)
    monkeypatch.setitem(CONTINUOUS_BATCHING_CONFIG, "semantic_poll_timeout_ms", 9)
    monkeypatch.setattr("core.qsg.ollama_adapter.OllamaQSGAdapter", _FakeAdapter)

    client = DeterministicOllama("granite4:tiny-h")

    cfg = captured["config"]
    assert captured["model_name"] == "granite4:tiny-h"
    assert captured["parent"] is client
    assert cfg.continuous_batching_enabled is True
    assert cfg.max_active_requests == 4
    assert cfg.max_pending_requests == 17
    assert cfg.scheduler_policy == "priority"
    assert cfg.batch_wait_timeout_ms == 7
    assert cfg.semantic_resonance_timeout_ms == 9
