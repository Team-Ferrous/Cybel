from types import SimpleNamespace

import pytest

from config.settings import GENERATION_PARAMS, QWEN35_SAMPLING_PROFILES
from core.model.chat_templates import (
    GRANITE_SYSTEM_PROMPT,
    CHAT_SUFFIXES,
    CHAT_TEMPLATES,
    format_chat_messages,
    format_strict_native_prompt,
    postprocess_strict_native_response,
    resolve_chat_template_name,
)
from core.native.native_qsg_engine import NativeQSGEngine
from core.ollama_client import DeterministicOllama
from core.qsg.ollama_adapter import OllamaQSGAdapter


class _TemplateLoader:
    def __init__(self, template=None):
        self.last_prompt = None
        if template is None:
            self.profile = None
        else:
            self.profile = SimpleNamespace(chat_template=template)

    def generate(self, prompt, params):
        self.last_prompt = prompt
        return "ok"

    def stream_generate(self, prompt, params):
        self.last_prompt = prompt
        yield "ok"


def setup_function():
    DeterministicOllama._loader_cache.clear()


def test_chat_templates_registry_formats_expected_tokens():
    granite = CHAT_TEMPLATES["granite"]("user", "hello")
    granite_role = CHAT_TEMPLATES["granite_role"]("user", "hello")
    chatml = CHAT_TEMPLATES["chatml"]("user", "hello")

    assert "<|start_of_role|>user<|end_of_role|>hello<|end_of_text|>" in granite
    assert granite_role == granite
    assert "<|im_start|>user\nhello<|im_end|>\n" == chatml
    assert CHAT_SUFFIXES["granite"] == "<|start_of_role|>assistant<|end_of_role|>"
    assert CHAT_SUFFIXES["granite_role"] == CHAT_SUFFIXES["granite"]
    assert CHAT_SUFFIXES["chatml"] == "<|im_start|>assistant\n"


def test_profile_template_takes_precedence_over_model_name(monkeypatch):
    fake_loader = _TemplateLoader(template="granite")
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )

    client = DeterministicOllama("qwen3.5:9b")
    result = client.chat([{"role": "user", "content": "Hello"}])

    assert result == "ok"
    assert fake_loader.last_prompt == format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        "chatml",
        system_prompt=None,
        inject_system_prompt=False,
    )


def test_model_name_fallback_when_profile_missing(monkeypatch):
    fake_loader = _TemplateLoader(template=None)
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )

    client = DeterministicOllama("qwen3.5:9b")
    result = client.chat([{"role": "user", "content": "Hello"}])

    assert result == "ok"
    assert "<|im_start|>user\nHello<|im_end|>\n" in fake_loader.last_prompt
    assert fake_loader.last_prompt.endswith(CHAT_SUFFIXES["chatml"])


def test_format_strict_native_prompt_wraps_granite_and_qwen_consistently():
    granite = format_strict_native_prompt("Hello", "granite", model_name="granite4:tiny-h")
    chatml = format_strict_native_prompt("Hello", "chatml", model_name="qwen3.5:9b")
    chatml_4b = format_strict_native_prompt("Hello", "chatml", model_name="qwen3.5:4b")

    assert granite.startswith("<|start_of_role|>system<|end_of_role|>")
    assert "professional, accurate, and safe." in granite
    assert granite.endswith(CHAT_SUFFIXES["granite"])
    assert chatml.startswith("<|im_start|>user\nHello<|im_end|>\n")
    assert chatml.endswith(CHAT_SUFFIXES["chatml"])
    assert chatml_4b.startswith("<|im_start|>user\nHello<|im_end|>\n")
    assert chatml_4b.endswith(CHAT_SUFFIXES["chatml"])


def test_postprocess_strict_native_response_strips_qwen_think_stub():
    text = "<think>\n\n</think>\n\nDeterministic answer."
    closer_only = "</think>\n\nDeterministic answer."

    assert (
        postprocess_strict_native_response(text, model_name="qwen3.5:9b")
        == "Deterministic answer."
    )
    assert (
        postprocess_strict_native_response(text, template_name="chatml")
        == "Deterministic answer."
    )
    assert (
        postprocess_strict_native_response(closer_only, model_name="qwen3.5:9b")
        == "Deterministic answer."
    )
    assert (
        postprocess_strict_native_response(text, model_name="qwen3.5:4b")
        == "Deterministic answer."
    )


def test_postprocess_strict_native_response_strips_unclosed_qwen_think_stub():
    text = "<think>\ninternal reasoning only"
    answer_after_blank = "<think>\ninternal reasoning\n\nDeterministic answer."

    assert postprocess_strict_native_response(text, model_name="qwen3.5:9b") == ""
    assert (
        postprocess_strict_native_response(
            answer_after_blank,
            model_name="qwen3.5:9b",
        )
        == "Deterministic answer."
    )


def test_engine_and_adapter_use_same_completion_prompt_shape():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(
        chat_template="chatml",
        family="qwen",
        model_name="qwen3.5:9b",
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._strict_prompt_contract = SimpleNamespace(
        template_name="chatml",
        system_prompt=None,
        assistant_prefix=None,
    )

    adapter = OllamaQSGAdapter.__new__(OllamaQSGAdapter)
    adapter.profile = SimpleNamespace(
        chat_template="chatml",
        family="qwen",
        model_name="qwen3.5:9b",
    )
    adapter.model_name = "qwen3.5:9b"

    prompt = "Write a haiku."

    assert engine.format_prompt(prompt) == adapter._format_prompt_for_model(prompt)


def test_adapter_strict_mode_pins_qwen_profile_default_without_explicit_override():
    adapter = OllamaQSGAdapter.__new__(OllamaQSGAdapter)
    adapter.model_name = "qwen3.5:4b"
    adapter.strict_native_qsg = True

    resolved = adapter._apply_sampling_profile({}, prompt="Fix this bug and add tests.")
    expected_profile = str(
        GENERATION_PARAMS.get("qwen35_sampling_profile", "instruct_deterministic")
    )

    assert resolved["qwen35_sampling_profile"] == expected_profile
    assert (
        resolved["temperature"]
        == QWEN35_SAMPLING_PROFILES[expected_profile]["temperature"]
    )


def test_shared_helper_matches_qwen_chat_prompt(monkeypatch):
    fake_loader = _TemplateLoader(template="chatml")
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )

    client = DeterministicOllama("qwen3.5:9b")
    _ = client.chat([{"role": "user", "content": "Hello"}])

    assert fake_loader.last_prompt == format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        "chatml",
        system_prompt=None,
        inject_system_prompt=False,
    )


def test_invalid_qwen_profile_template_falls_back_to_model_resolution(monkeypatch):
    fake_loader = _TemplateLoader(template="unsupported-template")
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )

    client = DeterministicOllama("qwen3.5:9b")
    _ = client.chat([{"role": "user", "content": "Hello"}])

    assert fake_loader.last_prompt == format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        "chatml",
        system_prompt=None,
        inject_system_prompt=False,
    )


def test_shared_helper_matches_granite_chat_prompt(monkeypatch):
    fake_loader = _TemplateLoader(template="granite")
    monkeypatch.setattr(
        "core.ollama_client.DeterministicOllama._get_loader",
        lambda self: fake_loader,
    )

    client = DeterministicOllama("granite4:tiny-h")
    _ = client.chat([{"role": "user", "content": "Hello"}])

    assert fake_loader.last_prompt == format_chat_messages(
        [{"role": "user", "content": "Hello"}],
        "granite",
        system_prompt=GRANITE_SYSTEM_PROMPT,
        inject_system_prompt=True,
    )


def test_strict_chat_template_resolution_fails_closed_for_unknown_model():
    with pytest.raises(RuntimeError, match="Strict native prompt formatting supports only"):
        resolve_chat_template_name("mistral:7b", strict=True)
