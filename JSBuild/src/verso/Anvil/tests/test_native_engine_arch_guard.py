import pytest

from core.native import engine


def test_preflight_arch_guard_blocks_known_unsupported_architecture(monkeypatch):
    monkeypatch.delenv("ANVIL_ALLOW_UNSAFE_ARCH_LOAD", raising=False)
    monkeypatch.setattr(
        engine, "_try_read_model_architecture", lambda _path: "granitehybrid"
    )

    message = engine._build_preflight_architecture_guard_message("/tmp/model.gguf")

    assert message is not None
    assert "Preflight blocked model load" in message
    assert "granitehybrid" in message


def test_preflight_arch_guard_can_be_overridden(monkeypatch):
    monkeypatch.setenv("ANVIL_ALLOW_UNSAFE_ARCH_LOAD", "1")
    monkeypatch.setattr(engine, "_try_read_model_architecture", lambda _path: "qwen35")

    message = engine._build_preflight_architecture_guard_message("/tmp/model.gguf")
    assert message is None


def test_backend_setting_allows_native_qsg_and_auto(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_ENGINE_BACKEND", raising=False)
    engine._validate_native_backend_setting()

    monkeypatch.setenv("ANVIL_NATIVE_ENGINE_BACKEND", "auto")
    engine._validate_native_backend_setting()

    monkeypatch.setenv("ANVIL_NATIVE_ENGINE_BACKEND", "native_qsg")
    engine._validate_native_backend_setting()


def test_backend_setting_rejects_non_native_values(monkeypatch):
    monkeypatch.setenv("ANVIL_NATIVE_ENGINE_BACKEND", "runner")

    with pytest.raises(RuntimeError, match="Supported native backends"):
        engine._validate_native_backend_setting()

    monkeypatch.setenv("ANVIL_NATIVE_ENGINE_BACKEND", "llama_cpp")
    with pytest.raises(RuntimeError, match="Supported native backends"):
        engine._validate_native_backend_setting()


def test_llama_cpp_engine_disabled_in_strict_native_mode(monkeypatch):
    monkeypatch.setitem(engine.PERFORMANCE_CONFIG, "strict_native_qsg", True)
    monkeypatch.delenv("ANVIL_ALLOW_LLAMA_CPP_ENGINE", raising=False)

    with pytest.raises(RuntimeError, match="disabled in strict native mode"):
        engine.LlamaCppInferenceEngine("/tmp/model.gguf")
