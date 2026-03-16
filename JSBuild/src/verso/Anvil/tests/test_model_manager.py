from __future__ import annotations

from types import SimpleNamespace

from core.model_manager import ModelManager


def test_model_manager_init_normalizes_supported_models() -> None:
    mm = ModelManager(["registry.ollama.ai/library/granite4:tiny-h", "qwen3.5:9b"])
    assert mm.models == ["granite4:tiny-h", "qwen3.5:9b"]


def test_model_manager_ensure_models_present_pulls_missing(monkeypatch) -> None:
    mm = ModelManager(["granite4:tiny-h"])
    monkeypatch.setattr(
        "core.model_manager.resolve_model_contract",
        lambda model: (_ for _ in ()).throw(RuntimeError(f"missing {model}")),
    )
    pulled: list[str] = []
    monkeypatch.setattr(mm, "pull_model", lambda model: pulled.append(model))

    missing = mm.ensure_models_present()

    assert missing == ["granite4:tiny-h"]
    assert pulled == ["granite4:tiny-h"]


def test_model_manager_warm_up_uses_native_client(monkeypatch) -> None:
    mm = ModelManager(["granite4:tiny-h"])
    fake_client = SimpleNamespace(generate=lambda prompt: f"warm:{prompt}")
    monkeypatch.setattr("core.model_manager.DeterministicOllama", lambda model: fake_client)

    result = mm.warm_up("granite4:tiny-h")

    assert result == "warm:hello"
