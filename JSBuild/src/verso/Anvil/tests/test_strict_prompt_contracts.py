from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from core.model.chat_templates import (
    format_strict_native_prompt,
    get_strict_prompt_contract,
    resolve_chat_template_name,
)
from core.model.model_contract import ModelContract, model_contract_snapshot
from core.native.native_qsg_engine import NativeQSGEngine


def test_qwen_strict_prompt_contract_is_explicit() -> None:
    contract = get_strict_prompt_contract("qwen3.5:9b", strict=True)
    contract_4b = get_strict_prompt_contract("qwen3.5:4b", strict=True)

    assert contract is not None
    assert contract.template_name == "chatml"
    assert contract.system_prompt is None
    assert contract.assistant_prefix is None
    assert "<think>" in contract.disallowed_output_prefixes
    assert contract_4b.template_name == "chatml"
    assert contract_4b.system_prompt is None
    assert contract_4b.assistant_prefix is None
    assert "<think>" in contract_4b.disallowed_output_prefixes


def test_granite_strict_prompt_contract_injects_system_prompt() -> None:
    prompt = format_strict_native_prompt(
        "Explain AVX2 briefly.",
        "granite",
        model_name="granite4:tiny-h",
    )

    assert prompt.startswith("<|start_of_role|>system<|end_of_role|>")
    assert "professional, accurate, and safe." in prompt
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_resolve_chat_template_name_fails_closed_for_unsupported_model() -> None:
    with pytest.raises(RuntimeError):
        resolve_chat_template_name("unsupported:model", strict=True)


def test_model_contract_snapshot_exposes_template_name() -> None:
    snapshot = model_contract_snapshot(
        ModelContract(
            canonical_name="qwen3.5:9b",
            template_name="chatml",
            strict_native_supported=True,
            manifest_path=Path("/tmp/manifest"),
            blob_path=Path("/tmp/blob"),
            manifest_sha256="sha256:manifest",
            expected_manifest_digest="sha256:manifest",
            manifest_digest="sha256:model",
            expected_model_digest="sha256:model",
            expected_digest="sha256:model",
            blob_size=123,
            digest_validated=True,
            local_sha256=None,
            quant_variant="manifest-pinned",
        )
    )

    assert snapshot["template_name"] == "chatml"
    assert snapshot["strict_native_supported"] is True


def test_strict_engine_generation_only_helpers_raise() -> None:
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(model_name="qwen3.5:9b")

    with pytest.raises(RuntimeError):
        engine.embed("hello")

    with pytest.raises(RuntimeError):
        engine.get_hidden_states([1, 2, 3])
