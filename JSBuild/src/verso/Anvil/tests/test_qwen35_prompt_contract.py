from types import SimpleNamespace

import pytest

from core.model.chat_templates import (
    CHAT_SUFFIXES,
    format_prompt_for_model,
    format_strict_native_prompt,
    get_strict_prompt_contract,
    postprocess_strict_native_response,
    resolve_chat_template_name,
)
from core.model.model_profile import _detect_chat_template


def test_qwen35_strict_prompt_contract_is_explicit_chatml_only() -> None:
    contract = get_strict_prompt_contract("qwen3.5:9b", strict=True)
    contract_4b = get_strict_prompt_contract("qwen3.5:4b", strict=True)

    assert contract.model_name == "qwen3.5:9b"
    assert contract.template_name == "chatml"
    assert contract.system_prompt is None
    assert contract.assistant_prefix is None
    assert contract.inject_system_prompt is False
    assert contract.disallowed_output_prefixes == (
        "<think>",
        "</think>",
        "Assistant:",
        "assistant:",
        "Assistant",
        "assistant",
    )
    assert contract_4b.model_name == "qwen3.5:4b"
    assert contract_4b.template_name == "chatml"
    assert contract_4b.inject_system_prompt is False


def test_qwen35_format_prompt_for_model_uses_bare_chatml_prompt() -> None:
    prompt = format_prompt_for_model("Hello", "qwen3.5:9b")
    prompt_4b = format_prompt_for_model("Hello", "qwen3.5:4b")

    assert prompt == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert prompt_4b == "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
    assert "You are a helpful assistant." not in prompt
    assert "<think>" not in prompt


def test_qwen35_format_strict_native_prompt_preserves_preformatted_chatml() -> None:
    prompt = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

    assert (
        format_strict_native_prompt(
            prompt,
            "chatml",
            model_name="qwen3.5:9b",
        )
        == prompt
    )


def test_qwen35_template_detection_prefers_strict_contract_over_metadata_noise() -> None:
    loader = SimpleNamespace(get_vocab_tokens=lambda: ["<|start_of_role|>user"])
    metadata = {
        "tokenizer.chat_template": "<|start_of_role|>user<|end_of_role|>{{ prompt }}",
        "general.architecture": "granitehybrid",
    }

    assert _detect_chat_template(loader, metadata, "qwen3.5:9b") == "chatml"
    assert _detect_chat_template(loader, metadata, "qwen3.5:4b") == "chatml"


def test_qwen35_strict_prompt_helpers_fail_closed_for_unsupported_models() -> None:
    with pytest.raises(
        RuntimeError,
        match="support only qwen3.5:4b, qwen3.5:9b and granite4:tiny-h",
    ):
        get_strict_prompt_contract("mistral:7b", strict=True)

    with pytest.raises(
        RuntimeError,
        match="supports only qwen3.5:4b, qwen3.5:9b and granite4:tiny-h",
    ):
        resolve_chat_template_name("mistral:7b", strict=True)

    with pytest.raises(
        RuntimeError,
        match="supports only qwen3.5:4b, qwen3.5:9b and granite4:tiny-h",
    ):
        resolve_chat_template_name("qwen3.5:4b-instruct", strict=True)


def test_qwen35_non_strict_contract_maps_variants_to_chatml() -> None:
    contract = get_strict_prompt_contract("qwen3.5:4b-instruct", strict=False)
    assert contract is not None
    assert contract.template_name == "chatml"
    assert contract.inject_system_prompt is False


def test_qwen35_postprocess_strips_leading_think_stub_but_keeps_plain_text() -> None:
    stripped = postprocess_strict_native_response(
        "<think>\n\n</think>\n\nDeterministic answer.",
        model_name="qwen3.5:9b",
    )
    untouched = postprocess_strict_native_response(
        "Already clean.",
        model_name="qwen3.5:9b",
    )

    assert stripped == "Deterministic answer."
    assert untouched == "Already clean."
    assert (
        postprocess_strict_native_response(
            "<think>\n\n</think>\n\nDeterministic answer.",
            model_name="qwen3.5:4b",
        )
        == "Deterministic answer."
    )
    assert CHAT_SUFFIXES["chatml"] == "<|im_start|>assistant\n"


def test_qwen35_variant_postprocess_does_not_require_exact_allowlist_name() -> None:
    stripped = postprocess_strict_native_response(
        "<think>\n\n</think>\n\nDeterministic answer.",
        model_name="qwen3.5:4b-instruct",
    )
    assert stripped == "Deterministic answer."


def test_qwen35_postprocess_removes_orphan_and_inline_think_tags() -> None:
    stripped = postprocess_strict_native_response(
        "Answer prefix </think> Final answer.",
        model_name="qwen3.5:9b",
    )
    block_stripped = postprocess_strict_native_response(
        "Visible <think>hidden draft</think> answer.",
        model_name="qwen3.5:9b",
    )

    assert stripped == "Answer prefix  Final answer."
    assert block_stripped == "Visible  answer."
    assert (
        postprocess_strict_native_response(
            "Answer prefix </think> Final answer.",
            model_name="qwen3.5:4b",
        )
        == "Answer prefix  Final answer."
    )


def test_qwen35_postprocess_strips_chatml_assistant_role_leakage() -> None:
    stripped = postprocess_strict_native_response(
        "Assistant: \n\nThe sentence is about AVX2.\n\nAssistant:",
        model_name="qwen3.5:9b",
    )
    bare = postprocess_strict_native_response(
        "Assistant\n\nThe sentence is about AVX2.\n\nAssistant",
        model_name="qwen3.5:9b",
    )

    assert stripped == "The sentence is about AVX2."
    assert bare == "The sentence is about AVX2."
    assert (
        postprocess_strict_native_response(
            "Assistant: \n\nThe sentence is about AVX2.\n\nAssistant:",
            model_name="qwen3.5:4b",
        )
        == "The sentence is about AVX2."
    )
