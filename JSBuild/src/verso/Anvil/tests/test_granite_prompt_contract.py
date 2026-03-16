from types import SimpleNamespace

import pytest

from core.model.chat_templates import (
    CHAT_SUFFIXES,
    GRANITE_SYSTEM_PROMPT,
    format_prompt_for_model,
    format_strict_native_prompt,
    get_strict_prompt_contract,
)
from core.native.native_qsg_engine import NativeQSGEngine


def _golden_granite_prompt(user_text: str) -> str:
    return (
        f"<|start_of_role|>system<|end_of_role|>{GRANITE_SYSTEM_PROMPT}<|end_of_text|>\n"
        f"<|start_of_role|>user<|end_of_role|>{user_text}<|end_of_text|>\n"
        f"{CHAT_SUFFIXES['granite']}"
    )


def _build_granite_engine() -> NativeQSGEngine:
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(
        chat_template="granite",
        family="granite",
        model_name="granite4:tiny-h",
    )
    engine.contract = {"model": "granite4:tiny-h"}
    engine._strict_prompt_contract = get_strict_prompt_contract("granite4:tiny-h")
    return engine


def test_granite_strict_prompt_contract_is_explicit() -> None:
    contract = get_strict_prompt_contract("granite4:tiny-h")

    assert contract is not None
    assert contract.model_name == "granite4:tiny-h"
    assert contract.template_name == "granite"
    assert contract.system_prompt == GRANITE_SYSTEM_PROMPT
    assert contract.assistant_prefix is None
    assert contract.inject_system_prompt is True
    assert contract.disallowed_output_prefixes == ()


def test_granite_prompt_for_model_matches_golden_contract() -> None:
    prompt = format_prompt_for_model("Hello", "granite4:tiny-h")

    assert prompt == _golden_granite_prompt("Hello")


def test_granite_preformatted_prompt_is_not_wrapped_twice() -> None:
    preformatted = _golden_granite_prompt("Hello")

    prompt = format_strict_native_prompt(
        preformatted,
        "granite",
        model_name="granite4:tiny-h",
    )

    assert prompt == preformatted


def test_granite_engine_format_prompt_matches_shared_contract() -> None:
    engine = _build_granite_engine()

    prompt = engine.format_prompt("Hello")

    assert prompt == _golden_granite_prompt("Hello")


def test_granite_strict_runtime_disables_non_generation_helpers() -> None:
    engine = _build_granite_engine()

    with pytest.raises(RuntimeError, match="disables Python forward-pass helpers"):
        engine._ensure_forward_pass()

    with pytest.raises(RuntimeError, match="disables embedding helpers"):
        engine.embed("Hello")

    with pytest.raises(RuntimeError, match="disables hidden-state helpers"):
        engine.get_hidden_states([1, 2, 3])


def test_granite_generate_text_uses_formatted_prompt_and_returns_deterministic_output() -> None:
    engine = _build_granite_engine()
    seen: dict[str, object] = {}

    def tokenize(text: str) -> list[int]:
        seen["prompt"] = text
        return [11, 22]

    def generate(
        prompt_tokens: list[int],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
    ) -> list[int]:
        seen["generate_args"] = {
            "prompt_tokens": list(prompt_tokens),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
        }
        return [11, 22, 301, 302]

    def detokenize(tokens: list[int]) -> str:
        assert tokens == [301, 302]
        return "Granite strict-native reply."

    engine.tokenize = tokenize
    engine.generate = generate
    engine.detokenize = detokenize

    result = engine.generate_text(
        "Hello",
        max_new_tokens=2,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        min_p=0.0,
        presence_penalty=0.0,
        repeat_penalty=1.0,
        repetition_penalty=1.0,
        seed=7,
    )

    assert seen["prompt"] == _golden_granite_prompt("Hello")
    assert seen["generate_args"] == {
        "prompt_tokens": [11, 22],
        "max_new_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    assert result == "Granite strict-native reply."
