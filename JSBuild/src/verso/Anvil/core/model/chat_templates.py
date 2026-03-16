"""Shared chat-template helpers for strict-native prompt formatting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from config.settings import PRODUCTION_MODEL_ALLOWLIST
from core.model.model_contract import canonicalize_model_name

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
GRANITE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Please ensure responses are professional, "
    "accurate, and safe."
)


@dataclass(frozen=True)
class PromptTemplateContract:
    model_name: str
    template_name: str
    system_prompt: str | None
    assistant_prefix: str | None = None
    inject_system_prompt: bool = True
    disallowed_output_prefixes: tuple[str, ...] = ()


CHAT_TEMPLATES = {
    "granite": lambda role, content: (
        f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
    ),
    "granite_role": lambda role, content: (
        f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
    ),
    "chatml": lambda role, content: f"<|im_start|>{role}\n{content}<|im_end|>\n",
}

CHAT_SUFFIXES = {
    "granite": "<|start_of_role|>assistant<|end_of_role|>",
    "granite_role": "<|start_of_role|>assistant<|end_of_role|>",
    "chatml": "<|im_start|>assistant\n",
}

PROMPT_CONTRACTS = {
    "granite4:tiny-h": PromptTemplateContract(
        model_name="granite4:tiny-h",
        template_name="granite",
        system_prompt=GRANITE_SYSTEM_PROMPT,
        inject_system_prompt=True,
    ),
    "qwen3.5:9b": PromptTemplateContract(
        model_name="qwen3.5:9b",
        template_name="chatml",
        system_prompt=None,
        assistant_prefix=None,
        inject_system_prompt=False,
        disallowed_output_prefixes=(
            "<think>",
            "</think>",
            "Assistant:",
            "assistant:",
            "Assistant",
            "assistant",
        ),
    ),
    "qwen3.5:4b": PromptTemplateContract(
        model_name="qwen3.5:4b",
        template_name="chatml",
        system_prompt=None,
        assistant_prefix=None,
        inject_system_prompt=False,
        disallowed_output_prefixes=(
            "<think>",
            "</think>",
            "Assistant:",
            "assistant:",
            "Assistant",
            "assistant",
        ),
    ),
}


def _is_qwen35_model_variant(model_name: str) -> bool:
    lowered = str(model_name or "").strip().lower()
    if not lowered:
        return False
    return "qwen3.5" in lowered or "qwen35" in lowered


def _canonical_or_raw_model_name(model_name: str) -> str:
    raw = str(model_name or "").strip()
    if not raw:
        return ""
    try:
        return canonicalize_model_name(raw)
    except Exception:
        return raw.lower()


def get_strict_prompt_contract(
    model_name: str,
    *,
    profile: Any | None = None,
    strict: bool = True,
) -> PromptTemplateContract | None:
    _ = profile
    canonical = _canonical_or_raw_model_name(model_name)
    contract = PROMPT_CONTRACTS.get(canonical)
    if contract is not None:
        return contract
    if not strict and _is_qwen35_model_variant(model_name):
        # Non-strict helpers keep qwen3.5 chatml formatting consistent across
        # tags/aliases while strict mode still fails closed to pinned models.
        return PROMPT_CONTRACTS["qwen3.5:4b"]
    if strict:
        raise RuntimeError(
            "Strict native prompt contracts support only qwen3.5:4b, "
            "qwen3.5:9b and granite4:tiny-h."
        )
    return None


def normalize_chat_template_name(template: Any) -> str:
    normalized = str(template or "").strip().lower()
    if normalized == "granite_role":
        return "granite"
    if normalized in CHAT_TEMPLATES:
        return normalized
    return ""


def default_chat_template_name(
    model_name: str,
    family: str | None = None,
    architecture: str | None = None,
) -> str:
    canonical = _canonical_or_raw_model_name(model_name)
    spec = PRODUCTION_MODEL_ALLOWLIST.get(canonical)
    if spec:
        return normalize_chat_template_name(spec.get("template"))

    lower_name = str(model_name or "").lower()
    lower_family = str(family or "").lower()
    lower_arch = str(architecture or "").lower()
    if (
        "qwen3.5" in lower_name
        or "qwen35" in lower_name
        or "qwen" in lower_arch
        or lower_family == "qwen"
    ):
        return "chatml"
    if (
        "granite4" in lower_name
        or "granite" in lower_arch
        or lower_family == "granite"
    ):
        return "granite"
    return ""


def resolve_prompt_contract(
    model_name: str,
    profile: Any | None = None,
    *,
    strict: bool = False,
    architecture: str | None = None,
    family: str | None = None,
) -> PromptTemplateContract:
    family = str(family or "")
    arch = str(architecture or "")
    if profile is not None:
        family = str(getattr(profile, "family", "") or family)
        arch = str(getattr(profile, "architecture", "") or arch)

    canonical = _canonical_or_raw_model_name(model_name)
    contract = PROMPT_CONTRACTS.get(canonical)
    if strict:
        if contract is None:
            raise RuntimeError(
                "Strict native prompt formatting supports only qwen3.5:4b, "
                "qwen3.5:9b and granite4:tiny-h."
            )
        return contract

    template_name = ""
    if profile is not None:
        template_name = normalize_chat_template_name(
            getattr(profile, "chat_template", "")
        )
        if template_name:
            return PromptTemplateContract(
                model_name=str(model_name or ""),
                template_name=template_name,
                system_prompt=(
                    GRANITE_SYSTEM_PROMPT
                    if template_name == "granite"
                    else DEFAULT_SYSTEM_PROMPT
                ),
                inject_system_prompt=template_name == "granite",
            )

    if contract is not None:
        return contract

    if not template_name:
        template_name = default_chat_template_name(
            model_name,
            family=family,
            architecture=arch,
        )
    if template_name:
        return PromptTemplateContract(
            model_name=str(model_name or ""),
            template_name=template_name,
            system_prompt=(
                GRANITE_SYSTEM_PROMPT
                if template_name == "granite"
                else DEFAULT_SYSTEM_PROMPT
            ),
            inject_system_prompt=template_name == "granite",
        )
    return PromptTemplateContract(
        model_name=str(model_name or ""),
        template_name="",
        system_prompt=None,
    )


def resolve_chat_template_name(
    model_name: str,
    profile: Any | None = None,
    *,
    strict: bool = False,
    architecture: str | None = None,
    family: str | None = None,
) -> str:
    return resolve_prompt_contract(
        model_name,
        profile=profile,
        strict=strict,
        architecture=architecture,
        family=family,
    ).template_name


def is_preformatted_prompt(text: str, template_name: str) -> bool:
    normalized = normalize_chat_template_name(template_name)
    if normalized == "chatml":
        return "<|im_start|>" in text and (
            "<|im_end|>" in text or text.endswith(CHAT_SUFFIXES["chatml"])
        )
    if normalized == "granite":
        return "<|start_of_role|>" in text and "<|end_of_role|>" in text
    return False


def format_chat_messages(
    messages: list[dict[str, Any]],
    template_name: str,
    assistant_prefix: str | None = None,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    inject_system_prompt: bool = True,
) -> str:
    normalized = normalize_chat_template_name(template_name)
    if normalized not in CHAT_TEMPLATES:
        raise RuntimeError(
            f"Unsupported chat template '{template_name}' for strict-native formatting."
        )

    formatter = CHAT_TEMPLATES[normalized]
    normalized_messages: list[dict[str, Any]] = list(messages)
    if inject_system_prompt and system_prompt and not any(
        str(msg.get("role", "")).strip().lower() == "system"
        for msg in normalized_messages
    ):
        normalized_messages = [
            {"role": "system", "content": system_prompt},
            *normalized_messages,
        ]

    prompt = ""
    for msg in normalized_messages:
        role = str(msg["role"]).strip().lower()
        content = str(msg["content"])
        if role == "tool":
            role = "system"
            content = f"Tool Response: {content}"
        prompt += formatter(role, content)

    prompt += CHAT_SUFFIXES[normalized]
    if assistant_prefix:
        prompt += assistant_prefix
    return prompt


def format_strict_native_prompt(
    prompt: str,
    template_name: str,
    *,
    model_name: str | None = None,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    assistant_prefix: str | None = None,
) -> str:
    text = str(prompt or "")
    if not text:
        return text
    normalized = normalize_chat_template_name(template_name)
    if not normalized:
        return text
    inject_system_prompt = normalized == "granite"
    if normalized == "granite" and not system_prompt:
        system_prompt = GRANITE_SYSTEM_PROMPT
    if normalized == "granite" and system_prompt == DEFAULT_SYSTEM_PROMPT:
        system_prompt = GRANITE_SYSTEM_PROMPT
    if model_name:
        contract = get_strict_prompt_contract(
            model_name,
            strict=False,
        )
        if contract is not None:
            if assistant_prefix is None:
                assistant_prefix = contract.assistant_prefix
            if system_prompt == DEFAULT_SYSTEM_PROMPT:
                system_prompt = contract.system_prompt
            inject_system_prompt = bool(contract.inject_system_prompt)
    if is_preformatted_prompt(text, normalized):
        return text
    return format_chat_messages(
        [{"role": "user", "content": text}],
        normalized,
        system_prompt=system_prompt,
        assistant_prefix=assistant_prefix,
        inject_system_prompt=inject_system_prompt,
    )


def format_prompt_for_model(prompt: str, model_name: str, profile: Any | None = None) -> str:
    contract = resolve_prompt_contract(model_name, profile=profile, strict=True)
    return format_strict_native_prompt(
        prompt,
        contract.template_name,
        model_name=contract.model_name,
        system_prompt=contract.system_prompt,
        assistant_prefix=contract.assistant_prefix,
    )


def _strip_leading_think_block(text: str) -> str:
    stripped = str(text or "").lstrip()
    if not stripped.startswith("<think>"):
        return str(text or "")
    end_idx = stripped.find("</think>")
    if end_idx < 0:
        after_opener = stripped[len("<think>") :].lstrip()
        blank_idx = after_opener.find("\n\n")
        if blank_idx < 0:
            return ""
        return after_opener[blank_idx + 2 :].lstrip()
    return stripped[end_idx + len("</think>") :].lstrip()


def _strip_leading_think_closer(text: str) -> str:
    stripped = str(text or "").lstrip()
    if not stripped.startswith("</think>"):
        return str(text or "")
    return stripped[len("</think>") :].lstrip()


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CHATML_ASSISTANT_PREFIX_RE = re.compile(r"^\s*assistant\s*:?\s*", re.IGNORECASE)
_CHATML_ASSISTANT_SUFFIX_RE = re.compile(r"\s*assistant\s*:?\s*$", re.IGNORECASE)
_CHATML_ASSISTANT_LINE_RE = re.compile(r"(?mi)^\s*assistant\s*:?\s*$")


def _strip_disallowed_think_tags(text: str) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", str(text or ""))
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _strip_chatml_assistant_markers(text: str) -> str:
    cleaned = str(text or "")
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = _CHATML_ASSISTANT_LINE_RE.sub("", cleaned)
        cleaned = _CHATML_ASSISTANT_PREFIX_RE.sub("", cleaned)
        cleaned = _CHATML_ASSISTANT_SUFFIX_RE.sub("", cleaned)
        cleaned = cleaned.strip()
    return cleaned


def postprocess_strict_native_response(
    text: str,
    *,
    model_name: str = "",
    template_name: str = "",
) -> str:
    result = str(text or "")
    contract = None
    if model_name:
        contract = get_strict_prompt_contract(model_name, strict=False)
    if (
        contract is not None
        and str(contract.template_name).strip().lower() == "chatml"
        and (
            _is_qwen35_model_variant(model_name)
            or _is_qwen35_model_variant(str(contract.model_name))
        )
    ):
        return _strip_chatml_assistant_markers(
            _strip_disallowed_think_tags(
                _strip_leading_think_closer(_strip_leading_think_block(result))
            )
        )
    if normalize_chat_template_name(template_name) == "chatml":
        return _strip_chatml_assistant_markers(
            _strip_disallowed_think_tags(
                _strip_leading_think_closer(_strip_leading_think_block(result))
            )
        )
    return result


def format_completion_prompt(
    prompt: str,
    template_name: str,
    *,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    assistant_prefix: str | None = None,
) -> str:
    return format_strict_native_prompt(
        prompt,
        template_name,
        model_name=None,
        system_prompt=system_prompt,
        assistant_prefix=assistant_prefix,
    )
