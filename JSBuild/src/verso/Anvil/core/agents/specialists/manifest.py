"""Manifest-backed overlays for specialist routing defaults."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "standards" / "specialist_manifest.yaml"
)


@lru_cache(maxsize=1)
def load_specialist_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        return {}
    payload = yaml.safe_load(_MANIFEST_PATH.read_text(encoding="utf-8")) or {}
    return payload if isinstance(payload, dict) else {}


def manifest_generic_role_aliases() -> dict[str, str]:
    aliases = load_specialist_manifest().get("generic_role_aliases") or {}
    return {
        str(key).strip().lower(): str(value).strip()
        for key, value in aliases.items()
        if str(key).strip() and str(value).strip()
    }


def manifest_role_domain_hints() -> dict[str, set[str]]:
    hints = load_specialist_manifest().get("role_domain_hints") or {}
    normalized: dict[str, set[str]] = {}
    for key, values in hints.items():
        tokens = {str(value).strip().lower() for value in (values or []) if str(value).strip()}
        if tokens:
            normalized[str(key).strip().lower()] = tokens
    return normalized


def manifest_question_domain_hints() -> dict[str, set[str]]:
    hints = load_specialist_manifest().get("question_domain_hints") or {}
    normalized: dict[str, set[str]] = {}
    for key, values in hints.items():
        tokens = {str(value).strip().lower() for value in (values or []) if str(value).strip()}
        if tokens:
            normalized[str(key).strip().lower()] = tokens
    return normalized


def manifest_prompt_key_overrides() -> dict[str, str]:
    mapping = load_specialist_manifest().get("prompt_key_overrides") or {}
    return {
        str(key).strip(): str(value).strip()
        for key, value in mapping.items()
        if str(key).strip() and str(value).strip()
    }
