#!/usr/bin/env python3
"""Validate Phase-2 prompt governance contracts and freshness guarantees."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REQUIRED_CONTRACT_KEYS = (
    "AAL_CLASSIFICATION",
    "APPLICABLE_RULE_IDS",
    "REQUIRED_ARTIFACTS",
    "BLOCKING_GATES",
    "AES_VISUALS_PACKS",
    "AES_VISUALS_SUMMARY",
)

FORBIDDEN_DIRECTIVES = (
    "skip verification",
    "ignore tests",
)

REQUIRED_AES_VISUAL_REFERENCES = (
    "aes_visuals/v1/",
    "aes_visuals/v2/",
)

STALE_TIMESTAMP_PATTERNS = (
    re.compile(r"artifacts/audit_visuals/\d{4}[-_]\d{2}[-_]\d{2}"),
    re.compile(r"artifacts/audit_visuals/\d{8}"),
)

STALE_LINK_SEGMENT_PATTERN = re.compile(r"\b(19|20)\d{2}[-_/]?\d{2}[-_/]?\d{2}\b")

MARKDOWN_LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _error(path: Path | str, message: str) -> dict[str, str]:
    return {"file": str(path), "message": message}


def _extract_contract_fields(prompt_text: str) -> dict[str, str]:
    match = re.search(
        r"<AES_PROMPT_CONTRACT>\s*(.*?)\s*</AES_PROMPT_CONTRACT>",
        prompt_text,
        flags=re.DOTALL,
    )
    if not match:
        return {}
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        fields[key.strip()] = value.strip()
    return fields


def _validate_contract_fields(
    prompt_text: str,
    source: str,
    errors: list[dict[str, str]],
) -> None:
    contract_fields = _extract_contract_fields(prompt_text)
    lower_prompt = prompt_text.lower()
    if not contract_fields:
        errors.append(_error(source, "missing <AES_PROMPT_CONTRACT> block"))
        return
    for key in REQUIRED_CONTRACT_KEYS:
        value = contract_fields.get(key, "")
        if not value or value.lower() == "none":
            errors.append(_error(source, f"contract key missing or empty: {key}"))
    for reference in REQUIRED_AES_VISUAL_REFERENCES:
        if reference not in lower_prompt:
            errors.append(_error(source, f"missing aes_visuals reference in prompt text: {reference}"))


def _validate_paths_in_markdown(path: Path, errors: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for raw_target in MARKDOWN_LINK_PATTERN.findall(text):
        target = raw_target.strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        normalized_target = target.split("#", 1)[0].split("?", 1)[0].strip()
        if STALE_LINK_SEGMENT_PATTERN.search(normalized_target):
            errors.append(_error(path, f"stale timestamp in local link: {target}"))
        resolved = (path.parent / normalized_target).resolve()
        if not resolved.exists():
            errors.append(_error(path, f"broken local link: {target}"))


def _validate_aes_visual_references(path: Path, errors: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8").lower()
    for reference in REQUIRED_AES_VISUAL_REFERENCES:
        if reference not in text:
            errors.append(_error(path, f"missing required aes_visuals reference: {reference}"))


def _validate_freshness(path: Path, errors: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for pattern in STALE_TIMESTAMP_PATTERNS:
        match = pattern.search(text)
        if match:
            errors.append(
                _error(path, f"stale hardcoded artifact timestamp reference: {match.group(0)}")
            )


def _validate_forbidden_directives(path: Path, errors: list[dict[str, str]]) -> None:
    text = path.read_text(encoding="utf-8").lower()
    for directive in FORBIDDEN_DIRECTIVES:
        if directive in text:
            errors.append(_error(path, f"forbidden directive detected: {directive}"))


def run(repo_root: Path) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    warnings: list[str] = []
    checked_files: list[str] = []

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    try:
        from core.prompts import PromptManager
    except Exception as exc:  # pragma: no cover - defensive for CI bootstrap issues
        errors.append(_error("core.prompts", f"failed to import PromptManager: {exc}"))
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "checked_files": checked_files,
            "required_contract_keys": list(REQUIRED_CONTRACT_KEYS),
        }

    prompt_manager = PromptManager()
    generated_master_prompt = prompt_manager.get_master_prompt(
        agent_name="PromptContractValidator",
        context_type="general",
        task_text="validate prompt governance contracts",
        workset_files=[],
    )
    _validate_contract_fields(generated_master_prompt, "generated:master_prompt", errors)
    checked_files.append("generated:master_prompt")

    generated_system_prompt = prompt_manager.get_system_prompt(
        workset_files=[],
        task_text="validate prompt governance contracts",
        role="master",
    )
    _validate_contract_fields(generated_system_prompt, "generated:system_prompt", errors)
    checked_files.append("generated:system_prompt")

    prompt_files = [
        repo_root / "prompts" / "GEMINI.md",
        repo_root / "prompts" / "shared_prompt_foundation.md",
    ]
    for prompt_file in prompt_files:
        if not prompt_file.exists():
            errors.append(_error(prompt_file, "missing required prompt file"))
            continue
        checked_files.append(str(prompt_file))
        _validate_paths_in_markdown(prompt_file, errors)
        _validate_aes_visual_references(prompt_file, errors)
        _validate_freshness(prompt_file, errors)
        _validate_forbidden_directives(prompt_file, errors)

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checked_files": checked_files,
        "required_contract_keys": list(REQUIRED_CONTRACT_KEYS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prompt contracts and freshness.")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    result = run(repo_root)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print("Prompt contract validation: OK")
        else:
            print("Prompt contract validation: FAILED")
            for item in result["errors"]:
                print(f"- {item['file']}: {item['message']}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
