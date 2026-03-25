from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from core.aes.checks.registry_anchor import REGISTERED_AES_CHECKS as _REGISTERED_AES_CHECKS
from saguaro.sentinel.engines.aes import AESEngine

_ALLOWED_EXECUTION_MODES = {
    "static",
    "artifact",
    "runtime_gate",
    "workflow_gate",
    "manual",
}


def _normalize_modes(include_modes: Iterable[str] | None) -> set[str]:
    if include_modes is None:
        return {"static"}
    normalized = {str(mode).strip().lower() for mode in include_modes if str(mode).strip()}
    invalid = sorted(mode for mode in normalized if mode not in _ALLOWED_EXECUTION_MODES)
    if invalid:
        raise ValueError(f"Unsupported AES execution mode filter(s): {', '.join(invalid)}")
    return normalized or {"static"}


def run_aes_lint(
    paths: list[str],
    repo_root: str = ".",
    aal: str | None = None,
    domain: str | None = None,
    include_modes: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    _ = _REGISTERED_AES_CHECKS
    engine = AESEngine(str(Path(repo_root).resolve()))
    engine.set_policy(
        {
            "verify_context": {"aal": aal, "domain": domain},
            "allow_ruff_rules": True,
        }
    )
    allowed_modes = _normalize_modes(include_modes)
    rule_modes = {
        str(rule.id): str(getattr(rule, "execution_mode", "static") or "static")
        for rule in engine.registry.rules
    }
    violations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str]] = set()
    targets = paths or ["."]
    for target in targets:
        for violation in engine.run(path_arg=target):
            rule_id = str(violation.get("rule_id", ""))
            if rule_modes.get(rule_id) not in allowed_modes:
                continue
            key = (
                rule_id,
                str(violation.get("file", "")),
                int(violation.get("line", 1) or 1),
                str(violation.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            violations.append(violation)
    return sorted(
        violations,
        key=lambda item: (
            str(item.get("file", "")),
            int(item.get("line", 1) or 1),
            str(item.get("rule_id", "")),
            str(item.get("message", "")),
        ),
    )


def _format_violation(violation: dict[str, Any]) -> str:
    path = str(violation.get("file", "."))
    line = int(violation.get("line", 1) or 1)
    rule_id = str(violation.get("rule_id", "AES-UNKNOWN"))
    message = str(violation.get("message", "AES violation"))
    return f"{path}:{line}:1: {rule_id} {message}"


def format_aes_lint(
    violations: list[dict[str, Any]], output_format: str = "text"
) -> str:
    if output_format == "json":
        return json.dumps(violations, indent=2)
    if output_format == "github":
        return "\n".join(
            "::error file={file},line={line},col=1,title={rule_id}::{message}".format(
                file=str(violation.get("file", ".")),
                line=int(violation.get("line", 1) or 1),
                rule_id=str(violation.get("rule_id", "AES-UNKNOWN")),
                message=str(violation.get("message", "AES violation")),
            )
            for violation in violations
        )
    return "\n".join(_format_violation(violation) for violation in violations)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run deterministic AES static lint with Ruff-style diagnostics."
    )
    parser.add_argument("paths", nargs="*", help="Files or directories to lint")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--aal", default=None, help="Optional AAL filter")
    parser.add_argument("--domain", default=None, help="Optional domain filter")
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=tuple(sorted(_ALLOWED_EXECUTION_MODES)),
        help="Execution mode(s) to include. Defaults to static only.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "github"),
        default="text",
        help="Output format",
    )
    args = parser.parse_args()

    violations = run_aes_lint(
        paths=[str(path) for path in args.paths],
        repo_root=args.repo,
        aal=args.aal,
        domain=args.domain,
        include_modes=args.modes,
    )
    rendered = format_aes_lint(violations, output_format=args.format)
    if rendered:
        print(rendered)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
