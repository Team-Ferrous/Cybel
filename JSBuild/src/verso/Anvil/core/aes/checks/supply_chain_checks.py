from __future__ import annotations

from pathlib import Path
from typing import Any

from core.aes.supply_chain import check_dependency_integrity


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_dependency_locking(source: str, filepath: str) -> list[dict[str, Any]]:
    violations = list(check_dependency_integrity(source, filepath))
    path = Path(filepath)
    if path.name == "pyproject.toml":
        parent = path.parent
        if not any((parent / name).exists() for name in ("poetry.lock", "uv.lock", "requirements.txt")):
            violations.append(
                _violation(
                    "AES-SUP-1",
                    filepath,
                    1,
                    "pyproject.toml changes require a lockfile or requirements baseline in the same project root.",
                )
            )
    return violations

