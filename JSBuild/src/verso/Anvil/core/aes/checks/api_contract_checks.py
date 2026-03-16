from __future__ import annotations

import ast
from typing import Any

from core.aes.policy_surfaces import is_error_contract_surface


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_public_api_contract_markers(source: str, filepath: str) -> list[dict[str, Any]]:
    normalized = filepath.replace("\\", "/").lower()
    if "/core/aes/checks/" in f"/{normalized}" or normalized.startswith("core/aes/checks/"):
        return []
    if not is_error_contract_surface(filepath, source):
        return []
    tree = ast.parse(source)
    violations: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        if node.returns is None:
            violations.append(
                _violation(
                    "AES-ERR-1",
                    filepath,
                    node.lineno,
                    "Public API function is missing a return annotation.",
                )
            )
        doc = ast.get_docstring(node) or ""
        if "Raises" not in doc and "Errors" not in doc:
            violations.append(
                _violation(
                    "AES-ERR-1",
                    filepath,
                    node.lineno,
                    "Public API function docstring must describe error behavior.",
                )
            )
    return violations
