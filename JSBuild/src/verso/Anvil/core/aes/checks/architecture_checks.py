from __future__ import annotations

from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_header_namespace_pollution(source: str, filepath: str) -> list[dict[str, Any]]:
    if filepath.endswith((".h", ".hpp")) and "using namespace std;" in source:
        return [
            _violation(
                "AES-CPP-2",
                filepath,
                1,
                "Headers must not use 'using namespace std;'.",
            )
        ]
    return []

