from __future__ import annotations

from typing import Any


def _violation(rule_id: str, filepath: str, line: int, message: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "filepath": filepath,
        "line": line,
        "message": message,
    }


def check_traceability_markers(source: str, filepath: str) -> list[dict[str, Any]]:
    if any(marker in source for marker in ("trace_id", "run_id", "span_id")):
        return []
    return [
        _violation(
            "AES-OBS-1",
            filepath,
            1,
            "Operational code is missing trace/run correlation markers.",
        )
    ]


def check_operational_logging_markers(source: str, filepath: str) -> list[dict[str, Any]]:
    lowered = source.lower()
    if any(token in lowered for token in ("logger.", "logging.", "structlog")):
        return []
    return [
        _violation(
            "AES-OBS-2",
            filepath,
            1,
            "Operational code is missing structured logging markers.",
        )
    ]
