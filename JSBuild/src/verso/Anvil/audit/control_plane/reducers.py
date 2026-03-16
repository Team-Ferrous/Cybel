from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

QUESTION_ORDER = (
    "intended",
    "happened",
    "observed",
    "concluded",
    "unresolved",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_digest(path: Path, *, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    if not path.exists():
        return {
            "path": relative,
            "exists": False,
            "bytes": 0,
            "sha256": "",
        }
    payload = path.read_bytes()
    return {
        "path": relative,
        "exists": True,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def question_for_artifact(relative_path: str) -> str:
    normalized = relative_path.lower()
    if any(token in normalized for token in ("manifest", "resolved", "assurance_plan")):
        return "intended"
    if any(
        token in normalized
        for token in (
            "events",
            "transcript",
            "checkpoint",
            "attempt",
            "phase",
            "failure",
            "ledger",
        )
    ):
        return "happened"
    if any(
        token in normalized
        for token in ("summary", "telemetry", "passport", "metrics_rollup", "verify")
    ):
        return "observed"
    if any(
        token in normalized
        for token in (
            "comparison",
            "variance",
            "closure",
            "handoff",
            "triage",
            "traceability",
        )
    ):
        return "concluded"
    return "unresolved"


def summarize_gate_status(passed: bool, *, issues: list[str]) -> str:
    if passed and not issues:
        return "pass"
    if passed:
        return "pass_with_advisories"
    if issues:
        return "fail"
    return "unknown"
