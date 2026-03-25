from __future__ import annotations

import hashlib
import json
import re
from itertools import count
from pathlib import Path
from typing import Any

from audit.store.writer import write_json_atomic

EVIDENCE_CAPSULE_SCHEMA_VERSION = "native_qsg_audit.evidence_capsule.v1"
LATENT_FEEDBACK_SCHEMA_VERSION = "native_qsg_audit.latent_feedback.v1"

_SEQUENCE_IDS = count(1)
_FAILED_TEST_PATTERNS = (
    re.compile(r"^FAILED\s+(.+)$", re.MULTILINE),
    re.compile(r"^ERROR\s+(.+)$", re.MULTILINE),
)
_COMPILER_DIAGNOSTIC_PATTERN = re.compile(
    r"^.*(?::\d+(?::\d+)?)?:\s*(?:fatal\s+)?(?:error|warning):.*$",
    re.MULTILINE | re.IGNORECASE,
)


def next_sequence_id() -> int:
    return int(next(_SEQUENCE_IDS))


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def extract_failed_tests(*texts: str) -> list[str]:
    matches: list[str] = []
    for text in texts:
        for pattern in _FAILED_TEST_PATTERNS:
            matches.extend(match.group(1).strip() for match in pattern.finditer(text or ""))
    return sorted({item for item in matches if item})


def extract_compiler_diagnostics(*texts: str) -> list[str]:
    diagnostics: list[str] = []
    for text in texts:
        diagnostics.extend(
            match.group(0).strip()
            for match in _COMPILER_DIAGNOSTIC_PATTERN.finditer(text or "")
        )
    return diagnostics[:20]


def build_latent_feedback_projection(
    *,
    exit_code: int,
    summary: str | None,
    benchmark_metrics: dict[str, Any],
    failing_tests: list[str],
    compiler_diagnostics: list[str],
) -> dict[str, Any]:
    summary_text = str(summary or "").strip()
    scalar_metrics: dict[str, float | None] = {}
    for key in (
        "wall_time_ms",
        "user_time_ms",
        "sys_time_ms",
        "max_rss_mb",
        "decode_tps",
        "e2e_tps",
        "ttft_ms",
        "gate_issue_count",
    ):
        value = benchmark_metrics.get(key)
        try:
            scalar_metrics[key] = None if value is None else float(value)
        except Exception:
            scalar_metrics[key] = None
    status_markers = [
        "exit_ok" if int(exit_code) == 0 else "exit_nonzero",
        "tests_failed" if failing_tests else "tests_passed",
        "compiler_diagnostics_present" if compiler_diagnostics else "compiler_clean",
    ]
    symbolic_markers = sorted(
        {
            str(item)
            for item in (
                benchmark_metrics.get("measurement_valid_marker"),
                benchmark_metrics.get("coherence_ok_marker"),
                benchmark_metrics.get("strict_path_stable_marker"),
            )
            if str(item or "").strip()
        }
    )
    return {
        "schema_version": LATENT_FEEDBACK_SCHEMA_VERSION,
        "status_markers": status_markers,
        "scalar_metrics": scalar_metrics,
        "symbolic_markers": symbolic_markers,
        "summary_features": {
            "summary_sha256": _sha256_text(summary_text) if summary_text else None,
            "summary_token_count": len(summary_text.split()) if summary_text else 0,
            "failing_test_count": len(failing_tests),
            "compiler_diagnostic_count": len(compiler_diagnostics),
        },
    }


def build_evidence_capsule(
    *,
    sequence_id: int,
    tool_run_id: str,
    source: str,
    command: list[str],
    cwd: str,
    exit_code: int,
    wall_time_ms: float,
    user_time_ms: float | None,
    sys_time_ms: float | None,
    max_rss_mb: float | None,
    stdout_path: str | None,
    stderr_path: str | None,
    artifact_paths: dict[str, str],
    failing_tests: list[str],
    compiler_diagnostics: list[str],
    benchmark_metrics: dict[str, Any],
    summary: str | None,
    replay: dict[str, Any],
    stdout_text: str = "",
    stderr_text: str = "",
) -> dict[str, Any]:
    metrics = dict(benchmark_metrics)
    metrics.setdefault("wall_time_ms", round(float(wall_time_ms), 3))
    metrics.setdefault("user_time_ms", None if user_time_ms is None else round(float(user_time_ms), 3))
    metrics.setdefault("sys_time_ms", None if sys_time_ms is None else round(float(sys_time_ms), 3))
    metrics.setdefault("max_rss_mb", None if max_rss_mb is None else round(float(max_rss_mb), 3))
    capsule = {
        "schema_version": EVIDENCE_CAPSULE_SCHEMA_VERSION,
        "sequence_id": int(sequence_id),
        "tool_run_id": str(tool_run_id),
        "source": str(source),
        "command": [str(part) for part in command],
        "cwd": str(cwd),
        "exit_code": int(exit_code),
        "wall_time_ms": round(float(wall_time_ms), 3),
        "user_time_ms": None if user_time_ms is None else round(float(user_time_ms), 3),
        "sys_time_ms": None if sys_time_ms is None else round(float(sys_time_ms), 3),
        "max_rss_mb": None if max_rss_mb is None else round(float(max_rss_mb), 3),
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "artifact_paths": {str(key): str(value) for key, value in artifact_paths.items()},
        "failing_tests": [str(item) for item in failing_tests],
        "compiler_diagnostics": [str(item) for item in compiler_diagnostics],
        "benchmark_metrics": metrics,
        "summary": str(summary) if summary is not None else None,
        "stdout_sha256": _sha256_text(stdout_text or ""),
        "stderr_sha256": _sha256_text(stderr_text or ""),
        "replay": {
            "checkpoint_metadata_path": replay.get("checkpoint_metadata_path"),
            "flight_recorder_timeline_path": replay.get("flight_recorder_timeline_path"),
            "terminal_transcript_path": replay.get("terminal_transcript_path"),
            "inspectable_without_model": bool(replay.get("inspectable_without_model", False)),
        },
    }
    capsule["latent_feedback_projection"] = build_latent_feedback_projection(
        exit_code=int(exit_code),
        summary=summary,
        benchmark_metrics=metrics,
        failing_tests=[str(item) for item in failing_tests],
        compiler_diagnostics=[str(item) for item in compiler_diagnostics],
    )
    capsule["result_hash"] = stable_result_hash(capsule)
    return capsule


def stable_result_hash(capsule: dict[str, Any]) -> str:
    payload = {key: value for key, value in capsule.items() if key != "result_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_text(encoded)


def write_evidence_capsule(path: Path, capsule: dict[str, Any]) -> None:
    write_json_atomic(path, capsule)

