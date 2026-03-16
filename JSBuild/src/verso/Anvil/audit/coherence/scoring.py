from __future__ import annotations

from typing import Any


def _as_list(payload: Any) -> list[str]:
    if isinstance(payload, list):
        return [str(item) for item in payload if str(item).strip()]
    return []


def evaluate_coherence(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize coherence fields emitted by the native benchmark harness."""
    raw_ok = bool(row.get("coherence_valid", True))
    raw_issues = _as_list(row.get("coherence_issues"))
    return {
        "ok": raw_ok and not raw_issues,
        "issues": raw_issues,
        "raw_ok": raw_ok,
        "raw_issues": raw_issues,
    }


def evaluate_signal_completeness(row: dict[str, Any]) -> dict[str, Any]:
    """Check required telemetry signals for native audit rows."""
    required = (
        "decode_throughput_tps",
        "end_to_end_throughput_tps",
        "ttft_ms",
        "measurement_valid",
        "coherence_valid",
        "native_build_sha256",
        "loaded_native_library",
        "sanctioned_backend_path",
        "tokenizer_backend",
        "backend_module",
        "backend_module_library",
        "backend_module_loaded",
        "full_qsg_enabled",
        "strict_path_stable",
        "native_isa_baseline",
        "native_backend_abi_match",
        "grover_enabled",
        "coconut_enabled",
    )
    missing: list[str] = []
    for key in required:
        value = row.get(key)
        if value is None:
            missing.append(key)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(key)
    return {
        "ok": not missing,
        "missing": missing,
    }

