from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import now_iso


SPC_REPORT_SCHEMA_VERSION = "native_qsg_suite.spc_report.v1"


def _summary_candidates(audit_root: Path, current_run_id: str) -> list[Path]:
    candidates: list[Path] = []
    for root in (audit_root, audit_root / "runs"):
        if not root.exists():
            continue
        for path in root.rglob("summary.json"):
            if current_run_id in str(path):
                continue
            candidates.append(path)
    return sorted(set(candidates), reverse=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _ewma(values: list[float], alpha: float = 0.3) -> list[float]:
    if not values:
        return []
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def _cusum(values: list[float], target: float) -> list[float]:
    total = 0.0
    result: list[float] = []
    for value in values:
        total += float(value) - target
        result.append(total)
    return result


def _same_topology(summary: dict[str, Any], topology_hash: str) -> bool:
    candidate_hash = str(
        (summary.get("topology_passport") or {}).get("topology_hash")
        or (summary.get("baseline_lineage") or {}).get("topology_hash")
        or ""
    )
    return bool(candidate_hash) and candidate_hash == topology_hash


def _same_host(summary: dict[str, Any], host_fingerprint: str) -> bool:
    candidate = str(
        (summary.get("topology_passport") or {}).get("host_fingerprint")
        or ((summary.get("host_compliance") or {}).get("host") or {}).get(
            "host_fingerprint"
        )
        or ""
    )
    return bool(candidate) and candidate == host_fingerprint


def build_spc_report(
    mission: MissionContext,
    *,
    topology_passport: dict[str, Any],
    compare_to: str,
) -> dict[str, Any]:
    topology_hash = str(topology_passport.get("topology_hash") or "")
    host_fingerprint = str(topology_passport.get("host_fingerprint") or "")
    history: list[dict[str, Any]] = []
    for candidate in _summary_candidates(mission.layout.root.parent, mission.run_id):
        summary = _load_json(candidate)
        if not summary:
            continue
        if compare_to == "last_same_topology" and topology_hash and not _same_topology(summary, topology_hash):
            continue
        if host_fingerprint and not _same_host(summary, host_fingerprint):
            continue
        history.append(summary)
        if len(history) >= 12:
            break
    current_models = {
        str(item.get("model") or ""): item
        for item in list(mission.summary.get("models") or [])
        if isinstance(item, dict)
    }
    model_reports: list[dict[str, Any]] = []
    drift_alert = False
    for model_name, model_summary in current_models.items():
        decode_values: list[float] = []
        ttft_values: list[float] = []
        for summary in reversed(history):
            models = {
                str(item.get("model") or ""): item
                for item in list(summary.get("models") or [])
                if isinstance(item, dict)
            }
            candidate = dict(models.get(model_name) or {})
            if not candidate:
                continue
            decode_values.append(float(candidate.get("decode_tps_p50", 0.0) or 0.0))
            ttft_values.append(float(candidate.get("ttft_ms_p95", 0.0) or 0.0))
        current_decode = float(model_summary.get("decode_tps_p50", 0.0) or 0.0)
        current_ttft = float(model_summary.get("ttft_ms_p95", 0.0) or 0.0)
        if decode_values:
            decode_values.append(current_decode)
        if ttft_values:
            ttft_values.append(current_ttft)
        decode_ewma = _ewma(decode_values)
        ttft_ewma = _ewma(ttft_values)
        decode_target = decode_values[-2] if len(decode_values) >= 2 else current_decode
        ttft_target = ttft_values[-2] if len(ttft_values) >= 2 else current_ttft
        decode_cusum = _cusum(decode_values, decode_target) if decode_values else []
        ttft_cusum = _cusum(ttft_values, ttft_target) if ttft_values else []
        decode_alert = len(decode_values) >= 3 and current_decode < min(decode_values[:-1])
        ttft_alert = len(ttft_values) >= 3 and current_ttft > max(ttft_values[:-1])
        drift_alert = drift_alert or decode_alert or ttft_alert
        model_reports.append(
            {
                "model": model_name,
                "history_count": len(decode_values) - 1 if decode_values else 0,
                "decode_tps_p50": {
                    "current": current_decode,
                    "ewma": decode_ewma[-1] if decode_ewma else current_decode,
                    "cusum": decode_cusum[-1] if decode_cusum else 0.0,
                    "alert": decode_alert,
                },
                "ttft_ms_p95": {
                    "current": current_ttft,
                    "ewma": ttft_ewma[-1] if ttft_ewma else current_ttft,
                    "cusum": ttft_cusum[-1] if ttft_cusum else 0.0,
                    "alert": ttft_alert,
                },
            }
        )
    status = "insufficient_history" if not history else ("alert" if drift_alert else "stable")
    return {
        "schema_version": SPC_REPORT_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "compare_to": compare_to,
        "topology_hash": topology_hash,
        "host_fingerprint": host_fingerprint,
        "history_runs_considered": len(history),
        "cohort_type": "same_host" if host_fingerprint else "mixed",
        "status": status,
        "drift_alert": drift_alert,
        "models": model_reports,
    }
