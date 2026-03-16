from __future__ import annotations

from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import now_iso

BASELINE_LINEAGE_SCHEMA_VERSION = "native_qsg_suite.baseline_lineage.v1"


def build_baseline_lineage(
    mission: MissionContext,
    *,
    topology_passport: dict[str, Any],
    compare_to: str,
) -> dict[str, Any]:
    baseline_summary = dict((mission.comparisons or {}).get("baseline") or {})
    comparator_catalog = {
        str(key): dict(value)
        for key, value in dict((mission.comparisons or {}).get("comparators") or {}).items()
        if isinstance(value, dict)
    }
    baseline_topology = dict(baseline_summary.get("topology_passport") or {})
    baseline_hash = str(
        baseline_topology.get("topology_hash")
        or baseline_summary.get("topology_hash")
        or ""
    )
    current_hash = str(topology_passport.get("topology_hash") or "")
    topology_match = (
        bool(baseline_summary) and baseline_hash == current_hash and bool(current_hash)
    )
    regression_signals: list[dict[str, Any]] = []
    baseline_models = {
        str(item.get("model") or ""): item
        for item in list(baseline_summary.get("models") or [])
        if isinstance(item, dict)
    }
    for model in list(mission.summary.get("models") or []):
        if not isinstance(model, dict):
            continue
        model_name = str(model.get("model") or "")
        baseline_model = dict(baseline_models.get(model_name) or {})
        if not baseline_model:
            continue
        baseline_decode = float(baseline_model.get("decode_tps_p50", 0.0) or 0.0)
        current_decode = float(model.get("decode_tps_p50", 0.0) or 0.0)
        baseline_ttft = float(baseline_model.get("ttft_ms_p95", 0.0) or 0.0)
        current_ttft = float(model.get("ttft_ms_p95", 0.0) or 0.0)
        regression_signals.append(
            {
                "model": model_name,
                "decode_tps_delta_pct": (
                    ((current_decode - baseline_decode) / baseline_decode) * 100.0
                    if baseline_decode > 0.0
                    else 0.0
                ),
                "ttft_ms_delta_pct": (
                    ((current_ttft - baseline_ttft) / baseline_ttft) * 100.0
                    if baseline_ttft > 0.0
                    else 0.0
                ),
            }
        )
    comparable = bool(baseline_summary) and (
        topology_match or not baseline_hash or not current_hash
    )
    comparator_mode = str(compare_to or "latest_compatible")
    comparator_classes = [
        {
            "mode": mode,
            "run_id": str(item.get("run_id") or ""),
            "overall_pass": bool(item.get("overall_pass", False)),
        }
        for mode, item in sorted(comparator_catalog.items())
    ]
    return {
        "schema_version": BASELINE_LINEAGE_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "comparator_mode": comparator_mode,
        "cohort_key": str(topology_passport.get("cohort_key") or ""),
        "comparator_classes": comparator_classes,
        "comparator_class_count": len(comparator_classes),
        "comparable": comparable,
        "topology_match": topology_match,
        "baseline": {
            "present": bool(baseline_summary),
            "run_id": str(baseline_summary.get("run_id") or ""),
            "topology_hash": baseline_hash,
            "summary": baseline_summary,
        },
        "regression_signals": regression_signals,
    }
