from __future__ import annotations

from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import now_iso

VARIANCE_BUDGET_SCHEMA_VERSION = "native_qsg_suite.variance_budget.v1"
DECODE_TPS_CV_BUDGET = 20.0
TTFT_CV_BUDGET = 20.0
DECODE_ACCOUNTING_MIN = 95.0
DECODE_ACCOUNTING_MAX = 105.0


def build_variance_budget(
    mission: MissionContext,
    *,
    topology_passport: dict[str, Any],
) -> dict[str, Any]:
    model_budgets: list[dict[str, Any]] = []
    recommendations: list[str] = []
    for model in list(mission.summary.get("models") or []):
        if not isinstance(model, dict):
            continue
        run_stability = dict(model.get("run_stability") or {})
        decode_cv = float(run_stability.get("decode_tps_cv_pct", 0.0) or 0.0)
        ttft_cv = float(run_stability.get("ttft_ms_cv_pct", 0.0) or 0.0)
        decode_accounted = float(model.get("decode_time_accounted_pct", 0.0) or 0.0)
        within_budget = (
            decode_cv <= DECODE_TPS_CV_BUDGET
            and ttft_cv <= TTFT_CV_BUDGET
            and DECODE_ACCOUNTING_MIN <= decode_accounted <= DECODE_ACCOUNTING_MAX
        )
        if not within_budget:
            recommendations.append(
                f"rerun {model.get('model')} with tighter host isolation or same-topology comparator"
            )
        model_budgets.append(
            {
                "model": str(model.get("model") or ""),
                "decode_tps_cv_pct": decode_cv,
                "ttft_ms_cv_pct": ttft_cv,
                "decode_time_accounted_pct": decode_accounted,
                "budgets": {
                    "decode_tps_cv_pct_max": DECODE_TPS_CV_BUDGET,
                    "ttft_ms_cv_pct_max": TTFT_CV_BUDGET,
                    "decode_time_accounted_pct_min": DECODE_ACCOUNTING_MIN,
                    "decode_time_accounted_pct_max": DECODE_ACCOUNTING_MAX,
                },
                "within_budget": within_budget,
            }
        )
    noise_signals = {
        "affinity_repair_attempted": bool(
            mission.preflight_payload.get("repair_attempted", False)
        ),
        "perf_available": bool(
            (mission.preflight_payload.get("perf") or {}).get("available", False)
        ),
        "cpu_governor": str(mission.preflight_payload.get("cpu_governor") or ""),
        "thp_mode": str(mission.preflight_payload.get("thp_mode") or ""),
        "saguaro_health_ok": bool(
            (mission.preflight_payload.get("saguaro") or {}).get("ok", False)
        ),
    }
    within_budget = bool(model_budgets) and all(
        bool(item.get("within_budget", False)) for item in model_budgets
    )
    if not model_budgets:
        recommendations.append(
            "collect at least one measured model summary before variance gating"
        )
    return {
        "schema_version": VARIANCE_BUDGET_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "topology_hash": str(topology_passport.get("topology_hash") or ""),
        "overall": {
            "within_budget": within_budget,
            "rerun_recommended": not within_budget,
        },
        "model_budgets": model_budgets,
        "noise_signals": noise_signals,
        "recommendations": recommendations,
    }
