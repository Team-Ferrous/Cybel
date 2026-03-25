from __future__ import annotations

import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

from audit.control_plane.mission import MissionContext
from audit.control_plane.reducers import now_iso

ADVISORY_BUNDLE_SCHEMA_VERSION = "native_qsg_suite.advisory_bundle.v1"


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


def _same_host(summary: dict[str, Any], host_fingerprint: str) -> bool:
    candidate = str(
        (summary.get("topology_passport") or {}).get("host_fingerprint")
        or ((summary.get("control_plane") or {}).get("host_identity") or {}).get(
            "host_fingerprint"
        )
        or ""
    )
    return bool(candidate) and candidate == host_fingerprint


def _model_metric_map(summary: dict[str, Any], key: str) -> dict[str, float]:
    return {
        str(item.get("model") or ""): float(item.get(key, 0.0) or 0.0)
        for item in list(summary.get("models") or [])
        if isinstance(item, dict) and str(item.get("model") or "").strip()
    }


def _quality_records(quality_payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        item for item in list(quality_payload.get(key) or []) if isinstance(item, dict)
    ]


def _lane_coverage(mission: MissionContext, lane_id: str) -> bool:
    return lane_id in set(mission.completed_lanes)


def _build_operator_control_room(
    mission: MissionContext,
    *,
    run_ledger: dict[str, Any],
    closure_result: dict[str, Any],
) -> dict[str, Any]:
    events = [
        item for item in list(run_ledger.get("events") or []) if isinstance(item, dict)
    ]
    counts_by_type: dict[str, int] = {}
    phases: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type") or "")
        phase = str(event.get("phase") or "")
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + 1
        if phase:
            phases[phase] = phases.get(phase, 0) + 1
    timeline = [
        {
            "timestamp": str(event.get("timestamp") or ""),
            "event_type": str(event.get("event_type") or ""),
            "phase": str(event.get("phase") or ""),
            "lane": str(event.get("lane") or ""),
            "message": str(event.get("message") or ""),
        }
        for event in events[-25:]
    ]
    return {
        "status": "attention" if closure_result.get("unresolved") else "clear",
        "event_count": len(events),
        "counts_by_type": counts_by_type,
        "counts_by_phase": phases,
        "unresolved_items": list(closure_result.get("unresolved") or []),
        "timeline": timeline,
    }


def _build_perturbation_matrix(mission: MissionContext) -> dict[str, Any]:
    summary_models = [
        item
        for item in list(mission.summary.get("models") or [])
        if isinstance(item, dict)
    ]
    quality_payload = dict(mission.summary.get("quality") or {})
    coherence_rows = _quality_records(quality_payload, "coherence")
    invariants: list[dict[str, Any]] = []
    invariants.append(
        {
            "invariant_id": "prompt_truncation_shadow",
            "status": "covered" if coherence_rows else "blocked",
            "passed": bool(coherence_rows),
            "reason": "" if coherence_rows else "quality_eval_missing",
        }
    )
    invariants.append(
        {
            "invariant_id": "minor_thread_perturbation",
            "status": (
                "covered" if _lane_coverage(mission, "thread_matrix") else "blocked"
            ),
            "passed": _lane_coverage(mission, "thread_matrix"),
            "reason": (
                ""
                if _lane_coverage(mission, "thread_matrix")
                else "thread_matrix_lane_disabled"
            ),
        }
    )
    accuracy_rows = _quality_records(quality_payload, "accuracy")
    invariants.append(
        {
            "invariant_id": "all_on_accuracy_truth",
            "status": "covered" if accuracy_rows else "blocked",
            "passed": bool(accuracy_rows),
            "reason": "" if accuracy_rows else "accuracy_eval_missing",
        }
    )
    return {
        "status": "covered" if summary_models else "blocked",
        "model_count": len(summary_models),
        "invariants": invariants,
    }


def _build_causal_hotspot_lab(mission: MissionContext) -> dict[str, Any]:
    hotspots = [
        item
        for item in list(mission.summary.get("kernel_hotspots") or [])
        if isinstance(item, dict)
    ]
    top = dict(hotspots[0] if hotspots else {})
    if not top:
        return {
            "status": "blocked",
            "reason": "kernel_hotspots_missing",
            "counterfactuals": [],
        }
    impact = float(top.get("impact_score", 0.0) or 0.0)
    recoverable = float(top.get("estimated_recoverable_gain_pct", 0.0) or 0.0)
    decode_share = float(top.get("pct_of_decode", 0.0) or 0.0)
    return {
        "status": "covered",
        "top_hotspot": top,
        "counterfactuals": [
            {
                "stage": str(top.get("kernel") or top.get("stage") or ""),
                "speedup_pct": 10.0,
                "estimated_e2e_gain_pct": round(max(0.0, decode_share * 0.10), 2),
                "priority": "high" if impact >= 0.5 else "medium",
            },
            {
                "stage": str(top.get("kernel") or top.get("stage") or ""),
                "speedup_pct": max(5.0, recoverable),
                "estimated_e2e_gain_pct": round(max(0.0, recoverable * 0.6), 2),
                "priority": "high" if recoverable >= 5.0 else "medium",
            },
        ],
    }


def _build_stage_graph_signature(
    mission: MissionContext, baseline_lineage: dict[str, Any]
) -> dict[str, Any]:
    signature = [
        {
            "model": str(item.get("model") or ""),
            "kernel": str(item.get("kernel") or item.get("stage") or ""),
            "cpp_file": str(item.get("cpp_file") or ""),
            "impact_score": round(float(item.get("impact_score", 0.0) or 0.0), 3),
        }
        for item in list(mission.summary.get("top_stage_hotspots") or [])[:10]
        if isinstance(item, dict)
    ]
    encoded = json.dumps(signature, sort_keys=True).encode("utf-8")
    signature_hash = __import__("hashlib").sha256(encoded).hexdigest()
    baseline = dict((baseline_lineage.get("baseline") or {}).get("summary") or {})
    baseline_signature = [
        {
            "model": str(item.get("model") or ""),
            "kernel": str(item.get("kernel") or item.get("stage") or ""),
            "cpp_file": str(item.get("cpp_file") or ""),
            "impact_score": round(float(item.get("impact_score", 0.0) or 0.0), 3),
        }
        for item in list(baseline.get("top_stage_hotspots") or [])[:10]
        if isinstance(item, dict)
    ]
    baseline_hash = (
        __import__("hashlib")
        .sha256(json.dumps(baseline_signature, sort_keys=True).encode("utf-8"))
        .hexdigest()
        if baseline_signature
        else ""
    )
    return {
        "status": "covered" if signature else "blocked",
        "signature_hash": signature_hash if signature else "",
        "baseline_signature_hash": baseline_hash,
        "changed": bool(
            signature and baseline_hash and signature_hash != baseline_hash
        ),
        "signature": signature,
    }


def _build_digital_twin(
    mission: MissionContext, topology_passport: dict[str, Any]
) -> dict[str, Any]:
    host_fingerprint = str(topology_passport.get("host_fingerprint") or "")
    history = []
    for path in _summary_candidates(mission.layout.root.parent, mission.run_id):
        payload = _load_json(path)
        if not payload:
            continue
        if host_fingerprint and not _same_host(payload, host_fingerprint):
            continue
        history.append(payload)
        if len(history) >= 8:
            break
    if not history:
        return {
            "status": "blocked",
            "reason": "same_host_history_missing",
            "models": [],
        }
    current_decode = _model_metric_map(mission.summary, "decode_tps_p50")
    models: list[dict[str, Any]] = []
    for model_name, current in current_decode.items():
        prior = [
            _model_metric_map(item, "decode_tps_p50").get(model_name, 0.0)
            for item in history
            if _model_metric_map(item, "decode_tps_p50").get(model_name, 0.0) > 0.0
        ]
        if not prior:
            continue
        expected = float(statistics.fmean(prior))
        models.append(
            {
                "model": model_name,
                "expected_decode_tps": expected,
                "observed_decode_tps": current,
                "delta_pct": (
                    ((current - expected) / expected * 100.0) if expected else 0.0
                ),
            }
        )
    return {"status": "covered", "history_count": len(history), "models": models}


def _build_scheduler_advice(mission: MissionContext) -> dict[str, Any]:
    quality_governance = dict(mission.summary.get("quality_governance") or {})
    issues = [
        str(item) for item in list(quality_governance.get("issues") or []) if str(item)
    ]
    recommendations: list[dict[str, Any]] = []
    if issues:
        recommendations.append(
            {
                "lane": "quality_eval",
                "priority": 1,
                "reason": "quality issues present",
                "informativeness": round(min(1.0, 0.4 + 0.1 * len(issues)), 2),
            }
        )
    if not _lane_coverage(mission, "thread_matrix"):
        recommendations.append(
            {
                "lane": "thread_matrix",
                "priority": 2,
                "reason": "thread perturbation coverage missing",
                "informativeness": 0.7,
            }
        )
    if not _lane_coverage(mission, "kernel_microbench"):
        recommendations.append(
            {
                "lane": "kernel_microbench",
                "priority": 3,
                "reason": "kernel pyramid layer missing",
                "informativeness": 0.65,
            }
        )
    return {
        "status": "covered",
        "recommendations": recommendations,
        "early_stop_recommended": not issues and len(mission.completed_lanes) >= 4,
    }


def _build_host_contract_simulation(
    topology_passport: dict[str, Any],
    variance_budget: dict[str, Any],
) -> dict[str, Any]:
    visible_threads = int(topology_passport.get("visible_threads", 0) or 0)
    return {
        "status": "covered" if visible_threads else "blocked",
        "simulations": [
            {
                "scenario": "perf_unavailable",
                "predicted_effect": "kernel_microbench degraded advisory mode",
                "closure_risk": "medium",
            },
            {
                "scenario": "reduced_visible_threads",
                "visible_threads": (
                    max(1, visible_threads // 2) if visible_threads else 0
                ),
                "predicted_effect": "thread-matrix and variance budget likely unstable",
                "closure_risk": (
                    "high"
                    if not (
                        (variance_budget.get("overall") or {}).get(
                            "within_budget", True
                        )
                    )
                    else "medium"
                ),
            },
        ],
    }


def _build_chaos_lab(
    mission: MissionContext, closure_result: dict[str, Any]
) -> dict[str, Any]:
    unresolved = list(closure_result.get("unresolved") or [])
    drills = [
        {
            "drill_id": "missing_perf_signal",
            "would_fail_closed": _lane_coverage(mission, "kernel_microbench"),
            "reason": "kernel microbench depends on perf capability for full fidelity",
        },
        {
            "drill_id": "missing_artifact_index",
            "would_fail_closed": True,
            "reason": "artifact completeness is a required closure question",
        },
    ]
    return {
        "status": "covered",
        "open_unresolved_count": len(unresolved),
        "drills": drills,
    }


def _build_benchmark_pyramid(mission: MissionContext) -> dict[str, Any]:
    top_hotspot = dict(
        (list(mission.summary.get("kernel_hotspots") or []) or [{}])[0] or {}
    )
    top_model = dict((list(mission.summary.get("models") or []) or [{}])[0] or {})
    coherence = dict(
        (
            _quality_records(dict(mission.summary.get("quality") or {}), "coherence")
            or [{}]
        )[0]
        or {}
    )
    if not top_hotspot:
        return {"status": "blocked", "cause_chain": []}
    return {
        "status": "covered",
        "cause_chain": [
            {
                "level": "kernel",
                "signal": str(
                    top_hotspot.get("kernel") or top_hotspot.get("stage") or ""
                ),
                "evidence": round(
                    float(top_hotspot.get("pct_of_decode", 0.0) or 0.0), 2
                ),
            },
            {
                "level": "system",
                "signal": str(top_model.get("model") or ""),
                "evidence": {
                    "decode_tps_p50": float(
                        top_model.get("decode_tps_p50", 0.0) or 0.0
                    ),
                    "ttft_ms_p95": float(top_model.get("ttft_ms_p95", 0.0) or 0.0),
                },
            },
            {
                "level": "quality",
                "signal": str(coherence.get("model") or top_model.get("model") or ""),
                "evidence": float(coherence.get("pass_rate", 0.0) or 0.0),
            },
        ],
    }


def _capture_semantic_impact(
    repo_root: Path, changed_files: list[str]
) -> dict[str, Any]:
    binary = repo_root / "venv" / "bin" / "saguaro"
    if not binary.exists():
        return {
            "status": "skipped",
            "reason": "missing ./venv/bin/saguaro",
            "recommendations": [],
        }
    recommendations: list[dict[str, Any]] = []
    for changed in changed_files[:3]:
        try:
            completed = subprocess.run(
                [str(binary), "impact", "--path", changed],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "skipped",
                "reason": "saguaro impact timeout",
                "recommendations": [],
            }
        if completed.returncode != 0:
            continue
        stdout = str(completed.stdout or "").strip()
        recommendations.append(
            {
                "path": changed,
                "impact_excerpt": stdout[:400],
                "recommended_lanes": _recommended_lanes_for_path(changed),
            }
        )
    if recommendations:
        return {"status": "covered", "recommendations": recommendations}
    return {
        "status": "fallback",
        "recommendations": [
            {
                "path": changed,
                "impact_excerpt": "",
                "recommended_lanes": _recommended_lanes_for_path(changed),
            }
            for changed in changed_files[:5]
        ],
    }


def _recommended_lanes_for_path(path: str) -> list[str]:
    lowered = str(path).lower()
    lanes = ["canonical_all_on"]
    if "memory" in lowered:
        lanes.append("memory_replay")
    if any(token in lowered for token in ("simd", "native", "kernel", "benchmark")):
        lanes.extend(["kernel_microbench", "thread_matrix"])
    if any(token in lowered for token in ("prompt", "quality", "eval")):
        lanes.append("quality_eval")
    return list(dict.fromkeys(lanes))


def build_advisory_bundle(
    mission: MissionContext,
    *,
    topology_passport: dict[str, Any],
    variance_budget: dict[str, Any],
    baseline_lineage: dict[str, Any],
    run_ledger: dict[str, Any],
    closure_result: dict[str, Any],
) -> dict[str, Any]:
    changed_files = list(
        (
            ((mission.assurance_plan or {}).get("context") or {}).get("changed_files")
            or []
        )
    )
    return {
        "schema_version": ADVISORY_BUNDLE_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "operator_control_room": _build_operator_control_room(
            mission, run_ledger=run_ledger, closure_result=closure_result
        ),
        "perturbation_matrix": _build_perturbation_matrix(mission),
        "causal_hotspot_lab": _build_causal_hotspot_lab(mission),
        "performance_digital_twin": _build_digital_twin(
            mission, topology_passport=topology_passport
        ),
        "self_calibrating_scheduler": _build_scheduler_advice(mission),
        "host_contract_simulator": _build_host_contract_simulation(
            topology_passport, variance_budget
        ),
        "chaos_lab": _build_chaos_lab(mission, closure_result),
        "stage_graph": _build_stage_graph_signature(mission, baseline_lineage),
        "benchmark_pyramid": _build_benchmark_pyramid(mission),
        "semantic_change_impact_gate": _capture_semantic_impact(
            mission.repo_root, changed_files
        ),
    }
