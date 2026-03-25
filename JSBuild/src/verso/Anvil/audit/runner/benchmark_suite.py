from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import statistics
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if os.getenv("ANVIL_BENCHMARK_SUITE_MODULE_BOOTSTRAPPED") != "1":
        env = os.environ.copy()
        env["ANVIL_BENCHMARK_SUITE_MODULE_BOOTSTRAPPED"] = "1"
        env["PYTHONPATH"] = (
            f"{repo_root}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else str(repo_root)
        )
        os.chdir(repo_root)
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "audit.runner.benchmark_suite", *sys.argv[1:]],
            env,
        )

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from audit.eval.native_logits import (
    NativeLogitScorer,
    evaluate_accuracy,
    evaluate_confidence,
    evaluate_perplexity,
    load_jsonl,
)
from audit.control_plane.ledger import reduce_runtime_state
from audit.control_plane.mission import (
    MissionNodeReceipt,
    compile_benchmark_mission_graph,
    dry_run_receipt,
)
from audit.control_plane import materialize_control_plane_artifacts
from audit.control_plane.topology import topology_hash_from_preflight
from audit.provenance.capture import capture_runtime_provenance
from audit.runtime_logging import SuiteEventLogger
from audit.runtime_logging import env_log_level
from audit.runtime_logging import env_ui_mode
from audit.runtime_logging import get_active_logger
from audit.runtime_logging import run_logged_subprocess
from audit.runtime_logging import set_active_logger
from audit.runner.common import load_optional_json_dict, read_json_dict
from audit.runner.assurance_control_plane import compile_assurance_plan
from audit.runner.assurance_control_plane import materialize_assurance_artifacts
from audit.runner import native_benchmark_runner as legacy_runner
from audit.runner.attempt_executor import AttemptSpec, execute_attempt
from audit.runner.suite_certification import (
    benchmark_harness_hash,
    load_tuning_contract,
    model_digest,
    write_tuning_contract,
)
from audit.runner.suite_preflight import run_preflight
from audit.runner.suite_profiles import (
    DEFAULT_ENABLED_LANES,
    BenchmarkSuiteSpec,
    compile_suite_profile,
    load_suite_profile,
)
from audit.store.schema_validation import validate_payload
from audit.store.suite_layout import (
    SuiteRunLayout,
    ensure_suite_layout,
    required_suite_artifacts,
    resolve_suite_layout,
)
from audit.store.writer import append_ndjson, read_ndjson, write_json_atomic
from core.model.chat_templates import resolve_prompt_contract
from core.native.runtime_telemetry import build_runtime_capability_ledger
from core.telemetry.black_box import BlackBoxRecorder
from saguaro.services.platform import EvidenceService, MetricsService

PROFILE_DIR = Path("audit/profiles")
DEFAULT_PROFILE = "silver"
CONTINUOUS_SCHEMA_VERSION = "continuous_qsg_benchmark.v1"
TERMINAL_STATES = {
    "completed_pass",
    "completed_fail",
    "failed_preflight",
    "interrupted_incomplete",
    "internal_error",
}
SUPPORTED_BENCHMARK_LANES = frozenset([*DEFAULT_ENABLED_LANES, "memory_replay"])
CALIBRATION_QUALITY_MINIMA = {
    "perplexity_max": 1_000_000.0,
    "coherence_pass_rate_min": 0.5,
    "accuracy_pass_rate_min": 0.25,
    "expected_calibration_error_max": 0.20,
}
QUALITY_GOVERNANCE_SCHEMA_VERSION = "native_qsg_suite.quality_governance.v1"
ACCEPTANCE_GOVERNANCE_SCHEMA_VERSION = "native_qsg_suite.acceptance_governance.v1"
COHERENCE_PRINTABLE_RATIO_MIN = 0.95
COHERENCE_REPEATED_8GRAM_RATIO_MAX = 0.2
SPECULATIVE_PREREQ_PHASE = "Phase 5"
NON_AR_PREREQ_PHASE = "Phase 8"
AR_GENERATION_MODES = frozenset({"ar_verify", "ar_recovery"})
CALIBRATION_MODES = frozenset({"probe", "search", "deep_search"})


def build_synthesis_governance_summary() -> dict[str, Any]:
    from benchmarks.synthesis_suite import SynthesisBenchmarkSuite

    return SynthesisBenchmarkSuite().run()


def _terminal(line: str) -> None:
    logger = get_active_logger()
    if logger is not None:
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="terminal",
            message=line,
        )
        return
    print(line, file=sys.stderr, flush=True)


def _parse_cpu_list(raw: str) -> set[int]:
    cpus: set[int] = set()
    text = str(raw or "").strip()
    if not text:
        return cpus
    for chunk in text.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start = int(start_text)
                end = int(end_text)
            except Exception:
                continue
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
            continue
        try:
            cpus.add(int(token))
        except Exception:
            continue
    return {cpu for cpu in cpus if cpu >= 0}


def _cpu_target_candidates() -> list[str]:
    candidates: list[str] = []
    env_target = str(os.getenv("ANVIL_SUITE_TARGET_CPUS", "")).strip()
    if env_target:
        candidates.append(env_target)
    for path in (
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/cpuset/cpuset.cpus",
        "/proc/self/status",
        "/sys/devices/system/cpu/online",
    ):
        try:
            raw = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if path.endswith("/status"):
            for line in raw.splitlines():
                if line.startswith("Cpus_allowed_list:"):
                    _, _, value = line.partition(":")
                    value = value.strip()
                    if value:
                        candidates.append(value)
                    break
            continue
        value = raw.strip()
        if value:
            candidates.append(value)
    logical = int(os.cpu_count() or 1)
    if logical > 0:
        candidates.append(f"0-{logical - 1}")
    return candidates


def _best_cpu_target() -> set[int]:
    best: set[int] = set()
    for raw in _cpu_target_candidates():
        parsed = _parse_cpu_list(raw)
        if len(parsed) > len(best):
            best = parsed
    return best


def _ensure_suite_affinity(spec: BenchmarkSuiteSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "repair_allowed": str(spec.affinity_policy or "").strip() != "certified_exact",
        "auto_expand_enabled": str(os.getenv("ANVIL_DISABLE_AUTO_AFFINITY_EXPAND", "0"))
        != "1",
        "attempted": False,
        "expanded": False,
        "repair_required": False,
        "before": [],
        "after": [],
        "target": [],
        "error": "",
    }
    if not payload["auto_expand_enabled"]:
        return payload
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        payload["error"] = "sched_affinity_unsupported"
        return payload
    try:
        before = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception as exc:
        payload["error"] = f"sched_getaffinity_failed:{type(exc).__name__}"
        return payload

    target_set = _best_cpu_target()
    target = sorted(int(cpu) for cpu in target_set)
    payload["before"] = before
    payload["target"] = target
    if not target:
        payload["error"] = "empty_affinity_target"
        payload["after"] = before
        return payload
    os.environ["ANVIL_SUITE_TARGET_CPUS"] = ",".join(str(cpu) for cpu in target)

    before_set = set(before)
    if target_set.issubset(before_set):
        payload["after"] = before
        return payload
    payload["repair_required"] = True
    if not bool(payload["repair_allowed"]):
        payload["after"] = before
        return payload

    payload["attempted"] = True
    try:
        os.sched_setaffinity(0, target_set)
        after = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception as exc:
        payload["error"] = f"sched_setaffinity_failed:{type(exc).__name__}:{exc}"
        payload["after"] = before
        return payload
    payload["after"] = after
    payload["expanded"] = len(after) > len(before)
    return payload


def _validate_execution_policy(spec: BenchmarkSuiteSpec) -> None:
    if int(spec.max_parallel_models) != 1:
        raise RuntimeError(
            "Native QSG benchmark suites must run models sequentially with `max_parallel_models: 1`."
        )
    if spec.affinity_policy not in {"repair_allowed", "certified_exact"}:
        raise RuntimeError(f"Unsupported suite affinity_policy: {spec.affinity_policy}")
    if spec.tuning_contract_policy not in {"optional", "required", "generate"}:
        raise RuntimeError(
            f"Unsupported suite tuning_contract_policy: {spec.tuning_contract_policy}"
        )
    unknown_lanes = [
        lane
        for lane in list(spec.enabled_lanes or [])
        if str(lane).strip() and str(lane) not in SUPPORTED_BENCHMARK_LANES
    ]
    if unknown_lanes:
        raise RuntimeError(
            f"Unsupported suite enabled_lanes: {', '.join(sorted(set(unknown_lanes)))}"
        )
    if spec.tuning_contract_policy != "generate" and "canonical_all_on" not in set(
        spec.enabled_lanes or []
    ):
        raise RuntimeError(
            "Benchmark suites must include `canonical_all_on` in enabled_lanes."
        )
    if spec.profile_name == "gold" and spec.tuning_contract_policy != "required":
        if any(item is None for item in list(spec.canonical_decode_threads or [])):
            raise RuntimeError(
                "Gold profile requires explicit canonical_decode_threads."
            )
        if any(item is None for item in list(spec.canonical_batch_threads or [])):
            raise RuntimeError(
                "Gold profile requires explicit canonical_batch_threads."
            )


def _enabled_benchmark_lanes(spec: BenchmarkSuiteSpec) -> list[str]:
    lanes = [
        str(lane)
        for lane in list(spec.enabled_lanes or DEFAULT_ENABLED_LANES)
        if str(lane).strip()
    ]
    deduped: list[str] = []
    seen: set[str] = set()
    for lane in lanes:
        if lane in seen:
            continue
        seen.add(lane)
        deduped.append(lane)
    return deduped


def _lane_enabled(spec: BenchmarkSuiteSpec, lane_id: str) -> bool:
    return lane_id in set(_enabled_benchmark_lanes(spec))


def _quality_lane_envs(spec: BenchmarkSuiteSpec) -> list[dict[str, Any]]:
    return [{"lane_id": "canonical_all_on", "ablation_id": "all_on", "env": {}}]


def _git_sha(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        return "nogit"
    return completed.stdout.strip() or "nogit"


def _default_run_id(repo_root: Path, spec: BenchmarkSuiteSpec) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime = capture_runtime_provenance(repo_root)
    host = str(((runtime.get("host") or {}).get("host_fingerprint")) or "unknown")
    git_sha = _git_sha(repo_root)
    return f"{timestamp}_{spec.profile_name}_{host}_{git_sha}"


def _profile_path(profile_name: str) -> Path:
    normalized = str(profile_name or "").strip().lower()
    if normalized == "smoke":
        normalized = "bronze"
    return PROFILE_DIR / f"native_qsg_{normalized}.yaml"


def _console_log(layout: SuiteRunLayout, line: str) -> None:
    layout.console_log.parent.mkdir(parents=True, exist_ok=True)
    with layout.console_log.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")
    logger = get_active_logger()
    if logger is not None:
        logger.emit(
            level="debug",
            source="benchmark_suite",
            event_type="console_log",
            message=line.rstrip(),
        )


def _write_json_artifact(
    *,
    layout: SuiteRunLayout,
    path: Path,
    payload: dict[str, Any],
    summary: str,
    phase: str | None = None,
    lane: str | None = None,
    attempt_id: str | None = None,
    model: str | None = None,
) -> None:
    write_json_atomic(path, payload)
    logger = get_active_logger()
    if logger is not None:
        logger.emit_artifact(
            source="benchmark_suite",
            kind=path.name,
            path=path,
            summary=summary,
            phase=phase,
            lane=lane,
            attempt_id=attempt_id,
            model=model,
        )


def _write_status(layout: SuiteRunLayout, **payload: Any) -> None:
    status = {
        "schema_version": "native_qsg_suite.status.v2",
        **payload,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_artifact(
        layout=layout,
        path=layout.suite_status_json,
        payload=status,
        summary=f"updated suite status state={status.get('state')}",
        phase=str(payload.get("state") or ""),
    )
    logger = get_active_logger()
    if logger is not None:
        logger.emit(
            level="debug",
            source="benchmark_suite",
            event_type="suite_state",
            message=f"state={status.get('state')}",
            phase=str(payload.get("state") or ""),
            payload={
                "state": str(status.get("state") or ""),
                "ok": bool(status.get("ok", True)),
                "artifact": layout.suite_status_json.relative_to(
                    layout.root
                ).as_posix(),
                "terminal_state": str(status.get("terminal_state") or ""),
                "run_exit_reason": str(status.get("run_exit_reason") or ""),
            },
        )


def _write_checkpoint(
    layout: SuiteRunLayout,
    *,
    completed_lanes: list[str],
    completed_attempt_ids: list[str],
    run_exit_reason: str | None = None,
    last_successful_lane: str | None = None,
) -> None:
    payload = {
        "schema_version": "native_qsg_suite.checkpoint.v2",
        "run_id": layout.run_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_lanes": completed_lanes,
        "completed_attempt_ids": completed_attempt_ids,
        "run_exit_reason": run_exit_reason,
        "last_successful_lane": last_successful_lane,
    }
    _write_json_artifact(
        layout=layout,
        path=layout.checkpoint_json,
        payload=payload,
        summary=(
            "updated checkpoint "
            f"lanes={len(completed_lanes)} attempts={len(completed_attempt_ids)}"
        ),
        phase=str(last_successful_lane or ""),
    )
    logger = get_active_logger()
    if logger is not None:
        logger.emit(
            level="debug",
            source="benchmark_suite",
            event_type="suite_checkpoint",
            message="updated reducer-compatible checkpoint",
            phase=str(last_successful_lane or ""),
            lane=str(last_successful_lane or ""),
            payload={
                "completed_lanes": list(completed_lanes),
                "completed_attempt_ids": list(completed_attempt_ids),
                "run_exit_reason": str(run_exit_reason or ""),
                "last_successful_lane": str(last_successful_lane or ""),
                "artifact": layout.checkpoint_json.relative_to(layout.root).as_posix(),
            },
        )


def _load_checkpoint(layout: SuiteRunLayout) -> dict[str, Any]:
    if layout.events_ndjson.exists():
        state = reduce_runtime_state(
            [row for row in read_ndjson(layout.events_ndjson) if isinstance(row, dict)]
        )
        if state.get("completed_lanes") or state.get("completed_attempt_ids"):
            return {
                "schema_version": "native_qsg_suite.checkpoint.v3",
                "run_id": layout.run_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "completed_lanes": list(state.get("completed_lanes") or []),
                "completed_attempt_ids": list(state.get("completed_attempt_ids") or []),
                "run_exit_reason": str(state.get("run_exit_reason") or ""),
                "last_successful_lane": state.get("last_successful_lane"),
                "terminal_state": str(state.get("terminal_state") or ""),
                "node_receipts": list(state.get("node_receipts") or []),
                "derived_from_ledger": True,
            }
    if not layout.checkpoint_json.exists():
        return {}
    return read_json_dict(layout.checkpoint_json)


def _emit_mission_receipt(
    *,
    node_id: str,
    phase: str,
    kind: str,
    status: str,
    blocking: bool,
    lane: str | None = None,
    attempt_id: str | None = None,
    model: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    logger = get_active_logger()
    if logger is None:
        return
    receipt = MissionNodeReceipt(
        node_id=node_id,
        phase=phase,
        kind=kind,
        status=status,
        blocking=blocking,
        lane=lane,
        attempt_id=attempt_id,
        model=model,
        details=details,
    )
    logger.emit(
        level=(
            "info"
            if status in {"completed", "dry_run"}
            else ("warn" if status == "failed" else "debug")
        ),
        source="benchmark_suite",
        event_type="mission_node_receipt",
        message=f"{node_id}:{status}",
        phase=phase,
        lane=str(lane or ""),
        attempt_id=str(attempt_id or ""),
        model=str(model or ""),
        payload=receipt.to_dict(),
    )


def _json_sha256(path: Path) -> str:
    payload = path.read_bytes()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _calibration_workload_digest(
    spec: BenchmarkSuiteSpec,
    *,
    target_profile: str,
) -> str:
    payload = {
        "profile_name": str(target_profile or spec.profile_name or ""),
        "models": list(spec.models),
        "canonical_max_new_tokens": int(spec.scenario_pack.canonical_max_new_tokens),
        "canonical_context_length": int(spec.scenario_pack.canonical_context_length),
        "continuous_concurrency": list(spec.scenario_pack.continuous_concurrency),
        "continuous_prompt_classes": list(spec.scenario_pack.continuous_prompt_classes),
        "continuous_scheduler_policies": list(
            spec.scenario_pack.continuous_scheduler_policies
        ),
        "calibration_surface": {
            "continuous_max_active_requests": list(
                spec.calibration_search.continuous_max_active_requests
            ),
            "continuous_batch_wait_timeout_ms": list(
                spec.calibration_search.continuous_batch_wait_timeout_ms
            ),
            "continuous_state_page_rows": list(
                spec.calibration_search.continuous_state_page_rows
            ),
            "continuous_max_prefill_rows_per_iteration": list(
                spec.calibration_search.continuous_max_prefill_rows_per_iteration
            ),
            "continuous_interleaved_streams": list(
                spec.calibration_search.continuous_interleaved_streams
            ),
        },
    }
    return f"sha256:{_stable_hash(payload)}"


def _prompt_contract_hash(models: list[str]) -> str:
    contracts = []
    for model in models:
        contract = resolve_prompt_contract(model, strict=False)
        contracts.append(
            {
                "model": model,
                "template_name": contract.template_name,
                "system_prompt": contract.system_prompt,
                "assistant_prefix": contract.assistant_prefix,
                "inject_system_prompt": contract.inject_system_prompt,
                "disallowed_output_prefixes": list(contract.disallowed_output_prefixes),
            }
        )
    return _stable_hash({"contracts": contracts})


def _feature_toggle_contract_hash(spec: BenchmarkSuiteSpec) -> str:
    return _stable_hash(
        {
            "all_on": _all_on_feature_toggles(),
            "lane_envs": _quality_lane_envs(spec),
            "ablations": [
                {
                    "ablation_id": item.ablation_id,
                    "env": dict(item.env),
                    "measured_runs": item.measured_runs,
                    "warmup_runs": item.warmup_runs,
                }
                for item in spec.ablations
            ],
        }
    )


def _memory_snapshot_hash(runtime_payload: dict[str, Any]) -> str:
    memory = dict(runtime_payload.get("memory") or {})
    return _stable_hash(memory)


def _build_metrics_rollup(summary: dict[str, Any]) -> dict[str, Any]:
    models = [
        item for item in list(summary.get("models") or []) if isinstance(item, dict)
    ]
    quality = dict(summary.get("quality") or {})

    confidence = [
        item for item in list(quality.get("confidence") or []) if isinstance(item, dict)
    ]
    perplexity = [
        item for item in list(quality.get("perplexity") or []) if isinstance(item, dict)
    ]
    coherence = [
        item for item in list(quality.get("coherence") or []) if isinstance(item, dict)
    ]
    accuracy = [
        item for item in list(quality.get("accuracy") or []) if isinstance(item, dict)
    ]
    confidence_by_model = {
        str(item.get("model") or ""): item
        for item in confidence
        if str(item.get("ablation_id") or "") == "all_on"
    }
    perplexity_by_model = {
        str(item.get("model") or ""): item
        for item in perplexity
        if str(item.get("ablation_id") or "") == "all_on"
    }
    coherence_by_model = {
        str(item.get("model") or ""): item
        for item in coherence
        if str(item.get("ablation_id") or "") == "all_on"
    }
    accuracy_by_model = {
        str(item.get("model") or ""): item
        for item in accuracy
        if str(item.get("ablation_id") or "") == "all_on"
    }

    return {
        "schema_version": "native_qsg_suite.metrics_rollup.v1",
        "run_id": summary.get("run_id"),
        "overall_pass": bool(summary.get("overall_pass", False)),
        "quality_governance_passed": bool(
            (summary.get("quality_governance") or {}).get("passed", False)
        ),
        "spc_status": str((summary.get("spc_report") or {}).get("status") or ""),
        "topology_hash": str(
            (summary.get("topology_passport") or {}).get("topology_hash") or ""
        ),
        "compare_to": str(
            (summary.get("baseline_lineage") or {}).get("comparator_mode") or ""
        ),
        "terminal_state": str(summary.get("terminal_state") or ""),
        "run_exit_reason": str(summary.get("run_exit_reason") or ""),
        "model_count": len(models),
        "quality_counts": {
            "confidence": len(confidence),
            "perplexity": len(perplexity),
            "coherence": len(coherence),
            "accuracy": len(accuracy),
        },
        "models": [
            {
                "model": str(item.get("model") or ""),
                "decode_tps_p50": float(item.get("decode_tps_p50", 0.0) or 0.0),
                "decode_tps_p95": float(item.get("decode_tps_p95", 0.0) or 0.0),
                "e2e_tps_p50": float(item.get("e2e_tps_p50", 0.0) or 0.0),
                "e2e_tps_p95": float(item.get("e2e_tps_p95", 0.0) or 0.0),
                "ttft_ms_p95": float(item.get("ttft_ms_p95", 0.0) or 0.0),
                "ttft_ms_p50": float(item.get("ttft_ms_p50", 0.0) or 0.0),
                "scheduler_queue_wait_ms_p95": float(
                    item.get("scheduler_queue_wait_ms_p95", 0.0) or 0.0
                ),
                "scheduler_iteration_ms_p95": float(
                    item.get("scheduler_iteration_ms_p95", 0.0) or 0.0
                ),
                "continuous_decode_tps_global_p50": float(
                    item.get("continuous_decode_tps_global_p50", 0.0) or 0.0
                ),
                "continuous_tpot_ms_p50": float(
                    item.get("continuous_tpot_ms_p50", 0.0) or 0.0
                ),
                "decode_time_accounted_pct": float(
                    item.get("decode_time_accounted_pct", 0.0) or 0.0
                ),
                "confidence_mean": float(
                    (confidence_by_model.get(str(item.get("model") or "")) or {}).get(
                        "mean_token_confidence",
                        0.0,
                    )
                    or 0.0
                ),
                "perplexity": float(
                    (perplexity_by_model.get(str(item.get("model") or "")) or {}).get(
                        "perplexity",
                        0.0,
                    )
                    or 0.0
                ),
                "coherence_pass_rate": float(
                    (coherence_by_model.get(str(item.get("model") or "")) or {}).get(
                        "pass_rate",
                        0.0,
                    )
                    or 0.0
                ),
                "accuracy_pass_rate": float(
                    (accuracy_by_model.get(str(item.get("model") or "")) or {}).get(
                        "pass_rate",
                        0.0,
                    )
                    or 0.0
                ),
                "accuracy_exact_match_rate": float(
                    (accuracy_by_model.get(str(item.get("model") or "")) or {}).get(
                        "exact_match_rate",
                        0.0,
                    )
                    or 0.0
                ),
                "pass": bool(item.get("pass", False)),
            }
            for item in models
        ],
    }


def _build_agent_handoff(summary: dict[str, Any]) -> dict[str, Any]:
    hotspots = [
        item
        for item in list(summary.get("kernel_hotspots") or [])
        if isinstance(item, dict)
    ]
    top_hotspots = hotspots[:10]
    next_hotspot = top_hotspots[0] if top_hotspots else {}
    return {
        "schema_version": "native_qsg_suite.agent_handoff.v1",
        "run_id": summary.get("run_id"),
        "overall_pass": bool(summary.get("overall_pass", False)),
        "failure_count": int(summary.get("failure_count", 0) or 0),
        "next_patch_target": {
            "kernel": str(next_hotspot.get("kernel") or ""),
            "cpp_file": str(next_hotspot.get("cpp_file") or ""),
            "cpp_function": str(next_hotspot.get("cpp_function") or ""),
            "evidence": {
                "impact_score": float(next_hotspot.get("impact_score", 0.0) or 0.0),
                "pct_of_decode": float(next_hotspot.get("pct_of_decode", 0.0) or 0.0),
                "cv_pct": float(next_hotspot.get("cv_pct", 0.0) or 0.0),
                "hotspot_confidence": float(
                    next_hotspot.get("hotspot_confidence", 0.0) or 0.0
                ),
                "estimated_recoverable_gain_pct": float(
                    next_hotspot.get("estimated_recoverable_gain_pct", 0.0) or 0.0
                ),
                "artifact_refs": dict(next_hotspot.get("artifact_refs") or {}),
            },
        },
        "top_hotspots": [
            {
                "model": str(item.get("model") or ""),
                "kernel": str(item.get("kernel") or ""),
                "cpp_file": str(item.get("cpp_file") or ""),
                "cpp_function": str(item.get("cpp_function") or ""),
                "impact_score": float(item.get("impact_score", 0.0) or 0.0),
                "pct_of_decode": float(item.get("pct_of_decode", 0.0) or 0.0),
                "hotspot_confidence": float(item.get("hotspot_confidence", 0.0) or 0.0),
                "estimated_recoverable_gain_pct": float(
                    item.get("estimated_recoverable_gain_pct", 0.0) or 0.0
                ),
                "why_hot": str(item.get("why_hot") or ""),
            }
            for item in top_hotspots
        ],
        "actionable_hotspot_bundle": [
            {
                "model": str(item.get("model") or ""),
                "kernel": str(item.get("kernel") or ""),
                "cpp_file": str(item.get("cpp_file") or ""),
                "cpp_function": str(item.get("cpp_function") or ""),
                "expected_recoverable_gain_pct": float(
                    item.get("estimated_recoverable_gain_pct", 0.0) or 0.0
                ),
                "hotspot_confidence": float(item.get("hotspot_confidence", 0.0) or 0.0),
                "evidence_refs": dict(item.get("artifact_refs") or {}),
            }
            for item in top_hotspots[:5]
        ],
        "control_plane": {
            "variance_within_budget": bool(
                ((summary.get("variance_budget") or {}).get("overall") or {}).get(
                    "within_budget", False
                )
            ),
            "spc_status": str((summary.get("spc_report") or {}).get("status") or ""),
            "compare_to": str(
                (summary.get("baseline_lineage") or {}).get("comparator_mode") or ""
            ),
        },
    }


def _write_artifact_index(layout: SuiteRunLayout) -> None:
    artifacts = []
    indexed_paths = list(required_suite_artifacts(layout))
    if layout.summary_failed_json.exists():
        indexed_paths.append(layout.summary_failed_json)
    for path in indexed_paths:
        if path == layout.index_json:
            continue
        if not path.exists():
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(layout.root)),
                "bytes": int(path.stat().st_size),
                "sha256": _json_sha256(path),
            }
        )
    payload = {
        "schema_version": "native_qsg_suite.index.v1",
        "run_id": layout.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    _write_json_artifact(
        layout=layout,
        path=layout.index_json,
        payload=payload,
        summary=f"indexed {len(artifacts)} artifacts",
        phase="finalizing",
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _thread_candidates(
    spec: BenchmarkSuiteSpec, runtime_payload: dict[str, Any]
) -> list[tuple[int | None, int | None, int]]:
    host = dict(runtime_payload.get("host") or {})
    logical = int(host.get("visible_threads", host.get("logical_cpus", 1)) or 1)
    physical = max(1, logical // 2) if logical >= 4 else logical
    decode_values = [None, max(1, physical - 2), physical]
    batch_values = [None, max(1, physical - 2), physical]
    matrix: list[tuple[int | None, int | None, int]] = []
    for decode_threads in decode_values:
        for batch_threads in batch_values:
            for ubatch in spec.scenario_pack.thread_matrix_ubatch:
                matrix.append((decode_threads, batch_threads, int(ubatch)))
    deduped: list[tuple[int | None, int | None, int]] = []
    seen: set[tuple[int | None, int | None, int]] = set()
    for item in matrix:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _canonical_thread_tuples(
    spec: BenchmarkSuiteSpec,
    runtime_payload: dict[str, Any],
) -> list[tuple[int | None, int | None, int]]:
    host = dict(runtime_payload.get("host") or {})
    logical = int(host.get("visible_threads", host.get("logical_cpus", 1)) or 1)
    host_default = max(1, logical)
    decode_values = [
        host_default if item is None else int(item)
        for item in list(spec.canonical_decode_threads or [None])
    ]
    batch_values = [
        host_default if item is None else int(item)
        for item in list(spec.canonical_batch_threads or [None])
    ]
    ubatch_values = list(spec.scenario_pack.thread_matrix_ubatch or [32])
    canonical_ubatch = int(ubatch_values[len(ubatch_values) // 2])
    candidates: list[tuple[int | None, int | None, int]] = []
    seen: set[tuple[int | None, int | None, int]] = set()
    for decode_threads in decode_values:
        for batch_threads in batch_values:
            item = (decode_threads, batch_threads, canonical_ubatch)
            if item in seen:
                continue
            seen.add(item)
            candidates.append(item)
    return candidates or [(host_default, host_default, canonical_ubatch)]


def _max_affinity_thread_tuple(
    spec: BenchmarkSuiteSpec,
    runtime_payload: dict[str, Any],
) -> tuple[int, int, int]:
    host = dict(runtime_payload.get("host") or {})
    logical = int(host.get("visible_threads", host.get("logical_cpus", 1)) or 1)
    ubatch_values = list(spec.scenario_pack.thread_matrix_ubatch or [32])
    canonical_ubatch = int(ubatch_values[len(ubatch_values) // 2])
    return (logical, logical, canonical_ubatch)


def _median_and_mad(values: list[float]) -> tuple[float, float]:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0, 0.0
    median = float(statistics.median(ordered))
    deviations = [abs(item - median) for item in ordered]
    mad = float(statistics.median(deviations)) if deviations else 0.0
    return median, mad


def _percentile_value(values: list[float], percentile: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    clamped = min(100.0, max(0.0, float(percentile)))
    position = (len(ordered) - 1) * (clamped / 100.0)
    lower_index = int(position)
    upper_index = min(len(ordered) - 1, lower_index + 1)
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return lower + (upper - lower) * fraction


def _float_list(values: list[Any]) -> list[float]:
    normalized: list[float] = []
    for value in values:
        try:
            normalized.append(float(value))
        except Exception:
            continue
    return normalized


def _runtime_metric(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    runtime = row.get("runtime")
    if isinstance(runtime, dict):
        return runtime.get(key)
    return None


def _attempt_metric_list(
    rows: list[dict[str, Any]], section: str, key: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        bucket = row.get(section)
        if isinstance(bucket, dict):
            raw = bucket.get(key)
        else:
            raw = row.get(key)
        try:
            values.append(float(raw))
        except Exception:
            continue
    return values


def _metric_summary(values: list[float]) -> dict[str, float]:
    normalized = _float_list(values)
    if not normalized:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": _percentile_value(normalized, 50.0),
        "p95": _percentile_value(normalized, 95.0),
    }


def _continuous_model_metrics(
    continuous_payload: list[dict[str, Any]], model: str
) -> dict[str, float]:
    results: list[dict[str, Any]] = []
    for item in continuous_payload:
        if str(item.get("model") or "") == model:
            results = [
                dict(row)
                for row in list(item.get("results") or [])
                if isinstance(row, dict)
            ]
            break
    if not results:
        return {}
    decode = _metric_summary(
        [float(row.get("decode_tps_global", 0.0) or 0.0) for row in results]
    )
    ttft_values = [float(row.get("ttft_ms_p95", 0.0) or 0.0) for row in results]
    ttft = _metric_summary(ttft_values)
    tpot = _metric_summary(
        [float(row.get("tpot_ms_p50", 0.0) or 0.0) for row in results]
    )
    queue_wait = _metric_summary(
        [float(row.get("queue_wait_ms_p95", 0.0) or 0.0) for row in results]
    )
    fairness_values = [float(row.get("fairness", 0.0) or 0.0) for row in results]
    return {
        "continuous_decode_tps_global_p50": decode["p50"],
        "continuous_decode_tps_global_p95": decode["p95"],
        "continuous_ttft_ms_p95_p50": ttft["p50"],
        "continuous_ttft_ms_p95_max": max(ttft_values) if ttft_values else 0.0,
        "continuous_tpot_ms_p50": tpot["p50"],
        "continuous_tpot_ms_p95": tpot["p95"],
        "continuous_queue_wait_ms_p95": queue_wait["p95"],
        "continuous_queue_wait_ms_p50": queue_wait["p50"],
        "continuous_fairness_min": min(fairness_values) if fairness_values else 0.0,
    }


def _augment_summary_with_quality_and_runtime(
    *,
    summary: dict[str, Any],
    quality_payload: dict[str, Any],
    continuous_payload: list[dict[str, Any]],
    attempt_rows: list[dict[str, Any]],
) -> None:
    measured_rows = [
        dict(row)
        for row in attempt_rows
        if isinstance(row, dict)
        and str(row.get("lane_id") or "") == "canonical_all_on"
        and str(row.get("ablation_id") or "") == "all_on"
        and not bool(row.get("warmup", False))
    ]
    quality_keys = ("perplexity", "confidence", "coherence", "accuracy")
    quality_index = {
        key: {
            str(item.get("model") or ""): dict(item)
            for item in list(quality_payload.get(key) or [])
            if isinstance(item, dict) and str(item.get("ablation_id") or "") == "all_on"
        }
        for key in quality_keys
    }
    for model_summary in list(summary.get("models") or []):
        if not isinstance(model_summary, dict):
            continue
        model_name = str(model_summary.get("model") or "")
        model_rows = [
            row for row in measured_rows if str(row.get("model_id") or "") == model_name
        ]
        if model_rows:
            wall = _metric_summary(
                _attempt_metric_list(model_rows, "throughput", "wall_tokens_per_second")
            )
            prefill = _metric_summary(
                _attempt_metric_list(model_rows, "throughput", "prefill_tps")
            )
            decode = _metric_summary(
                _attempt_metric_list(model_rows, "throughput", "decode_tps")
            )
            e2e = _metric_summary(
                _attempt_metric_list(model_rows, "throughput", "e2e_tps")
            )
            ttft = _metric_summary(
                _attempt_metric_list(model_rows, "latency", "ttft_ms")
            )
            model_summary["wall_tps_p50"] = wall["p50"]
            model_summary["wall_tps_p95"] = wall["p95"]
            model_summary["prefill_tps_p50"] = prefill["p50"]
            model_summary["prefill_tps_p95"] = prefill["p95"]
            model_summary["decode_tps_p50"] = decode["p50"]
            model_summary["decode_tps_p95"] = decode["p95"]
            model_summary["e2e_tps_p50"] = e2e["p50"]
            model_summary["e2e_tps_p95"] = e2e["p95"]
            model_summary["ttft_ms_p50"] = ttft["p50"]
            model_summary["ttft_ms_p95"] = ttft["p95"]

            for field_name, summary_key in (
                ("p10_ms", "per_token_latency_p10_ms"),
                ("p25_ms", "per_token_latency_p25_ms"),
                ("p50_ms", "per_token_latency_p50_ms"),
                ("p75_ms", "per_token_latency_p75_ms"),
                ("p95_ms", "per_token_latency_p95_ms"),
                ("p99_ms", "per_token_latency_p99_ms"),
                ("stddev_ms", "per_token_latency_stddev_ms"),
                ("min_ms", "per_token_latency_min_ms"),
                ("max_ms", "per_token_latency_max_ms"),
            ):
                values = _attempt_metric_list(model_rows, "latency", field_name)
                model_summary[summary_key] = (
                    _percentile_value(values, 50.0) if values else 0.0
                )

            queue_wait = _metric_summary(
                _float_list(
                    [
                        _runtime_metric(row, "scheduler_queue_wait_ms")
                        for row in model_rows
                    ]
                )
            )
            iteration = _metric_summary(
                _float_list(
                    [
                        _runtime_metric(row, "scheduler_iteration_ms")
                        for row in model_rows
                    ]
                )
            )
            model_summary["scheduler_queue_wait_ms_p50"] = queue_wait["p50"]
            model_summary["scheduler_queue_wait_ms_p95"] = queue_wait["p95"]
            model_summary["scheduler_iteration_ms_p50"] = iteration["p50"]
            model_summary["scheduler_iteration_ms_p95"] = iteration["p95"]

            stage_metrics = {}
            for stage_key in (
                "graph_prefill_avg_ms",
                "graph_decode_avg_ms",
                "sample_avg_ms",
                "logits_processor_avg_ms",
                "penalty_avg_ms",
                "suppression_avg_ms",
            ):
                values = _float_list(
                    [_runtime_metric(row, stage_key) for row in model_rows]
                )
                if values:
                    stage_metrics[stage_key] = _percentile_value(values, 50.0)
            if stage_metrics:
                model_summary["runtime_stage_avg_ms"] = stage_metrics

            proposed_parallel = sum(
                int(row.get("accepted_parallel_tokens", 0) or 0)
                + int(row.get("rejected_parallel_tokens", 0) or 0)
                for row in model_rows
            )
            accepted_parallel = sum(
                int(row.get("accepted_parallel_tokens", 0) or 0) for row in model_rows
            )
            model_summary["accepted_parallel_tokens_total"] = accepted_parallel
            model_summary["proposed_parallel_tokens_total"] = proposed_parallel
            model_summary["parallel_acceptance_rate"] = (
                float(accepted_parallel) / float(proposed_parallel)
                if proposed_parallel
                else 0.0
            )

        continuous_metrics = _continuous_model_metrics(continuous_payload, model_name)
        model_summary.update(continuous_metrics)

        confidence = quality_index["confidence"].get(model_name, {})
        coherence = quality_index["coherence"].get(model_name, {})
        perplexity = quality_index["perplexity"].get(model_name, {})
        accuracy = quality_index["accuracy"].get(model_name, {})
        if perplexity:
            model_summary["perplexity"] = float(
                perplexity.get("perplexity", 0.0) or 0.0
            )
        if confidence:
            model_summary["confidence_mean"] = float(
                confidence.get("mean_token_confidence", 0.0) or 0.0
            )
            model_summary["confidence_p95"] = float(
                confidence.get("p95_token_confidence", 0.0) or 0.0
            )
            model_summary["expected_calibration_error"] = float(
                confidence.get("expected_calibration_error", 0.0) or 0.0
            )
        if coherence:
            model_summary["coherence_pass_rate"] = float(
                coherence.get("pass_rate", 0.0) or 0.0
            )
        if accuracy:
            model_summary["accuracy_pass_rate"] = float(
                accuracy.get("pass_rate", 0.0) or 0.0
            )
            model_summary["accuracy_exact_match_rate"] = float(
                accuracy.get("exact_match_rate", 0.0) or 0.0
            )
            model_summary["accuracy_contains_match_rate"] = float(
                accuracy.get("contains_match_rate", 0.0) or 0.0
            )
            model_summary["accuracy_option_match_rate"] = float(
                accuracy.get("option_match_rate", 0.0) or 0.0
            )


def _calibration_seed_candidates(
    max_threads: int,
    ubatches: list[int],
) -> list[tuple[int, int, int]]:
    normalized_ubatches = sorted(
        {int(ubatch) for ubatch in ubatches if int(ubatch) > 0}
    )
    if not normalized_ubatches:
        normalized_ubatches = [32]
    canonical_ubatch = min(
        normalized_ubatches, key=lambda value: (abs(value - 32), value)
    )
    seeds = list(
        dict.fromkeys(
            max(1, candidate)
            for candidate in (
                (max_threads + 1) // 2,
                (3 * max_threads + 3) // 4,
                max_threads,
            )
        )
    )
    asymmetry_step = max(1, min(4, max_threads // 4))
    candidates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    def _append(item: tuple[int, int, int]) -> None:
        if item in seen:
            return
        seen.add(item)
        candidates.append(item)

    for threads in seeds:
        for ubatch in normalized_ubatches:
            _append((int(threads), int(threads), int(ubatch)))

    decode_heavy = (
        int(max_threads),
        max(1, int(max_threads) - asymmetry_step),
        int(canonical_ubatch),
    )
    batch_heavy = (
        max(1, int(max_threads) - asymmetry_step),
        int(max_threads),
        int(canonical_ubatch),
    )
    _append(decode_heavy)
    _append(batch_heavy)
    return candidates


def _calibration_refined_candidates(
    leaders: list[tuple[int, int, int]],
    *,
    max_threads: int,
) -> list[tuple[int, int, int]]:
    candidates: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    step = 2 if max_threads >= 8 else 1
    deltas = (
        (-step, 0),
        (step, 0),
        (0, -step),
        (0, step),
        (-step, -step),
        (step, step),
    )
    for decode_threads, batch_threads, ubatch in leaders[:2]:
        for decode_delta, batch_delta in deltas:
            item = (
                min(max_threads, max(1, int(decode_threads) + decode_delta)),
                min(max_threads, max(1, int(batch_threads) + batch_delta)),
                int(ubatch),
            )
            if item in seen:
                continue
            seen.add(item)
            candidates.append(item)
    return candidates


def _calibration_budget_settings(
    spec: BenchmarkSuiteSpec,
    mode: str,
) -> dict[str, Any]:
    normalized = str(mode or "search").strip().lower()
    if normalized not in CALIBRATION_MODES:
        normalized = "search"
    base = {
        "mode": normalized,
        "stage1_warmup_runs": 0,
        "stage1_measured_runs": 1,
        "stage2_measured_runs": 1,
        "stage1_top_k": max(1, int(spec.calibration_search.stage1_top_k or 1)),
        "stage2_top_k": max(1, int(spec.calibration_search.stage2_top_k or 1)),
        "scheduler_top_k": max(1, int(spec.calibration_search.scheduler_top_k or 1)),
        "max_scheduler_candidates": max(
            1, int(spec.calibration_search.max_scheduler_candidates or 1)
        ),
        "quality_sample_limit": int(spec.quality_eval.max_samples_per_family or 0),
        "kernel_iterations": int(spec.kernel_microbench.iterations or 0),
    }
    if normalized == "probe":
        base.update(
            {
                "stage2_measured_runs": 0,
                "stage1_top_k": 1,
                "stage2_top_k": 1,
                "scheduler_top_k": 1,
                "max_scheduler_candidates": min(4, base["max_scheduler_candidates"]),
                "quality_sample_limit": max(2, base["quality_sample_limit"] or 0),
                "kernel_iterations": int(
                    spec.calibration_search.probe_kernel_iterations or 0
                ),
            }
        )
    elif normalized == "deep_search":
        base.update(
            {
                "stage2_measured_runs": 2,
                "stage1_top_k": max(base["stage1_top_k"], 4),
                "stage2_top_k": max(base["stage2_top_k"], 3),
                "scheduler_top_k": max(base["scheduler_top_k"], 3),
                "max_scheduler_candidates": max(base["max_scheduler_candidates"], 24),
                "kernel_iterations": max(
                    int(spec.calibration_search.deep_search_kernel_iterations or 0),
                    int(spec.kernel_microbench.iterations or 0),
                ),
            }
        )
    else:
        base.update(
            {
                "scheduler_top_k": max(base["scheduler_top_k"], 2),
                "max_scheduler_candidates": max(base["max_scheduler_candidates"], 12),
                "kernel_iterations": max(
                    int(spec.calibration_search.search_kernel_iterations or 0), 0
                ),
            }
        )
    return base


def _calibration_quality_spec(
    spec: BenchmarkSuiteSpec,
    *,
    sample_limit: int,
) -> BenchmarkSuiteSpec:
    quality_eval = replace(
        spec.quality_eval,
        max_samples_per_family=(
            max(
                int(sample_limit or 0),
                int(spec.quality_eval.max_samples_per_family or 0),
            )
            if int(spec.quality_eval.max_samples_per_family or 0) > 0
            else int(sample_limit or 0)
        ),
    )
    return replace(spec, quality_eval=quality_eval)


def _continuous_surface_values(
    spec: BenchmarkSuiteSpec,
    *,
    mode: str,
) -> dict[str, list[Any]]:
    budget = _calibration_budget_settings(spec, mode)
    active_requests = list(spec.calibration_search.continuous_max_active_requests or [])
    if not active_requests:
        active_requests = list(spec.scenario_pack.continuous_concurrency or [1, 2, 4])
    surface = {
        "max_active_requests": list(
            dict.fromkeys(max(1, int(item)) for item in active_requests)
        ),
        "batch_wait_timeout_ms": list(
            dict.fromkeys(
                max(1, int(item))
                for item in list(
                    spec.calibration_search.continuous_batch_wait_timeout_ms
                    or [1, 2, 4]
                )
            )
        ),
        "state_page_rows": list(
            dict.fromkeys(
                max(1, int(item))
                for item in list(
                    spec.calibration_search.continuous_state_page_rows or [64, 128, 256]
                )
            )
        ),
        "max_prefill_rows_per_iteration": list(
            dict.fromkeys(
                max(1, int(item))
                for item in list(
                    spec.calibration_search.continuous_max_prefill_rows_per_iteration
                    or [512, 1024]
                )
            )
        ),
        "continuous_interleaved_streams": list(
            dict.fromkeys(
                bool(item)
                for item in list(
                    spec.calibration_search.continuous_interleaved_streams or [False]
                )
            )
        ),
    }
    if str(mode or "").strip().lower() == "probe":
        for key in (
            "max_active_requests",
            "batch_wait_timeout_ms",
            "state_page_rows",
            "max_prefill_rows_per_iteration",
        ):
            surface[key] = surface[key][:1]
        surface["continuous_interleaved_streams"] = surface[
            "continuous_interleaved_streams"
        ][:1]
    max_candidates = int(budget["max_scheduler_candidates"])
    candidate_count = (
        len(surface["max_active_requests"])
        * len(surface["batch_wait_timeout_ms"])
        * len(surface["state_page_rows"])
        * len(surface["max_prefill_rows_per_iteration"])
        * len(surface["continuous_interleaved_streams"])
        * max(1, len(spec.scenario_pack.continuous_scheduler_policies))
        * max(1, len(spec.scenario_pack.continuous_prompt_classes))
        * max(1, len(spec.scenario_pack.continuous_concurrency))
    )
    while (
        candidate_count > max_candidates
        and len(surface["continuous_interleaved_streams"]) > 1
    ):
        surface["continuous_interleaved_streams"] = surface[
            "continuous_interleaved_streams"
        ][:1]
        candidate_count = max_candidates
    return surface


def _scheduler_candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in list(payload.get("results") or []):
        if not isinstance(item, dict):
            continue
        fairness = float(item.get("fairness", 0.0) or 0.0)
        decode_tps_global = float(item.get("decode_tps_global", 0.0) or 0.0)
        rows.append(
            {
                "scheduler_policy": str(item.get("scheduler_policy") or ""),
                "concurrency": int(item.get("concurrency", 0) or 0),
                "max_active_requests": int(item.get("max_active_requests", 0) or 0),
                "batch_wait_timeout_ms": int(item.get("batch_wait_timeout_ms", 0) or 0),
                "state_page_rows": int(item.get("state_page_rows", 0) or 0),
                "max_prefill_rows_per_iteration": int(
                    item.get("max_prefill_rows_per_iteration", 0) or 0
                ),
                "continuous_interleaved_streams": bool(
                    item.get("continuous_interleaved_streams", False)
                ),
                "ttft_ms_p95": float(item.get("ttft_ms_p95", 0.0) or 0.0),
                "tpot_ms_p50": float(item.get("tpot_ms_p50", 0.0) or 0.0),
                "queue_wait_ms_p95": float(item.get("queue_wait_ms_p95", 0.0) or 0.0),
                "scheduler_iteration_ms_p95": float(
                    item.get("scheduler_iteration_ms_p95", 0.0) or 0.0
                ),
                "fairness": fairness,
                "decode_tps_global": decode_tps_global,
                "decode_goodput_tps": float(
                    item.get("decode_goodput_tps", decode_tps_global * fairness) or 0.0
                ),
                "state_fragmentation_ratio": float(
                    item.get("state_fragmentation_ratio", 0.0) or 0.0
                ),
                "drift_overhead_percent": float(
                    item.get("drift_overhead_percent", 0.0) or 0.0
                ),
                "continuous_metrics": dict(item.get("continuous_metrics") or {}),
            }
        )
    return rows


def _select_scheduler_candidates(
    spec: BenchmarkSuiteSpec,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fairness_floor = float(spec.calibration_search.fairness_floor or 0.0)
    queue_wait_ceiling = float(
        spec.calibration_search.queue_wait_p95_ceiling_ms or float("inf")
    )
    ranked = sorted(
        rows,
        key=lambda item: (
            float(item.get("fairness", 0.0) or 0.0) < fairness_floor,
            float(item.get("queue_wait_ms_p95", 0.0) or 0.0) > queue_wait_ceiling,
            -float(item.get("decode_goodput_tps", 0.0) or 0.0),
            float(item.get("ttft_ms_p95", 0.0) or 0.0),
            float(item.get("state_fragmentation_ratio", 0.0) or 0.0),
            float(item.get("drift_overhead_percent", 0.0) or 0.0),
        ),
    )
    return ranked[: max(1, int(spec.calibration_search.scheduler_top_k or 1))]


def _kernel_hotspot_metrics(
    *,
    model: str,
    kernel_summary: dict[str, Any],
) -> dict[str, Any]:
    for payload in list(kernel_summary.get("models") or []):
        if str(payload.get("model") or "") != str(model):
            continue
        runs = [
            dict(item)
            for item in list(payload.get("runs") or [])
            if isinstance(item, dict)
        ]
        ranked = sorted(
            runs,
            key=lambda item: (
                -float(item.get("estimated_recoverable_gain_pct", 0.0) or 0.0),
                float(item.get("cv_pct", 0.0) or 0.0),
            ),
        )
        top = ranked[:3]
        return {
            "top_hotspots": top,
            "hotspot_penalty": sum(
                float(item.get("estimated_recoverable_gain_pct", 0.0) or 0.0)
                for item in top
            ),
        }
    return {"top_hotspots": [], "hotspot_penalty": 0.0}


def _candidate_metrics(
    *,
    attempt_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    lane_id: str,
    model: str,
    candidate: tuple[int, int, int],
) -> dict[str, Any]:
    decode_threads, batch_threads, ubatch = candidate
    rows = [
        row
        for row in attempt_rows
        if str(row.get("lane_id") or "") == lane_id
        and str(row.get("model_id") or "") == model
        and not bool(row.get("warmup", False))
        and int((row.get("thread_config") or {}).get("decode_threads") or 0)
        == int(decode_threads)
        and int((row.get("thread_config") or {}).get("batch_threads") or 0)
        == int(batch_threads)
        and int((row.get("thread_config") or {}).get("ubatch") or 0) == int(ubatch)
    ]
    failures = [
        row
        for row in failure_rows
        if str(row.get("attempt_id") or "")
        in {str(item.get("attempt_id") or "") for item in rows}
    ]
    decode_values = [
        float((row.get("throughput") or {}).get("decode_tps", 0.0) or 0.0)
        for row in rows
    ]
    ttft_values = [
        float((row.get("latency") or {}).get("ttft_ms", 0.0) or 0.0) for row in rows
    ]
    decode_median, decode_mad = _median_and_mad(decode_values)
    ttft_median, ttft_mad = _median_and_mad(ttft_values)
    decode_mean = float(statistics.fmean(decode_values)) if decode_values else 0.0
    decode_stdev = (
        float(statistics.stdev(decode_values)) if len(decode_values) >= 2 else 0.0
    )
    decode_cv_pct = ((decode_stdev / decode_mean) * 100.0) if decode_mean > 0.0 else 0.0

    kernel_series: dict[str, list[float]] = {}
    for row in rows:
        runtime = dict(row.get("runtime") or {})
        stage_ms = dict(runtime.get("graph_stage_ms") or {})
        for stage_name, value in stage_ms.items():
            kernel_series.setdefault(str(stage_name), []).append(float(value or 0.0))
    kernel_summary = {
        stage_name: {
            "median_ms": _median_and_mad(values)[0],
            "mad_ms": _median_and_mad(values)[1],
        }
        for stage_name, values in sorted(kernel_series.items())
        if values
    }

    pmu_series: dict[str, list[float]] = {}
    observed_runs = 0
    for row in rows:
        measurement = dict(row.get("measurement") or {})
        pmu = dict(measurement.get("pmu") or {})
        if bool(pmu.get("observed", False)):
            observed_runs += 1
        for source_key, target_key in (
            ("cycles", "cycles"),
            ("instructions", "instructions"),
            ("ipc", "ipc"),
            ("cache_miss_rate", "cache_miss_rate"),
            ("context_switches", "context_switches"),
            ("cpu_migrations", "cpu_migrations"),
            ("page_faults", "page_faults"),
        ):
            value = pmu.get(source_key)
            if value is None:
                continue
            pmu_series.setdefault(target_key, []).append(float(value))
    pmu_summary: dict[str, float | int] = {"observed_runs": int(observed_runs)}
    for metric_name, values in sorted(pmu_series.items()):
        if not values:
            continue
        median, mad = _median_and_mad(values)
        pmu_summary[f"{metric_name}_median"] = median
        pmu_summary[f"{metric_name}_mad"] = mad

    coherence_values = [
        1.0 if bool((row.get("coherence") or {}).get("ok", False)) else 0.0
        for row in rows
    ]
    coherence_median, coherence_mad = _median_and_mad(coherence_values)
    accepted_parallel_tokens_total = sum(
        int(row.get("accepted_parallel_tokens", 0) or 0) for row in rows
    )
    rejected_parallel_tokens_total = sum(
        int(row.get("rejected_parallel_tokens", 0) or 0) for row in rows
    )
    proposed_parallel_tokens_total = sum(
        int(row.get("proposed_parallel_tokens", 0) or 0) for row in rows
    )
    speculative_attempts = sum(
        1
        for row in rows
        if any(
            int(row.get(metric_name, 0) or 0) > 0
            for metric_name in (
                "accepted_parallel_tokens",
                "rejected_parallel_tokens",
                "proposed_parallel_tokens",
            )
        )
    )
    draft_acceptance_ratio = (
        float(accepted_parallel_tokens_total) / float(proposed_parallel_tokens_total)
        if proposed_parallel_tokens_total > 0
        else 0.0
    )
    queue_wait_values = [
        float(_runtime_metric(row, "scheduler_queue_wait_ms") or 0.0) for row in rows
    ]
    iteration_values = [
        float(_runtime_metric(row, "scheduler_iteration_ms") or 0.0) for row in rows
    ]
    ok = bool(rows) and not failures and decode_cv_pct <= 5.0
    return {
        "candidate": {
            "decode_threads": int(decode_threads),
            "batch_threads": int(batch_threads),
            "ubatch": int(ubatch),
        },
        "rows": rows,
        "failure_count": len(failures),
        "decode_tps_median": decode_median,
        "decode_tps_mad": decode_mad,
        "ttft_ms_median": ttft_median,
        "ttft_ms_mad": ttft_mad,
        "decode_tps_cv_pct": decode_cv_pct,
        "kernel_summary": kernel_summary,
        "pmu_summary": pmu_summary,
        "quality_summary": {
            "coherence_ok_median": coherence_median,
            "coherence_ok_mad": coherence_mad,
        },
        "queue_wait_ms_p95": _percentile_value(queue_wait_values, 0.95),
        "scheduler_iteration_ms_p95": _percentile_value(iteration_values, 0.95),
        "speculative_summary": {
            "accepted_parallel_tokens_total": int(accepted_parallel_tokens_total),
            "rejected_parallel_tokens_total": int(rejected_parallel_tokens_total),
            "proposed_parallel_tokens_total": int(proposed_parallel_tokens_total),
            "draft_acceptance_ratio": float(draft_acceptance_ratio),
            "speculative_attempts": int(speculative_attempts),
        },
        "accepted_parallel_tokens_total": int(accepted_parallel_tokens_total),
        "rejected_parallel_tokens_total": int(rejected_parallel_tokens_total),
        "proposed_parallel_tokens_total": int(proposed_parallel_tokens_total),
        "draft_acceptance_ratio": float(draft_acceptance_ratio),
        "speculative_attempts": int(speculative_attempts),
        "ok": ok,
    }


def _rank_calibration_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def _pmu_value(item: dict[str, Any], key: str, default: float) -> float:
        pmu = dict(item.get("pmu_summary") or {})
        value = pmu.get(key)
        if value is None:
            return default
        return float(value)

    return sorted(
        candidates,
        key=lambda item: (
            not bool(item.get("ok", False)),
            bool(item.get("safety_envelope_breached", False)),
            not bool(item.get("quality_constraints_pass", True)),
            float(item.get("fairness", 1.0) or 1.0)
            < float(item.get("fairness_floor", 0.0) or 0.0),
            float(item.get("queue_wait_ms_p95", 0.0) or 0.0)
            > float(
                item.get("queue_wait_ms_p95_ceiling", float("inf")) or float("inf")
            ),
            -float(
                item.get(
                    "decode_goodput_tps",
                    item.get("decode_tps_median", 0.0),
                )
                or 0.0
            ),
            -float(item.get("decode_tps_median", 0.0) or 0.0),
            float(item.get("ttft_ms_p95", item.get("ttft_ms_median", 0.0)) or 0.0),
            float(item.get("decode_tps_cv_pct", 0.0) or 0.0),
            -float(item.get("draft_acceptance_ratio", 0.0) or 0.0),
            -int(item.get("accepted_parallel_tokens_total", 0) or 0),
            float(item.get("hotspot_penalty", 0.0) or 0.0),
            not bool((item.get("pmu_summary") or {}).get("observed_runs", 0)),
            -_pmu_value(item, "ipc_median", -1.0),
            _pmu_value(item, "cpu_migrations_median", float("inf")),
            _pmu_value(item, "context_switches_median", float("inf")),
            float(item.get("ttft_ms_median", 0.0) or 0.0),
        ),
    )


def _model_thread_contracts(
    *,
    repo_root: Path,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
) -> dict[str, tuple[int, int, int]]:
    contracts = dict(preflight_payload.get("tuning_contracts") or {})
    if spec.tuning_contract_policy != "required":
        return {}
    host_contract = dict(preflight_payload.get("host_contract") or {})
    fingerprint = str(host_contract.get("host_fingerprint") or "").strip()
    resolved: dict[str, tuple[int, int, int]] = {}
    for model in spec.models:
        contract_info = dict(contracts.get(model) or {})
        if str(contract_info.get("readiness_state") or "").strip() != "ready":
            continue
        thread_config = dict(contract_info.get("thread_config") or {})
        if not thread_config and fingerprint:
            payload, _ = load_tuning_contract(repo_root, fingerprint, model)
            if not payload:
                continue
            if str(payload.get("readiness_state") or "ready").strip() != "ready":
                continue
            thread_config = dict((payload or {}).get("thread_config") or {})
        resolved[model] = (
            int(thread_config.get("decode_threads") or 0),
            int(thread_config.get("batch_threads") or 0),
            int(thread_config.get("ubatch") or 0),
        )
    return resolved


def _model_continuous_contracts(
    *,
    repo_root: Path,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    contracts = dict(preflight_payload.get("tuning_contracts") or {})
    if spec.tuning_contract_policy != "required":
        return {}
    host_contract = dict(preflight_payload.get("host_contract") or {})
    fingerprint = str(host_contract.get("host_fingerprint") or "").strip()
    resolved: dict[str, dict[str, Any]] = {}
    for model in spec.models:
        contract_info = dict(contracts.get(model) or {})
        if str(contract_info.get("readiness_state") or "").strip() != "ready":
            continue
        continuous_config = dict(contract_info.get("continuous_config") or {})
        pager_config = dict(contract_info.get("pager_config") or {})
        if (not continuous_config or not pager_config) and fingerprint:
            payload, _ = load_tuning_contract(repo_root, fingerprint, model)
            if not payload:
                continue
            if str(payload.get("readiness_state") or "ready").strip() != "ready":
                continue
            continuous_config = dict((payload or {}).get("continuous_config") or {})
            pager_config = dict((payload or {}).get("pager_config") or {})
        if continuous_config or pager_config:
            resolved[model] = {
                "continuous_config": continuous_config,
                "pager_config": pager_config,
            }
    return resolved


def _continuous_thread_tuple(
    *,
    spec: BenchmarkSuiteSpec,
    runtime_payload: dict[str, Any],
    model: str,
    model_thread_overrides: dict[str, tuple[int, int, int]] | None,
) -> tuple[int, int, int]:
    if model_thread_overrides and model in model_thread_overrides:
        override = tuple(int(item) for item in model_thread_overrides[model])
        if all(item > 0 for item in override):
            return override
    decode_threads, batch_threads, ubatch = _canonical_thread_tuples(
        spec, runtime_payload
    )[0]
    return int(decode_threads or 0), int(batch_threads or 0), int(ubatch or 0)


def _all_on_feature_toggles() -> dict[str, bool]:
    return {
        "timecrystal": True,
        "grover": True,
        "coconut": True,
    }


def _feature_toggles_for_env(env: dict[str, str]) -> dict[str, bool]:
    toggles = _all_on_feature_toggles()
    if env.get("ANVIL_TC_STABILIZER") == "0":
        toggles["timecrystal"] = False
    if env.get("ANVIL_NATIVE_QSG_USE_GROVER") == "0":
        toggles["grover"] = False
    if env.get("ANVIL_NATIVE_QSG_USE_COCONUT") == "0":
        toggles["coconut"] = False
    return toggles


def _record_attempt(
    *,
    layout: SuiteRunLayout,
    spec: AttemptSpec,
    repo_root: Path,
    attempt_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    completed_attempt_ids: set[str],
) -> None:
    logger = get_active_logger()
    if logger is not None:
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="attempt_start",
            message=f"running attempt {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            payload={
                "thread_tuple": (
                    f"{spec.decode_threads or 'auto'}x{spec.batch_threads or 'auto'}x"
                    f"{spec.ubatch or 'auto'}"
                ),
                "phase": "attempt",
                "lane": str(spec.lane_id or ""),
                "attempt_id": spec.attempt_id,
                "model": spec.model,
            },
        )
    attempt_record, phases, _ = execute_attempt(spec, repo_root=repo_root)
    attempt_record["run_id"] = layout.run_id
    validate_payload("attempt_record.schema.json", attempt_record)
    append_ndjson(layout.native_attempts_ndjson, attempt_record)
    attempt_rows.append(attempt_record)
    for phase in phases:
        phase["run_id"] = layout.run_id
        append_ndjson(layout.native_phases_ndjson, phase)
    if not bool((attempt_record.get("status") or {}).get("ok", False)):
        failure = {
            "run_id": layout.run_id,
            "attempt_id": spec.attempt_id,
            "model": spec.model,
            "error": "attempt_failed",
            "error_type": "GateFailure",
            "failure_kind": "gate_failure",
            "gate_issues": list(
                (attempt_record.get("status") or {}).get("issues") or []
            ),
            "normalized_issues": list(
                (attempt_record.get("status") or {}).get("issues") or []
            ),
            "traceback": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        append_ndjson(layout.native_failures_ndjson, failure)
        failure_rows.append(failure)
        if logger is not None:
            logger.emit(
                level="warn",
                source="benchmark_suite",
                event_type="attempt_failure",
                message=f"attempt {spec.attempt_id} failed gate checks",
                phase="attempt",
                lane=str(spec.lane_id or ""),
                attempt_id=spec.attempt_id,
                model=spec.model,
                payload={
                    "issues": list(
                        (attempt_record.get("status") or {}).get("issues") or []
                    )
                },
            )
    completed_attempt_ids.add(spec.attempt_id)
    if logger is not None:
        throughput = dict(attempt_record.get("throughput") or {})
        latency = dict(attempt_record.get("latency") or {})
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="attempt_complete",
            message=f"finished attempt {spec.attempt_id}",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            payload={
                "completed_attempts": len(completed_attempt_ids),
                "decode_tps": throughput.get("decode_tps"),
                "e2e_tps": throughput.get("e2e_tps"),
                "ttft_ms": latency.get("ttft_ms"),
            },
        )
        logger.emit(
            level="debug",
            source="benchmark_suite",
            event_type="attempt_finish",
            message=f"attempt {spec.attempt_id} finalized for ledger reducers",
            phase="attempt",
            lane=str(spec.lane_id or ""),
            attempt_id=spec.attempt_id,
            model=spec.model,
            payload={
                "status": str((attempt_record.get("status") or {}).get("state") or ""),
                "passed": bool((attempt_record.get("status") or {}).get("ok", False)),
            },
        )


def _attempt_artifact_paths(
    layout: SuiteRunLayout, attempt_id: str
) -> tuple[str, dict[str, str]]:
    root = layout.artifacts_dir / "native" / _safe_name(attempt_id)
    root.mkdir(parents=True, exist_ok=True)
    return str(root), {
        "attempt_artifact_dir": str(root),
        "stdout_log": str(root / "stdout.log"),
        "stderr_log": str(root / "stderr.log"),
        "evidence_capsule": str(root / "evidence_capsule.json"),
        "checkpoint_metadata": str(layout.checkpoint_json),
        "flight_recorder_timeline": str(layout.events_ndjson),
        "terminal_transcript": str(layout.terminal_transcript_log),
    }


def _lane_attempts(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    lane_id: str,
    ablation_id: str,
    env_overrides: dict[str, str],
    prompt: str,
    warmup_runs: int,
    measured_runs: int,
    thread_matrix: list[tuple[int | None, int | None, int]],
    use_thread_matrix: bool,
    attempt_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    completed_attempt_ids: set[str],
    resume_completed: set[str],
    model_thread_overrides: dict[str, tuple[int, int, int]] | None = None,
) -> None:
    run_counter = 0
    matrix = list(thread_matrix) if thread_matrix else [(None, None, 32)]
    if not use_thread_matrix and not matrix:
        matrix = [(None, None, 32)]
    logger = get_active_logger()
    planned_attempts = 0
    for model in spec.models:
        matrix_for_model = matrix
        if model_thread_overrides and model in model_thread_overrides:
            decode_threads, batch_threads, ubatch = model_thread_overrides[model]
            matrix_for_model = [(decode_threads, batch_threads, ubatch)]
        for _decode_threads, _batch_threads, _ubatch in matrix_for_model:
            planned_attempts += int(warmup_runs) + int(measured_runs)
    if logger is not None:
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="lane_start",
            message=f"starting lane {lane_id} ({ablation_id})",
            phase=lane_id,
            lane=lane_id,
            payload={
                "phase": lane_id,
                "lane": lane_id,
                "planned_attempts": len(completed_attempt_ids) + planned_attempts,
            },
        )
    feature_toggles = _feature_toggles_for_env(env_overrides)
    for model_index, model in enumerate(spec.models):
        matrix_for_model = matrix
        if model_thread_overrides and model in model_thread_overrides:
            decode_threads, batch_threads, ubatch = model_thread_overrides[model]
            matrix_for_model = [(decode_threads, batch_threads, ubatch)]
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="model_start",
                message=f"starting model {model} in lane {lane_id}",
                phase=lane_id,
                lane=lane_id,
                model=model,
            )
        for combo_index, (decode_threads, batch_threads, ubatch) in enumerate(
            matrix_for_model
        ):
            for warmup_flag, count in ((True, warmup_runs), (False, measured_runs)):
                for run_index in range(int(count)):
                    run_counter += 1
                    attempt_id = (
                        f"{lane_id}-m{model_index:02d}-c{combo_index:02d}-"
                        f"{'warm' if warmup_flag else 'measure'}-r{run_index:02d}-"
                        f"{_safe_name(model)}"
                    )
                    if attempt_id in resume_completed:
                        completed_attempt_ids.add(attempt_id)
                        continue
                    artifact_dir, artifact_paths = _attempt_artifact_paths(
                        layout, attempt_id
                    )
                    if logger is not None:
                        logger.emit(
                            level="debug",
                            source="benchmark_suite",
                            event_type="attempt_scheduled",
                            message=(
                                f"scheduled {attempt_id} "
                                f"threads={decode_threads or 'auto'}x{batch_threads or 'auto'}x{ubatch}"
                            ),
                            phase=lane_id,
                            lane=lane_id,
                            attempt_id=attempt_id,
                            model=model,
                            payload={
                                "thread_tuple": (
                                    f"{decode_threads or 'auto'}x{batch_threads or 'auto'}x{ubatch}"
                                ),
                                "planned_attempts": len(completed_attempt_ids)
                                + planned_attempts,
                            },
                        )
                    attempt_spec = AttemptSpec(
                        attempt_id=attempt_id,
                        model=model,
                        prompt=prompt,
                        max_new_tokens=spec.scenario_pack.canonical_max_new_tokens,
                        context_length=spec.scenario_pack.canonical_context_length,
                        decode_threads=decode_threads,
                        batch_threads=batch_threads,
                        ubatch=ubatch,
                        sampling_profile=None,
                        coherence_first=True,
                        min_new_tokens_before_eos=12,
                        require_openmp=True,
                        require_avx2=True,
                        require_mmap=False,
                        host_access="privileged" if spec.require_perf else "user",
                        collect_hw_counters="required" if spec.require_perf else "auto",
                        require_grover=feature_toggles["grover"],
                        require_coconut=feature_toggles["coconut"],
                        force_parallel_decode=bool(spec.force_parallel_decode),
                        forbid_autoregressive_fallback=bool(
                            spec.forbid_autoregressive_fallback
                        ),
                        warmup=warmup_flag,
                        run_index=run_index,
                        lane_id=lane_id,
                        ablation_id=ablation_id,
                        feature_toggles=feature_toggles,
                        dataset_id="canonical_generation",
                        prompt_id=f"{lane_id}:{ablation_id}",
                        artifact_paths=artifact_paths,
                        env_overrides=env_overrides,
                        artifacts_dir=artifact_dir,
                    )
                    _record_attempt(
                        layout=layout,
                        spec=attempt_spec,
                        repo_root=repo_root,
                        attempt_rows=attempt_rows,
                        failure_rows=failure_rows,
                        completed_attempt_ids=completed_attempt_ids,
                    )
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="model_complete",
                message=f"completed model {model} in lane {lane_id}",
                phase=lane_id,
                lane=lane_id,
                model=model,
                payload={"completed_attempts": len(completed_attempt_ids)},
            )
    if logger is not None:
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="lane_complete",
            message=f"completed lane {lane_id}",
            phase=lane_id,
            lane=lane_id,
            payload={"completed_attempts": len(completed_attempt_ids)},
        )


def _run_continuous_surface(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    runtime_payload: dict[str, Any],
    model_thread_overrides: dict[str, tuple[int, int, int]] | None = None,
    model_continuous_overrides: dict[str, dict[str, Any]] | None = None,
    max_active_requests_values: list[int] | None = None,
    batch_wait_timeout_values: list[int] | None = None,
    state_page_rows_values: list[int] | None = None,
    max_prefill_rows_values: list[int] | None = None,
    interleaved_stream_values: list[bool] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    logger = get_active_logger()
    for model in spec.models:
        out_path = layout.continuous_dir / f"{_safe_name(model)}.json"
        stdout_path = layout.continuous_dir / f"{_safe_name(model)}.stdout.log"
        stderr_path = layout.continuous_dir / f"{_safe_name(model)}.stderr.log"
        decode_threads, batch_threads, ubatch = _continuous_thread_tuple(
            spec=spec,
            runtime_payload=runtime_payload,
            model=model,
            model_thread_overrides=model_thread_overrides,
        )
        continuous_override = dict((model_continuous_overrides or {}).get(model) or {})
        continuous_config = dict(continuous_override.get("continuous_config") or {})
        pager_config = dict(continuous_override.get("pager_config") or {})
        cmd = [
            sys.executable,
            str((repo_root / "benchmarks" / "continuous_qsg_benchmark.py").resolve()),
            "--model",
            model,
            "--json",
            "--json-out",
            str(out_path),
            "--max-new-tokens",
            str(int(spec.scenario_pack.canonical_max_new_tokens)),
            "--context-length",
            str(int(spec.scenario_pack.canonical_context_length)),
            "--concurrency",
            ",".join(str(item) for item in spec.scenario_pack.continuous_concurrency),
            "--max-active-requests",
            ",".join(
                str(int(item))
                for item in (
                    max_active_requests_values
                    or list(
                        [continuous_config.get("max_active_requests")]
                        if continuous_config.get("max_active_requests") is not None
                        else []
                    )
                    or spec.calibration_search.continuous_max_active_requests
                    or spec.scenario_pack.continuous_concurrency
                )
            ),
            "--decode-threads",
            str(int(decode_threads)),
            "--batch-threads",
            str(int(batch_threads)),
            "--ubatch",
            str(int(ubatch)),
            "--state-page-rows",
            ",".join(
                str(int(item))
                for item in (
                    state_page_rows_values
                    or list(
                        [pager_config.get("state_page_rows")]
                        if pager_config.get("state_page_rows") is not None
                        else []
                    )
                    or spec.calibration_search.continuous_state_page_rows
                    or [128]
                )
            ),
            "--batch-wait-timeout-ms",
            ",".join(
                str(int(item))
                for item in (
                    batch_wait_timeout_values
                    or list(
                        [continuous_config.get("batch_wait_timeout_ms")]
                        if continuous_config.get("batch_wait_timeout_ms") is not None
                        else []
                    )
                    or spec.calibration_search.continuous_batch_wait_timeout_ms
                    or [2]
                )
            ),
            "--max-prefill-rows-per-iteration",
            ",".join(
                str(int(item))
                for item in (
                    max_prefill_rows_values
                    or list(
                        [continuous_config.get("max_prefill_rows_per_iteration")]
                        if continuous_config.get("max_prefill_rows_per_iteration")
                        is not None
                        else []
                    )
                    or spec.calibration_search.continuous_max_prefill_rows_per_iteration
                    or [1024]
                )
            ),
            "--interleaved-streams",
            ",".join(
                "true" if bool(item) else "false"
                for item in (
                    interleaved_stream_values
                    or list(
                        [continuous_config.get("continuous_interleaved_streams")]
                        if "continuous_interleaved_streams" in continuous_config
                        else []
                    )
                    or spec.calibration_search.continuous_interleaved_streams
                    or [False]
                )
            ),
        ]
        for prompt_class in spec.scenario_pack.continuous_prompt_classes:
            cmd.extend(["--prompt-class", prompt_class])
        for policy in spec.scenario_pack.continuous_scheduler_policies:
            cmd.extend(["--scheduler-policy", policy])
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="continuous_start",
                message=f"starting continuous scheduler surface for {model}",
                phase="continuous_scheduler",
                lane="continuous_scheduler",
                model=model,
            )
        continuous_env = os.environ.copy()
        if spec.force_parallel_decode:
            continuous_env["ANVIL_FORCE_PARALLEL_DECODE"] = "1"
        if spec.forbid_autoregressive_fallback:
            continuous_env["ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK"] = "1"
            continuous_env["ANVIL_PARALLEL_AR_RECOVERY_ENABLED"] = "0"
        completed = run_logged_subprocess(
            cmd=cmd,
            cwd=repo_root,
            env=continuous_env,
            source="continuous_benchmark",
            phase="continuous_scheduler",
            lane="continuous_scheduler",
            model=model,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"continuous benchmark failed for {model}: {completed.stderr or completed.stdout}"
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        results.append(payload)
        if logger is not None:
            logger.emit_artifact(
                source="benchmark_suite",
                kind=out_path.name,
                path=out_path,
                summary=f"wrote continuous scheduler payload for {model}",
                phase="continuous_scheduler",
                lane="continuous_scheduler",
                model=model,
            )
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="continuous_complete",
                message=f"completed continuous scheduler surface for {model}",
                phase="continuous_scheduler",
                lane="continuous_scheduler",
                model=model,
            )
    return results


def _quality_temp_corpus(
    *,
    source: Path,
    destination: Path,
    limit: int | None,
) -> Path:
    rows = load_jsonl(source)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return destination


def _quality_informativeness(row: dict[str, Any]) -> float:
    prompt = str(
        row.get("prompt")
        or row.get("text")
        or row.get("question")
        or row.get("input")
        or ""
    )
    must_include = len(list(row.get("must_include") or []))
    must_not_include = len(list(row.get("must_not_include") or []))
    min_words = int(row.get("min_words", 0) or 0)
    return round(
        min(10.0, len(prompt.split()) / 8.0)
        + must_include * 1.5
        + must_not_include * 1.5
        + min_words / 20.0,
        3,
    )


def _adaptive_quality_subset(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        enumerate(rows),
        key=lambda item: (-_quality_informativeness(item[1]), item[0]),
    )
    selected_indices = sorted(index for index, _row in ranked[: max(1, int(limit))])
    return [rows[index] for index in selected_indices]


def _coherence_max_new_tokens(
    spec: BenchmarkSuiteSpec,
    item: dict[str, Any],
) -> int:
    base_budget = max(1, int(spec.scenario_pack.canonical_max_new_tokens or 1))
    explicit_budget = max(0, int(item.get("max_new_tokens", 0) or 0))
    min_words = max(0, int(item.get("min_words", 0) or 0))
    rubric_budget = max(0, min_words * 3)
    return max(base_budget, explicit_budget, rubric_budget)


def _text_printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / float(len(text))


def _repeated_ngram_ratio(text: str, ngram_size: int) -> float:
    words = [part for part in text.split() if part]
    if len(words) < ngram_size:
        return 0.0
    ngrams = [
        tuple(words[index : index + ngram_size])
        for index in range(len(words) - ngram_size + 1)
    ]
    if not ngrams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    repeated = 0
    for ngram in ngrams:
        if ngram in seen:
            repeated += 1
        else:
            seen.add(ngram)
    return repeated / float(len(ngrams))


def _quality_records(
    quality_payload: dict[str, Any],
    key: str,
    *,
    ablation_id: str | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in list(quality_payload.get(key) or []):
        if not isinstance(item, dict):
            continue
        if (
            ablation_id is not None
            and str(item.get("ablation_id") or "") != ablation_id
        ):
            continue
        results.append(dict(item))
    return results


def _entropy_bucket(value: float) -> str:
    entropy = float(value or 0.0)
    if entropy < 0.5:
        return "low"
    if entropy < 1.0:
        return "medium"
    return "high"


def _latent_projection_report(
    coherence_records: list[dict[str, Any]],
    *,
    pending: bool = False,
) -> dict[str, Any]:
    if pending:
        return {
            "status": "pending",
            "passed": False,
            "records": 0,
            "projection_required_records": 0,
            "finalized_decode_records": 0,
            "finalized_diff_records": 0,
            "raw_control_tag_records": 0,
            "leaked_control_tag_records": 0,
            "faithfulness_failures": 0,
            "evidence_missing_records": 0,
            "empty_projection_records": 0,
        }

    projection_required_records = 0
    finalized_decode_records = 0
    finalized_diff_records = 0
    raw_control_tag_records = 0
    leaked_control_tag_records = 0
    faithfulness_failures = 0
    evidence_missing_records = 0
    empty_projection_records = 0

    for record in coherence_records:
        raw_text = str(record.get("raw_generated_text") or "")
        generated_text = str(record.get("generated_text") or "")
        finalized_decode_used = bool(record.get("finalized_decode_used", False))
        finalized_differs = bool(record.get("finalized_differs_from_raw", False))
        raw_control_tags = [
            str(tag)
            for tag in list(record.get("raw_control_tags") or [])
            if str(tag).strip()
        ]
        leaked_control_tags = [
            str(tag)
            for tag in list(record.get("leaked_control_tags") or [])
            if str(tag).strip()
        ]
        projection_required = bool(raw_control_tags) or finalized_differs
        if projection_required:
            projection_required_records += 1
        if finalized_decode_used:
            finalized_decode_records += 1
        if finalized_differs:
            finalized_diff_records += 1
        if raw_control_tags:
            raw_control_tag_records += 1
        if leaked_control_tags:
            leaked_control_tag_records += 1
        if raw_text.strip() and not generated_text.strip():
            empty_projection_records += 1
        if projection_required and not raw_text.strip():
            evidence_missing_records += 1
        if leaked_control_tags:
            faithfulness_failures += 1
        elif raw_control_tags and (
            not generated_text.strip()
            or not finalized_differs
            or any(tag in generated_text for tag in raw_control_tags)
        ):
            faithfulness_failures += 1

    status = "covered" if coherence_records else "missing"
    return {
        "status": status,
        "passed": status == "covered"
        and faithfulness_failures == 0
        and evidence_missing_records == 0
        and empty_projection_records == 0,
        "records": len(coherence_records),
        "projection_required_records": projection_required_records,
        "finalized_decode_records": finalized_decode_records,
        "finalized_diff_records": finalized_diff_records,
        "raw_control_tag_records": raw_control_tag_records,
        "leaked_control_tag_records": leaked_control_tag_records,
        "faithfulness_failures": faithfulness_failures,
        "evidence_missing_records": evidence_missing_records,
        "empty_projection_records": empty_projection_records,
    }


def _quality_governance_report(quality_payload: dict[str, Any]) -> dict[str, Any]:
    perplexity_rows = _quality_records(quality_payload, "perplexity")
    confidence_rows = _quality_records(quality_payload, "confidence")
    coherence_rows = _quality_records(quality_payload, "coherence")
    accuracy_rows = _quality_records(quality_payload, "accuracy")
    pending_state = str(quality_payload.get("state") or "").strip().lower()
    if pending_state == "pending":
        latent_projection = _latent_projection_report([], pending=True)
        family_status = {
            "held_out_perplexity": {
                "status": "pending",
                "records": 0,
                "tokens_scored": 0,
            },
            "confidence_calibration": {
                "status": "pending",
                "records": 0,
                "tokens_scored": 0,
            },
            "coherence_rubric": {
                "status": "pending",
                "records": 0,
                "samples": 0,
            },
            "task_accuracy": {
                "status": "pending",
                "records": 0,
                "samples": 0,
            },
            "structural_validity": {
                "status": "pending",
                "records": 0,
            },
            "degeneration_checks": {
                "status": "pending",
                "records": 0,
            },
            "latent_to_text_faithfulness": {
                "status": "pending",
                "records": 0,
            },
            "evidence_capsule_fidelity": {
                "status": "pending",
                "records": 0,
            },
        }
        return {
            "schema_version": QUALITY_GOVERNANCE_SCHEMA_VERSION,
            "status": "pending",
            "passed": False,
            "issues": [],
            "benchmark_families": family_status,
            "coherence": {
                "records": 0,
                "utf8_invalid_records": 0,
                "empty_output_records": 0,
                "printable_ratio_failures": 0,
                "control_tag_leak_records": 0,
                "repeated_8gram_failures": 0,
            },
            "calibration": {
                "records": 0,
                "mean_expected_calibration_error": 0.0,
                "mean_entropy": 0.0,
            },
            "accuracy": {
                "records": 0,
                "samples": 0,
                "mean_pass_rate": 0.0,
                "mean_exact_match_rate": 0.0,
            },
            "entropy_buckets": {
                "low": {"samples": 0, "mean_confidence": 0.0},
                "medium": {"samples": 0, "mean_confidence": 0.0},
                "high": {"samples": 0, "mean_confidence": 0.0},
            },
            "latent_projection": latent_projection,
        }
    all_on_confidence = _quality_records(
        quality_payload,
        "confidence",
        ablation_id="all_on",
    )
    all_on_coherence = _quality_records(
        quality_payload,
        "coherence",
        ablation_id="all_on",
    )
    confidence_records = [
        dict(record)
        for item in all_on_confidence
        for record in list(item.get("records") or [])
        if isinstance(record, dict)
    ]
    coherence_records = [
        dict(record)
        for item in all_on_coherence
        for record in list(item.get("records") or [])
        if isinstance(record, dict)
    ]

    entropy_buckets = {
        "low": {"samples": 0, "mean_confidence": 0.0},
        "medium": {"samples": 0, "mean_confidence": 0.0},
        "high": {"samples": 0, "mean_confidence": 0.0},
    }
    for record in confidence_records:
        bucket = _entropy_bucket(float(record.get("mean_entropy", 0.0) or 0.0))
        entropy_buckets[bucket]["samples"] += 1
        entropy_buckets[bucket]["mean_confidence"] += float(
            record.get("mean_confidence", 0.0) or 0.0
        )
    for bucket in entropy_buckets.values():
        samples = int(bucket["samples"] or 0)
        bucket["mean_confidence"] = (
            float(bucket["mean_confidence"]) / float(samples) if samples else 0.0
        )

    total_coherence = len(coherence_records)
    utf8_invalid = sum(
        1 for record in coherence_records if not bool(record.get("utf8_valid", False))
    )
    empty_outputs = sum(
        1
        for record in coherence_records
        if not str(record.get("generated_text") or "").strip()
    )
    leaked_control_tags = sum(
        1
        for record in coherence_records
        if list(record.get("leaked_control_tags") or [])
    )
    printable_failures = sum(
        1
        for record in coherence_records
        if float(record.get("printable_ratio", 0.0) or 0.0)
        < COHERENCE_PRINTABLE_RATIO_MIN
    )
    repeated_8gram_failures = sum(
        1
        for record in coherence_records
        if float(record.get("repeated_8gram_ratio", 0.0) or 0.0)
        > COHERENCE_REPEATED_8GRAM_RATIO_MAX
    )
    latent_projection = _latent_projection_report(coherence_records)
    family_status = {
        "held_out_perplexity": {
            "status": "covered" if perplexity_rows else "missing",
            "records": len(perplexity_rows),
            "tokens_scored": sum(
                int(item.get("tokens_scored", 0) or 0) for item in perplexity_rows
            ),
        },
        "confidence_calibration": {
            "status": "covered" if confidence_rows else "missing",
            "records": len(confidence_rows),
            "tokens_scored": sum(
                int(item.get("tokens_scored", 0) or 0) for item in confidence_rows
            ),
        },
        "coherence_rubric": {
            "status": "covered" if coherence_rows else "missing",
            "records": len(coherence_rows),
            "samples": total_coherence,
        },
        "task_accuracy": {
            "status": "covered" if accuracy_rows else "missing",
            "records": len(accuracy_rows),
            "samples": sum(int(item.get("samples", 0) or 0) for item in accuracy_rows),
        },
        "structural_validity": {
            "status": "covered" if total_coherence else "missing",
            "records": total_coherence,
        },
        "degeneration_checks": {
            "status": "covered" if total_coherence else "missing",
            "records": total_coherence,
        },
        "latent_to_text_faithfulness": {
            "status": str(latent_projection.get("status") or "missing"),
            "records": int(latent_projection.get("records", 0) or 0),
        },
        "evidence_capsule_fidelity": {
            "status": str(latent_projection.get("status") or "missing"),
            "records": int(latent_projection.get("records", 0) or 0),
        },
    }
    issues = [
        f"missing_benchmark_family:{name}"
        for name, item in family_status.items()
        if str(item.get("status") or "") == "missing"
    ]
    if int(latent_projection.get("faithfulness_failures", 0) or 0) > 0:
        issues.append(
            "latent_faithfulness_failures:"
            f"{int(latent_projection['faithfulness_failures'])}"
        )
    if int(latent_projection.get("evidence_missing_records", 0) or 0) > 0:
        issues.append(
            "evidence_capsule_fidelity_failures:"
            f"{int(latent_projection['evidence_missing_records'])}"
        )
    if int(latent_projection.get("empty_projection_records", 0) or 0) > 0:
        issues.append(
            "latent_projection_empty_records:"
            f"{int(latent_projection['empty_projection_records'])}"
        )
    return {
        "schema_version": QUALITY_GOVERNANCE_SCHEMA_VERSION,
        "passed": not issues,
        "issues": issues,
        "benchmark_families": family_status,
        "coherence": {
            "records": total_coherence,
            "utf8_invalid_records": utf8_invalid,
            "empty_output_records": empty_outputs,
            "printable_ratio_failures": printable_failures,
            "control_tag_leak_records": leaked_control_tags,
            "repeated_8gram_failures": repeated_8gram_failures,
        },
        "calibration": {
            "records": len(confidence_records),
            "mean_expected_calibration_error": (
                sum(
                    float(item.get("expected_calibration_error", 0.0) or 0.0)
                    for item in all_on_confidence
                )
                / float(len(all_on_confidence))
                if all_on_confidence
                else 0.0
            ),
            "mean_entropy": (
                sum(
                    float(item.get("mean_entropy", 0.0) or 0.0)
                    for item in all_on_confidence
                )
                / float(len(all_on_confidence))
                if all_on_confidence
                else 0.0
            ),
        },
        "accuracy": {
            "records": len(accuracy_rows),
            "samples": sum(int(item.get("samples", 0) or 0) for item in accuracy_rows),
            "mean_pass_rate": (
                sum(float(item.get("pass_rate", 0.0) or 0.0) for item in accuracy_rows)
                / float(len(accuracy_rows))
                if accuracy_rows
                else 0.0
            ),
            "mean_exact_match_rate": (
                sum(
                    float(item.get("exact_match_rate", 0.0) or 0.0)
                    for item in accuracy_rows
                )
                / float(len(accuracy_rows))
                if accuracy_rows
                else 0.0
            ),
        },
        "entropy_buckets": entropy_buckets,
        "latent_projection": latent_projection,
    }


def _quality_record_for_model(
    quality_payload: dict[str, Any],
    key: str,
    *,
    model: str,
) -> dict[str, Any]:
    for item in list(quality_payload.get(key) or []):
        if (
            str(item.get("model") or "") == model
            and str(item.get("ablation_id") or "") == "all_on"
        ):
            return dict(item)
    return {}


def _quality_gate_for_model(
    quality_payload: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    perplexity = _quality_record_for_model(quality_payload, "perplexity", model=model)
    confidence = _quality_record_for_model(quality_payload, "confidence", model=model)
    coherence = _quality_record_for_model(quality_payload, "coherence", model=model)
    accuracy = _quality_record_for_model(quality_payload, "accuracy", model=model)

    issues: list[str] = []
    perplexity_value = float(perplexity.get("perplexity", 0.0) or 0.0)
    perplexity_tokens = int(perplexity.get("tokens_scored", 0) or 0)
    if perplexity_tokens <= 0:
        issues.append("perplexity_tokens_scored=0")
    if (
        not 0.0
        < perplexity_value
        <= float(CALIBRATION_QUALITY_MINIMA["perplexity_max"])
    ):
        issues.append(f"perplexity_out_of_range:{perplexity_value}")

    confidence_tokens = int(confidence.get("tokens_scored", 0) or 0)
    mean_confidence = float(confidence.get("mean_token_confidence", 0.0) or 0.0)
    p95_confidence = float(confidence.get("p95_token_confidence", 0.0) or 0.0)
    expected_calibration_error = float(
        confidence.get("expected_calibration_error", 0.0) or 0.0
    )
    if confidence_tokens <= 0:
        issues.append("confidence_tokens_scored=0")
    if mean_confidence <= 0.0:
        issues.append(f"mean_token_confidence={mean_confidence}")
    if p95_confidence <= 0.0:
        issues.append(f"p95_token_confidence={p95_confidence}")
    if (
        not 0.0
        <= expected_calibration_error
        <= float(CALIBRATION_QUALITY_MINIMA["expected_calibration_error_max"])
    ):
        issues.append(f"expected_calibration_error={expected_calibration_error}")

    coherence_pass_rate = float(coherence.get("pass_rate", 0.0) or 0.0)
    if coherence_pass_rate < float(
        CALIBRATION_QUALITY_MINIMA["coherence_pass_rate_min"]
    ):
        issues.append(f"coherence_pass_rate={coherence_pass_rate}")
    records = list(coherence.get("records") or [])
    if not records:
        issues.append("coherence_records_missing")
    elif all(not str(item.get("generated_text") or "").strip() for item in records):
        issues.append("coherence_generated_text_empty")

    accuracy_samples = int(accuracy.get("samples", 0) or 0)
    accuracy_pass_rate = float(accuracy.get("pass_rate", 0.0) or 0.0)
    accuracy_exact_match_rate = float(accuracy.get("exact_match_rate", 0.0) or 0.0)
    if accuracy_samples <= 0:
        issues.append("accuracy_samples=0")
    if accuracy_pass_rate < float(CALIBRATION_QUALITY_MINIMA["accuracy_pass_rate_min"]):
        issues.append(f"accuracy_pass_rate={accuracy_pass_rate}")

    evidence = {
        "perplexity": {
            "tokens_scored": perplexity_tokens,
            "perplexity": perplexity_value,
        },
        "confidence": {
            "tokens_scored": confidence_tokens,
            "mean_token_confidence": mean_confidence,
            "p95_token_confidence": p95_confidence,
            "expected_calibration_error": expected_calibration_error,
        },
        "coherence": {
            "pass_rate": coherence_pass_rate,
            "record_count": len(records),
        },
        "accuracy": {
            "samples": accuracy_samples,
            "pass_rate": accuracy_pass_rate,
            "exact_match_rate": accuracy_exact_match_rate,
        },
    }
    return {
        "passed": not issues,
        "issues": issues,
        "evidence": evidence,
    }


def _kv_cache_quantization_mode_from_env(*payloads: dict[str, Any]) -> str:
    for payload in payloads:
        env = dict(payload.get("env") or {})
        raw = str(env.get("ANVIL_KV_QUANT") or "").strip().lower()
        if not raw:
            continue
        if raw in {"0", "false", "off", "none", "fp32", "f32"}:
            return "fp32"
        if raw in {"1", "true", "on", "q8", "int8"}:
            return "q8"
        return raw
    raw = str(os.getenv("ANVIL_KV_QUANT") or "").strip().lower()
    if raw in {"1", "true", "on", "q8", "int8"}:
        return "q8"
    return "fp32"


def _run_quality_evals(
    *,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
) -> dict[str, Any]:
    logger = get_active_logger()
    perplexity_source = Path(spec.quality_eval.perplexity_corpus)
    confidence_source = Path(spec.quality_eval.confidence_corpus)
    coherence_source = Path(spec.quality_eval.rubric_corpus)
    accuracy_source = Path(spec.quality_eval.accuracy_corpus)
    payload = {
        "schema_version": "native_qsg_suite.quality.v1",
        "perplexity": [],
        "confidence": [],
        "coherence": [],
        "accuracy": [],
        "governance": {},
        "adaptive_selection": {},
    }
    lane_envs = _quality_lane_envs(spec)
    adaptive_top_k = int(spec.quality_eval.adaptive_top_k or 0)

    for model in spec.models:
        for lane in lane_envs:
            env = dict(lane["env"])
            if logger is not None:
                logger.emit(
                    level="info",
                    source="benchmark_suite",
                    event_type="quality_eval_start",
                    message=f"starting quality evals for {model} ({lane['ablation_id']})",
                    phase="quality_eval",
                    lane=str(lane["lane_id"]),
                    model=model,
                )
            ppl_rows = load_jsonl(perplexity_source)
            conf_rows = load_jsonl(confidence_source)
            rubric_rows = load_jsonl(coherence_source)
            accuracy_rows = (
                load_jsonl(accuracy_source) if accuracy_source.exists() else []
            )
            selected_limit = int(spec.quality_eval.max_samples_per_family or 0)
            if adaptive_top_k > 0:
                selected_limit = adaptive_top_k
            if selected_limit > 0:
                ppl_rows = _adaptive_quality_subset(ppl_rows, selected_limit)
                conf_rows = _adaptive_quality_subset(conf_rows, selected_limit)
                rubric_rows = _adaptive_quality_subset(rubric_rows, selected_limit)
                accuracy_rows = _adaptive_quality_subset(accuracy_rows, selected_limit)
            ppl_path = (
                layout.eval_dir
                / f"tmp_{_safe_name(model)}_{lane['ablation_id']}_perplexity.jsonl"
            )
            ppl_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in ppl_rows),
                encoding="utf-8",
            )
            conf_path = (
                layout.eval_dir
                / f"tmp_{_safe_name(model)}_{lane['ablation_id']}_confidence.jsonl"
            )
            conf_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in conf_rows),
                encoding="utf-8",
            )
            payload["adaptive_selection"][f"{model}:{lane['ablation_id']}"] = {
                "adaptive_top_k": adaptive_top_k,
                "shadow_mode": bool(spec.quality_eval.shadow_mode),
                "perplexity_selected": len(ppl_rows),
                "confidence_selected": len(conf_rows),
                "coherence_selected": len(rubric_rows),
                "accuracy_selected": len(accuracy_rows),
            }
            perplexity = evaluate_perplexity(
                model=model,
                corpus_path=ppl_path,
                context_length=spec.scenario_pack.canonical_context_length,
                env_overrides=env,
            )
            perplexity["lane_id"] = lane["lane_id"]
            perplexity["ablation_id"] = lane["ablation_id"]
            append_ndjson(layout.quality_attempts_ndjson, perplexity)
            payload["perplexity"].append(perplexity)
            if logger is not None:
                logger.emit(
                    level="debug",
                    source="benchmark_suite",
                    event_type="quality_metric",
                    message=f"perplexity={float(perplexity.get('perplexity', 0.0) or 0.0):.4f}",
                    phase="quality_eval",
                    lane=str(lane["lane_id"]),
                    model=model,
                )

            confidence = evaluate_confidence(
                model=model,
                corpus_path=conf_path,
                context_length=spec.scenario_pack.canonical_context_length,
                env_overrides=env,
            )
            confidence["lane_id"] = lane["lane_id"]
            confidence["ablation_id"] = lane["ablation_id"]
            append_ndjson(layout.quality_attempts_ndjson, confidence)
            payload["confidence"].append(confidence)
            if logger is not None:
                logger.emit(
                    level="debug",
                    source="benchmark_suite",
                    event_type="quality_metric",
                    message=(
                        "mean_token_confidence="
                        f"{float(confidence.get('mean_token_confidence', 0.0) or 0.0):.4f}"
                    ),
                    phase="quality_eval",
                    lane=str(lane["lane_id"]),
                    model=model,
                )

            accuracy_path = (
                layout.eval_dir
                / f"tmp_{_safe_name(model)}_{lane['ablation_id']}_accuracy.jsonl"
            )
            accuracy_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in list(accuracy_rows)
                ),
                encoding="utf-8",
            )
            accuracy = evaluate_accuracy(
                model=model,
                corpus_path=accuracy_path,
                context_length=spec.scenario_pack.canonical_context_length,
                env_overrides=env,
            )
            accuracy["lane_id"] = lane["lane_id"]
            accuracy["ablation_id"] = lane["ablation_id"]
            append_ndjson(layout.quality_attempts_ndjson, accuracy)
            payload["accuracy"].append(accuracy)
            if logger is not None:
                logger.emit(
                    level="debug",
                    source="benchmark_suite",
                    event_type="quality_metric",
                    message=(
                        f"accuracy_pass_rate={float(accuracy.get('pass_rate', 0.0) or 0.0):.4f}"
                    ),
                    phase="quality_eval",
                    lane=str(lane["lane_id"]),
                    model=model,
                )

            scorer = NativeLogitScorer(
                model=model,
                context_length=spec.scenario_pack.canonical_context_length,
                env_overrides=env,
            )
            try:
                coherence_records: list[dict[str, Any]] = []
                for item in rubric_rows:
                    prompt = str(item.get("prompt") or "")
                    prompt_tokens = scorer.engine.prepare_prompt_tokens(prompt)
                    coherence_max_new_tokens = _coherence_max_new_tokens(spec, item)
                    output_tokens = scorer.engine.generate(
                        prompt_tokens=list(prompt_tokens),
                        max_new_tokens=coherence_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=0,
                    )
                    generated_tokens = output_tokens[len(prompt_tokens) :]
                    raw_generated = scorer.engine.detokenize(generated_tokens)
                    decode_generated_tokens = getattr(
                        scorer.engine,
                        "decode_generated_tokens",
                        None,
                    )
                    if callable(decode_generated_tokens):
                        generated = str(decode_generated_tokens(generated_tokens))
                    else:
                        finalize_response_text = getattr(
                            scorer.engine,
                            "finalize_response_text",
                            None,
                        )
                        generated = (
                            str(finalize_response_text(raw_generated))
                            if callable(finalize_response_text)
                            else str(raw_generated)
                        )
                    words = len(str(generated).split())
                    issues: list[str] = []
                    for marker in list(item.get("must_include") or []):
                        if str(marker).lower() not in str(generated).lower():
                            issues.append(f"missing:{marker}")
                    for marker in list(item.get("must_not_include") or []):
                        if str(marker) in str(generated):
                            issues.append(f"forbidden:{marker}")
                    min_words = int(item.get("min_words", 0) or 0)
                    if words < min_words:
                        issues.append(f"min_words:{words}<{min_words}")
                    printable_ratio = _text_printable_ratio(str(generated))
                    repeated_4gram_ratio = _repeated_ngram_ratio(str(generated), 4)
                    repeated_8gram_ratio = _repeated_ngram_ratio(str(generated), 8)
                    if not str(generated).strip():
                        issues.append("empty_output")
                    if printable_ratio < 0.95:
                        issues.append(f"printable_ratio={printable_ratio:.3f}")
                    if repeated_8gram_ratio > 0.2:
                        issues.append(
                            f"repeated_8gram_ratio={repeated_8gram_ratio:.3f}"
                        )
                    try:
                        str(generated).encode("utf-8")
                        utf8_valid = True
                    except UnicodeEncodeError:
                        utf8_valid = False
                        issues.append("utf8_invalid")
                    leaked_control_tags = [
                        tag for tag in ("<think>", "</think>") if tag in str(generated)
                    ]
                    raw_control_tags = [
                        tag
                        for tag in ("<think>", "</think>")
                        if tag in str(raw_generated)
                    ]
                    for tag in leaked_control_tags:
                        issues.append(f"leaked_control_tag:{tag}")
                    coherence_records.append(
                        {
                            "sample_id": item.get("sample_id"),
                            "prompt": prompt,
                            "generated_text": str(generated),
                            "raw_generated_text": str(raw_generated),
                            "finalized_decode_used": bool(
                                callable(decode_generated_tokens)
                            ),
                            "finalized_differs_from_raw": str(generated)
                            != str(raw_generated),
                            "printable_ratio": printable_ratio,
                            "repeated_4gram_ratio": repeated_4gram_ratio,
                            "repeated_8gram_ratio": repeated_8gram_ratio,
                            "max_new_tokens": coherence_max_new_tokens,
                            "utf8_valid": utf8_valid,
                            "leaked_control_tags": leaked_control_tags,
                            "raw_control_tags": raw_control_tags,
                            "issues": sorted(set(issues)),
                            "ok": not issues,
                        }
                    )
                coherence = {
                    "schema_version": "native_qsg_suite.coherence.v1",
                    "model": model,
                    "lane_id": lane["lane_id"],
                    "ablation_id": lane["ablation_id"],
                    "pass_rate": (
                        sum(1 for item in coherence_records if item["ok"])
                        / float(len(coherence_records))
                        if coherence_records
                        else 0.0
                    ),
                    "records": coherence_records,
                }
                append_ndjson(layout.quality_attempts_ndjson, coherence)
                payload["coherence"].append(coherence)
                if logger is not None:
                    logger.emit(
                        level="info",
                        source="benchmark_suite",
                        event_type="quality_eval_complete",
                        message=(
                            f"completed quality evals for {model} ({lane['ablation_id']}) "
                            f"pass_rate={float(coherence.get('pass_rate', 0.0) or 0.0):.4f}"
                        ),
                        phase="quality_eval",
                        lane=str(lane["lane_id"]),
                        model=model,
                    )
            finally:
                scorer.close()
    payload["governance"] = _quality_governance_report(payload)
    _write_json_artifact(
        layout=layout,
        path=layout.quality_summary_json,
        payload=payload,
        summary="wrote quality summary",
        phase="quality_eval",
    )
    return payload


def _quality_baseline_for_model(
    quality_payload: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, field_name in (
        ("perplexity", "perplexity"),
        ("confidence", "mean_token_confidence"),
        ("coherence", "pass_rate"),
        ("accuracy", "pass_rate"),
    ):
        matches = [
            float(item.get(field_name, 0.0) or 0.0)
            for item in list(quality_payload.get(key) or [])
            if str(item.get("model") or "") == model
            and str(item.get("ablation_id") or "") == "all_on"
        ]
        median, mad = _median_and_mad(matches)
        result[key] = {
            "median": median,
            "mad": mad,
        }
    return result


def _profile_signature_sha256(spec: BenchmarkSuiteSpec) -> str:
    payload = json.dumps(asdict(spec), sort_keys=True, ensure_ascii=True).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _calibration_contract_payload(
    *,
    repo_root: Path,
    spec: BenchmarkSuiteSpec,
    host_contract: dict[str, Any],
    host_contract_sha256: str,
    model_name: str,
    model_contract: dict[str, Any],
    winner: dict[str, Any],
    scheduler_winner: dict[str, Any],
    quality_payload: dict[str, Any],
    kv_cache_quantization: str,
    calibration_mode: str,
    calibration_source: str,
    calibration_target_profile: str,
    workload_digest: str,
    hotspot_metrics: dict[str, Any],
) -> dict[str, Any]:
    candidate = dict(winner.get("candidate") or {})
    quality_gate = _quality_gate_for_model(quality_payload, model=model_name)
    quant_variant = str(model_contract.get("quant_variant") or "").strip()
    continuous_config = {
        "scheduler_policy": str(scheduler_winner.get("scheduler_policy") or "fcfs"),
        "max_active_requests": int(scheduler_winner.get("max_active_requests", 1) or 1),
        "batch_wait_timeout_ms": int(
            scheduler_winner.get("batch_wait_timeout_ms", 2) or 2
        ),
        "max_prefill_rows_per_iteration": int(
            scheduler_winner.get("max_prefill_rows_per_iteration", 1024) or 1024
        ),
        "continuous_interleaved_streams": bool(
            scheduler_winner.get("continuous_interleaved_streams", False)
        ),
    }
    pager_config = {
        "state_page_rows": int(scheduler_winner.get("state_page_rows", 128) or 128),
        "state_compaction_soft_threshold": 0.18,
        "state_compaction_hard_threshold": 0.30,
    }
    admission = {
        "decision": "probe" if str(calibration_mode) == "probe" else "search",
        "budget_tier": str(calibration_mode or "search"),
        "invocation_source": str(calibration_source or "manual"),
        "workload_digest": str(workload_digest or ""),
        "target_profile": str(calibration_target_profile or spec.profile_name),
    }
    objective_vector = {
        "decode_tps_median": float(winner.get("decode_tps_median", 0.0) or 0.0),
        "ttft_ms_median": float(winner.get("ttft_ms_median", 0.0) or 0.0),
        "ttft_ms_p95": float(
            scheduler_winner.get("ttft_ms_p95", winner.get("ttft_ms_median", 0.0))
            or 0.0
        ),
        "queue_wait_ms_p95": float(
            scheduler_winner.get("queue_wait_ms_p95", 0.0) or 0.0
        ),
        "fairness": float(scheduler_winner.get("fairness", 1.0) or 1.0),
        "decode_goodput_tps": float(
            scheduler_winner.get(
                "decode_goodput_tps", winner.get("decode_tps_median", 0.0)
            )
            or 0.0
        ),
        "hotspot_penalty": float(hotspot_metrics.get("hotspot_penalty", 0.0) or 0.0),
    }
    safe_envelope = {
        "fairness_floor": float(spec.calibration_search.fairness_floor or 0.0),
        "queue_wait_ms_p95_ceiling": float(
            spec.calibration_search.queue_wait_p95_ceiling_ms or 0.0
        ),
        "quality_regression_policy": "fail_closed",
        "decode_tps_regression_floor_pct": float(
            spec.calibration_search.decode_tps_regression_floor_pct or 0.0
        ),
        "hotspot_watch_penalty": float(
            hotspot_metrics.get("hotspot_penalty", 0.0) or 0.0
        ),
    }
    capability_ledger = build_runtime_capability_ledger(
        {
            "model": model_name,
            "digest": model_digest(model_contract),
            "decode_threads": int(candidate.get("decode_threads") or 0),
            "batch_threads": int(candidate.get("batch_threads") or 0),
            "ubatch": int(candidate.get("ubatch") or 0),
            "generation_mode": "parallel_hybrid",
            "scheduler_policy": str(continuous_config.get("scheduler_policy") or ""),
            "max_active_requests": int(
                continuous_config.get("max_active_requests", 0) or 0
            ),
            "batch_wait_timeout_ms": int(
                continuous_config.get("batch_wait_timeout_ms", 0) or 0
            ),
            "max_prefill_rows_per_iteration": int(
                continuous_config.get("max_prefill_rows_per_iteration", 0) or 0
            ),
            "continuous_interleaved_streams": bool(
                continuous_config.get("continuous_interleaved_streams", False)
            ),
            "state_page_rows": int(pager_config.get("state_page_rows", 0) or 0),
            "state_compaction_soft_threshold": float(
                pager_config.get("state_compaction_soft_threshold", 0.0) or 0.0
            ),
            "state_compaction_hard_threshold": float(
                pager_config.get("state_compaction_hard_threshold", 0.0) or 0.0
            ),
            "workload_digest": str(workload_digest or ""),
            "budget_tier": str(calibration_mode or "search"),
            "admission_decision": str(admission.get("decision") or ""),
            "affinity_policy": str(spec.affinity_policy or ""),
            "parallel_decode_allowed": True,
            "full_qsg_enabled": True,
            "full_graph_enabled": True,
            "qsg_processors_native_enabled": True,
            "batched_prefill_native_enabled": True,
            "native_backend_abi_match": True,
            "perf_event_access": True,
        },
        host_fingerprint=str(host_contract.get("host_fingerprint") or ""),
        certification_state="ready",
        source="calibration_contract",
    )
    return {
        "schema_version": "native_qsg_suite.tuning_contract.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness_state": "ready",
        "source_phase": str(calibration_target_profile or spec.profile_name),
        "host_contract_id": str(
            host_contract.get("contract_id")
            or host_contract.get("host_fingerprint")
            or ""
        ),
        "host_fingerprint": str(host_contract.get("host_fingerprint") or ""),
        "model": model_name,
        "model_digest": model_digest(model_contract),
        "profile_schema_version": spec.schema_version,
        "benchmark_harness_hash": benchmark_harness_hash(repo_root),
        "contract_hashes": {
            "host_contract_sha256": host_contract_sha256,
            "profile_signature_sha256": _profile_signature_sha256(spec),
        },
        "thread_config": {
            "decode_threads": int(candidate.get("decode_threads") or 0),
            "batch_threads": int(candidate.get("batch_threads") or 0),
            "ubatch": int(candidate.get("ubatch") or 0),
        },
        "continuous_config": continuous_config,
        "pager_config": pager_config,
        "admission": admission,
        "objective_vector": objective_vector,
        "safe_envelope": safe_envelope,
        "rollback_policy": {
            "triggers": [
                "quality_gate_failed",
                "fairness_floor_breach",
                "queue_wait_ceiling_breach",
                "workload_digest_mismatch",
            ],
            "hotspot_watch_kernels": list(hotspot_metrics.get("top_hotspots") or []),
        },
        "capability_ledger": capability_ledger,
        "quantization_profile": {
            "weights": quant_variant,
            "kv_cache": kv_cache_quantization,
        },
        "calibration_stats": {
            "decode_tps_median": float(winner.get("decode_tps_median", 0.0) or 0.0),
            "decode_tps_mad": float(winner.get("decode_tps_mad", 0.0) or 0.0),
            "ttft_ms_median": float(winner.get("ttft_ms_median", 0.0) or 0.0),
            "ttft_ms_mad": float(winner.get("ttft_ms_mad", 0.0) or 0.0),
            "decode_tps_cv_pct": float(winner.get("decode_tps_cv_pct", 0.0) or 0.0),
            "accepted_parallel_tokens_total": int(
                winner.get("accepted_parallel_tokens_total", 0) or 0
            ),
            "rejected_parallel_tokens_total": int(
                winner.get("rejected_parallel_tokens_total", 0) or 0
            ),
            "proposed_parallel_tokens_total": int(
                winner.get("proposed_parallel_tokens_total", 0) or 0
            ),
            "draft_acceptance_ratio": float(
                winner.get("draft_acceptance_ratio", 0.0) or 0.0
            ),
            "speculative_attempts": int(winner.get("speculative_attempts", 0) or 0),
            "speculative_metrics": dict(winner.get("speculative_summary") or {}),
            "scheduler_metrics": dict(scheduler_winner or {}),
            "kernel_metrics": dict(winner.get("kernel_summary") or {}),
            "pmu_metrics": dict(winner.get("pmu_summary") or {}),
            "hotspot_metrics": dict(hotspot_metrics or {}),
            "quality_metrics": _quality_baseline_for_model(
                quality_payload, model=model_name
            ),
            "measured_runs": len(list(winner.get("rows") or [])),
        },
        "quality_gate": quality_gate,
        "quality_gate_version": "native_qsg_suite.quality_gate.v1",
    }


def _run_calibration(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
    attempt_rows: list[dict[str, Any]],
    failure_rows: list[dict[str, Any]],
    completed_attempt_ids: set[str],
    resume_completed: set[str],
    calibration_mode: str = "search",
    calibration_source: str = "manual",
    calibration_target_profile: str = "",
) -> dict[str, Any]:
    logger = get_active_logger()
    host_contract = dict(preflight_payload.get("host_contract") or {})
    host_contract_sha256 = str(preflight_payload.get("host_contract_sha256") or "")
    models = dict(preflight_payload.get("models") or {})
    kv_cache_quantization = _kv_cache_quantization_mode_from_env(
        dict(preflight_payload.get("runtime") or {}),
        dict(preflight_payload.get("launch_runtime") or {}),
    )
    max_threads = int(host_contract.get("required_visible_threads", 0) or 0)
    if max_threads <= 0:
        raise RuntimeError(
            "Calibration requires a certified host contract with required_visible_threads."
        )
    budget = _calibration_budget_settings(spec, calibration_mode)
    stage1_warmup_runs = int(budget.get("stage1_warmup_runs", 0) or 0)
    stage1_measured_runs = int(budget.get("stage1_measured_runs", 1) or 1)
    stage2_measured_runs = int(budget.get("stage2_measured_runs", 1) or 1)
    workload_digest = _calibration_workload_digest(
        spec, target_profile=calibration_target_profile or spec.profile_name
    )

    winners: dict[str, dict[str, Any]] = {}
    thread_frontiers: dict[str, list[dict[str, Any]]] = {}
    scheduler_frontiers: dict[str, list[dict[str, Any]]] = {}
    for model_name in spec.models:
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="calibration_model_start",
                message=f"starting calibration search for {model_name}",
                phase="lane_calibration",
                lane="calibration",
                model=model_name,
            )
        model_spec = replace(spec, models=[model_name])
        stage1_candidates = _calibration_seed_candidates(
            max_threads,
            list(spec.scenario_pack.thread_matrix_ubatch or [32]),
        )
        _lane_attempts(
            repo_root=repo_root,
            layout=layout,
            spec=model_spec,
            lane_id="calibration_stage1",
            ablation_id="all_on",
            env_overrides={},
            prompt=spec.scenario_pack.canonical_prompt,
            warmup_runs=stage1_warmup_runs,
            measured_runs=stage1_measured_runs,
            thread_matrix=stage1_candidates,
            use_thread_matrix=True,
            attempt_rows=attempt_rows,
            failure_rows=failure_rows,
            completed_attempt_ids=completed_attempt_ids,
            resume_completed=resume_completed,
        )
        stage1_results = _rank_calibration_candidates(
            [
                _candidate_metrics(
                    attempt_rows=attempt_rows,
                    failure_rows=failure_rows,
                    lane_id="calibration_stage1",
                    model=model_name,
                    candidate=item,
                )
                for item in stage1_candidates
            ]
        )
        thread_frontiers[model_name] = list(stage1_results)
        stage1_leaders = [
            tuple(
                int((item.get("candidate") or {}).get(key) or 0)
                for key in ("decode_threads", "batch_threads", "ubatch")
            )
            for item in stage1_results
            if bool(item.get("ok", False))
        ][: max(1, int(budget.get("stage1_top_k", 1) or 1))]
        stage2_candidates = [
            item
            for item in _calibration_refined_candidates(
                stage1_leaders, max_threads=max_threads
            )
            if item not in stage1_candidates
        ]
        if stage2_candidates and stage2_measured_runs > 0:
            _lane_attempts(
                repo_root=repo_root,
                layout=layout,
                spec=model_spec,
                lane_id="calibration_stage2",
                ablation_id="all_on",
                env_overrides={},
                prompt=spec.scenario_pack.canonical_prompt,
                warmup_runs=0,
                measured_runs=stage2_measured_runs,
                thread_matrix=stage2_candidates,
                use_thread_matrix=True,
                attempt_rows=attempt_rows,
                failure_rows=failure_rows,
                completed_attempt_ids=completed_attempt_ids,
                resume_completed=resume_completed,
            )
        thread_results = _rank_calibration_candidates(
            stage1_results
            + [
                _candidate_metrics(
                    attempt_rows=attempt_rows,
                    failure_rows=failure_rows,
                    lane_id="calibration_stage2",
                    model=model_name,
                    candidate=item,
                )
                for item in stage2_candidates
            ]
        )
        thread_frontiers[model_name] = list(thread_results)
        thread_finalists = [
            dict(item) for item in thread_results if bool(item.get("ok", False))
        ][: max(1, int(budget.get("stage2_top_k", 1) or 1))]
        if not thread_finalists:
            raise RuntimeError(
                f"Calibration failed to find a stable candidate for {model_name}."
            )
        surface = _continuous_surface_values(model_spec, mode=calibration_mode)
        combined_candidates: list[dict[str, Any]] = []
        for thread_winner in thread_finalists:
            thread_candidate = dict(thread_winner.get("candidate") or {})
            thread_tuple = (
                int(thread_candidate.get("decode_threads") or 0),
                int(thread_candidate.get("batch_threads") or 0),
                int(thread_candidate.get("ubatch") or 0),
            )
            scheduler_candidates: list[dict[str, Any]] = []
            if (
                model_spec.scenario_pack.continuous_concurrency
                and model_spec.scenario_pack.continuous_prompt_classes
                and model_spec.scenario_pack.continuous_scheduler_policies
            ):
                continuous_payloads = _run_continuous_surface(
                    repo_root=repo_root,
                    layout=layout,
                    spec=model_spec,
                    runtime_payload=dict(preflight_payload.get("runtime") or {}),
                    model_thread_overrides={model_name: thread_tuple},
                    max_active_requests_values=list(
                        surface.get("max_active_requests") or []
                    ),
                    batch_wait_timeout_values=list(
                        surface.get("batch_wait_timeout_ms") or []
                    ),
                    state_page_rows_values=list(surface.get("state_page_rows") or []),
                    max_prefill_rows_values=list(
                        surface.get("max_prefill_rows_per_iteration") or []
                    ),
                    interleaved_stream_values=list(
                        surface.get("continuous_interleaved_streams") or []
                    ),
                )
                for payload in continuous_payloads:
                    scheduler_candidates.extend(_scheduler_candidate_rows(payload))
            selected_scheduler = _select_scheduler_candidates(
                model_spec, scheduler_candidates
            ) or [
                {
                    "scheduler_policy": "fcfs",
                    "max_active_requests": max(
                        1,
                        int(
                            (
                                surface.get("max_active_requests")
                                or model_spec.scenario_pack.continuous_concurrency
                                or [1]
                            )[0]
                        ),
                    ),
                    "batch_wait_timeout_ms": max(
                        1, int((surface.get("batch_wait_timeout_ms") or [2])[0])
                    ),
                    "state_page_rows": max(
                        1, int((surface.get("state_page_rows") or [128])[0])
                    ),
                    "max_prefill_rows_per_iteration": max(
                        1,
                        int(
                            (surface.get("max_prefill_rows_per_iteration") or [1024])[0]
                        ),
                    ),
                    "continuous_interleaved_streams": bool(
                        (surface.get("continuous_interleaved_streams") or [False])[0]
                    ),
                    "ttft_ms_p95": float(
                        thread_winner.get("ttft_ms_median", 0.0) or 0.0
                    ),
                    "queue_wait_ms_p95": float(
                        thread_winner.get("queue_wait_ms_p95", 0.0) or 0.0
                    ),
                    "scheduler_iteration_ms_p95": float(
                        thread_winner.get("scheduler_iteration_ms_p95", 0.0) or 0.0
                    ),
                    "fairness": 1.0,
                    "decode_goodput_tps": float(
                        thread_winner.get("decode_tps_median", 0.0) or 0.0
                    ),
                    "decode_tps_global": float(
                        thread_winner.get("decode_tps_median", 0.0) or 0.0
                    ),
                    "state_fragmentation_ratio": 0.0,
                    "drift_overhead_percent": 0.0,
                }
            ]
            for scheduler_winner in selected_scheduler:
                combined = dict(thread_winner)
                combined.update(
                    {
                        "scheduler_policy": str(
                            scheduler_winner.get("scheduler_policy") or "fcfs"
                        ),
                        "max_active_requests": int(
                            scheduler_winner.get("max_active_requests", 1) or 1
                        ),
                        "batch_wait_timeout_ms": int(
                            scheduler_winner.get("batch_wait_timeout_ms", 2) or 2
                        ),
                        "state_page_rows": int(
                            scheduler_winner.get("state_page_rows", 128) or 128
                        ),
                        "max_prefill_rows_per_iteration": int(
                            scheduler_winner.get("max_prefill_rows_per_iteration", 1024)
                            or 1024
                        ),
                        "continuous_interleaved_streams": bool(
                            scheduler_winner.get(
                                "continuous_interleaved_streams", False
                            )
                        ),
                        "ttft_ms_p95": float(
                            scheduler_winner.get(
                                "ttft_ms_p95",
                                thread_winner.get("ttft_ms_median", 0.0),
                            )
                            or 0.0
                        ),
                        "queue_wait_ms_p95": float(
                            scheduler_winner.get("queue_wait_ms_p95", 0.0) or 0.0
                        ),
                        "fairness": float(scheduler_winner.get("fairness", 1.0) or 1.0),
                        "decode_goodput_tps": float(
                            scheduler_winner.get(
                                "decode_goodput_tps",
                                thread_winner.get("decode_tps_median", 0.0),
                            )
                            or 0.0
                        ),
                        "state_fragmentation_ratio": float(
                            scheduler_winner.get("state_fragmentation_ratio", 0.0)
                            or 0.0
                        ),
                        "drift_overhead_percent": float(
                            scheduler_winner.get("drift_overhead_percent", 0.0) or 0.0
                        ),
                        "continuous_metrics": dict(
                            scheduler_winner.get("continuous_metrics") or {}
                        ),
                        "fairness_floor": float(
                            model_spec.calibration_search.fairness_floor or 0.0
                        ),
                        "queue_wait_ms_p95_ceiling": float(
                            model_spec.calibration_search.queue_wait_p95_ceiling_ms
                            or 0.0
                        ),
                    }
                )
                combined["quality_constraints_pass"] = float(
                    combined.get("fairness", 1.0) or 1.0
                ) >= float(combined.get("fairness_floor", 0.0) or 0.0) and float(
                    combined.get("queue_wait_ms_p95", 0.0) or 0.0
                ) <= float(
                    combined.get("queue_wait_ms_p95_ceiling", float("inf"))
                    or float("inf")
                )
                combined["safety_envelope_breached"] = not bool(
                    combined["quality_constraints_pass"]
                )
                combined_candidates.append(combined)
        scheduler_frontiers[model_name] = list(combined_candidates)
        final_results = _rank_calibration_candidates(
            combined_candidates or thread_finalists
        )
        if not final_results:
            raise RuntimeError(
                f"Calibration failed to select a scheduler surface for {model_name}."
            )
        winners[model_name] = dict(final_results[0])
        if logger is not None:
            winner = dict(winners[model_name].get("candidate") or {})
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="calibration_model_complete",
                message=(
                    f"winner for {model_name}: "
                    f"{winner.get('decode_threads')}x{winner.get('batch_threads')}x{winner.get('ubatch')}"
                ),
                phase="lane_calibration",
                lane="calibration",
                model=model_name,
            )

    quality_spec = _calibration_quality_spec(
        spec, sample_limit=int(budget.get("quality_sample_limit", 0) or 0)
    )
    quality_payload = _run_quality_evals(layout=layout, spec=quality_spec)
    kernel_summary = _empty_kernel_summary()
    kernel_iterations = int(budget.get("kernel_iterations", 0) or 0)
    if kernel_iterations > 0 and spec.kernel_microbench.target_kernels:
        kernel_spec = replace(
            spec,
            kernel_microbench=replace(
                spec.kernel_microbench,
                iterations=kernel_iterations,
                warmups=min(
                    int(spec.kernel_microbench.warmups or 0),
                    max(0, kernel_iterations // 4),
                ),
            ),
        )
        kernel_summary = _run_kernel_harness(
            layout=layout,
            spec=kernel_spec,
            preflight_payload=preflight_payload,
        )
    contracts: dict[str, Any] = {}
    quality_gates: dict[str, Any] = {}
    rejected_models: dict[str, list[str]] = {}
    for model_name, winner in winners.items():
        quality_gate = _quality_gate_for_model(quality_payload, model=model_name)
        quality_gates[model_name] = quality_gate
        safety_issues: list[str] = []
        if float(winner.get("fairness", 1.0) or 1.0) < float(
            spec.calibration_search.fairness_floor or 0.0
        ):
            safety_issues.append(
                f"fairness_below_floor={float(winner.get('fairness', 0.0) or 0.0):.3f}"
            )
        if float(winner.get("queue_wait_ms_p95", 0.0) or 0.0) > float(
            spec.calibration_search.queue_wait_p95_ceiling_ms or 0.0
        ):
            safety_issues.append(
                "queue_wait_ms_p95="
                f"{float(winner.get('queue_wait_ms_p95', 0.0) or 0.0):.3f}"
            )
        if not bool(quality_gate.get("passed", False)) or safety_issues:
            rejected_models[model_name] = [
                str(item) for item in list(quality_gate.get("issues") or [])
            ]
            rejected_models[model_name].extend(safety_issues)
            if logger is not None:
                logger.emit(
                    level="warn",
                    source="benchmark_suite",
                    event_type="calibration_quality_gate_failed",
                    message=(
                        f"calibration quality gate rejected {model_name}: "
                        + ", ".join(rejected_models[model_name])
                    ),
                    phase="lane_calibration",
                    lane="calibration",
                    model=model_name,
                    payload=dict(quality_gate.get("evidence") or {}),
                )
            continue
        hotspot_metrics = _kernel_hotspot_metrics(
            model=model_name, kernel_summary=kernel_summary
        )
        payload = _calibration_contract_payload(
            repo_root=repo_root,
            spec=spec,
            host_contract=host_contract,
            host_contract_sha256=host_contract_sha256,
            model_name=model_name,
            model_contract=dict(models.get(model_name) or {}),
            winner=winner,
            scheduler_winner=winner,
            quality_payload=quality_payload,
            kv_cache_quantization=kv_cache_quantization,
            calibration_mode=str(calibration_mode or "search"),
            calibration_source=str(calibration_source or "manual"),
            calibration_target_profile=str(
                calibration_target_profile or spec.profile_name
            ),
            workload_digest=workload_digest,
            hotspot_metrics=hotspot_metrics,
        )
        path = write_tuning_contract(
            repo_root,
            host_fingerprint=str(host_contract.get("host_fingerprint") or ""),
            model=model_name,
            payload=payload,
        )
        if logger is not None:
            logger.emit_artifact(
                source="benchmark_suite",
                kind=path.name,
                path=path,
                summary=f"wrote tuning contract for {model_name}",
                phase="lane_calibration",
                lane="calibration",
                model=model_name,
            )
        contracts[model_name] = {
            "path": str(path),
            "thread_config": dict(payload.get("thread_config") or {}),
            "continuous_config": dict(payload.get("continuous_config") or {}),
            "pager_config": dict(payload.get("pager_config") or {}),
            "admission": dict(payload.get("admission") or {}),
            "calibration_stats": dict(payload.get("calibration_stats") or {}),
        }
    if rejected_models:
        rejected = ", ".join(
            f"{model}({';'.join(issues)})"
            for model, issues in sorted(rejected_models.items())
        )
        raise RuntimeError(
            "Calibration quality gate rejected candidate contracts: " f"{rejected}"
        )
    return {
        "schema_version": "native_qsg_suite.calibration.v1",
        "contracts": contracts,
        "quality": quality_payload,
        "winners": winners,
        "thread_frontiers": thread_frontiers,
        "scheduler_frontiers": scheduler_frontiers,
        "quality_gates": quality_gates,
        "kernel_summary": kernel_summary,
        "admission": {
            "budget_tier": str(calibration_mode or "search"),
            "invocation_source": str(calibration_source or "manual"),
            "target_profile": str(calibration_target_profile or spec.profile_name),
            "workload_digest": workload_digest,
        },
    }


def _threshold_delta(
    baseline: float,
    current: float,
    *,
    higher_is_better: bool,
) -> float:
    if higher_is_better:
        return baseline - current
    return current - baseline


def _apply_tuning_baseline_gates(
    *,
    summary: dict[str, Any],
    preflight_payload: dict[str, Any],
    repo_root: Path,
    failure_rows: list[dict[str, Any]],
) -> None:
    host_contract = dict(preflight_payload.get("host_contract") or {})
    if not host_contract:
        return
    fingerprint = str(host_contract.get("host_fingerprint") or "")
    quality_payload = dict(summary.get("quality") or {})
    quality_maps = {
        "perplexity": {
            str(item.get("model") or ""): float(item.get("perplexity", 0.0) or 0.0)
            for item in list(quality_payload.get("perplexity") or [])
            if str(item.get("ablation_id") or "") == "all_on"
        },
        "confidence": {
            str(item.get("model") or ""): float(
                item.get("mean_token_confidence", 0.0) or 0.0
            )
            for item in list(quality_payload.get("confidence") or [])
            if str(item.get("ablation_id") or "") == "all_on"
        },
        "coherence": {
            str(item.get("model") or ""): float(item.get("pass_rate", 0.0) or 0.0)
            for item in list(quality_payload.get("coherence") or [])
            if str(item.get("ablation_id") or "") == "all_on"
        },
        "accuracy": {
            str(item.get("model") or ""): float(item.get("pass_rate", 0.0) or 0.0)
            for item in list(quality_payload.get("accuracy") or [])
            if str(item.get("ablation_id") or "") == "all_on"
        },
    }
    checks: list[dict[str, Any]] = []
    for model_summary in list(summary.get("models") or []):
        model_name = str(model_summary.get("model") or "")
        contract, _ = load_tuning_contract(repo_root, fingerprint, model_name)
        if contract is None:
            continue
        calibration_stats = dict(contract.get("calibration_stats") or {})
        model_checks: list[dict[str, Any]] = []

        def _record_check(
            *,
            metric: str,
            current: float,
            baseline: float,
            mad: float,
            higher_is_better: bool,
            absolute_floor_pct: float = 0.0,
        ) -> None:
            delta = _threshold_delta(
                baseline, current, higher_is_better=higher_is_better
            )
            warn_threshold = max(3.0 * mad, abs(baseline) * absolute_floor_pct)
            fail_threshold = max(5.0 * mad, abs(baseline) * absolute_floor_pct)
            status = "pass"
            if delta > fail_threshold:
                status = "fail"
            elif delta > warn_threshold:
                status = "warn"
            entry = {
                "metric": metric,
                "status": status,
                "current": current,
                "baseline": baseline,
                "mad": mad,
                "delta": delta,
            }
            model_checks.append(entry)
            if status == "fail":
                failure_rows.append(
                    {
                        "run_id": str(summary.get("run_id") or ""),
                        "attempt_id": "",
                        "model": model_name,
                        "error": f"tuning_baseline:{metric}",
                        "error_type": "CalibrationRegression",
                        "failure_kind": "gate_failure",
                        "gate_issues": [f"{metric}_regression"],
                        "normalized_issues": [f"{metric}_regression"],
                        "traceback": "",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        _record_check(
            metric="decode_tps",
            current=float(model_summary.get("decode_tps_p50", 0.0) or 0.0),
            baseline=float(calibration_stats.get("decode_tps_median", 0.0) or 0.0),
            mad=float(calibration_stats.get("decode_tps_mad", 0.0) or 0.0),
            higher_is_better=True,
            absolute_floor_pct=0.03,
        )
        _record_check(
            metric="ttft_ms",
            current=float(model_summary.get("ttft_ms_p95", 0.0) or 0.0),
            baseline=float(calibration_stats.get("ttft_ms_median", 0.0) or 0.0),
            mad=float(calibration_stats.get("ttft_ms_mad", 0.0) or 0.0),
            higher_is_better=False,
            absolute_floor_pct=0.05,
        )
        kernel_metrics = dict(calibration_stats.get("kernel_metrics") or {})
        stage_map = {
            str(item.get("stage") or ""): float(item.get("total_ms_mean", 0.0) or 0.0)
            for item in list(model_summary.get("stage_hotspots") or [])
            if str(item.get("stage") or "").strip()
        }
        for stage_name, baseline_metrics in kernel_metrics.items():
            current_value = float(stage_map.get(stage_name, 0.0) or 0.0)
            _record_check(
                metric=f"kernel:{stage_name}",
                current=current_value,
                baseline=float(dict(baseline_metrics).get("median_ms", 0.0) or 0.0),
                mad=float(dict(baseline_metrics).get("mad_ms", 0.0) or 0.0),
                higher_is_better=False,
            )
        quality_metrics = dict(calibration_stats.get("quality_metrics") or {})
        _record_check(
            metric="quality:perplexity",
            current=float(quality_maps["perplexity"].get(model_name, 0.0) or 0.0),
            baseline=float(
                dict(quality_metrics.get("perplexity") or {}).get("median", 0.0) or 0.0
            ),
            mad=float(
                dict(quality_metrics.get("perplexity") or {}).get("mad", 0.0) or 0.0
            ),
            higher_is_better=False,
        )
        _record_check(
            metric="quality:confidence",
            current=float(quality_maps["confidence"].get(model_name, 0.0) or 0.0),
            baseline=float(
                dict(quality_metrics.get("confidence") or {}).get("median", 0.0) or 0.0
            ),
            mad=float(
                dict(quality_metrics.get("confidence") or {}).get("mad", 0.0) or 0.0
            ),
            higher_is_better=True,
        )
        _record_check(
            metric="quality:coherence",
            current=float(quality_maps["coherence"].get(model_name, 0.0) or 0.0),
            baseline=float(
                dict(quality_metrics.get("coherence") or {}).get("median", 0.0) or 0.0
            ),
            mad=float(
                dict(quality_metrics.get("coherence") or {}).get("mad", 0.0) or 0.0
            ),
            higher_is_better=True,
        )
        _record_check(
            metric="quality:accuracy",
            current=float(quality_maps["accuracy"].get(model_name, 0.0) or 0.0),
            baseline=float(
                dict(quality_metrics.get("accuracy") or {}).get("median", 0.0) or 0.0
            ),
            mad=float(
                dict(quality_metrics.get("accuracy") or {}).get("mad", 0.0) or 0.0
            ),
            higher_is_better=True,
        )
        model_summary["tuning_baseline_checks"] = model_checks
        checks.append({"model": model_name, "checks": model_checks})
    summary["tuning_baseline_checks"] = checks


def _run_kernel_harness(
    *,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
) -> dict[str, Any]:
    from benchmarks.native_kernel_microbench import run_kernel_microbench

    results: list[dict[str, Any]] = []
    logger = get_active_logger()
    perf_payload = dict(preflight_payload.get("perf") or {})
    degraded_reason = None
    if not bool(perf_payload.get("available", False)):
        degraded_reason = str(perf_payload.get("reason") or "perf_unavailable")
    for model in spec.models:
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="kernel_microbench_start",
                message=f"starting kernel microbench for {model}",
                phase="kernel_microbench",
                lane="kernel_microbench",
                model=model,
            )
        payload = run_kernel_microbench(
            model=model,
            context_length=spec.scenario_pack.canonical_context_length,
            warmups=spec.kernel_microbench.warmups,
            iterations=spec.kernel_microbench.iterations,
            target_kernels=spec.kernel_microbench.target_kernels,
            synthetic_lengths=spec.kernel_microbench.synthetic_lengths,
            degraded_reason=degraded_reason,
        )
        result_path = layout.kernel_dir / f"{_safe_name(model)}.json"
        _write_json_artifact(
            layout=layout,
            path=result_path,
            payload=payload,
            summary=f"wrote kernel microbench payload for {model}",
            phase="kernel_microbench",
            lane="kernel_microbench",
            model=model,
        )
        for row in list(payload.get("runs") or []):
            append_ndjson(layout.kernel_attempts_ndjson, row)
        results.append(payload)
        if logger is not None:
            logger.emit(
                level="info",
                source="benchmark_suite",
                event_type="kernel_microbench_complete",
                message=f"completed kernel microbench for {model}",
                phase="kernel_microbench",
                lane="kernel_microbench",
                model=model,
            )
    summary = {
        "schema_version": "native_qsg_suite.kernel_summary.v1",
        "degraded_reason": degraded_reason,
        "models": results,
    }
    _write_json_artifact(
        layout=layout,
        path=layout.kernel_summary_json,
        payload=summary,
        summary="wrote kernel summary",
        phase="kernel_microbench",
    )
    return summary


def _run_memory_replay_lane(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
) -> dict[str, Any]:
    logger = get_active_logger()
    config = spec.memory_replay
    if not str(config.cases_path or "").strip():
        payload = _empty_memory_replay_summary()
        _write_json_artifact(
            layout=layout,
            path=layout.memory_replay_summary_json,
            payload=payload,
            summary="memory replay lane skipped",
            phase="memory_replay",
            lane="memory_replay",
        )
        return payload
    cmd = [
        sys.executable,
        str(
            (repo_root / "benchmarks" / "almf_retrieval_replay_benchmark.py").resolve()
        ),
        "--db-path",
        str(config.db_path),
        "--campaign-id",
        str(config.campaign_id),
        "--cases",
        str(config.cases_path),
        "--out-root",
        str(layout.eval_dir / "memory_replay_runs"),
        "--run-id",
        f"{layout.run_id}_memory_replay",
    ]
    if str(config.storage_root or "").strip():
        cmd.extend(["--storage-root", str(config.storage_root)])
    if logger is not None:
        logger.emit(
            level="info",
            source="benchmark_suite",
            event_type="memory_replay_start",
            message="starting memory replay lane",
            phase="memory_replay",
            lane="memory_replay",
        )
    completed = run_logged_subprocess(
        cmd=cmd,
        cwd=repo_root,
        env=os.environ.copy(),
        source="memory_replay_benchmark",
        phase="memory_replay",
        lane="memory_replay",
        stdout_path=layout.eval_dir / "memory_replay.stdout.log",
        stderr_path=layout.eval_dir / "memory_replay.stderr.log",
    )
    if completed.returncode != 0:
        payload = {
            "schema_version": "native_qsg_suite.memory_replay.v1",
            "state": "failed",
            "reason": str(
                completed.stderr or completed.stdout or "memory_replay_failed"
            ),
            "results": {},
        }
    else:
        payload = json.loads(str(completed.stdout or "{}") or "{}")
        payload["schema_version"] = "native_qsg_suite.memory_replay.v1"
        payload["state"] = "completed"
        payload["gate_thresholds"] = dict(config.gate_thresholds)
    _write_json_artifact(
        layout=layout,
        path=layout.memory_replay_summary_json,
        payload=payload,
        summary="wrote memory replay summary",
        phase="memory_replay",
        lane="memory_replay",
    )
    return payload


def _compat_baseline_candidates(base_dir: Path, current_run_id: str) -> list[Path]:
    candidates: list[Path] = []
    for root in (base_dir, base_dir / "runs"):
        if not root.exists():
            continue
        for path in root.rglob("summary.json"):
            if current_run_id in str(path):
                continue
            candidates.append(path)
    return sorted(set(candidates), reverse=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    return load_optional_json_dict(path)


def _resolve_baseline(
    *,
    audit_root: Path,
    current_run_id: str,
    model_set: list[str],
    compare_to: str = "latest_compatible",
    current_topology_hash: str = "",
    current_host_fingerprint: str = "",
    current_prompt_contract_hash: str = "",
) -> dict[str, Any] | None:
    for candidate in _compat_baseline_candidates(audit_root, current_run_id):
        summary = _load_json(candidate)
        if not summary:
            continue
        models = [
            str(item.get("model") or "")
            for item in list(summary.get("models") or [])
            if isinstance(item, dict)
        ]
        if models and sorted(models) == sorted(model_set):
            if str(compare_to or "") == "last_same_topology":
                candidate_hash = str(
                    (summary.get("topology_passport") or {}).get("topology_hash") or ""
                )
                if current_topology_hash and candidate_hash != current_topology_hash:
                    continue
            elif str(compare_to or "") == "last_same_host":
                candidate_host = str(
                    ((summary.get("host_compliance") or {}).get("host") or {}).get(
                        "host_fingerprint"
                    )
                    or ""
                )
                if (
                    current_host_fingerprint
                    and candidate_host != current_host_fingerprint
                ):
                    continue
            elif str(compare_to or "") == "last_green_main":
                if not bool(summary.get("overall_pass", False)):
                    continue
            elif str(compare_to or "") == "last_same_prompt_contract":
                candidate_prompt_hash = str(
                    (summary.get("comparisons") or {}).get("prompt_contract_hash")
                    or str(summary.get("prompt_contract_hash") or "")
                )
                if (
                    current_prompt_contract_hash
                    and candidate_prompt_hash != current_prompt_contract_hash
                ):
                    continue
            return summary
    return None


def _resolve_comparator_catalog(
    *,
    audit_root: Path,
    current_run_id: str,
    model_set: list[str],
    current_topology_hash: str,
    current_host_fingerprint: str,
    current_prompt_contract_hash: str,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for mode in (
        "latest_compatible",
        "last_same_topology",
        "last_same_host",
        "last_green_main",
        "last_same_prompt_contract",
    ):
        baseline = _resolve_baseline(
            audit_root=audit_root,
            current_run_id=current_run_id,
            model_set=model_set,
            compare_to=mode,
            current_topology_hash=current_topology_hash,
            current_host_fingerprint=current_host_fingerprint,
            current_prompt_contract_hash=current_prompt_contract_hash,
        )
        if baseline:
            catalog[mode] = baseline
    return catalog


def _quality_index(
    quality_payload: dict[str, Any], key: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for item in list(quality_payload.get(key) or []):
        if not isinstance(item, dict):
            continue
        indexed[(str(item.get("model") or ""), str(item.get("ablation_id") or ""))] = (
            item
        )
    return indexed


def _ablation_deltas(
    quality_payload: dict[str, Any],
    attempt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    confidence_by_key = _quality_index(quality_payload, "confidence")
    perplexity_by_key = _quality_index(quality_payload, "perplexity")
    canonical_rows = [
        row
        for row in attempt_rows
        if str(row.get("lane_id") or "") == "canonical_all_on"
        and str(row.get("ablation_id") or "") == "all_on"
        and not bool(row.get("warmup", False))
    ]
    baseline_by_model: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        baseline_by_model.setdefault(str(row.get("model_id") or ""), row)
    deltas: list[dict[str, Any]] = []
    for row in attempt_rows:
        if bool(row.get("warmup", False)):
            continue
        ablation_id = str(row.get("ablation_id") or "")
        if ablation_id in {"", "all_on"}:
            continue
        model = str(row.get("model_id") or "")
        baseline = baseline_by_model.get(model)
        if baseline is None:
            continue
        baseline_decode = float(
            (baseline.get("throughput") or {}).get("decode_tps", 0.0) or 0.0
        )
        baseline_ttft = float(
            (baseline.get("latency") or {}).get("ttft_ms", 0.0) or 0.0
        )
        current_decode = float(
            (row.get("throughput") or {}).get("decode_tps", 0.0) or 0.0
        )
        current_ttft = float((row.get("latency") or {}).get("ttft_ms", 0.0) or 0.0)
        base_ppl = float(
            (perplexity_by_key.get((model, "all_on")) or {}).get("perplexity", 0.0)
            or 0.0
        )
        curr_ppl = float(
            (perplexity_by_key.get((model, ablation_id)) or {}).get("perplexity", 0.0)
            or 0.0
        )
        base_conf = float(
            (confidence_by_key.get((model, "all_on")) or {}).get(
                "mean_token_confidence", 0.0
            )
            or 0.0
        )
        curr_conf = float(
            (confidence_by_key.get((model, ablation_id)) or {}).get(
                "mean_token_confidence", 0.0
            )
            or 0.0
        )
        deltas.append(
            {
                "model": model,
                "ablation_id": ablation_id,
                "decode_tps_delta_pct": (
                    ((current_decode - baseline_decode) / baseline_decode) * 100.0
                    if baseline_decode > 0.0
                    else 0.0
                ),
                "ttft_delta_pct": (
                    ((current_ttft - baseline_ttft) / baseline_ttft) * 100.0
                    if baseline_ttft > 0.0
                    else 0.0
                ),
                "perplexity_delta_pct": (
                    ((curr_ppl - base_ppl) / base_ppl) * 100.0
                    if base_ppl > 0.0
                    else 0.0
                ),
                "confidence_delta_pct": (
                    ((curr_conf - base_conf) / base_conf) * 100.0
                    if base_conf > 0.0
                    else 0.0
                ),
            }
        )
    return deltas


def _fuse_hotspots(
    attempt_rows: list[dict[str, Any]],
    kernel_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    canonical_rows = [
        row
        for row in attempt_rows
        if str(row.get("lane_id") or "") == "canonical_all_on"
        and str(row.get("ablation_id") or "") == "all_on"
        and not bool(row.get("warmup", False))
    ]
    by_model_kernel: dict[tuple[str, str], dict[str, Any]] = {}
    for model_payload in list(kernel_summary.get("models") or []):
        model = str(model_payload.get("model") or "")
        for row in list(model_payload.get("runs") or []):
            if not isinstance(row, dict):
                continue
            by_model_kernel[(model, str(row.get("kernel") or ""))] = row
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in canonical_rows:
        runtime = dict(row.get("runtime") or {})
        stage_ms = dict(runtime.get("graph_stage_ms") or {})
        decode_ms = float(runtime.get("runtime_decode_seconds", 0.0) or 0.0) * 1000.0
        if decode_ms <= 0.0:
            continue
        model = str(row.get("model_id") or "")
        for kernel, total_ms in stage_ms.items():
            grouped.setdefault((model, str(kernel)), []).append(
                (float(total_ms) / decode_ms) * 100.0
            )

    fused: list[dict[str, Any]] = []
    for (model, kernel), pct_samples in grouped.items():
        kernel_row = by_model_kernel.get((model, kernel), {})
        microbench_cv_pct = float(kernel_row.get("cv_pct", 0.0) or 0.0)
        attempts_cv_pct = (
            float(
                statistics.pstdev(pct_samples) / statistics.fmean(pct_samples) * 100.0
            )
            if len(pct_samples) > 1 and statistics.fmean(pct_samples) > 0.0
            else 0.0
        )
        combined_cv_pct = (microbench_cv_pct + attempts_cv_pct) / 2.0
        stability_weight = max(0.2, 1.0 - combined_cv_pct / 100.0)
        recoverable_gain_pct = float(
            kernel_row.get("estimated_recoverable_gain_pct", 0.0) or 0.0
        )
        recoverable_weight = max(0.1, min(1.0, recoverable_gain_pct / 100.0))
        calls_mean = float(kernel_row.get("calls_mean", 0.0) or 0.0)
        call_density_weight = max(0.1, min(1.0, calls_mean / 1024.0))
        pct_of_decode = float(statistics.fmean(pct_samples))
        impact_score = (
            pct_of_decode * stability_weight * recoverable_weight * call_density_weight
        )
        stability_confidence = max(
            0.0,
            min(1.0, 1.0 - ((combined_cv_pct / 100.0) * 0.7)),
        )
        fused.append(
            {
                "model": model,
                "lane": "canonical_all_on",
                "kernel": kernel,
                "cpp_file": str(kernel_row.get("cpp_file") or "unknown"),
                "cpp_function": str(kernel_row.get("cpp_function") or "unknown"),
                "pct_of_decode": pct_of_decode,
                "cv_pct": microbench_cv_pct,
                "stability_across_attempts_cv_pct": attempts_cv_pct,
                "hotspot_confidence": stability_confidence,
                "impact_score": impact_score,
                "artifact_refs": {
                    "kernel_summary": str(
                        (Path("kernel") / f"{_safe_name(model)}.json")
                    ),
                    "native_attempts": "native/attempts.ndjson",
                },
                "why_hot": f"{kernel} owns {pct_of_decode:.2f}% of decode time on the canonical lane",
                "how_stable": (
                    f"kernel_cv={microbench_cv_pct:.2f}%, "
                    f"attempt_cv={attempts_cv_pct:.2f}%"
                ),
                "estimated_recoverable_gain_pct": recoverable_gain_pct,
                "evidence": {
                    "sample_count": len(pct_samples),
                    "microbench_cv_pct": microbench_cv_pct,
                    "attempts_cv_pct": attempts_cv_pct,
                    "recoverable_gain_pct": recoverable_gain_pct,
                },
            }
        )
    fused.sort(key=lambda item: float(item["impact_score"]), reverse=True)
    return fused


def _build_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Native QSG Benchmark Suite",
        "",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Overall pass: `{summary.get('overall_pass')}`",
        f"- Certification state: `{summary.get('certification_state', 'unknown')}`",
        "",
        "## Models",
    ]
    for model in list(summary.get("models") or []):
        if not isinstance(model, dict):
            continue
        lines.append(
            f"- `{model.get('model')}` decode_tps_p50={float(model.get('decode_tps_p50', 0.0)):.3f} "
            f"ttft_ms_p95={float(model.get('ttft_ms_p95', 0.0)):.3f}"
        )
    lines.extend(["", "## Hotspots"])
    for hotspot in list(summary.get("kernel_hotspots") or [])[:10]:
        lines.append(
            f"- `{hotspot.get('model')}` `{hotspot.get('kernel')}` "
            f"impact={float(hotspot.get('impact_score', 0.0)):.3f} "
            f"decode_share={float(hotspot.get('pct_of_decode', 0.0)):.2f}%"
        )
    triage = dict(summary.get("agent_triage") or {})
    if triage:
        lines.extend(
            [
                "",
                "## Triage",
                f"- Next action: `{triage.get('next_action')}`",
            ]
        )
    variance = dict(summary.get("variance_budget") or {})
    lineage = dict(summary.get("baseline_lineage") or {})
    closure = dict(summary.get("closure_result") or {})
    spc = dict(summary.get("spc_report") or {})
    traceability = dict(summary.get("traceability_graph") or {})
    advisory_bundle = dict(summary.get("advisory_bundle") or {})
    memory_replay = dict(summary.get("memory_replay") or {})
    if variance:
        lines.extend(
            [
                "",
                "## Variance Budget",
                f"- Within budget: `{((variance.get('overall') or {}).get('within_budget'))}`",
                f"- Rerun recommended: `{((variance.get('overall') or {}).get('rerun_recommended'))}`",
            ]
        )
    if lineage:
        lines.extend(
            [
                "",
                "## Comparator Lineage",
                f"- Comparator mode: `{lineage.get('comparator_mode')}`",
                f"- Comparable: `{lineage.get('comparable')}`",
                f"- Topology match: `{lineage.get('topology_match')}`",
            ]
        )
    if closure:
        lines.extend(
            [
                "",
                "## Closure Coverage",
                f"- Closure pass: `{closure.get('overall_pass')}`",
                f"- Unresolved items: `{len(closure.get('unresolved') or [])}`",
            ]
        )
    if spc:
        lines.extend(
            [
                "",
                "## SPC Drift Status",
                f"- Status: `{spc.get('status')}`",
                f"- History runs considered: `{spc.get('history_runs_considered')}`",
            ]
        )
    if traceability:
        lines.extend(
            [
                "",
                "## Prompt And Memory Traceability",
                f"- Traceability nodes: `{len(traceability.get('nodes') or [])}`",
                f"- Prompt contract hash: `{summary.get('prompt_contract_hash', '')}`",
                f"- Memory snapshot hash: `{summary.get('memory_snapshot_hash', '')}`",
                f"- Feature toggle hash: `{summary.get('feature_toggle_hash', '')}`",
            ]
        )
    if memory_replay:
        lines.extend(
            [
                "",
                "## Memory Replay",
                f"- State: `{memory_replay.get('state') or memory_replay.get('status')}`",
                f"- Gate pass: `{((memory_replay.get('results') or {}).get('benchmark_gates') or {}).get('passed', False)}`",
            ]
        )
    if advisory_bundle:
        stage_graph = dict(advisory_bundle.get("stage_graph") or {})
        impact_gate = dict(advisory_bundle.get("semantic_change_impact_gate") or {})
        lines.extend(
            [
                "",
                "## Advisory Systems",
                f"- Stage graph changed: `{stage_graph.get('changed', False)}`",
                f"- Semantic impact status: `{impact_gate.get('status', '')}`",
                f"- Scheduler early stop: `{((advisory_bundle.get('self_calibrating_scheduler') or {}).get('early_stop_recommended', False))}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_report_html(summary: dict[str, Any]) -> str:
    rows = []
    for hotspot in list(summary.get("kernel_hotspots") or [])[:20]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(hotspot.get('model') or ''))}</td>"
            f"<td>{html.escape(str(hotspot.get('kernel') or ''))}</td>"
            f"<td>{float(hotspot.get('impact_score', 0.0)):.3f}</td>"
            f"<td>{float(hotspot.get('pct_of_decode', 0.0)):.2f}%</td>"
            f"<td>{html.escape(str(hotspot.get('cpp_file') or ''))}</td>"
            "</tr>"
        )
    return (
        "<html><head><title>Native QSG Benchmark Suite</title></head><body>"
        f"<h1>Run {html.escape(str(summary.get('run_id') or ''))}</h1>"
        f"<p>Overall pass: {html.escape(str(summary.get('overall_pass')))}</p>"
        f"<p>Variance status: {html.escape(str(((summary.get('variance_budget') or {}).get('overall') or {}).get('within_budget')))}</p>"
        f"<p>Comparator: {html.escape(str((summary.get('baseline_lineage') or {}).get('comparator_mode') or ''))}</p>"
        f"<p>SPC: {html.escape(str((summary.get('spc_report') or {}).get('status') or ''))}</p>"
        "<table border='1' cellpadding='4' cellspacing='0'>"
        "<tr><th>Model</th><th>Kernel</th><th>Impact</th><th>Decode Share</th><th>C++ File</th></tr>"
        + "".join(rows)
        + "</table></body></html>"
    )


def _build_executive_summary(summary: dict[str, Any]) -> str:
    triage = dict(summary.get("agent_triage") or {})
    top = dict(triage.get("top_hotspot") or {})
    return (
        "# Executive Summary\n\n"
        f"- Run ID: `{summary.get('run_id')}`\n"
        f"- Overall pass: `{summary.get('overall_pass')}`\n"
        f"- Certification state: `{summary.get('certification_state', 'unknown')}`\n"
        f"- Failures: `{summary.get('failure_count')}`\n"
        f"- Top hotspot: `{top.get('model', 'n/a')}` / `{top.get('kernel', 'n/a')}`\n"
        f"- Next action: `{triage.get('next_action', 'No hotspot available')}`\n"
        f"- Comparator: `{((summary.get('baseline_lineage') or {}).get('comparator_mode') or 'latest_compatible')}`\n"
        f"- SPC status: `{((summary.get('spc_report') or {}).get('status') or 'unknown')}`\n"
    )


def _build_publication_manifest(
    *, layout: SuiteRunLayout, summary: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "native_qsg_suite.publication_manifest.v1",
        "run_id": str(summary.get("run_id") or ""),
        "profile_name": str(summary.get("profile_name") or ""),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "publishable": bool(summary.get("overall_pass", False))
        and str(summary.get("profile_name") or "") == "platinum",
        "certification_state": str(summary.get("certification_state") or ""),
        "compare_to": str((summary.get("comparisons") or {}).get("compare_to") or ""),
        "baseline_run_id": str(summary.get("baseline_run_id") or ""),
        "summary_path": layout.summary_json.relative_to(layout.root).as_posix(),
        "metrics_rollup_path": layout.metrics_rollup_json.relative_to(
            layout.root
        ).as_posix(),
        "agent_handoff_path": layout.agent_handoff_json.relative_to(
            layout.root
        ).as_posix(),
        "quality_summary_path": layout.quality_summary_json.relative_to(
            layout.root
        ).as_posix(),
        "kernel_summary_path": layout.kernel_summary_json.relative_to(
            layout.root
        ).as_posix(),
        "models": [
            {
                "model": str(item.get("model") or ""),
                "decode_tps_p50": float(item.get("decode_tps_p50", 0.0) or 0.0),
                "ttft_ms_p95": float(item.get("ttft_ms_p95", 0.0) or 0.0),
                "continuous_decode_tps_global_p50": float(
                    item.get("continuous_decode_tps_global_p50", 0.0) or 0.0
                ),
                "accuracy_pass_rate": float(item.get("accuracy_pass_rate", 0.0) or 0.0),
                "coherence_pass_rate": float(
                    item.get("coherence_pass_rate", 0.0) or 0.0
                ),
            }
            for item in list(summary.get("models") or [])
            if isinstance(item, dict)
        ],
    }


def _write_reports(layout: SuiteRunLayout, summary: dict[str, Any]) -> None:
    summary_md = _build_summary_markdown(summary)
    layout.summary_md.write_text(summary_md, encoding="utf-8")
    layout.report_md.write_text(summary_md, encoding="utf-8")
    layout.executive_summary_md.write_text(
        _build_executive_summary(summary), encoding="utf-8"
    )
    layout.report_html.write_text(_build_report_html(summary), encoding="utf-8")
    publication_manifest = dict(summary.get("publication_manifest") or {})
    publication_manifest_path = layout.reports_dir / "publication_manifest.json"
    if publication_manifest:
        publication_manifest_path.write_text(
            json.dumps(publication_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    logger = get_active_logger()
    if logger is not None:
        artifact_rows = [
            (layout.summary_md, "wrote summary markdown"),
            (layout.report_md, "wrote report markdown"),
            (layout.executive_summary_md, "wrote executive summary"),
            (layout.report_html, "wrote html report"),
        ]
        if publication_manifest:
            artifact_rows.append(
                (publication_manifest_path, "wrote publication manifest")
            )
        for path, summary_text in artifact_rows:
            logger.emit_artifact(
                source="benchmark_suite",
                kind=path.name,
                path=path,
                summary=summary_text,
                phase="finalizing",
            )
    _write_json_artifact(
        layout=layout,
        path=layout.metrics_rollup_json,
        payload=_build_metrics_rollup(summary),
        summary="wrote metrics rollup",
        phase="finalizing",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.agent_handoff_json,
        payload=_build_agent_handoff(summary),
        summary="wrote agent handoff",
        phase="finalizing",
    )


def _emit_preflight_failure(
    layout: SuiteRunLayout, payload: dict[str, Any], profile_name: str
) -> None:
    failures = [
        str(item) for item in list(payload.get("failures") or []) if str(item).strip()
    ]
    remediations = dict(payload.get("remediations") or {})
    _terminal(f"Benchmark suite preflight failed for profile `{profile_name}`.")
    _terminal(f"Run folder: {layout.root}")
    for failure in failures:
        _terminal(f"- {failure}")
        remediation = str(remediations.get(failure) or "").strip()
        if remediation:
            _terminal(f"  remediation: {remediation}")
    if profile_name == "gold":
        _terminal(
            "This host can still run the reduced non-strict suite with: ./scripts/run_native_qsg_suite.sh --profile bronze"
        )


def _emit_terminal_result(layout: SuiteRunLayout, summary: dict[str, Any]) -> None:
    _terminal(f"Benchmark suite finished. Run folder: {layout.root}")
    _terminal(f"Overall pass: {summary.get('overall_pass')}")
    _terminal(f"Certification state: {summary.get('certification_state', 'unknown')}")
    triage = dict(summary.get("agent_triage") or {})
    next_action = str(triage.get("next_action") or "").strip()
    if next_action:
        _terminal(f"Next action: {next_action}")


def _build_triage(summary: dict[str, Any]) -> dict[str, Any]:
    hotspots = list(summary.get("kernel_hotspots") or [])
    top = hotspots[0] if hotspots else {}
    return {
        "schema_version": "native_qsg_suite.triage.v1",
        "run_id": summary.get("run_id"),
        "top_hotspot": top,
        "next_action": (
            f"Inspect {top.get('cpp_file')}::{top.get('cpp_function')}"
            if top
            else "No hotspot available"
        ),
    }


def _verify_required_artifacts(layout: SuiteRunLayout) -> list[str]:
    missing: list[str] = []
    for path in required_suite_artifacts(layout):
        if not path.exists():
            missing.append(str(path))
    return missing


def _empty_kernel_summary() -> dict[str, Any]:
    return {"schema_version": "native_qsg_suite.kernel_summary.v1", "models": []}


def _empty_quality_summary() -> dict[str, Any]:
    payload = {
        "schema_version": "native_qsg_suite.quality.v1",
        "state": "pending",
        "perplexity": [],
        "confidence": [],
        "coherence": [],
        "accuracy": [],
    }
    payload["governance"] = _quality_governance_report(payload)
    return payload


def _empty_memory_replay_summary() -> dict[str, Any]:
    return {
        "schema_version": "native_qsg_suite.memory_replay.v1",
        "state": "skipped",
        "reason": "memory_replay_lane_not_requested",
        "results": {},
    }


def _is_speculative_row(row: dict[str, Any]) -> bool:
    generation_mode = str(row.get("generation_mode") or "").strip().lower()
    benchmark_label = str(row.get("benchmark_label") or "").strip().lower()
    if bool(row.get("speculative_decode", False)):
        return True
    if int(row.get("accepted_parallel_tokens", 0) or 0) > 0:
        return True
    if int(row.get("rejected_parallel_tokens", 0) or 0) > 0:
        return True
    return any(
        marker in generation_mode or marker in benchmark_label
        for marker in ("spec", "jacobi", "medusa", "hydra", "lookup", "parallel")
    )


def _is_non_ar_row(row: dict[str, Any]) -> bool:
    generation_mode = str(row.get("generation_mode") or "").strip().lower()
    benchmark_label = str(row.get("benchmark_label") or "").strip().lower()
    return any(
        marker in generation_mode or marker in benchmark_label
        for marker in ("diffusion", "masked", "block")
    )


def _strict_native_decode_receipt(
    spec: BenchmarkSuiteSpec,
    attempt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    required = bool(spec.force_parallel_decode or spec.forbid_autoregressive_fallback)
    measured_rows = [
        dict(row)
        for row in attempt_rows
        if isinstance(row, dict) and not bool(row.get("warmup", False))
    ]
    observed_modes = sorted(
        {
            str(row.get("generation_mode") or "").strip().lower()
            for row in measured_rows
            if str(row.get("generation_mode") or "").strip()
        }
    )
    observed_labels = sorted(
        {
            str(row.get("benchmark_label") or "").strip().lower()
            for row in measured_rows
            if str(row.get("benchmark_label") or "").strip()
        }
    )
    observed_non_ar_modes = sorted(
        {
            str(row.get("generation_mode") or "").strip().lower()
            for row in measured_rows
            if _is_non_ar_row(row) and str(row.get("generation_mode") or "").strip()
        }
    )
    observed_ar_attempt_ids = sorted(
        {
            str(row.get("attempt_id") or "").strip()
            for row in measured_rows
            if str(row.get("generation_mode") or "").strip().lower()
            in AR_GENERATION_MODES
            and str(row.get("attempt_id") or "").strip()
        }
    )
    issues: list[str] = []
    missing_generation_mode_count = sum(
        1 for row in measured_rows if not str(row.get("generation_mode") or "").strip()
    )
    if required and missing_generation_mode_count:
        issues.append(f"generation_mode_missing:{missing_generation_mode_count}")
    if spec.forbid_autoregressive_fallback:
        observed_ar_modes = sorted(
            {
                str(row.get("generation_mode") or "").strip().lower()
                for row in measured_rows
                if str(row.get("generation_mode") or "").strip().lower()
                in AR_GENERATION_MODES
            }
        )
        if observed_ar_modes:
            issues.append(
                "observed_autoregressive_modes:" + ",".join(observed_ar_modes)
            )
    else:
        observed_ar_modes = []

    missing_force_env: list[str] = []
    missing_forbid_env: list[str] = []
    missing_disable_env: list[str] = []
    for row in measured_rows:
        attempt_id = str(row.get("attempt_id") or "").strip() or "<unknown>"
        provenance = dict(row.get("provenance") or {})
        env_overrides = dict(provenance.get("env_overrides") or {})
        if (
            spec.force_parallel_decode
            and str(env_overrides.get("ANVIL_FORCE_PARALLEL_DECODE") or "") != "1"
        ):
            missing_force_env.append(attempt_id)
        if (
            spec.forbid_autoregressive_fallback
            and str(env_overrides.get("ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK") or "")
            != "1"
        ):
            missing_forbid_env.append(attempt_id)
        if (
            spec.forbid_autoregressive_fallback
            and str(env_overrides.get("ANVIL_PARALLEL_AR_RECOVERY_ENABLED") or "")
            != "0"
        ):
            missing_disable_env.append(attempt_id)
    if missing_force_env:
        issues.append(
            "force_parallel_env_missing:" + ",".join(sorted(missing_force_env))
        )
    if missing_forbid_env:
        issues.append(
            "forbid_autoregressive_env_missing:" + ",".join(sorted(missing_forbid_env))
        )
    if missing_disable_env:
        issues.append(
            "parallel_ar_recovery_disable_missing:"
            + ",".join(sorted(missing_disable_env))
        )

    return {
        "required": required,
        "force_parallel_decode": bool(spec.force_parallel_decode),
        "forbid_autoregressive_fallback": bool(spec.forbid_autoregressive_fallback),
        "attempt_count": len(measured_rows),
        "observed_generation_modes": observed_modes,
        "observed_benchmark_labels": observed_labels,
        "observed_non_ar_modes": observed_non_ar_modes,
        "observed_ar_modes": observed_ar_modes,
        "observed_ar_attempt_ids": observed_ar_attempt_ids,
        "passed": not issues,
        "issues": issues,
    }


def _append_strict_native_decode_failures(
    *,
    summary: dict[str, Any],
    receipt: dict[str, Any],
    failure_rows: list[dict[str, Any]],
) -> None:
    if not bool(receipt.get("required", False)) or bool(receipt.get("passed", False)):
        return
    error = "strict_native_decode_contract_failed"
    if any(str(item.get("error") or "") == error for item in failure_rows):
        return
    issues = [
        str(item) for item in list(receipt.get("issues") or []) if str(item).strip()
    ]
    failure_rows.append(
        {
            "run_id": str(summary.get("run_id") or ""),
            "attempt_id": "",
            "model": "",
            "error": error,
            "error_type": "StrictNativeDecodeFailure",
            "failure_kind": "gate_failure",
            "gate_issues": issues,
            "normalized_issues": issues,
            "traceback": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )


def _group_speculative_acceptance(
    attempt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in attempt_rows:
        if not isinstance(row, dict) or not _is_speculative_row(row):
            continue
        prompt_category = str(row.get("prompt_category") or "unknown")
        temperature_band = str(row.get("temperature_band") or "unknown")
        generation_mode = str(row.get("generation_mode") or "unknown")
        key = (prompt_category, temperature_band, generation_mode)
        bucket = grouped.setdefault(
            key,
            {
                "prompt_category": prompt_category,
                "temperature_band": temperature_band,
                "generation_mode": generation_mode,
                "runs": 0,
                "accepted_parallel_tokens": 0,
                "rejected_parallel_tokens": 0,
            },
        )
        bucket["runs"] += 1
        bucket["accepted_parallel_tokens"] += int(
            row.get("accepted_parallel_tokens", 0) or 0
        )
        bucket["rejected_parallel_tokens"] += int(
            row.get("rejected_parallel_tokens", 0) or 0
        )
    rows: list[dict[str, Any]] = []
    for item in sorted(
        grouped.values(),
        key=lambda row: (
            str(row["prompt_category"]),
            str(row["temperature_band"]),
            str(row["generation_mode"]),
        ),
    ):
        attempts = int(item["accepted_parallel_tokens"]) + int(
            item["rejected_parallel_tokens"]
        )
        item["speculative_attempts"] = attempts
        item["acceptance_rate"] = (
            float(item["accepted_parallel_tokens"]) / float(attempts)
            if attempts
            else 0.0
        )
        rows.append(item)
    return rows


def _acceptance_governance_report(
    *,
    summary: dict[str, Any],
    quality_payload: dict[str, Any],
    attempt_rows: list[dict[str, Any]],
    resume_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _runtime_metric(row: dict[str, Any], key: str) -> Any:
        if key in row:
            return row.get(key)
        runtime = row.get("runtime")
        if isinstance(runtime, dict):
            return runtime.get(key)
        return None

    quality_governance = dict(
        quality_payload.get("governance") or _quality_governance_report(quality_payload)
    )
    artifact_checks = [
        {
            "artifact": "performance_summary",
            "status": "present" if list(summary.get("models") or []) else "missing",
        },
        {
            "artifact": "quality_perplexity",
            "status": (
                "present"
                if _quality_records(quality_payload, "perplexity")
                else "missing"
            ),
        },
        {
            "artifact": "quality_confidence",
            "status": (
                "present"
                if _quality_records(quality_payload, "confidence")
                else "missing"
            ),
        },
        {
            "artifact": "quality_coherence",
            "status": (
                "present"
                if _quality_records(quality_payload, "coherence")
                else "missing"
            ),
        },
        {
            "artifact": "quality_accuracy",
            "status": (
                "present"
                if _quality_records(quality_payload, "accuracy")
                else "missing"
            ),
        },
    ]
    artifact_issues = [
        f"missing_artifact:{item['artifact']}"
        for item in artifact_checks
        if str(item.get("status") or "") == "missing"
    ]
    speculative_rows = _group_speculative_acceptance(attempt_rows)
    non_ar_rows = [
        dict(row)
        for row in attempt_rows
        if isinstance(row, dict) and _is_non_ar_row(row)
    ]
    blocked_items: list[dict[str, Any]] = []
    speculative_status = "covered" if speculative_rows else "blocked_by_prerequisite"
    if speculative_status != "covered":
        blocked_items.append(
            {
                "scope": "speculative_first_class_benchmarking",
                "prerequisite_phase": SPECULATIVE_PREREQ_PHASE,
                "reason": "No speculative generation mode attempts were observed in this run.",
            }
        )
    non_ar_status = "covered" if non_ar_rows else "blocked_by_prerequisite"
    if non_ar_status != "covered":
        blocked_items.append(
            {
                "scope": "non_ar_first_class_benchmarking",
                "prerequisite_phase": NON_AR_PREREQ_PHASE,
                "reason": "No block, masked, or diffusion generation mode attempts were observed in this run.",
            }
        )
    drift_rows = [dict(row) for row in attempt_rows if isinstance(row, dict) and row]
    drift_signal_records = 0
    drift_missing_signal_rows = 0
    drift_mean_values: list[float] = []
    drift_max_values: list[float] = []
    auto_downgrade_events = 0
    for row in drift_rows:
        signal_values = [
            _runtime_metric(row, "drift_mean"),
            _runtime_metric(row, "drift_max"),
            _runtime_metric(row, "drift_overhead_percent"),
            _runtime_metric(row, "drift_auto_downgrade_events"),
        ]
        if all(value is None for value in signal_values):
            drift_missing_signal_rows += 1
            continue
        drift_signal_records += 1
        drift_mean = _runtime_metric(row, "drift_mean")
        drift_max = _runtime_metric(row, "drift_max")
        if drift_mean is not None:
            drift_mean_values.append(float(drift_mean or 0.0))
        if drift_max is not None:
            drift_max_values.append(float(drift_max or 0.0))
        auto_downgrade_events += int(
            _runtime_metric(row, "drift_auto_downgrade_events") or 0
        )
    hidden_drift = {
        "status": (
            "covered"
            if drift_signal_records
            else ("missing" if drift_rows else "missing")
        ),
        "passed": drift_signal_records > 0
        and drift_missing_signal_rows == 0
        and auto_downgrade_events == 0,
        "records": len(drift_rows),
        "signal_records": drift_signal_records,
        "missing_signal_rows": drift_missing_signal_rows,
        "auto_downgrade_events": auto_downgrade_events,
        "mean_drift_mean": (
            sum(drift_mean_values) / float(len(drift_mean_values))
            if drift_mean_values
            else 0.0
        ),
        "max_drift_max": max(drift_max_values) if drift_max_values else 0.0,
    }
    hidden_drift_issues: list[str] = []
    if drift_rows and drift_signal_records == 0:
        hidden_drift_issues.append("hidden_drift_signals_missing:all")
    elif drift_missing_signal_rows > 0:
        hidden_drift_issues.append(
            f"hidden_drift_signals_missing:{drift_missing_signal_rows}"
        )
    if auto_downgrade_events > 0:
        hidden_drift_issues.append(
            f"hidden_drift_auto_downgrade_events:{auto_downgrade_events}"
        )

    context = dict(resume_context or {})
    resume_requested = bool(context.get("requested", False))
    resumed_attempt_ids = sorted(
        str(item)
        for item in list(context.get("resumed_attempt_ids") or [])
        if str(item).strip()
    )
    completed_lanes = [
        str(item)
        for item in list(context.get("completed_lanes") or [])
        if str(item).strip()
    ]
    checkpoint_artifact_present = bool(
        context.get("checkpoint_artifact_present", False)
    )
    quality_artifact_present = bool(context.get("quality_artifact_present", False))
    resume_quality_issues: list[str] = []
    if resume_requested or resumed_attempt_ids or completed_lanes:
        if not checkpoint_artifact_present:
            resume_quality_issues.append("resume_checkpoint_missing")
        if resumed_attempt_ids and int(summary.get("completed_attempts", 0) or 0) < len(
            resumed_attempt_ids
        ):
            resume_quality_issues.append(
                "resume_completed_attempts_truncated:"
                f"{int(summary.get('completed_attempts', 0) or 0)}"
                f"<{len(resumed_attempt_ids)}"
            )
        if (
            any(lane in {"calibration", "quality_eval"} for lane in completed_lanes)
            and not quality_artifact_present
        ):
            resume_quality_issues.append("resume_quality_artifact_missing")
    resume_quality = {
        "status": (
            "covered"
            if (resume_requested or resumed_attempt_ids or completed_lanes)
            else "not_requested"
        ),
        "passed": not resume_quality_issues,
        "requested": resume_requested,
        "resumed_attempts": len(resumed_attempt_ids),
        "completed_lanes": completed_lanes,
        "checkpoint_artifact_present": checkpoint_artifact_present,
        "quality_artifact_present": quality_artifact_present,
    }
    issues = (
        [
            str(item)
            for item in list(quality_governance.get("issues") or [])
            if str(item).strip()
        ]
        + artifact_issues
        + hidden_drift_issues
        + resume_quality_issues
    )
    return {
        "schema_version": ACCEPTANCE_GOVERNANCE_SCHEMA_VERSION,
        "passed": not issues,
        "issues": issues,
        "blocked_items": blocked_items,
        "artifact_completeness": {
            "passed": not artifact_issues,
            "checks": artifact_checks,
        },
        "quality_evidence": quality_governance,
        "hidden_drift": hidden_drift,
        "resume_quality": resume_quality,
        "mode_coverage": {
            "autoregressive_baseline": {
                "status": "covered" if attempt_rows else "missing",
                "runs": len(attempt_rows),
            },
            "speculative": {
                "status": speculative_status,
                "rows": speculative_rows,
            },
            "non_ar": {
                "status": non_ar_status,
                "runs": len(non_ar_rows),
            },
        },
    }


def _append_governance_failures(
    *,
    summary: dict[str, Any],
    governance: dict[str, Any],
    failure_rows: list[dict[str, Any]],
) -> None:
    seen = {
        str(item.get("error") or "") for item in failure_rows if isinstance(item, dict)
    }
    for issue in list(governance.get("issues") or []):
        issue_text = str(issue).strip()
        if not issue_text:
            continue
        error = f"quality_governance:{issue_text}"
        if error in seen:
            continue
        failure_rows.append(
            {
                "run_id": str(summary.get("run_id") or ""),
                "attempt_id": "",
                "model": "",
                "error": error,
                "error_type": "QualityGovernanceFailure",
                "failure_kind": "gate_failure",
                "gate_issues": [issue_text],
                "normalized_issues": [issue_text],
                "traceback": "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        seen.add(error)


def _empty_comparisons() -> dict[str, Any]:
    return {
        "schema_version": "native_qsg_suite.comparisons.v1",
        "baseline": None,
        "comparators": {},
        "deltas": [],
    }


def _planned_lane_total(spec: BenchmarkSuiteSpec) -> int:
    if spec.tuning_contract_policy == "generate":
        return 3
    return len(_enabled_benchmark_lanes(spec))


def _touch_minimal_lane_artifacts(layout: SuiteRunLayout) -> None:
    _write_json_artifact(
        layout=layout,
        path=layout.kernel_summary_json,
        payload=_empty_kernel_summary(),
        summary="initialized empty kernel summary",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.quality_summary_json,
        payload=_empty_quality_summary(),
        summary="initialized empty quality summary",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.comparisons_json,
        payload=_empty_comparisons(),
        summary="initialized empty comparisons",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.memory_replay_summary_json,
        payload=_empty_memory_replay_summary(),
        summary="initialized empty memory replay summary",
    )
    layout.console_log.touch(exist_ok=True)
    layout.events_ndjson.touch(exist_ok=True)
    layout.terminal_transcript_log.touch(exist_ok=True)
    layout.native_attempts_ndjson.touch(exist_ok=True)
    layout.native_phases_ndjson.touch(exist_ok=True)
    layout.native_failures_ndjson.touch(exist_ok=True)
    layout.kernel_attempts_ndjson.touch(exist_ok=True)
    layout.quality_attempts_ndjson.touch(exist_ok=True)


def _summary_payload(
    *,
    run_id: str,
    quality_payload: dict[str, Any],
    kernel_hotspots: list[dict[str, Any]],
    host_compliance: dict[str, Any],
    failure_rows: list[dict[str, Any]],
    planned_attempts: int,
    completed_attempts: int,
    run_exit_reason: str,
    terminal_state: str,
    last_successful_lane: str | None,
    overall_pass: bool,
    certification_state: str,
    calibration: dict[str, Any] | None = None,
    resume_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    triage = _build_triage({"run_id": run_id, "kernel_hotspots": kernel_hotspots})
    payload = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": bool(overall_pass),
        "failure_count": len(failure_rows),
        "failure_counts": {
            "total": len(failure_rows),
            "gate_failure": sum(
                1
                for item in failure_rows
                if str(item.get("failure_kind") or "") == "gate_failure"
            ),
            "execution_failure": sum(
                1
                for item in failure_rows
                if str(item.get("failure_kind") or "") == "execution_failure"
            ),
        },
        "failed_attempt_ids": [
            str(item.get("attempt_id") or "")
            for item in failure_rows
            if str(item.get("attempt_id") or "").strip()
        ],
        "completed_attempts": int(completed_attempts),
        "planned_attempts": int(planned_attempts),
        "pass": bool(overall_pass),
        "quality": quality_payload,
        "ablation_deltas": [],
        "kernel_hotspots": kernel_hotspots,
        "host_compliance": host_compliance,
        "agent_triage": triage,
        "continuous": {
            "schema_version": CONTINUOUS_SCHEMA_VERSION,
            "results": [],
        },
        "quality_governance": _acceptance_governance_report(
            summary={"models": [], "run_id": run_id},
            quality_payload=quality_payload,
            attempt_rows=[],
            resume_context=resume_context,
        ),
        "certification_state": certification_state,
        "run_exit_reason": run_exit_reason,
        "terminal_state": terminal_state,
        "last_successful_lane": last_successful_lane,
    }
    if isinstance(calibration, dict) and calibration:
        payload["calibration"] = calibration
    return payload


def _tuning_receipt_payload(
    *,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
    calibration_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    contracts = dict((calibration_bundle or {}).get("contracts") or {})
    winners = dict((calibration_bundle or {}).get("winners") or {})
    admission = dict((calibration_bundle or {}).get("admission") or {})
    return {
        "schema_version": "native_qsg_suite.tuning_receipt.v1",
        "profile_name": str(spec.profile_name),
        "tuning_policy": str(spec.tuning_contract_policy),
        "tuning_state": str(preflight_payload.get("tuning_state") or ""),
        "refresh_models": list(preflight_payload.get("refresh_models") or []),
        "admission": admission,
        "contracts": contracts,
        "selected_runtime_envelope": {
            model: {
                "thread_config": dict((payload or {}).get("thread_config") or {}),
                "continuous_config": dict(
                    (payload or {}).get("continuous_config") or {}
                ),
                "pager_config": dict((payload or {}).get("pager_config") or {}),
            }
            for model, payload in contracts.items()
        },
        "winners": winners,
    }


def _tuning_remediation_payload(
    *,
    spec: BenchmarkSuiteSpec,
    preflight_payload: dict[str, Any],
    calibration_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    refresh_models = [
        str(item)
        for item in list(preflight_payload.get("refresh_models") or [])
        if str(item).strip()
    ]
    winner_packets = []
    for model, winner in dict((calibration_bundle or {}).get("winners") or {}).items():
        hotspot_metrics = dict(
            (dict((calibration_bundle or {}).get("contracts") or {}).get(model) or {})
            .get("calibration_stats", {})
            .get("hotspot_metrics", {})
        )
        winner_packets.append(
            {
                "model": model,
                "fairness": float(winner.get("fairness", 1.0) or 1.0),
                "queue_wait_ms_p95": float(winner.get("queue_wait_ms_p95", 0.0) or 0.0),
                "decode_goodput_tps": float(
                    winner.get(
                        "decode_goodput_tps", winner.get("decode_tps_median", 0.0)
                    )
                    or 0.0
                ),
                "hotspot_watch": list(hotspot_metrics.get("top_hotspots") or []),
            }
        )
    recommendations: list[str] = []
    if refresh_models:
        recommendations.append(
            f"refresh_or_rebuild_tuning_contracts:{','.join(refresh_models)}"
        )
    for packet in winner_packets:
        if float(packet.get("fairness", 1.0) or 1.0) < float(
            spec.calibration_search.fairness_floor or 0.0
        ):
            recommendations.append(f"raise_scheduler_fairness:{packet['model']}")
        if float(packet.get("queue_wait_ms_p95", 0.0) or 0.0) > float(
            spec.calibration_search.queue_wait_p95_ceiling_ms or 0.0
        ):
            recommendations.append(f"reduce_queue_wait:{packet['model']}")
        if list(packet.get("hotspot_watch") or []):
            recommendations.append(f"optimize_hotspot_watchlist:{packet['model']}")
    return {
        "schema_version": "native_qsg_suite.tuning_remediation.v1",
        "profile_name": str(spec.profile_name),
        "refresh_models": refresh_models,
        "recommendations": recommendations,
        "winner_packets": winner_packets,
    }


def _final_certification_state(preflight_payload: dict[str, Any], passed: bool) -> str:
    certified = (
        str(preflight_payload.get("certification_state") or "") == "certified_candidate"
    )
    if certified:
        return "certified_pass" if passed else "certified_fail"
    return "non_certified_pass" if passed else "non_certified_fail"


def _failed_certification_state(current: str) -> str:
    normalized = str(current or "").strip()
    if normalized.startswith("certified"):
        return "certified_fail"
    return "non_certified_fail"


def _fallback_summary_for_persistence(
    summary: dict[str, Any],
    *,
    errors: list[str],
) -> dict[str, Any]:
    host_compliance = dict(summary.get("host_compliance") or {})
    host_compliance["summary_persistence"] = {
        "errors": list(errors),
        "fallback_summary": True,
    }
    payload = _summary_payload(
        run_id=str(summary.get("run_id") or ""),
        quality_payload=dict(summary.get("quality") or _empty_quality_summary()),
        kernel_hotspots=[
            item
            for item in list(summary.get("kernel_hotspots") or [])
            if isinstance(item, dict)
        ],
        host_compliance=host_compliance,
        failure_rows=[],
        planned_attempts=int(summary.get("planned_attempts", 0) or 0),
        completed_attempts=int(summary.get("completed_attempts", 0) or 0),
        run_exit_reason="summary_persistence_failed",
        terminal_state="internal_error",
        last_successful_lane=(
            str(summary.get("last_successful_lane"))
            if summary.get("last_successful_lane") is not None
            else None
        ),
        overall_pass=False,
        certification_state=_failed_certification_state(
            str(summary.get("certification_state") or "")
        ),
        calibration=(
            dict(summary.get("calibration") or {})
            if isinstance(summary.get("calibration"), dict)
            else None
        ),
        resume_context=dict(
            dict(summary.get("quality_governance") or {}).get("resume_quality") or {}
        ),
    )
    payload["models"] = [
        item for item in list(summary.get("models") or []) if isinstance(item, dict)
    ]
    payload["top_stage_hotspots"] = [
        item
        for item in list(summary.get("top_stage_hotspots") or [])
        if isinstance(item, dict)
    ]
    payload["stage_origin_map"] = dict(summary.get("stage_origin_map") or {})
    payload["baseline_floors"] = dict(summary.get("baseline_floors") or {})
    payload["failure_count"] = int(summary.get("failure_count", 0) or 0)
    payload["failure_counts"] = {
        "total": int(
            (summary.get("failure_counts") or {}).get(
                "total",
                summary.get("failure_count", 0) or 0,
            )
            or 0
        ),
        "gate_failure": int(
            (summary.get("failure_counts") or {}).get("gate_failure", 0) or 0
        ),
        "execution_failure": int(
            (summary.get("failure_counts") or {}).get("execution_failure", 0) or 0
        ),
    }
    payload["failed_attempt_ids"] = [
        str(item)
        for item in list(summary.get("failed_attempt_ids") or [])
        if str(item).strip()
    ]
    payload["ablation_deltas"] = [
        item
        for item in list(summary.get("ablation_deltas") or [])
        if isinstance(item, dict)
    ]
    payload["agent_triage"] = dict(summary.get("agent_triage") or {})
    payload["continuous"] = dict(
        summary.get("continuous")
        or {"schema_version": CONTINUOUS_SCHEMA_VERSION, "results": []}
    )
    payload["quality_governance"] = dict(summary.get("quality_governance") or {})
    payload["memory_replay"] = dict(
        summary.get("memory_replay") or _empty_memory_replay_summary()
    )
    payload["comparisons"] = dict(summary.get("comparisons") or _empty_comparisons())
    payload["baseline_run_id"] = str(summary.get("baseline_run_id") or "")
    payload["prompt_hash"] = str(summary.get("prompt_hash") or "")
    payload["prompt_contract_hash"] = str(summary.get("prompt_contract_hash") or "")
    payload["memory_snapshot_hash"] = str(summary.get("memory_snapshot_hash") or "")
    payload["feature_toggle_hash"] = str(summary.get("feature_toggle_hash") or "")
    payload["advisory_bundle"] = dict(summary.get("advisory_bundle") or {})
    payload["black_box"] = dict(summary.get("black_box") or {})
    return payload


def _summary_failed_payload(
    *,
    summary: dict[str, Any],
    persisted_summary: dict[str, Any],
    comparisons: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "native_qsg_suite.summary_failed.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": list(errors),
        "raw_summary": summary,
        "persisted_summary": persisted_summary,
        "comparisons": comparisons,
    }


def _write_fallback_reports(
    layout: SuiteRunLayout,
    *,
    summary: dict[str, Any],
    error_text: str,
) -> None:
    layout.summary_md.write_text(
        f"# Native QSG Summary\n\nFallback summary due to persistence failure.\n\n{error_text}\n",
        encoding="utf-8",
    )
    layout.report_md.write_text(
        f"# Native QSG Report\n\nReport generation failed.\n\n{error_text}\n",
        encoding="utf-8",
    )
    layout.executive_summary_md.write_text(
        f"# Executive Summary\n\nSummary persistence failed.\n\n{error_text}\n",
        encoding="utf-8",
    )
    layout.report_html.write_text(
        (
            "<html><body><h1>Native QSG Report</h1>"
            f"<p>{html.escape(error_text)}</p></body></html>"
        ),
        encoding="utf-8",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.metrics_rollup_json,
        payload={
            "schema_version": "native_qsg_suite.metrics_rollup.v1",
            "error": error_text,
            "overall_pass": bool(summary.get("overall_pass", False)),
        },
        summary="wrote fallback metrics rollup",
        phase="finalizing",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.agent_handoff_json,
        payload={
            "schema_version": "native_qsg_suite.agent_handoff.v1",
            "error": error_text,
            "run_id": str(summary.get("run_id") or ""),
        },
        summary="wrote fallback agent handoff",
        phase="finalizing",
    )


def _persist_summary_bundle(
    *,
    layout: SuiteRunLayout,
    summary: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    persisted_summary = dict(summary)
    try:
        validate_payload("summary.schema.json", summary)
    except Exception as exc:
        errors.append(f"summary_validation:{type(exc).__name__}:{exc}")
        persisted_summary = _fallback_summary_for_persistence(summary, errors=errors)
    _write_json_artifact(
        layout=layout,
        path=layout.summary_json,
        payload=persisted_summary,
        summary="wrote suite summary",
        phase="finalizing",
    )
    try:
        _write_json_artifact(
            layout=layout,
            path=layout.comparisons_json,
            payload=comparisons,
            summary="wrote comparison bundle",
            phase="finalizing",
        )
    except Exception as exc:
        errors.append(f"comparisons_write:{type(exc).__name__}:{exc}")
    try:
        _write_reports(layout, persisted_summary)
    except Exception as exc:
        errors.append(f"report_write:{type(exc).__name__}:{exc}")
        _write_fallback_reports(
            layout,
            summary=persisted_summary,
            error_text=str(exc),
        )
    try:
        _write_json_artifact(
            layout=layout,
            path=layout.triage_json,
            payload=dict(persisted_summary.get("agent_triage") or {}),
            summary="wrote triage bundle",
            phase="finalizing",
        )
    except Exception as exc:
        errors.append(f"triage_write:{type(exc).__name__}:{exc}")
    if errors:
        persisted_summary = _fallback_summary_for_persistence(summary, errors=errors)
        _write_json_artifact(
            layout=layout,
            path=layout.summary_json,
            payload=persisted_summary,
            summary="wrote fallback suite summary",
            phase="finalizing",
        )
        try:
            _write_json_artifact(
                layout=layout,
                path=layout.summary_failed_json,
                payload=_summary_failed_payload(
                    summary=summary,
                    persisted_summary=persisted_summary,
                    comparisons=comparisons,
                    errors=errors,
                ),
                summary="wrote failed summary bundle",
                phase="finalizing",
            )
        except Exception:
            pass
        _terminal("Summary persistence fallback engaged.")
        for error in errors:
            _terminal(f"- {error}")
    return {
        "ok": not errors,
        "summary": persisted_summary,
        "errors": errors,
    }


def _sync_benchmark_truth_bundle(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    preflight_payload: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    existing_bundle = _load_json(layout.evidence_bundle_json) or {}
    artifact_index = _load_json(layout.index_json) or {}
    black_box_manifest = _load_json(layout.black_box_manifest_json) or {}
    event_store_export = _load_json(layout.telemetry_event_store_export_json) or {}
    convergence = {
        "schema_version": "native_qsg_suite.silver_convergence.v1",
        "run_id": str(summary.get("run_id") or ""),
        "verdict": "pass" if bool(summary.get("overall_pass", False)) else "fail",
        "certification_state": str(summary.get("certification_state") or ""),
        "admission_passed": bool(
            (preflight_payload.get("admission_manifest") or {}).get("passed", False)
        ),
        "graph_preflight_passed": bool(
            (preflight_payload.get("graph_preflight") or {}).get("passed", False)
        ),
        "runtime_gates_passed": bool(
            (summary.get("assurance") or {}).get("runtime_gates_passed", False)
        ),
        "quality_governance_passed": bool(
            (summary.get("quality_governance") or {}).get("passed", False)
        ),
        "failure_count": int(summary.get("failure_count", 0) or 0),
        "residual_risk": list(
            (summary.get("quality_governance") or {}).get("issues") or []
        ),
    }
    bundle_payload = {
        **existing_bundle,
        "schema_version": "native_qsg_suite.benchmark_truth_bundle.v1",
        "run_id": str(summary.get("run_id") or ""),
        "profile_name": spec.profile_name,
        "summary_path": str(layout.summary_json),
        "comparisons_path": str(layout.comparisons_json),
        "artifact_index_path": str(layout.index_json),
        "kernel_summary_path": str(layout.kernel_summary_json),
        "quality_summary_path": str(layout.quality_summary_json),
        "memory_replay_summary_path": str(layout.memory_replay_summary_json),
        "event_store_export_path": str(layout.telemetry_event_store_export_json),
        "black_box_manifest_path": str(layout.black_box_manifest_json),
        "artifact_index": artifact_index,
        "manifest": manifest,
        "admission_manifest": dict(preflight_payload.get("admission_manifest") or {}),
        "graph_preflight": dict(preflight_payload.get("graph_preflight") or {}),
        "capability_ledger": dict(preflight_payload.get("capability_ledger") or {}),
        "quality_governance": dict(summary.get("quality_governance") or {}),
        "black_box": black_box_manifest,
        "event_store_export": event_store_export.get("closure_summary") or {},
        "comparisons": comparisons,
        "convergence_manifest": convergence,
    }
    write_json_atomic(layout.evidence_bundle_json, bundle_payload)
    advisory_bundle = _load_json(layout.advisory_bundle_json) or {}
    advisory_bundle["silver_convergence"] = convergence
    write_json_atomic(layout.advisory_bundle_json, advisory_bundle)

    evidence_path = EvidenceService(str(repo_root)).write_bundle(
        "benchmark", bundle_payload
    )
    metrics_record = MetricsService(str(repo_root)).write_run(
        "benchmark",
        {
            "run_id": str(summary.get("run_id") or ""),
            "profile_name": spec.profile_name,
            "slo_status": (
                "pass" if bool(summary.get("overall_pass", False)) else "fail"
            ),
            "certification_state": str(summary.get("certification_state") or ""),
            "bundle_path": evidence_path,
            "convergence_verdict": convergence["verdict"],
            "failure_count": int(summary.get("failure_count", 0) or 0),
        },
    )
    return {
        "bundle_path": str(layout.evidence_bundle_json),
        "platform_evidence_path": evidence_path,
        "metrics_record": metrics_record,
        "convergence_manifest": convergence,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=str, default=DEFAULT_PROFILE)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument(
        "--calibration-mode",
        choices=tuple(sorted(CALIBRATION_MODES)),
        default="search",
    )
    parser.add_argument("--calibration-source", type=str, default="manual")
    parser.add_argument("--calibration-target-profile", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=str, default="audit")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--mission-dump", action="store_true")
    parser.add_argument("--mission-node", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variance-report-only", action="store_true")
    parser.add_argument("--closure-advisory", action="store_true")
    parser.add_argument("--capsule", action="store_true")
    parser.add_argument(
        "--compare-to",
        choices=(
            "latest_compatible",
            "last_same_topology",
            "last_same_host",
            "last_green_main",
            "last_same_prompt_contract",
        ),
        default="latest_compatible",
    )
    parser.add_argument(
        "--ui-mode",
        choices=("glassbox", "raw", "dashboard", "plain"),
        default=env_ui_mode(),
    )
    parser.add_argument(
        "--log-level",
        choices=("trace", "debug", "info"),
        default=env_log_level(),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variance_report_only = bool(getattr(args, "variance_report_only", False))
    mission_dump = bool(getattr(args, "mission_dump", False))
    closure_advisory = bool(getattr(args, "closure_advisory", False))
    capsule_enabled = bool(getattr(args, "capsule", False))
    calibration_mode = str(getattr(args, "calibration_mode", "search") or "search")
    calibration_source = str(getattr(args, "calibration_source", "manual") or "manual")
    calibration_target_profile = str(
        getattr(args, "calibration_target_profile", "") or ""
    )
    compare_to = str(
        getattr(args, "compare_to", "latest_compatible") or "latest_compatible"
    )
    repo_root = Path(__file__).resolve().parents[2]
    profile_path = _profile_path(str(args.profile))
    if not profile_path.exists():
        raise RuntimeError(f"Benchmark profile not found: {profile_path}")
    spec = load_suite_profile(profile_path)
    models_override = [
        str(item).strip()
        for item in str(getattr(args, "models", "") or "").split(",")
        if str(item).strip()
    ]
    if models_override:
        spec = replace(spec, models=models_override)
    _validate_execution_policy(spec)
    run_id = str(args.run_id or _default_run_id(repo_root, spec))
    layout = resolve_suite_layout(Path(str(args.out_root)), run_id)
    ensure_suite_layout(layout)
    prompt_hash = hashlib.sha256(
        str(spec.scenario_pack.canonical_prompt or "").encode("utf-8")
    ).hexdigest()
    prompt_contract_hash = _prompt_contract_hash(spec.models)
    feature_toggle_hash = _feature_toggle_contract_hash(spec)
    logger = SuiteEventLogger(
        run_id=run_id,
        run_root=layout.root,
        events_path=layout.events_ndjson,
        transcript_path=layout.terminal_transcript_log,
        console_log_path=layout.console_log,
        ui_mode=str(args.ui_mode),
        log_level=str(args.log_level),
    )
    logger.start()
    set_active_logger(logger)
    black_box = BlackBoxRecorder(str(repo_root))
    black_box.start_run(
        run_id=run_id,
        task_id=f"benchmark_suite::{run_id}",
        task=f"benchmark suite profile {spec.profile_name}",
        metadata={"profile": spec.profile_name, "compare_to": compare_to},
    )
    launch_runtime = capture_runtime_provenance(repo_root)
    memory_snapshot_hash = _memory_snapshot_hash(launch_runtime)
    affinity_adjustment = _ensure_suite_affinity(spec)
    logger.set_state(total_lanes=_planned_lane_total(spec))
    _terminal(f"Starting benchmark suite profile `{spec.profile_name}`")
    _terminal(f"Run folder: {layout.root}")
    _terminal("Model execution policy: sequential (max_parallel_models=1)")
    if affinity_adjustment.get("repair_required") and not affinity_adjustment.get(
        "repair_allowed"
    ):
        before = affinity_adjustment.get("before") or []
        target = affinity_adjustment.get("target") or []
        _terminal(
            f"CPU affinity repair required but disallowed by policy: visible_threads {len(before)} -> {len(target)}"
        )
    elif affinity_adjustment.get("attempted"):
        before = affinity_adjustment.get("before") or []
        after = affinity_adjustment.get("after") or []
        _terminal(
            f"CPU affinity auto-expand attempted: visible_threads {len(before)} -> {len(after)}"
        )
    elif str(affinity_adjustment.get("error") or "").strip():
        _terminal(
            f"warning: suite affinity auto-expand unavailable ({affinity_adjustment['error']})"
        )

    terminal_state = "internal_error"
    run_exit_reason = "internal_error:uninitialized"
    last_successful_lane: str | None = None
    exit_code = 1
    summary: dict[str, Any] | None = None
    assurance_plan: dict[str, Any] | None = None
    assurance_context: Any = None
    preflight_payload: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    comparisons = _empty_comparisons()
    attempt_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    completed_lanes: list[str] = []
    completed_attempt_ids: set[str] = set()
    resume_completed: set[str] = set()
    if bool(args.resume):
        checkpoint = _load_checkpoint(layout)
        resume_completed = {
            str(item)
            for item in list(checkpoint.get("completed_attempt_ids") or [])
            if str(item).strip()
        }
        completed_lanes = [
            str(item)
            for item in list(checkpoint.get("completed_lanes") or [])
            if str(item).strip()
        ]
        last_successful_lane = (
            str(checkpoint.get("last_successful_lane") or "").strip() or None
        )

    if variance_report_only:
        existing_summary = _load_json(layout.summary_json)
        if not existing_summary:
            raise RuntimeError(
                "--variance-report-only requires an existing run summary for the selected --run-id"
            )
        existing_preflight = _load_json(layout.preflight_json) or {}
        existing_manifest = _load_json(layout.manifest_json) or {}
        existing_comparisons = (
            _load_json(layout.comparisons_json) or _empty_comparisons()
        )
        existing_assurance_plan = _load_json(layout.assurance_plan_json)
        checkpoint = _load_checkpoint(layout)
        completed_lanes = [
            str(item)
            for item in list(checkpoint.get("completed_lanes") or [])
            if str(item).strip()
        ]
        control_plane_payload = materialize_control_plane_artifacts(
            repo_root=repo_root,
            layout=layout,
            spec=spec,
            manifest=existing_manifest,
            launch_runtime=dict(
                existing_manifest.get("launch_runtime") or launch_runtime
            ),
            preflight_payload=existing_preflight,
            summary=existing_summary,
            comparisons=existing_comparisons,
            assurance_plan=existing_assurance_plan,
            completed_lanes=completed_lanes,
            compare_to=compare_to,
            mission_dump=mission_dump,
            variance_report_only=True,
            closure_advisory=closure_advisory,
            capsule_enabled=capsule_enabled,
        )
        _terminal(f"Variance report regenerated: {layout.variance_budget_json}")
        if mission_dump:
            _terminal(f"Mission graph: {layout.mission_graph_json}")
        if capsule_enabled:
            archive_path = str(
                dict(control_plane_payload.get("capsule_archive") or {}).get("path")
                or ""
            ).strip()
            if archive_path:
                _terminal(f"Capsule archive: {layout.root / archive_path}")
        return 0

    _write_json_artifact(
        layout=layout,
        path=layout.suite_resolved_json,
        payload=compile_suite_profile(spec),
        summary=f"resolved suite profile {spec.profile_name}",
        phase="initialized",
    )
    _write_status(layout, run_id=run_id, state="initialized", ok=True)
    _write_checkpoint(
        layout,
        completed_lanes=completed_lanes,
        completed_attempt_ids=sorted(completed_attempt_ids | resume_completed),
        run_exit_reason="in_progress",
        last_successful_lane=last_successful_lane,
    )
    manifest = {
        "schema_version": "native_qsg_suite.manifest.v1",
        "run_id": run_id,
        "profile": spec.profile_name,
        "experiment": args.experiment,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "models": spec.models,
        "profile_path": str(profile_path),
        "prompt_hash": prompt_hash,
        "prompt_contract_hash": prompt_contract_hash,
        "memory_snapshot_hash": memory_snapshot_hash,
        "feature_toggle_hash": feature_toggle_hash,
        "compare_to": compare_to,
        "launch_runtime": launch_runtime,
        "affinity_adjustment": affinity_adjustment,
    }
    _write_json_artifact(
        layout=layout,
        path=layout.manifest_json,
        payload=manifest,
        summary="wrote run manifest",
        phase="initialized",
    )
    assurance_plan, assurance_context = compile_assurance_plan(
        repo_root=repo_root,
        run_id=run_id,
        profile_path=profile_path,
        spec=spec,
    )
    _write_json_artifact(
        layout=layout,
        path=layout.assurance_plan_json,
        payload=assurance_plan,
        summary="wrote assurance plan",
        phase="initialized",
    )
    _write_json_artifact(
        layout=layout,
        path=layout.closure_matrix_json,
        payload={
            "schema_version": str(assurance_plan.get("schema_version") or ""),
            "run_id": run_id,
            "closure_matrix": list(assurance_plan.get("closure_matrix") or []),
        },
        summary="wrote closure matrix",
        phase="initialized",
    )
    mission_graph = compile_benchmark_mission_graph(
        spec,
        assurance_plan=assurance_plan,
        compare_to=compare_to,
        variance_report_only=variance_report_only,
        closure_advisory=closure_advisory,
        capsule_enabled=capsule_enabled,
    )
    if (
        bool(getattr(args, "dry_run", False))
        and str(getattr(args, "mission_node", "")).strip()
    ):
        receipt = dry_run_receipt(mission_graph, str(args.mission_node))
        _write_json_artifact(
            layout=layout,
            path=layout.mission_graph_json,
            payload=mission_graph.to_dict(),
            summary="wrote mission graph for dry-run replay",
            phase="initialized",
        )
        _write_json_artifact(
            layout=layout,
            path=layout.mission_receipts_json,
            payload={
                "schema_version": "native_qsg_suite.mission_receipts.v1",
                "run_id": run_id,
                "receipts": [receipt],
            },
            summary="wrote dry-run mission receipt",
            phase="initialized",
        )
        _emit_mission_receipt(
            node_id=str(args.mission_node),
            phase=str(receipt.get("phase") or ""),
            kind=str(receipt.get("kind") or ""),
            status="dry_run",
            blocking=bool(receipt.get("blocking", False)),
            lane=str(receipt.get("lane") or "") or None,
            details=dict(receipt.get("details") or {}),
        )
        write_json_atomic(layout.black_box_manifest_json, {"status": "dry_run"})
        set_active_logger(None)
        logger.close()
        return 0

    try:
        black_box.record_event("preflight_start", phase="preflight", status="started")
        _emit_mission_receipt(
            node_id="preflight",
            phase="preflight",
            kind="host_contract",
            status="started",
            blocking=True,
        )
        _write_status(layout, run_id=run_id, state="preflight", ok=True)
        preflight = run_preflight(
            repo_root=repo_root,
            spec=spec,
            telemetry_dir=layout.telemetry_dir,
            launch_runtime=launch_runtime,
            affinity_adjustment=affinity_adjustment,
        )
        preflight_payload = dict(preflight.payload or {})
        preflight_payload["suite_affinity_adjustment"] = affinity_adjustment
        _write_json_artifact(
            layout=layout,
            path=layout.preflight_json,
            payload=preflight_payload,
            summary=f"wrote preflight payload ok={preflight.ok}",
            phase="preflight",
        )
        _write_json_artifact(
            layout=layout,
            path=layout.environment_json,
            payload={
                "launch_runtime": preflight_payload.get("launch_runtime")
                or launch_runtime,
                "runtime": preflight_payload.get("runtime") or {},
                "host": preflight_payload.get("host") or {},
                "models": preflight_payload.get("models") or {},
                "cpu_governor": str(preflight_payload.get("cpu_governor") or ""),
                "thp_mode": str(preflight_payload.get("thp_mode") or ""),
                "perf_event_paranoid": str(
                    preflight_payload.get("perf_event_paranoid") or ""
                ),
                "lscpu": preflight_payload.get("lscpu") or {},
                "memory": preflight_payload.get("memory") or {},
                "benchmark_metadata": preflight_payload.get("benchmark_metadata") or {},
                "graph_preflight": preflight_payload.get("graph_preflight") or {},
                "capability_ledger": preflight_payload.get("capability_ledger") or {},
                "admission_manifest": preflight_payload.get("admission_manifest") or {},
                "launch_affinity": preflight_payload.get("launch_affinity") or [],
                "post_adjustment_affinity": preflight_payload.get(
                    "post_adjustment_affinity"
                )
                or [],
                "repair_allowed": bool(preflight_payload.get("repair_allowed", False)),
                "repair_attempted": bool(
                    preflight_payload.get("repair_attempted", False)
                ),
                "repair_required": bool(
                    preflight_payload.get("repair_required", False)
                ),
                "certified_host_match": bool(
                    preflight_payload.get("certified_host_match", False)
                ),
                "certification_state": str(
                    preflight_payload.get("certification_state") or ""
                ),
                "benchmark_harness_hash": str(
                    preflight_payload.get("benchmark_harness_hash") or ""
                ),
            },
            summary="wrote environment snapshot",
            phase="preflight",
        )
        _console_log(layout, f"preflight ok={preflight.ok}")
        _emit_mission_receipt(
            node_id="preflight",
            phase="preflight",
            kind="host_contract",
            status="completed" if preflight.ok else "failed",
            blocking=True,
            details={
                "certification_state": preflight_payload.get("certification_state")
            },
        )
        _write_checkpoint(
            layout,
            completed_lanes=completed_lanes,
            completed_attempt_ids=sorted(completed_attempt_ids | resume_completed),
            run_exit_reason="in_progress",
            last_successful_lane=last_successful_lane,
        )
        _write_status(layout, run_id=run_id, state="preflight", ok=preflight.ok)

        if spec.strict_host and not preflight.ok:
            terminal_state = "failed_preflight"
            run_exit_reason = "preflight_failed"
            _touch_minimal_lane_artifacts(layout)
            summary = _summary_payload(
                run_id=run_id,
                quality_payload=_empty_quality_summary(),
                kernel_hotspots=[],
                host_compliance=preflight_payload,
                failure_rows=[
                    {
                        "failure_kind": "gate_failure",
                        "attempt_id": "",
                    }
                    for _ in list(preflight_payload.get("failures") or [])
                ],
                planned_attempts=0,
                completed_attempts=0,
                run_exit_reason=run_exit_reason,
                terminal_state=terminal_state,
                last_successful_lane=last_successful_lane,
                overall_pass=False,
                certification_state=_final_certification_state(
                    preflight_payload, False
                ),
            )
            persist_result = _persist_summary_bundle(
                layout=layout, summary=summary, comparisons=comparisons
            )
            summary = dict(persist_result.get("summary") or summary)
            if not bool(persist_result.get("ok", False)):
                terminal_state = str(summary.get("terminal_state") or "internal_error")
                run_exit_reason = str(
                    summary.get("run_exit_reason") or "summary_persistence_failed"
                )
            _emit_preflight_failure(layout, preflight_payload, spec.profile_name)
            exit_code = 1
        else:
            _touch_minimal_lane_artifacts(layout)
            attempt_rows = read_ndjson(layout.native_attempts_ndjson)
            failure_rows = read_ndjson(layout.native_failures_ndjson)
            completed_attempt_ids.update(
                str(row.get("attempt_id"))
                for row in attempt_rows
                if str(row.get("attempt_id") or "").strip()
            )
            completed_attempt_ids.update(resume_completed)

            runtime_payload = dict(preflight_payload.get("runtime") or {})
            thread_matrix = _thread_candidates(spec, runtime_payload)
            canonical_thread_tuples = _canonical_thread_tuples(spec, runtime_payload)
            max_affinity_threads = _max_affinity_thread_tuple(spec, runtime_payload)
            host = dict(runtime_payload.get("host") or {})
            logical = int(host.get("logical_cpus", 1) or 1)
            visible = int(host.get("visible_threads", logical) or logical)
            model_thread_overrides = _model_thread_contracts(
                repo_root=repo_root,
                spec=spec,
                preflight_payload=preflight_payload,
            )
            model_continuous_overrides = _model_continuous_contracts(
                repo_root=repo_root,
                spec=spec,
                preflight_payload=preflight_payload,
            )
            tuning_state = str(preflight_payload.get("tuning_state") or "").strip()
            if model_thread_overrides:
                sample_threads = next(iter(model_thread_overrides.values()))
                _terminal(
                    f"Thread budget: visible_threads={visible}, logical_cpus={logical}, "
                    f"max_affinity={max_affinity_threads[0]}x{max_affinity_threads[1]}, "
                    f"tuned_headline={sample_threads[0]}x{sample_threads[1]}"
                )
            else:
                _terminal(
                    f"Thread budget: visible_threads={visible}, logical_cpus={logical}, "
                    f"max_affinity={max_affinity_threads[0]}x{max_affinity_threads[1]}, "
                    f"canonical={canonical_thread_tuples[0][0]}x{canonical_thread_tuples[0][1]}"
                )
                if (
                    tuning_state in {"stale", "missing", "mixed"}
                    and spec.tuning_contract_policy in {"required", "generate"}
                ):
                    _terminal(
                        f"warning: tuning_state={tuning_state}; running canonical fallback threads before calibration refresh."
                    )
            if visible < logical:
                _terminal(
                    "warning: process visibility is CPU-limited; suite will use visible threads only."
                )

            continuous_payload: list[dict[str, Any]] = []
            kernel_summary: dict[str, Any] = _empty_kernel_summary()
            calibration_bundle: dict[str, Any] | None = None
            baseline_summary: dict[str, Any] | None = None
            comparator_catalog: dict[str, dict[str, Any]] = {}
            summary_rows: list[dict[str, Any]] = []
            quality_payload: dict[str, Any] = _empty_quality_summary()
            memory_replay_payload: dict[str, Any] = _empty_memory_replay_summary()

            if spec.tuning_contract_policy == "generate":
                if "calibration" not in completed_lanes:
                    logger.set_state(
                        phase="lane_calibration",
                        lane="calibration",
                        completed_lanes=len(completed_lanes),
                    )
                    _write_status(
                        layout, run_id=run_id, state="lane_calibration", ok=True
                    )
                    calibration_bundle = _run_calibration(
                        repo_root=repo_root,
                        layout=layout,
                        spec=spec,
                        preflight_payload=preflight_payload,
                        attempt_rows=attempt_rows,
                        failure_rows=failure_rows,
                        completed_attempt_ids=completed_attempt_ids,
                        resume_completed=resume_completed,
                        calibration_mode=calibration_mode,
                        calibration_source=calibration_source,
                        calibration_target_profile=calibration_target_profile,
                    )
                    completed_lanes.append("calibration")
                    _emit_mission_receipt(
                        node_id="calibration",
                        phase="calibration",
                        kind="tuning",
                        status="completed",
                        blocking=False,
                        lane="calibration",
                    )
                    logger.set_state(
                        completed_lanes=len(completed_lanes), lane="calibration"
                    )
                    last_successful_lane = "calibration"
                    _write_checkpoint(
                        layout,
                        completed_lanes=completed_lanes,
                        completed_attempt_ids=sorted(completed_attempt_ids),
                        run_exit_reason="in_progress",
                        last_successful_lane=last_successful_lane,
                    )
                else:
                    quality_payload = json.loads(
                        layout.quality_summary_json.read_text(encoding="utf-8")
                    )
                    calibration_bundle = {
                        "schema_version": "native_qsg_suite.calibration.v2",
                        "contracts": {},
                        "quality": quality_payload,
                        "winners": {},
                    }
                quality_payload = dict((calibration_bundle or {}).get("quality") or {})
                summary_rows = [
                    row
                    for winner in dict(
                        (calibration_bundle or {}).get("winners") or {}
                    ).values()
                    for row in list(winner.get("rows") or [])
                ]
            else:
                if (
                    _lane_enabled(spec, "canonical_all_on")
                    and "canonical_all_on" not in completed_lanes
                ):
                    logger.set_state(
                        phase="lane_canonical_all_on",
                        lane="canonical_all_on",
                        completed_lanes=len(completed_lanes),
                    )
                    _write_status(
                        layout, run_id=run_id, state="lane_canonical_all_on", ok=True
                    )
                    _lane_attempts(
                        repo_root=repo_root,
                        layout=layout,
                        spec=spec,
                        lane_id="canonical_all_on",
                        ablation_id="all_on",
                        env_overrides={},
                        prompt=spec.scenario_pack.canonical_prompt,
                        warmup_runs=spec.scenario_pack.warmup_runs,
                        measured_runs=spec.scenario_pack.measured_runs,
                        thread_matrix=canonical_thread_tuples,
                        use_thread_matrix=False,
                        attempt_rows=attempt_rows,
                        failure_rows=failure_rows,
                        completed_attempt_ids=completed_attempt_ids,
                        resume_completed=resume_completed,
                        model_thread_overrides=model_thread_overrides or None,
                    )
                    completed_lanes.append("canonical_all_on")
                    _emit_mission_receipt(
                        node_id="canonical_all_on",
                        phase="native_surface",
                        kind="lane",
                        status="completed",
                        blocking=True,
                        lane="canonical_all_on",
                    )
                    logger.set_state(
                        completed_lanes=len(completed_lanes), lane="canonical_all_on"
                    )
                    last_successful_lane = "canonical_all_on"
                    _write_checkpoint(
                        layout,
                        completed_lanes=completed_lanes,
                        completed_attempt_ids=sorted(completed_attempt_ids),
                        run_exit_reason="in_progress",
                        last_successful_lane=last_successful_lane,
                    )

                if (
                    _lane_enabled(spec, "thread_matrix")
                    and "thread_matrix" not in completed_lanes
                ):
                    logger.set_state(
                        phase="lane_thread_matrix",
                        lane="thread_matrix",
                        completed_lanes=len(completed_lanes),
                    )
                    _write_status(
                        layout, run_id=run_id, state="lane_thread_matrix", ok=True
                    )
                    _lane_attempts(
                        repo_root=repo_root,
                        layout=layout,
                        spec=spec,
                        lane_id="thread_matrix",
                        ablation_id="all_on",
                        env_overrides={},
                        prompt=spec.scenario_pack.canonical_prompt,
                        warmup_runs=0,
                        measured_runs=1,
                        thread_matrix=thread_matrix,
                        use_thread_matrix=True,
                        attempt_rows=attempt_rows,
                        failure_rows=failure_rows,
                        completed_attempt_ids=completed_attempt_ids,
                        resume_completed=resume_completed,
                    )
                    completed_lanes.append("thread_matrix")
                    _emit_mission_receipt(
                        node_id="thread_matrix",
                        phase="native_surface",
                        kind="lane",
                        status="completed",
                        blocking=False,
                        lane="thread_matrix",
                    )
                    logger.set_state(
                        completed_lanes=len(completed_lanes), lane="thread_matrix"
                    )
                    last_successful_lane = "thread_matrix"
                    _write_checkpoint(
                        layout,
                        completed_lanes=completed_lanes,
                        completed_attempt_ids=sorted(completed_attempt_ids),
                        run_exit_reason="in_progress",
                        last_successful_lane=last_successful_lane,
                    )

                if _lane_enabled(spec, "continuous_scheduler"):
                    if "continuous_scheduler" not in completed_lanes:
                        logger.set_state(
                            phase="lane_continuous_scheduler",
                            lane="continuous_scheduler",
                            completed_lanes=len(completed_lanes),
                        )
                        _write_status(
                            layout,
                            run_id=run_id,
                            state="lane_continuous_scheduler",
                            ok=True,
                        )
                        continuous_payload = _run_continuous_surface(
                            repo_root=repo_root,
                            layout=layout,
                            spec=spec,
                            runtime_payload=runtime_payload,
                            model_thread_overrides=model_thread_overrides or None,
                            model_continuous_overrides=(
                                model_continuous_overrides or None
                            ),
                        )
                        completed_lanes.append("continuous_scheduler")
                        _emit_mission_receipt(
                            node_id="continuous_scheduler",
                            phase="continuous",
                            kind="lane",
                            status="completed",
                            blocking=False,
                            lane="continuous_scheduler",
                        )
                        logger.set_state(
                            completed_lanes=len(completed_lanes),
                            lane="continuous_scheduler",
                        )
                        last_successful_lane = "continuous_scheduler"
                        _write_checkpoint(
                            layout,
                            completed_lanes=completed_lanes,
                            completed_attempt_ids=sorted(completed_attempt_ids),
                            run_exit_reason="in_progress",
                            last_successful_lane=last_successful_lane,
                        )
                    else:
                        for model in spec.models:
                            path = layout.continuous_dir / f"{_safe_name(model)}.json"
                            if path.exists():
                                continuous_payload.append(
                                    json.loads(path.read_text(encoding="utf-8"))
                                )

                if _lane_enabled(spec, "kernel_microbench"):
                    if "kernel_microbench" not in completed_lanes:
                        logger.set_state(
                            phase="lane_kernel_microbench",
                            lane="kernel_microbench",
                            completed_lanes=len(completed_lanes),
                        )
                        _write_status(
                            layout,
                            run_id=run_id,
                            state="lane_kernel_microbench",
                            ok=True,
                        )
                        kernel_summary = _run_kernel_harness(
                            layout=layout,
                            spec=spec,
                            preflight_payload=preflight_payload,
                        )
                        completed_lanes.append("kernel_microbench")
                        _emit_mission_receipt(
                            node_id="kernel_microbench",
                            phase="kernel_harness",
                            kind="lane",
                            status="completed",
                            blocking=False,
                            lane="kernel_microbench",
                        )
                        logger.set_state(
                            completed_lanes=len(completed_lanes),
                            lane="kernel_microbench",
                        )
                        last_successful_lane = "kernel_microbench"
                        _write_checkpoint(
                            layout,
                            completed_lanes=completed_lanes,
                            completed_attempt_ids=sorted(completed_attempt_ids),
                            run_exit_reason="in_progress",
                            last_successful_lane=last_successful_lane,
                        )
                    else:
                        kernel_summary = json.loads(
                            layout.kernel_summary_json.read_text(encoding="utf-8")
                        )

                if _lane_enabled(spec, "quality_eval"):
                    if "quality_eval" not in completed_lanes:
                        logger.set_state(
                            phase="lane_quality_eval",
                            lane="quality_eval",
                            completed_lanes=len(completed_lanes),
                        )
                        _write_status(
                            layout, run_id=run_id, state="lane_quality_eval", ok=True
                        )
                        quality_payload = _run_quality_evals(layout=layout, spec=spec)
                        completed_lanes.append("quality_eval")
                        _emit_mission_receipt(
                            node_id="quality_eval",
                            phase="quality_eval",
                            kind="lane",
                            status="completed",
                            blocking=False,
                            lane="quality_eval",
                        )
                        logger.set_state(
                            completed_lanes=len(completed_lanes), lane="quality_eval"
                        )
                        last_successful_lane = "quality_eval"
                        _write_checkpoint(
                            layout,
                            completed_lanes=completed_lanes,
                            completed_attempt_ids=sorted(completed_attempt_ids),
                            run_exit_reason="in_progress",
                            last_successful_lane=last_successful_lane,
                        )
                    else:
                        quality_payload = json.loads(
                            layout.quality_summary_json.read_text(encoding="utf-8")
                        )

                if _lane_enabled(spec, "memory_replay"):
                    if "memory_replay" not in completed_lanes:
                        logger.set_state(
                            phase="lane_memory_replay",
                            lane="memory_replay",
                            completed_lanes=len(completed_lanes),
                        )
                        _write_status(
                            layout, run_id=run_id, state="lane_memory_replay", ok=True
                        )
                        memory_replay_payload = _run_memory_replay_lane(
                            repo_root=repo_root,
                            layout=layout,
                            spec=spec,
                        )
                        completed_lanes.append("memory_replay")
                        _emit_mission_receipt(
                            node_id="memory_replay",
                            phase="memory_replay",
                            kind="lane",
                            status="completed",
                            blocking=False,
                            lane="memory_replay",
                        )
                        logger.set_state(
                            completed_lanes=len(completed_lanes), lane="memory_replay"
                        )
                        last_successful_lane = "memory_replay"
                        _write_checkpoint(
                            layout,
                            completed_lanes=completed_lanes,
                            completed_attempt_ids=sorted(completed_attempt_ids),
                            run_exit_reason="in_progress",
                            last_successful_lane=last_successful_lane,
                        )
                    else:
                        memory_replay_payload = json.loads(
                            layout.memory_replay_summary_json.read_text(
                                encoding="utf-8"
                            )
                        )
                else:
                    memory_replay_payload = _empty_memory_replay_summary()

                summary_rows = [
                    row
                    for row in attempt_rows
                    if str(row.get("lane_id") or "") == "canonical_all_on"
                    and str(row.get("ablation_id") or "") == "all_on"
                ]
                baseline_summary = _resolve_baseline(
                    audit_root=Path(str(args.out_root)),
                    current_run_id=run_id,
                    model_set=spec.models,
                    compare_to=compare_to,
                    current_topology_hash=topology_hash_from_preflight(
                        spec=spec,
                        preflight_payload=preflight_payload,
                    ),
                    current_host_fingerprint=str(
                        ((preflight_payload.get("host") or {}).get("host_fingerprint"))
                        or ""
                    ),
                    current_prompt_contract_hash=prompt_contract_hash,
                )
                comparator_catalog = _resolve_comparator_catalog(
                    audit_root=Path(str(args.out_root)),
                    current_run_id=run_id,
                    model_set=spec.models,
                    current_topology_hash=topology_hash_from_preflight(
                        spec=spec,
                        preflight_payload=preflight_payload,
                    ),
                    current_host_fingerprint=str(
                        ((preflight_payload.get("host") or {}).get("host_fingerprint"))
                        or ""
                    ),
                    current_prompt_contract_hash=prompt_contract_hash,
                )

            _write_status(layout, run_id=run_id, state="finalizing", ok=True)
            summary = legacy_runner._build_summary(
                summary_rows,
                baseline_summary=baseline_summary,
                baseline_mode=baseline_summary is None,
                model_order=spec.models,
            )
            summary["schema_version"] = "native_qsg_audit.v3"
            summary["run_id"] = run_id
            summary["profile_name"] = spec.profile_name
            summary["prompt_hash"] = prompt_hash
            summary["prompt_contract_hash"] = prompt_contract_hash
            summary["memory_snapshot_hash"] = memory_snapshot_hash
            summary["feature_toggle_hash"] = feature_toggle_hash
            summary["quality"] = quality_payload
            _augment_summary_with_quality_and_runtime(
                summary=summary,
                quality_payload=quality_payload,
                continuous_payload=continuous_payload,
                attempt_rows=attempt_rows,
            )
            summary["ablation_deltas"] = []
            summary["kernel_hotspots"] = _fuse_hotspots(summary_rows, kernel_summary)
            summary["host_compliance"] = preflight_payload
            summary["continuous"] = {
                "schema_version": CONTINUOUS_SCHEMA_VERSION,
                "results": continuous_payload,
            }
            summary["memory_replay"] = memory_replay_payload
            if calibration_bundle is not None:
                summary["calibration"] = {
                    "contracts": dict(calibration_bundle.get("contracts") or {}),
                    "admission": dict(calibration_bundle.get("admission") or {}),
                    "winners": dict(calibration_bundle.get("winners") or {}),
                    "thread_frontiers": dict(
                        calibration_bundle.get("thread_frontiers") or {}
                    ),
                    "scheduler_frontiers": dict(
                        calibration_bundle.get("scheduler_frontiers") or {}
                    ),
                    "kernel_summary": dict(
                        calibration_bundle.get("kernel_summary") or {}
                    ),
                }
                _write_json_artifact(
                    layout=layout,
                    path=layout.reports_dir / "tuning_receipt.json",
                    payload=_tuning_receipt_payload(
                        spec=spec,
                        preflight_payload=preflight_payload,
                        calibration_bundle=calibration_bundle,
                    ),
                    summary="wrote tuning receipt",
                    phase="calibration",
                    lane="calibration",
                )
                _write_json_artifact(
                    layout=layout,
                    path=layout.reports_dir / "tuning_remediation.json",
                    payload=_tuning_remediation_payload(
                        spec=spec,
                        preflight_payload=preflight_payload,
                        calibration_bundle=calibration_bundle,
                    ),
                    summary="wrote tuning remediation packet",
                    phase="calibration",
                    lane="calibration",
                )
            comparisons = {
                "schema_version": "native_qsg_suite.comparisons.v1",
                "baseline": baseline_summary,
                "comparators": comparator_catalog,
                "deltas": summary["ablation_deltas"],
                "compare_to": compare_to,
                "selected_baseline_run_id": str(
                    (baseline_summary or {}).get("run_id") or ""
                ),
                "prompt_contract_hash": prompt_contract_hash,
                "memory_snapshot_hash": memory_snapshot_hash,
                "feature_toggle_hash": feature_toggle_hash,
            }
            summary["comparisons"] = comparisons
            summary["baseline_run_id"] = str(
                (baseline_summary or {}).get("run_id") or ""
            )
            if spec.tuning_contract_policy == "required":
                _apply_tuning_baseline_gates(
                    summary=summary,
                    preflight_payload=preflight_payload,
                    repo_root=repo_root,
                    failure_rows=failure_rows,
                )
            summary["quality_governance"] = _acceptance_governance_report(
                summary=summary,
                quality_payload=quality_payload,
                attempt_rows=attempt_rows,
            )
            summary["assurance"] = {
                "strict_native_decode": _strict_native_decode_receipt(
                    spec,
                    attempt_rows,
                )
            }
            _append_governance_failures(
                summary=summary,
                governance=summary["quality_governance"],
                failure_rows=failure_rows,
            )
            _append_strict_native_decode_failures(
                summary=summary,
                receipt=dict(summary["assurance"].get("strict_native_decode") or {}),
                failure_rows=failure_rows,
            )
            summary["failure_count"] = len(failure_rows)
            summary["failure_counts"] = {
                "total": len(failure_rows),
                "gate_failure": sum(
                    1
                    for item in failure_rows
                    if str(item.get("failure_kind") or "") == "gate_failure"
                ),
                "execution_failure": sum(
                    1
                    for item in failure_rows
                    if str(item.get("failure_kind") or "") == "execution_failure"
                ),
            }
            summary["failed_attempt_ids"] = [
                str(item.get("attempt_id") or "")
                for item in failure_rows
                if str(item.get("attempt_id") or "").strip()
            ]
            summary["completed_attempts"] = len(completed_attempt_ids)
            summary["planned_attempts"] = len(attempt_rows)
            summary["agent_triage"] = _build_triage(summary)
            summary["overall_pass"] = (
                bool(summary.get("overall_pass", False)) and not failure_rows
            )
            summary["pass"] = bool(summary["overall_pass"])
            terminal_state = "completed_pass" if summary["pass"] else "completed_fail"
            run_exit_reason = "suite_pass" if summary["pass"] else "suite_fail"
            summary["certification_state"] = _final_certification_state(
                preflight_payload, summary["pass"]
            )
            if spec.profile_name == "platinum":
                summary["publication_manifest"] = _build_publication_manifest(
                    layout=layout,
                    summary=summary,
                )
            summary["run_exit_reason"] = run_exit_reason
            summary["terminal_state"] = terminal_state
            summary["last_successful_lane"] = last_successful_lane
            persist_result = _persist_summary_bundle(
                layout=layout, summary=summary, comparisons=comparisons
            )
            summary = dict(persist_result.get("summary") or summary)
            if bool(persist_result.get("ok", False)):
                _emit_terminal_result(layout, summary)
                exit_code = 0 if summary["pass"] else 1
            else:
                terminal_state = str(summary.get("terminal_state") or "internal_error")
                run_exit_reason = str(
                    summary.get("run_exit_reason") or "summary_persistence_failed"
                )
                exit_code = 1
    except KeyboardInterrupt:
        terminal_state = "interrupted_incomplete"
        run_exit_reason = "keyboard_interrupt"
        _terminal("Benchmark suite interrupted by signal.")
        exit_code = 130
    except Exception as exc:  # pragma: no cover - exercised in integration
        terminal_state = "internal_error"
        run_exit_reason = f"internal_error:{type(exc).__name__}"
        _terminal(f"Benchmark suite failed: {exc}")
        _console_log(layout, traceback.format_exc())
        exit_code = 1
    finally:
        if terminal_state not in TERMINAL_STATES:
            invalid_state = terminal_state
            terminal_state = "internal_error"
            run_exit_reason = f"invalid_terminal_state:{invalid_state}"
        if summary is None:
            _touch_minimal_lane_artifacts(layout)
            summary = _summary_payload(
                run_id=run_id,
                quality_payload=_empty_quality_summary(),
                kernel_hotspots=[],
                host_compliance=preflight_payload,
                failure_rows=failure_rows,
                planned_attempts=len(attempt_rows),
                completed_attempts=len(completed_attempt_ids),
                run_exit_reason=run_exit_reason,
                terminal_state=terminal_state,
                last_successful_lane=last_successful_lane,
                overall_pass=False,
                certification_state=_final_certification_state(
                    preflight_payload, False
                ),
            )
            persist_result = _persist_summary_bundle(
                layout=layout, summary=summary, comparisons=comparisons
            )
            summary = dict(persist_result.get("summary") or summary)
            if not bool(persist_result.get("ok", False)):
                terminal_state = str(summary.get("terminal_state") or "internal_error")
                run_exit_reason = str(
                    summary.get("run_exit_reason") or "summary_persistence_failed"
                )
                exit_code = 1
        else:
            summary["run_exit_reason"] = run_exit_reason
            summary["terminal_state"] = terminal_state
            summary["last_successful_lane"] = last_successful_lane
            summary["pass"] = bool(summary.get("overall_pass", False))
            summary["certification_state"] = _final_certification_state(
                preflight_payload,
                bool(summary.get("overall_pass", False)),
            )
            persist_result = _persist_summary_bundle(
                layout=layout, summary=summary, comparisons=comparisons
            )
            summary = dict(persist_result.get("summary") or summary)
            if not bool(persist_result.get("ok", False)):
                terminal_state = str(summary.get("terminal_state") or "internal_error")
                run_exit_reason = str(
                    summary.get("run_exit_reason") or "summary_persistence_failed"
                )
                exit_code = 1

        _write_artifact_index(layout)
        if (
            summary is not None
            and assurance_plan is not None
            and assurance_context is not None
        ):
            try:
                assurance_payload = materialize_assurance_artifacts(
                    repo_root=repo_root,
                    layout=layout,
                    spec=spec,
                    context=assurance_context,
                    plan=assurance_plan,
                    summary=summary,
                    preflight_payload=preflight_payload,
                )
                runtime_gates = dict(assurance_payload.get("runtime_gates") or {})
                assurance_summary = dict(summary.get("assurance") or {})
                assurance_summary.update(
                    {
                        "assurance_level": str(
                            assurance_plan.get("assurance_level") or ""
                        ),
                        "evidence_class": str(
                            assurance_plan.get("evidence_class") or ""
                        ),
                        "required_runtime_gates": list(
                            assurance_plan.get("required_runtime_gates") or []
                        ),
                        "runtime_gates_passed": bool(
                            runtime_gates.get("passed", False)
                        ),
                        "missing_artifacts": list(
                            runtime_gates.get("missing_artifacts") or []
                        ),
                    }
                )
                summary["assurance"] = assurance_summary
                if not bool(runtime_gates.get("passed", False)):
                    summary["overall_pass"] = False
                    summary["pass"] = False
                    terminal_state = "completed_fail"
                    run_exit_reason = "assurance_runtime_gates_failed"
                    summary["run_exit_reason"] = run_exit_reason
                    summary["terminal_state"] = terminal_state
                    _terminal("Assurance runtime gates failed.")
                    for item in list(runtime_gates.get("missing_artifacts") or []):
                        _terminal(f"- {item}")
                    exit_code = 1
                    persist_result = _persist_summary_bundle(
                        layout=layout, summary=summary, comparisons=comparisons
                    )
                    summary = dict(persist_result.get("summary") or summary)
            except Exception as exc:
                terminal_state = "internal_error"
                run_exit_reason = f"assurance_artifacts_failed:{type(exc).__name__}"
                summary["overall_pass"] = False
                summary["pass"] = False
                summary["run_exit_reason"] = run_exit_reason
                summary["terminal_state"] = terminal_state
                _terminal(f"Assurance control-plane write failed: {exc}")
                exit_code = 1
                persist_result = _persist_summary_bundle(
                    layout=layout, summary=summary, comparisons=comparisons
                )
                summary = dict(persist_result.get("summary") or summary)
        if summary is not None:
            try:
                control_plane_payload = materialize_control_plane_artifacts(
                    repo_root=repo_root,
                    layout=layout,
                    spec=spec,
                    manifest=manifest,
                    launch_runtime=launch_runtime,
                    preflight_payload=preflight_payload,
                    summary=summary,
                    comparisons=comparisons,
                    assurance_plan=assurance_plan,
                    completed_lanes=completed_lanes,
                    compare_to=compare_to,
                    mission_dump=mission_dump,
                    closure_advisory=closure_advisory,
                    capsule_enabled=capsule_enabled,
                )
                summary["control_plane"] = dict(
                    control_plane_payload.get("summary") or {}
                )
                summary["topology_passport"] = dict(
                    control_plane_payload.get("topology_passport") or {}
                )
                summary["variance_budget"] = dict(
                    control_plane_payload.get("variance_budget") or {}
                )
                summary["baseline_lineage"] = dict(
                    control_plane_payload.get("baseline_lineage") or {}
                )
                summary["closure_result"] = dict(
                    control_plane_payload.get("closure_result") or {}
                )
                summary["spc_report"] = dict(
                    control_plane_payload.get("spc_report") or {}
                )
                summary["traceability_graph"] = dict(
                    control_plane_payload.get("traceability_graph") or {}
                )
                summary["saguaro_verify"] = dict(
                    control_plane_payload.get("saguaro_verify") or {}
                )
                summary["advisory_bundle"] = dict(
                    control_plane_payload.get("advisory_bundle") or {}
                )
                summary["capsule_archive"] = dict(
                    control_plane_payload.get("capsule_archive") or {}
                )
                persist_result = _persist_summary_bundle(
                    layout=layout, summary=summary, comparisons=comparisons
                )
                summary = dict(persist_result.get("summary") or summary)
            except Exception as exc:
                terminal_state = "internal_error"
                run_exit_reason = f"control_plane_failed:{type(exc).__name__}"
                summary["overall_pass"] = False
                summary["pass"] = False
                summary["run_exit_reason"] = run_exit_reason
                summary["terminal_state"] = terminal_state
                _terminal(f"Benchmark control-plane write failed: {exc}")
                exit_code = 1
                persist_result = _persist_summary_bundle(
                    layout=layout, summary=summary, comparisons=comparisons
                )
                summary = dict(persist_result.get("summary") or summary)
        try:
            logger.event_store.export_run(
                run_id,
                output_path=str(layout.telemetry_event_store_export_json),
                limit=20_000,
            )
        except Exception:
            pass
        try:
            black_box_manifest = black_box.finalize(
                stop_reason=run_exit_reason,
                success=(
                    bool(summary.get("overall_pass", False))
                    if summary is not None
                    else False
                ),
                extra_metadata={
                    "terminal_state": terminal_state,
                    "last_successful_lane": last_successful_lane,
                },
            )
            write_json_atomic(layout.black_box_manifest_json, black_box_manifest)
            if summary is not None:
                summary["black_box"] = {
                    "path": str(
                        layout.black_box_manifest_json.relative_to(layout.root)
                    ),
                    "timeline_count": int(
                        black_box_manifest.get("timeline_count", 0) or 0
                    ),
                    "replay": dict(black_box_manifest.get("replay") or {}),
                }
        except Exception as exc:
            if summary is not None:
                summary["black_box"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        _write_artifact_index(layout)
        if summary is not None:
            try:
                bundle_sync = _sync_benchmark_truth_bundle(
                    repo_root=repo_root,
                    layout=layout,
                    spec=spec,
                    summary=summary,
                    manifest=manifest,
                    preflight_payload=preflight_payload,
                    comparisons=comparisons,
                )
                summary.setdefault("control_plane", {})["benchmark_truth_bundle"] = {
                    "bundle_path": str(bundle_sync.get("bundle_path") or ""),
                    "platform_evidence_path": str(
                        bundle_sync.get("platform_evidence_path") or ""
                    ),
                }
                summary.setdefault("advisory_bundle", {})["silver_convergence"] = dict(
                    bundle_sync.get("convergence_manifest") or {}
                )
                persist_result = _persist_summary_bundle(
                    layout=layout, summary=summary, comparisons=comparisons
                )
                summary = dict(persist_result.get("summary") or summary)
            except Exception as exc:
                summary.setdefault("control_plane", {})["benchmark_truth_bundle"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                persist_result = _persist_summary_bundle(
                    layout=layout, summary=summary, comparisons=comparisons
                )
                summary = dict(persist_result.get("summary") or summary)
        missing = _verify_required_artifacts(layout)
        if missing:
            terminal_state = "internal_error"
            run_exit_reason = "missing_required_artifacts"
            summary["overall_pass"] = False
            summary["pass"] = False
            summary["run_exit_reason"] = run_exit_reason
            summary["terminal_state"] = terminal_state
            summary["certification_state"] = _final_certification_state(
                preflight_payload, False
            )
            summary.setdefault("host_compliance", {})["missing_artifacts"] = missing
            persist_result = _persist_summary_bundle(
                layout=layout, summary=summary, comparisons=comparisons
            )
            summary = dict(persist_result.get("summary") or summary)
            _write_artifact_index(layout)
            _terminal("Missing required artifacts:")
            for item in missing:
                _terminal(f"- {item}")
            if exit_code == 0:
                exit_code = 1

        _write_checkpoint(
            layout,
            completed_lanes=completed_lanes,
            completed_attempt_ids=sorted(completed_attempt_ids),
            run_exit_reason=run_exit_reason,
            last_successful_lane=last_successful_lane,
        )
        _write_status(
            layout,
            run_id=run_id,
            state=terminal_state,
            ok=terminal_state == "completed_pass",
            run_exit_reason=run_exit_reason,
            last_successful_lane=last_successful_lane,
            terminal=True,
        )
        set_active_logger(None)
        logger.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
