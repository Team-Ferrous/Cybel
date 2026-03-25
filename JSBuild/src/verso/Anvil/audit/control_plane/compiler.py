from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from audit.control_plane.advisory import build_advisory_bundle
from audit.control_plane.capsules import build_capsule_archive, build_capsule_manifest
from audit.control_plane.comparators import build_baseline_lineage
from audit.control_plane.ledger import build_run_ledger
from audit.control_plane.mission import MissionContext, compile_mission_graph
from audit.control_plane.spc import build_spc_report
from audit.control_plane.topology import build_topology_passport
from audit.control_plane.traceability import (
    build_traceability_graph,
    capture_saguaro_verify_envelope,
    skipped_saguaro_verify_envelope,
)
from audit.control_plane.variance import build_variance_budget
from audit.control_plane.reducers import now_iso, summarize_gate_status
from audit.runner.suite_profiles import BenchmarkSuiteSpec
from audit.store.schema_validation import validate_payload
from audit.store.suite_layout import SuiteRunLayout, required_suite_artifacts
from audit.store.writer import write_json_atomic, write_ndjson_atomic

CLOSURE_RESULT_SCHEMA_VERSION = "native_qsg_suite.closure_result.v1"


def _write_validated(path: Path, schema_name: str, payload: dict[str, Any]) -> None:
    validate_payload(schema_name, payload)
    write_json_atomic(path, payload)


def _build_closure_result(
    mission: MissionContext,
    *,
    saguaro_verify: dict[str, Any],
    closure_advisory: bool,
    ignored_paths: set[Path] | None = None,
) -> dict[str, Any]:
    ignored = {path.resolve() for path in (ignored_paths or set())}
    artifacts = []
    missing = []
    for path in required_suite_artifacts(mission.layout):
        if path.resolve() in ignored:
            continue
        relative = path.relative_to(mission.layout.root).as_posix()
        exists = path.exists()
        artifacts.append({"path": relative, "exists": exists})
        if not exists:
            missing.append(relative)
    runtime_gates = {
        "required": list(
            (mission.assurance_plan or {}).get("required_runtime_gates") or []
        ),
        "passed": bool(
            (mission.summary.get("assurance") or {}).get("runtime_gates_passed", True)
        ),
        "missing_artifacts": list(
            (mission.summary.get("assurance") or {}).get("missing_artifacts") or []
        ),
    }
    unresolved = list(missing)
    unresolved.extend(
        str(item) for item in runtime_gates["missing_artifacts"] if str(item).strip()
    )
    return {
        "schema_version": CLOSURE_RESULT_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "advisory_only": bool(closure_advisory),
        "overall_pass": not unresolved
        and bool(mission.summary.get("overall_pass", False)),
        "artifacts": artifacts,
        "runtime_gates": runtime_gates,
        "saguaro_verify": {
            "status": str(saguaro_verify.get("status") or ""),
            "available": bool(saguaro_verify.get("available", False)),
            "gate_status": summarize_gate_status(
                bool(saguaro_verify.get("available", False)),
                issues=(
                    [str(saguaro_verify.get("error") or "")]
                    if saguaro_verify.get("error")
                    else []
                ),
            ),
        },
        "required_questions": [
            "intended",
            "happened",
            "observed",
            "concluded",
            "unresolved",
        ],
        "unresolved": unresolved,
    }


def materialize_control_plane_artifacts(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    manifest: dict[str, Any],
    launch_runtime: dict[str, Any],
    preflight_payload: dict[str, Any],
    summary: dict[str, Any],
    comparisons: dict[str, Any],
    assurance_plan: dict[str, Any] | None,
    completed_lanes: list[str],
    verify_runner: Callable[..., Any] | None = None,
    compare_to: str = "latest_compatible",
    mission_dump: bool = False,
    variance_report_only: bool = False,
    closure_advisory: bool = False,
    capsule_enabled: bool = False,
) -> dict[str, Any]:
    mission = MissionContext(
        repo_root=repo_root,
        layout=layout,
        spec=spec,
        run_id=layout.run_id,
        manifest=manifest,
        launch_runtime=launch_runtime,
        preflight_payload=preflight_payload,
        summary=summary,
        comparisons=comparisons,
        assurance_plan=assurance_plan,
        completed_lanes=tuple(completed_lanes),
    )
    terminal_state = str(summary.get("terminal_state") or "")
    if variance_report_only or terminal_state.startswith("completed"):
        saguaro_verify = capture_saguaro_verify_envelope(
            repo_root,
            command_runner=verify_runner,
        )
    else:
        saguaro_verify = skipped_saguaro_verify_envelope(
            reason=f"terminal_state:{terminal_state or 'unknown'}"
        )
    saguaro_verify["run_id"] = layout.run_id
    mission_graph = compile_mission_graph(
        spec,
        assurance_plan=assurance_plan,
        compare_to=compare_to,
        variance_report_only=variance_report_only,
        closure_advisory=closure_advisory,
        capsule_enabled=capsule_enabled,
    )
    _write_validated(layout.mission_graph_json, "mission_graph.schema.json", mission_graph)
    topology_passport = build_topology_passport(
        mission,
        saguaro_verify=saguaro_verify,
    )
    variance_budget = build_variance_budget(
        mission,
        topology_passport=topology_passport,
    )
    baseline_lineage = build_baseline_lineage(
        mission,
        topology_passport=topology_passport,
        compare_to=compare_to,
    )
    spc_report = build_spc_report(
        mission,
        topology_passport=topology_passport,
        compare_to=compare_to,
    )
    _write_validated(
        layout.saguaro_verify_json, "saguaro_verify.schema.json", saguaro_verify
    )
    _write_validated(
        layout.topology_passport_json,
        "topology_passport.schema.json",
        topology_passport,
    )
    _write_validated(
        layout.variance_budget_json,
        "variance_budget.schema.json",
        variance_budget,
    )
    _write_validated(
        layout.baseline_lineage_json,
        "baseline_lineage.schema.json",
        baseline_lineage,
    )
    _write_validated(layout.spc_report_json, "spc_report.schema.json", spc_report)
    run_ledger = build_run_ledger(mission)
    _write_validated(layout.run_ledger_json, "run_ledger.schema.json", run_ledger)
    write_json_atomic(
        layout.mission_receipts_json,
        {
            "schema_version": "native_qsg_suite.mission_receipts.v1",
            "run_id": mission.run_id,
            "receipts": list(run_ledger.get("node_receipts") or []),
        },
    )
    write_ndjson_atomic(
        layout.run_ledger_ndjson,
        [
            {
                "run_id": mission.run_id,
                "kind": "lifecycle",
                **row,
            }
            for row in list(run_ledger.get("lifecycle") or [])
        ]
        + [
            {
                "run_id": mission.run_id,
                "kind": "event",
                **row,
            }
            for row in list(run_ledger.get("events") or [])
        ]
        + [
            {
                "run_id": mission.run_id,
                "kind": "receipt",
                **row,
            }
            for row in list(run_ledger.get("node_receipts") or [])
        ],
    )
    closure_result = _build_closure_result(
        mission,
        saguaro_verify=saguaro_verify,
        closure_advisory=closure_advisory,
        ignored_paths={
            layout.traceability_graph_json,
            layout.advisory_bundle_json,
            layout.run_capsule_manifest_json,
            layout.closure_result_json,
            layout.telemetry_event_store_export_json,
            layout.black_box_manifest_json,
        },
    )
    advisory_bundle = build_advisory_bundle(
        mission,
        topology_passport=topology_passport,
        variance_budget=variance_budget,
        baseline_lineage=baseline_lineage,
        run_ledger=run_ledger,
        closure_result=closure_result,
    )
    write_json_atomic(layout.advisory_bundle_json, advisory_bundle)
    traceability_graph = build_traceability_graph(
        mission,
        topology_passport=topology_passport,
        variance_budget=variance_budget,
        baseline_lineage=baseline_lineage,
        run_ledger=run_ledger,
        closure_result=closure_result,
        saguaro_verify=saguaro_verify,
    )
    _write_validated(
        layout.traceability_graph_json,
        "traceability_graph.schema.json",
        traceability_graph,
    )
    closure_result = _build_closure_result(
        mission,
        saguaro_verify=saguaro_verify,
        closure_advisory=closure_advisory,
        ignored_paths={
            layout.advisory_bundle_json,
            layout.run_capsule_manifest_json,
            layout.closure_result_json,
            layout.telemetry_event_store_export_json,
            layout.black_box_manifest_json,
        },
    )
    _write_validated(
        layout.closure_result_json,
        "closure_result.schema.json",
        closure_result,
    )
    artifact_paths = list(required_suite_artifacts(layout))
    capsule_manifest = build_capsule_manifest(mission, artifact_paths=artifact_paths)
    _write_validated(
        layout.run_capsule_manifest_json,
        "capsule_manifest.schema.json",
        capsule_manifest,
    )
    capsule_archive = None
    if capsule_enabled:
        capsule_archive = build_capsule_archive(
            mission,
            output_path=layout.run_capsule_tgz,
            artifact_paths=artifact_paths,
        )
    closure_result = _build_closure_result(
        mission,
        saguaro_verify=saguaro_verify,
        closure_advisory=closure_advisory,
        ignored_paths={
            layout.closure_result_json,
            layout.telemetry_event_store_export_json,
            layout.black_box_manifest_json,
        },
    )
    _write_validated(
        layout.closure_result_json,
        "closure_result.schema.json",
        closure_result,
    )
    return {
        "summary": {
            "profile_name": spec.profile_name,
            "completed_lanes": list(completed_lanes),
            "topology_hash": str(topology_passport.get("topology_hash") or ""),
            "cohort_key": str(topology_passport.get("cohort_key") or ""),
            "variance_within_budget": bool(
                (variance_budget.get("overall") or {}).get("within_budget", False)
            ),
            "history_aware_comparison": bool(baseline_lineage.get("comparable", False)),
            "closure_pass": bool(closure_result.get("overall_pass", False)),
            "spc_status": str(spc_report.get("status") or ""),
            "mission_node_count": len(mission_graph.get("nodes") or []),
            "host_identity": {
                "host_fingerprint": str(topology_passport.get("host_fingerprint") or ""),
                "topology_hash": str(topology_passport.get("topology_hash") or ""),
            },
            "capsule_archive": (
                str(capsule_archive.get("path") or "") if capsule_archive else ""
            ),
            "mission_dump": bool(mission_dump),
        },
        "mission_graph": mission_graph,
        "topology_passport": topology_passport,
        "variance_budget": variance_budget,
        "baseline_lineage": baseline_lineage,
        "closure_result": closure_result,
        "traceability_graph": traceability_graph,
        "saguaro_verify": saguaro_verify,
        "spc_report": spc_report,
        "advisory_bundle": advisory_bundle,
        "capsule_archive": capsule_archive or {},
    }
