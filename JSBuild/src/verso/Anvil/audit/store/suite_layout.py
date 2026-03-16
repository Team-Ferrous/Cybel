from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SuiteRunLayout:
    run_id: str
    root: Path
    events_ndjson: Path
    terminal_transcript_log: Path
    manifest_json: Path
    suite_resolved_json: Path
    preflight_json: Path
    environment_json: Path
    checkpoint_json: Path
    suite_status_json: Path
    assurance_plan_json: Path
    closure_matrix_json: Path
    change_manifest_json: Path
    traceability_json: Path
    evidence_bundle_json: Path
    runtime_gates_json: Path
    telemetry_contract_json: Path
    chronicle_json: Path
    summary_json: Path
    summary_failed_json: Path
    summary_md: Path
    report_html: Path
    triage_json: Path
    console_log: Path
    report_md: Path
    executive_summary_md: Path
    index_json: Path
    agent_handoff_json: Path
    metrics_rollup_json: Path
    run_ledger_json: Path
    run_ledger_ndjson: Path
    mission_receipts_json: Path
    mission_graph_json: Path
    topology_passport_json: Path
    variance_budget_json: Path
    baseline_lineage_json: Path
    closure_result_json: Path
    traceability_graph_json: Path
    spc_report_json: Path
    advisory_bundle_json: Path
    run_capsule_manifest_json: Path
    run_capsule_tgz: Path
    saguaro_verify_json: Path
    telemetry_event_store_db: Path
    telemetry_event_store_export_json: Path
    black_box_manifest_json: Path
    native_dir: Path
    kernel_dir: Path
    continuous_dir: Path
    eval_dir: Path
    telemetry_dir: Path
    comparison_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    native_attempts_ndjson: Path
    native_phases_ndjson: Path
    native_failures_ndjson: Path
    kernel_attempts_ndjson: Path
    kernel_summary_json: Path
    quality_attempts_ndjson: Path
    quality_summary_json: Path
    memory_replay_summary_json: Path
    comparisons_json: Path


def resolve_suite_layout(base_dir: Path, run_id: str) -> SuiteRunLayout:
    root = base_dir / "runs" / str(run_id)
    native_dir = root / "native"
    kernel_dir = root / "kernel"
    continuous_dir = root / "continuous"
    eval_dir = root / "eval"
    telemetry_dir = root / "telemetry"
    comparison_dir = root / "comparison"
    reports_dir = root / "reports"
    artifacts_dir = root / "artifacts"
    return SuiteRunLayout(
        run_id=str(run_id),
        root=root,
        events_ndjson=root / "events.ndjson",
        terminal_transcript_log=root / "terminal_transcript.log",
        manifest_json=root / "manifest.json",
        suite_resolved_json=root / "suite_resolved.json",
        preflight_json=root / "preflight.json",
        environment_json=root / "environment.json",
        checkpoint_json=root / "checkpoint.json",
        suite_status_json=root / "suite_status.json",
        assurance_plan_json=root / "assurance_plan.json",
        closure_matrix_json=root / "closure_matrix.json",
        change_manifest_json=root / "change_manifest.json",
        traceability_json=root / "traceability.json",
        evidence_bundle_json=root / "evidence_bundle.json",
        runtime_gates_json=root / "runtime_gates.json",
        telemetry_contract_json=root / "telemetry_contract.json",
        chronicle_json=root / "chronicle.json",
        summary_json=root / "summary.json",
        summary_failed_json=root / "summary_failed.json",
        summary_md=root / "summary.md",
        report_html=root / "report.html",
        triage_json=root / "triage.json",
        console_log=root / "console.log",
        report_md=reports_dir / "report.md",
        executive_summary_md=reports_dir / "executive_summary.md",
        index_json=root / "index.json",
        agent_handoff_json=root / "agent_handoff.json",
        metrics_rollup_json=root / "metrics_rollup.json",
        run_ledger_json=root / "run_ledger.json",
        run_ledger_ndjson=root / "run_ledger.ndjson",
        mission_receipts_json=root / "mission_receipts.json",
        mission_graph_json=root / "mission_graph.json",
        topology_passport_json=root / "topology_passport.json",
        variance_budget_json=root / "variance_budget.json",
        baseline_lineage_json=comparison_dir / "baseline_lineage.json",
        closure_result_json=root / "closure_result.json",
        traceability_graph_json=root / "traceability_graph.json",
        spc_report_json=root / "spc_report.json",
        advisory_bundle_json=root / "advisory_bundle.json",
        run_capsule_manifest_json=root / "run_capsule.manifest.json",
        run_capsule_tgz=artifacts_dir / "run_capsule.tar.gz",
        saguaro_verify_json=root / "saguaro_verify.json",
        telemetry_event_store_db=telemetry_dir / "event_store.db",
        telemetry_event_store_export_json=telemetry_dir / "event_store_export.json",
        black_box_manifest_json=telemetry_dir / "black_box_manifest.json",
        native_dir=native_dir,
        kernel_dir=kernel_dir,
        continuous_dir=continuous_dir,
        eval_dir=eval_dir,
        telemetry_dir=telemetry_dir,
        comparison_dir=comparison_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        native_attempts_ndjson=native_dir / "attempts.ndjson",
        native_phases_ndjson=native_dir / "phases.ndjson",
        native_failures_ndjson=native_dir / "failures.ndjson",
        kernel_attempts_ndjson=kernel_dir / "attempts.ndjson",
        kernel_summary_json=kernel_dir / "summary.json",
        quality_attempts_ndjson=eval_dir / "quality_attempts.ndjson",
        quality_summary_json=eval_dir / "quality_summary.json",
        memory_replay_summary_json=eval_dir / "memory_replay_summary.json",
        comparisons_json=comparison_dir / "comparisons.json",
    )


def ensure_suite_layout(layout: SuiteRunLayout) -> None:
    for path in (
        layout.root,
        layout.native_dir,
        layout.kernel_dir,
        layout.continuous_dir,
        layout.eval_dir,
        layout.telemetry_dir,
        layout.comparison_dir,
        layout.reports_dir,
        layout.artifacts_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def required_suite_artifacts(layout: SuiteRunLayout) -> list[Path]:
    return [
        layout.events_ndjson,
        layout.terminal_transcript_log,
        layout.manifest_json,
        layout.suite_resolved_json,
        layout.preflight_json,
        layout.environment_json,
        layout.checkpoint_json,
        layout.suite_status_json,
        layout.assurance_plan_json,
        layout.closure_matrix_json,
        layout.change_manifest_json,
        layout.traceability_json,
        layout.evidence_bundle_json,
        layout.runtime_gates_json,
        layout.telemetry_contract_json,
        layout.chronicle_json,
        layout.summary_json,
        layout.summary_md,
        layout.report_html,
        layout.triage_json,
        layout.console_log,
        layout.report_md,
        layout.executive_summary_md,
        layout.index_json,
        layout.agent_handoff_json,
        layout.metrics_rollup_json,
        layout.run_ledger_json,
        layout.run_ledger_ndjson,
        layout.mission_receipts_json,
        layout.mission_graph_json,
        layout.topology_passport_json,
        layout.variance_budget_json,
        layout.baseline_lineage_json,
        layout.closure_result_json,
        layout.traceability_graph_json,
        layout.spc_report_json,
        layout.advisory_bundle_json,
        layout.run_capsule_manifest_json,
        layout.saguaro_verify_json,
        layout.telemetry_event_store_db,
        layout.telemetry_event_store_export_json,
        layout.black_box_manifest_json,
        layout.native_attempts_ndjson,
        layout.native_phases_ndjson,
        layout.native_failures_ndjson,
        layout.kernel_attempts_ndjson,
        layout.kernel_summary_json,
        layout.quality_attempts_ndjson,
        layout.quality_summary_json,
        layout.memory_replay_summary_json,
        layout.comparisons_json,
    ]
