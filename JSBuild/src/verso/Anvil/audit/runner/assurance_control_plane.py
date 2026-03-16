from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jsonschema

from audit.runner import suite_certification
from audit.runner.suite_profiles import BenchmarkSuiteSpec
from audit.store.suite_layout import SuiteRunLayout
from audit.runtime_logging import get_active_logger
from core.aes.compliance_context import ComplianceContext
from core.aes.obligation_engine import ObligationEngine
from core.aes.rule_registry import AESRuleRegistry
from core.aes.runtime_gate_runner import RuntimeGateRunner


ASSURANCE_PLAN_SCHEMA_VERSION = "native_qsg_suite.assurance_plan.v1"
RUNTIME_GATE_REPORT_SCHEMA_VERSION = "native_qsg_suite.runtime_gates.v1"
PROFILE_HOT_PATHS = [
    "benchmarks/native_qsg_benchmark.py",
    "benchmarks/continuous_qsg_benchmark.py",
    "core/native/native_qsg_engine.py",
]
FAILURE_TAXONOMY = [
    {
        "failure_class": "policy_failure",
        "description": "A compiled AES rule or blocking obligation failed.",
    },
    {
        "failure_class": "structural_schema_failure",
        "description": "A required artifact was missing or schema-invalid.",
    },
    {
        "failure_class": "environment_preflight_failure",
        "description": "The host contract, preflight, or runtime baseline was not satisfied.",
    },
    {
        "failure_class": "runtime_transport_failure",
        "description": "A benchmark subprocess or runtime transport failed.",
    },
    {
        "failure_class": "performance_regression",
        "description": "A hot-path or throughput regression crossed policy thresholds.",
    },
    {
        "failure_class": "quality_coherence_failure",
        "description": "Output quality, coherence, or telemetry quality regressed.",
    },
    {
        "failure_class": "evidence_closure_failure",
        "description": "Required evidence, traceability, or receipts were incomplete.",
    },
    {
        "failure_class": "review_workflow_failure",
        "description": "Review or waiver governance was incomplete or stale.",
    },
]


class ClosureExecutor:
    """Map assurance obligations to explicit runtime gate and artifact closure state."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.runtime_gate_runner = RuntimeGateRunner(str(repo_root))

    def evaluate(
        self,
        *,
        context: ComplianceContext,
        plan: dict[str, Any],
        preflight_payload: dict[str, Any],
    ) -> dict[str, Any]:
        runtime_gate_summary = self.runtime_gate_runner.evaluate(
            context,
            list(plan.get("required_runtime_gates") or []),
            thresholds=dict(plan.get("thresholds") or {}),
        )
        artifact_states = [
            {
                "artifact": str(artifact),
                "state": (
                    "covered"
                    if not runtime_gate_summary.missing_artifacts
                    or str(artifact) not in runtime_gate_summary.missing_artifacts
                    else "missing"
                ),
            }
            for artifact in list(plan.get("required_artifacts") or [])
        ]
        return {
            "runtime_gate_summary": runtime_gate_summary,
            "artifact_states": artifact_states,
            "preflight_ok": bool(preflight_payload.get("ok", True)),
        }


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_receipt(
    repo_root: Path,
    schema_name: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    schema_path = repo_root / "standards" / "schemas" / schema_name
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=payload, schema=schema)
    return {
        "schema": schema_name,
        "schema_sha256": _sha256(schema_path),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "valid": True,
    }


def build_compliance_context(
    *,
    repo_root: Path,
    run_id: str,
    profile_path: Path,
    spec: BenchmarkSuiteSpec,
) -> ComplianceContext:
    changed_files: list[str] = []
    for rel_path in suite_certification.HARNESS_FILES:
        candidate = repo_root / rel_path
        if candidate.exists():
            changed_files.append(_repo_relative(repo_root, candidate))
    changed_files.append(_repo_relative(repo_root, profile_path))
    changed_files = list(dict.fromkeys(changed_files))
    hot_paths = [path for path in PROFILE_HOT_PATHS if path in changed_files]
    return ComplianceContext(
        run_id=run_id,
        aal=str(spec.assurance_level or "AAL-2").upper(),
        domains=[],
        changed_files=changed_files,
        hot_paths=hot_paths,
        public_api_changes=[],
        dependency_changes=[],
        trace_id=f"trace::{run_id}",
        evidence_bundle_id=f"evidence::{run_id}",
        waiver_ids=[],
        red_team_required=False,
    )


def compile_assurance_plan(
    *,
    repo_root: Path,
    run_id: str,
    profile_path: Path,
    spec: BenchmarkSuiteSpec,
) -> tuple[dict[str, Any], ComplianceContext]:
    context = build_compliance_context(
        repo_root=repo_root,
        run_id=run_id,
        profile_path=profile_path,
        spec=spec,
    )
    obligations = ObligationEngine().evaluate(context)
    registry = AESRuleRegistry()
    registry.load()

    rule_entries: list[dict[str, Any]] = []
    for rule_id in obligations.required_rule_ids:
        rule = registry.get_rule(rule_id)
        if rule is None:
            rule_entries.append(
                {
                    "rule_id": rule_id,
                    "status": "unowned",
                    "failure_class": "policy_failure",
                    "owning_verifier": "missing_rule_registry_entry",
                }
            )
            continue
        execution_mode = str(rule.execution_mode or "static")
        failure_class = {
            "static": "policy_failure",
            "artifact": "structural_schema_failure",
            "runtime_gate": "evidence_closure_failure",
            "workflow_gate": "review_workflow_failure",
            "manual": "review_workflow_failure",
        }.get(execution_mode, "policy_failure")
        owning_verifier = {
            "static": "saguaro_verify",
            "artifact": "schema_validation",
            "runtime_gate": "runtime_gate_runner",
            "workflow_gate": "suite_workflow",
            "manual": "manual_review",
        }.get(execution_mode, "suite_workflow")
        rule_entries.append(
            {
                "rule_id": rule.id,
                "title": rule.title,
                "severity": rule.severity,
                "status": rule.status,
                "execution_mode": execution_mode,
                "required_artifacts": list(rule.required_artifacts),
                "required_runtime_gates": [],
                "owning_verifier": owning_verifier,
                "closure_path": execution_mode,
                "failure_class": failure_class,
                "source_refs": list(rule.source_refs),
            }
        )

    thresholds_path = repo_root / "standards" / "aes" / "thresholds.yaml"
    obligations_path = repo_root / "standards" / "AES_OBLIGATIONS.json"
    rules_path = repo_root / "standards" / "AES_RULES.json"
    catalog_path = repo_root / "standards" / "aes" / "rule_catalog.yaml"
    plan = {
        "schema_version": ASSURANCE_PLAN_SCHEMA_VERSION,
        "assurance_model_version": "2026-03-10",
        "run_id": run_id,
        "profile_name": spec.profile_name,
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "assurance_level": context.aal,
        "evidence_class": str(spec.evidence_class or "exploratory"),
        "context": context.to_dict(),
        "matched_obligations": list(obligations.matched_obligations),
        "required_rule_ids": list(obligations.required_rule_ids),
        "required_runtime_gates": list(obligations.required_runtime_gates),
        "required_artifacts": list(obligations.required_artifacts),
        "thresholds": dict(obligations.thresholds),
        "failure_taxonomy": list(FAILURE_TAXONOMY),
        "closure_matrix": rule_entries,
        "policy_sources": {
            "rules": {
                "path": _repo_relative(repo_root, rules_path),
                "sha256": _sha256(rules_path),
            },
            "obligations": {
                "path": _repo_relative(repo_root, obligations_path),
                "sha256": _sha256(obligations_path),
            },
            "catalog": {
                "path": _repo_relative(repo_root, catalog_path),
                "sha256": _sha256(catalog_path),
            },
            "thresholds": {
                "path": _repo_relative(repo_root, thresholds_path),
                "sha256": _sha256(thresholds_path),
            },
        },
    }
    return plan, context


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def materialize_assurance_artifacts(
    *,
    repo_root: Path,
    layout: SuiteRunLayout,
    spec: BenchmarkSuiteSpec,
    context: ComplianceContext,
    plan: dict[str, Any],
    summary: dict[str, Any],
    preflight_payload: dict[str, Any],
) -> dict[str, Any]:
    compliance_dir = repo_root / ".anvil" / "compliance" / context.run_id
    compliance_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {
        "summary_json": _repo_relative(repo_root, layout.summary_json),
        "preflight_json": _repo_relative(repo_root, layout.preflight_json),
        "environment_json": _repo_relative(repo_root, layout.environment_json),
        "suite_resolved_json": _repo_relative(repo_root, layout.suite_resolved_json),
    }
    author = str(os.getenv("USER", "benchmark_suite") or "benchmark_suite")
    change_manifest = {
        "run_id": context.run_id,
        "changed_files": list(context.changed_files),
        "aal": context.aal,
        "domains": list(context.domains),
        "hot_paths": list(context.hot_paths),
        "public_api_changes": list(context.public_api_changes),
        "dependency_changes": list(context.dependency_changes),
        "required_rule_ids": list(plan.get("required_rule_ids") or []),
        "required_runtime_gates": list(plan.get("required_runtime_gates") or []),
        "trace_id": context.trace_id,
        "evidence_bundle_id": context.evidence_bundle_id,
    }
    chronicle = {
        "baseline_id": str(summary.get("baseline_run_id") or "baseline::none"),
        "result_id": context.run_id,
        "regression_percent": float(
            summary.get("decode_tps_delta_pct")
            or summary.get("regression_percent")
            or 0.0
        ),
    }
    telemetry_contract = {
        "contract_id": context.run_id,
        "required_fields": list(
            dict.fromkeys(
                list(
                    (
                        dict(plan.get("thresholds") or {})
                        .get("telemetry", {})
                        .get("required_fields", [])
                    )
                    or []
                )
            )
        ),
        "loggers": [
            "benchmark_suite",
            "attempt_executor",
            "continuous_benchmark",
        ],
        "profile_name": spec.profile_name,
    }
    traceability = {
        "trace_id": context.trace_id,
        "run_id": context.run_id,
        "requirement_id": (
            (list(plan.get("required_rule_ids") or [None])[0])
            or f"AES-RUN::{context.run_id}"
        ),
        "design_ref": "ASSURANCE_PIPELINE_HIGH_ASSURANCE_ROADMAP.md",
        "code_refs": list(context.changed_files),
        "test_refs": [
            "tests/audit/test_benchmark_suite.py",
            "tests/test_native_qsg_benchmark.py",
        ],
        "verification_refs": [
            f"python -m audit.runner.benchmark_suite --profile {spec.profile_name}",
            "pytest tests/audit/test_benchmark_suite.py -q",
            "pytest tests/test_native_qsg_benchmark.py -q",
        ],
        "aal": context.aal,
        "owner": author,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "changed_files": list(context.changed_files),
        "evidence_bundle_id": context.evidence_bundle_id,
    }
    evidence_bundle = {
        "bundle_id": context.evidence_bundle_id,
        "change_id": context.run_id,
        "trace_id": context.trace_id,
        "changed_files": list(context.changed_files),
        "aal": context.aal,
        "chronicle_snapshot": f"chronicle::{context.run_id}::snapshot",
        "chronicle_diff": f"chronicle::{context.run_id}::diff",
        "verification_report_path": _repo_relative(repo_root, layout.summary_json),
        "red_team_report_path": "runtime://not-required",
        "review_signoffs": [],
        "waivers": list(context.waiver_ids),
        "author": author,
        "evidence_class": str(spec.evidence_class or "exploratory"),
        "artifacts": artifact_hashes,
    }

    plan_receipts = [
        _schema_receipt(repo_root, "change_manifest.schema.json", change_manifest),
        _schema_receipt(repo_root, "traceability.schema.json", traceability),
        _schema_receipt(repo_root, "telemetry_contract.schema.json", telemetry_contract),
        _schema_receipt(repo_root, "chronicle_report.schema.json", chronicle),
        _schema_receipt(repo_root, "evidence_bundle.schema.json", evidence_bundle),
    ]

    for target in (
        layout.assurance_plan_json,
        compliance_dir / "assurance_plan.json",
    ):
        _write_json(target, plan)
    for target in (
        layout.closure_matrix_json,
        compliance_dir / "closure_matrix.json",
    ):
        _write_json(
            target,
            {
                "schema_version": ASSURANCE_PLAN_SCHEMA_VERSION,
                "run_id": context.run_id,
                "closure_matrix": list(plan.get("closure_matrix") or []),
            },
        )
    for target in (layout.change_manifest_json, compliance_dir / "change_manifest.json"):
        _write_json(target, change_manifest)
    for target in (layout.traceability_json, compliance_dir / "traceability.json"):
        _write_json(target, traceability)
    for target in (
        layout.telemetry_contract_json,
        compliance_dir / "telemetry_contract.json",
    ):
        _write_json(target, telemetry_contract)
    for target in (layout.chronicle_json, compliance_dir / "chronicle.json"):
        _write_json(target, chronicle)
    for target in (
        layout.evidence_bundle_json,
        compliance_dir / "evidence_bundle.json",
    ):
        _write_json(target, evidence_bundle)

    closure_executor = ClosureExecutor(repo_root)
    closure_state = closure_executor.evaluate(
        context=context,
        plan=plan,
        preflight_payload=preflight_payload,
    )
    runtime_gate_summary = closure_state["runtime_gate_summary"]
    runtime_gates = {
        "schema_version": RUNTIME_GATE_REPORT_SCHEMA_VERSION,
        "run_id": context.run_id,
        "trace_id": context.trace_id,
        "required_runtime_gates": list(plan.get("required_runtime_gates") or []),
        "missing_artifacts": list(runtime_gate_summary.missing_artifacts),
        "results": [asdict(item) for item in runtime_gate_summary.results],
        "artifact_states": list(closure_state.get("artifact_states") or []),
        "schema_validation_receipts": plan_receipts,
        "passed": bool(runtime_gate_summary.passed),
        "preflight_ok": bool(closure_state.get("preflight_ok", True)),
    }
    _schema_receipt(repo_root, "runtime_gate_report.schema.json", runtime_gates)
    for target in (layout.runtime_gates_json, compliance_dir / "runtime_gates.json"):
        _write_json(target, runtime_gates)
    logger = get_active_logger()
    if logger is not None:
        for result in runtime_gate_summary.results:
            logger.emit(
                level="info" if result.passed else ("debug" if result.status == "skipped" else "warn"),
                source="assurance_control_plane",
                event_type="runtime_gate_result",
                message=result.message,
                phase="assurance",
                lane="assurance",
                payload={
                    "gate_id": result.gate_id,
                    "status": result.status,
                    "passed": result.passed,
                    "missing_artifacts": list(result.missing_artifacts),
                    "required_artifacts": list(result.required_artifacts),
                    "skipped_reason": result.skipped_reason,
                },
            )
    return {
        "context": context.to_dict(),
        "plan": plan,
        "runtime_gates": runtime_gates,
    }
