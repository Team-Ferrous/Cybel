from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable

from audit.control_plane.mission import MissionContext
from audit.control_plane.nodes import TraceabilityEdge, TraceabilityNode
from audit.control_plane.reducers import now_iso

TRACEABILITY_GRAPH_SCHEMA_VERSION = "native_qsg_suite.traceability_graph.v1"
SAGUARO_VERIFY_SCHEMA_VERSION = "native_qsg_suite.saguaro_verify.v1"
SAGUARO_VERIFY_COMMAND = (
    "./venv/bin/saguaro verify . --engines native,ruff,semantic --format json"
)


def skipped_saguaro_verify_envelope(*, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SAGUARO_VERIFY_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "command": SAGUARO_VERIFY_COMMAND,
        "available": False,
        "status": "skipped",
        "returncode": -1,
        "summary": {
            "status": "skipped",
            "violation_count": 0,
        },
        "reason": reason,
    }


def capture_saguaro_verify_envelope(
    repo_root: Path,
    *,
    command_runner: Callable[..., Any] | None = None,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    runner = command_runner or subprocess.run
    command = SAGUARO_VERIFY_COMMAND.split()
    payload = {
        "schema_version": SAGUARO_VERIFY_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "command": SAGUARO_VERIFY_COMMAND,
        "available": False,
        "status": "unavailable",
        "returncode": -1,
        "summary": {},
    }
    binary = repo_root / "venv" / "bin" / "saguaro"
    if not binary.exists():
        payload["error"] = "missing ./venv/bin/saguaro"
        return payload
    try:
        completed = runner(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        payload["status"] = "timeout"
        payload["error"] = f"TimeoutExpired: {exc}"
        payload["summary"] = {
            "status": "timeout",
            "violation_count": 0,
        }
        return payload
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload
    payload["available"] = True
    payload["returncode"] = int(completed.returncode)
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if stderr:
        payload["stderr"] = stderr
    try:
        report = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        report = {"raw_stdout": stdout}
    summary = {
        "status": str(
            report.get("status") or ("ok" if completed.returncode == 0 else "error")
        ),
        "violation_count": int(
            len(report.get("violations") or [])
            if isinstance(report.get("violations"), list)
            else int(report.get("count", 0) or 0)
        ),
    }
    payload["status"] = summary["status"]
    payload["summary"] = summary
    if report:
        payload["report"] = report
    return payload


def build_traceability_graph(
    mission: MissionContext,
    *,
    topology_passport: dict[str, Any],
    variance_budget: dict[str, Any],
    baseline_lineage: dict[str, Any],
    run_ledger: dict[str, Any],
    closure_result: dict[str, Any],
    saguaro_verify: dict[str, Any],
) -> dict[str, Any]:
    nodes = [
        TraceabilityNode(
            node_id="intent.manifest",
            question="intended",
            label="Benchmark mission intent",
            artifact_ref="manifest.json",
            metadata={
                "profile_name": mission.spec.profile_name,
                "models": list(mission.spec.models),
            },
        ),
        TraceabilityNode(
            node_id="happened.ledger",
            question="happened",
            label="Run ledger",
            artifact_ref="run_ledger.json",
            metadata={"event_count": len(run_ledger.get("events") or [])},
        ),
        TraceabilityNode(
            node_id="observed.topology",
            question="observed",
            label="Topology passport",
            artifact_ref="topology_passport.json",
            metadata={"topology_hash": topology_passport.get("topology_hash")},
        ),
        TraceabilityNode(
            node_id="concluded.variance",
            question="concluded",
            label="Variance budget",
            artifact_ref="variance_budget.json",
            metadata={
                "within_budget": (variance_budget.get("overall") or {}).get(
                    "within_budget"
                )
            },
        ),
        TraceabilityNode(
            node_id="concluded.lineage",
            question="concluded",
            label="Baseline lineage",
            artifact_ref="baseline_lineage.json",
            metadata={"comparable": baseline_lineage.get("comparable")},
        ),
        TraceabilityNode(
            node_id="observed.prompt_contract",
            question="observed",
            label="Prompt contract lineage",
            artifact_ref="manifest.json",
            metadata={
                "prompt_hash": mission.manifest.get("prompt_hash"),
                "prompt_contract_hash": mission.manifest.get("prompt_contract_hash"),
            },
        ),
        TraceabilityNode(
            node_id="observed.memory_snapshot",
            question="observed",
            label="Memory snapshot lineage",
            artifact_ref="manifest.json",
            metadata={
                "memory_snapshot_hash": mission.manifest.get("memory_snapshot_hash"),
                "feature_toggle_hash": mission.manifest.get("feature_toggle_hash"),
            },
        ),
        TraceabilityNode(
            node_id="concluded.verify",
            question="concluded",
            label="Saguaro verify envelope",
            artifact_ref="saguaro_verify.json",
            metadata={"status": saguaro_verify.get("status")},
        ),
        TraceabilityNode(
            node_id="unresolved.closure",
            question="unresolved",
            label="Closure result",
            artifact_ref="closure_result.json",
            metadata={"unresolved_count": len(closure_result.get("unresolved") or [])},
        ),
    ]
    edges = [
        TraceabilityEdge(
            edge_id="edge.intent_to_happened",
            src="intent.manifest",
            dst="happened.ledger",
            relation="executed_as",
            evidence=["manifest.json", "run_ledger.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.happened_to_observed",
            src="happened.ledger",
            dst="observed.topology",
            relation="observed_under",
            evidence=["events.ndjson", "topology_passport.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.observed_to_variance",
            src="observed.topology",
            dst="concluded.variance",
            relation="stability_assessed_by",
            evidence=["summary.json", "variance_budget.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.observed_to_lineage",
            src="observed.topology",
            dst="concluded.lineage",
            relation="compared_against",
            evidence=["comparisons.json", "baseline_lineage.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.intent_to_prompt",
            src="intent.manifest",
            dst="observed.prompt_contract",
            relation="parameterized_by",
            evidence=["manifest.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.intent_to_memory",
            src="intent.manifest",
            dst="observed.memory_snapshot",
            relation="bounded_by",
            evidence=["manifest.json"],
        ),
        TraceabilityEdge(
            edge_id="edge.verify_to_closure",
            src="concluded.verify",
            dst="unresolved.closure",
            relation="constrains",
            evidence=["saguaro_verify.json", "closure_result.json"],
        ),
    ]
    question_counts = {
        question: sum(1 for node in nodes if node.question == question)
        for question in ("intended", "happened", "observed", "concluded", "unresolved")
    }
    return {
        "schema_version": TRACEABILITY_GRAPH_SCHEMA_VERSION,
        "run_id": mission.run_id,
        "generated_at": now_iso(),
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
        "question_counts": question_counts,
    }
