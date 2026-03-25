from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit.runner.suite_profiles import BenchmarkSuiteSpec
from audit.store.suite_layout import SuiteRunLayout

MISSION_GRAPH_SCHEMA_VERSION = "native_qsg_suite.mission_graph.v1"


@dataclass(frozen=True)
class MissionContext:
    repo_root: Path
    layout: SuiteRunLayout
    spec: BenchmarkSuiteSpec
    run_id: str
    manifest: dict[str, Any]
    launch_runtime: dict[str, Any]
    preflight_payload: dict[str, Any]
    summary: dict[str, Any]
    comparisons: dict[str, Any]
    assurance_plan: dict[str, Any] | None
    completed_lanes: tuple[str, ...]


@dataclass(frozen=True)
class MissionNode:
    node_id: str
    phase: str
    kind: str
    enabled: bool
    blocking: bool
    artifact: str
    depends_on: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "phase": self.phase,
            "kind": self.kind,
            "enabled": self.enabled,
            "blocking": self.blocking,
            "artifact": self.artifact,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class MissionNodeReceipt:
    node_id: str
    phase: str
    kind: str
    status: str
    blocking: bool
    lane: str | None = None
    attempt_id: str | None = None
    model: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "phase": self.phase,
            "kind": self.kind,
            "status": self.status,
            "blocking": self.blocking,
            "lane": str(self.lane or ""),
            "attempt_id": str(self.attempt_id or ""),
            "model": str(self.model or ""),
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True)
class BenchmarkMissionGraph:
    schema_version: str
    profile_name: str
    compare_to: str
    variance_report_only: bool
    closure_advisory: bool
    capsule_enabled: bool
    required_runtime_gates: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    nodes: tuple[MissionNode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_name": self.profile_name,
            "compare_to": self.compare_to,
            "variance_report_only": self.variance_report_only,
            "closure_advisory": self.closure_advisory,
            "capsule_enabled": self.capsule_enabled,
            "required_runtime_gates": list(self.required_runtime_gates),
            "required_artifacts": list(self.required_artifacts),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [
                {"src": node_id, "dst": item.node_id}
                for item in self.nodes
                for node_id in item.depends_on
            ],
        }


def compile_benchmark_mission_graph(
    spec: BenchmarkSuiteSpec,
    *,
    assurance_plan: dict[str, Any] | None,
    compare_to: str,
    variance_report_only: bool,
    closure_advisory: bool,
    capsule_enabled: bool,
) -> BenchmarkMissionGraph:
    enabled_lanes = set(spec.enabled_lanes)
    nodes = (
        MissionNode(
            node_id="preflight",
            phase="preflight",
            kind="host_contract",
            enabled=True,
            blocking=True,
            artifact="preflight.json",
        ),
        MissionNode(
            node_id="calibration",
            phase="calibration",
            kind="tuning",
            enabled=str(spec.tuning_contract_policy or "") == "generate",
            blocking=False,
            artifact="telemetry/calibration.json",
            depends_on=("preflight",),
        ),
        MissionNode(
            node_id="canonical_all_on",
            phase="native_surface",
            kind="lane",
            enabled="canonical_all_on" in enabled_lanes,
            blocking=True,
            artifact="native/attempts.ndjson",
            depends_on=("preflight",),
        ),
        MissionNode(
            node_id="thread_matrix",
            phase="native_surface",
            kind="lane",
            enabled="thread_matrix" in enabled_lanes,
            blocking=False,
            artifact="native/phases.ndjson",
            depends_on=("canonical_all_on",),
        ),
        MissionNode(
            node_id="continuous_scheduler",
            phase="continuous",
            kind="lane",
            enabled="continuous_scheduler" in enabled_lanes,
            blocking=False,
            artifact="continuous",
            depends_on=("canonical_all_on",),
        ),
        MissionNode(
            node_id="kernel_microbench",
            phase="kernel_harness",
            kind="lane",
            enabled="kernel_microbench" in enabled_lanes,
            blocking=False,
            artifact="kernel/summary.json",
            depends_on=("canonical_all_on",),
        ),
        MissionNode(
            node_id="quality_eval",
            phase="quality_eval",
            kind="lane",
            enabled="quality_eval" in enabled_lanes,
            blocking=False,
            artifact="eval/quality_summary.json",
            depends_on=("canonical_all_on",),
        ),
        MissionNode(
            node_id="memory_replay",
            phase="memory_replay",
            kind="lane",
            enabled="memory_replay" in enabled_lanes,
            blocking=False,
            artifact="eval/memory_replay_summary.json",
            depends_on=("quality_eval",),
        ),
        MissionNode(
            node_id="assurance",
            phase="assurance",
            kind="governance",
            enabled=True,
            blocking=not closure_advisory,
            artifact="runtime_gates.json",
            depends_on=("canonical_all_on", "quality_eval"),
        ),
        MissionNode(
            node_id="control_plane",
            phase="control_plane",
            kind="reducer",
            enabled=True,
            blocking=True,
            artifact="mission_graph.json",
            depends_on=("assurance",),
        ),
        MissionNode(
            node_id="persistence",
            phase="persistence",
            kind="artifact_bundle",
            enabled=True,
            blocking=True,
            artifact="summary.json",
            depends_on=("control_plane",),
        ),
    )
    return BenchmarkMissionGraph(
        schema_version=MISSION_GRAPH_SCHEMA_VERSION,
        profile_name=spec.profile_name,
        compare_to=compare_to,
        variance_report_only=variance_report_only,
        closure_advisory=closure_advisory,
        capsule_enabled=capsule_enabled,
        required_runtime_gates=tuple(
            (assurance_plan or {}).get("required_runtime_gates") or []
        ),
        required_artifacts=tuple(
            (assurance_plan or {}).get("required_artifacts") or []
        ),
        nodes=nodes,
    )


def mission_node_by_id(
    graph: BenchmarkMissionGraph, node_id: str
) -> MissionNode | None:
    for node in graph.nodes:
        if node.node_id == node_id:
            return node
    return None


def dry_run_receipt(graph: BenchmarkMissionGraph, node_id: str) -> dict[str, Any]:
    node = mission_node_by_id(graph, node_id)
    if node is None:
        raise KeyError(f"Unknown mission node: {node_id}")
    return MissionNodeReceipt(
        node_id=node.node_id,
        phase=node.phase,
        kind=node.kind,
        status="dry_run",
        blocking=node.blocking,
        lane=node.node_id if node.kind == "lane" else None,
        details={"artifact": node.artifact, "depends_on": list(node.depends_on)},
    ).to_dict()


def compile_mission_graph(
    spec: BenchmarkSuiteSpec,
    *,
    assurance_plan: dict[str, Any] | None,
    compare_to: str,
    variance_report_only: bool,
    closure_advisory: bool,
    capsule_enabled: bool,
) -> dict[str, Any]:
    return compile_benchmark_mission_graph(
        spec,
        assurance_plan=assurance_plan,
        compare_to=compare_to,
        variance_report_only=variance_report_only,
        closure_advisory=closure_advisory,
        capsule_enabled=capsule_enabled,
    ).to_dict()
