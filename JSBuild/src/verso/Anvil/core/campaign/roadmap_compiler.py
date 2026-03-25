"""Evidence-driven roadmap compilation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List

from core.campaign.phase_packet import PhasePacketBuilder
from core.campaign.roadmap_validator import RoadmapValidator
from core.campaign.task_graph import CampaignTaskGraph, TaskNode


@dataclass
class RoadmapItem:
    item_id: str
    phase_id: str
    title: str
    type: str
    repo_scope: List[str]
    owner_type: str
    depends_on: List[str] = field(default_factory=list)
    description: str = ""
    objective: str = ""
    success_metrics: List[str] = field(default_factory=list)
    required_evidence: List[str] = field(default_factory=list)
    required_artifacts: List[str] = field(default_factory=list)
    telemetry_contract: Dict[str, object] = field(default_factory=dict)
    allowed_writes: List[str] = field(default_factory=list)
    rollback_criteria: List[str] = field(default_factory=list)
    promotion_gate: Dict[str, object] = field(default_factory=dict)
    exit_gate: Dict[str, object] = field(default_factory=dict)
    status: str = "planned"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RoadmapPhaseDocument:
    phase_id: str
    name: str
    order: int
    tasks: List[Dict[str, object]] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    estimated_iterations: int = 1
    artifact_folder: str = ""


PHASE_SEQUENCE = [
    ("intake", "Intake & Configuration"),
    ("research", "Research Loop"),
    ("eid", "EID R&D Loop"),
    ("questionnaire", "Architecture Questionnaire"),
    ("feature_map", "Feature Map"),
    ("roadmap_draft", "Draft Roadmap"),
    ("development", "Development Loop"),
    ("analysis_upgrade", "Code Analysis & Upgrade"),
    ("deep_test_audit", "Deep Test & Audit"),
    ("convergence", "Convergence Audit"),
]

PHASE_ALIASES = {
    "architecture": "questionnaire",
    "artifact_store": "questionnaire",
    "questionnaire": "questionnaire",
    "feature_map": "feature_map",
    "development": "development",
    "research": "research",
    "eid": "eid",
    "roadmap_draft": "roadmap_draft",
}


class RoadmapCompiler:
    """Compile a typed roadmap graph from artifacted campaign inputs."""

    def __init__(self, state_store, campaign_id: str):
        self.state_store = state_store
        self.campaign_id = campaign_id

    def compile(
        self,
        *,
        features: Iterable[Dict[str, object]],
        questions: Iterable[Dict[str, object]],
        hypotheses: Iterable[Dict[str, object]] | None = None,
        repo_dossiers: Iterable[Dict[str, object]] | None = None,
        experiment_lanes: Iterable[Dict[str, object]] | None = None,
        research_clusters: Iterable[Dict[str, object]] | None = None,
        objective: str = "",
    ) -> List[RoadmapItem]:
        items: List[RoadmapItem] = []
        for index, feature in enumerate(features, start=1):
            selection_state = str(feature.get("selection_state") or "defer")
            if selection_state == "defer":
                continue
            feature_id = str(feature["feature_id"])
            depends_on = feature.get("depends_on")
            if depends_on is None and feature.get("depends_on_json"):
                depends_on = json.loads(feature["depends_on_json"])
            evidence_links = feature.get("evidence_links")
            if evidence_links is None and feature.get("evidence_links_json"):
                evidence_links = json.loads(feature["evidence_links_json"])
            items.append(
                RoadmapItem(
                    item_id=f"roadmap_{feature_id}",
                    phase_id=str(feature.get("category") or "development"),
                    title=str(feature["name"]),
                    type="feature",
                    repo_scope=["target"],
                    owner_type="ImplementationEngineerSubagent",
                    depends_on=list(depends_on or []),
                    description=str(feature.get("description") or ""),
                    objective=objective or str(feature.get("name") or ""),
                    success_metrics=[
                        f"selected_feature:{feature_id}",
                        "telemetry:present",
                    ],
                    required_evidence=list(evidence_links or []),
                    required_artifacts=["feature_map"],
                    telemetry_contract={
                        "minimum": ["wall_time", "artifact_emission_status"]
                    },
                    allowed_writes=["target"],
                    rollback_criteria=["feature regression detected", "tests failing"],
                    promotion_gate={"feature_selected": feature_id},
                    exit_gate={"feature_selected": feature_id},
                    metadata={"ordinal": index},
                )
            )

        unresolved = [
            question
            for question in questions
            if question.get("current_status") not in {"answered", "waived"}
            and question.get("blocking_level") in {"critical", "high"}
        ]
        if unresolved:
            items.append(
                RoadmapItem(
                    item_id="roadmap_blocking_questions",
                    phase_id="questionnaire",
                    title="Resolve blocking architecture questions",
                    type="decision",
                    repo_scope=["artifact_store"],
                    owner_type="ArchitectureAdjudicatorSubagent",
                    description="Blocking questions must be resolved before development promotion.",
                    objective=objective or "Resolve blocking architecture questions",
                    success_metrics=["blocking_questions:0"],
                    required_artifacts=["architecture"],
                    telemetry_contract={"minimum": ["artifact_emission_status"]},
                    allowed_writes=["artifact_store"],
                    rollback_criteria=["blocking questions remain unresolved"],
                    promotion_gate={"resolved_questions": len(unresolved)},
                    exit_gate={"resolved_questions": len(unresolved)},
                )
            )

        for index, hypothesis in enumerate(hypotheses or [], start=1):
            if not hypothesis.get("promotable"):
                continue
            hypothesis_id = str(hypothesis.get("hypothesis_id") or f"hypothesis_{index}")
            items.append(
                RoadmapItem(
                    item_id=f"roadmap_eid_{hypothesis_id}",
                    phase_id="eid",
                    title=f"Validate EID hypothesis {index}",
                    type="research_track",
                    repo_scope=["artifact_store", "target"],
                    owner_type="EIDMasterSubagent",
                    description=str(hypothesis.get("statement") or ""),
                    objective=objective or str(hypothesis.get("statement") or ""),
                    success_metrics=[
                        "telemetry_contract_satisfied",
                        "determinism_pass",
                    ],
                    required_evidence=list(hypothesis.get("source_basis") or []),
                    required_artifacts=["research", "experiments"],
                    telemetry_contract={
                        "minimum": [
                            "wall_time_seconds",
                            "telemetry_contract_satisfied",
                        ]
                    },
                    allowed_writes=["artifact_store"],
                    rollback_criteria=["kill criteria triggered", "promotion gate failed"],
                    promotion_gate={
                        "hypothesis_id": hypothesis_id,
                        "minimum_innovation_score": hypothesis.get("innovation_score", 0),
                    },
                    exit_gate={"promotable": True},
                    metadata={"rank": index, "kind": "eid_hypothesis"},
                )
            )

        for index, lane in enumerate(experiment_lanes or [], start=1):
            lane_id = str(lane.get("lane_id") or f"lane_{index}")
            items.append(
                RoadmapItem(
                    item_id=f"roadmap_lane_{lane_id}",
                    phase_id="development",
                    title=f"Run experiment lane {lane.get('name') or index}",
                    type="experiment_lane",
                    repo_scope=["target", "artifact_store"],
                    owner_type="ExperimentLane",
                    description=str(lane.get("description") or lane.get("objective_function") or ""),
                    objective=objective or str(lane.get("objective_function") or ""),
                    success_metrics=["telemetry_contract_satisfied", "correctness_pass"],
                    required_evidence=list(lane.get("read_only_scope") or []),
                    required_artifacts=["experiments", "telemetry"],
                    telemetry_contract=dict(lane.get("telemetry_contract") or {}),
                    allowed_writes=list(lane.get("allowed_writes") or ["target"]),
                    rollback_criteria=list(lane.get("rollback_criteria") or []),
                    promotion_gate=dict(lane.get("promotion_policy") or {}),
                    exit_gate={"lane_id": lane_id},
                    metadata={
                        "lane_type": lane.get("lane_type"),
                        "caller_mode": lane.get("caller_mode"),
                        "commands": list(lane.get("commands") or []),
                        "editable_scope": list(lane.get("editable_scope") or []),
                        "read_only_scope": list(lane.get("read_only_scope") or []),
                        "allowed_writes": list(lane.get("allowed_writes") or ["target"]),
                        "promotion_policy": dict(lane.get("promotion_policy") or {}),
                        "speculation_variants": list(
                            lane.get("speculation_variants") or []
                        ),
                    },
                )
            )

        for index, dossier in enumerate(repo_dossiers or [], start=1):
            comparative = dict(dossier.get("comparative_evidence") or {})
            recipes = list(
                dossier.get("migration_recipes")
                or comparative.get("migration_recipes")
                or []
            )
            frontier_packets = list(
                dossier.get("frontier_packets")
                or comparative.get("frontier_packets")
                or []
            )
            ledger = list(
                dossier.get("port_ledger")
                or comparative.get("port_ledger")
                or []
            )
            comparative_emitted = False
            for recipe_index, recipe in enumerate(recipes[:2], start=1):
                evidence = [
                    str(recipe.get("recipe_id") or ""),
                    str(recipe.get("source_path") or ""),
                    str(recipe.get("target_insertion_path") or ""),
                    *[
                        str(item)
                        for item in list(recipe.get("verification_requirements") or [])
                    ],
                ]
                items.append(
                    RoadmapItem(
                        item_id=f"roadmap_comparative_recipe_{index}_{recipe_index}",
                        phase_id="convergence",
                        title=str(
                            recipe.get("title")
                            or f"Review comparative recipe {recipe_index}"
                        ),
                        type="comparative_recipe",
                        repo_scope=["analysis_local", "artifact_store", "target"],
                        owner_type="ComparativeMigrationArchitectSubagent",
                        description=(
                            "Promote comparative migration evidence into an explicit "
                            "port, rewrite, or integration plan."
                        ),
                        objective=objective or "Promote comparative migration recipe",
                        success_metrics=[
                            "comparative_recipe_reviewed",
                            "comparative_evidence_closed",
                        ],
                        required_evidence=[item for item in evidence if item],
                        required_artifacts=["research", "comparative_reports"],
                        telemetry_contract={
                            "minimum": [
                                "artifact_emission_status",
                                "comparison_backend",
                            ]
                        },
                        allowed_writes=["target", "artifact_store"],
                        rollback_criteria=[
                            "comparative recipe disproven by verification",
                            "port posture rejected",
                        ],
                        promotion_gate={
                            "repo_id": dossier.get("repo_id"),
                            "posture": recipe.get("posture"),
                        },
                        exit_gate={"recipe_id": recipe.get("recipe_id")},
                        metadata={
                            "posture": recipe.get("posture"),
                            "relation_type": recipe.get("relation_type"),
                            "source_corpus_id": recipe.get("source_corpus_id"),
                            "target_corpus_id": recipe.get("target_corpus_id"),
                        },
                    )
                )
                comparative_emitted = True
            for packet_index, packet in enumerate(frontier_packets[:2], start=1):
                items.append(
                    RoadmapItem(
                        item_id=f"roadmap_comparative_frontier_{index}_{packet_index}",
                        phase_id="eid",
                        title=str(
                            packet.get("title")
                            or f"Run comparative frontier packet {packet_index}"
                        ),
                        type="comparative_frontier",
                        repo_scope=["analysis_local", "artifact_store", "target"],
                        owner_type="ComparativeFrontierPlannerSubagent",
                        description=(
                            "Convert comparative mechanism evidence into an experiment "
                            "or implementation track with explicit posture."
                        ),
                        objective=objective or "Advance comparative frontier evidence",
                        success_metrics=[
                            "comparative_frontier_packet_reviewed",
                            "experiment_track_acceptance_rate",
                        ],
                        required_evidence=[
                            item
                            for item in (
                                str(packet.get("packet_id") or ""),
                                str(packet.get("source_path") or ""),
                                str(packet.get("target_path") or ""),
                            )
                            if item
                        ],
                        required_artifacts=["research", "experiments"],
                        telemetry_contract={
                            "minimum": [
                                "artifact_emission_status",
                                "comparative_frontier_count",
                            ]
                        },
                        allowed_writes=["artifact_store"],
                        rollback_criteria=[
                            "frontier packet loses comparative support",
                            "experiment track rejected",
                        ],
                        promotion_gate={
                            "repo_id": dossier.get("repo_id"),
                            "posture": packet.get("posture"),
                            "minimum_priority": packet.get("priority", 0),
                        },
                        exit_gate={"packet_id": packet.get("packet_id")},
                        metadata={
                            "recommended_tracks": list(
                                packet.get("recommended_tracks") or []
                            ),
                            "priority": packet.get("priority"),
                            "posture": packet.get("posture"),
                        },
                    )
                )
                comparative_emitted = True
            if not recipes and ledger:
                top_relation = ledger[0]
                items.append(
                    RoadmapItem(
                        item_id=f"roadmap_port_ledger_{index}",
                        phase_id="analysis_upgrade",
                        title=(
                            f"Review {top_relation.get('relation_type', 'comparative')} "
                            f"candidate {top_relation.get('source_path')}"
                        ),
                        type="comparative_port_candidate",
                        repo_scope=["analysis_local", "artifact_store", "target"],
                        owner_type="ComparativeRepoAnalystSubagent",
                        description=(
                            "Review the highest-confidence comparative port candidate "
                            "before promotion into development."
                        ),
                        objective=objective or "Review comparative port candidate",
                        success_metrics=[
                            "port_candidate_reviewed",
                            "comparative_evidence_closed",
                        ],
                        required_evidence=[
                            item
                            for item in (
                                str(top_relation.get("source_path") or ""),
                                str(top_relation.get("target_path") or ""),
                                str(top_relation.get("relation_type") or ""),
                            )
                            if item
                        ],
                        required_artifacts=["research", "comparative_reports"],
                        telemetry_contract={
                            "minimum": [
                                "artifact_emission_status",
                                "relation_precision_audit_score",
                            ]
                        },
                        allowed_writes=["artifact_store"],
                        rollback_criteria=["candidate disproven by comparative audit"],
                        promotion_gate={
                            "repo_id": dossier.get("repo_id"),
                            "posture": top_relation.get("posture"),
                        },
                        exit_gate={
                            "source_path": top_relation.get("source_path"),
                            "target_path": top_relation.get("target_path"),
                        },
                        metadata={
                            "relation_score": top_relation.get("relation_score"),
                            "posture": top_relation.get("posture"),
                        },
                    )
                )
                comparative_emitted = True
            if comparative_emitted:
                continue
            top_candidate = next(iter(dossier.get("reuse_candidates") or []), {})
            if not top_candidate:
                continue
            items.append(
                RoadmapItem(
                    item_id=f"roadmap_repo_dossier_{index}",
                    phase_id="analysis_upgrade",
                    title=f"Analyze reuse candidate {top_candidate.get('path')}",
                    type="analysis",
                    repo_scope=["analysis_local"],
                    owner_type="RepoAnalystSubagent",
                    description="Promote the top repo dossier reuse candidate into the roadmap.",
                    objective=objective or "Leverage analysis repo evidence",
                    success_metrics=["reuse_candidate_reviewed"],
                    required_evidence=[str(top_candidate.get("path") or "")],
                    required_artifacts=["research"],
                    telemetry_contract={"minimum": ["artifact_emission_status"]},
                    allowed_writes=["artifact_store"],
                    rollback_criteria=["candidate disproven by analysis"],
                    promotion_gate={"repo_id": dossier.get("repo_id")},
                    exit_gate={"candidate_path": top_candidate.get("path")},
                )
            )

        for index, cluster in enumerate(research_clusters or [], start=1):
            label = str(cluster.get("label") or cluster.get("topic") or f"cluster_{index}")
            items.append(
                RoadmapItem(
                    item_id=f"roadmap_cluster_{index}",
                    phase_id="research",
                    title=f"Resolve research cluster {label}",
                    type="research",
                    repo_scope=["analysis_external", "artifact_store"],
                    owner_type="ResearchLoopSubagent",
                    description="Carry unresolved research clusters forward as first-class roadmap work.",
                    objective=objective or label,
                    success_metrics=["cluster_resolved"],
                    required_evidence=list(cluster.get("members") or []),
                    required_artifacts=["research"],
                    telemetry_contract={"minimum": ["artifact_emission_status"]},
                    allowed_writes=["artifact_store"],
                    rollback_criteria=["cluster remains under-evidenced"],
                    promotion_gate={"cluster_score": cluster.get("score", 0)},
                    exit_gate={"cluster_id": cluster.get("cluster_id")},
                )
            )

        for item in items:
            payload = asdict(item)
            payload["campaign_id"] = self.campaign_id
            self.state_store.record_roadmap_item(payload)
        return items

    @staticmethod
    def build_task_graph(items: Iterable[RoadmapItem]) -> CampaignTaskGraph:
        graph = CampaignTaskGraph()
        for item in items:
            graph.add_node(
                TaskNode(
                    item_id=item.item_id,
                    phase_id=item.phase_id,
                    title=item.title,
                    depends_on=item.depends_on,
                    metadata={"type": item.type, "owner_type": item.owner_type},
                )
            )
        return graph

    @staticmethod
    def build_phase_documents(
        items: Iterable[RoadmapItem],
    ) -> List[RoadmapPhaseDocument]:
        grouped: Dict[str, List[Dict[str, object]]] = {}
        for item in items:
            canonical_phase = PHASE_ALIASES.get(item.phase_id, item.phase_id)
            grouped.setdefault(canonical_phase, []).append(asdict(item))

        phase_documents: List[RoadmapPhaseDocument] = []
        for order, (phase_id, name) in enumerate(PHASE_SEQUENCE, start=1):
            tasks = grouped.get(phase_id, [])
            dependencies = [
                prev_phase
                for prev_order, (prev_phase, _) in enumerate(PHASE_SEQUENCE, start=1)
                if prev_order < order and grouped.get(prev_phase)
            ]
            success_criteria: List[str] = []
            for task in tasks:
                success_criteria.extend(
                    [
                        str(metric)
                        for metric in task.get("success_metrics", [])
                        if metric
                    ]
                )
            phase_documents.append(
                RoadmapPhaseDocument(
                    phase_id=phase_id,
                    name=name,
                    order=order,
                    tasks=tasks,
                    dependencies=dependencies[-2:],
                    success_criteria=sorted(set(success_criteria)),
                    estimated_iterations=max(1, len(tasks)),
                    artifact_folder=f"phases/{order:02d}_{phase_id}/",
                )
            )
        return phase_documents

    @staticmethod
    def render_phase_pack(items: Iterable[RoadmapItem]) -> Dict[str, str]:
        phase_documents = [
            asdict(phase)
            for phase in RoadmapCompiler.build_phase_documents(items)
        ]
        packets = PhasePacketBuilder.build(phase_documents)
        payloads: Dict[str, str] = {}
        for order, packet in enumerate(packets, start=1):
            payloads[f"phase_{order:02d}.json"] = json.dumps(
                packet,
                indent=2,
                default=str,
            )
        return payloads

    @staticmethod
    def validate(
        items: Iterable[RoadmapItem],
        *,
        objective: str = "",
    ) -> List[str]:
        item_payloads = [asdict(item) for item in items]
        phase_documents = [
            asdict(phase)
            for phase in RoadmapCompiler.build_phase_documents(items)
        ]
        packets = PhasePacketBuilder.build(phase_documents, objective=objective)
        return RoadmapValidator().validate(item_payloads, packets)
