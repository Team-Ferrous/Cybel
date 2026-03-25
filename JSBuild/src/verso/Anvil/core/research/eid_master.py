"""Experimental innovative design orchestration."""

from __future__ import annotations

from core.memory.fabric import MemoryFabricStore, MemoryProjector
from core.qsg.latent_bridge import QSGLatentBridge
from typing import Any

from core.research.eid_scheduler import EIDScheduler
from core.research.experiment_design import ExperimentDesignService
from core.research.hypothesis_ranker import HypothesisRanker
from core.research.simulator_planner import SimulatorPlanner


class EIDMasterLoop:
    """Second-order innovation branch for research and experiment ideation."""

    def __init__(self, *, experiment_runner=None, state_store=None, campaign_id: str = "") -> None:
        self.rank = HypothesisRanker()
        self.scheduler = EIDScheduler()
        self.design = ExperimentDesignService()
        self.simulator_planner = SimulatorPlanner()
        self.experiment_runner = experiment_runner
        self.state_store = state_store
        self.campaign_id = campaign_id
        self.memory_fabric = MemoryFabricStore(state_store) if state_store is not None else None
        self.memory_projector = MemoryProjector() if state_store is not None else None
        self.latent_bridge = (
            QSGLatentBridge(self.memory_fabric, self.memory_projector)
            if state_store is not None
            else None
        )

    def run(
        self,
        objective: str,
        hypotheses: list[dict[str, Any]] | Any,
        repo_dossiers: list[dict[str, Any]] | Any = None,
        *,
        execute_tracks: bool = False,
        max_tracks: int = 2,
        workspace_root: str = ".",
        metadata_path: str = "campaign.json",
    ) -> dict[str, Any]:
        ranked = self.rank.rank(hypotheses, objective=objective)
        dossier_list = list(repo_dossiers or [])
        specialist_packets = self.scheduler.schedule(
            objective, ranked, repo_dossiers=dossier_list
        )
        proposals = self.scheduler.build_proposals(
            objective,
            ranked,
            specialist_packets,
            max_tracks=max_tracks,
        )
        funded_proposals = [item for item in proposals if item.get("accepted_bid")]
        simulator_plans = self.simulator_planner.plan(
            objective,
            ranked,
            funded_proposals=funded_proposals,
        )
        ranked_experiments = self._rank_experiments(ranked)
        experimental_tracks = self.design.design(
            objective,
            funded_proposals or proposals,
            workspace_root=workspace_root,
            metadata_path=metadata_path,
        )
        lane_runs: list[dict[str, Any]] = []
        if execute_tracks and self.experiment_runner is not None:
            for track in experimental_tracks[: max(1, int(max_tracks))]:
                lane_runs.append(self.experiment_runner.run_lane(track))
        self._persist_specialists(specialist_packets)
        hardware_recommendations = self._hardware_fit_recommendations(objective, ranked)
        simplifications = self._feature_simplifications(ranked)
        dossier_actions = self._repo_dossier_actions(dossier_list)
        comparative_frontier = [
            packet
            for dossier in dossier_list
            for packet in list(dossier.get("frontier_packets") or [])
        ]
        comparative_frontier.sort(
            key=lambda item: (-float(item.get("priority") or 0.0), str(item.get("title") or ""))
        )
        exchange_book = [dict(item.get("exchange_book") or {}) for item in ranked]
        accepted_bid_count = len(funded_proposals)
        funded_ids = {
            str(item.get("hypothesis_id") or "")
            for item in funded_proposals
        }
        portfolio_summary = {
            "accepted_bid_count": accepted_bid_count,
            "funded_hypothesis_ids": sorted(funded_ids),
            "portfolio_diversity_index": len(
                {
                    str(item.get("diversity_bucket") or "general")
                    for item in funded_proposals
                }
            ),
            "mean_execution_cost_estimate": round(
                sum(float(item.get("execution_cost_estimate") or 0.0) for item in funded_proposals)
                / max(1, accepted_bid_count),
                3,
            ),
        }
        latent_capture = None
        if self.memory_fabric is not None and ranked:
            hypothesis_memory_ids = []
            for item in ranked[:3]:
                memory_id = self.memory_fabric.resolve_alias(
                    campaign_id=self.campaign_id,
                    source_table="hypotheses",
                    source_id=str(item.get("hypothesis_id") or ""),
                )
                if memory_id:
                    hypothesis_memory_ids.append(memory_id)
            branch_memory = self.memory_fabric.create_memory(
                memory_kind="latent_branch",
                payload_json={
                    "objective": objective,
                    "hypothesis_ids": [str(item.get("hypothesis_id") or "") for item in ranked[:3]],
                    "experimental_tracks": [track.get("name") for track in experimental_tracks[:3]],
                },
                campaign_id=self.campaign_id,
                workspace_id=self.campaign_id,
                source_system="eid_master",
                summary_text=f"EID latent branch for {objective}",
                lifecycle_state="ranked",
                importance_score=0.9,
                confidence_score=0.7,
            )
            self.memory_projector.project_memory(self.memory_fabric, branch_memory)
            latent_capture = self.latent_bridge.capture_summary_package(
                memory_id=branch_memory.memory_id,
                summary_text=branch_memory.summary_text,
                capture_stage="hypothesis_ranking",
                supporting_memory_ids=hypothesis_memory_ids,
                creation_reason="eid hypothesis ranking",
            )
        return {
            "innovation_hypotheses": ranked,
            "specialist_packets": specialist_packets,
            "eid_proposals": proposals,
            "funded_proposals": funded_proposals,
            "exchange_book": exchange_book,
            "portfolio_summary": portfolio_summary,
            "simulator_plans": simulator_plans,
            "ranked_experiments": ranked_experiments,
            "experimental_tracks": experimental_tracks,
            "lane_runs": lane_runs,
            "hardware_fit_recommendations": hardware_recommendations,
            "feature_simplifications": simplifications,
            "repo_dossier_actions": dossier_actions,
            "comparative_frontier": comparative_frontier[:5],
            "repo_dossiers": dossier_list,
            "latent_capture": latent_capture,
            "whitepaper_outline": {
                "sections": [
                    "Innovation candidates",
                    "Exchange book and funded bids",
                    "Specialist packets",
                    "Experimental tracks",
                    "Rejected alternatives",
                    "Hardware-fit implications",
                    "Measurement plan",
                    "Repo dossier integration",
                ],
                "top_promotions": [
                    item["hypothesis_id"]
                    for item in ranked[:3]
                    if item.get("promotable") and item["hypothesis_id"] in funded_ids
                ],
            },
        }

    def _persist_specialists(self, specialist_packets: list[dict[str, Any]]) -> None:
        if self.state_store is None or not self.campaign_id:
            return
        for packet in specialist_packets:
            self.state_store.insert_json_row(
                "specialist_assignments",
                campaign_id=self.campaign_id,
                payload=packet,
                id_field="assignment_id",
                id_value=str(packet.get("assignment_id") or ""),
                extra={"specialist_role": str(packet.get("specialist_role") or "")},
            )

    @staticmethod
    def _rank_experiments(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        experiments: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            for index, name in enumerate(
                hypothesis.get("required_experiments") or [], start=1
            ):
                experiments.append(
                    {
                        "experiment_name": name,
                        "hypothesis_id": hypothesis["hypothesis_id"],
                        "priority": round(
                            float(hypothesis["innovation_score"]) - index * 0.05, 3
                        ),
                        "success_metrics": [
                            "artifact_replayability",
                            "telemetry_contract_satisfied",
                        ],
                        "kill_criteria": [
                            "telemetry contract failure",
                            "determinism regression",
                        ],
                    }
                )
        return sorted(
            experiments, key=lambda item: (-item["priority"], item["experiment_name"])
        )

    @staticmethod
    def _hardware_fit_recommendations(
        objective: str,
        hypotheses: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        text = objective.lower()
        recommendations: list[dict[str, Any]] = []
        if any(term in text for term in {"simd", "cpu", "native", "openmp"}):
            recommendations.append(
                {
                    "title": "Promote benchmark-first native optimization",
                    "rationale": "Objective explicitly targets CPU/native optimization surfaces.",
                }
            )
        if any(
            "hardware" in str(item.get("statement", "")).lower() for item in hypotheses
        ):
            recommendations.append(
                {
                    "title": "Keep implementation gated on benchmark evidence",
                    "rationale": "Hypotheses indicate hardware-fit uncertainty that should remain measured.",
                }
            )
        return recommendations

    @staticmethod
    def _feature_simplifications(
        hypotheses: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        simplifications: list[dict[str, Any]] = [
            {
                "title": "Prefer canonical campaign artifacts over prompt reconstruction",
                "impact": "reduces context churn",
            }
        ]
        if any(
            "telemetry" in str(item.get("statement", "")).lower() for item in hypotheses
        ):
            simplifications.append(
                {
                    "title": "Collapse duplicate telemetry paths into one contract-driven layer",
                    "impact": "improves reproducibility and lowers instrumentation drift",
                }
            )
        return simplifications

    @staticmethod
    def _repo_dossier_actions(
        repo_dossiers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for dossier in repo_dossiers:
            comparative = dict(dossier.get("comparative_evidence") or {})
            phase_packets = list(
                dossier.get("phase_packets") or comparative.get("phase_packets") or []
            )
            programs = list(
                dossier.get("native_migration_programs")
                or comparative.get("native_migration_programs")
                or []
            )
            recipes = list(dossier.get("migration_recipes") or comparative.get("migration_recipes") or [])
            ledger = list(dossier.get("port_ledger") or comparative.get("port_ledger") or [])
            frontier = list(dossier.get("frontier_packets") or comparative.get("frontier_packets") or [])
            frontier.sort(
                key=lambda item: (
                    -float(item.get("priority") or 0.0),
                    str(item.get("title") or ""),
                )
            )
            if phase_packets:
                packet = phase_packets[0]
                actions.append(
                    {
                        "title": f"Promote phase packet {packet.get('phase_id')}",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": (
                            f"Comparative roadmap promotion prepared objective {packet.get('objective')}."
                        ),
                        "phase_id": packet.get("phase_id"),
                        "priority": float(
                            (packet.get("telemetry_contract") or {}).get(
                                "portfolio_rank_score",
                                0.0,
                            )
                        ),
                        "allowed_writes": list(packet.get("allowed_writes") or []),
                    }
                )
                continue
            if programs:
                program = programs[0]
                actions.append(
                    {
                        "title": f"Promote native migration program {program.get('feature_family')}",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": (
                            f"Comparative synthesis elevated {program.get('feature_family')} into a gated native program."
                        ),
                        "posture": program.get("posture"),
                        "source_path": program.get("source_path"),
                        "target_path": program.get("target_path"),
                        "priority": float(program.get("priority") or 0.0),
                        "program_id": program.get("program_id"),
                    }
                )
                continue
            if recipes:
                recipe = recipes[0]
                matching_packet = next(
                    (
                        packet
                        for packet in frontier
                        if str(packet.get("source_path") or "") == str(recipe.get("source_path") or "")
                    ),
                    frontier[0] if frontier else {},
                )
                actions.append(
                    {
                        "title": recipe.get("title") or "Evaluate comparative migration recipe",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": (
                            f"Comparative analysis recommends posture {recipe.get('posture', 'unknown')} "
                            f"for {recipe.get('source_path', '<unknown>')}."
                        ),
                        "posture": recipe.get("posture"),
                        "source_path": recipe.get("source_path"),
                        "target_insertion_path": recipe.get("target_insertion_path"),
                        "frontier_packet_id": matching_packet.get("packet_id"),
                        "recommended_tracks": list(
                            matching_packet.get("recommended_tracks") or []
                        ),
                        "priority": float(matching_packet.get("priority") or 0.0),
                    }
                )
                continue
            if frontier:
                packet = frontier[0]
                actions.append(
                    {
                        "title": packet.get("title") or "Run comparative frontier packet",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": packet.get("rationale")
                        or "Comparative frontier packet surfaced a high-priority experiment track.",
                        "posture": packet.get("posture"),
                        "source_path": packet.get("source_path"),
                        "target_path": packet.get("target_path"),
                        "frontier_packet_id": packet.get("packet_id"),
                        "recommended_tracks": list(packet.get("recommended_tracks") or []),
                        "priority": float(packet.get("priority") or 0.0),
                    }
                )
                continue
            if ledger:
                top_relation = ledger[0]
                actions.append(
                    {
                        "title": f"Probe {top_relation.get('relation_type', 'relation')} candidate first",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": top_relation.get("rationale")
                        or "Comparative port ledger surfaced a high-confidence candidate.",
                        "posture": top_relation.get("posture"),
                        "source_path": top_relation.get("source_path"),
                        "target_path": top_relation.get("target_path"),
                    }
                )
                continue
            candidates = list(dossier.get("reuse_candidates") or [])
            if candidates:
                top = candidates[0]
                actions.append(
                    {
                        "title": f"Evaluate reuse candidate {top.get('path')} first",
                        "repo_id": dossier.get("repo_id"),
                        "rationale": (
                            "Repo dossier ranks this path highest by symbol/reference reuse score."
                        ),
                    }
                )
        return actions[:5]
