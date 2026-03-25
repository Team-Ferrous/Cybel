"""Hypothesis generation and registry."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from core.campaign.state_store import CampaignStateStore
from core.memory.fabric import MemoryEdge, MemoryFabricStore, MemoryProjector


class HypothesisLab:
    """Generates and stores testable hypotheses from evidence gaps."""

    def __init__(self, campaign_id: str, state_store: CampaignStateStore):
        self.campaign_id = campaign_id
        self.state_store = state_store
        self.memory_fabric = MemoryFabricStore(state_store)
        self.memory_projector = MemoryProjector()

    def generate(
        self,
        objective: str,
        evidence: list[dict[str, Any]] | Any,
    ) -> list[dict[str, Any]]:
        evidence_list = list(evidence)
        topic_groups: dict[str, list[dict[str, Any]]] = {}
        topic_basis: list[str] = []
        counterexample_refs = [
            str(item.get("claim_id") or "")
            for item in evidence_list
            if item.get("claim_id") and float(item.get("confidence") or 0.0) < 0.6
        ]
        for item in evidence_list:
            topic = str(item.get("topic") or item.get("name") or "").strip()
            if not topic:
                continue
            if topic not in topic_basis:
                topic_basis.append(topic)
            topic_groups.setdefault(topic, []).append(item)

        hypotheses = [
            self._build_hypothesis(
                objective=objective,
                topic="campaign_runtime",
                supporting=evidence_list[:4],
                statement=(
                    f"Building explicit autonomy control-plane artifacts will improve "
                    f"reproducibility for '{objective}'."
                ),
                motivation="Roadmap requires deterministic pause/resume and typed artifacts.",
                target_subsystems=["campaign_runtime", "artifacts"],
                required_experiments=["artifact_resume_replay"],
                risk="medium implementation complexity",
                diversity_bucket="control_plane",
                counterexample_refs=counterexample_refs,
            )
        ]
        for topic in topic_basis[:4]:
            lowered = topic.lower()
            supporting = topic_groups.get(topic, [])
            if "telemetry" in lowered:
                hypotheses.append(
                    self._build_hypothesis(
                        objective=objective,
                        topic=topic,
                        supporting=supporting,
                        statement="A contract-driven telemetry layer will reduce audit churn and missing measurements.",
                        motivation="Evidence references observability and telemetry obligations.",
                        target_subsystems=["telemetry", "audit_engine"],
                        required_experiments=["telemetry_contract_replay"],
                        risk="low implementation complexity",
                        diversity_bucket="observability",
                        counterexample_refs=counterexample_refs,
                    )
                )
            elif "analysis" in lowered or "repo" in lowered or "dossier" in lowered:
                hypotheses.append(
                    self._build_hypothesis(
                        objective=objective,
                        topic=topic,
                        supporting=supporting,
                        statement="Pre-computed repo analysis packs can reduce campaign planning latency without reducing evidence fidelity.",
                        motivation="Repo evidence should be reusable and immutable across phases.",
                        target_subsystems=["repo_cache", "research_store"],
                        required_experiments=["analysis_pack_reuse_benchmark"],
                        risk="medium cache invalidation complexity",
                        diversity_bucket="repo_intelligence",
                        counterexample_refs=counterexample_refs,
                    )
                )
            elif "determin" in lowered or "replay" in lowered:
                hypotheses.append(
                    self._build_hypothesis(
                        objective=objective,
                        topic=topic,
                        supporting=supporting,
                        statement="Replay-bound prompt and task identities will reduce semantic drift during promotion.",
                        motivation="Determinism evidence suggests replayability should remain a first-class contract.",
                        target_subsystems=["campaign_runtime", "prompts"],
                        required_experiments=["artifact_resume_replay"],
                        risk="low implementation complexity",
                        diversity_bucket="determinism",
                        counterexample_refs=counterexample_refs,
                    )
                )
        seen_statements: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for item in hypotheses:
            statement = str(item.get("statement") or "")
            if statement in seen_statements:
                continue
            seen_statements.add(statement)
            deduped.append(item)
        self._persist(deduped)
        return deduped

    def _build_hypothesis(
        self,
        *,
        objective: str,
        topic: str,
        supporting: list[dict[str, Any]],
        statement: str,
        motivation: str,
        target_subsystems: list[str],
        required_experiments: list[str],
        risk: str,
        diversity_bucket: str,
        counterexample_refs: list[str],
    ) -> dict[str, Any]:
        supporting_claim_ids = [
            str(item.get("claim_id") or "")
            for item in supporting
            if item.get("claim_id")
        ]
        confidence_values = [float(item.get("confidence") or 0.0) for item in supporting]
        applicability_values = [
            float(item.get("applicability_score") or item.get("confidence") or 0.0)
            for item in supporting
        ]
        evidence_coverage_seed = round(
            min(1.0, len(supporting_claim_ids) / 3.0) * 0.55
            + (sum(confidence_values) / max(1, len(confidence_values))) * 0.45,
            3,
        )
        execution_cost_estimate = round(
            1.0 + len(required_experiments) * 0.45 + len(target_subsystems) * 0.2,
            3,
        )
        return {
            "hypothesis_id": str(uuid.uuid4()),
            "statement": statement,
            "motivation": motivation,
            "source_basis": sorted(
                {
                    str(item.get("topic") or topic)
                    for item in supporting
                    if str(item.get("topic") or topic).strip()
                }
            ),
            "target_subsystems": target_subsystems,
            "expected_upside": f"Improves {objective.lower() or 'campaign execution'} through {topic}.",
            "risk": risk,
            "required_experiments": required_experiments,
            "status": "proposed",
            "supersedes": [],
            "supporting_claim_ids": supporting_claim_ids,
            "evidence_refs": supporting_claim_ids,
            "counterexample_refs": counterexample_refs[:3],
            "evidence_coverage_seed": evidence_coverage_seed,
            "execution_cost_estimate": execution_cost_estimate,
            "diversity_bucket": diversity_bucket,
            "applicability_score": round(
                sum(applicability_values) / max(1, len(applicability_values)),
                3,
            ),
        }

    def _persist(self, hypotheses: list[dict[str, Any]]) -> None:
        for item in hypotheses:
            now = time.time()
            self.state_store.execute(
                """
                INSERT INTO hypotheses (
                    campaign_id, hypothesis_id, statement, status, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.campaign_id,
                    item["hypothesis_id"],
                    item["statement"],
                    item["status"],
                    json.dumps(item, default=str),
                    now,
                    now,
                ),
            )
            memory = self.memory_fabric.create_memory(
                memory_kind="hypothesis",
                payload_json=item,
                campaign_id=self.campaign_id,
                workspace_id=self.campaign_id,
                source_system="hypothesis_lab",
                summary_text=str(item.get("statement") or ""),
                provenance_json={
                    "source_table": "hypotheses",
                    "hypothesis_id": item["hypothesis_id"],
                    "supporting_claim_ids": item.get("supporting_claim_ids") or [],
                },
                hypothesis_id=str(item["hypothesis_id"]),
                lifecycle_state=str(item.get("status") or "proposed"),
                importance_score=0.8,
                confidence_score=0.55,
            )
            self.memory_fabric.register_alias(
                memory.memory_id,
                "hypotheses",
                str(item["hypothesis_id"]),
                campaign_id=self.campaign_id,
            )
            self.memory_projector.project_memory(
                self.memory_fabric,
                memory,
                include_multivector=True,
            )
            for claim_id in item.get("supporting_claim_ids") or []:
                claim_memory_id = self.memory_fabric.resolve_alias(
                    campaign_id=self.campaign_id,
                    source_table="research_claims",
                    source_id=str(claim_id),
                )
                if claim_memory_id:
                    self.memory_fabric.add_edge(
                        MemoryEdge(
                            src_memory_id=memory.memory_id,
                            dst_memory_id=claim_memory_id,
                            edge_type="supports",
                            evidence_json={"claim_id": claim_id},
                        )
                    )
