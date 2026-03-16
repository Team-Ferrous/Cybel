"""Specialist scheduling for the EID R&D organization."""

from __future__ import annotations

import uuid
from typing import Any


class EIDScheduler:
    """Maps ranked hypotheses onto deterministic specialist packets."""

    SPECIALIST_ROLES = (
        "hypothesis_generator",
        "counterfactual_strategist",
        "math_analyst",
        "software_architecture",
        "hardware_optimization",
        "telemetry_systems",
        "determinism_compliance",
        "market_analysis",
    )

    def schedule(
        self,
        objective: str,
        ranked_hypotheses: list[dict[str, Any]] | Any,
        *,
        repo_dossiers: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        objective_text = objective.lower()
        hypotheses = list(ranked_hypotheses or [])
        packets: list[dict[str, Any]] = []
        for index, hypothesis in enumerate(hypotheses[:4], start=1):
            roles = self._roles_for_hypothesis(hypothesis, objective_text)
            for role in roles:
                packets.append(
                    {
                        "assignment_id": f"eid_assign_{uuid.uuid4().hex[:12]}",
                        "specialist_role": role,
                        "hypothesis_id": hypothesis.get("hypothesis_id"),
                        "priority": round(float(hypothesis.get("innovation_score") or 0.0) - index * 0.01, 3),
                        "objective": objective,
                        "focus": hypothesis.get("statement", ""),
                        "source_basis": list(hypothesis.get("source_basis") or []),
                        "repo_dossier_count": len(repo_dossiers or []),
                        "expected_outputs": [
                            "hypothesis_statement",
                            "required_experiments",
                            "kill_criteria",
                            "fallback_path",
                        ],
                    }
                )
        return packets

    def build_proposals(
        self,
        objective: str,
        ranked_hypotheses: list[dict[str, Any]] | Any,
        specialist_packets: list[dict[str, Any]] | Any,
        *,
        max_tracks: int = 2,
    ) -> list[dict[str, Any]]:
        packets_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
        for packet in specialist_packets:
            packets_by_hypothesis.setdefault(str(packet.get("hypothesis_id") or ""), []).append(packet)

        proposals: list[dict[str, Any]] = []
        accepted_bids = self._accepted_hypotheses(ranked_hypotheses, max_tracks=max_tracks)
        for hypothesis in ranked_hypotheses:
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            proposal_packets = packets_by_hypothesis.get(hypothesis_id, [])
            roles = sorted(
                {
                    str(packet.get("specialist_role") or "")
                    for packet in proposal_packets
                    if packet.get("specialist_role")
                }
            )
            proposals.append(
                {
                    "proposal_id": f"eid_prop_{uuid.uuid4().hex[:12]}",
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_statement": hypothesis.get("statement", ""),
                    "source_basis": list(hypothesis.get("source_basis") or []),
                    "novelty_claim": self._novelty_claim(hypothesis),
                    "expected_upside": hypothesis.get("expected_upside", ""),
                    "risk": hypothesis.get("risk", ""),
                    "required_experiments": list(hypothesis.get("required_experiments") or []),
                    "kill_criteria": self._kill_criteria(hypothesis),
                    "fallback_path": self._fallback_path(hypothesis),
                    "specialist_roles": roles,
                    "telemetry_contract": {
                        "required_metrics": [
                            "wall_time_seconds",
                            "command_count",
                            "success_count",
                            "failure_count",
                            "determinism_pass",
                        ]
                    },
                    "blast_radius": {
                        "editable_scope": ["artifacts/experiments", "artifacts/telemetry"],
                        "read_only_scope": ["analysis_local", "analysis_external"],
                    },
                    "objective": objective,
                    "promotable": bool(hypothesis.get("promotable")),
                    "innovation_score": hypothesis.get("innovation_score"),
                    "evidence_coverage_score": float(hypothesis.get("evidence_coverage_score") or 0.0),
                    "counterexample_debt": float(hypothesis.get("counterexample_debt") or 0.0),
                    "execution_cost_estimate": float(hypothesis.get("execution_cost_estimate") or 0.0),
                    "portfolio_weight": float(hypothesis.get("portfolio_weight") or 0.0),
                    "diversity_bucket": str(hypothesis.get("diversity_bucket") or "general"),
                    "accepted_bid": hypothesis_id in accepted_bids,
                    "accepted_bid_count": len(accepted_bids),
                    "funding_rationale": self._funding_rationale(
                        hypothesis,
                        accepted=hypothesis_id in accepted_bids,
                    ),
                }
            )
        return proposals

    def _roles_for_hypothesis(self, hypothesis: dict[str, Any], objective_text: str) -> list[str]:
        statement = str(hypothesis.get("statement") or "").lower()
        roles = ["hypothesis_generator", "counterfactual_strategist", "determinism_compliance"]
        if "telemetry" in statement or "telemetry" in objective_text:
            roles.append("telemetry_systems")
        if "hardware" in statement or any(term in objective_text for term in {"cpu", "native", "simd", "openmp"}):
            roles.append("hardware_optimization")
        if any(term in statement for term in {"architecture", "artifact", "repo", "cache"}):
            roles.append("software_architecture")
        if "market" in statement:
            roles.append("market_analysis")
        return sorted(set(roles))

    @staticmethod
    def _novelty_claim(hypothesis: dict[str, Any]) -> str:
        basis = list(hypothesis.get("source_basis") or [])
        if basis:
            return f"Combines evidence from {', '.join(basis[:3])} into one bounded experimental track."
        return "Introduces a new bounded R&D track that was not previously captured in the campaign artifacts."

    @staticmethod
    def _kill_criteria(hypothesis: dict[str, Any]) -> list[str]:
        criteria = ["telemetry contract failure", "determinism regression"]
        if "high" in str(hypothesis.get("risk") or "").lower():
            criteria.append("risk exceeds bounded blast radius")
        return criteria

    @staticmethod
    def _fallback_path(hypothesis: dict[str, Any]) -> str:
        target_subsystems = ", ".join(hypothesis.get("target_subsystems") or [])
        if target_subsystems:
            return f"Revert to current implementation and document follow-up work for {target_subsystems}."
        return "Revert to the current roadmap path and keep the idea as a deferred hypothesis."

    @staticmethod
    def _accepted_hypotheses(
        ranked_hypotheses: list[dict[str, Any]] | Any,
        *,
        max_tracks: int,
    ) -> set[str]:
        accepted: set[str] = set()
        seen_buckets: set[str] = set()
        for hypothesis in ranked_hypotheses:
            if len(accepted) >= max(1, int(max_tracks)):
                break
            hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
            bucket = str(hypothesis.get("diversity_bucket") or "general")
            if hypothesis_id in accepted:
                continue
            if bucket in seen_buckets and len(accepted) + 1 < max(1, int(max_tracks)):
                continue
            if not bool(hypothesis.get("promotable")):
                continue
            accepted.add(hypothesis_id)
            seen_buckets.add(bucket)
        if not accepted:
            first = next(iter(ranked_hypotheses or []), None)
            if first is not None:
                accepted.add(str(first.get("hypothesis_id") or ""))
        return accepted

    @staticmethod
    def _funding_rationale(hypothesis: dict[str, Any], *, accepted: bool) -> str:
        if accepted:
            return (
                "Funded because its evidence coverage, portfolio weight, and diversity "
                "justify bounded execution."
            )
        if float(hypothesis.get("counterexample_debt") or 0.0) > 0.8:
            return "Deferred because counterexample debt is too high for the current execution budget."
        return "Deferred because higher-value or more diverse tracks exhausted the bounded execution budget."
