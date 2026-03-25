"""Ranks EID hypotheses using novelty, evidence, and determinism fit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HypothesisExchangeBook:
    hypothesis_id: str
    evidence_coverage_score: float
    counterexample_debt: float
    execution_cost_estimate: float
    portfolio_weight: float
    diversity_bucket: str
    accepted_bid: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "evidence_coverage_score": round(self.evidence_coverage_score, 3),
            "counterexample_debt": round(self.counterexample_debt, 3),
            "execution_cost_estimate": round(self.execution_cost_estimate, 3),
            "portfolio_weight": round(self.portfolio_weight, 3),
            "diversity_bucket": self.diversity_bucket,
            "accepted_bid": self.accepted_bid,
        }


class HypothesisRanker:
    """Scores EID proposals for bounded autonomous experimentation."""

    NOVELTY_KEYWORDS = {
        "deterministic": 0.7,
        "telemetry": 0.6,
        "benchmark": 0.5,
        "hardware": 0.5,
        "simulation": 0.4,
        "architecture": 0.4,
        "autonomy": 0.4,
    }

    def rank(
        self,
        hypotheses: list[dict[str, Any]] | Any,
        *,
        objective: str = "",
    ) -> list[dict[str, Any]]:
        objective_text = objective.lower()
        ranked: list[dict[str, Any]] = []
        for item in hypotheses:
            statement = str(item.get("statement") or "")
            statement_lower = statement.lower()
            source_basis = list(item.get("source_basis") or [])
            experiments = list(item.get("required_experiments") or [])
            target_subsystems = [str(value).lower() for value in item.get("target_subsystems") or []]
            evidence_refs = list(item.get("evidence_refs") or item.get("supporting_claim_ids") or [])
            counterexample_refs = list(item.get("counterexample_refs") or [])
            novelty_score = 1.0 + min(len(source_basis), 4) * 0.22
            evidence_coverage_score = round(
                min(1.0, len(evidence_refs) / 4.0) * 0.55
                + min(1.0, len(source_basis) / 4.0) * 0.25
                + min(float(item.get("applicability_score") or 0.0), 1.0) * 0.2,
                3,
            )
            evidence_score = 1.0 + evidence_coverage_score * 2.1
            subsystem_bonus = 0.0
            risk_penalty = 0.0
            determinism_bonus = 0.4 if "determin" in statement_lower else 0.0
            objective_fit = 0.0
            counterexample_debt = round(
                min(1.5, len(counterexample_refs) * 0.35)
                + (0.25 if "counterexample" in statement_lower else 0.0),
                3,
            )
            execution_cost_estimate = round(
                float(item.get("execution_cost_estimate") or 0.8)
                + len(experiments) * 0.35
                + len(target_subsystems) * 0.08,
                3,
            )

            for keyword, bonus in self.NOVELTY_KEYWORDS.items():
                if keyword in statement_lower:
                    novelty_score += bonus
            if any(
                term in target_subsystems
                for term in {"campaign_runtime", "artifacts", "telemetry", "repo_cache"}
            ):
                subsystem_bonus += 0.7
            if any(term in objective_text for term in {"cpu", "native", "autonomy", "determin"}):
                objective_fit += 0.5
            risk_text = str(item.get("risk") or "").lower()
            if "high" in risk_text or "complex" in risk_text:
                risk_penalty += 0.9
            elif "medium" in risk_text:
                risk_penalty += 0.45

            portfolio_weight = round(
                max(
                    0.0,
                    evidence_coverage_score * 1.8
                    + novelty_score * 0.3
                    + subsystem_bonus * 0.4
                    + determinism_bonus * 0.5
                    + objective_fit * 0.35
                    - counterexample_debt * 0.7
                    - execution_cost_estimate * 0.15
                    - risk_penalty * 0.5,
                ),
                3,
            )
            score = (
                novelty_score
                + evidence_score
                + subsystem_bonus
                + determinism_bonus
                + objective_fit
                + portfolio_weight
                - counterexample_debt
                - risk_penalty
                - execution_cost_estimate * 0.08
            )
            exchange_book = HypothesisExchangeBook(
                hypothesis_id=str(item.get("hypothesis_id") or ""),
                evidence_coverage_score=evidence_coverage_score,
                counterexample_debt=counterexample_debt,
                execution_cost_estimate=execution_cost_estimate,
                portfolio_weight=portfolio_weight,
                diversity_bucket=str(item.get("diversity_bucket") or "general"),
            )
            ranked.append(
                {
                    **item,
                    "novelty_score": round(novelty_score, 3),
                    "evidence_score": round(evidence_score, 3),
                    "evidence_coverage_score": evidence_coverage_score,
                    "counterexample_debt": counterexample_debt,
                    "execution_cost_estimate": execution_cost_estimate,
                    "portfolio_weight": portfolio_weight,
                    "determinism_bonus": round(determinism_bonus, 3),
                    "objective_fit": round(objective_fit, 3),
                    "innovation_score": round(score, 3),
                    "promotable": score >= 4.2 and evidence_coverage_score >= 0.25,
                    "exchange_book": exchange_book.as_dict(),
                    "score_breakdown": {
                        "novelty_score": round(novelty_score, 3),
                        "evidence_score": round(evidence_score, 3),
                        "evidence_coverage_score": evidence_coverage_score,
                        "subsystem_bonus": round(subsystem_bonus, 3),
                        "determinism_bonus": round(determinism_bonus, 3),
                        "objective_fit": round(objective_fit, 3),
                        "counterexample_debt": round(-counterexample_debt, 3),
                        "execution_cost_estimate": round(-execution_cost_estimate, 3),
                        "portfolio_weight": portfolio_weight,
                        "risk_penalty": round(-risk_penalty, 3),
                    },
                }
            )
        return sorted(
            ranked,
            key=lambda item: (-float(item.get("innovation_score") or 0.0), str(item.get("hypothesis_id") or "")),
        )
