"""Ranks innovation hypotheses for the EID branch."""

from __future__ import annotations

from typing import Any


class InnovationRanker:
    """Scores proposed innovations by upside, risk, evidence quality, and fit."""

    BOOST_KEYWORDS = {
        "accuracy": 1.3,
        "performance": 1.2,
        "telemetry": 0.9,
        "reliability": 1.0,
        "maintainability": 0.8,
        "hardware": 1.1,
        "ux": 0.7,
        "simplify": 0.9,
    }

    def rank(self, hypotheses: list[dict[str, Any]] | Any) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for item in hypotheses:
            statement = str(item.get("statement") or "").lower()
            target_subsystems = [str(value).lower() for value in item.get("target_subsystems") or []]
            experiment_count = len(item.get("required_experiments") or [])
            source_basis = len(item.get("source_basis") or [])
            risk_text = str(item.get("risk") or "").lower()
            score = 2.0 + min(source_basis, 3) * 0.4 + min(experiment_count, 4) * 0.25
            breakdown = {
                "evidence_basis": min(source_basis, 3) * 0.4,
                "experiment_depth": min(experiment_count, 4) * 0.25,
                "keyword_boost": 0.0,
                "risk_penalty": 0.0,
                "subsystem_bonus": 0.0,
            }
            for keyword, boost in self.BOOST_KEYWORDS.items():
                if keyword in statement:
                    score += boost
                    breakdown["keyword_boost"] += boost
            if any(term in target_subsystems for term in {"runtime", "campaign_runtime", "artifacts", "telemetry"}):
                score += 0.8
                breakdown["subsystem_bonus"] += 0.8
            if "high" in risk_text or "complex" in risk_text:
                score -= 0.9
                breakdown["risk_penalty"] -= 0.9
            elif "medium" in risk_text:
                score -= 0.45
                breakdown["risk_penalty"] -= 0.45
            ranked.append(
                {
                    **item,
                    "innovation_score": round(score, 3),
                    "score_breakdown": breakdown,
                    "promotable": score >= 3.5,
                }
            )
        return sorted(
            ranked,
            key=lambda item: (
                -float(item["innovation_score"]),
                str(item.get("hypothesis_id") or ""),
            ),
        )
