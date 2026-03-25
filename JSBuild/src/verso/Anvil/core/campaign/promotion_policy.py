"""Promotion policies for shared experiment lanes."""

from __future__ import annotations

from typing import Any, Dict, List

from core.campaign.lane_interfaces import PromotionDecision


class PromotionPolicyEngine:
    """Applies keep/discard decisions from a scored experiment bundle."""

    DEFAULT_THRESHOLDS = {
        "experimental_eid": 0.0,
        "conservative_release": 0.5,
        "strict_audit": 1.0,
    }

    def evaluate(
        self,
        scorecard: Dict[str, Any],
        telemetry_check: Dict[str, Any],
        *,
        policy: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        policy = dict(policy or {})
        policy_name = str(policy.get("name") or "experimental_eid")
        minimum_score = float(
            policy.get(
                "minimum_score",
                self.DEFAULT_THRESHOLDS.get(policy_name, 0.0),
            )
        )
        reasons: List[str] = []
        if not telemetry_check.get("contract_satisfied"):
            reasons.append("telemetry contract failed")
        if float(scorecard.get("determinism_penalty") or 0.0) > 0:
            reasons.append("determinism penalty applied")
        if float(scorecard.get("regression_penalty") or 0.0) > float(
            policy.get("max_regression_penalty", 0.0)
        ):
            reasons.append("regression penalty exceeded policy")
        if float(scorecard.get("score") or 0.0) < minimum_score:
            reasons.append("score below promotion threshold")

        verdict = "keep" if not reasons else "discard"
        decision = PromotionDecision(
            verdict=verdict,
            score=float(scorecard.get("score") or 0.0),
            reasons=reasons,
            metric_deltas=dict(scorecard.get("metric_deltas") or {}),
            policy_name=policy_name,
        )
        return {
            **decision.to_dict(),
            "minimum_score": minimum_score,
            "telemetry_contract_satisfied": bool(
                telemetry_check.get("contract_satisfied")
            ),
        }
