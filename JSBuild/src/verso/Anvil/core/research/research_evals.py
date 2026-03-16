"""Evaluation harness for research and crawl behavior."""

from __future__ import annotations

from typing import Any


class ResearchEvaluationHarness:
    """Produces measurable eval summaries for research loops."""

    def evaluate(
        self,
        frontier_size: int,
        claims: list[dict[str, Any]] | Any,
    ) -> dict[str, Any]:
        claim_list = list(claims)
        confidences = [float(item.get("confidence", 0.0)) for item in claim_list]
        unique_topics = {str(item.get("topic") or "general") for item in claim_list}
        return {
            "frontier_size": frontier_size,
            "claims_collected": len(claim_list),
            "unique_topics": len(unique_topics),
            "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "measurable": True,
            "replayable": True,
        }
