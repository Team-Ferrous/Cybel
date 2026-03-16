"""Utilities for ranking."""

from typing import Any


class FeedbackRanker:
    """Implements Learning-to-Rank by adjusting weights based on user feedback."""

    def __init__(self) -> None:
        """Initialize the instance."""
        self.weights = {"semantic": 1.0, "text": 1.0}

    def update_weights(self, session_data: dict[str, Any]) -> None:
        """Updates retrieval weights based on session success.
        If a file was 'accepted' (read + edited), boost its features.
        """
        if session_data.get("resolution") == "success":
            # Simplistic reinforcement
            self.weights["semantic"] += 0.01

    def re_rank(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Re-orders search results based on learned weights."""
        # Stub logic: sort by score * weight
        return sorted(
            results,
            key=lambda x: x.get("score", 0) * self.weights["semantic"],
            reverse=True,
        )
