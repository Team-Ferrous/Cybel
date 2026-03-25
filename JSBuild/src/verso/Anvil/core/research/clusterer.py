"""Simple topic clustering for research evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List


class TopicClusterer:
    """Groups evidence by topic for roadmap and stop-proof synthesis."""

    def cluster(self, claims: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[str]] = defaultdict(list)
        for claim in claims:
            buckets[str(claim.get("topic") or "general")].append(str(claim.get("summary") or ""))
        output: List[Dict[str, Any]] = []
        for index, (topic, summaries) in enumerate(sorted(buckets.items())):
            output.append(
                {
                    "cluster_id": f"cluster_{index}",
                    "topic": topic,
                    "label": topic.replace("_", " ").title(),
                    "members": summaries,
                    "score": float(len(summaries)),
                }
            )
        return output
