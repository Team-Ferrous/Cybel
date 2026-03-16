"""Community summaries for ALMF memory graphs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable


class MemoryCommunityBuilder:
    """Group memory objects into coarse communities for sensemaking."""

    def build(self, memories: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        buckets: dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for memory in memories:
            repo_context = str(memory.get("repo_context") or "")
            memory_kind = str(memory.get("memory_kind") or "unknown")
            bucket_key = repo_context or memory_kind
            buckets[bucket_key].append(memory)
        communities = []
        for bucket_key, items in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
            communities.append(
                {
                    "community_id": bucket_key,
                    "member_count": len(items),
                    "memory_kinds": sorted({str(item.get("memory_kind") or "unknown") for item in items}),
                    "summary": "; ".join(
                        str(item.get("summary_text") or "")[:80]
                        for item in items[:3]
                        if item.get("summary_text")
                    ),
                }
            )
        return {"communities": communities, "community_count": len(communities)}
