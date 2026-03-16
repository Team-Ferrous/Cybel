"""Temporal hierarchy helpers for ALMF."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable


class MemoryTemporalTreeBuilder:
    """Build a simple temporal hierarchy from memory rows."""

    def build(self, memories: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        by_day: dict[str, list[Dict[str, Any]]] = defaultdict(list)
        for memory in memories:
            observed_at = float(memory.get("observed_at") or memory.get("created_at") or 0.0)
            day_key = datetime.fromtimestamp(observed_at, tz=timezone.utc).strftime("%Y-%m-%d")
            by_day[day_key].append(memory)
        days = []
        for day_key in sorted(by_day):
            items = by_day[day_key]
            kinds: dict[str, int] = defaultdict(int)
            for item in items:
                kinds[str(item.get("memory_kind") or "unknown")] += 1
            days.append(
                {
                    "day": day_key,
                    "count": len(items),
                    "memory_kinds": dict(sorted(kinds.items())),
                    "memory_ids": [str(item.get("memory_id") or "") for item in items],
                }
            )
        return {
            "days": days,
            "total_days": len(days),
            "total_memories": sum(day["count"] for day in days),
        }
