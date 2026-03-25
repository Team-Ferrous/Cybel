from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DefectEvent:
    created_at: datetime
    resolved_at: datetime | None = None
    injected_in_change: str | None = None


@dataclass
class ProcessMetricsTracker:
    defects: list[DefectEvent] = field(default_factory=list)

    def record_defect(self, created_at: datetime, injected_in_change: str | None = None) -> int:
        self.defects.append(DefectEvent(created_at=created_at, injected_in_change=injected_in_change))
        return len(self.defects) - 1

    def resolve_defect(self, index: int, resolved_at: datetime) -> None:
        self.defects[index].resolved_at = resolved_at

    def defect_injection_rate(self, change_count: int) -> float:
        if change_count <= 0:
            return 0.0
        injected = sum(1 for event in self.defects if event.injected_in_change)
        return injected / change_count

    def mttr_hours(self) -> float:
        resolved_durations = []
        for event in self.defects:
            if event.resolved_at is None:
                continue
            resolved_durations.append((event.resolved_at - event.created_at).total_seconds() / 3600.0)
        if not resolved_durations:
            return 0.0
        return sum(resolved_durations) / len(resolved_durations)

    def rca_stats(self) -> dict[str, float]:
        total = len(self.defects)
        resolved = sum(1 for event in self.defects if event.resolved_at is not None)
        unresolved = total - resolved
        return {
            "total_defects": float(total),
            "resolved_defects": float(resolved),
            "unresolved_defects": float(unresolved),
            "resolution_rate": (resolved / total) if total else 0.0,
        }
