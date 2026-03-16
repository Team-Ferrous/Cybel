"""Feature inventory and selection-map generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List


@dataclass
class FeatureEntry:
    feature_id: str
    name: str
    category: str
    description: str
    default_state: str = "defer"
    selection_state: str = "defer"
    requires_user_confirmation: bool = False
    depends_on: List[str] = field(default_factory=list)
    mutually_exclusive_with: List[str] = field(default_factory=list)
    evidence_links: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    maintenance_cost: float = 0.0
    market_value: float = 0.0
    hardware_impact: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


class FeatureMapBuilder:
    """Build feature inventory artifacts with deterministic checklist rendering."""

    def __init__(self, state_store, campaign_id: str):
        self.state_store = state_store
        self.campaign_id = campaign_id

    def build_from_candidates(
        self,
        candidates: Iterable[Dict[str, object]],
    ) -> List[FeatureEntry]:
        entries: List[FeatureEntry] = []
        for candidate in candidates:
            entries.append(
                FeatureEntry(
                    feature_id=str(candidate["feature_id"]),
                    name=str(candidate["name"]),
                    category=str(candidate.get("category") or "uncategorized"),
                    description=str(candidate.get("description") or ""),
                    default_state=str(candidate.get("default_state") or "defer"),
                    selection_state=str(candidate.get("selection_state") or candidate.get("default_state") or "defer"),
                    requires_user_confirmation=bool(candidate.get("requires_user_confirmation", False)),
                    depends_on=list(candidate.get("depends_on") or []),
                    mutually_exclusive_with=list(candidate.get("mutually_exclusive_with") or []),
                    evidence_links=list(candidate.get("evidence_links") or []),
                    complexity_score=float(candidate.get("complexity_score", 0.0)),
                    maintenance_cost=float(candidate.get("maintenance_cost", 0.0)),
                    market_value=float(candidate.get("market_value", 0.0)),
                    hardware_impact=float(candidate.get("hardware_impact", 0.0)),
                    metadata=dict(candidate.get("metadata") or {}),
                )
            )
        return entries

    def persist(self, entries: Iterable[FeatureEntry]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for entry in entries:
            payload = asdict(entry)
            payload["campaign_id"] = self.campaign_id
            self.state_store.record_feature(payload)
            rows.append(payload)
        return rows

    @staticmethod
    def render_checklist(entries: Iterable[FeatureEntry]) -> str:
        lines: List[str] = []
        for entry in sorted(entries, key=lambda item: (item.category, item.name)):
            state = entry.selection_state
            if state == "selected":
                marker = "[x]"
            elif state == "defer":
                marker = "[ ] defer"
            else:
                marker = "[ ]"
            lines.append(f"{marker} {entry.category}: {entry.name} - {entry.description}")
        return "\n".join(lines)
