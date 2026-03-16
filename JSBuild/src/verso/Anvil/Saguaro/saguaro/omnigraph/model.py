"""Typed omni-graph contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OmniNode:
    """A node in the joined omni-graph."""

    id: str
    type: str
    label: str
    file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OmniRelation:
    """Typed relation with evidence and uncertainty."""

    id: str
    src_type: str
    src_id: str
    dst_type: str
    dst_id: str
    relation_type: str
    evidence_types: list[str]
    confidence: float
    verified: bool
    drift_state: str
    generation_id: str
    notes: list[str] = field(default_factory=list)
    uncertainty: dict[str, Any] = field(default_factory=dict)
    evidence_spans: list[dict[str, Any]] = field(default_factory=list)
    evidence_mix: list[str] = field(default_factory=list)
    confidence_components: dict[str, Any] = field(default_factory=dict)
    parser_uncertainty: str = "unknown"
    counterevidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
