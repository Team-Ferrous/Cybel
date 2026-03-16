from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArtifactNode:
    artifact_id: str
    question: str
    path: str
    summary: str
    required: bool
    exists: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "question": self.question,
            "path": self.path,
            "summary": self.summary,
            "required": self.required,
            "exists": self.exists,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceabilityNode:
    node_id: str
    question: str
    label: str
    artifact_ref: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "question": self.question,
            "label": self.label,
            "artifact_ref": self.artifact_ref,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceabilityEdge:
    edge_id: str
    src: str
    dst: str
    relation: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.edge_id,
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "evidence": list(self.evidence),
        }
