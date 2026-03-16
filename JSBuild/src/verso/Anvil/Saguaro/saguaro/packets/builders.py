"""Build bounded-context packets from traceability and omnigraph state."""

from __future__ import annotations

import hashlib
import os
from typing import Any

from saguaro.omnigraph.store import OmniGraphStore
from saguaro.packets.model import (
    MappingPacket,
    RequirementPacket,
    ReviewPacket,
    WitnessPacket,
)
from saguaro.validation.engine import ValidationEngine


class PacketBuilder:
    """Create requirement and witness packets."""

    def __init__(self, repo_path: str, graph_service: Any | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.validation = ValidationEngine(self.repo_path, graph_service=graph_service)
        self.omnigraph = OmniGraphStore(self.repo_path, graph_service=graph_service)

    def build_task_packet(self, task: str) -> dict[str, Any]:
        """Build a generic mapping packet."""
        graph = self.omnigraph.load()
        needle = task.lower()
        candidates = [
            node
            for node in graph.get("nodes", {}).values()
            if needle in str(node).lower()
        ][:10]
        packet = MappingPacket(
            id=self._packet_id("map", task),
            task=task,
            candidates=candidates,
            constraints=[
                "Prefer deterministic mappings over prose-only reasoning.",
                "Escalate when witness coverage is missing.",
            ],
        )
        return packet.to_dict()

    def review_packet(self, packet_id: str) -> dict[str, Any]:
        """Return a stored-like review from current graph state."""
        graph = self.omnigraph.load()
        for node in graph.get("nodes", {}).values():
            if packet_id in json_safe(node):
                return ReviewPacket(
                    id=self._packet_id("review", packet_id),
                    packet_id=packet_id,
                    findings=[],
                    related_node=node,
                ).to_dict()
        return {"status": "missing", "packet_id": packet_id}

    def witness_packet(self, requirement_id: str) -> dict[str, Any]:
        """Build a witness packet from validation output."""
        validation = self.validation.validate_requirement(requirement_id)
        if validation.get("status") != "ok":
            return validation
        packet = WitnessPacket(
            id=self._packet_id("wit", requirement_id),
            requirement_id=requirement_id,
            witnesses=list(validation.get("witnesses", [])),
        )
        return packet.to_dict()

    def requirement_packet(self, requirement_id: str) -> dict[str, Any]:
        """Build a requirement packet from omni-graph state."""
        explanation = self.omnigraph.explain(requirement_id)
        if explanation.get("status") != "ok":
            return explanation
        requirement = next(
            (item for item in explanation.get("nodes", []) if item.get("id") == requirement_id),
            None,
        )
        packet = RequirementPacket(
            id=self._packet_id("req", requirement_id),
            requirement_id=requirement_id,
            summary=str(requirement.get("label") if requirement else requirement_id),
            evidence=list(explanation.get("relations", [])),
            related_nodes=list(explanation.get("nodes", [])),
        )
        return packet.to_dict()

    @staticmethod
    def _packet_id(prefix: str, value: str) -> str:
        return f"{prefix.upper()}-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:12].upper()}"


def json_safe(value: Any) -> str:
    """Stable stringification for search."""
    return str(value).lower()
