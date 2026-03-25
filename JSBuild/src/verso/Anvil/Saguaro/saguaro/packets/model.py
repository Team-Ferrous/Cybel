"""Packet data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RequirementPacket:
    """Bounded context packet for one requirement."""

    id: str
    requirement_id: str
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    related_nodes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MappingPacket:
    """Packet for mapping a task to repo artifacts."""

    id: str
    task: str
    candidates: list[dict[str, Any]]
    constraints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WitnessPacket:
    """Packet for witness review."""

    id: str
    requirement_id: str
    witnesses: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PatchIntentPacket:
    """Packet describing a bounded patch objective."""

    id: str
    task: str
    requirement_ids: list[str] = field(default_factory=list)
    candidate_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CounterexamplePacket:
    """Packet carrying counterexample review context."""

    id: str
    requirement_id: str
    counterexamples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewPacket:
    """Packet for bounded review output."""

    id: str
    packet_id: str
    findings: list[dict[str, Any]] = field(default_factory=list)
    related_node: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
