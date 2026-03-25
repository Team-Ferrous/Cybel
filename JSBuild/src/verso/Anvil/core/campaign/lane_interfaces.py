"""Shared experiment-lane interfaces for campaign execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class IdeaCandidate:
    """A bounded idea that can be executed by the shared lane runtime."""

    idea_id: str
    source_type: str
    title: str
    objective: str
    hypothesis_id: str = ""
    description: str = ""
    editable_scope: List[str] = field(default_factory=list)
    read_only_scope: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkContract:
    """Metric contract for an experiment lane."""

    contract_id: str
    name: str
    required_metrics: List[str] = field(default_factory=list)
    optional_metrics: List[str] = field(default_factory=list)
    objective_metric: str = "score"
    direction: str = "maximize"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PromotionDecision:
    """The keep/discard verdict for an experiment lane run."""

    verdict: str
    score: float
    reasons: List[str] = field(default_factory=list)
    metric_deltas: Dict[str, float] = field(default_factory=dict)
    policy_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LaneTask:
    """Full execution packet for the shared lane runtime."""

    lane_id: str
    caller_mode: str
    lane_type: str
    name: str
    objective_function: str
    commands: List[str | Dict[str, Any]] = field(default_factory=list)
    description: str = ""
    editable_scope: List[str] = field(default_factory=list)
    read_only_scope: List[str] = field(default_factory=list)
    benchmark_contract: Dict[str, Any] = field(default_factory=dict)
    promotion_policy: Dict[str, Any] = field(default_factory=dict)
    telemetry_contract: Dict[str, Any] = field(default_factory=dict)
    rollback_criteria: List[str] = field(default_factory=list)
    kill_criteria: List[str] = field(default_factory=list)
    allowed_writes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
