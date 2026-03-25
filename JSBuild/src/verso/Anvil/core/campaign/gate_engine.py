"""Artifact and state promotion gate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GateRule:
    state: str
    required_artifact_families: List[str] = field(default_factory=list)
    required_approved_families: List[str] = field(default_factory=list)
    require_no_blocking_questions: bool = False
    require_feature_confirmation: bool = False
    required_repo_roles: List[str] = field(default_factory=list)


@dataclass
class GateDecision:
    allowed: bool
    state: str
    missing_requirements: List[str] = field(default_factory=list)


class GateEngine:
    """Apply deterministic promotion rules using campaign snapshots."""

    def __init__(self, rules: List[GateRule] | None = None) -> None:
        self.rules = {rule.state: rule for rule in (rules or self.default_rules())}

    @staticmethod
    def default_rules() -> List[GateRule]:
        return [
            GateRule(state="REPO_INGESTION", required_repo_roles=["target"]),
            GateRule(
                state="FEATURE_MAP_WAIT",
                required_artifact_families=["architecture", "feature_map"],
                require_no_blocking_questions=True,
            ),
            GateRule(
                state="ROADMAP_WAIT",
                required_artifact_families=["roadmap_draft"],
                required_approved_families=["feature_map"],
                require_no_blocking_questions=True,
                require_feature_confirmation=True,
            ),
            GateRule(
                state="DEVELOPMENT",
                required_artifact_families=["roadmap_final"],
                required_approved_families=["roadmap_final"],
            ),
            GateRule(
                state="CLOSURE",
                required_artifact_families=["closure", "audits", "telemetry"],
                required_approved_families=["roadmap_final"],
            ),
        ]

    def evaluate(self, state: str, snapshot: Dict[str, object]) -> GateDecision:
        rule = self.rules.get(state)
        if rule is None:
            return GateDecision(allowed=True, state=state)

        missing: List[str] = []
        artifact_families = set(snapshot.get("artifact_families", []))
        approved_families = set(snapshot.get("approved_families", []))
        repo_roles = set(snapshot.get("repo_roles", []))
        blocking_questions = int(snapshot.get("blocking_questions", 0))
        pending_feature_confirmation = int(
            snapshot.get("pending_feature_confirmation", 0)
        )

        for family in rule.required_artifact_families:
            if family not in artifact_families:
                missing.append(f"artifact:{family}")
        for family in rule.required_approved_families:
            if family not in approved_families:
                missing.append(f"approval:{family}")
        for role in rule.required_repo_roles:
            if role not in repo_roles:
                missing.append(f"repo:{role}")
        if rule.require_no_blocking_questions and blocking_questions:
            missing.append("blocking_questions")
        if rule.require_feature_confirmation and pending_feature_confirmation:
            missing.append("pending_feature_confirmation")

        return GateDecision(allowed=not missing, state=state, missing_requirements=missing)
