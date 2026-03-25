"""Structured task packet schema for specialist routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List


@dataclass
class RepoConstraint:
    allowed_repos: List[str] = field(default_factory=list)
    forbidden_repos: List[str] = field(default_factory=list)
    required_repo_roles: List[str] = field(default_factory=list)


@dataclass
class ArtifactExpectation:
    required_artifacts: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)


@dataclass
class AESConstraint:
    aal: str = "AAL-3"
    action_type: str = "code_modification"
    waiver_ids: List[str] = field(default_factory=list)
    required_reviewers: List[str] = field(default_factory=list)


@dataclass
class TelemetryContract:
    required_metrics: List[str] = field(default_factory=lambda: ["wall_time"])
    logs_required: bool = True


@dataclass
class FailureEscalation:
    escalate_to: List[str] = field(default_factory=list)
    retry_policy: Dict[str, object] = field(default_factory=dict)
    stop_conditions: List[str] = field(default_factory=list)


@dataclass
class EvidenceEnvelopeRequirement:
    required: bool = True
    schema_version: str = "phase1"
    required_fields: List[str] = field(
        default_factory=lambda: [
            "fallback_mode",
            "saguaro_failures",
            "prompt_profile",
            "specialist_prompt_key",
        ]
    )
    allowed_fallback_modes: List[str] = field(
        default_factory=lambda: ["fallback_static_scan"]
    )


@dataclass
class SovereignPolicy:
    enabled: bool = False
    policy_block: str = ""
    injection_mode: str = "append"


@dataclass
class TaskPacket:
    task_packet_id: str
    objective: str
    specialist_role: str
    repo_constraint: RepoConstraint = field(default_factory=RepoConstraint)
    artifact_expectation: ArtifactExpectation = field(default_factory=ArtifactExpectation)
    aes_constraint: AESConstraint = field(default_factory=AESConstraint)
    telemetry_contract: TelemetryContract = field(default_factory=TelemetryContract)
    failure_escalation: FailureEscalation = field(default_factory=FailureEscalation)
    allowed_tools: List[str] = field(default_factory=list)
    prompt_profile: str = "default"
    specialist_prompt_key: str = ""
    evidence_envelope_requirement: EvidenceEnvelopeRequirement = field(
        default_factory=EvidenceEnvelopeRequirement
    )
    sovereign_policy: SovereignPolicy = field(default_factory=SovereignPolicy)
    evidence_bundle: Dict[str, object] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
