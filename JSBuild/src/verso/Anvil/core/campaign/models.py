"""Typed models for the autonomy campaign control plane."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CampaignStateName(str, Enum):
    INTAKE = "INTAKE"
    REPO_INGESTION = "REPO_INGESTION"
    REPO_ACQUISITION = "REPO_ACQUISITION"
    RESEARCH = "RESEARCH"
    RESEARCH_RECONCILIATION = "RESEARCH_RECONCILIATION"
    QUESTIONNAIRE_BUILD = "QUESTIONNAIRE_BUILD"
    QUESTIONNAIRE_WAIT = "QUESTIONNAIRE_WAIT"
    FEATURE_MAP_BUILD = "FEATURE_MAP_BUILD"
    FEATURE_MAP_WAIT = "FEATURE_MAP_WAIT"
    EID_LAB = "EID_LAB"
    ROADMAP_DRAFT = "ROADMAP_DRAFT"
    ROADMAP_RECONCILIATION = "ROADMAP_RECONCILIATION"
    ROADMAP_WAIT = "ROADMAP_WAIT"
    DEVELOPMENT = "DEVELOPMENT"
    REMEDIATION = "REMEDIATION"
    SOAK_TEST = "SOAK_TEST"
    AUDIT = "AUDIT"
    CLOSURE = "CLOSURE"


class RepoRole(str, Enum):
    TARGET = "target"
    ANALYSIS_LOCAL = "analysis_local"
    ANALYSIS_EXTERNAL = "analysis_external"
    ARTIFACT_STORE = "artifact_store"
    BENCHMARK_FIXTURE = "benchmark_fixture"


class RepoWritePolicy(str, Enum):
    IMMUTABLE = "immutable"
    ARTIFACT_ONLY = "artifact_only"
    PHASE_GATED_WRITE = "phase_gated_write"
    SANDBOXED_WRITE = "sandboxed_write"


class OwnershipAccessMode(str, Enum):
    ANALYSIS_READONLY = "analysis_readonly"
    ANALYSIS_EXTRACT_ONLY = "analysis_extract_only"
    TARGET_PLAN_ONLY = "target_plan_only"
    TARGET_WRITE = "target_write"
    ARTIFACT_WRITE = "artifact_write"
    AUDIT_READONLY = "audit_readonly"


class ArtifactFamily(str, Enum):
    INTAKE = "intake"
    ARCHITECTURE = "architecture"
    FEATURE_MAP = "feature_map"
    RESEARCH = "research"
    ROADMAP_DRAFT = "roadmap_draft"
    ROADMAP_FINAL = "roadmap_final"
    EXPERIMENTS = "experiments"
    TELEMETRY = "telemetry"
    AUDITS = "audits"
    WHITEPAPERS = "whitepapers"
    CLOSURE = "closure"


class ArtifactApprovalState(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


@dataclass
class RepoRegistration:
    repo_id: str
    name: str
    origin: str
    revision: str
    local_path: str
    role: str
    write_policy: str
    topic_tags: List[str] = field(default_factory=list)
    trust_level: str = "standard"
    ingestion_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoopDefinition:
    loop_id: str
    purpose: str
    inputs: List[str]
    produced_artifacts: List[str]
    allowed_repo_roles: List[str]
    allowed_tools: List[str]
    stop_conditions: List[str]
    escalation_conditions: List[str]
    retry_policy: Dict[str, Any]
    telemetry_contract: Dict[str, Any]
    promotion_effect: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArtifactRecord:
    artifact_id: str
    campaign_id: str
    family: str
    name: str
    version: int
    canonical_path: str
    rendered_path: Optional[str]
    approval_state: str = ArtifactApprovalState.PENDING.value
    blocking: bool = False
    provenance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureQuestion:
    question_id: str
    question: str
    why_it_matters: str
    decision_scope: str
    blocking_level: str
    answer_mode: str
    default_policy: str
    linked_roadmap_items: List[str] = field(default_factory=list)
    current_status: str = "open"
    answer: Optional[str] = None
    reasoning_bundle: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureEntry:
    feature_id: str
    name: str
    category: str
    description: str
    default_state: str
    selection_state: str
    requires_user_confirmation: bool
    depends_on: List[str] = field(default_factory=list)
    mutually_exclusive_with: List[str] = field(default_factory=list)
    evidence_links: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    maintenance_cost: float = 0.0
    market_value: float = 0.0
    hardware_impact: str = "neutral"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoadmapItem:
    item_id: str
    phase_id: str
    title: str
    type: str
    repo_scope: List[str]
    owner_type: str
    depends_on: List[str]
    description: str
    success_metrics: List[str]
    required_evidence: List[str]
    required_artifacts: List[str]
    telemetry_contract: Dict[str, Any]
    exit_gate: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskPacket:
    task_packet_id: str
    objective: str
    allowed_repos: List[str]
    forbidden_repos: List[str]
    allowed_tools: List[str]
    expected_artifacts: List[str]
    evidence_bundle: List[str]
    aes_metadata: Dict[str, Any]
    success_metrics: List[str]
    telemetry_contract: Dict[str, Any]
    failure_escalation_path: str
    specialist_roles: List[str] = field(default_factory=list)
    repo_permissions: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisRecord:
    hypothesis_id: str
    statement: str
    motivation: str
    source_basis: List[str]
    target_subsystems: List[str]
    expected_upside: str
    risk: str
    required_experiments: List[str]
    status: str = "proposed"
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRecord:
    experiment_id: str
    environment: str
    commands: List[str]
    inputs: Dict[str, Any]
    metrics: List[str]
    pass_fail_rule: str
    artifacts_emitted: List[str]
    follow_up_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetryRecord:
    run_id: str
    span_name: str
    wall_time_seconds: float
    cpu_time_seconds: float
    rss_mb: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditFinding:
    finding_id: str
    category: str
    severity: str
    summary: str
    evidence_links: List[str] = field(default_factory=list)
    remediation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompletionProof:
    campaign_id: str
    summary: str
    completed_items: List[str]
    waivers: List[str]
    research_stop_reason: str
    audit_cycles: int
    artifact_inventory: List[str]
    reproducibility_instructions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
