"""Core data models for the DARE subsystem."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class KBEntry:
    category: str
    topic: str
    path: str
    content: str
    source: str
    confidence: str
    created: str
    updated: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    campaign_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RepoProfile:
    repo_path: str
    repo_name: str
    role: str
    read_only: bool
    file_count: int
    loc: int
    language_breakdown: Dict[str, int]
    entry_points: List[str]
    modules: List[str]
    test_files: List[str]
    build_files: List[str]
    dependency_graph: Dict[str, List[str]]
    detected_patterns: Dict[str, int]
    tech_stack: List[str]
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerAnalysis:
    layer_type: str
    files: List[str]
    opportunities: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisFinding:
    category: str
    title: str
    detail: str
    confidence: str = "medium"
    severity: str = "medium"
    evidence: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisReport:
    repo_path: str
    summary: str
    architecture: str
    findings: List[AnalysisFinding] = field(default_factory=list)
    layer_analyses: List[LayerAnalysis] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["findings"] = [item.to_dict() for item in self.findings]
        payload["layer_analyses"] = [item.to_dict() for item in self.layer_analyses]
        return payload


@dataclass
class ResearchSource:
    source_type: str
    title: str
    url: str
    summary: str
    confidence: str = "medium"
    published: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchReport:
    topic: str
    summary: str
    sources: List[ResearchSource] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    coverage: Dict[str, int] = field(default_factory=dict)
    novelty_score: float = 0.0
    exhausted: bool = False
    generated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["sources"] = [item.to_dict() for item in self.sources]
        return payload


@dataclass
class CompetitorProfile:
    name: str
    url: str
    summary: str
    stars: Optional[int] = None
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompetitiveReport:
    domain: str
    summary: str
    competitors: List[CompetitorProfile] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["competitors"] = [item.to_dict() for item in self.competitors]
        return payload


@dataclass
class ExperimentResult:
    name: str
    hypothesis: str
    verdict: str
    notes: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InnovationProposal:
    name: str
    description: str
    rationale: str
    evidence_topics: List[str] = field(default_factory=list)
    expected_impact: str = ""
    confidence: str = "medium"
    experiment_plan: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComponentSpec:
    name: str
    objective: str
    files: List[str] = field(default_factory=list)
    tests: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    estimated_duration: str = "1 phase"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhaseSpec:
    phase_id: str
    name: str
    objective: str
    files: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    estimated_duration: str = "1 phase"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TechnicalRoadmap:
    objective: str
    summary: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    proposals: List[InnovationProposal] = field(default_factory=list)
    components: List[ComponentSpec] = field(default_factory=list)
    phases: List[PhaseSpec] = field(default_factory=list)
    evidence_topics: List[str] = field(default_factory=list)
    confidence: str = "medium"
    generated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["proposals"] = [item.to_dict() for item in self.proposals]
        payload["components"] = [item.to_dict() for item in self.components]
        payload["phases"] = [item.to_dict() for item in self.phases]
        return payload

    def to_markdown(self) -> str:
        lines = [f"# Technical Roadmap", "", f"## Objective", self.objective, "", "## Summary", self.summary]
        if self.constraints:
            lines.extend(["", "## Constraints"])
            for key, value in sorted(self.constraints.items()):
                lines.append(f"- **{key}**: {value}")
        if self.proposals:
            lines.extend(["", "## Innovation Proposals"])
            for proposal in self.proposals:
                lines.append(f"### {proposal.name}")
                lines.append(proposal.description)
                if proposal.rationale:
                    lines.append(f"- Rationale: {proposal.rationale}")
                if proposal.expected_impact:
                    lines.append(f"- Expected Impact: {proposal.expected_impact}")
                if proposal.experiment_plan:
                    lines.append("- Experiment Plan:")
                    for step in proposal.experiment_plan:
                        lines.append(f"  - {step}")
        if self.components:
            lines.extend(["", "## Components"])
            for component in self.components:
                lines.append(f"- **{component.name}**: {component.objective}")
        if self.phases:
            lines.extend(["", "## Phases"])
            for phase in self.phases:
                lines.append(f"- **{phase.name}**: {phase.objective}")
        return "\n".join(lines).strip() + "\n"


@dataclass
class DareState:
    repos: List[str] = field(default_factory=list)
    repo_profiles: List[Dict[str, Any]] = field(default_factory=list)
    analysis_reports: List[Dict[str, Any]] = field(default_factory=list)
    research_reports: List[Dict[str, Any]] = field(default_factory=list)
    competitive_reports: List[Dict[str, Any]] = field(default_factory=list)
    roadmap: Optional[Dict[str, Any]] = None
    campaign_path: Optional[str] = None
    last_report_path: Optional[str] = None
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
