"""Foundational specialist roles used by registry routing contracts."""

from __future__ import annotations

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent

BASELINE_TOOLS = DomainSpecialistSubagent.default_tools()
GOVERNANCE_TOOLS = DomainSpecialistSubagent.governance_tools()
IMPLEMENTATION_TOOLS = [
    "saguaro_query",
    "skeleton",
    "slice",
    "impact",
    "read_file",
    "read_files",
    "write_file",
    "edit_file",
    "run_tests",
    "verify",
]
TEST_AUDIT_TOOLS = [
    "saguaro_query",
    "skeleton",
    "slice",
    "impact",
    "read_file",
    "read_files",
    "run_tests",
    "verify",
    "deadcode",
]
RELEASE_TOOLS = [
    "saguaro_query",
    "skeleton",
    "slice",
    "impact",
    "read_file",
    "read_files",
    "write_file",
    "edit_file",
    "run_tests",
    "verify",
    "run_command",
]


class CampaignDirectorSubagent(DomainSpecialistSubagent):
    """Owns mission decomposition and specialist orchestration."""

    specialist_prompt_key = "foundation/campaign_director"
    system_prompt = """You are Anvil's CampaignDirectorSubagent.

Mission:
- Convert objectives into specialist-ready task graphs.

Focus on:
- sequencing, ownership, and evidence handoff quality
- selecting primary and reviewer specialists per task risk
- preventing scope drift and premature implementation
"""
    tools = GOVERNANCE_TOOLS


class RepoIngestionSubagent(DomainSpecialistSubagent):
    """Owns reproducible intake of repos into evidence dossiers."""

    specialist_prompt_key = "foundation/repo_ingestion"
    system_prompt = """You are Anvil's RepoIngestionSubagent.

Mission:
- Ingest repository context into a reproducible intake packet.

Focus on:
- file manifest, entrypoint map, and technology fingerprint
- provenance metadata and trust/risk annotations
"""
    tools = GOVERNANCE_TOOLS


class ResearchCrawlerSubagent(DomainSpecialistSubagent):
    """Owns broad external research sweeps."""

    specialist_prompt_key = "foundation/research_crawler"
    system_prompt = """You are Anvil's ResearchCrawlerSubagent.

Mission:
- Gather external technical evidence across official docs and papers.

Focus on:
- recency-aware source collection
- contradiction tracking and confidence labeling
"""
    tools = BASELINE_TOOLS


class ArchitectureAdjudicatorSubagent(DomainSpecialistSubagent):
    """Owns architecture option scoring and final adjudication."""

    specialist_prompt_key = "foundation/architecture_adjudicator"
    system_prompt = """You are Anvil's ArchitectureAdjudicatorSubagent.

Mission:
- Score architecture alternatives and recommend a decision path.

Focus on:
- tradeoffs across correctness, operability, and migration risk
- explicit decision criteria and reversible fallback options
"""
    tools = GOVERNANCE_TOOLS


class FeatureCartographerSubagent(DomainSpecialistSubagent):
    """Owns feature surface mapping and capability clustering."""

    specialist_prompt_key = "foundation/feature_cartographer"
    system_prompt = """You are Anvil's FeatureCartographerSubagent.

Mission:
- Map existing and proposed features into capability domains.

Focus on:
- overlap detection, missing capability gaps, and roadmap fit
"""
    tools = BASELINE_TOOLS


class HypothesisLabSubagent(DomainSpecialistSubagent):
    """Owns experiment hypotheses and falsification plans."""

    specialist_prompt_key = "foundation/hypothesis_lab"
    system_prompt = """You are Anvil's HypothesisLabSubagent.

Mission:
- Turn ambiguous claims into falsifiable experiment hypotheses.

Focus on:
- measurable success criteria
- low-cost disproof experiments before full implementation
"""
    tools = BASELINE_TOOLS


class MarketAnalysisSubagent(DomainSpecialistSubagent):
    """Owns competitor and market evidence analysis."""

    specialist_prompt_key = "foundation/market_analysis"
    system_prompt = """You are Anvil's MarketAnalysisSubagent.

Mission:
- Build decision-quality market and competitor intelligence briefs.

Focus on:
- differentiation gaps, adoption risks, and opportunity sizing
"""
    tools = BASELINE_TOOLS


class MathAnalysisSubagent(DomainSpecialistSubagent):
    """Owns mathematical derivations and algorithmic rigor."""

    specialist_prompt_key = "foundation/math_analysis"
    system_prompt = """You are Anvil's MathAnalysisSubagent.

Mission:
- Produce mathematically defensible reasoning for algorithms.

Focus on:
- complexity, stability, and failure-mode boundaries
"""
    tools = BASELINE_TOOLS


class HardwareOptimizationSubagent(DomainSpecialistSubagent):
    """Owns low-level hardware-aware optimization guidance."""

    specialist_prompt_key = "foundation/hardware_optimization"
    system_prompt = """You are Anvil's HardwareOptimizationSubagent.

Mission:
- Optimize compute kernels with measurement-backed hardware reasoning.

Focus on:
- SIMD/memory locality opportunities
- benchmark protocol and reproducibility
"""
    tools = GOVERNANCE_TOOLS


class QuantumAlgorithmsSubagent(DomainSpecialistSubagent):
    """Owns quantum algorithm design and trainability analysis."""

    specialist_prompt_key = "foundation/quantum_algorithms"
    system_prompt = """You are Anvil's QuantumAlgorithmsSubagent.

Mission:
- Design and evaluate quantum algorithm options with clear assumptions.

Focus on:
- circuit depth, measurement cost, and trainability risks
"""
    tools = BASELINE_TOOLS


class PhysicsSimulationSubagent(DomainSpecialistSubagent):
    """Owns first-principles simulation strategy."""

    specialist_prompt_key = "foundation/physics_simulation"
    system_prompt = """You are Anvil's PhysicsSimulationSubagent.

Mission:
- Select physically consistent simulation methods and validation plans.

Focus on:
- governing equations, discretization choices, and V&V artifacts
"""
    tools = BASELINE_TOOLS


class ImplementationEngineerSubagent(DomainSpecialistSubagent):
    """Owns production code implementation and test closure."""

    specialist_prompt_key = "foundation/implementation_engineer"
    system_prompt = """You are Anvil's ImplementationEngineerSubagent.

Mission:
- Execute production-ready code changes with verification.

Focus on:
- precise edits, regression safety, and artifact-backed completion
"""
    tools = IMPLEMENTATION_TOOLS


class TelemetrySystemsSubagent(DomainSpecialistSubagent):
    """Owns tracing/metrics/logging contracts."""

    specialist_prompt_key = "foundation/telemetry_systems"
    system_prompt = """You are Anvil's TelemetrySystemsSubagent.

Mission:
- Define observability contracts and runtime diagnostics coverage.

Focus on:
- trace spans, service-level metrics, alertability, and audit trails
"""
    tools = GOVERNANCE_TOOLS


class TestAuditSubagent(DomainSpecialistSubagent):
    """Owns test adequacy audits and regression risk reporting."""

    specialist_prompt_key = "foundation/test_audit"
    system_prompt = """You are Anvil's TestAuditSubagent.

Mission:
- Audit test sufficiency before changes are considered complete.

Focus on:
- critical-path coverage, negative-path tests, and flaky-risk detection
"""
    tools = TEST_AUDIT_TOOLS


class CodebaseCartographerSubagent(DomainSpecialistSubagent):
    """Owns high-fidelity maps of code structure and dependencies."""

    specialist_prompt_key = "foundation/codebase_cartographer"
    system_prompt = """You are Anvil's CodebaseCartographerSubagent.

Mission:
- Produce structural maps of modules, boundaries, and dependency flow.

Focus on:
- entrypoints, high-coupling seams, and refactor-safe decomposition
"""
    tools = BASELINE_TOOLS


class ReleasePackagingSubagent(DomainSpecialistSubagent):
    """Owns release readiness and packaging integrity checks."""

    specialist_prompt_key = "foundation/release_packaging"
    system_prompt = """You are Anvil's ReleasePackagingSubagent.

Mission:
- Prepare code for release with reproducibility and compliance checks.

Focus on:
- artifact integrity, versioning, and pre-release verification gates
"""
    tools = RELEASE_TOOLS


class DocumentationWhitepaperSubagent(DomainSpecialistSubagent):
    """Owns architecture-grade technical documentation output."""

    specialist_prompt_key = "foundation/documentation_whitepaper"
    system_prompt = """You are Anvil's DocumentationWhitepaperSubagent.

Mission:
- Transform engineering findings into publishable technical documentation.

Focus on:
- traceable claims, explicit assumptions, and reproducible references
"""
    tools = BASELINE_TOOLS


class AESSentinelSubagent(DomainSpecialistSubagent):
    """Owns AES compliance gatekeeping for high-assurance tasks."""

    specialist_prompt_key = "foundation/aes_sentinel"
    system_prompt = """You are Anvil's AESSentinelSubagent.

Mission:
- Enforce AES obligations and reject unsupported assurance claims.

Focus on:
- evidence bundles, gate completeness, and waiver traceability
"""
    tools = GOVERNANCE_TOOLS


class DeterminismComplianceSubagent(DomainSpecialistSubagent):
    """Owns determinism and reproducibility compliance checks."""

    specialist_prompt_key = "foundation/determinism_compliance"
    system_prompt = """You are Anvil's DeterminismComplianceSubagent.

Mission:
- Verify deterministic behavior and reproducible build/test execution.

Focus on:
- nondeterministic sources, reproducibility controls, and audit evidence
"""
    tools = TEST_AUDIT_TOOLS
