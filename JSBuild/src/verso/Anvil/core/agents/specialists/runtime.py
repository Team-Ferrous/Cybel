"""Runtime helpers for specialist routing and instantiation."""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Iterable, Optional, Type

from core.agents.specialists.manifest import (
    manifest_generic_role_aliases,
    manifest_question_domain_hints,
    manifest_role_domain_hints,
)
from core.agents.specialists.registry import RoutingDecision, SpecialistRegistry

if TYPE_CHECKING:
    from core.agents.subagent import SubAgent as SubAgentType

GENERIC_ROLE_ALIASES = {
    "researcher": "ResearchLibrarianSubagent",
    "architect": "SoftwareArchitectureSubagent",
    "analyzer": "RepoCampaignAnalysisSubagent",
    "repo_analyzer": "RepoCampaignAnalysisSubagent",
    "implementer": "ImplementationEngineerSubagent",
    "validator": "TestAuditSubagent",
    "tester": "TestAuditSubagent",
    "planner": "CampaignDirectorSubagent",
    "campaign_director": "CampaignDirectorSubagent",
    "documentation": "DocumentationWhitepaperSubagent",
    "observability": "TelemetrySystemsSubagent",
    "determinism": "DeterminismComplianceSubagent",
    "deadcode": "DeadCodeTriageSubagent",
}

ROLE_DOMAIN_HINTS = {
    "researcher": {"research", "web"},
    "architect": {"architecture", "design"},
    "analyzer": {"repo", "analysis_local"},
    "implementer": {"implementation", "code"},
    "validator": {"audit", "test"},
    "tester": {"audit", "test"},
    "planner": {"planning", "campaign"},
    "documentation": {"docs", "whitepaper"},
    "observability": {"telemetry", "observability"},
    "determinism": {"determinism", "reproducibility"},
    "deadcode": {"deadcode", "dead_code", "code_cleanup"},
}

QUESTION_DOMAIN_HINTS = {
    "research": {"research", "web"},
    "architecture": {"architecture", "design"},
    "investigation": {"repo", "analysis_local"},
}

KNOWN_DOMAIN_FAMILIES = (
    "aerospace",
    "architecture",
    "em",
    "engineering",
    "governance",
    "industrial",
    "repo",
    "robotics",
    "science",
    "surrogates",
    "toolchain",
)

_STATIC_ROLE_CLASS_MAP = {
    "CampaignDirectorSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "CampaignDirectorSubagent",
    ),
    "RepoIngestionSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "RepoIngestionSubagent",
    ),
    "ResearchCrawlerSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "ResearchCrawlerSubagent",
    ),
    "ArchitectureAdjudicatorSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "ArchitectureAdjudicatorSubagent",
    ),
    "FeatureCartographerSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "FeatureCartographerSubagent",
    ),
    "HypothesisLabSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "HypothesisLabSubagent",
    ),
    "MarketAnalysisSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "MarketAnalysisSubagent",
    ),
    "MathAnalysisSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "MathAnalysisSubagent",
    ),
    "HardwareOptimizationSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "HardwareOptimizationSubagent",
    ),
    "QuantumAlgorithmsSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "QuantumAlgorithmsSubagent",
    ),
    "PhysicsSimulationSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "PhysicsSimulationSubagent",
    ),
    "ImplementationEngineerSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "ImplementationEngineerSubagent",
    ),
    "TelemetrySystemsSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "TelemetrySystemsSubagent",
    ),
    "TestAuditSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "TestAuditSubagent",
    ),
    "CodebaseCartographerSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "CodebaseCartographerSubagent",
    ),
    "ReleasePackagingSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "ReleasePackagingSubagent",
    ),
    "DocumentationWhitepaperSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "DocumentationWhitepaperSubagent",
    ),
    "AESSentinelSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "AESSentinelSubagent",
    ),
    "DeterminismComplianceSubagent": (
        "core.agents.domain.foundation.foundational_subagents",
        "DeterminismComplianceSubagent",
    ),
    "ResearchSubagent": ("core.agents.researcher", "ResearchSubagent"),
    "RepoAnalysisSubagent": ("core.agents.repo_analyzer", "RepoAnalysisSubagent"),
    "DebugSubagent": ("core.agents.debugger", "DebugSubagent"),
    "ImplementationSubagent": ("core.agents.implementor", "ImplementationSubagent"),
    "TestingSubagent": ("core.agents.tester", "TestingSubagent"),
    "PlanningSubagent": ("core.agents.planner_agent", "PlanningSubagent"),
    "ArchitectSubagent": ("core.agents.architect_subagent", "ArchitectSubagent"),
    "MLResearchSubagent": ("core.agents.ml_research_subagent", "MLResearchSubagent"),
    "HPCSubagent": ("core.agents.hpc_subagent", "HPCSubagent"),
    "CPUOptimizerSubagent": ("core.agents.hpc_subagent", "CPUOptimizerSubagent"),
    "CompetitorSubagent": ("core.agents.competitor_subagent", "CompetitorSubagent"),
    "MathSubagent": ("core.agents.math_subagent", "MathSubagent"),
}


def _graph_dependency_anchor_for_dynamic_specialists() -> None:
    """Declare dynamic specialist modules for static graph reachability."""
    from core.agents.domain import base_domain_subagent  # noqa: F401
    from core.agents.domain.aerospace import (  # noqa: F401
        aeronautics_subagent,
        aerospace_systems_subagent,
        controls_gnc_subagent,
        fluids_subagent,
        mdo_trade_study_subagent,
        propulsion_subagent,
        simulation_vnv_subagent,
        thermal_systems_subagent,
        thruster_subagent,
    )
    from core.agents.domain.architecture import (  # noqa: F401
        adr_steward_subagent,
        counterfactual_strategist_subagent,
        software_architecture_subagent,
        system_cartographer_subagent,
    )
    from core.agents.domain.em import (  # noqa: F401
        electric_propulsion_subagent,
        electromagnetism_subagent,
        electromechanical_systems_subagent,
        magnetics_manufacturing_subagent,
        magnetics_subagent,
        magnetohydrodynamics_subagent,
        magnetoresponsive_materials_subagent,
        piezoelectric_smart_materials_subagent,
        plasma_physics_subagent,
    )
    from core.agents.domain.engineering import (  # noqa: F401
        cad_geometry_subagent,
        dfm_negotiator_subagent,
        electrical_engineering_subagent,
        embedded_systems_subagent,
        gcode_safety_marshal_subagent,
        materials_manufacturing_subagent,
        mechanical_engineering_subagent,
    )
    from core.agents.domain.foundation import foundational_subagents  # noqa: F401
    from core.agents.domain.governance import (  # noqa: F401
        compatibility_migration_architect_subagent,
        failure_economist_subagent,
        finops_model_economist_subagent,
        fitness_function_engineer_subagent,
        observability_contract_engineer_subagent,
        platform_dx_architect_subagent,
        supply_chain_provenance_auditor_subagent,
        threat_boundary_analyst_subagent,
    )
    from core.agents.domain.industrial import (  # noqa: F401
        additive_manufacturing_subagent,
        industrial_economics_subagent,
        industrial_iot_subagent,
        manufacturing_automation_subagent,
        process_control_subagent,
    )
    from core.agents.domain.repo import repo_campaign_analysis_subagent  # noqa: F401
    from core.agents.domain.robotics import (  # noqa: F401
        autonomous_systems_vnv_subagent,
        drone_autonomy_subagent,
        robotics_kinematics_subagent,
        sensor_fusion_subagent,
        uav_swarm_subagent,
    )
    from core.agents.domain.science import (  # noqa: F401
        formal_methods_subagent,
        quantum_computing_subagent,
        quantum_mechanics_subagent,
        research_librarian_subagent,
        scientific_computing_subagent,
    )
    from core.agents.domain.surrogates import (  # noqa: F401
        cfd_surrogate_subagent,
        digital_twin_architect_subagent,
        em_surrogate_subagent,
        heat_transfer_surrogate_subagent,
        mhd_surrogate_subagent,
        simulation_acceleration_subagent,
    )
    from core.agents.domain.toolchain import (  # noqa: F401
        abi_cartographer_subagent,
        benchmark_engineer_subagent,
        binary_layout_surgeon_subagent,
        counterexample_coroner_subagent,
        dead_code_triage_subagent,
        determinism_sheriff_subagent,
        integration_treaty_broker_subagent,
        lowering_blacksmith_subagent,
        pmu_pathologist_subagent,
        translation_validation_prosecutor_subagent,
    )


def normalize_requested_role(role: Optional[str]) -> str:
    token = str(role or "").strip()
    if not token:
        return ""
    lowered = token.lower()
    if lowered in manifest_generic_role_aliases():
        return manifest_generic_role_aliases()[lowered]
    if lowered in GENERIC_ROLE_ALIASES:
        return GENERIC_ROLE_ALIASES[lowered]
    return token


def route_specialist(
    *,
    registry: SpecialistRegistry,
    objective: str,
    requested_role: Optional[str] = None,
    aal: str = "AAL-3",
    domains: Optional[Iterable[str]] = None,
    question_type: str = "",
    repo_roles: Optional[Iterable[str]] = None,
    required_artifacts: Optional[Iterable[str]] = None,
) -> RoutingDecision:
    requested = normalize_requested_role(requested_role)
    domain_set = {str(item).strip().lower() for item in (domains or []) if str(item).strip()}
    role_key = str(requested_role or "").strip().lower()
    domain_set.update(manifest_role_domain_hints().get(role_key, set()))
    domain_set.update(ROLE_DOMAIN_HINTS.get(role_key, set()))
    question_key = str(question_type or "").strip().lower()
    domain_set.update(manifest_question_domain_hints().get(question_key, set()))
    domain_set.update(QUESTION_DOMAIN_HINTS.get(question_key, set()))

    baseline = registry.route(
        objective=objective,
        domains=sorted(domain_set),
        repo_roles=list(repo_roles or []),
        question_type=question_type,
        aal=aal,
        required_artifacts=required_artifacts,
    )

    if requested:
        if requested in registry.catalog:
            reviewers = [role for role in baseline.reviewer_roles if role != requested]
            reasons = list(baseline.reasons)
            reasons.append("requested_role_explicit")
            return RoutingDecision(
                primary_role=requested,
                reviewer_roles=reviewers,
                reasons=reasons,
            )
        if resolve_specialist_class(requested) is not None:
            reasons = list(baseline.reasons)
            reasons.append("requested_role_resolved")
            return RoutingDecision(
                primary_role=requested,
                reviewer_roles=list(baseline.reviewer_roles),
                reasons=reasons,
            )

    if requested_role and not baseline.reasons:
        baseline.reasons.append("requested_role_rerouted")
    return baseline


def build_specialist_subagent(
    *,
    role: str,
    task: str,
    parent_name: str = "Master",
    brain: Any = None,
    console: Any = None,
    parent_agent: Any = None,
    quiet: bool = False,
    message_bus: Any = None,
    ownership_registry: Any = None,
    complexity_profile: Any = None,
    context_budget: Optional[int] = None,
    coconut_context_vector: Any = None,
    coconut_depth: Optional[int] = None,
    prompt_profile: str = "sovereign_build",
    specialist_prompt_key: str = "",
    sovereign_build_policy_block: str = "",
    sovereign_build_policy_enabled: bool = True,
    prompt_injection: str = "",
) -> "SubAgentType":
    from core.agents.subagent import SubAgent

    cls = resolve_specialist_class(role, specialist_prompt_key=specialist_prompt_key) or SubAgent
    kwargs = {
        "task": task,
        "parent_name": parent_name,
        "brain": brain,
        "console": console,
        "parent_agent": parent_agent,
        "quiet": quiet,
        "message_bus": message_bus,
        "ownership_registry": ownership_registry,
        "complexity_profile": complexity_profile,
        "context_budget": context_budget,
        "coconut_context_vector": coconut_context_vector,
        "coconut_depth": coconut_depth,
        "prompt_profile": prompt_profile,
        "specialist_prompt_key": specialist_prompt_key,
        "sovereign_build_policy_block": sovereign_build_policy_block,
        "sovereign_build_policy_enabled": sovereign_build_policy_enabled,
    }
    if prompt_injection:
        kwargs["prompt_injection"] = prompt_injection
    return cls(**kwargs)


@lru_cache(maxsize=256)
def resolve_specialist_class(
    role: str,
    specialist_prompt_key: str = "",
) -> Optional[Type[Any]]:
    token = str(role or "").strip()
    if not token:
        return None

    prompt_key = str(specialist_prompt_key or "").strip()
    if "/" in prompt_key:
        family, base_name = prompt_key.split("/", maxsplit=1)
        module_name = f"core.agents.domain.{family}.{base_name}_subagent"
        try:
            module = import_module(module_name)
            cls = getattr(module, token, None)
            if isinstance(cls, type):
                return cls
        except Exception:
            pass

    static_mapping = _STATIC_ROLE_CLASS_MAP.get(token)
    if static_mapping:
        module_name, class_name = static_mapping
        try:
            module = import_module(module_name)
            cls = getattr(module, class_name, None)
            return cls if isinstance(cls, type) else None
        except Exception:
            return None

    snake = _camel_to_snake(token).removesuffix("_subagent")
    for family in KNOWN_DOMAIN_FAMILIES:
        module_name = f"core.agents.domain.{family}.{snake}_subagent"
        try:
            module = import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, token, None)
        if isinstance(cls, type):
            return cls
    return None


def _camel_to_snake(text: str) -> str:
    source = str(text or "").strip()
    token = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", source)
    token = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", token).lower()
    token = re.sub(r"_+", "_", token)
    return token.strip("_")
