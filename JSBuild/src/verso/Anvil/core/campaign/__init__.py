"""Campaign orchestration package."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AgentResult": ("core.campaign.base_campaign", "AgentResult"),
    "ArchitectureQuestion": ("core.campaign.questionnaire", "ArchitectureQuestion"),
    "BaseCampaignLoop": ("core.campaign.base_campaign", "BaseCampaignLoop"),
    "CampaignArtifactRegistry": (
        "core.campaign.artifact_registry",
        "CampaignArtifactRegistry",
    ),
    "CampaignAuditEngine": ("core.campaign.audit_engine", "CampaignAuditEngine"),
    "CampaignGenerator": ("core.campaign.campaign_generator", "CampaignGenerator"),
    "CampaignManifest": ("core.campaign.manifest", "CampaignManifest"),
    "CampaignPhaseSpec": ("core.campaign.manifest", "CampaignPhaseSpec"),
    "CampaignReport": ("core.campaign.base_campaign", "CampaignReport"),
    "CampaignRunner": ("core.campaign.runner", "CampaignRunner"),
    "CampaignRepoCache": ("core.campaign.repo_cache", "CampaignRepoCache"),
    "CampaignRepoRecord": ("core.campaign.repo_registry", "CampaignRepoRecord"),
    "CampaignRepoRegistry": ("core.campaign.repo_registry", "CampaignRepoRegistry"),
    "CampaignState": ("core.campaign.base_campaign", "CampaignState"),
    "CampaignStateStore": ("core.campaign.state_store", "CampaignStateStore"),
    "CampaignTaskGraph": ("core.campaign.task_graph", "CampaignTaskGraph"),
    "CampaignTelemetry": ("core.campaign.telemetry", "CampaignTelemetry"),
    "CampaignWorkspace": ("core.campaign.workspace", "CampaignWorkspace"),
    "CompletionEngine": ("core.campaign.completion_engine", "CompletionEngine"),
    "FeatureEntry": ("core.campaign.feature_map", "FeatureEntry"),
    "FeatureMapBuilder": ("core.campaign.feature_map", "FeatureMapBuilder"),
    "GateDecision": ("core.campaign.gate_engine", "GateDecision"),
    "GateEngine": ("core.campaign.gate_engine", "GateEngine"),
    "GateRule": ("core.campaign.gate_engine", "GateRule"),
    "LoopDefinition": ("core.campaign.loop_scheduler", "LoopDefinition"),
    "LoopExecutionResult": (
        "core.campaign.loop_scheduler",
        "LoopExecutionResult",
    ),
    "LoopScheduler": ("core.campaign.loop_scheduler", "LoopScheduler"),
    "ManifestLoader": ("core.campaign.manifest", "ManifestLoader"),
    "PhaseStatus": ("core.campaign.base_campaign", "PhaseStatus"),
    "QuestionnaireBuilder": ("core.campaign.questionnaire", "QuestionnaireBuilder"),
    "RoadmapCompiler": ("core.campaign.roadmap_compiler", "RoadmapCompiler"),
    "RoadmapItem": ("core.campaign.roadmap_compiler", "RoadmapItem"),
    "TaskNode": ("core.campaign.task_graph", "TaskNode"),
    "TelemetryContract": ("core.campaign.telemetry", "TelemetryContract"),
    "TheLedger": ("core.campaign.ledger", "TheLedger"),
    "gate": ("core.campaign.base_campaign", "gate"),
    "phase": ("core.campaign.base_campaign", "phase"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
