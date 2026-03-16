"""YAML campaign manifest parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class CampaignGateSpec:
    assertions: List[str] = field(default_factory=list)


@dataclass
class CampaignPhaseSpec:
    id: str
    name: str
    depends_on: List[str] = field(default_factory=list)
    iteration: str = "none"
    agent_tasks: List[str] = field(default_factory=list)
    per_file_agent_task: Optional[str] = None
    gate: CampaignGateSpec = field(default_factory=CampaignGateSpec)
    memory_policy: str = "keep_model"


@dataclass
class CampaignRepoSpec:
    name: str
    path: str
    role: str
    write_policy: str = "immutable"
    trust_level: str = "medium"


@dataclass
class CampaignLoopSpec:
    loop_id: str
    purpose: str
    stop_conditions: List[str] = field(default_factory=list)
    controlling_state: Optional[str] = None


@dataclass
class CampaignManifest:
    name: str
    config: Dict[str, Any]
    phases: List[CampaignPhaseSpec]
    repos: List[CampaignRepoSpec] = field(default_factory=list)
    loops: List[CampaignLoopSpec] = field(default_factory=list)


class ManifestLoader:
    """Parse and validate campaign specification files."""

    def load_yaml(self, yaml_path: str) -> CampaignManifest:
        with open(yaml_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        campaign = payload.get("campaign") or {}
        name = str(campaign.get("name") or "Generated Campaign")
        config = dict(campaign.get("config") or {})
        repos = [self._parse_repo(item) for item in (campaign.get("repos") or [])]
        loops = [self._parse_loop(item) for item in (campaign.get("loops") or [])]

        raw_phases = list(campaign.get("phases") or [])
        phases = [self._parse_phase(item) for item in raw_phases]
        self._validate_dependencies(phases)

        return CampaignManifest(
            name=name,
            config=config,
            phases=phases,
            repos=repos,
            loops=loops,
        )

    def _parse_phase(self, payload: Dict[str, Any]) -> CampaignPhaseSpec:
        phase_id = str(payload.get("id") or "phase")
        phase_name = str(payload.get("name") or phase_id.replace("_", " ").title())
        depends_on = [str(item) for item in payload.get("depends_on") or []]

        gate_payload = payload.get("gate") or {}
        assertions = [str(item) for item in gate_payload.get("assertions") or []]
        gate = CampaignGateSpec(assertions=assertions)

        return CampaignPhaseSpec(
            id=phase_id,
            name=phase_name,
            depends_on=depends_on,
            iteration=str(payload.get("iteration") or "none"),
            agent_tasks=[str(item) for item in payload.get("agent_tasks") or []],
            per_file_agent_task=payload.get("per_file_agent_task"),
            gate=gate,
            memory_policy=str(payload.get("memory_policy") or "keep_model"),
        )

    def _parse_repo(self, payload: Dict[str, Any]) -> CampaignRepoSpec:
        return CampaignRepoSpec(
            name=str(payload.get("name") or payload.get("path") or "repo"),
            path=str(payload.get("path") or "."),
            role=str(payload.get("role") or "analysis_local"),
            write_policy=str(payload.get("write_policy") or "immutable"),
            trust_level=str(payload.get("trust_level") or "medium"),
        )

    def _parse_loop(self, payload: Dict[str, Any]) -> CampaignLoopSpec:
        return CampaignLoopSpec(
            loop_id=str(payload.get("loop_id") or "loop"),
            purpose=str(payload.get("purpose") or ""),
            stop_conditions=[str(item) for item in payload.get("stop_conditions") or []],
            controlling_state=payload.get("controlling_state"),
        )

    @staticmethod
    def _validate_dependencies(phases: List[CampaignPhaseSpec]) -> None:
        known_ids = {phase.id for phase in phases}
        for phase in phases:
            for dependency in phase.depends_on:
                if dependency not in known_ids:
                    raise ValueError(
                        f"Phase '{phase.id}' depends on unknown phase '{dependency}'"
                    )
