"""Roadmap phase manager backed by file ownership boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.ownership.ownership_models import PhaseStartResult, PhaseStatus


@dataclass
class Phase:
    phase_id: str
    files: List[str]
    predecessors: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    campaign_id: Optional[str] = None
    repo_id: Optional[str] = None
    access_mode: str = "target_write"
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    blocked_reason: Optional[str] = None


class PhaseManager:
    """Coordinates phase transitions and file ownership boundaries."""

    def __init__(self, ownership_registry, task_graph=None):
        self.registry = ownership_registry
        self.graph = task_graph
        self.phases: Dict[str, Phase] = {}

    def register_phase(
        self,
        phase_id: str,
        tasks,
        files: List[str],
        predecessors: Optional[List[str]] = None,
        campaign_id: Optional[str] = None,
        repo_id: Optional[str] = None,
        access_mode: str = "target_write",
    ) -> None:
        task_ids = []
        for task in tasks or []:
            task_id = getattr(task, "id", None) if not isinstance(task, dict) else task.get("id")
            if task_id:
                task_ids.append(str(task_id))

        self.phases[phase_id] = Phase(
            phase_id=phase_id,
            files=list(dict.fromkeys(files or [])),
            predecessors=list(predecessors or []),
            task_ids=task_ids,
            campaign_id=campaign_id,
            repo_id=repo_id,
            access_mode=access_mode,
        )

    def begin_phase(self, phase_id: str) -> PhaseStartResult:
        phase = self.phases.get(phase_id)
        if phase is None:
            return PhaseStartResult(ready=False, phase_id=phase_id, blocked_reason="unknown_phase")

        missing = [
            predecessor
            for predecessor in phase.predecessors
            if self.phases.get(predecessor) is None
            or self.phases[predecessor].status != "completed"
        ]
        if missing:
            phase.status = "blocked"
            phase.blocked_reason = "predecessors_incomplete"
            return PhaseStartResult(
                ready=False,
                phase_id=phase_id,
                blocked_reason=phase.blocked_reason,
                missing_predecessors=missing,
            )

        claim = self.registry.claim_files(
            agent_id=f"Phase:{phase_id}",
            files=phase.files,
            mode="exclusive",
            phase_id=phase_id,
            task_id=None,
            campaign_id=phase.campaign_id,
            repo_id=phase.repo_id,
            agent_role="phase_manager",
            access_mode=phase.access_mode,
        )
        if not claim.success:
            phase.status = "blocked"
            phase.blocked_reason = "ownership_conflict"
            return PhaseStartResult(
                ready=False,
                phase_id=phase_id,
                blocked_reason=phase.blocked_reason,
                missing_predecessors=missing,
            )

        phase.status = "active"
        phase.started_at = time.time()
        phase.blocked_reason = None
        return PhaseStartResult(ready=True, phase_id=phase_id)

    def complete_phase(self, phase_id: str) -> None:
        phase = self.phases.get(phase_id)
        if phase is None:
            return

        self.registry.release_files(agent_id=f"Phase:{phase_id}", files=phase.files)
        phase.status = "completed"
        phase.completed_at = time.time()
        phase.blocked_reason = None

    def get_phase_status(self) -> Dict[str, PhaseStatus]:
        status: Dict[str, PhaseStatus] = {}
        for phase_id, phase in self.phases.items():
            status[phase_id] = PhaseStatus(
                phase_id=phase_id,
                status=phase.status,
                files=list(phase.files),
                predecessors=list(phase.predecessors),
                active_tasks=list(phase.task_ids),
                blocked_reason=phase.blocked_reason,
            )
        return status
