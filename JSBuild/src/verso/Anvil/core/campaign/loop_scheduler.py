"""Deterministic loop registration and dispatch for autonomy campaigns."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass
class LoopDefinition:
    loop_id: str
    purpose: str
    inputs: List[str] = field(default_factory=list)
    produced_artifacts: List[str] = field(default_factory=list)
    allowed_repo_roles: List[str] = field(default_factory=list)
    allowed_tools: List[str] = field(default_factory=list)
    stop_conditions: List[str] = field(default_factory=list)
    escalation_conditions: List[str] = field(default_factory=list)
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    telemetry_contract: Dict[str, Any] = field(default_factory=dict)
    promotion_effect: Optional[str] = None
    controlling_state: Optional[str] = None
    concurrent_family: Optional[str] = None


@dataclass
class LoopExecutionResult:
    loop_id: str
    status: str
    started_at: float
    completed_at: float
    artifacts_emitted: List[str] = field(default_factory=list)
    notes: Dict[str, Any] = field(default_factory=dict)


class LoopScheduler:
    """Register and execute loop families under explicit contracts."""

    def __init__(self) -> None:
        self._loops: Dict[str, LoopDefinition] = {}

    def register(self, loop: LoopDefinition) -> None:
        if not loop.stop_conditions:
            raise ValueError(f"Loop '{loop.loop_id}' must declare stop conditions")
        self._loops[loop.loop_id] = loop

    def register_many(self, loops: Iterable[LoopDefinition]) -> None:
        for loop in loops:
            self.register(loop)

    def get(self, loop_id: str) -> Optional[LoopDefinition]:
        return self._loops.get(loop_id)

    def list_loops(self) -> List[LoopDefinition]:
        return [self._loops[key] for key in sorted(self._loops)]

    def loops_for_state(self, state: str) -> List[LoopDefinition]:
        return [
            loop
            for loop in self.list_loops()
            if loop.controlling_state == state
        ]

    def choose_next(
        self,
        *,
        state: str,
        completed_loop_ids: Optional[Iterable[str]] = None,
    ) -> Optional[LoopDefinition]:
        completed = set(completed_loop_ids or [])
        for loop in self.loops_for_state(state):
            if loop.loop_id not in completed:
                return loop
        return None

    def execute(
        self,
        loop_id: str,
        executor: Callable[[LoopDefinition], Dict[str, Any]],
    ) -> LoopExecutionResult:
        loop = self._loops[loop_id]
        started = time.time()
        payload = executor(loop) or {}
        completed = time.time()
        return LoopExecutionResult(
            loop_id=loop_id,
            status=str(payload.get("status", "completed")),
            started_at=started,
            completed_at=completed,
            artifacts_emitted=list(payload.get("artifacts_emitted", [])),
            notes=dict(payload.get("notes", {})),
        )
