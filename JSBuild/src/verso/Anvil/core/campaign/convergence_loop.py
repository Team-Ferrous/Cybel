"""Shared convergence-loop primitive for autonomy phases."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.campaign.state_store import CampaignStateStore


@dataclass
class LoopResult:
    iteration: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceReport:
    converged: bool
    iterations: int
    history: List[LoopResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    wall_time_seconds: float = 0.0


class ConvergenceLoop(ABC):
    """Loop that runs until a convergence predicate holds."""

    max_iterations: int = 100
    min_iterations: int = 1

    def __init__(
        self,
        *,
        campaign_id: str = "",
        loop_name: str = "",
        state_store: Optional[CampaignStateStore] = None,
    ) -> None:
        self.campaign_id = campaign_id
        self.loop_name = loop_name or self.__class__.__name__.lower()
        self.state_store = state_store

    @abstractmethod
    def iterate(self, iteration: int, state: dict[str, Any]) -> LoopResult: ...

    @abstractmethod
    def is_converged(self, history: list[LoopResult]) -> bool: ...

    @abstractmethod
    def on_converged(self, history: list[LoopResult]) -> dict[str, Any]: ...

    def run(self, initial_state: dict[str, Any]) -> ConvergenceReport:
        started = time.time()
        history: List[LoopResult] = []
        state = dict(initial_state)
        converged = False
        for iteration in range(1, self.max_iterations + 1):
            result = self.iterate(iteration, state)
            history.append(result)
            state.update(result.payload)
            converged = iteration >= self.min_iterations and self.is_converged(history)
            if self.state_store and self.campaign_id:
                self.state_store.record_convergence_checkpoint(
                    self.campaign_id,
                    self.loop_name,
                    iteration,
                    result.metrics,
                    converged=converged,
                )
            if converged:
                break
        summary = self.on_converged(history) if converged else {"state": state}
        return ConvergenceReport(
            converged=converged,
            iterations=len(history),
            history=history,
            summary=summary,
            wall_time_seconds=time.time() - started,
        )
