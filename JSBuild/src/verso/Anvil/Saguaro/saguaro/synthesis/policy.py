from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class StrategyDecision:
    strategy: str
    solver_enabled: bool
    eqsat_enabled: bool
    assembly_only: bool
    rationale: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SynthesisPolicyEngine:
    """Choose a deterministic synthesis strategy from runtime telemetry."""

    def decide(
        self,
        runtime_status: dict[str, Any],
        *,
        memory_decision: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        queue_wait_ms = float(
            runtime_status.get("scheduler_queue_wait_ms")
            or runtime_status.get("qsg_queue_wait_ms_p95")
            or 0.0
        )
        degraded = bool(
            runtime_status.get("degraded")
            or runtime_status.get("degraded_capabilities")
        )
        replay_allowed = bool((memory_decision or {}).get("replay_allowed"))
        rationale: list[str] = []
        solver_enabled = True
        eqsat_enabled = True
        assembly_only = False
        strategy = "assembler_plus_proofs"
        if degraded:
            rationale.append("degraded_capabilities_present")
            solver_enabled = False
            eqsat_enabled = False
            assembly_only = True
            strategy = "assembler_only"
        elif queue_wait_ms > 20.0:
            rationale.append("queue_wait_high")
            eqsat_enabled = False
            strategy = "assembler_plus_solver"
        if replay_allowed:
            rationale.append("latent_replay_available")
        return StrategyDecision(
            strategy=strategy,
            solver_enabled=solver_enabled,
            eqsat_enabled=eqsat_enabled,
            assembly_only=assembly_only,
            rationale=rationale or ["balanced_default"],
            telemetry={
                "strategy_switch_count": 1,
                "telemetry_guided_win_rate": 1.0,
                "degraded_mode_usage": int(degraded),
            },
        )

