from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .spec import SagSpec


@dataclass(slots=True)
class ForbiddenFlow:
    source: str
    sink: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class BudgetConstraint:
    name: str
    limit: float
    current: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(slots=True)
class EffectEvaluation:
    allowed: bool
    blockers: list[str] = field(default_factory=list)
    forbidden_flows: list[ForbiddenFlow] = field(default_factory=list)
    budgets: list[BudgetConstraint] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blockers": list(self.blockers),
            "forbidden_flows": [item.as_dict() for item in self.forbidden_flows],
            "budgets": [item.as_dict() for item in self.budgets],
            "telemetry": dict(self.telemetry),
        }


class SynthesisEffectEngine:
    """Evaluate effect and resource constraints before synthesis execution."""

    _FORBIDDEN_PHRASES = {
        "skip verification": "sanctioned_verification_bypass",
        "bypass verification": "sanctioned_verification_bypass",
        "os.system": "unsanctioned_subprocess_use",
        "subprocess.run": "unsanctioned_subprocess_use",
    }

    def evaluate_spec(
        self,
        spec: SagSpec | dict[str, Any],
        *,
        runtime_status: dict[str, Any] | None = None,
        queue_wait_budget_ms: float = 20.0,
    ) -> EffectEvaluation:
        normalized = spec if isinstance(spec, SagSpec) else SagSpec.from_dict(spec)
        blockers: list[str] = []
        flows: list[ForbiddenFlow] = []
        objective = normalized.objective.lower()
        for needle, reason in self._FORBIDDEN_PHRASES.items():
            if needle in objective:
                blockers.append(reason)
                flows.append(
                    ForbiddenFlow(
                        source="objective",
                        sink=needle,
                        reason=reason,
                    )
                )
        verification_commands = list(normalized.verification.commands)
        sanctioned = any(
            "saguaro verify" in command or command.startswith("pytest ")
            for command in verification_commands
        )
        if not sanctioned:
            blockers.append("missing_sanctioned_verification_path")
        budgets: list[BudgetConstraint] = []
        queue_wait_ms = float(
            (runtime_status or {}).get("scheduler_queue_wait_ms")
            or (runtime_status or {}).get("qsg_queue_wait_ms_p95")
            or 0.0
        )
        budgets.append(
            BudgetConstraint(
                name="scheduler_queue_wait_ms",
                limit=queue_wait_budget_ms,
                current=queue_wait_ms,
            )
        )
        if queue_wait_ms > queue_wait_budget_ms and normalized.verification.proofs_required:
            blockers.append("resource_budget_reject_count")
        telemetry = {
            "forbidden_flow_block_count": len(flows),
            "resource_budget_reject_count": int(queue_wait_ms > queue_wait_budget_ms),
            "effect_inference_coverage": 1.0 if verification_commands else 0.5,
        }
        return EffectEvaluation(
            allowed=not blockers,
            blockers=blockers,
            forbidden_flows=flows,
            budgets=budgets,
            telemetry=telemetry,
        )

