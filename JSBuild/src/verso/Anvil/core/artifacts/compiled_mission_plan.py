"""Typed mission plan persisted separately from free-form prompt text."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MissionPlanStep:
    """One executable step in a compiled mission plan."""

    step_id: str
    tool: str
    arguments: dict[str, Any]
    target_files: list[str] = field(default_factory=list)
    status: str = "pending"
    depends_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "arguments": dict(self.arguments),
            "target_files": list(self.target_files),
            "status": self.status,
            "depends_on": list(self.depends_on),
        }


@dataclass
class CompiledMissionPlan:
    """Typed mission plan artifact for replay and resume."""

    run_id: str
    trace_id: str
    task: str
    prompt_plan_hash: str
    created_at: str = field(default_factory=_now_iso)
    steps: list[MissionPlanStep] = field(default_factory=list)
    plan_outline: list[str] = field(default_factory=list)
    thread_context: dict[str, Any] = field(default_factory=dict)
    runtime_control: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def _infer_target_files(arguments: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for key in ("file_path", "path", "target_file", "source", "destination"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        return list(dict.fromkeys(candidates))

    @classmethod
    def from_action_plan(
        cls,
        *,
        run_id: str,
        trace_id: str,
        task: str,
        action_plan: str,
        tool_calls: list[dict[str, Any]],
        thread_context: dict[str, Any] | None = None,
        runtime_control: dict[str, Any] | None = None,
    ) -> "CompiledMissionPlan":
        digest = hashlib.sha256(action_plan.encode("utf-8")).hexdigest()
        outline = [
            line.strip()
            for line in action_plan.splitlines()
            if line.strip() and line.lstrip()[:2] in {"1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."}
        ]
        steps: list[MissionPlanStep] = []
        previous_step_id: str | None = None
        for index, tool_call in enumerate(tool_calls, start=1):
            tool = str(tool_call.get("tool") or "unknown")
            arguments = dict(tool_call.get("args") or {})
            step_id = f"step-{index:03d}"
            depends_on = [previous_step_id] if previous_step_id else []
            steps.append(
                MissionPlanStep(
                    step_id=step_id,
                    tool=tool,
                    arguments=arguments,
                    target_files=cls._infer_target_files(arguments),
                    depends_on=depends_on,
                )
            )
            previous_step_id = step_id
        return cls(
            run_id=run_id,
            trace_id=trace_id,
            task=task,
            prompt_plan_hash=digest,
            steps=steps,
            plan_outline=outline,
            thread_context=dict(thread_context or {}),
            runtime_control=dict(runtime_control or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trace_id": self.trace_id,
            "task": self.task,
            "prompt_plan_hash": self.prompt_plan_hash,
            "created_at": self.created_at,
            "step_count": len(self.steps),
            "plan_outline": list(self.plan_outline),
            "thread_context": dict(self.thread_context),
            "runtime_control": dict(self.runtime_control),
            "steps": [step.to_dict() for step in self.steps],
        }

    def save(self, path: str | Path) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return str(output)
