from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .spec import SagSpec


@dataclass(slots=True)
class SpecLintIssue:
    severity: str
    code: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(slots=True)
class SpecLintResult:
    is_valid: bool
    completeness_score: float
    issues: list[SpecLintIssue] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[SpecLintIssue]:
        return [item for item in self.issues if item.severity == "error"]

    @property
    def warnings(self) -> list[SpecLintIssue]:
        return [item for item in self.issues if item.severity == "warning"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "completeness_score": self.completeness_score,
            "issues": [item.as_dict() for item in self.issues],
            "telemetry": dict(self.telemetry),
        }


def lint_sagspec(spec: SagSpec | dict[str, Any]) -> SpecLintResult:
    normalized = spec if isinstance(spec, SagSpec) else SagSpec.from_dict(dict(spec or {}))
    issues: list[SpecLintIssue] = []
    if not normalized.objective.strip():
        issues.append(SpecLintIssue("error", "missing_objective", "Objective is required"))
    if not normalized.target_files:
        issues.append(SpecLintIssue("error", "missing_target_files", "At least one target file is required"))
    if not normalized.outputs:
        issues.append(SpecLintIssue("error", "missing_outputs", "At least one typed output is required"))
    if not normalized.verification.commands:
        issues.append(
            SpecLintIssue(
                "error",
                "missing_verification",
                "At least one verification command is required",
            )
        )
    if not normalized.inputs:
        issues.append(SpecLintIssue("warning", "missing_inputs", "Inputs were inferred implicitly"))
    if not normalized.constraints:
        issues.append(
            SpecLintIssue(
                "warning",
                "missing_constraints",
                "No explicit constraints declared; deterministic proof surface is weaker",
            )
        )
    completeness = normalized.completeness_score()
    telemetry = {
        "constraint_count": len(normalized.constraints),
        "missing_field_count": len([item for item in issues if item.severity == "error"]),
        "warning_count": len([item for item in issues if item.severity == "warning"]),
    }
    return SpecLintResult(
        is_valid=not any(item.severity == "error" for item in issues),
        completeness_score=completeness,
        issues=issues,
        telemetry=telemetry,
    )
