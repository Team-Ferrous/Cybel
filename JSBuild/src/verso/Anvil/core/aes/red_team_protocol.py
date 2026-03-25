from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class RedTeamValidationResult:
    required: bool
    passed: bool
    missing_artifacts: List[str]
    unresolved_critical_findings: List[str]


class RedTeamProtocol:
    """Validate required red-team closure artifacts for high-assurance work."""

    REQUIRED_ARTIFACTS = (
        "fmea.json",
        "fta_paths.json",
        "cwe_mapping.json",
        "residual_risk.md",
    )

    HIGH_AAL = {"AAL-0", "AAL-1"}

    def required_artifacts(self, aal: str, red_team_required: bool) -> List[str]:
        normalized = str(aal or "AAL-3").upper()
        if normalized in self.HIGH_AAL or red_team_required:
            return list(self.REQUIRED_ARTIFACTS)
        return []

    def validate(
        self,
        artifacts: Dict[str, Any] | None,
        aal: str,
        red_team_required: bool,
    ) -> RedTeamValidationResult:
        artifact_map = dict(artifacts or {})
        required = self.required_artifacts(aal, red_team_required)
        missing = [name for name in required if not artifact_map.get(name)]
        unresolved = self._find_unresolved_critical(artifact_map)
        passed = not missing and not unresolved
        return RedTeamValidationResult(
            required=bool(required),
            passed=passed,
            missing_artifacts=missing,
            unresolved_critical_findings=unresolved,
        )

    def build_placeholder_bundle(
        self,
        aal: str,
        red_team_required: bool,
    ) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "status": "missing",
                "required": True,
                "aal": str(aal or "AAL-3").upper(),
            }
            for name in self.required_artifacts(aal, red_team_required)
        }

    def _find_unresolved_critical(
        self,
        artifacts: Dict[str, Any],
    ) -> List[str]:
        unresolved: List[str] = []
        for name in self.REQUIRED_ARTIFACTS:
            value = artifacts.get(name)
            if value is None:
                continue
            if isinstance(value, dict):
                critical_open = (
                    value.get("critical_open")
                    or value.get("critical_findings_open")
                    or value.get("unresolved_critical")
                )
                if isinstance(critical_open, (int, float)) and critical_open > 0:
                    unresolved.append(f"{name}: {int(critical_open)} critical open")
            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
                items = list(value)
                for idx, item in enumerate(items):
                    if isinstance(item, dict) and str(item.get("severity", "")).upper() in {
                        "P0",
                        "P1",
                        "CRITICAL",
                    } and not item.get("resolved"):
                        unresolved.append(f"{name}[{idx}] unresolved critical finding")
        return unresolved
