from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.aes.compliance_context import ComplianceContext


@dataclass(frozen=True)
class ObligationResult:
    required_rule_ids: list[str]
    required_runtime_gates: list[str]
    required_artifacts: list[str]
    matched_obligations: list[str]
    thresholds: dict[str, Any]


class ObligationEngine:
    def __init__(self, obligations_path: str = "standards/AES_OBLIGATIONS.json") -> None:
        self.obligations_path = Path(obligations_path)
        self.payload = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.obligations_path.exists():
            return {
                "defaults": {},
                "thresholds": {},
                "obligations": [],
            }
        return json.loads(self.obligations_path.read_text(encoding="utf-8"))

    def evaluate(self, context: ComplianceContext) -> ObligationResult:
        required_rule_ids: list[str] = []
        required_runtime_gates: list[str] = []
        required_artifacts: list[str] = []
        matched: list[str] = []

        defaults = self.payload.get("defaults", {}) or {}
        required_rule_ids.extend(defaults.get("required_rule_ids", []) or [])
        required_runtime_gates.extend(defaults.get("required_runtime_gates", []) or [])
        required_artifacts.extend(defaults.get("required_artifacts", []) or [])

        for entry in self.payload.get("obligations", []) or []:
            when = entry.get("when", {}) or {}
            if not self._matches(when, context):
                continue
            matched.append(str(entry.get("id", "unknown")))
            required_rule_ids.extend(entry.get("required_rule_ids", []) or [])
            required_runtime_gates.extend(entry.get("required_runtime_gates", []) or [])
            required_artifacts.extend(entry.get("required_artifacts", []) or [])

        return ObligationResult(
            required_rule_ids=list(dict.fromkeys(str(item) for item in required_rule_ids)),
            required_runtime_gates=list(
                dict.fromkeys(str(item) for item in required_runtime_gates)
            ),
            required_artifacts=list(dict.fromkeys(str(item) for item in required_artifacts)),
            matched_obligations=matched,
            thresholds=self.payload.get("thresholds", {}) or {},
        )

    def _matches(self, condition: dict[str, Any], context: ComplianceContext) -> bool:
        aal = str(context.aal).upper()
        if "aal" in condition:
            allowed = {str(item).upper() for item in condition.get("aal", []) or []}
            if allowed and aal not in allowed:
                return False

        if condition.get("hot_path") is True and not context.hot_paths:
            return False

        if condition.get("dependency_changes") is True and not context.dependency_changes:
            return False

        required_domains = {str(item) for item in condition.get("domains", []) or []}
        if required_domains and required_domains.isdisjoint(set(context.domains)):
            return False

        return True

