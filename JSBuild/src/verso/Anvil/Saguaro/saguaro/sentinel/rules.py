"""SAGUARO rule loading for Sentinel engines."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def _derive_closure_level(severity: str) -> str:
    sev = (severity or "").upper()
    if sev in {"P0", "P1", "AAL-0", "AAL-1", "ERROR"}:
        return "blocking"
    if sev in {"P2", "AAL-2", "WARN", "WARNING"}:
        return "guarded"
    return "advisory"


def _derive_requires_artifact(rule_id: str, message: str) -> bool:
    rid = (rule_id or "").upper()
    text = (message or "").lower()
    return rid.startswith(("AES-TR", "AES-REV")) or any(
        token in text for token in ("trace", "evidence", "waiver", "review")
    )


@dataclass
class Rule:
    """Provide Rule support."""
    id: str
    pattern: str | None
    message: str
    severity: str = "P2"
    scope: str = "**"
    replacement: str | None = None
    engine: str = "native"
    enforcement_kind: str = "native"
    requires_artifact: bool = False
    closure_level: str = "guarded"
    domain: list[str] = field(default_factory=lambda: ["universal"])
    aal: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    negative_lookahead: str | None = None

    def check(self, content: str) -> list[tuple[int, str]]:
        """Return `(line_number, line_text)` matches for regex-backed rules."""
        if not self.pattern:
            return []

        violations: list[tuple[int, str]] = []
        try:
            regex = re.compile(self.pattern)
            neg_regex = (
                re.compile(self.negative_lookahead) if self.negative_lookahead else None
            )
            for line_no, line in enumerate(content.splitlines(), start=1):
                if not regex.search(line):
                    continue
                if neg_regex and neg_regex.search(line):
                    continue
                violations.append((line_no, line.strip()))
        except re.error as exc:
            logger.error("Invalid regex for rule %s: %s", self.id, exc)

        return violations


class RuleLoader:
    """Load rules from structured AES registry and legacy regex files."""

    @staticmethod
    def _legacy_compat_enabled() -> bool:
        value = os.environ.get("SAGUARO_ALLOW_LEGACY_RULES", "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _load_authoritative_rules(repo_path: str) -> list[Rule]:
        rules_path = os.path.join(repo_path, "standards", "AES_RULES.json")
        if not os.path.exists(rules_path):
            return []

        try:
            data = json.loads(open(rules_path, encoding="utf-8").read())
        except Exception as exc:
            logger.error("Failed to load %s: %s", rules_path, exc)
            return []

        rules: list[Rule] = []
        for item in data:
            rule_id = str(item.get("id", "unknown"))
            severity = str(item.get("severity", "P2"))
            message = str(item.get("text", ""))
            rules.append(
                Rule(
                    id=rule_id,
                    pattern=item.get("pattern"),
                    message=message,
                    severity=severity,
                    scope=item.get("scope", "**"),
                    replacement=item.get("replacement"),
                    engine=item.get("engine", "native"),
                    enforcement_kind=item.get(
                        "enforcement_kind", item.get("engine", "native")
                    ),
                    requires_artifact=bool(
                        item.get(
                            "requires_artifact",
                            _derive_requires_artifact(rule_id, message),
                        )
                    ),
                    closure_level=item.get(
                        "closure_level", _derive_closure_level(severity)
                    ),
                    domain=list(item.get("domain") or ["universal"]),
                    aal=severity if severity.startswith("AAL-") else item.get("aal"),
                    evidence_refs=list(item.get("evidence_refs") or []),
                    negative_lookahead=item.get("negative_lookahead"),
                )
            )
        return rules

    @staticmethod
    def _load_legacy_rules(repo_path: str) -> list[Rule]:
        rules_path = os.path.join(repo_path, ".saguaro.rules")
        if not os.path.exists(rules_path):
            return []

        try:
            with open(rules_path, encoding="utf-8") as handle:
                data: Any = yaml.safe_load(handle) or []
        except Exception as exc:
            logger.error("Failed to load %s: %s", rules_path, exc)
            return []

        raw_rules = data.get("rules", []) if isinstance(data, dict) else data
        rules: list[Rule] = []
        for item in raw_rules:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("id", "unknown"))
            severity = str(item.get("severity", "P2"))
            message = str(item.get("message", ""))
            rules.append(
                Rule(
                    id=rule_id,
                    pattern=item.get("pattern"),
                    message=message,
                    severity=severity,
                    scope=item.get("scope", item.get("file_pattern", "**")),
                    replacement=item.get("replacement"),
                    engine=item.get("engine", "native"),
                    enforcement_kind=item.get(
                        "enforcement_kind", item.get("engine", "native")
                    ),
                    requires_artifact=bool(
                        item.get(
                            "requires_artifact",
                            _derive_requires_artifact(rule_id, message),
                        )
                    ),
                    closure_level=item.get(
                        "closure_level", _derive_closure_level(severity)
                    ),
                    domain=list(item.get("domain") or ["universal"]),
                    aal=item.get("aal"),
                    evidence_refs=list(item.get("evidence_refs") or []),
                    negative_lookahead=item.get("negative_lookahead"),
                )
            )
        return rules

    @staticmethod
    def load(repo_path: str) -> list[Rule]:
        """Load rules with authoritative AES registry precedence."""
        structured_rules = RuleLoader._load_authoritative_rules(repo_path)
        legacy_rules = RuleLoader._load_legacy_rules(repo_path)
        legacy_compat = RuleLoader._legacy_compat_enabled()

        if structured_rules:
            if legacy_rules and not legacy_compat:
                logger.info(
                    "Ignoring legacy .saguaro.rules because authoritative "
                    "standards/AES_RULES.json is present."
                )
                return structured_rules

            deduped: list[Rule] = []
            seen_ids: set[str] = set()
            for rule in [*structured_rules, *legacy_rules]:
                if rule.id in seen_ids:
                    continue
                deduped.append(rule)
                seen_ids.add(rule.id)
            return deduped

        if legacy_rules and legacy_compat:
            logger.warning(
                "Using legacy .saguaro.rules via SAGUARO_ALLOW_LEGACY_RULES "
                "compatibility mode. Migrate to standards/AES_RULES.json."
            )
            return legacy_rules

        if legacy_rules and not legacy_compat:
            logger.error(
                "Legacy .saguaro.rules detected but ignored in strict mode. "
                "Set SAGUARO_ALLOW_LEGACY_RULES=1 only for temporary compatibility."
            )
            return []

        if not structured_rules and not legacy_rules:
            logger.info(
                "No Sentinel rules found at standards/AES_RULES.json or .saguaro.rules."
            )
            return []
        return []
