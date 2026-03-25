import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

SEVERITY_ORDER = ("AAL-0", "AAL-1", "AAL-2", "AAL-3")


@dataclass(frozen=True)
class AESRule:
    id: str
    section: str
    text: str
    severity: str
    engine: str
    auto_fixable: bool
    domains: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    check_function: Optional[str] = None
    pattern: Optional[str] = None
    negative_lookahead: Optional[str] = None
    cwe: Optional[list[str]] = None
    svl_min: Optional[str] = None
    title: str = ""
    source_version: str = "v2"
    source_refs: list[str] = field(default_factory=list)
    precedence: int = 100
    selectors: dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "static"
    parameters: dict[str, Any] = field(default_factory=dict)
    required_artifacts: list[str] = field(default_factory=list)
    waiverable: bool = False
    rollout_stage: str = "ratchet"
    status: str = "advisory"
    section_title: str = ""
    fix_strategy: str = ""
    fix_handler: str = ""
    fix_safety: str = ""
    fix_confidence: float = 0.0
    preferred_tool: str = ""
    suppression_policy: str = ""
    dedupe_group: str = ""
    preferred_disposition: str = ""

    @property
    def domain(self) -> list[str]:
        return list(self.domains)

    @property
    def language(self) -> list[str]:
        return list(self.languages)


class AESRuleRegistry:
    """Loads and queries the machine-readable AES rule registry."""

    def __init__(self) -> None:
        self._rules: list[AESRule] = []

    def load(self, rules_path: str = "standards/AES_RULES.json") -> None:
        data = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self._rules = [self._coerce_rule(entry) for entry in data]

    def _coerce_rule(self, entry: dict[str, Any]) -> AESRule:
        payload = dict(entry)
        payload["domains"] = list(
            payload.get("domains", payload.get("domain", []) or [])
        )
        payload["languages"] = list(
            payload.get("languages", payload.get("language", []) or [])
        )
        payload.pop("domain", None)
        payload.pop("language", None)
        payload.setdefault("title", payload.get("text", payload.get("id", "")))
        payload.setdefault("source_version", "v2")
        payload.setdefault("source_refs", [])
        payload.setdefault("precedence", 100)
        payload.setdefault("selectors", {})
        payload.setdefault("execution_mode", "static")
        payload.setdefault("parameters", {})
        payload.setdefault("required_artifacts", [])
        payload.setdefault("waiverable", False)
        payload.setdefault("rollout_stage", "ratchet")
        payload.setdefault("status", "")
        payload.setdefault("section_title", "")
        payload.setdefault("fix_strategy", "")
        payload.setdefault("fix_handler", "")
        payload.setdefault("fix_safety", "")
        payload.setdefault("fix_confidence", 0.0)
        payload.setdefault("preferred_tool", "")
        payload.setdefault("suppression_policy", "")
        payload.setdefault("dedupe_group", "")
        payload.setdefault("preferred_disposition", "")
        return AESRule(**payload)

    @property
    def rules(self) -> list[AESRule]:
        return list(self._rules)

    def get_rules_for_domain(self, domain: str) -> list[AESRule]:
        return [rule for rule in self._rules if domain in rule.domains]

    def get_rules_for_aal(self, aal: str) -> list[AESRule]:
        threshold = SEVERITY_ORDER.index(aal)
        return [
            rule for rule in self._rules if SEVERITY_ORDER.index(rule.severity) <= threshold
        ]

    def get_rules_for_engine(self, engine: str) -> list[AESRule]:
        return [rule for rule in self._rules if rule.engine == engine]

    def get_rule(self, rule_id: str) -> Optional[AESRule]:
        for rule in self._rules:
            if rule.id == rule_id:
                return rule
        return None

    def get_rules_for_selector(self, **selectors: Any) -> list[AESRule]:
        if not selectors:
            return list(self._rules)
        matches: list[AESRule] = []
        for rule in self._rules:
            rule_selectors = rule.selectors or {}
            include = True
            for key, value in selectors.items():
                expected = rule_selectors.get(key)
                if expected is None:
                    continue
                if isinstance(expected, list):
                    if value not in expected:
                        include = False
                        break
                elif expected != value:
                    include = False
                    break
            if include:
                matches.append(rule)
        return matches

    def get_artifact_rules(self) -> list[AESRule]:
        return [rule for rule in self._rules if rule.execution_mode == "artifact"]

    def get_runtime_gate_rules(self) -> list[AESRule]:
        return [rule for rule in self._rules if rule.execution_mode == "runtime_gate"]

    def get_blocking_rules_for_rollout(self, rollout_stage: str = "ratchet") -> list[AESRule]:
        return [
            rule
            for rule in self._rules
            if rule.rollout_stage == rollout_stage and rule.status == "blocking"
        ]

    def get_check_function(self, rule_id: str) -> Optional[Callable[..., Any]]:
        rule = self.get_rule(rule_id)
        if rule is None or not rule.check_function:
            return None
        module_name, function_name = rule.check_function.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
