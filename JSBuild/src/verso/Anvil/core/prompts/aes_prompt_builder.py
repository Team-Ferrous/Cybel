from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from core.aes import AALClassifier, AESRuleRegistry, DomainDetector
from core.token_budget import PromptContextBudgetPolicy, assemble_prompt_with_budget


class AESPromptBuilder:
    """Build AES-aware prompt payloads and parseable governance contracts."""

    VISUAL_PACK_VERSIONS = ("v1", "v2")

    REQUIRED_CONTRACT_KEYS = (
        "AAL_CLASSIFICATION",
        "APPLICABLE_RULE_IDS",
        "REQUIRED_ARTIFACTS",
        "BLOCKING_GATES",
        "AES_VISUALS_PACKS",
        "AES_VISUALS_SUMMARY",
    )

    CONTROL_PLANE_KEYS = (
        "TRACE_ID",
        "GRAPH_SNAPSHOT_ID",
        "POLICY_POSTURE",
        "TOOLCHAIN_STATE_VECTOR",
        "CHANGED_FILES",
        "RUNTIME_POSTURE",
    )

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        budget_policy: Optional[PromptContextBudgetPolicy] = None,
    ) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.standards_dir = self.repo_root / "standards"
        self.prompts_dir = self.repo_root / "prompts"
        self.visuals_dir = self.repo_root / "aes_visuals"
        self.legacy_visuals_dir = self.prompts_dir / "aes_visuals"
        self.budget_policy = budget_policy or PromptContextBudgetPolicy()
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.rule_registry = AESRuleRegistry()
        rules_path = self.standards_dir / "AES_RULES.json"
        if rules_path.exists():
            self.rule_registry.load(str(rules_path))

    def _read_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _resolve_visual_pack_source(self, version: str) -> tuple[Optional[Path], str]:
        candidates = (
            (
                self.visuals_dir / version / "PROMPT_GUIDANCE.md",
                "canonical_prompt_guidance",
            ),
            (
                self.visuals_dir / version / "directives.json",
                "canonical_directives_json",
            ),
            (
                self.legacy_visuals_dir / version / "PROMPT_GUIDANCE.md",
                "legacy_prompt_guidance",
            ),
            (
                self.legacy_visuals_dir / version / "directives.json",
                "legacy_directives_json",
            ),
        )
        for candidate, source_kind in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate, source_kind
        return None, "missing"

    @staticmethod
    def _relative_to_repo(path: Path, repo_root: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _truncate_summary(text: str, max_chars: int = 220) -> str:
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _summarize_visual_text_guidance(text: str, *, max_lines: int = 2) -> str:
        normalized_lines: list[str] = []
        for raw_line in text.splitlines():
            line = " ".join(raw_line.strip().split())
            if not line or line.startswith("#"):
                continue
            if line[0] in {"-", "*"}:
                line = line[1:].strip()
            if len(line) >= 2 and line[0].isdigit() and line[1] in {".", ")"}:
                line = line[2:].strip()
            if line:
                normalized_lines.append(line)
            if len(normalized_lines) >= max_lines:
                break
        if not normalized_lines:
            return "present but empty; fallback to AES condensed policy"
        summary = " | ".join(normalized_lines)
        return AESPromptBuilder._truncate_summary(summary)

    def _summarize_visual_json_guidance(self, path: Path, version: str) -> str:
        raw_text = self._read_text(path)
        if not raw_text:
            return "JSON directives pack present but empty; fallback to AES condensed policy"
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return "JSON directives pack unreadable; fallback to AES condensed policy"

        artifact = ""
        profile = ""
        directives_count = 0
        sample_ids: list[str] = []
        if isinstance(payload, dict):
            artifact = str(payload.get("artifact", "")).strip()
            profile = str(payload.get("profile", "")).strip()
            directives = payload.get("directives")
            if isinstance(directives, list):
                directives_count = len(directives)
                for directive in directives[:3]:
                    if isinstance(directive, dict):
                        directive_id = str(
                            directive.get("directive_id") or directive.get("id", "")
                        ).strip()
                        if directive_id:
                            sample_ids.append(directive_id)
        elif isinstance(payload, list):
            directives_count = len(payload)

        identity = ":".join(part for part in (artifact, profile or version) if part)
        if not identity:
            identity = f"aes_visuals:{version}"
        summary = f"JSON directives pack ({identity}) with {directives_count} directives"
        if sample_ids:
            summary = f"{summary}; sample ids: {', '.join(sample_ids)}"
        return self._truncate_summary(summary)

    def _aes_visual_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for version in self.VISUAL_PACK_VERSIONS:
            reference = f"aes_visuals/{version}"
            expected_path = f"aes_visuals/{version}/PROMPT_GUIDANCE.md"
            visual_path, source_kind = self._resolve_visual_pack_source(version)
            if visual_path:
                source = self._relative_to_repo(visual_path, self.repo_root)
                if visual_path.suffix.lower() == ".json":
                    summary = self._summarize_visual_json_guidance(visual_path, version)
                else:
                    summary = self._summarize_visual_text_guidance(
                        self._read_text(visual_path)
                    )
                status = (
                    "present_legacy_fallback"
                    if source_kind.startswith("legacy")
                    else "present"
                )
            else:
                source = expected_path
                summary = "missing; fallback to AES condensed and domain rules only"
                status = "missing"
            entries.append(
                {
                    "reference": reference,
                    "source": source,
                    "summary": summary,
                    "status": status,
                }
            )
        return entries

    def _aes_visual_contract_fields(self) -> tuple[list[str], str]:
        entries = self._aes_visual_entries()
        packs = [
            f"{entry['reference']}:{entry['status']}:{entry['source']}" for entry in entries
        ]
        summary = " || ".join(
            f"{entry['reference']}={entry['summary']}" for entry in entries
        )
        return packs, summary

    def _aes_visual_prompt_section(self) -> str:
        lines = ["AES Visual Guidance (bounded summaries):"]
        for entry in self._aes_visual_entries():
            lines.append(
                "- "
                f"{entry['reference']} ({entry['status']} @ {entry['source']}): "
                f"{entry['summary']}"
            )
        return "\n".join(lines)

    def _resolve_files(self, task_files: Optional[Iterable[str]]) -> list[str]:
        if not task_files:
            return []
        resolved: list[str] = []
        for item in task_files:
            if not item:
                continue
            path = Path(item)
            if path.exists():
                resolved.append(str(path))
        return sorted(set(resolved))

    def _collect_rule_ids(self, aal: str, domains: list[str]) -> list[str]:
        rules = list(self.rule_registry.get_rules_for_aal(aal))
        for domain in domains:
            rules.extend(self.rule_registry.get_rules_for_domain(domain))
        deduped = {rule.id for rule in rules}
        return sorted(deduped)

    @staticmethod
    def _required_artifacts_for_aal(aal: str) -> list[str]:
        if aal in {"AAL-0", "AAL-1"}:
            return [
                "traceability_record",
                "evidence_bundle",
                "review_signoff",
                "red_team_report",
            ]
        if aal == "AAL-2":
            return ["evidence_backed_output", "verification_summary"]
        return ["basic_verification_note"]

    @staticmethod
    def _blocking_gates_for_aal(aal: str) -> list[str]:
        gates = ["missing_prompt_contract"]
        if aal in {"AAL-0", "AAL-1"}:
            gates.extend(
                [
                    "missing_traceability_record",
                    "missing_evidence_bundle",
                    "missing_review_signoff",
                    "missing_red_team_report",
                ]
            )
        elif aal == "AAL-2":
            gates.append("missing_verification_summary")
        return gates

    def _domain_rule_sections(self, domains: list[str]) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        for domain in domains:
            path = self.standards_dir / "domain_rules" / f"{domain}.md"
            text = self._read_text(path)
            if text:
                sections.append((f"domain_rules/{domain}", text))
        return sections

    @staticmethod
    def _aal_checklist(aal: str) -> str:
        if aal in {"AAL-0", "AAL-1"}:
            return "\n".join(
                [
                    "High-AAL Checklist:",
                    "- produce traceability + evidence bundle IDs before synthesis",
                    "- include explicit verification and red-team closure status",
                    "- treat missing artifacts as blocking failures",
                ]
            )
        return "\n".join(
            [
                "Standard Checklist:",
                "- provide evidence-backed conclusions",
                "- include verification notes for modified behavior",
            ]
        )

    def _classify_context(
        self,
        task_text: str = "",
        task_files: Optional[Iterable[str]] = None,
    ) -> tuple[str, list[str], list[str]]:
        files = self._resolve_files(task_files)
        if files:
            aal = self.aal_classifier.classify_changeset(files)
            domains = sorted(self.domain_detector.detect_domains(files))
        elif task_text:
            aal = self.aal_classifier.classify_text(task_text)
            domains = []
        else:
            aal = "AAL-3"
            domains = []
        rule_ids = self._collect_rule_ids(aal, domains)
        return aal, domains, rule_ids

    def build_contract(
        self,
        role: str,
        task_text: str = "",
        task_files: Optional[Iterable[str]] = None,
        aal_override: Optional[str] = None,
        additional_rule_ids: Optional[Iterable[str]] = None,
        contract_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        aal, domains, rule_ids = self._classify_context(task_text, task_files)
        if aal_override:
            aal = aal_override
            rule_ids = self._collect_rule_ids(aal, domains)
        if additional_rule_ids:
            rule_ids = sorted(set(rule_ids).union({str(item) for item in additional_rule_ids}))
        visual_packs, visual_summary = self._aes_visual_contract_fields()
        contract = {
            "PROMPT_ROLE": role,
            "AAL_CLASSIFICATION": aal,
            "APPLICABLE_RULE_IDS": rule_ids,
            "REQUIRED_ARTIFACTS": self._required_artifacts_for_aal(aal),
            "BLOCKING_GATES": self._blocking_gates_for_aal(aal),
            "AES_VISUALS_PACKS": visual_packs,
            "AES_VISUALS_SUMMARY": visual_summary,
            "ACTIVE_DOMAINS": domains,
            "DROPPED_SECTIONS": [],
            "TOKEN_BUDGET": 0,
            "TOKEN_USAGE": 0,
        }
        contract.update(self._normalize_control_plane_context(contract_context))
        return contract

    def _normalize_control_plane_context(
        self,
        contract_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        context = dict(contract_context or {})
        toolchain_vector = context.get("toolchain_state_vector") or []
        if isinstance(toolchain_vector, dict):
            toolchain_vector = [toolchain_vector]
        normalized = {
            "TRACE_ID": str(context.get("trace_id") or "none"),
            "GRAPH_SNAPSHOT_ID": str(context.get("graph_snapshot_id") or "none"),
            "POLICY_POSTURE": str(context.get("policy_posture") or "unknown"),
            "TOOLCHAIN_STATE_VECTOR": [
                item
                for item in (
                    toolchain_vector if isinstance(toolchain_vector, list) else []
                )
                if item
            ],
            "CHANGED_FILES": [
                str(item)
                for item in list(context.get("changed_files") or [])
                if str(item).strip()
            ],
            "RUNTIME_POSTURE": str(context.get("runtime_posture") or "unknown"),
        }
        return normalized

    def validate_contract(self, contract: Dict[str, Any]) -> list[str]:
        missing: list[str] = []
        for key in self.REQUIRED_CONTRACT_KEYS:
            value = contract.get(key)
            if value is None:
                missing.append(key)
            elif isinstance(value, (list, tuple, set)) and not value:
                missing.append(key)
            elif isinstance(value, str) and not value.strip():
                missing.append(key)
        return missing

    def _finalize_contract(
        self,
        contract: Dict[str, Any],
        budget_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(contract)
        merged["TOKEN_BUDGET"] = budget_result["token_budget"]
        merged["TOKEN_USAGE"] = budget_result["token_usage"]
        merged["DROPPED_SECTIONS"] = budget_result["dropped_sections"]
        return merged

    @staticmethod
    def _value_as_scalar(value: Any) -> str:
        if isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value if str(item).strip()]
            return ",".join(values) if values else "none"
        if value is None:
            return "none"
        text = str(value).strip()
        return text or "none"

    def render_contract_block(self, contract: Dict[str, Any]) -> str:
        lines = ["<AES_PROMPT_CONTRACT>"]
        ordered_keys = [
            "PROMPT_ROLE",
            "AAL_CLASSIFICATION",
            "APPLICABLE_RULE_IDS",
            "REQUIRED_ARTIFACTS",
            "BLOCKING_GATES",
            "AES_VISUALS_PACKS",
            "AES_VISUALS_SUMMARY",
            "TRACE_ID",
            "GRAPH_SNAPSHOT_ID",
            "POLICY_POSTURE",
            "TOOLCHAIN_STATE_VECTOR",
            "CHANGED_FILES",
            "RUNTIME_POSTURE",
            "ACTIVE_DOMAINS",
            "TOKEN_BUDGET",
            "TOKEN_USAGE",
            "DROPPED_SECTIONS",
        ]
        for key in ordered_keys:
            if key not in contract:
                continue
            lines.append(f"{key}={self._value_as_scalar(contract.get(key))}")
        lines.append("</AES_PROMPT_CONTRACT>")
        return "\n".join(lines)

    def build_master_prompt(
        self,
        task_text: str = "",
        task_files: Optional[Iterable[str]] = None,
        contract_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        contract = self.build_contract(
            role="master",
            task_text=task_text,
            task_files=task_files,
            contract_context=contract_context,
        )
        sections = [
            ("aes_condensed", self._read_text(self.standards_dir / "AES_CONDENSED.md")),
            ("aes_visuals", self._aes_visual_prompt_section()),
            (
                "master_orchestration",
                "\n".join(
                    [
                        "Master Prompt Governance:",
                        "- keep orchestration deterministic and evidence-first",
                        "- enforce blocking gates when required artifacts are absent",
                        "- route deep domain work to subagents with scoped rule packs",
                    ]
                ),
            ),
        ]
        budget_result = assemble_prompt_with_budget(
            sections=sections,
            token_budget=self.budget_policy.master_prompt_tokens,
            reserve_tokens=self.budget_policy.response_reserve_tokens,
        )
        return budget_result["text"], self._finalize_contract(contract, budget_result)

    def build_subagent_prompt(
        self,
        role: str,
        task_files: Optional[Iterable[str]] = None,
        task_text: str = "",
        contract_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        contract = self.build_contract(
            role=f"subagent:{role}",
            task_text=task_text,
            task_files=task_files,
            contract_context=contract_context,
        )
        domains = contract.get("ACTIVE_DOMAINS", [])
        aal = contract.get("AAL_CLASSIFICATION", "AAL-3")
        sections = [
            ("aes_condensed", self._read_text(self.standards_dir / "AES_CONDENSED.md")),
            ("aes_visuals", self._aes_visual_prompt_section()),
            *self._domain_rule_sections(domains),
            ("aal_checklist", self._aal_checklist(aal)),
        ]
        budget_result = assemble_prompt_with_budget(
            sections=sections,
            token_budget=self.budget_policy.subagent_prompt_tokens,
            reserve_tokens=self.budget_policy.response_reserve_tokens,
        )
        return budget_result["text"], self._finalize_contract(contract, budget_result)

    def build_verification_prompt(
        self,
        aal: str,
        violations: Optional[Iterable[str]] = None,
        contract_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Any]]:
        normalized_violations = [str(item).strip() for item in (violations or []) if str(item).strip()]
        contract = self.build_contract(
            role="verification",
            aal_override=aal,
            additional_rule_ids=["AES-ARCH-1"],
            contract_context=contract_context,
        )
        violation_lines = "\n".join(f"- {item}" for item in normalized_violations[:50]) or "- none"
        sections = [
            ("aes_condensed", self._read_text(self.standards_dir / "AES_CONDENSED.md")),
            ("aes_visuals", self._aes_visual_prompt_section()),
            ("aal_checklist", self._aal_checklist(aal)),
            ("violations", f"Violations Under Review:\n{violation_lines}"),
        ]
        budget_result = assemble_prompt_with_budget(
            sections=sections,
            token_budget=self.budget_policy.verification_prompt_tokens,
            reserve_tokens=0,
        )
        return budget_result["text"], self._finalize_contract(contract, budget_result)
