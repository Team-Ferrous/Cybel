"""Utilities for aes."""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from saguaro.architecture import ArchitectureAnalyzer


def _bootstrap_repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "core" / "aes").is_dir():
            return candidate
    return Path(__file__).resolve().parents[4]


try:
    from core.aes import (
        AALClassifier,
        AESRuleRegistry,
        ComplianceContext,
        DomainDetector,
        ObligationEngine,
        RuntimeGateRunner,
    )
except ModuleNotFoundError:  # pragma: no cover - runtime bootstrap for CLI entrypoints
    repo_root = _bootstrap_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from core.aes import (
        AALClassifier,
        AESRuleRegistry,
        ComplianceContext,
        DomainDetector,
        ObligationEngine,
        RuntimeGateRunner,
    )

from .base import BaseEngine

logger = logging.getLogger(__name__)

_AAL_ORDER = {"AAL-0": 0, "AAL-1": 1, "AAL-2": 2, "AAL-3": 3}
_EXT_TO_LANG = {
    ".py": "python",
    ".pyi": "python",
    ".c": "c",
    ".cc": "c++",
    ".cpp": "c++",
    ".cxx": "c++",
    ".h": "c++",
    ".hpp": "c++",
    ".md": "md",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".txt": "txt",
}
_TEXT_EXTENSIONS = set(_EXT_TO_LANG)
_NON_EXECUTABLE_ENGINES = {"human"}
_GOVERNED_DOC_PREFIXES = ("standards/", "prompts/", "aes_visuals/")
_GOVERNED_DOC_NAMES = {"AGENTS.md", "GEMINI.md"}
_DEFAULT_EXCLUDED_PATHS = [
    ".anvil/",
    ".saguaro/",
    "build/",
    "dist/",
    "venv/",
    "__pycache__/",
    "Saguaro/",
    "core/native/build/",
    "saguaro/native/build_release/",
    "saguaro/native/build_test/",
]
_GATE_TO_RULE_IDS = {
    "traceability_gate": ("AES-TR-1",),
    "evidence_closure_gate": ("AES-TR-2",),
    "review_independence_gate": ("AES-REV-1",),
    "waiver_validity_gate": ("AES-TR-3",),
    "chronicle_gate": ("AES-OBS-1",),
    "telemetry_contract_gate": ("AES-OBS-2",),
    "supply_chain_gate": ("AES-SUP-3", "AES-SUP-4"),
    "domain_report_gate": ("AES-ML-5", "AES-HPC-4", "AES-QC-4", "AES-PHYS-1"),
}


def _severity_to_p(severity: str) -> str:
    normalized = (severity or "").upper()
    if normalized in {"P0", "AAL-0"}:
        return "P0"
    if normalized in {"P1", "AAL-1", "ERROR"}:
        return "P1"
    if normalized in {"P2", "AAL-2", "WARN", "WARNING"}:
        return "P2"
    if normalized in {"P3", "AAL-3", "INFO"}:
        return "P3"
    return "P2"


def _closure_from_p(severity: str, aal: str) -> str:
    if severity in {"P0", "P1"}:
        return "blocking"
    if severity == "P2":
        return "guarded" if aal in {"AAL-0", "AAL-1"} else "advisory"
    return "advisory"


class AESEngine(BaseEngine):
    """Structured AES enforcement engine backed by standards/AES_RULES.json."""

    def __init__(self, repo_path: str) -> None:
        """Initialize the instance."""
        super().__init__(repo_path)
        self.registry = AESRuleRegistry()
        rules_path = os.path.join(repo_path, "standards", "AES_RULES.json")
        self.registry.load(rules_path)
        self.obligations = ObligationEngine(
            os.path.join(repo_path, "standards", "AES_OBLIGATIONS.json")
        )
        self.runtime_gate_runner = RuntimeGateRunner(repo_path)
        self.classifier = AALClassifier()
        self.domain_detector = DomainDetector()

    def run(self, path_arg: str = ".") -> list[dict[str, Any]]:
        """Handle run."""
        files = self._collect_target_files(path_arg)
        verify_ctx = dict(self.policy_config.get("verify_context", {}))
        aal_filter = self._parse_aal_filter(verify_ctx.get("aal"))
        domain_filter = self._parse_domain_filter(verify_ctx.get("domain"))
        change_manifest = self._load_change_manifest(
            verify_ctx.get("change_manifest_path")
        )
        compliance_context = self._hydrate_compliance_context(
            verify_ctx.get("compliance_context"),
            change_manifest=change_manifest,
        )
        governed_context = self._should_run_governance_checks(
            verify_ctx=verify_ctx,
            compliance_context=compliance_context,
            change_manifest=change_manifest,
        )
        obligation_result = (
            self.obligations.evaluate(compliance_context) if governed_context else None
        )

        violations: list[dict[str, Any]] = []
        file_aal_map: dict[str, str] = {}
        strictest_seen = "AAL-3"

        for file_path in files:
            rel_path = os.path.relpath(file_path, self.repo_path).replace("\\", "/")
            language = _EXT_TO_LANG.get(Path(file_path).suffix.lower(), "unknown")

            try:
                source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            file_aal = self.classifier.classify_file(file_path)
            file_svl = AALClassifier.map_aal_to_svl(file_aal)
            file_aal_map[rel_path] = file_aal
            if _AAL_ORDER[file_aal] < _AAL_ORDER[strictest_seen]:
                strictest_seen = file_aal
            domains = self.domain_detector.detect_domains([file_path]) or {"universal"}

            for rule in self._ordered_rules():
                if not self._rule_matches_engine(getattr(rule, "engine", "")):
                    continue
                if not self._rule_matches_aal(rule.severity, aal_filter):
                    continue
                if not self._rule_matches_language(rule.language, language):
                    continue
                if not self._rule_matches_domain(rule.domain, domains, domain_filter):
                    continue

                execution_mode = str(
                    getattr(rule, "execution_mode", "static") or "static"
                )
                check_function = self.registry.get_check_function(rule.id)
                if execution_mode in {"static", "artifact"} and check_function:
                    violations.extend(
                        self._run_check_function(
                            check_function=check_function,
                            rule=rule,
                            rel_path=rel_path,
                            file_aal=file_aal,
                            file_svl=file_svl.value,
                            domains=domains,
                            source=source,
                        )
                    )
                if execution_mode == "static" and getattr(rule, "pattern", None):
                    violations.extend(
                        self._run_pattern_check(
                            rule=rule,
                            rel_path=rel_path,
                            file_aal=file_aal,
                            file_svl=file_svl.value,
                            domains=domains,
                            source=source,
                        )
                    )

        if not files:
            strictest_seen = str(compliance_context.aal or strictest_seen)
        violations.extend(
            self._artifact_checks(
                strictest_seen=strictest_seen,
                verify_ctx=verify_ctx,
                file_aal_map=file_aal_map,
                compliance_context=compliance_context,
                change_manifest=change_manifest,
            )
        )
        violations.extend(
            self._execution_mode_rule_checks(
                strictest_seen=strictest_seen,
                verify_ctx=verify_ctx,
                file_aal_map=file_aal_map,
                compliance_context=compliance_context,
                change_manifest=change_manifest,
                obligation_result=obligation_result,
            )
        )
        violations.extend(self._prompt_policy_checks(strictest_seen=strictest_seen))
        violations.extend(self._architecture_findings(path_arg))
        return self._sorted_unique_violations(violations)

    def _collect_target_files(self, path_arg: str) -> list[str]:
        target_abs = (
            path_arg
            if os.path.isabs(path_arg)
            else os.path.abspath(os.path.join(self.repo_path, path_arg))
        )
        repo_abs = os.path.abspath(self.repo_path)
        if not target_abs.startswith(os.path.abspath(self.repo_path)):
            return []
        if self._is_excluded_path(target_abs):
            return []
        default_repo_target = target_abs == repo_abs

        if os.path.isfile(target_abs):
            suffix = Path(target_abs).suffix.lower()
            rel = os.path.relpath(target_abs, self.repo_path).replace("\\", "/")
            return (
                [target_abs]
                if self._should_scan_path(rel, suffix, enforce_authoritative=False)
                else []
            )

        if not os.path.isdir(target_abs):
            return []

        files: list[str] = []
        for root, dirs, names in os.walk(target_abs):
            dirs[:] = sorted(
                [
                    d
                    for d in dirs
                    if d
                    not in {
                        ".git",
                        ".saguaro",
                        "__pycache__",
                        "venv",
                        "Saguaro",
                        "node_modules",
                        "build",
                        "dist",
                    }
                    and not self._is_excluded_path(os.path.join(root, d))
                ]
            )
            for name in sorted(names):
                candidate = os.path.join(root, name)
                if self._is_excluded_path(candidate):
                    continue
                suffix = Path(name).suffix.lower()
                rel = os.path.relpath(candidate, self.repo_path).replace("\\", "/")
                if self._should_scan_path(
                    rel,
                    suffix,
                    enforce_authoritative=default_repo_target,
                ):
                    files.append(candidate)
        return files

    def _is_excluded_path(self, candidate: str) -> bool:
        rel = os.path.relpath(candidate, self.repo_path).replace("\\", "/")
        excluded = []
        excluded.extend(
            self.policy_config.get("aes_excluded_reference_roots", ["Saguaro/"]) or []
        )
        excluded.extend(
            self.policy_config.get("excluded_paths", _DEFAULT_EXCLUDED_PATHS) or []
        )
        return any(
            rel == str(root).rstrip("/") or rel.startswith(str(root).rstrip("/") + "/")
            for root in excluded
        )

    def _should_scan_path(
        self,
        rel_path: str,
        suffix: str,
        *,
        enforce_authoritative: bool,
    ) -> bool:
        if suffix not in _TEXT_EXTENSIONS:
            return False
        if suffix in {".md", ".txt"}:
            return (
                rel_path.startswith(_GOVERNED_DOC_PREFIXES)
                or Path(rel_path).name in _GOVERNED_DOC_NAMES
            )
        if (
            rel_path.startswith(_GOVERNED_DOC_PREFIXES)
            or rel_path == "standards/AES_RULES.json"
        ):
            return True
        return not (
            enforce_authoritative and not self._is_authoritative_runtime_path(rel_path)
        )

    def _is_authoritative_runtime_path(self, rel_path: str) -> bool:
        root = (
            str(
                self.policy_config.get("aes_authoritative_package_root", "saguaro")
                or ""
            )
            .strip()
            .strip("/")
        )
        if not root:
            return True
        if not (Path(self.repo_path) / root).exists():
            return True
        normalized = rel_path.replace("\\", "/").lstrip("./")
        return normalized == root or normalized.startswith(root + "/")

    @staticmethod
    def _hydrate_compliance_context(
        payload: dict[str, Any] | None,
        change_manifest: dict[str, Any],
    ) -> ComplianceContext:
        merged = dict(payload or {})
        manifest = dict(change_manifest or {})
        for key in (
            "run_id",
            "aal",
            "domains",
            "changed_files",
            "hot_paths",
            "public_api_changes",
            "dependency_changes",
            "required_rule_ids",
            "required_runtime_gates",
        ):
            if not merged.get(key) and manifest.get(key):
                merged[key] = manifest.get(key)
        return ComplianceContext.from_mapping(merged)

    @staticmethod
    def _should_run_governance_checks(
        verify_ctx: dict[str, Any],
        compliance_context: ComplianceContext,
        change_manifest: dict[str, Any],
    ) -> bool:
        if any(
            bool(verify_ctx.get(flag))
            for flag in ("require_trace", "require_evidence", "require_valid_waivers")
        ):
            return True
        if change_manifest:
            return True
        return any(
            (
                compliance_context.domains,
                compliance_context.evidence_bundle_id,
                compliance_context.waiver_ids,
                compliance_context.changed_files,
                compliance_context.hot_paths,
                compliance_context.public_api_changes,
                compliance_context.dependency_changes,
                compliance_context.required_rule_ids,
                compliance_context.required_runtime_gates,
            )
        )

    @staticmethod
    def _load_change_manifest(change_manifest_path: Any) -> dict[str, Any]:
        if not change_manifest_path:
            return {}
        path = Path(str(change_manifest_path))
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _change_manifest_status(
        self, change_manifest: dict[str, Any]
    ) -> tuple[bool, str]:
        if not change_manifest:
            return False, "Missing change manifest for high-assurance verification."

        schema_path = (
            Path(self.repo_path)
            / "standards"
            / "schemas"
            / "change_manifest.schema.json"
        )
        if not schema_path.exists():
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "standards"
                / "schemas"
                / "change_manifest.schema.json"
            )
        if not schema_path.exists():
            return True, ""

        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=change_manifest, schema=schema)
        except Exception as exc:
            return False, f"Invalid change manifest: {exc}"

        return True, ""

    @staticmethod
    def _parse_aal_filter(raw: Any) -> set[str] | None:
        if raw is None or raw == "":
            return None
        if isinstance(raw, str):
            tokens = [token.strip() for token in raw.split(",") if token.strip()]
        else:
            tokens = [str(token).strip() for token in raw if str(token).strip()]

        normalized: set[str] = set()
        for token in tokens:
            upper = token.upper()
            if upper.startswith("AAL-"):
                normalized.add(upper)
            elif token.isdigit():
                normalized.add(f"AAL-{token}")
        return normalized or None

    @staticmethod
    def _parse_domain_filter(raw: Any) -> set[str] | None:
        if raw is None or raw == "":
            return None
        if isinstance(raw, str):
            tokens = [
                token.strip().lower() for token in raw.split(",") if token.strip()
            ]
        else:
            tokens = [str(token).strip().lower() for token in raw if str(token).strip()]
        return set(tokens) or None

    @staticmethod
    def _rule_matches_aal(rule_severity: str, aal_filter: set[str] | None) -> bool:
        if not aal_filter:
            return True
        rule_aal = AESEngine._resolve_rule_aal(rule_severity)
        if rule_aal is None:
            return False
        threshold = min(_AAL_ORDER.get(aal, 3) for aal in aal_filter)
        return _AAL_ORDER[rule_aal] <= threshold

    @staticmethod
    def _resolve_rule_aal(rule_severity: str) -> str | None:
        normalized = (rule_severity or "").upper()
        if normalized.startswith("AAL-"):
            return normalized
        return {
            "P0": "AAL-0",
            "P1": "AAL-1",
            "P2": "AAL-2",
            "P3": "AAL-3",
            "ERROR": "AAL-1",
            "WARN": "AAL-2",
            "WARNING": "AAL-2",
            "INFO": "AAL-3",
        }.get(normalized)

    @staticmethod
    def _rule_matches_language(
        rule_languages: Iterable[str], file_language: str
    ) -> bool:
        if not rule_languages:
            return True
        return file_language in set(rule_languages)

    @staticmethod
    def _rule_matches_domain(
        rule_domains: Iterable[str],
        file_domains: set[str],
        domain_filter: set[str] | None,
    ) -> bool:
        domains = set(rule_domains or ["universal"])
        if domain_filter and not domains.intersection(domain_filter):
            return False
        if "universal" in domains:
            return domain_filter is None
        return bool(domains.intersection(file_domains))

    def _rule_matches_engine(self, engine_name: str) -> bool:
        normalized = str(engine_name or "").lower()
        if normalized in _NON_EXECUTABLE_ENGINES:
            return False
        if normalized == "ruff":
            return bool(self.policy_config.get("allow_ruff_rules"))
        return True

    def _ordered_rules(self) -> list[Any]:
        return sorted(
            self.registry.rules,
            key=lambda rule: (
                int(getattr(rule, "precedence", 100) or 100),
                str(getattr(rule, "id", "")),
            ),
        )

    @staticmethod
    def _sorted_unique_violations(
        violations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            violations,
            key=lambda item: (
                str(item.get("file", "")),
                int(item.get("line", 0) or 0),
                str(item.get("rule_id", "")),
                str(item.get("context", "")),
                str(item.get("message", "")),
            ),
        )
        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, int, str, str, str]] = set()
        for item in ordered:
            key = (
                str(item.get("file", "")),
                int(item.get("line", 0) or 0),
                str(item.get("rule_id", "")),
                str(item.get("context", "")),
                str(item.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _architecture_findings(self, path_arg: str) -> list[dict[str, Any]]:
        policy_path = Path(self.repo_path) / "standards" / "REPO_LAYOUT.yaml"
        if not policy_path.exists():
            return []
        analyzer = ArchitectureAnalyzer(self.repo_path)
        report = analyzer.verify(path_arg)
        return list(report.get("findings", []))

    def _run_check_function(
        self,
        check_function: Any,
        rule: Any,
        rel_path: str,
        file_aal: str,
        file_svl: str,
        domains: set[str],
        source: str,
    ) -> list[dict[str, Any]]:
        abs_path = os.path.join(self.repo_path, rel_path)
        try:
            raw_violations = check_function(source, abs_path) or []
        except SyntaxError:
            return []
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.debug("AES check function failed (%s): %s", rule.id, exc)
            return []

        normalized: list[dict[str, Any]] = []
        for entry in raw_violations:
            line = int(entry.get("line", 1))
            message = str(entry.get("message", rule.text))
            rule_id = str(entry.get("rule_id", rule.id))
            severity = _severity_to_p(str(rule.severity))
            normalized.append(
                self._build_rule_violation(
                    rel_path=rel_path,
                    line=line,
                    rule_id=rule_id,
                    message=message,
                    severity=severity,
                    file_aal=file_aal,
                    file_svl=file_svl,
                    domains=domains,
                    context=str(entry.get("context", "")),
                    cwe=getattr(rule, "cwe", None),
                    status=str(getattr(rule, "status", "")),
                )
            )
        return normalized

    def _run_pattern_check(
        self,
        rule: Any,
        rel_path: str,
        file_aal: str,
        file_svl: str,
        domains: set[str],
        source: str,
    ) -> list[dict[str, Any]]:
        pattern_text = str(getattr(rule, "pattern", "") or "")
        if not pattern_text:
            return []
        try:
            pattern = re.compile(pattern_text, re.IGNORECASE | re.MULTILINE)
        except re.error as exc:
            logger.debug("Invalid AES rule pattern (%s): %s", rule.id, exc)
            return []

        guard_text = str(getattr(rule, "negative_lookahead", "") or "").strip()
        guard = None
        if guard_text:
            try:
                guard = re.compile(guard_text, re.IGNORECASE | re.MULTILINE)
            except re.error as exc:
                logger.debug("Invalid AES rule guard pattern (%s): %s", rule.id, exc)
                guard = None

        source_lines = source.splitlines()
        violations: list[dict[str, Any]] = []
        severity = _severity_to_p(str(rule.severity))
        for match in pattern.finditer(source):
            line_no = source.count("\n", 0, match.start()) + 1
            if guard:
                start = max(1, line_no - 2)
                end = min(len(source_lines), line_no + 2)
                window = "\n".join(source_lines[start - 1 : end])
                if guard.search(window):
                    continue

            context = source_lines[line_no - 1].strip() if source_lines else ""
            violations.append(
                self._build_rule_violation(
                    rel_path=rel_path,
                    line=line_no,
                    rule_id=str(rule.id),
                    message=str(rule.text),
                    severity=severity,
                    file_aal=file_aal,
                    file_svl=file_svl,
                    domains=domains,
                    context=context[:300],
                    cwe=getattr(rule, "cwe", None),
                    status=str(getattr(rule, "status", "")),
                )
            )
        return violations

    @staticmethod
    def _build_rule_violation(
        rel_path: str,
        line: int,
        rule_id: str,
        message: str,
        severity: str,
        file_aal: str,
        file_svl: str,
        domains: set[str],
        context: str,
        cwe: list[str] | None = None,
        status: str = "",
    ) -> dict[str, Any]:
        normalized_status = str(status or "").strip().lower()
        if normalized_status in {"blocking", "guarded", "advisory"}:
            closure_level = normalized_status
        elif normalized_status == "advisory_pending_threshold":
            closure_level = "guarded"
        elif normalized_status == "blocking":
            closure_level = "blocking"
        else:
            closure_level = _closure_from_p(severity, file_aal)
        return {
            "file": rel_path,
            "line": line,
            "rule_id": rule_id,
            "message": message,
            "severity": severity,
            "aal": file_aal,
            "svl": file_svl,
            "domain": sorted(domains) if domains else ["universal"],
            "closure_level": closure_level,
            "evidence_refs": [],
            "context": context,
            "cwe": list(cwe or []),
            "status": normalized_status or None,
        }

    def _artifact_checks(
        self,
        strictest_seen: str,
        verify_ctx: dict[str, Any],
        file_aal_map: dict[str, str],
        compliance_context: ComplianceContext,
        change_manifest: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        strict_mode_block = bool(
            self.policy_config.get("block_on_missing_artifacts")
        ) and strictest_seen in {"AAL-0", "AAL-1"}
        if (
            not self._should_run_governance_checks(
                verify_ctx=verify_ctx,
                compliance_context=compliance_context,
                change_manifest=change_manifest or {},
            )
            and not strict_mode_block
        ):
            return []
        require_trace = bool(verify_ctx.get("require_trace")) or strict_mode_block
        require_evidence = bool(verify_ctx.get("require_evidence")) or strict_mode_block
        require_valid_waivers = bool(verify_ctx.get("require_valid_waivers"))
        if strict_mode_block:
            require_valid_waivers = True

        violations: list[dict[str, Any]] = []
        if self.policy_config.get(
            "aes_require_change_manifest", True
        ) and strictest_seen in {"AAL-0", "AAL-1"}:
            manifest_ok, manifest_msg = self._change_manifest_status(
                change_manifest or {}
            )
            if not manifest_ok:
                violations.append(
                    self._artifact_violation(
                        rule_id="AES-AG-1",
                        message=manifest_msg,
                        strictest_seen=strictest_seen,
                        status="blocking",
                    )
                )
        if require_trace and not self._has_traceability_records():
            violations.append(
                self._artifact_violation(
                    rule_id="AES-TR-1",
                    message="Missing traceability records for high-assurance verification.",
                    strictest_seen=strictest_seen,
                    status="blocking",
                )
            )

        evidence_status = self._evidence_status()
        if require_evidence and not evidence_status["present"]:
            violations.append(
                self._artifact_violation(
                    rule_id="AES-TR-2",
                    message="Missing evidence bundle required for closure.",
                    strictest_seen=strictest_seen,
                    status="blocking",
                )
            )

        review_ok, review_msg = self._review_independence_status(
            strictest_seen, evidence_status.get("bundles", [])
        )
        if require_evidence and not review_ok:
            violations.append(
                self._artifact_violation(
                    rule_id="AES-REV-1",
                    message=review_msg,
                    strictest_seen=strictest_seen,
                    status="blocking",
                )
            )

        waiver_ok, waiver_msg = self._waiver_status()
        if require_valid_waivers and not waiver_ok:
            violations.append(
                self._artifact_violation(
                    rule_id="AES-TR-3",
                    message=waiver_msg,
                    strictest_seen=strictest_seen,
                    status="blocking",
                )
            )

        # Ensure artifact violations carry file-level assurance context.
        if violations and file_aal_map:
            strictest_file = min(
                sorted(file_aal_map.items()),
                key=lambda item: (_AAL_ORDER[item[1]], item[0]),
            )[0]
            for violation in violations:
                violation["file"] = strictest_file
        return violations

    @staticmethod
    def _artifact_violation(
        rule_id: str, message: str, strictest_seen: str, status: str = ""
    ) -> dict[str, Any]:
        severity = "P1" if strictest_seen in {"AAL-0", "AAL-1"} else "P2"
        svl = AALClassifier.map_aal_to_svl(strictest_seen).value
        normalized_status = str(status or "").strip().lower()
        return {
            "file": "standards",
            "line": 1,
            "rule_id": rule_id,
            "message": message,
            "severity": severity,
            "aal": strictest_seen,
            "svl": svl,
            "domain": ["universal"],
            "closure_level": normalized_status
            or ("blocking" if severity == "P1" else "guarded"),
            "evidence_refs": [],
            "context": "",
            "status": normalized_status or None,
        }

    def _rule_ids_for_runtime_gate(
        self, gate_id: str, compliance_context: ComplianceContext
    ) -> set[str]:
        rule_ids = set(_GATE_TO_RULE_IDS.get(gate_id, ()))
        if gate_id != "domain_report_gate":
            return rule_ids

        domains = set(compliance_context.domains or [])
        scoped: set[str] = set()
        if "ml" in domains:
            scoped.add("AES-ML-5")
        if "hpc" in domains:
            scoped.add("AES-HPC-4")
        if "quantum" in domains:
            scoped.add("AES-QC-4")
        if "physics" in domains:
            scoped.add("AES-PHYS-1")
        return scoped or rule_ids

    def _runtime_gate_violations(
        self,
        runtime_summary: Any,
        applicable_rule_ids: set[str],
        strictest_file: str,
        strictest_seen: str,
        compliance_context: ComplianceContext,
    ) -> list[dict[str, Any]]:
        violations: list[dict[str, Any]] = []
        svl = AALClassifier.map_aal_to_svl(strictest_seen).value
        domains = set(compliance_context.domains or ["universal"])

        for result in getattr(runtime_summary, "results", []) or []:
            if bool(getattr(result, "passed", False)):
                continue
            candidate_rule_ids = self._rule_ids_for_runtime_gate(
                str(getattr(result, "gate_id", "")),
                compliance_context,
            )
            if applicable_rule_ids:
                candidate_rule_ids &= applicable_rule_ids
            if not candidate_rule_ids:
                candidate_rule_ids = {"AES-AG-2"}

            missing_artifacts = [
                str(item) for item in getattr(result, "missing_artifacts", []) or []
            ]
            suffix = (
                f" Missing or invalid artifacts: {', '.join(missing_artifacts)}"
                if missing_artifacts
                else ""
            )
            message = (
                f"{getattr(result, 'message', 'Runtime gate failed')}.{suffix}".rstrip(
                    "."
                )
            )

            for rule_id in sorted(candidate_rule_ids):
                rule = self.registry.get_rule(rule_id)
                status = (
                    str(getattr(rule, "status", "advisory")) if rule else "advisory"
                )
                severity = (
                    _severity_to_p(str(getattr(rule, "severity", "AAL-2")))
                    if rule is not None
                    else ("P1" if strictest_seen in {"AAL-0", "AAL-1"} else "P2")
                )
                violations.append(
                    self._build_rule_violation(
                        rel_path=strictest_file,
                        line=1,
                        rule_id=rule_id,
                        message=message,
                        severity=severity,
                        file_aal=strictest_seen,
                        file_svl=svl,
                        domains=domains,
                        context=f"runtime_gate:{getattr(result, 'gate_id', '')}",
                        cwe=getattr(rule, "cwe", None) if rule is not None else None,
                        status=status,
                    )
                )

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for violation in violations:
            key = (
                str(violation.get("rule_id", "")),
                str(violation.get("context", "")),
                str(violation.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(violation)
        return deduped

    def _execution_mode_rule_checks(
        self,
        strictest_seen: str,
        verify_ctx: dict[str, Any],
        file_aal_map: dict[str, str],
        compliance_context: ComplianceContext,
        change_manifest: dict[str, Any],
        obligation_result: Any,
    ) -> list[dict[str, Any]]:
        if obligation_result is None:
            return []
        violations: list[dict[str, Any]] = []
        applicable_rule_ids = set(
            getattr(obligation_result, "required_rule_ids", []) or []
        )
        applicable_rule_ids.update(compliance_context.required_rule_ids)
        if not applicable_rule_ids:
            return []
        applicable_gate_ids = set(
            getattr(obligation_result, "required_runtime_gates", []) or []
        )
        applicable_gate_ids.update(compliance_context.required_runtime_gates)
        strictest_file = (
            min(
                sorted(file_aal_map.items()),
                key=lambda item: (_AAL_ORDER[item[1]], item[0]),
            )[0]
            if file_aal_map
            else "."
        )
        if applicable_gate_ids:
            runtime_summary = self.runtime_gate_runner.evaluate(
                compliance_context,
                sorted(applicable_gate_ids),
                thresholds=getattr(obligation_result, "thresholds", {}),
            )
            violations.extend(
                self._runtime_gate_violations(
                    runtime_summary=runtime_summary,
                    applicable_rule_ids=applicable_rule_ids,
                    strictest_file=strictest_file,
                    strictest_seen=strictest_seen,
                    compliance_context=compliance_context,
                )
            )
        for rule in self.registry.rules:
            mode = str(getattr(rule, "execution_mode", "static") or "static")
            if mode not in {"workflow_gate", "runtime_gate", "manual"}:
                continue
            if str(getattr(rule, "id", "")) not in applicable_rule_ids:
                continue
            if not self._rule_matches_aal(
                rule.severity, self._parse_aal_filter(verify_ctx.get("aal"))
            ):
                continue
            if mode == "manual":
                if compliance_context.waiver_ids:
                    continue
                violations.append(
                    self._rule_violation_from_catalog(
                        rule=rule,
                        rel_path=strictest_file,
                        file_aal=strictest_seen,
                        file_svl=AALClassifier.map_aal_to_svl(strictest_seen).value,
                        domains=set(compliance_context.domains or ["universal"]),
                        message=f"Manual AES rule requires explicit waiver or human review record: {rule.title}",
                        status="advisory",
                    )
                )
                continue

            missing = self._missing_required_artifacts(
                rule.required_artifacts,
                compliance_context=compliance_context,
                change_manifest=change_manifest,
            )
            if missing:
                violations.append(
                    self._rule_violation_from_catalog(
                        rule=rule,
                        rel_path=strictest_file,
                        file_aal=strictest_seen,
                        file_svl=AALClassifier.map_aal_to_svl(strictest_seen).value,
                        domains=set(compliance_context.domains or ["universal"]),
                        message=f"Missing required artifacts for {rule.id}: {', '.join(missing)}",
                        status=str(getattr(rule, "status", "")),
                    )
                )
        return violations

    def _missing_required_artifacts(
        self,
        required_artifacts: Iterable[str],
        compliance_context: ComplianceContext,
        change_manifest: dict[str, Any],
    ) -> list[str]:
        missing: list[str] = []
        run_dir = (
            Path(self.repo_path) / ".anvil" / "compliance" / compliance_context.run_id
        )
        for artifact in required_artifacts or []:
            if artifact == "change_manifest.json" and change_manifest:
                continue
            candidate = Path(self.repo_path) / str(artifact)
            runtime_candidate = run_dir / str(artifact)
            standards_candidate = Path(self.repo_path) / "standards" / str(artifact)
            if (
                candidate.exists()
                or runtime_candidate.exists()
                or standards_candidate.exists()
            ):
                continue
            missing.append(str(artifact))
        return missing

    def _rule_violation_from_catalog(
        self,
        rule: Any,
        rel_path: str,
        file_aal: str,
        file_svl: str,
        domains: set[str],
        message: str,
        status: str,
    ) -> dict[str, Any]:
        severity = _severity_to_p(str(getattr(rule, "severity", "AAL-2")))
        return self._build_rule_violation(
            rel_path=rel_path,
            line=1,
            rule_id=str(getattr(rule, "id", "AES-UNKNOWN")),
            message=message,
            severity=severity,
            file_aal=file_aal,
            file_svl=file_svl,
            domains=domains,
            context="catalog_execution_mode_check",
            cwe=getattr(rule, "cwe", None),
            status=status,
        )

    def _has_traceability_records(self) -> bool:
        trace_file = (
            Path(self.repo_path) / "standards" / "traceability" / "TRACEABILITY.jsonl"
        )
        if not trace_file.exists():
            return False
        try:
            return any(
                line.strip()
                for line in trace_file.read_text(encoding="utf-8").splitlines()
            )
        except OSError:
            return False

    def _evidence_status(self) -> dict[str, Any]:
        candidates = [
            Path(self.repo_path) / "standards" / "evidence_bundle.json",
            Path(self.repo_path) / "standards" / "evidence" / "bundle.json",
        ]
        candidates.extend(
            sorted((Path(self.repo_path) / "standards" / "evidence").glob("*.json"))
        )
        candidates.extend(
            sorted((Path(self.repo_path) / ".anvil" / "evidence").glob("*.json"))
        )

        bundles: list[dict[str, Any]] = []
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                bundles.append(payload)

        return {"present": bool(bundles), "bundles": bundles}

    def _review_independence_status(
        self, strictest_seen: str, bundles: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        matrix_path = Path(self.repo_path) / "standards" / "review_matrix.yaml"
        required_reviews = 0
        if matrix_path.exists():
            try:
                matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8")) or {}
                required_reviews = int(
                    matrix.get("aal_levels", {})
                    .get(strictest_seen, {})
                    .get("independent_reviews", 0)
                )
            except Exception:
                required_reviews = 0

        if required_reviews <= 0:
            return True, ""
        if not bundles:
            return False, "Missing evidence bundle with review signoffs."

        for bundle in bundles:
            signoffs = bundle.get("review_signoffs") or []
            approved = [
                signoff
                for signoff in signoffs
                if isinstance(signoff, dict) and signoff.get("decision") == "approved"
            ]
            if len(approved) >= required_reviews:
                return True, ""
        return (
            False,
            f"Insufficient independent review signoffs: requires {required_reviews} approvals.",
        )

    def _waiver_status(self) -> tuple[bool, str]:
        waiver_dir = Path(self.repo_path) / "standards" / "waivers"
        if not waiver_dir.exists():
            return True, ""

        waiver_files = (
            sorted(waiver_dir.glob("*.yaml"))
            + sorted(waiver_dir.glob("*.yml"))
            + sorted(waiver_dir.glob("*.json"))
        )
        verify_ctx = dict(self.policy_config.get("verify_context", {}))
        as_of = str(verify_ctx.get("waiver_as_of", "") or "").strip()
        today = dt.date.today()
        if as_of:
            try:
                today = dt.date.fromisoformat(as_of)
            except ValueError:
                return False, "waiver_as_of is not ISO date"
        for waiver_file in waiver_files:
            try:
                if waiver_file.suffix == ".json":
                    payload = json.loads(waiver_file.read_text(encoding="utf-8"))
                else:
                    payload = yaml.safe_load(waiver_file.read_text(encoding="utf-8"))
            except Exception:
                return False, f"Invalid waiver format: {waiver_file.name}"

            if not isinstance(payload, dict):
                return False, f"Invalid waiver payload: {waiver_file.name}"
            required = {
                "waiver_id",
                "rule_id",
                "change_scope",
                "compensating_control",
                "risk_owner",
                "expiry",
                "remediation_ticket",
            }
            if not required.issubset(payload):
                return False, f"Waiver missing required fields: {waiver_file.name}"
            expiry = str(payload.get("expiry", "")).strip()
            try:
                expiry_date = dt.date.fromisoformat(expiry)
            except ValueError:
                return False, f"Waiver expiry is not ISO date: {waiver_file.name}"
            if expiry_date < today:
                return False, f"Waiver expired: {waiver_file.name}"

        return True, ""

    def _prompt_policy_checks(self, strictest_seen: str) -> list[dict[str, Any]]:
        script_path = Path(self.repo_path) / "scripts" / "validate_prompt_contracts.py"
        if not script_path.exists():
            return []

        try:
            proc = subprocess.run(
                [sys.executable, str(script_path), "--repo", self.repo_path, "--json"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except Exception as exc:
            severity = "P1" if strictest_seen in {"AAL-0", "AAL-1"} else "P2"
            svl = AALClassifier.map_aal_to_svl(strictest_seen).value
            return [
                {
                    "file": "scripts/validate_prompt_contracts.py",
                    "line": 1,
                    "rule_id": "AES-PRM-1",
                    "message": f"Prompt policy validator execution failed: {exc}",
                    "severity": severity,
                    "aal": strictest_seen,
                    "svl": svl,
                    "domain": ["universal"],
                    "closure_level": _closure_from_p(severity, strictest_seen),
                    "evidence_refs": [],
                    "context": "prompt_contract_validation",
                }
            ]

        output = (proc.stdout or "").strip()
        payload: dict[str, Any] = {}
        if output:
            try:
                payload = json.loads(output)
            except Exception:
                payload = {}

        validator_errors = (
            payload.get("errors", []) if isinstance(payload, dict) else []
        )
        if proc.returncode == 0 and not validator_errors:
            return []

        severity = "P1" if strictest_seen in {"AAL-0", "AAL-1"} else "P2"
        svl = AALClassifier.map_aal_to_svl(strictest_seen).value
        violations: list[dict[str, Any]] = []
        if not validator_errors:
            validator_errors = [
                {
                    "file": "prompts",
                    "message": (
                        "Prompt contract validation failed without structured error output."
                    ),
                }
            ]

        for item in validator_errors[:50]:
            file_path = str(item.get("file", "prompts"))
            message = str(item.get("message", "Prompt policy violation detected"))
            violations.append(
                {
                    "file": file_path,
                    "line": 1,
                    "rule_id": "AES-PRM-1",
                    "message": message,
                    "severity": severity,
                    "aal": strictest_seen,
                    "svl": svl,
                    "domain": ["universal"],
                    "closure_level": _closure_from_p(severity, strictest_seen),
                    "evidence_refs": [],
                    "context": "prompt_contract_validation",
                }
            )
        return violations
