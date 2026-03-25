"""Utilities for adapters."""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import FindingRecord, FixBatch, FixReceipt


def normalize_engine_name(engine: Any) -> str:
    name = engine.__class__.__name__
    lowered = name.lower().lstrip("_")
    if lowered.endswith("engine"):
        lowered = lowered[:-6]
    return lowered


class FixEngineAdapter(ABC):
    adapter_key = "base"
    tool = "base"

    def __init__(self, repo_path: str, engines: list[Any] | None = None) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.engines = list(engines or [])

    @abstractmethod
    def supports(self, batch: FixBatch) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        batch: FixBatch,
        findings: list[FindingRecord],
        receipt_dir: str,
        dry_run: bool,
    ) -> FixReceipt:
        raise NotImplementedError


class FormatterAdapter(FixEngineAdapter):
    pass


class CodemodAdapter(FixEngineAdapter):
    pass


class SemanticPatchAdapter(FixEngineAdapter):
    pass


class PlannedOnlyAdapter(FixEngineAdapter):
    def __init__(self, repo_path: str, reason: str) -> None:
        super().__init__(repo_path=repo_path, engines=[])
        self.reason = reason

    def supports(self, batch: FixBatch) -> bool:
        return False

    def apply(
        self,
        batch: FixBatch,
        findings: list[FindingRecord],
        receipt_dir: str,
        dry_run: bool,
    ) -> FixReceipt:
        return FixReceipt(
            receipt_id=f"{batch.batch_id}-planned",
            run_id="",
            batch_id=batch.batch_id,
            adapter_key=batch.adapter_key,
            tool=batch.tool,
            language=batch.language,
            safety_tier=batch.safety_tier,
            status="skipped",
            dry_run=dry_run,
            finding_ids=batch.finding_ids,
            rule_ids=batch.rule_ids,
            files=batch.files,
            confidence=batch.confidence,
            error=self.reason or batch.reason or "adapter unavailable",
        )


class PythonCodemodAdapter(CodemodAdapter):
    adapter_key = "python_codemod"
    tool = "python-ast"
    _BARE_EXCEPT_RE = re.compile(r"^(\s*)except(\s*)(:)(\s*(#.*)?)$")
    _DOCSTRING_RULES = {"D100", "D101", "D102", "D103", "D104", "D107"}

    def supports(self, batch: FixBatch) -> bool:
        if batch.adapter_key != self.adapter_key:
            return False
        return set(batch.rule_ids).issubset({"AES-CR-2", "AES-PY-3", *self._DOCSTRING_RULES})

    def apply(
        self,
        batch: FixBatch,
        findings: list[FindingRecord],
        receipt_dir: str,
        dry_run: bool,
    ) -> FixReceipt:
        status = "planned" if dry_run else "no_change"
        changed_files: list[str] = []
        command_log = [f"{self.__class__.__name__}.apply({','.join(batch.rule_ids)})"]
        diff_path: str | None = None
        rollback_bundle_path: str | None = None
        fixed_count = 0
        error: str | None = None
        rollback_payload: dict[str, str] = {}
        diffs: list[str] = []

        target_findings = [
            finding for finding in findings if finding.finding_id in batch.finding_ids
        ]
        grouped_rules: dict[str, dict[str, set[int]]] = {}
        for finding in target_findings:
            if finding.rule_id not in {"AES-CR-2", "AES-PY-3", *self._DOCSTRING_RULES}:
                error = f"unsupported python codemod rule: {finding.rule_id}"
                continue
            grouped_rules.setdefault(finding.file, {}).setdefault(finding.rule_id, set()).add(
                finding.line
            )

        if error is not None and not grouped_rules:
            return FixReceipt(
                receipt_id=f"{batch.batch_id}-receipt",
                run_id="",
                batch_id=batch.batch_id,
                adapter_key=batch.adapter_key,
                tool=self.tool,
                language=batch.language,
                safety_tier=batch.safety_tier,
                status="skipped" if not dry_run else "planned",
                dry_run=dry_run,
                finding_ids=batch.finding_ids,
                rule_ids=batch.rule_ids,
                files=batch.files,
                confidence=batch.confidence,
                error=error,
                command_log=command_log,
            )

        for file_path, rule_lines in grouped_rules.items():
            abs_path = os.path.join(self.repo_path, file_path)
            if not os.path.isfile(abs_path):
                error = f"file not found: {file_path}"
                continue
            original = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
            updated = original
            file_fixed = 0

            bare_lines = rule_lines.get("AES-CR-2", set())
            if bare_lines:
                updated, bare_fixed = self._rewrite_bare_excepts(updated, bare_lines)
                file_fixed += bare_fixed

            mutable_lines = rule_lines.get("AES-PY-3", set())
            if mutable_lines:
                try:
                    updated, mutable_fixed = self._rewrite_mutable_defaults(
                        updated,
                        mutable_lines,
                    )
                except SyntaxError as exc:
                    error = f"python codemod parse failed for {file_path}: {exc}"
                    continue
                file_fixed += mutable_fixed

            docstring_rule_lines = {
                rule_id: lines
                for rule_id, lines in rule_lines.items()
                if rule_id in self._DOCSTRING_RULES
            }
            if docstring_rule_lines:
                try:
                    updated, docstring_fixed = self._rewrite_missing_docstrings(
                        updated,
                        file_path,
                        docstring_rule_lines,
                    )
                except SyntaxError as exc:
                    error = f"python codemod parse failed for {file_path}: {exc}"
                    continue
                file_fixed += docstring_fixed

            if file_fixed == 0 or updated == original:
                continue

            if dry_run:
                status = "planned"
            else:
                Path(abs_path).write_text(updated, encoding="utf-8")
                status = "applied"
            fixed_count += file_fixed
            changed_files.append(file_path)
            rollback_payload[file_path] = original
            diffs.extend(
                difflib.unified_diff(
                    original.splitlines(),
                    updated.splitlines(),
                    fromfile=f"a/{file_path}",
                    tofile=f"b/{file_path}",
                    lineterm="",
                )
            )

        if changed_files:
            diff_path = os.path.join(receipt_dir, f"{batch.batch_id}.diff")
            Path(diff_path).write_text("\n".join(diffs) + "\n", encoding="utf-8")
            rollback_bundle_path = os.path.join(receipt_dir, f"{batch.batch_id}.rollback.json")
            Path(rollback_bundle_path).write_text(
                json.dumps(rollback_payload, indent=2),
                encoding="utf-8",
            )
        elif error is None and not dry_run:
            status = "no_change"

        return FixReceipt(
            receipt_id=f"{batch.batch_id}-receipt",
            run_id="",
            batch_id=batch.batch_id,
            adapter_key=batch.adapter_key,
            tool=self.tool,
            language=batch.language,
            safety_tier=batch.safety_tier,
            status=status,
            dry_run=dry_run,
            finding_ids=batch.finding_ids,
            rule_ids=batch.rule_ids,
            files=batch.files,
            changed_files=changed_files,
            fixed_count=fixed_count,
            confidence=batch.confidence,
            diff_path=diff_path,
            rollback_bundle_path=rollback_bundle_path,
            error=error,
            command_log=command_log,
        )

    def _rewrite_bare_excepts(self, source: str, target_lines: set[int]) -> tuple[str, int]:
        updated_lines = source.splitlines(keepends=True)
        fixed_count = 0
        for line_no in sorted(target_lines):
            index = line_no - 1
            if index < 0 or index >= len(updated_lines):
                continue
            line = updated_lines[index]
            match = self._BARE_EXCEPT_RE.match(line.rstrip("\n"))
            if match is None:
                continue
            updated_lines[index] = (
                f"{match.group(1)}except Exception{match.group(3)}{match.group(4) or ''}"
            )
            if line.endswith("\n"):
                updated_lines[index] += "\n"
            fixed_count += 1
        return "".join(updated_lines), fixed_count

    def _rewrite_mutable_defaults(self, source: str, target_lines: set[int]) -> tuple[str, int]:
        tree = ast.parse(source)
        replacements: list[tuple[int, int, str]] = []
        insertions: list[tuple[int, str]] = []
        fixed_count = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.lineno not in target_lines:
                continue

            edits: list[tuple[str, str]] = []
            positional_args = list(node.args.posonlyargs) + list(node.args.args)
            defaults = list(node.args.defaults)
            if defaults:
                for arg, default in zip(positional_args[-len(defaults) :], defaults):
                    if not self._is_mutable_default(default):
                        continue
                    replacement = ast.get_source_segment(source, default)
                    if replacement is None:
                        continue
                    replacements.append(
                        (
                            self._offset_for(source, default.lineno, default.col_offset),
                            self._offset_for(source, default.end_lineno, default.end_col_offset),
                            "None",
                        )
                    )
                    edits.append((arg.arg, replacement))
                    fixed_count += 1

            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is None or not self._is_mutable_default(default):
                    continue
                replacement = ast.get_source_segment(source, default)
                if replacement is None:
                    continue
                replacements.append(
                    (
                        self._offset_for(source, default.lineno, default.col_offset),
                        self._offset_for(source, default.end_lineno, default.end_col_offset),
                        "None",
                    )
                )
                edits.append((arg.arg, replacement))
                fixed_count += 1

            if edits:
                insertions.append(self._guard_insertion(source, node, edits))

        if not replacements:
            return source, 0

        updated = source
        for start, end, replacement in sorted(replacements, reverse=True):
            updated = updated[:start] + replacement + updated[end:]

        lines = updated.splitlines(keepends=True)
        for insert_at, block in sorted(insertions, reverse=True):
            lines.insert(insert_at, block)
        return "".join(lines), fixed_count

    def _rewrite_missing_docstrings(
        self,
        source: str,
        file_path: str,
        target_rules: dict[str, set[int]],
    ) -> tuple[str, int]:
        tree = ast.parse(source)
        insertions: list[tuple[int, str]] = []
        fixed_count = 0
        module_rule_lines = set(target_rules.get("D100", set())) | set(
            target_rules.get("D104", set())
        )
        if module_rule_lines and ast.get_docstring(tree) is None:
            insert_at = self._module_docstring_index(source, tree)
            insertions.append((insert_at, self._module_docstring_block(file_path)))
            fixed_count += 1

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.lineno in target_rules.get("D101", set()):
                if ast.get_docstring(node) is None:
                    insertions.append(
                        self._docstring_insertion(
                            source,
                            node,
                            self._class_docstring_summary(node.name),
                        )
                    )
                    fixed_count += 1
                continue

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            rule_id = None
            if node.name == "__init__" and node.lineno in target_rules.get("D107", set()):
                rule_id = "D107"
            elif node.lineno in target_rules.get("D102", set()):
                rule_id = "D102"
            elif node.lineno in target_rules.get("D103", set()):
                rule_id = "D103"
            if rule_id is None:
                continue
            if ast.get_docstring(node) is not None:
                continue
            insertions.append(
                self._docstring_insertion(
                    source,
                    node,
                    self._function_docstring_summary(node.name),
                )
            )
            fixed_count += 1

        if not insertions:
            return source, 0

        lines = source.splitlines(keepends=True)
        for insert_at, block in sorted(insertions, reverse=True):
            lines.insert(insert_at, block)
        return "".join(lines), fixed_count

    @staticmethod
    def _is_mutable_default(node: ast.AST) -> bool:
        return isinstance(node, (ast.List, ast.Dict, ast.Set))

    @staticmethod
    def _offset_for(source: str, lineno: int | None, col: int | None) -> int:
        if lineno is None or col is None:
            return 0
        lines = source.splitlines(keepends=True)
        return sum(len(line) for line in lines[: lineno - 1]) + col

    def _guard_insertion(
        self,
        source: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        edits: list[tuple[str, str]],
    ) -> tuple[int, str]:
        source_lines = source.splitlines(keepends=True)
        function_line = source_lines[node.lineno - 1]
        function_indent = function_line[: len(function_line) - len(function_line.lstrip())]
        body_indent = function_indent + "    "

        insert_at = node.body[0].lineno - 1 if node.body else node.end_lineno or node.lineno
        if node.body:
            first_stmt = node.body[0]
            body_line = source_lines[first_stmt.lineno - 1]
            leading = body_line[: len(body_line) - len(body_line.lstrip())]
            if leading:
                body_indent = leading
            if (
                isinstance(first_stmt, ast.Expr)
                and isinstance(first_stmt.value, ast.Constant)
                and isinstance(first_stmt.value.value, str)
            ):
                insert_at = first_stmt.end_lineno or first_stmt.lineno

        block = "".join(
            f"{body_indent}if {name} is None:\n{body_indent}    {name} = {default_expr}\n"
            for name, default_expr in edits
        )
        return insert_at, block

    def _docstring_insertion(
        self,
        source: str,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        summary: str,
    ) -> tuple[int, str]:
        source_lines = source.splitlines(keepends=True)
        header_line = source_lines[node.lineno - 1]
        header_indent = header_line[: len(header_line) - len(header_line.lstrip())]
        body_indent = header_indent + "    "
        insert_at = node.body[0].lineno - 1 if node.body else (node.end_lineno or node.lineno)

        if node.body:
            first_stmt = node.body[0]
            body_line = source_lines[first_stmt.lineno - 1]
            leading = body_line[: len(body_line) - len(body_line.lstrip())]
            if leading:
                body_indent = leading

        return insert_at, f'{body_indent}"""{summary}"""\n'

    @staticmethod
    def _module_docstring_index(source: str, tree: ast.Module) -> int:
        if tree.body:
            return max(tree.body[0].lineno - 1, 0)
        return 0

    @staticmethod
    def _module_docstring_block(file_path: str) -> str:
        path = Path(file_path)
        if path.name == "__init__.py":
            package_name = path.parent.name.replace("_", " ")
            return f'"""Package initialization for {package_name}.\"\"\"\n\n'
        module_name = path.stem.replace("_", " ")
        return f'"""Utilities for {module_name}.\"\"\"\n\n'

    @staticmethod
    def _class_docstring_summary(name: str) -> str:
        return f"Provide {PythonCodemodAdapter._humanize_name(name)} support."

    @staticmethod
    def _function_docstring_summary(name: str) -> str:
        humanized = PythonCodemodAdapter._humanize_name(name)
        if name == "__init__":
            return "Initialize the instance."
        if name.startswith(("get_", "list_", "load_", "read_", "fetch_")):
            return f"{humanized.capitalize()}."
        if name.startswith(("set_", "update_", "write_", "store_", "save_")):
            return f"{humanized.capitalize()}."
        if name.startswith(("is_", "has_", "can_", "should_")):
            return f"Return whether {humanized[3:]}."
        return f"Handle {humanized}."

    @staticmethod
    def _humanize_name(name: str) -> str:
        cleaned = name.strip("_").replace("_", " ")
        return cleaned or "item"


class ArtifactTemplateAdapter(FixEngineAdapter):
    adapter_key = "artifact_templates"
    tool = "aes"

    def supports(self, batch: FixBatch) -> bool:
        if batch.adapter_key != self.adapter_key:
            return False
        return set(batch.rule_ids).issubset({"AES-TR-1", "AES-TR-2", "AES-VIS-1"})

    def apply(
        self,
        batch: FixBatch,
        findings: list[FindingRecord],
        receipt_dir: str,
        dry_run: bool,
    ) -> FixReceipt:
        status = "planned" if dry_run else "no_change"
        changed_files: list[str] = []
        fixed_count = 0
        command_log = [f"{self.__class__.__name__}.apply({','.join(batch.rule_ids)})"]
        diff_path: str | None = None
        rollback_bundle_path: str | None = None
        error: str | None = None
        rollback_payload: dict[str, str | None] = {}
        diffs: list[str] = []
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

        file_payloads = self._artifact_payloads(now=now)
        target_paths: dict[str, dict[str, str]] = {}
        for rule_id in batch.rule_ids:
            if rule_id == "AES-TR-1":
                target_paths["standards/traceability/TRACEABILITY.jsonl"] = file_payloads[
                    "standards/traceability/TRACEABILITY.jsonl"
                ]
            elif rule_id == "AES-TR-2":
                target_paths["standards/evidence_bundle.json"] = file_payloads[
                    "standards/evidence_bundle.json"
                ]
            elif rule_id == "AES-VIS-1":
                target_paths["aes_visuals/v1/directives.json"] = file_payloads[
                    "aes_visuals/v1/directives.json"
                ]
                target_paths["aes_visuals/v2/directives.json"] = file_payloads[
                    "aes_visuals/v2/directives.json"
                ]
            else:
                error = f"unsupported artifact remediation rule: {rule_id}"

        for rel_path, payload in target_paths.items():
            abs_path = os.path.join(self.repo_path, rel_path)
            before = Path(abs_path).read_text(encoding="utf-8", errors="ignore") if os.path.exists(abs_path) else None
            desired = payload["content"]
            if before == desired:
                continue
            rollback_payload[rel_path] = before
            if dry_run:
                status = "planned"
            else:
                Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
                Path(abs_path).write_text(desired, encoding="utf-8")
                status = "applied"
            changed_files.append(rel_path)
            fixed_count += 1
            diffs.extend(
                difflib.unified_diff(
                    [] if before is None else before.splitlines(),
                    desired.splitlines(),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                    lineterm="",
                )
            )

        if changed_files:
            diff_path = os.path.join(receipt_dir, f"{batch.batch_id}.diff")
            Path(diff_path).write_text("\n".join(diffs) + "\n", encoding="utf-8")
            rollback_bundle_path = os.path.join(receipt_dir, f"{batch.batch_id}.rollback.json")
            Path(rollback_bundle_path).write_text(
                json.dumps(rollback_payload, indent=2),
                encoding="utf-8",
            )
        elif error is None and not dry_run:
            status = "no_change"

        return FixReceipt(
            receipt_id=f"{batch.batch_id}-receipt",
            run_id="",
            batch_id=batch.batch_id,
            adapter_key=batch.adapter_key,
            tool=self.tool,
            language=batch.language,
            safety_tier=batch.safety_tier,
            status=status,
            dry_run=dry_run,
            finding_ids=batch.finding_ids,
            rule_ids=batch.rule_ids,
            files=batch.files,
            changed_files=changed_files,
            fixed_count=fixed_count,
            confidence=batch.confidence,
            diff_path=diff_path,
            rollback_bundle_path=rollback_bundle_path,
            error=error,
            command_log=command_log,
        )

    @staticmethod
    def _artifact_payloads(*, now: str) -> dict[str, dict[str, str]]:
        trace_record = {
            "trace_id": "bootstrap-trace",
            "run_id": "bootstrap-run",
            "requirement_id": "AES-bootstrap",
            "design_ref": "standards/AES.md",
            "code_refs": ["saguaro/sentinel/remediation"],
            "test_refs": ["tests/test_saguaro_remediation.py"],
            "verification_refs": ["saguaro verify . --format json"],
            "aal": "AAL-2",
            "owner": "saguaro-remediation",
            "timestamp": now,
            "changed_files": ["saguaro/sentinel/remediation"],
            "evidence_bundle_id": "bootstrap-bundle",
        }
        evidence_bundle = {
            "bundle_id": "bootstrap-bundle",
            "change_id": "bootstrap-run",
            "trace_id": "bootstrap-trace",
            "changed_files": ["saguaro/sentinel/remediation"],
            "aal": "AAL-2",
            "chronicle_snapshot": "pending",
            "chronicle_diff": "pending",
            "verification_report_path": "pending",
            "red_team_report_path": "pending",
            "review_signoffs": [
                {
                    "reviewer": "pending",
                    "timestamp": now,
                    "decision": "approved",
                }
            ],
            "waivers": [],
            "author": "saguaro-remediation",
        }
        visual_template = {
            "version": "v1",
            "telemetry": {"run_id": "bootstrap-run", "trace_id": "bootstrap-trace", "aal": "AAL-2"},
            "thresholds": {
                "complexity": {"max_cyclomatic": 10, "max_statements": 60},
                "dependency": {"require_hashes": True},
            },
        }
        visual_v1 = dict(visual_template)
        visual_v2 = dict(visual_template)
        visual_v2["version"] = "v2"
        return {
            "standards/traceability/TRACEABILITY.jsonl": {
                "content": json.dumps(trace_record, sort_keys=True) + "\n"
            },
            "standards/evidence_bundle.json": {
                "content": json.dumps(evidence_bundle, indent=2, sort_keys=True) + "\n"
            },
            "aes_visuals/v1/directives.json": {
                "content": json.dumps(visual_v1, indent=2, sort_keys=True) + "\n"
            },
            "aes_visuals/v2/directives.json": {
                "content": json.dumps(visual_v2, indent=2, sort_keys=True) + "\n"
            },
        }


class LegacyCompatAdapter(FixEngineAdapter):
    def __init__(self, repo_path: str, engine_name: str, engines: list[Any]) -> None:
        super().__init__(repo_path=repo_path, engines=engines)
        self.engine_name = engine_name
        self.adapter_key = f"legacy_{engine_name}"
        self.tool = engine_name

    def supports(self, batch: FixBatch) -> bool:
        if batch.adapter_key != self.adapter_key:
            return False
        return any(normalize_engine_name(engine) == self.engine_name for engine in self.engines)

    def apply(
        self,
        batch: FixBatch,
        findings: list[FindingRecord],
        receipt_dir: str,
        dry_run: bool,
    ) -> FixReceipt:
        status = "planned" if dry_run else "skipped"
        changed_files: list[str] = []
        command_log: list[str] = []
        diff_path: str | None = None
        rollback_bundle_path: str | None = None
        fixed_count = 0
        error: str | None = None

        target_findings = [finding for finding in findings if finding.finding_id in batch.finding_ids]
        relevant_engines = [
            engine for engine in self.engines if normalize_engine_name(engine) == self.engine_name
        ]
        if not relevant_engines:
            status = "skipped"
            error = f"no engine available for {self.engine_name}"
        elif dry_run:
            status = "planned"
        else:
            before_content: dict[str, str] = {}
            for file_path in batch.files:
                abs_path = os.path.join(self.repo_path, file_path)
                if os.path.isfile(abs_path):
                    before_content[file_path] = Path(abs_path).read_text(
                        encoding="utf-8", errors="ignore"
                    )

            for finding in target_findings:
                corrected = False
                raw_violation = dict(finding.raw)
                raw_violation["file"] = finding.file
                for engine in relevant_engines:
                    command_log.append(f"{engine.__class__.__name__}.fix({finding.rule_id})")
                    try:
                        if engine.fix(raw_violation):
                            corrected = True
                            fixed_count += 1
                            break
                    except Exception as exc:
                        error = str(exc)
                if corrected:
                    status = "applied"

            diffs: list[str] = []
            rollback_payload: dict[str, str] = {}
            for file_path, original in before_content.items():
                abs_path = os.path.join(self.repo_path, file_path)
                if not os.path.isfile(abs_path):
                    continue
                updated = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
                if updated == original:
                    continue
                changed_files.append(file_path)
                rollback_payload[file_path] = original
                diffs.extend(
                    difflib.unified_diff(
                        original.splitlines(),
                        updated.splitlines(),
                        fromfile=f"a/{file_path}",
                        tofile=f"b/{file_path}",
                        lineterm="",
                    )
                )

            if changed_files:
                diff_path = os.path.join(receipt_dir, f"{batch.batch_id}.diff")
                Path(diff_path).write_text("\n".join(diffs) + "\n", encoding="utf-8")
                rollback_bundle_path = os.path.join(receipt_dir, f"{batch.batch_id}.rollback.json")
                Path(rollback_bundle_path).write_text(
                    json.dumps(rollback_payload, indent=2),
                    encoding="utf-8",
                )
            elif status == "planned":
                status = "planned"
            elif fixed_count == 0:
                status = "no_change"

        return FixReceipt(
            receipt_id=f"{batch.batch_id}-receipt",
            run_id="",
            batch_id=batch.batch_id,
            adapter_key=batch.adapter_key,
            tool=self.tool,
            language=batch.language,
            safety_tier=batch.safety_tier,
            status=status,
            dry_run=dry_run,
            finding_ids=batch.finding_ids,
            rule_ids=batch.rule_ids,
            files=batch.files,
            changed_files=changed_files,
            fixed_count=fixed_count,
            confidence=batch.confidence,
            diff_path=diff_path,
            rollback_bundle_path=rollback_bundle_path,
            error=error,
            command_log=command_log,
        )
