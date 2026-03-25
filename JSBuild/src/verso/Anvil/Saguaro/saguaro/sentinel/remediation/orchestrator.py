"""Utilities for orchestrator."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from .adapters import (
    ArtifactTemplateAdapter,
    LegacyCompatAdapter,
    PlannedOnlyAdapter,
    PythonCodemodAdapter,
)
from .capabilities import LanguageCapabilityRegistry
from .classifier import FindingClassifier
from .models import (
    FIX_CLASS_AES_ARTIFACT,
    FIX_CLASS_AST_CODEMOD,
    FIX_CLASS_MANUAL,
    FIX_CLASS_NATIVE_SEMANTIC,
    FIX_CLASS_RULE_TUNE,
    FIX_CLASS_STOCK_LINTER,
    SAFETY_SEMANTIC,
    FindingRecord,
    FixBatch,
    FixPlan,
    FixReceipt,
    RollbackReceipt,
)
from .toolchains import ToolchainManager

_FIX_MODE_ORDER = {"safe": 0, "guarded": 1, "full": 2}
_SAFETY_ORDER = {
    "safe-format": 0,
    "safe-lint": 0,
    "safe-structured": 0,
    "artifact-only": 0,
    "guarded-structured": 1,
    "semantic": 2,
}


class FixSafetyPolicy:
    def allows(self, batch: FixBatch, fix_mode: str) -> tuple[bool, str]:
        mode = _FIX_MODE_ORDER.get(str(fix_mode or "safe").lower(), 0)
        required = _SAFETY_ORDER.get(batch.safety_tier, 2)
        if mode < required:
            return False, f"{batch.safety_tier} requires --fix-mode {'guarded' if required == 1 else 'full'}"
        if batch.fix_class == FIX_CLASS_NATIVE_SEMANTIC:
            return False, "native semantic patch adapter is scaffolded but not enabled in MVP"
        if batch.safety_tier == SAFETY_SEMANTIC:
            return False, "semantic remediation is scaffolded but not enabled in MVP"
        return True, ""


class FixOrchestrator:
    def __init__(
        self,
        repo_path: str,
        verifier: Any,
        receipt_dir: str | None = None,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.verifier = verifier
        raw_receipt_dir = receipt_dir or os.path.join(self.repo_path, ".anvil", "fix_receipts")
        if os.path.isabs(raw_receipt_dir):
            self.receipt_dir = os.path.abspath(raw_receipt_dir)
        else:
            self.receipt_dir = os.path.abspath(os.path.join(self.repo_path, raw_receipt_dir))
        self.toolchains = ToolchainManager(self.repo_path)
        self.registry = LanguageCapabilityRegistry(self.repo_path, toolchains=self.toolchains)
        self.classifier = FindingClassifier(self.repo_path)
        self.safety = FixSafetyPolicy()
        self.adapters = {
            "legacy_ruff": LegacyCompatAdapter(self.repo_path, "ruff", verifier.engines),
            "legacy_native": LegacyCompatAdapter(self.repo_path, "native", verifier.engines),
            "python_codemod": PythonCodemodAdapter(self.repo_path, verifier.engines),
            "clang_native": PlannedOnlyAdapter(
                self.repo_path, "clang formatter/fix adapter is install-gated"
            ),
            "cpp_semantic": PlannedOnlyAdapter(
                self.repo_path, "native semantic patch adapter is scaffolded but not installed"
            ),
            "rust_tooling": PlannedOnlyAdapter(
                self.repo_path, "rust remediation adapter is install-gated"
            ),
            "rust_semantic": PlannedOnlyAdapter(
                self.repo_path, "rust semantic adapter is install-gated"
            ),
            "go_tooling": PlannedOnlyAdapter(
                self.repo_path, "go remediation adapter is install-gated"
            ),
            "go_semantic": PlannedOnlyAdapter(
                self.repo_path, "go semantic adapter is install-gated"
            ),
            "java_tooling": PlannedOnlyAdapter(
                self.repo_path, "java remediation adapter is install-gated"
            ),
            "java_semantic": PlannedOnlyAdapter(
                self.repo_path, "java semantic adapter is install-gated"
            ),
            "kotlin_tooling": PlannedOnlyAdapter(
                self.repo_path, "kotlin remediation adapter is install-gated"
            ),
            "kotlin_semantic": PlannedOnlyAdapter(
                self.repo_path, "kotlin semantic adapter is install-gated"
            ),
            "web_formatter": PlannedOnlyAdapter(
                self.repo_path, "web formatter adapter is install-gated"
            ),
            "web_linter": PlannedOnlyAdapter(
                self.repo_path, "web linter adapter is install-gated"
            ),
            "web_codemod": PlannedOnlyAdapter(
                self.repo_path, "web codemod adapter is install-gated"
            ),
            "web_markup": PlannedOnlyAdapter(
                self.repo_path, "html remediation adapter is install-gated"
            ),
            "web_styles": PlannedOnlyAdapter(
                self.repo_path, "css/scss remediation adapter is install-gated"
            ),
            "web_semantic": PlannedOnlyAdapter(
                self.repo_path, "web semantic adapter is install-gated"
            ),
            "artifact_templates": ArtifactTemplateAdapter(self.repo_path, verifier.engines),
            "config_formatter": PlannedOnlyAdapter(
                self.repo_path, "config formatter adapter is install-gated"
            ),
            "config_schema": PlannedOnlyAdapter(
                self.repo_path, "config schema adapter is install-gated"
            ),
            "docs_formatter": PlannedOnlyAdapter(
                self.repo_path, "docs formatter adapter is install-gated"
            ),
            "docs_semantic": PlannedOnlyAdapter(
                self.repo_path, "docs semantic adapter is install-gated"
            ),
            "shell_formatter": PlannedOnlyAdapter(
                self.repo_path, "shell formatter adapter is install-gated"
            ),
        }

    def normalize_findings(
        self,
        findings: list[dict[str, Any]],
    ) -> list[FindingRecord]:
        normalized: list[FindingRecord] = []
        seen: set[tuple[str, int, str, str]] = set()
        for index, raw in enumerate(findings):
            raw_file = str(raw.get("file", "") or "").strip()
            rel_file, abs_path = self._normalize_path(raw_file)
            if not rel_file:
                rel_file = "."
            key = (
                rel_file,
                int(raw.get("line", 0) or 0),
                str(raw.get("rule_id", "UNKNOWN")),
                str(raw.get("message", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            capability = self.registry.detect(rel_file)
            finding = FindingRecord(
                finding_id=f"finding-{index}",
                file=rel_file,
                abs_path=abs_path,
                file_id=self._file_id(rel_file),
                line=int(raw.get("line", 0) or 0),
                rule_id=str(raw.get("rule_id", "UNKNOWN")),
                message=str(raw.get("message", "")),
                severity=str(raw.get("severity", "P2")),
                aal=str(raw.get("aal") or "AAL-3"),
                domain=list(raw.get("domain") or ["universal"]),
                closure_level=str(raw.get("closure_level", "")),
                context=str(raw.get("context", "")),
                engine=self._infer_engine(raw),
                language=capability.language,
                metadata={"capability": capability.to_dict()},
                raw=dict(raw),
            )
            fix_class, safety_tier = self.classifier.classify(finding, capability)
            finding.fix_class = fix_class
            finding.metadata["safety_tier"] = safety_tier
            normalized.append(finding)
        return normalized

    def plan(
        self,
        findings: list[FindingRecord],
        *,
        target_path: str,
        fix_mode: str,
        dry_run: bool,
        max_files: int | None = None,
    ) -> FixPlan:
        run_id = f"fix-{uuid.uuid4().hex[:12]}"
        buckets: dict[tuple[str, str, str, str], list[FindingRecord]] = {}
        files_seen: set[str] = set()
        for finding in findings:
            if max_files is not None and finding.file not in files_seen and len(files_seen) >= max_files:
                continue
            files_seen.add(finding.file)
            adapter_key, tool = self._select_adapter(finding)
            bucket_key = (
                adapter_key,
                finding.language,
                finding.file,
                self._batch_group(finding),
            )
            buckets.setdefault(bucket_key, []).append(finding)

        batches: list[FixBatch] = []
        batch_index = 0
        for (_, language, file_path, _), bucket in sorted(
            buckets.items(),
            key=lambda item: (item[0][1], item[0][2], item[0][0], item[0][3]),
        ):
            first = bucket[0]
            adapter_key, tool = self._select_adapter(first)
            safety_tier = str(first.metadata.get("safety_tier", "guarded-structured"))
            allowed, reason = self.safety.allows(
                FixBatch(
                    batch_id="probe",
                    adapter_key=adapter_key,
                    tool=tool,
                    language=language,
                    fix_class=first.fix_class,
                    safety_tier=safety_tier,
                    finding_ids=[],
                    rule_ids=[],
                    files=[file_path],
                    confidence=0.0,
                    supported=False,
                    toolchain_profile=str(
                        (first.metadata.get("capability") or {}).get("toolchain_profile") or ""
                    )
                    or None,
                    toolchain_state=str(
                        (first.metadata.get("capability") or {}).get("toolchain_state", "unmanaged")
                    ),
                    toolchain_source=str(
                        (first.metadata.get("capability") or {}).get(
                            "toolchain_source", "unmanaged"
                        )
                    ),
                    toolchain_tools=dict(
                        (first.metadata.get("capability") or {}).get("resolved_tools") or {}
                    ),
                    toolchain_message=str(
                        (first.metadata.get("capability") or {}).get("toolchain_message", "")
                    ),
                ),
                fix_mode=fix_mode,
            )
            adapter = self.adapters.get(adapter_key)
            probe_batch = FixBatch(
                batch_id="probe",
                adapter_key=adapter_key,
                tool=tool,
                language=language,
                fix_class=first.fix_class,
                safety_tier=safety_tier,
                finding_ids=[finding.finding_id for finding in bucket],
                rule_ids=sorted({finding.rule_id for finding in bucket}),
                files=[file_path],
                confidence=0.0,
                supported=False,
            )
            supported = adapter is not None and allowed and adapter.supports(probe_batch)
            if adapter is not None and allowed and not supported and not reason:
                reason = "adapter is registered but does not support this finding batch"
            if first.fix_class in {FIX_CLASS_RULE_TUNE, FIX_CLASS_MANUAL}:
                supported = False
                if not reason:
                    reason = "finding is plan-only and requires manual or rule-tuning follow-up"
            batch_index += 1
            batches.append(
                FixBatch(
                    batch_id=f"batch-{batch_index:04d}",
                    adapter_key=adapter_key,
                    tool=tool,
                    language=language,
                    fix_class=first.fix_class,
                    safety_tier=safety_tier,
                    finding_ids=[finding.finding_id for finding in bucket],
                    rule_ids=sorted({finding.rule_id for finding in bucket}),
                    files=[file_path],
                    confidence=self._batch_confidence(first.fix_class, supported),
                    supported=supported,
                    reason=reason,
                    toolchain_profile=str(
                        (first.metadata.get("capability") or {}).get("toolchain_profile") or ""
                    )
                    or None,
                    toolchain_state=str(
                        (first.metadata.get("capability") or {}).get("toolchain_state", "unmanaged")
                    ),
                    toolchain_source=str(
                        (first.metadata.get("capability") or {}).get(
                            "toolchain_source", "unmanaged"
                        )
                    ),
                    toolchain_tools=dict(
                        (first.metadata.get("capability") or {}).get("resolved_tools") or {}
                    ),
                    toolchain_message=str(
                        (first.metadata.get("capability") or {}).get("toolchain_message", "")
                    ),
                )
            )

        return FixPlan(
            run_id=run_id,
            target_path=target_path,
            fix_mode=fix_mode,
            dry_run=dry_run,
            batch_count=len(batches),
            finding_count=len(findings),
            batches=batches,
        )

    def execute(
        self,
        *,
        findings: list[dict[str, Any]],
        target_path: str,
        verification_kwargs: dict[str, Any],
        fix_mode: str = "safe",
        dry_run: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        normalized = self.normalize_findings(findings)
        normalized_by_id = {finding.finding_id: finding for finding in normalized}
        plan = self.plan(
            normalized,
            target_path=target_path,
            fix_mode=fix_mode,
            dry_run=dry_run,
            max_files=max_files,
        )

        receipt_dir = self._prepare_receipt_dir(plan.run_id)
        receipts: list[FixReceipt] = []
        fixed_total = 0
        relevant_profiles = sorted(
            {
                batch.toolchain_profile
                for batch in plan.batches
                if batch.toolchain_profile
            }
        )
        state_vectors = getattr(self.toolchains, "state_vectors", None)
        if callable(state_vectors):
            toolchain_state_vector = [
                vector.to_dict() for vector in state_vectors(relevant_profiles)
            ]
        else:
            toolchain_state_vector = []
        pre_verification = {
            "run_id": plan.run_id,
            "target_path": target_path,
            "fix_mode": fix_mode,
            "finding_count": len(normalized),
            "strict_aal": any(
                str(finding.aal).upper() in {"AAL-0", "AAL-1"} for finding in normalized
            ),
            "violation_ids": [finding.finding_id for finding in normalized],
        }

        for batch in plan.batches:
            adapter = self.adapters.get(batch.adapter_key)
            if adapter is None or not batch.supported:
                receipts.append(
                    FixReceipt(
                        receipt_id=f"{batch.batch_id}-receipt",
                        run_id=plan.run_id,
                        batch_id=batch.batch_id,
                        adapter_key=batch.adapter_key,
                        tool=batch.tool,
                        language=batch.language,
                        safety_tier=batch.safety_tier,
                        status="skipped" if not dry_run else "planned",
                        dry_run=dry_run,
                        finding_ids=batch.finding_ids,
                        rule_ids=batch.rule_ids,
                        files=batch.files,
                        confidence=batch.confidence,
                        error=batch.reason or "adapter unavailable",
                        toolchain_profile=batch.toolchain_profile,
                        toolchain_state=batch.toolchain_state,
                        toolchain_source=batch.toolchain_source,
                        toolchain_tools=batch.toolchain_tools,
                        toolchain_message=batch.toolchain_message,
                        verification_envelope={
                            "phase": "pre_mutation",
                            "status": "skipped",
                            "reason": batch.reason or "adapter unavailable",
                            "finding_ids": list(batch.finding_ids),
                        },
                        rollback_coordinates={
                            "receipt_dir": receipt_dir,
                            "rollback_bundle_path": None,
                        },
                    )
                )
                continue

            if batch.toolchain_profile:
                resolution = self.toolchains.resolve(
                    batch.toolchain_profile,
                    auto_bootstrap=not dry_run,
                )
                batch.toolchain_state = resolution.state
                batch.toolchain_source = resolution.source
                batch.toolchain_tools = dict(resolution.tool_paths)
                batch.toolchain_message = resolution.message
                if not resolution.installed:
                    receipts.append(
                        FixReceipt(
                            receipt_id=f"{batch.batch_id}-receipt",
                            run_id=plan.run_id,
                            batch_id=batch.batch_id,
                            adapter_key=batch.adapter_key,
                            tool=batch.tool,
                            language=batch.language,
                            safety_tier=batch.safety_tier,
                            status="skipped" if not dry_run else "planned",
                            dry_run=dry_run,
                            finding_ids=batch.finding_ids,
                            rule_ids=batch.rule_ids,
                            files=batch.files,
                            confidence=batch.confidence,
                            error=f"toolchain profile '{batch.toolchain_profile}' unavailable: {resolution.message}",
                            toolchain_profile=batch.toolchain_profile,
                            toolchain_state=batch.toolchain_state,
                            toolchain_source=batch.toolchain_source,
                            toolchain_tools=batch.toolchain_tools,
                            toolchain_message=batch.toolchain_message,
                            verification_envelope={
                                "phase": "pre_mutation",
                                "status": "skipped",
                                "reason": resolution.message,
                                "finding_ids": list(batch.finding_ids),
                            },
                            rollback_coordinates={
                                "receipt_dir": receipt_dir,
                                "rollback_bundle_path": None,
                            },
                        )
                    )
                    continue

            receipt = adapter.apply(
                batch=batch,
                findings=normalized,
                receipt_dir=receipt_dir,
                dry_run=dry_run,
            )
            receipt.run_id = plan.run_id
            receipt.toolchain_profile = batch.toolchain_profile
            receipt.toolchain_state = batch.toolchain_state
            receipt.toolchain_source = batch.toolchain_source
            receipt.toolchain_tools = dict(batch.toolchain_tools)
            receipt.toolchain_message = batch.toolchain_message
            receipt.rollback_coordinates = {
                "receipt_dir": receipt_dir,
                "rollback_bundle_path": receipt.rollback_bundle_path,
            }
            receipt.verification_envelope = {
                "phase": "post_mutation",
                "status": receipt.status,
                "finding_ids": list(batch.finding_ids),
                "rule_ids": list(batch.rule_ids),
                "strict_aal": any(
                    str(normalized_by_id[finding_id].aal).upper() in {"AAL-0", "AAL-1"}
                    for finding_id in batch.finding_ids
                    if finding_id in normalized_by_id
                ),
                "toolchain_profile": batch.toolchain_profile,
                "toolchain_state": batch.toolchain_state,
            }
            if not dry_run and receipt.changed_files:
                validation_target = os.path.join(self.repo_path, receipt.changed_files[0])
                receipt.validation = {
                    "post_batch_violation_count": len(
                        self.verifier.verify_all(path_arg=validation_target, **verification_kwargs)
                    )
                }
                receipt.verification_envelope["post_batch_violation_count"] = int(
                    receipt.validation.get("post_batch_violation_count", 0) or 0
                )
            receipts.append(receipt)
            fixed_total += receipt.fixed_count

        final_violations = (
            findings
            if dry_run
            else self.verifier.verify_all(path_arg=target_path, **verification_kwargs)
        )
        post_verification = {
            "run_id": plan.run_id,
            "target_path": target_path,
            "status": "pass" if not final_violations else "fail",
            "violation_count": len(final_violations),
        }
        receipt_summary = [
            {
                "receipt_id": receipt.receipt_id,
                "batch_id": receipt.batch_id,
                "status": receipt.status,
                "files": list(receipt.files),
                "changed_files": list(receipt.changed_files),
                "toolchain_profile": receipt.toolchain_profile,
                "toolchain_state": receipt.toolchain_state,
                "reverification_violations": int(
                    (receipt.validation or {}).get("post_batch_violation_count", 0) or 0
                ),
                "rollback_bundle_path": receipt.rollback_bundle_path,
                "receipt_dir": receipt_dir,
            }
            for receipt in receipts
        ]
        if not toolchain_state_vector:
            deduped_vectors: dict[str, dict[str, Any]] = {}
            for receipt in receipts:
                profile = str(receipt.toolchain_profile or "").strip()
                if not profile:
                    continue
                available_tools = sorted(
                    tool for tool, path in (receipt.toolchain_tools or {}).items() if path
                )
                missing_tools = sorted(
                    tool for tool, path in (receipt.toolchain_tools or {}).items() if not path
                )
                deduped_vectors[profile] = {
                    "profile": profile,
                    "qualification_state": (
                        "ready" if receipt.toolchain_state != "missing" else "missing"
                    ),
                    "state": receipt.toolchain_state,
                    "source": receipt.toolchain_source,
                    "installed": receipt.toolchain_state != "missing",
                    "available_tools": available_tools,
                    "missing_tools": missing_tools,
                    "bootstrap_attempted": False,
                    "bootstrap_skipped": False,
                    "message": receipt.toolchain_message,
                }
            toolchain_state_vector = list(deduped_vectors.values())
        receipts_path = os.path.join(receipt_dir, "receipts.json")
        Path(receipts_path).write_text(
            json.dumps([receipt.to_dict() for receipt in receipts], indent=2),
            encoding="utf-8",
        )
        return {
            "plan": plan.to_dict(),
            "fix_plan": plan.to_dict(),
            "normalized_findings": [finding.to_dict() for finding in normalized],
            "receipts": [receipt.to_dict() for receipt in receipts],
            "fix_receipts": [receipt.to_dict() for receipt in receipts],
            "receipt_summary": receipt_summary,
            "receipt_dir": receipt_dir,
            "final_violations": final_violations,
            "fixed": fixed_total,
            "receipts_path": receipts_path,
            "toolchain_state_vector": toolchain_state_vector,
            "verification_envelope": {
                "pre_mutation": pre_verification,
                "post_mutation": post_verification,
                "receipt_summary": receipt_summary,
            },
        }

    def rollback(self, receipt_dir: str) -> list[RollbackReceipt]:
        receipts_path = os.path.join(receipt_dir, "receipts.json")
        if not os.path.exists(receipts_path):
            return [
                RollbackReceipt(
                    receipt_id="unknown",
                    restored_files=[],
                    status="error",
                    error="receipts.json not found",
                )
            ]
        payload = json.loads(Path(receipts_path).read_text(encoding="utf-8"))
        results: list[RollbackReceipt] = []
        for receipt in payload:
            rollback_path = receipt.get("rollback_bundle_path")
            restored_files: list[str] = []
            if not rollback_path or not os.path.exists(rollback_path):
                results.append(
                    RollbackReceipt(
                        receipt_id=str(receipt.get("receipt_id", "unknown")),
                        restored_files=[],
                        status="skipped",
                    )
                )
                continue
            bundle = json.loads(Path(rollback_path).read_text(encoding="utf-8"))
            for rel_path, content in bundle.items():
                abs_path = os.path.join(self.repo_path, rel_path)
                if content is None:
                    if os.path.exists(abs_path):
                        os.remove(abs_path)
                else:
                    Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(abs_path).write_text(content, encoding="utf-8")
                restored_files.append(rel_path)
            results.append(
                RollbackReceipt(
                    receipt_id=str(receipt.get("receipt_id", "unknown")),
                    restored_files=restored_files,
                    status="restored",
                )
            )
        return results

    def _normalize_path(self, raw_file: str) -> tuple[str, str | None]:
        if not raw_file:
            return "", None
        candidate = raw_file
        if os.path.isabs(candidate):
            abs_path = os.path.abspath(candidate)
        else:
            abs_path = os.path.abspath(os.path.join(self.repo_path, candidate))
        try:
            rel_path = os.path.relpath(abs_path, self.repo_path).replace("\\", "/")
        except ValueError:
            rel_path = candidate.replace("\\", "/").lstrip("./")
        if rel_path.startswith(".."):
            rel_path = candidate.replace("\\", "/").lstrip("./")
        return rel_path, abs_path

    def _infer_engine(self, raw: dict[str, Any]) -> str:
        if raw.get("engine"):
            return str(raw["engine"]).lower()
        rule_id = str(raw.get("rule_id", ""))
        if rule_id.startswith("AES-"):
            return "aes"
        if rule_id.startswith("SEMANTIC-"):
            return "semantic"
        if rule_id == "MYPY":
            return "mypy"
        if rule_id == "VULTURE":
            return "vulture"
        return "ruff"

    @staticmethod
    def _file_id(rel_path: str) -> str:
        return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]

    def _select_adapter(self, finding: FindingRecord) -> tuple[str, str]:
        capability = dict(finding.metadata.get("capability") or {})
        if finding.fix_class == FIX_CLASS_STOCK_LINTER:
            adapter_key = str(capability.get("fix_adapter") or "")
            if adapter_key:
                tool = "ruff" if adapter_key == "legacy_ruff" else adapter_key
                return adapter_key, tool
        if finding.fix_class == FIX_CLASS_NATIVE_SEMANTIC:
            adapter_key = str(capability.get("semantic_patch_adapter") or "cpp_semantic")
            return adapter_key, "clang-tidy" if adapter_key == "cpp_semantic" else adapter_key
        if finding.fix_class == FIX_CLASS_AST_CODEMOD:
            adapter_key = str(capability.get("codemod_adapter") or "python_codemod")
            return adapter_key, "python-ast" if adapter_key == "python_codemod" else adapter_key
        if finding.fix_class == FIX_CLASS_AES_ARTIFACT:
            return "artifact_templates", "aes"
        if finding.language in {"json", "yaml", "toml"}:
            return "config_formatter", "formatter"
        if finding.language in {"md", "txt"}:
            return "docs_formatter", "formatter"
        if finding.language == "shell":
            return "shell_formatter", "shfmt"
        if capability.get("fix_adapter"):
            adapter_key = str(capability["fix_adapter"])
            return adapter_key, adapter_key
        return "manual", "manual"

    def _batch_group(self, finding: FindingRecord) -> str:
        adapter_key, _ = self._select_adapter(finding)
        if adapter_key != "python_codemod":
            return finding.fix_class or finding.rule_id
        if finding.rule_id.startswith("D"):
            return "python_docstrings"
        if finding.rule_id.startswith("ANN"):
            return "python_annotations"
        if finding.rule_id == "AES-PY-3":
            return "python_mutable_defaults"
        if finding.rule_id == "AES-CR-2":
            return "python_bare_except"
        if finding.rule_id == "AES-PY-6":
            return "python_open_context"
        if finding.rule_id == "AES-ERR-2":
            return "python_exception_contracts"
        return finding.rule_id

    @staticmethod
    def _batch_confidence(fix_class: str, supported: bool) -> float:
        if not supported:
            return 0.0
        if fix_class == FIX_CLASS_STOCK_LINTER:
            return 0.9
        if fix_class == FIX_CLASS_AES_ARTIFACT:
            return 0.75
        if fix_class == FIX_CLASS_AST_CODEMOD:
            return 0.65
        if fix_class == FIX_CLASS_NATIVE_SEMANTIC:
            return 0.6
        return 0.25

    def _prepare_receipt_dir(self, run_id: str) -> str:
        receipt_dir = os.path.join(self.receipt_dir, run_id)
        os.makedirs(receipt_dir, exist_ok=True)
        return receipt_dir
