from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from core.aes.compliance_context import ComplianceContext


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    passed: bool
    status: str
    required_artifacts: list[str]
    missing_artifacts: list[str]
    message: str
    skipped_reason: str = ""


class RuntimeGate:
    gate_id = "base_gate"
    _schema_root = Path(__file__).resolve().parents[3] / "standards" / "schemas"

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return True

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return []

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return None

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: Any,
        thresholds: dict[str, Any],
        compliance_context: ComplianceContext,
    ) -> str | None:
        return None

    @staticmethod
    def _normalize_path_values(values: Any) -> set[str]:
        if not isinstance(values, list):
            return set()
        normalized: set[str] = set()
        for item in values:
            text = str(item or "").strip().replace("\\", "/").lstrip("./")
            if text:
                normalized.add(text)
        return normalized

    @staticmethod
    def _validate_context_field(
        payload: dict[str, Any],
        payload_key: str,
        expected: Any,
        *,
        required: bool = True,
    ) -> str | None:
        if expected in {None, ""} and not required:
            return None
        if payload_key not in payload:
            return f"missing {payload_key}"
        actual = str(payload.get(payload_key) or "").strip()
        expected_text = str(expected or "").strip()
        if not actual:
            return f"missing {payload_key}"
        if expected_text and actual != expected_text:
            return (
                f"{payload_key} does not match compliance context "
                f"({actual} != {expected_text})"
            )
        return None

    def _validate_path_coverage(
        self,
        payload: dict[str, Any],
        payload_key: str,
        expected_paths: list[str],
    ) -> str | None:
        expected = self._normalize_path_values(expected_paths)
        if not expected:
            return None
        actual = self._normalize_path_values(payload.get(payload_key))
        if not actual:
            return f"missing {payload_key} coverage for changed files"
        missing = sorted(expected - actual)
        if missing:
            return f"{payload_key} missing changed file refs: {', '.join(missing)}"
        return None

    def _resolve_artifact_path(
        self, repo_root: str, compliance_context: ComplianceContext, artifact: str
    ) -> Path | None:
        repo_path = Path(repo_root)
        candidates = (
            repo_path / ".anvil" / "compliance" / compliance_context.run_id / artifact,
            repo_path / artifact,
            repo_path / "standards" / artifact,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def run(
        self,
        repo_root: str,
        compliance_context: ComplianceContext,
        thresholds: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        required = self.required_artifacts(compliance_context)
        missing: list[str] = []
        invalid: dict[str, str] = {}
        resolved: dict[str, str] = {}
        thresholds = thresholds or {}
        for artifact in required:
            path = self._resolve_artifact_path(repo_root, compliance_context, artifact)
            if path is None:
                missing.append(artifact)
                continue
            resolved[artifact] = str(path)
            if path.suffix != ".json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                invalid[artifact] = f"invalid json: {exc}"
                continue

            schema_name = self.schema_for_artifact(artifact, compliance_context)
            if schema_name:
                schema_path = Path(repo_root) / "standards" / "schemas" / schema_name
                if not schema_path.exists():
                    schema_path = self._schema_root / schema_name
                try:
                    schema = json.loads(schema_path.read_text(encoding="utf-8"))
                    jsonschema.validate(instance=payload, schema=schema)
                except Exception as exc:
                    invalid[artifact] = f"schema validation failed: {exc}"
                    continue

            payload_error = self.validate_artifact_payload(
                artifact=artifact,
                payload=payload,
                thresholds=thresholds,
                compliance_context=compliance_context,
            )
            if payload_error:
                invalid[artifact] = payload_error
        return {
            "gate_id": self.gate_id,
            "required_artifacts": required,
            "missing_artifacts": missing,
            "invalid_artifacts": invalid,
            "resolved_artifacts": resolved,
            "context": compliance_context.to_dict(),
        }

    def evaluate(
        self, report: dict[str, Any], thresholds: dict[str, Any] | None = None
    ) -> GateResult:
        missing = [str(item) for item in report.get("missing_artifacts", []) or []]
        invalid = {
            str(key): str(value)
            for key, value in (report.get("invalid_artifacts", {}) or {}).items()
        }
        if invalid:
            missing.extend(invalid.keys())
        if not missing:
            message = "ok"
        elif invalid:
            message = "; ".join(
                ["missing or invalid required runtime artifacts"]
                + [f"{artifact}: {reason}" for artifact, reason in invalid.items()]
            )
        else:
            message = "missing required runtime artifacts"
        return GateResult(
            gate_id=self.gate_id,
            passed=not missing,
            status="passed" if not missing else "failed",
            required_artifacts=[
                str(item) for item in report.get("required_artifacts", []) or []
            ],
            missing_artifacts=missing,
            message=message,
        )
