from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class TraceabilityGate(RuntimeGate):
    gate_id = "traceability_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["traceability.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return compliance_context.aal in {"AAL-0", "AAL-1"}

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return "traceability.schema.json" if artifact == "traceability.json" else None

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        del thresholds
        if artifact != "traceability.json" or not isinstance(payload, dict):
            return None
        for key, expected, required in (
            ("trace_id", compliance_context.trace_id, True),
            ("run_id", compliance_context.run_id, True),
            ("aal", compliance_context.aal, True),
            ("evidence_bundle_id", compliance_context.evidence_bundle_id, False),
        ):
            error = self._validate_context_field(
                payload,
                key,
                expected,
                required=required,
            )
            if error:
                return error
        return self._validate_path_coverage(
            payload,
            "changed_files",
            compliance_context.changed_files,
        ) or self._validate_path_coverage(
            payload,
            "code_refs",
            compliance_context.changed_files,
        )
