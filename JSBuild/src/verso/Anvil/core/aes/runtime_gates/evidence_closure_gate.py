from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class EvidenceClosureGate(RuntimeGate):
    gate_id = "evidence_closure_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["evidence_bundle.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return compliance_context.aal in {"AAL-0", "AAL-1"}

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return (
            "evidence_bundle.schema.json" if artifact == "evidence_bundle.json" else None
        )

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        del thresholds
        if artifact != "evidence_bundle.json" or not isinstance(payload, dict):
            return None
        for key, expected, required in (
            ("bundle_id", compliance_context.evidence_bundle_id, False),
            ("change_id", compliance_context.run_id, True),
            ("trace_id", compliance_context.trace_id, True),
            ("aal", compliance_context.aal, True),
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
        )
