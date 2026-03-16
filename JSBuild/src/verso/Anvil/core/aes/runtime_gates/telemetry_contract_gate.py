from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class TelemetryContractGate(RuntimeGate):
    gate_id = "telemetry_contract_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["telemetry_contract.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return bool(compliance_context.hot_paths)

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return "telemetry_contract.schema.json" if artifact == "telemetry_contract.json" else None

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        if artifact != "telemetry_contract.json" or not isinstance(payload, dict):
            return None
        required = (
            thresholds.get("telemetry", {}).get("required_fields", [])
            if isinstance(thresholds.get("telemetry", {}), dict)
            else []
        )
        declared = {str(item) for item in payload.get("required_fields", []) or []}
        missing = [field for field in required if str(field) not in declared]
        if missing:
            return f"missing telemetry required_fields entries: {', '.join(missing)}"
        return None
