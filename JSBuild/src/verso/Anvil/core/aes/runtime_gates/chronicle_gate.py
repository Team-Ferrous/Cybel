from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class ChronicleGate(RuntimeGate):
    gate_id = "chronicle_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["chronicle.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return bool(compliance_context.hot_paths)

    def schema_for_artifact(
        self, artifact: str, compliance_context: ComplianceContext
    ) -> str | None:
        return "chronicle_report.schema.json" if artifact == "chronicle.json" else None

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        if artifact != "chronicle.json" or not isinstance(payload, dict):
            return None
        perf = thresholds.get("hot_path_perf", {})
        if not isinstance(perf, dict):
            return None
        block = perf.get("regression_block_percent")
        regression = payload.get("regression_percent")
        if isinstance(block, (int, float)) and isinstance(regression, (int, float)):
            if float(regression) > float(block):
                return (
                    "regression_percent exceeds AES blocking threshold "
                    f"({regression} > {block})"
                )
        return None
