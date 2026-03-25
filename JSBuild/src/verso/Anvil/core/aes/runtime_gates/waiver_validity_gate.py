from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class WaiverValidityGate(RuntimeGate):
    gate_id = "waiver_validity_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        if compliance_context.waiver_ids:
            return ["waivers.json"]
        return []

