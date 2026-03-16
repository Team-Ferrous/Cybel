from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate


class SupplyChainGate(RuntimeGate):
    gate_id = "supply_chain_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return ["sbom.json"]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return bool(compliance_context.dependency_changes)

