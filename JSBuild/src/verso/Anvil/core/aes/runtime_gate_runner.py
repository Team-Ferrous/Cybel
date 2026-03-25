from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import GateResult, RuntimeGate
from core.aes.runtime_gates.chronicle_gate import ChronicleGate
from core.aes.runtime_gates.domain_report_gate import DomainReportGate
from core.aes.runtime_gates.evidence_closure_gate import EvidenceClosureGate
from core.aes.runtime_gates.review_independence_gate import ReviewIndependenceGate
from core.aes.runtime_gates.supply_chain_gate import SupplyChainGate
from core.aes.runtime_gates.telemetry_contract_gate import TelemetryContractGate
from core.aes.runtime_gates.traceability_gate import TraceabilityGate
from core.aes.runtime_gates.waiver_validity_gate import WaiverValidityGate


@dataclass(frozen=True)
class RuntimeGateSummary:
    passed: bool
    results: list[GateResult]
    missing_artifacts: list[str]


class RuntimeGateRunner:
    def __init__(self, repo_root: str) -> None:
        self.repo_root = Path(repo_root).resolve()
        self._gates: dict[str, RuntimeGate] = {
            gate.gate_id: gate
            for gate in (
                TraceabilityGate(),
                EvidenceClosureGate(),
                ReviewIndependenceGate(),
                WaiverValidityGate(),
                ChronicleGate(),
                TelemetryContractGate(),
                SupplyChainGate(),
                DomainReportGate(),
            )
        }

    def evaluate(
        self,
        context: ComplianceContext,
        gate_ids: list[str],
        thresholds: dict[str, Any] | None = None,
    ) -> RuntimeGateSummary:
        results: list[GateResult] = []
        missing_artifacts: list[str] = []
        thresholds = thresholds or {}

        for gate_id in gate_ids:
            gate = self._gates.get(gate_id)
            if gate is None:
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        passed=False,
                        status="failed",
                        required_artifacts=[],
                        missing_artifacts=[],
                        message=f"Unknown runtime gate: {gate_id}",
                    )
                )
                continue
            if not gate.applies(context):
                results.append(
                    GateResult(
                        gate_id=gate_id,
                        passed=True,
                        status="skipped",
                        required_artifacts=gate.required_artifacts(context),
                        missing_artifacts=[],
                        message=f"Skipped runtime gate {gate_id}",
                        skipped_reason="not_applicable",
                    )
                )
                continue
            report = gate.run(str(self.repo_root), context, thresholds=thresholds)
            result = gate.evaluate(report, thresholds)
            results.append(result)
            if result.status != "skipped":
                missing_artifacts.extend(result.missing_artifacts)

        return RuntimeGateSummary(
            passed=all(result.passed for result in results),
            results=results,
            missing_artifacts=list(dict.fromkeys(missing_artifacts)),
        )
