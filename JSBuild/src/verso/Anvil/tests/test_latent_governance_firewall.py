from __future__ import annotations

from core.aes import ComplianceContext
from core.aes.runtime_gates.telemetry_contract_gate import TelemetryContractGate


def test_latent_governance_firewall_requires_declared_telemetry_fields() -> None:
    gate = TelemetryContractGate()
    context = ComplianceContext(run_id="run-1", hot_paths=["core/qsg/grover.py"])
    thresholds = {
        "telemetry": {
            "required_fields": ["capability_digest", "delta_watermark"],
        }
    }

    failure = gate.validate_artifact_payload(
        "telemetry_contract.json",
        {"required_fields": ["capability_digest"]},
        thresholds,
        context,
    )
    success = gate.validate_artifact_payload(
        "telemetry_contract.json",
        {"required_fields": ["capability_digest", "delta_watermark"]},
        thresholds,
        context,
    )

    assert "delta_watermark" in str(failure)
    assert success is None
