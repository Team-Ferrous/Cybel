from __future__ import annotations

from core.aes.compliance_context import ComplianceContext
from core.aes.runtime_gates.base import RuntimeGate

_DOMAIN_TO_REPORT = {
    "ml": "domain_reports/ml_run_manifest.json",
    "hpc": "domain_reports/hpc_report.json",
    "quantum": "domain_reports/quantum_report.json",
    "physics": "domain_reports/physics_report.json",
}


class DomainReportGate(RuntimeGate):
    gate_id = "domain_report_gate"

    def required_artifacts(self, compliance_context: ComplianceContext) -> list[str]:
        return [
            report
            for domain, report in _DOMAIN_TO_REPORT.items()
            if domain in set(compliance_context.domains)
        ]

    def applies(self, compliance_context: ComplianceContext) -> bool:
        return bool(set(compliance_context.domains).intersection(_DOMAIN_TO_REPORT))

    def validate_artifact_payload(
        self,
        artifact: str,
        payload: object,
        thresholds: dict[str, object],
        compliance_context: ComplianceContext,
    ) -> str | None:
        if not isinstance(payload, dict):
            return "domain report must be a JSON object"
        domain_thresholds = thresholds.get("domain_reports", {})
        if not isinstance(domain_thresholds, dict):
            return None
        for domain, report_path in _DOMAIN_TO_REPORT.items():
            if artifact != report_path or domain not in compliance_context.domains:
                continue
            required = domain_thresholds.get(domain, {})
            if not isinstance(required, dict):
                return None
            missing = [
                str(item)
                for item in required.get("required", []) or []
                if str(item) not in payload
            ]
            if missing:
                return (
                    f"{domain} domain report is missing required keys: {', '.join(missing)}"
                )
        return None
