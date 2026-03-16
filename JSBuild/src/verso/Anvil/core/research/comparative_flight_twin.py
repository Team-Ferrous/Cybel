from __future__ import annotations

from typing import Any


class ComparativeFlightTwin:
    """Static risk twin for comparative migration programs."""

    def simulate(
        self,
        *,
        program: dict[str, Any],
        target_pack: dict[str, Any],
    ) -> dict[str, Any]:
        impact = dict(program.get("impact_assessment") or {})
        affected_tests = list(program.get("affected_tests") or [])
        build_depth = str(
            ((target_pack.get("build_fingerprint") or {}).get("build_fingerprint_depth") or "shallow")
        )
        risk = float(impact.get("impact_score") or 0.0)
        if build_depth == "shallow":
            risk += 0.08
        if not affected_tests:
            risk += 0.07
        risk = min(0.99, risk)
        return {
            "schema_version": "comparative_flight_twin.v1",
            "predicted_breakage_risk": round(risk, 4),
            "simulated_test_fallout": max(1, len(affected_tests) or 1),
            "rollback_trigger_rate": round(min(0.95, risk * 0.8), 4),
            "risk_level": "high" if risk >= 0.72 else ("medium" if risk >= 0.42 else "low"),
            "monitors": [
                "targeted tests",
                "build graph consistency",
                "saguaro semantic verification",
            ],
        }
