from __future__ import annotations

from core.research.comparative_flight_twin import ComparativeFlightTwin
from core.research.eid_master import EIDMasterLoop


def test_flight_twin_predicts_static_risk() -> None:
    twin = ComparativeFlightTwin()
    result = twin.simulate(
        program={
            "impact_assessment": {"impact_score": 0.6},
            "affected_tests": [],
        },
        target_pack={"build_fingerprint": {"build_fingerprint_depth": "shallow"}},
    )
    assert result["predicted_breakage_risk"] >= 0.6
    assert result["risk_level"] in {"medium", "high"}


def test_eid_prefers_phase_packets_over_programs() -> None:
    actions = EIDMasterLoop._repo_dossier_actions(
        [
            {
                "repo_id": "candidate",
                "phase_packets": [
                    {
                        "phase_id": "development",
                        "objective": "reporting",
                        "telemetry_contract": {"portfolio_rank_score": 0.9},
                        "allowed_writes": ["Saguaro/saguaro/analysis/report.py"],
                    }
                ],
                "native_migration_programs": [
                    {"feature_family": "reporting", "priority": 0.8}
                ],
            }
        ]
    )
    assert actions[0]["phase_id"] == "development"
