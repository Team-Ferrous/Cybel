from __future__ import annotations

from core.campaign.control_plane import CampaignControlPlane
from core.env_manager import EnvironmentManager


def test_native_capability_profile_and_governance_proposals(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        EnvironmentManager,
        "_native_capability_manifest",
        staticmethod(
            lambda: {
                "schema_version": "native_capability_manifest.v1",
                "summary": {"capability_count": 1, "available_count": 1, "degraded_count": 0},
                "capabilities": [{"capability": "native_ops", "status": "available"}],
            }
        ),
    )
    profile = EnvironmentManager(root_dir=str(tmp_path)).capture_profile()
    assert profile["native_capabilities"]["summary"]["available_count"] == 1

    control = CampaignControlPlane.create(
        "governance_test",
        "Governance Test",
        str(tmp_path / "campaigns"),
        objective="Turn telemetry into draft rules",
        root_dir=str(tmp_path),
    )
    monkeypatch.setattr(
        control.audit_engine,
        "run",
        lambda scope="operator": {"findings": [], "summary": {"finding_count": 0}},
    )
    control.state_store.record_telemetry(
        control.campaign_id,
        telemetry_kind="verification_lane",
        payload={"all_passed": False, "counterexamples": ["failed sentinel"]},
    )

    result = control.run_audit()
    proposals = result["rule_proposals"]

    assert proposals["proposal_count"] >= 1
    assert proposals["path"].endswith("_rule_proposals.json")
