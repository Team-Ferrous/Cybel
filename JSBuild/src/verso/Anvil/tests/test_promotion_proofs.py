import time

from core.campaign.control_plane import CampaignControlPlane


def test_campaign_control_plane_emits_completion_proof_bundle(tmp_path):
    control = CampaignControlPlane.create(
        f"proof_{int(time.time())}",
        "Promotion Proof",
        str(tmp_path / "campaigns"),
        objective="Emit a closure proof",
        root_dir=str(tmp_path),
    )
    control.event_store.emit(
        event_type="campaign.verification_lane",
        payload={"all_passed": True, "artifacts": ["artifacts/verification/report.json"]},
        source="test",
        run_id=control.campaign_id,
    )

    proof = control.build_completion_proof()

    assert proof["proof"]["closure_allowed"] in {True, False}
    assert proof["mission_capsule"]["artifact_refs"] == ["artifacts/verification/report.json"]
    assert proof["path"].endswith(".json")
