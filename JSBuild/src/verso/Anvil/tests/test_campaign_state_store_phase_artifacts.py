from core.campaign.state_store import CampaignStateStore


def test_phase_artifacts_and_convergence_persist_across_reopen(tmp_path):
    db_path = tmp_path / "state.db"
    store = CampaignStateStore(str(db_path))

    store.record_phase_artifact(
        "campaign-1",
        "01_intake",
        "campaign-1:intake_manifest",
        "manifest",
        "/tmp/intake.json",
        metadata={"source": "test"},
    )
    store.record_convergence_checkpoint(
        "campaign-1",
        "research",
        1,
        {"claim_count": 3, "remaining_frontier": 0},
        converged=True,
    )
    store.close()

    reopened = CampaignStateStore(str(db_path))
    phase_artifacts = reopened.list_phase_artifacts("campaign-1")
    convergence = reopened.list_convergence_checkpoints("campaign-1", "research")

    assert len(phase_artifacts) == 1
    assert phase_artifacts[0]["phase_id"] == "01_intake"
    assert phase_artifacts[0]["metadata"]["source"] == "test"
    assert len(convergence) == 1
    assert convergence[0]["metrics"]["claim_count"] == 3
    assert convergence[0]["converged"] == 1
