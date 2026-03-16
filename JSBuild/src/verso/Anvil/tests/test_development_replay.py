from core.campaign.control_plane import CampaignControlPlane
from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGInferenceEngine


def test_control_plane_development_replay_uses_almf(tmp_path):
    campaigns_dir = tmp_path / "campaigns"
    control = CampaignControlPlane.create(
        "campaign-1",
        "Campaign 1",
        str(campaigns_dir),
        objective="Improve telemetry replay",
        root_dir=str(tmp_path),
    )
    control.run_research(
        sources=[
            {
                "topic": "telemetry",
                "url": "memory://telemetry",
                "title": "Telemetry",
                "content": "A contract-driven telemetry layer reduces missing metrics.",
                "summary": "Telemetry contract reduces missing metrics.",
            }
        ]
    )
    eid = control.run_eid()
    hypothesis_id = eid["hypotheses"][0]["hypothesis_id"]
    hypothesis_memory_id = control.memory_fabric.resolve_alias(
        campaign_id="campaign-1",
        source_table="hypotheses",
        source_id=hypothesis_id,
    )
    control.latent_bridge.capture_summary_package(
        memory_id=hypothesis_memory_id,
        summary_text="development replay branch",
        capture_stage="hypothesis_ranking",
    )
    engine = QSGInferenceEngine(
        config=QSGConfig(continuous_batching_enabled=True, batch_wait_timeout_ms=1),
        stream_producer=lambda request: iter(["resume"]),
    )

    result = control.run_development_replay(hypothesis_id, engine=engine)

    assert result["memory_id"] == hypothesis_memory_id
    assert result["replay"]["restored"] is True
    assert result["replay"]["memory_tier_decision"]["selected_tier"] == "latent_replay"
    assert result["memory_tier_decision"]["selected_tier"] == "latent_replay"
    assert result["repo_delta_memory"]["source_stage"] == "development_replay"
    descriptor = result["mission_replay_descriptor"]
    assert descriptor["memory_id"] == hypothesis_memory_id
    assert descriptor["restored"] is True
    assert descriptor["repo_delta_memory_id"]
    assert descriptor["replay_tape_path"]
    assert result["evidence_results"]
