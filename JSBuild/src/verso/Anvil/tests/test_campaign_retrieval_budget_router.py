from __future__ import annotations

from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane


def test_retrieval_policy_records_saguaro_and_fallback_routes(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"retrieval-{uuid4().hex[:8]}",
        "Retrieval Policy",
        str(tmp_path / "campaigns"),
        objective="Keep retrieval budget explicit.",
        root_dir=str(tmp_path / "repo"),
    )

    saguaro_decision = control.retrieval_policy.decide(
        campaign_id=control.campaign_id,
        query="repo dossier summary",
    )
    fallback_decision = control.retrieval_policy.decide(
        campaign_id=control.campaign_id,
        query="browser fallback",
        fallback_reason="index_unavailable",
        evidence_quality="medium",
    )

    telemetry = control.state_store.list_telemetry(control.campaign_id)
    replay = control.event_store.export_run(
        control.campaign_id,
        output_path=str(tmp_path / "retrieval_replay.json"),
    )

    assert saguaro_decision.route == "saguaro"
    assert saguaro_decision.reason == "saguaro_authoritative"
    assert fallback_decision.route == "fallback"
    assert fallback_decision.reason == "index_unavailable"
    assert [item["route"] for item in telemetry if item["telemetry_kind"] == "retrieval_policy"] == [
        "saguaro",
        "fallback",
    ]
    assert [
        event["event_type"]
        for event in replay["events"]
        if event["event_type"] == "campaign.retrieval_policy"
    ] == ["campaign.retrieval_policy", "campaign.retrieval_policy"]


def test_retrieval_policy_renders_repo_dossier_brief_with_reuse_and_risk_signals(
    tmp_path,
) -> None:
    control = CampaignControlPlane.create(
        f"dossier-{uuid4().hex[:8]}",
        "Repo Dossier",
        str(tmp_path / "campaigns"),
        objective="Summarize repo-native evidence.",
        root_dir=str(tmp_path / "repo"),
    )

    markdown, payload = control.retrieval_policy.render_repo_dossier_brief(
        campaign_id=control.campaign_id,
        repos=[
            {
                "repo_id": "target",
                "role": "target",
                "local_path": str(tmp_path / "repo"),
            }
        ],
        repo_dossiers=[
            {
                "repo_id": "target",
                "entry_points": ["main.py"],
                "build_files": ["setup.py"],
                "test_files": ["tests/test_campaign_control_kernel.py"],
                "reuse_candidates": [{"path": "core/campaign/control_plane.py"}],
                "risk_signals": [{"kind": "blast_radius", "level": "medium"}],
                "tech_stack": ["python", "sqlite"],
            }
        ],
    )

    assert payload["repo_count"] == 1
    assert payload["repos"][0]["reuse_candidates"][0]["path"] == "core/campaign/control_plane.py"
    assert payload["repos"][0]["risk_signals"][0]["kind"] == "blast_radius"
    assert "# Repo Dossier Brief" in markdown
    assert "main.py" in markdown
