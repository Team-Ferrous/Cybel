from __future__ import annotations

from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane


def test_campaign_closure_proof_binds_replay_checks_and_latent_evidence(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"closure-{uuid4().hex[:8]}",
        "Closure Safety Case",
        str(tmp_path / "campaigns"),
        objective="Compile a closure proof from replay evidence.",
        root_dir=str(tmp_path / "repo"),
    )

    control.run_research(
        sources=[
            {
                "topic": "telemetry",
                "url": "memory://telemetry",
                "title": "Telemetry",
                "content": "Telemetry contracts keep lane evidence complete.",
                "summary": "Telemetry contracts keep lane evidence complete.",
            }
        ]
    )
    eid = control.run_eid()
    control.state_store.record_roadmap_item(
        {
            "campaign_id": control.campaign_id,
            "item_id": "roadmap-closure",
            "phase_id": "development",
            "title": "Close the governed loop",
            "type": "experiment_lane",
            "repo_scope": ["target"],
            "owner_type": "ExperimentLane",
            "depends_on": [],
            "description": "Closure lane.",
            "objective": "Close the loop.",
            "status": "completed",
            "success_metrics": ["closure_allowed"],
            "required_evidence": ["eid_summary"],
            "required_artifacts": ["closure"],
            "telemetry_contract": {},
            "allowed_writes": ["target"],
            "promotion_gate": {},
            "exit_gate": {},
            "metadata": {},
        }
    )
    control.state_store.record_completion_check(
        {
            "campaign_id": control.campaign_id,
            "check_id": "closure-check-1",
            "passed": True,
            "summary": "Replay and telemetry witnesses present.",
        }
    )
    control.event_store.emit(
        "campaign.phase",
        {"phase": "eid"},
        source="test",
        run_id=control.campaign_id,
    )
    control.event_store.record_checkpoint(
        control.campaign_id,
        "eid",
        "completed",
        metadata={"artifact": "eid_summary"},
        artifacts=["artifacts/experiments/eid_summary.json"],
    )

    replay = control.event_store.export_run(
        control.campaign_id,
        output_path=str(tmp_path / "closure_replay.json"),
    )
    closure = control.build_completion_proof()

    assert eid["eid"]["latent_capture"]["latent_package_id"].startswith("latent_")
    assert replay["replay"]["inspectable_without_model"] is True
    assert replay["replay"]["checkpoint_count"] == 1
    assert closure["proof"]["completed_roadmap_items"] == 1
    assert closure["proof"]["telemetry_event_count"] >= 1
    assert closure["proof"]["completion_checks"][0]["check_id"] == "closure-check-1"
    assert closure["proof"]["closure_allowed"] is True


def test_audit_findings_block_closure_safety_case_when_roadmap_final_is_missing(
    tmp_path,
) -> None:
    control = CampaignControlPlane.create(
        f"closure-audit-{uuid4().hex[:8]}",
        "Closure Audit",
        str(tmp_path / "campaigns"),
        objective="Surface blocking closure findings.",
        root_dir=str(tmp_path / "repo"),
    )

    audit = control.run_audit()
    closure = control.build_completion_proof()

    assert audit["summary"]["finding_count"] == 1
    assert audit["findings"][0]["severity"] == "high"
    assert closure["proof"]["open_material_findings"] == 1
    assert closure["proof"]["closure_allowed"] is False
