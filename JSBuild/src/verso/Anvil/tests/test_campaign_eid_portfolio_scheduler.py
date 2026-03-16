from __future__ import annotations

from uuid import uuid4

from core.campaign.control_plane import CampaignControlPlane
from core.research.eid_scheduler import EIDScheduler


def test_eid_scheduler_assigns_portfolio_roles_and_blast_radius() -> None:
    scheduler = EIDScheduler()
    ranked_hypothesis = {
        "hypothesis_id": "hyp-1",
        "statement": "Hardware telemetry architecture for repo cache determinism",
        "source_basis": ["telemetry", "repo analysis"],
        "required_experiments": ["telemetry_contract_replay"],
        "risk": "low implementation complexity",
        "target_subsystems": ["repo_cache", "telemetry"],
        "promotable": True,
        "innovation_score": 4.2,
    }

    packets = scheduler.schedule(
        "Improve cpu telemetry determinism",
        [ranked_hypothesis],
        repo_dossiers=[{"repo_id": "target"}],
    )
    proposals = scheduler.build_proposals(
        "Improve cpu telemetry determinism",
        [ranked_hypothesis],
        packets,
    )
    roles = {packet["specialist_role"] for packet in packets}

    assert {
        "counterfactual_strategist",
        "determinism_compliance",
        "hardware_optimization",
        "hypothesis_generator",
        "software_architecture",
        "telemetry_systems",
    } <= roles
    assert proposals[0]["blast_radius"]["editable_scope"] == [
        "artifacts/experiments",
        "artifacts/telemetry",
    ]
    assert proposals[0]["promotable"] is True
    assert proposals[0]["fallback_path"].endswith("repo_cache, telemetry.")


def test_eid_master_emits_ranked_tracks_simulator_plans_and_latent_capture(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"eid-portfolio-{uuid4().hex[:8]}",
        "EID Portfolio",
        str(tmp_path / "campaigns"),
        objective="Run simulator-first native telemetry work.",
        root_dir=str(tmp_path / "repo"),
    )

    result = control.eid_master.run(
        "Run simulator-first native telemetry work.",
        [
            {
                "hypothesis_id": "hyp-hardware",
                "statement": "Hardware telemetry simulator path",
                "motivation": "Search space must be bounded before implementation.",
                "source_basis": ["telemetry", "repo analysis"],
                "target_subsystems": ["telemetry", "repo_cache"],
                "required_experiments": ["telemetry_contract_replay"],
                "risk": "low implementation complexity",
            },
            {
                "hypothesis_id": "hyp-inverse",
                "statement": "Inverse design benchmark for deterministic runtime",
                "motivation": "Search space uncertainty should be reduced first.",
                "source_basis": ["determinism"],
                "target_subsystems": ["campaign_runtime"],
                "required_experiments": ["artifact_resume_replay"],
                "risk": "medium implementation complexity",
            },
        ],
        repo_dossiers=[
            {
                "repo_id": "target",
                "frontier_packets": [
                    {
                        "packet_id": "target:frontier:1",
                        "title": "Experiment native_rewrite for core/campaign/control_plane.py",
                        "priority": 0.82,
                        "posture": "native_rewrite",
                        "source_path": "core/campaign/control_plane.py",
                        "target_path": "core/campaign/control_plane.py",
                        "recommended_tracks": ["comparative_spike"],
                    }
                ],
            }
        ],
        execute_tracks=False,
        workspace_root=control.workspace.root_dir,
        metadata_path=control.workspace.metadata_path,
    )

    plan_ids = {plan["plan_id"] for plan in result["simulator_plans"]}

    assert "simulator_first" in plan_ids
    assert "hardware_fit_eval" in plan_ids
    assert "inverse_design_loop" in plan_ids
    assert result["experimental_tracks"][0]["metadata"]["priority"] >= result["experimental_tracks"][-1]["metadata"]["priority"]
    assert result["repo_dossier_actions"][0]["repo_id"] == "target"
    assert result["repo_dossier_actions"][0]["frontier_packet_id"] == "target:frontier:1"
    assert result["comparative_frontier"][0]["packet_id"] == "target:frontier:1"
    assert result["latent_capture"]["mode"] == "captured"
