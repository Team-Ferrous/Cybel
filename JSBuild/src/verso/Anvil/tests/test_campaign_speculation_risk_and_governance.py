from __future__ import annotations

import sys

from core.campaign.control_plane import CampaignControlPlane


def test_roadmap_risk_repo_twin_and_governance_flow(tmp_path, monkeypatch) -> None:
    control = CampaignControlPlane.create(
        "campaign-risk",
        "Campaign Risk",
        str(tmp_path / "campaigns"),
        objective="Ship a governed roadmap.",
        root_dir=str(tmp_path),
    )

    questionnaire = control.build_questionnaire()
    assert questionnaire["count"] >= 1
    control.approve_artifact(f"{control.campaign_id}:architecture_questionnaire", state="accepted")

    feature_map = control.build_feature_map()
    assert feature_map["count"] >= 1
    control.approve_artifact(f"{control.campaign_id}:feature_map", state="approved")

    roadmap = control.build_roadmap()
    assert roadmap["roadmap_risk"]["summary"]["item_count"] >= 1
    assert roadmap["items"][0]["metadata"]["risk"]["risk_level"] in {"low", "medium", "high"}

    promoted = control.promote_final_roadmap()
    assert promoted["repo_twin"]["path"].endswith("repo_twin_roadmap_promoted.json")

    packet = control.create_task_packet(
        packet_id="packet-1",
        objective="Implement guarded telemetry",
        allowed_repos=["target"],
        aes_metadata={"packet_kind": "implementation"},
    )
    result = control.execute_task_packet(
        packet["task_packet_id"],
        lambda payload: {
            "summary": payload["objective"],
            "changed_files": [],
            "verification": "pending",
        },
    )
    assert result["accepted"] is True

    dashboard = control.build_specialist_dashboard()
    assert dashboard["summary"]["accepted_count"] >= 1

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
    audit = control.run_audit()
    rule_id = audit["rule_proposals"]["proposals"][0]["rule_id"]
    adoption = control.adopt_rule_proposal(rule_id, notes="human approved")
    outcome = control.record_rule_outcome(
        rule_id,
        outcome_status="stable",
        regression_delta=-0.1,
        notes="regressions reduced",
    )
    status = control.rule_proposals.status(control.campaign_id)

    assert adoption["rule_id"] == rule_id
    assert outcome["outcome_status"] == "stable"
    assert status["adoption_count"] == 1
    assert status["outcome_count"] == 1


def test_speculation_runs_two_branches_and_promotes_explicit_choice(tmp_path) -> None:
    control = CampaignControlPlane.create(
        "campaign-spec",
        "Campaign Spec",
        str(tmp_path / "campaigns"),
        objective="Run bounded speculation.",
        root_dir=str(tmp_path),
    )
    editable = control.workspace.root_dir + "/editable.txt"
    with open(editable, "w", encoding="utf-8") as handle:
        handle.write("source\n")

    control.state_store.record_roadmap_item(
        {
            "campaign_id": control.campaign_id,
            "item_id": "roadmap_lane_overlay",
            "phase_id": "development",
            "title": "Speculative overlay lane",
            "type": "experiment_lane",
            "repo_scope": ["target"],
            "owner_type": "ExperimentLane",
            "depends_on": [],
            "description": "Compare control and candidate branches.",
            "objective": "Compare speculative branches.",
            "success_metrics": ["correctness_pass"],
            "required_evidence": [],
            "required_artifacts": ["experiments"],
            "telemetry_contract": {
                "required_metrics": ["correctness_pass", "determinism_pass"],
                "minimum_success_count": 1,
            },
            "allowed_writes": ["target"],
            "promotion_gate": {"minimum_score": 0.0},
            "exit_gate": {"branch_count": 2},
            "metadata": {
                "editable_scope": ["editable.txt"],
                "speculation_variants": [
                    {
                        "name": "control",
                        "commands": [
                            {
                                "label": "control",
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "print('correctness_pass=1\\ndeterminism_pass=1')",
                                ],
                            }
                        ],
                    },
                    {
                        "name": "candidate",
                        "commands": [
                            {
                                "label": "candidate",
                                "argv": [
                                    sys.executable,
                                    "-c",
                                    "from pathlib import Path; Path('editable.txt').write_text('candidate\\n', encoding='utf-8'); print('correctness_pass=1\\ndeterminism_pass=1')",
                                ],
                            }
                        ],
                    },
                ],
            },
        }
    )

    comparison = control.run_speculative_roadmap_item("roadmap_lane_overlay")
    assert len(comparison["branches"]) == 2
    candidate = next(
        item for item in comparison["branches"] if item["variant"] == "candidate"
    )
    assert candidate["branch_metrics"]["changed_files"] == ["editable.txt"]

    promoted = control.promote_speculative_branch(
        comparison["comparison_id"],
        candidate["lane_id"],
    )
    with open(editable, "r", encoding="utf-8") as handle:
        contents = handle.read()

    assert promoted["promoted"]["promoted_files"] == ["editable.txt"]
    assert contents == "candidate\n"
