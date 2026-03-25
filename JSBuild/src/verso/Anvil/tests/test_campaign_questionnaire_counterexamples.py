from __future__ import annotations

from uuid import uuid4

import pytest

from core.campaign.control_plane import CampaignControlPlane


def test_questionnaire_builder_preserves_counterexample_scoring_metadata(
    tmp_path,
) -> None:
    control = CampaignControlPlane.create(
        f"questionnaire-{uuid4().hex[:8]}",
        "Questionnaire",
        str(tmp_path / "campaigns"),
        objective="Resolve architecture counterexamples.",
        root_dir=str(tmp_path / "repo"),
    )

    built = control.questionnaire.build(
        directives=[],
        unresolved_unknowns=[
            {
                "question_id": "q-counterexample",
                "question": "Which cache invalidation boundary removes the replay counterexample?",
                "why_it_matters": "This answer prunes a risky implementation branch.",
                "blocking_level": "critical",
                "linked_roadmap_items": ["roadmap-cache-boundary"],
                "metadata": {
                    "question_value_score": 0.93,
                    "branch_pruning_estimate": 4,
                },
            }
        ],
    )
    persisted = control.build_questionnaire(questions=built)

    assert persisted["questions"][0]["linked_roadmap_items"] == ["roadmap-cache-boundary"]
    assert persisted["questions"][0]["metadata"]["question_value_score"] == 0.93
    assert persisted["questions"][0]["metadata"]["branch_pruning_estimate"] == 4
    assert control.questionnaire.pending_blockers()[0]["question_id"] == "q-counterexample"


def test_transition_policy_blocks_until_questionnaire_is_accepted(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"questionnaire-transition-{uuid4().hex[:8]}",
        "Questionnaire Transition",
        str(tmp_path / "campaigns"),
        objective="Resolve blockers before feature mapping.",
        root_dir=str(tmp_path / "repo"),
    )

    control.build_questionnaire()

    with pytest.raises(ValueError, match="Blocking questionnaire items"):
        control.continue_campaign()

    control.approve_artifact(
        f"{control.campaign_id}:architecture_questionnaire",
        state="accepted",
    )
    event = control.continue_campaign()

    assert event["to_state"] == "FEATURE_MAP_WAIT"
    assert event["metadata"]["blocking_questions"] >= 1
