from core.campaign.artifact_registry import (
    ArtifactApprovalState,
    ArtifactDependencyEdge,
    ArtifactProvenanceLink,
    CampaignArtifactRegistry,
)


def test_publish_approve_and_version_artifact(tmp_path):
    registry = CampaignArtifactRegistry(str(tmp_path))
    version_one = registry.publish(
        artifact_id="campaign:architecture",
        family="architecture",
        name="questionnaire",
        canonical_payload={"questions": [{"question_id": "q1", "question": "Choose API?"}]},
        provenance=[ArtifactProvenanceLink(kind="directive", target="objective")],
        dependencies=[ArtifactDependencyEdge("campaign:architecture", "campaign:intake")],
        blocking=True,
    )

    assert version_one.version == 1
    assert version_one.blocking is True

    registry.approve(
        "campaign:architecture",
        approved_by="user",
        state=ArtifactApprovalState.APPROVED.value,
    )
    registry.set_blocking("campaign:architecture", blocking=False)

    version_two = registry.publish(
        artifact_id="campaign:architecture",
        family="architecture",
        name="questionnaire",
        canonical_payload={"questions": [{"question_id": "q2", "question": "Choose DB?"}]},
    )

    record = registry.get_artifact("campaign:architecture")
    assert version_two.version == 2
    assert registry.get_latest_version("campaign:architecture").version == 2
    assert registry.get_resume_pointer("campaign:architecture") == 2
    assert record.latest_approval_state == ArtifactApprovalState.APPROVED.value
    assert registry.get_blocking_artifacts() == []
    assert "campaign:architecture" in registry.export_index()
