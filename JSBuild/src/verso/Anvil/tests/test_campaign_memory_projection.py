import time

from core.campaign.control_plane import CampaignControlPlane


def test_campaign_memory_projection_supports_coordination_retrieval(tmp_path):
    control = CampaignControlPlane.create(
        f"memory_{int(time.time())}",
        "Memory Projection",
        str(tmp_path / "campaigns"),
        objective="Project coordination state",
        root_dir=str(tmp_path),
    )

    memory = control.memory_fabric.create_memory(
        "coordination_verdict",
        {"decision": "split_by_file", "files": ["core/a.py"]},
        campaign_id=control.campaign_id,
        repo_context="target",
        task_packet_id="packet-1",
        summary_text="Architect split work by file ownership.",
    )
    control.memory_projector.project_memory(control.memory_fabric, memory)
    result = control.memory_planner.retrieve(
        campaign_id=control.campaign_id,
        query_text="architect split file ownership",
        repo_context="target",
        limit=1,
    )

    assert result["results"]
    assert result["results"][0]["memory_id"] == memory.memory_id
