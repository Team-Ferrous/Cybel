from __future__ import annotations

import time

from core.campaign.control_plane import CampaignControlPlane


def test_campaign_control_kernel_generates_dossier_and_injects_task_packets(tmp_path) -> None:
    campaigns_dir = tmp_path / "campaigns"
    target_repo = tmp_path / "target"
    analysis_repo = tmp_path / "analysis"
    target_repo.mkdir()
    analysis_repo.mkdir()
    (analysis_repo / "module.py").write_text("def reuse_me(x):\n    return x + 1\n", encoding="utf-8")

    control = CampaignControlPlane.create(
        f"campaign_{int(time.time())}",
        "Kernel Test",
        str(campaigns_dir),
        objective="Build a truthful campaign kernel",
        root_dir=str(target_repo),
    )
    control.acquire_repos(repo_specs=[str(analysis_repo)])

    brief = control.ensure_repo_dossier_brief()
    assert brief["repo_count"] >= 1
    assert brief["path"].endswith("repo_dossier_brief.md")

    packet = control.create_task_packet(
        packet_id="packet-1",
        objective="Implement reuse-aware change",
        allowed_repos=["target"],
    )
    metadata = packet["metadata"]

    assert "repo_dossier_summary" in metadata
    assert metadata["retrieval_policy"]["route"] == "saguaro"
    assert metadata["memory_read_id"].startswith("read_")

    control.transition_to("RESEARCH", "test_setup")
    event = control.continue_campaign()
    assert event["to_state"] == "RESEARCH_RECONCILIATION"
    assert event["loop_id"] == "research_loop"
