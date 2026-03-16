from core.campaign.control_plane import CampaignControlPlane
from core.campaign.worktree_manager import CampaignWorktreeManager


class _PassingVerifier:
    def verify_changes(self, modified_files):
        return {
            "syntax": {"passed": True},
            "lint": {"passed": True},
            "tests": {"passed": True},
            "sentinel": {"passed": True},
            "all_passed": True,
            "runtime_symbols": [],
            "counterexamples": [],
        }


def test_verified_delta_can_be_promoted_and_recorded(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "core" / "sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("print('base')\n", encoding="utf-8")

    control = CampaignControlPlane.create(
        "delta_dist",
        "Delta Dist",
        str(tmp_path / "campaigns"),
        objective="Promote verified deltas",
        root_dir=str(root),
    )
    manager = CampaignWorktreeManager(str(root))
    manager.prepare("lane-a", editable_scope=["core/sample.py"])
    lane_file = root / ".anvil_lane_runtime" / "lane-a" / "workspace" / "core" / "sample.py"
    lane_file.write_text("print('verified')\n", encoding="utf-8")

    lane = control.create_verification_lane(_PassingVerifier())
    verify_payload = lane.run(["core/sample.py"], campaign_id=control.campaign_id, task_packet_id="lane-a")
    promoted = manager.promote("lane-a")
    control.state_ledger.record_changes(changed_files=[str(source)], reason="delta_distribution")

    assert verify_payload["promotion_blocked"] is False
    assert promoted["promoted_files"] == ["core/sample.py"]
    assert control.state_ledger.delta_watermark()["changed_paths"] == ["core/sample.py"]
