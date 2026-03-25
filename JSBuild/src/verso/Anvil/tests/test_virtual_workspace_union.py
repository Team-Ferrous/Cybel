from core.campaign.worktree_manager import CampaignWorktreeManager


def test_virtual_workspace_union_collects_changed_files_across_lanes(tmp_path):
    base = tmp_path / "repo"
    base.mkdir()
    source = base / "core" / "sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("print('v1')\n", encoding="utf-8")

    manager = CampaignWorktreeManager(str(base))
    manager.prepare("lane-a", editable_scope=["core/sample.py"])
    manager.prepare("lane-b", editable_scope=["core/sample.py"])

    (base / ".anvil_lane_runtime" / "lane-a" / "workspace" / "core" / "sample.py").write_text(
        "print('lane-a')\n",
        encoding="utf-8",
    )
    (base / ".anvil_lane_runtime" / "lane-b" / "workspace" / "core" / "sample.py").write_text(
        "print('lane-b')\n",
        encoding="utf-8",
    )

    union = manager.virtual_union_snapshot(["lane-a", "lane-b"])

    assert union["lane_count"] == 2
    assert union["file_count"] == 1
    assert union["files"]["core/sample.py"]["lane_id"] in {"lane-a", "lane-b"}
