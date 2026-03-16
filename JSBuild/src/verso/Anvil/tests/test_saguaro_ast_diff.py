from __future__ import annotations

from pathlib import Path

from saguaro.indexing.tracker import IndexTracker


def test_ast_diff_tracks_added_removed_changed_segments(tmp_path: Path) -> None:
    tracker = IndexTracker(str(tmp_path / "tracking.json"))
    file_path = str(tmp_path / "module.py")

    baseline = [
        {
            "segment_id": "seg:function:alpha",
            "segment_hash": "h-alpha-v1",
            "name": "alpha",
            "type": "function",
            "line": 1,
        },
        {
            "segment_id": "seg:function:beta",
            "segment_hash": "h-beta-v1",
            "name": "beta",
            "type": "function",
            "line": 10,
        },
    ]
    initial = tracker.update_ast_segments(file_path, baseline)
    assert sorted(initial["added_segments"]) == ["seg:function:alpha", "seg:function:beta"]

    # alpha moved but keeps its stable segment id/hash, beta changed, gamma added.
    next_state = [
        {
            "segment_id": "seg:function:alpha",
            "segment_hash": "h-alpha-v1",
            "name": "alpha",
            "type": "function",
            "line": 40,
        },
        {
            "segment_id": "seg:function:beta",
            "segment_hash": "h-beta-v2",
            "name": "beta",
            "type": "function",
            "line": 12,
        },
        {
            "segment_id": "seg:function:gamma",
            "segment_hash": "h-gamma-v1",
            "name": "gamma",
            "type": "function",
            "line": 80,
        },
    ]
    diff = tracker.diff_ast_segments(file_path, next_state)
    assert diff["added_segments"] == ["seg:function:gamma"]
    assert diff["removed_segments"] == []
    assert diff["changed_segments"] == ["seg:function:beta"]
    assert "seg:function:alpha" in diff["unchanged_segments"]
