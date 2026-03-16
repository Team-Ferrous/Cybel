from __future__ import annotations

import time

from core.campaign.control_plane import CampaignControlPlane


class _FakeVerifier:
    def verify_changes(self, modified_files):
        return {
            "syntax": {"passed": True, "errors": []},
            "tests": {"passed": True, "output": "", "skipped": False},
            "lint": {"passed": True, "warnings": []},
            "sentinel": {"passed": True, "violations": []},
            "all_passed": True,
            "runtime_symbols": [],
            "counterexamples": [],
        }


def test_timeline_assembler_captures_memory_and_verification_signals(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"timeline_{int(time.time())}",
        "Timeline Test",
        str(tmp_path / "campaigns"),
        objective="Replay a governed mission",
        root_dir=str(tmp_path),
    )
    packet = control.create_task_packet(
        packet_id="packet-1",
        objective="Change one file safely",
        allowed_repos=["target"],
    )

    lane = control.create_verification_lane(_FakeVerifier())
    payload = lane.run(
        ["core/example.py"],
        campaign_id=control.campaign_id,
        task_packet_id="packet-1",
        read_id=packet["metadata"]["memory_read_id"],
    )
    assert payload["all_passed"] is True

    timeline = control.build_mission_timeline()
    assert timeline["summary"]["telemetry_count"] >= 1
    assert timeline["summary"]["memory_read_count"] >= 1
    assert timeline["summary"]["memory_feedback_count"] >= 1
    assert timeline["path"].endswith("mission_timeline.json")
