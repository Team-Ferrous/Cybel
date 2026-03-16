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


def test_task_packet_executor_enforces_result_obligations(tmp_path) -> None:
    control = CampaignControlPlane.create(
        f"packet_{int(time.time())}",
        "Packet Test",
        str(tmp_path / "campaigns"),
        objective="Execute structured packet work",
        root_dir=str(tmp_path),
    )
    control.create_task_packet(
        packet_id="packet-1",
        objective="Implement a change",
        allowed_repos=["target"],
    )

    invalid = control.execute_task_packet(
        "packet-1",
        lambda packet: {"summary": "missing changed files"},
    )
    assert invalid["accepted"] is False
    assert "changed_files" in invalid["missing_result_fields"]

    valid = control.execute_task_packet(
        "packet-1",
        lambda packet: {
            "summary": "implemented",
            "changed_files": ["core/example.py"],
            "verification": {"planned": True},
        },
        verifier=_FakeVerifier(),
    )
    assert valid["accepted"] is True
    assert valid["verification_lane"]["all_passed"] is True
