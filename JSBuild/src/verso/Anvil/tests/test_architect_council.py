from core.architect.architect_plane import ArchitectPlane


def test_architect_council_returns_top_three_candidates():
    plane = ArchitectPlane(instance_id="inst-a")
    elected = plane.elect_leader(
        [
            {"instance_id": "inst-a", "connected": True, "verification_state": "ready", "analysis_capacity": 0.9},
            {"instance_id": "inst-b", "connected": True, "verification_state": "ready", "analysis_capacity": 0.7},
            {"instance_id": "inst-c", "connected": True, "verification_state": "warming", "analysis_capacity": 0.8},
            {"instance_id": "inst-d", "connected": False, "verification_state": "ready", "analysis_capacity": 1.0},
        ]
    )

    assert elected["leader_id"] == "inst-a"
    assert elected["council"] == ["inst-a", "inst-b", "inst-c"]
