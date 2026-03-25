from core.architect.architect_plane import ArchitectPlane


def test_architect_plane_elects_promotable_leader_and_rebalances_files():
    plane = ArchitectPlane(instance_id="inst-a")
    presence = {
        "peers": [
            {
                "instance_id": "inst-a",
                "campaign_id": "cmp-1",
                "verification_state": "ready",
                "connected": True,
                "analysis_capacity": 0.9,
                "verification_capacity": 0.8,
                "active_claim_count": 1,
            },
            {
                "instance_id": "inst-b",
                "campaign_id": "cmp-1",
                "verification_state": "warming",
                "connected": True,
                "analysis_capacity": 0.4,
                "verification_capacity": 0.5,
                "active_claim_count": 3,
            },
        ]
    }

    decision = plane.arbitrate(
        local_plan={
            "instance_id": "inst-a",
            "tasks": [{"id": "t-1", "task": "Refactor presence"}],
            "files": ["core/networking/peer_discovery.py", "shared_kernel/event_store.py"],
        },
        remote_plans=[
            {
                "instance_id": "inst-b",
                "tasks": [{"id": "t-2", "task": "Add verification telemetry"}],
                "files": ["shared_kernel/event_store.py"],
            }
        ],
        presence=presence,
        ownership_snapshot={"file_owners": {"shared_kernel/event_store.py": [{"owner": "inst-c"}]}},
        campaign_id="cmp-1",
    )

    assert decision.leader_id == "inst-a"
    assert decision.merged_execution_contract["task_count"] == 2
    assert decision.merged_execution_contract["ownership_rebalance_required"] is True
    assert decision.merged_execution_contract["file_assignments"]["inst-a"] == [
        "core/networking/peer_discovery.py"
    ]
