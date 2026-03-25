from core.collaboration.negotiation import CollaborationNegotiator
from core.collaboration.task_announcer import OverlapResult, TaskAnnouncer


def test_task_announcer_preserves_phase_symbol_and_verification_metadata():
    task = {
        "id": "task-1",
        "instruction": "Refactor peer presence mesh",
        "context_files": ["core/networking/peer_discovery.py"],
        "phase_id": "development",
        "campaign_id": "cmp-1",
        "context_symbols": ["PeerDiscovery.refresh"],
        "verification_targets": ["tests/test_repo_presence.py"],
    }

    announced = TaskAnnouncer._task_to_dict(task)

    assert announced["phase_id"] == "development"
    assert announced["campaign_id"] == "cmp-1"
    assert announced["context_symbols"] == ["PeerDiscovery.refresh"]
    assert announced["verification_targets"] == ["tests/test_repo_presence.py"]


def test_negotiation_accepts_overlap_and_emits_shared_handoff():
    overlap = OverlapResult(
        local_task_id="local",
        remote_task_id="remote",
        similarity_score=0.9,
        overlap_type="complementary",
        local_files=["core/networking/peer_discovery.py"],
        remote_files=["core/networking/peer_discovery.py"],
    )
    negotiator = CollaborationNegotiator()
    proposal_id = negotiator.propose_collaboration(
        overlap=overlap,
        local_context="Need to sequence presence and transport changes",
        local_plan={"tasks": [{"id": "local", "task": "Presence mesh"}], "files": overlap.local_files},
    )

    response = negotiator.receive_proposal(
        {
            "proposal_id": proposal_id,
            "overlap": overlap.__dict__,
            "local_plan": {"tasks": [{"id": "local", "task": "Presence mesh"}], "files": overlap.local_files},
            "remote_plan": {"tasks": [{"id": "remote", "task": "Transport provider"}], "files": overlap.remote_files},
        }
    )

    assert response.status == "accepted"
    assert response.merged_plan["handoffs"][0]["type"] == "shared_files"
