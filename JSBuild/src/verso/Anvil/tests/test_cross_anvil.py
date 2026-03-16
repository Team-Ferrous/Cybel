import asyncio
import time

from core.collaboration.context_sharing import ContextShareProtocol
from core.collaboration.negotiation import CollaborationNegotiator
from core.collaboration.task_announcer import OverlapResult, TaskAnnouncer
from core.networking.instance_identity import AnvilInstance
from core.networking.peer_discovery import FileSystemDiscovery
from core.networking.peer_transport import PeerTransport
from core.ownership.ownership_crdt import OwnershipCRDT
from core.ownership.ownership_sync import OwnershipSyncProtocol


def make_instance(instance_id: str, project_hash: str = "proj-hash") -> AnvilInstance:
    now = time.time()
    return AnvilInstance(
        instance_id=instance_id,
        hostname=f"host-{instance_id}",
        user="tester",
        project_root="/tmp/repo",
        project_hash=project_hash,
        listen_address="127.0.0.1:0",
        capabilities=["ownership_v1", "collaboration_v1"],
        started_at=now,
        last_seen=now,
    )


def test_peer_discovery_filesystem(tmp_path):
    peers_dir = tmp_path / "peers"
    inst1 = make_instance("inst-1")
    inst2 = make_instance("inst-2")

    fs1 = FileSystemDiscovery(inst1, peers_dir=str(peers_dir))
    fs2 = FileSystemDiscovery(inst2, peers_dir=str(peers_dir))

    fs1.announce()
    fs2.announce()

    peers = fs1.browse()
    assert any(peer.instance_id == "inst-2" for peer in peers)


async def _connect_bidirectional(transport_a: PeerTransport, transport_b: PeerTransport):
    await transport_a.connect(transport_b.instance)
    await transport_b.connect(transport_a.instance)


def test_ownership_sync_on_connect():
    inst1 = make_instance("inst-a")
    inst2 = make_instance("inst-b")

    transport1 = PeerTransport(inst1)
    transport2 = PeerTransport(inst2)

    crdt1 = OwnershipCRDT(inst1.instance_id)
    crdt2 = OwnershipCRDT(inst2.instance_id)

    crdt1.claim("core/a.py", "agent-a")

    sync1 = OwnershipSyncProtocol(crdt1, transport1)
    sync2 = OwnershipSyncProtocol(crdt2, transport2)

    async def scenario():
        await _connect_bidirectional(transport1, transport2)
        await sync1.on_peer_connected(inst2.instance_id)
        await sync2.on_peer_connected(inst1.instance_id)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert "core/a.py" in crdt2.state


def test_ownership_claim_propagation():
    inst1 = make_instance("inst-a")
    inst2 = make_instance("inst-b")

    transport1 = PeerTransport(inst1)
    transport2 = PeerTransport(inst2)

    crdt1 = OwnershipCRDT(inst1.instance_id)
    crdt2 = OwnershipCRDT(inst2.instance_id)

    sync1 = OwnershipSyncProtocol(crdt1, transport1)
    OwnershipSyncProtocol(crdt2, transport2)

    async def scenario():
        await _connect_bidirectional(transport1, transport2)
        entry = crdt1.claim("core/a.py", "agent-a")
        await sync1.on_ownership_change(entry)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    assert "core/a.py" in crdt2.state


def test_cross_instance_conflict_resolution():
    inst1 = make_instance("inst-a")
    inst2 = make_instance("inst-b")

    transport1 = PeerTransport(inst1)
    transport2 = PeerTransport(inst2)

    crdt1 = OwnershipCRDT(inst1.instance_id)
    crdt2 = OwnershipCRDT(inst2.instance_id)

    sync1 = OwnershipSyncProtocol(crdt1, transport1)
    sync2 = OwnershipSyncProtocol(crdt2, transport2)

    async def scenario():
        await _connect_bidirectional(transport1, transport2)
        entry1 = crdt1.claim("core/a.py", "agent-a")
        entry2 = crdt2.claim("core/a.py", "agent-b")
        await sync1.on_ownership_change(entry1)
        await sync2.on_ownership_change(entry2)
        await asyncio.sleep(0.05)

    asyncio.run(scenario())

    expected_instance = max(inst1.instance_id, inst2.instance_id)
    assert crdt1.state["core/a.py"].owner_instance_id == expected_instance
    assert crdt2.state["core/a.py"].owner_instance_id == expected_instance


def test_task_overlap_detection():
    announcer = TaskAnnouncer(transport=None, overlap_threshold=0.5)
    local = [
        {
            "id": "l1",
            "instruction": "Refactor authentication token parser",
            "context_files": ["core/auth.py"],
        }
    ]
    remote = [
        {
            "id": "r1",
            "instruction": "Improve auth token parsing reliability",
            "context_files": ["core/auth.py"],
        }
    ]

    overlaps = announcer.detect_overlap(local, remote)
    assert overlaps
    assert overlaps[0].overlap_type in {"duplicate", "complementary", "conflicting"}


def test_collaboration_negotiation_flow():
    negotiator = CollaborationNegotiator()
    overlap = OverlapResult(
        local_task_id="l1",
        remote_task_id="r1",
        similarity_score=0.82,
        overlap_type="complementary",
        local_files=["core/a.py"],
        remote_files=["core/b.py"],
    )

    proposal_id = negotiator.propose_collaboration(
        overlap=overlap,
        local_context="Need to coordinate auth changes",
        local_plan={"tasks": [{"id": "l1", "task": "Update auth flow"}], "files": ["core/a.py"]},
    )
    response = negotiator.receive_proposal(
        {
            "proposal_id": proposal_id,
            "overlap": overlap.__dict__,
            "local_plan": {"tasks": [{"id": "l1", "task": "Update auth flow"}], "files": ["core/a.py"]},
            "remote_plan": {"tasks": [{"id": "r1", "task": "Add auth tests"}], "files": ["tests/test_auth.py"]},
        }
    )

    assert response.status == "accepted"
    assert response.merged_plan is not None


def test_context_sharing_redaction():
    protocol = ContextShareProtocol()
    task = {"id": "t1", "instruction": "Fix API auth", "context_files": ["core/auth.py"]}
    history = [
        {"role": "user", "content": "api_key=SECRET12345 and token=abcde12345"},
        {"role": "assistant", "content": "Working on auth flow"},
    ]

    shared = protocol.prepare_shareable_context(task, history, sharing_level="full")
    joined = "\n".join(item["content"] for item in shared.get("messages", []))

    assert "SECRET12345" not in joined
    assert "abcde12345" not in joined
