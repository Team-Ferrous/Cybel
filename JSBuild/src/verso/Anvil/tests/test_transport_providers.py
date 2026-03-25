import asyncio
import time

from core.networking.instance_identity import AnvilInstance
from core.networking.peer_transport import PeerTransport


def _instance(instance_id: str) -> AnvilInstance:
    now = time.time()
    return AnvilInstance(
        instance_id=instance_id,
        hostname=f"host-{instance_id}",
        user="tester",
        project_root="/tmp/repo",
        project_hash="repo-hash",
        repo_branch="main",
        repo_head="deadbeef",
        repo_dirty=False,
        listen_address="127.0.0.1:0",
        capabilities=["ownership_v1"],
        started_at=now,
        last_seen=now,
    )


def test_in_memory_transport_provider_delivers_messages():
    left = PeerTransport(_instance("left"), provider="in_memory")
    right = PeerTransport(_instance("right"), provider="in_memory")
    received = []
    right.on_message(received.append)

    async def scenario():
        await left.connect(right.instance)
        await right.connect(left.instance)
        await left.send("right", {"type": "presence", "payload": {"ok": True}})

    asyncio.run(scenario())

    assert received[0]["type"] == "presence"
    assert received[0]["transport_provider"] == "in_memory"

    left.close()
    right.close()


def test_filesystem_transport_provider_cross_process_mailbox(tmp_path):
    root = tmp_path / "transport"
    left = PeerTransport(_instance("left"), provider="filesystem", transport_root=str(root))
    right = PeerTransport(_instance("right"), provider="filesystem", transport_root=str(root))
    received = []
    right.on_message(received.append)

    async def scenario():
        await left.connect(right.instance)
        await right.connect(left.instance)
        await left.send("right", {"type": "ownership.claim", "payload": {"file": "core/a.py"}})

    asyncio.run(scenario())
    time.sleep(0.2)

    assert received
    assert received[0]["type"] == "ownership.claim"
    assert received[0]["transport_provider"] == "filesystem"

    left.close()
    right.close()
