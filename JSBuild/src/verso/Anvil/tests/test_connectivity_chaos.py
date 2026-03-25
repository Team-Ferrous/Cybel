import time

from core.architect.architect_plane import ArchitectPlane
from core.ownership.file_ownership import FileOwnershipRegistry
from saguaro.workset import WorksetManager


class _DummyBus:
    def publish(self, topic, sender, payload, priority=None, metadata=None):
        _ = (topic, sender, payload, priority, metadata)


class _DummyEventStore:
    def emit(self, event_type, payload, source=None, metadata=None):
        _ = (event_type, payload, source, metadata)


def _registry(tmp_path):
    manager = WorksetManager(saguaro_dir=str(tmp_path / ".saguaro"), repo_path=str(tmp_path))
    return FileOwnershipRegistry(
        workset_manager=manager,
        message_bus=_DummyBus(),
        event_store=_DummyEventStore(),
        instance_id="local",
        default_ttl_seconds=1,
    )


def test_connectivity_chaos_failover_re_elects_leader_and_releases_stale_claims(tmp_path):
    plane = ArchitectPlane(instance_id="inst-b")
    leader = plane.elect_leader(
        [
            {
                "instance_id": "inst-a",
                "connected": True,
                "verification_state": "ready",
                "analysis_capacity": 0.8,
                "verification_capacity": 0.8,
                "last_seen": time.time() - 120,
            },
            {
                "instance_id": "inst-b",
                "connected": True,
                "verification_state": "ready",
                "analysis_capacity": 0.7,
                "verification_capacity": 0.7,
                "last_seen": time.time(),
            },
        ]
    )
    registry = _registry(tmp_path)
    registry.claim_files("agent-a", ["core/a.py"])
    time.sleep(1.1)
    registry.reap_expired_leases()

    assert leader["leader_id"] == "inst-b"
    assert registry.claim_files("agent-b", ["core/a.py"]).success is True
