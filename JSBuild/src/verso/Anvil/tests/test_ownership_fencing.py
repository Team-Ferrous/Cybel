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
    )


def test_fencing_token_is_rotated_on_reclaim(tmp_path):
    registry = _registry(tmp_path)

    first = registry.claim_files("agent-a", ["core/a.py"])
    first_token = registry.current_fencing_token("core/a.py")
    registry.release_files("agent-a", ["core/a.py"])
    second = registry.claim_files("agent-a", ["core/a.py"])
    second_token = registry.current_fencing_token("core/a.py")

    assert first.success is True
    assert second.success is True
    assert first_token != second_token
    assert registry.can_access("agent-a", "core/a.py", fencing_token=first_token).allowed is False
    assert registry.can_access("agent-a", "core/a.py", fencing_token=second_token).allowed is True
