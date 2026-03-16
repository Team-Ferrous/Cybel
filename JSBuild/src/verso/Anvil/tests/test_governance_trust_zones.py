from core.ownership.file_ownership import FileOwnershipRegistry, build_trust_zone_resolver
from saguaro.workset import WorksetManager


class _DummyBus:
    def publish(self, topic, sender, payload, priority=None, metadata=None):
        _ = (topic, sender, payload, priority, metadata)


class _DummyEventStore:
    def emit(self, event_type, payload, source=None, metadata=None):
        _ = (event_type, payload, source, metadata)


def _registry(tmp_path, *, local_zone: str):
    manager = WorksetManager(saguaro_dir=str(tmp_path / ".saguaro"), repo_path=str(tmp_path))
    return FileOwnershipRegistry(
        workset_manager=manager,
        message_bus=_DummyBus(),
        event_store=_DummyEventStore(),
        instance_id="local",
        repo_policy_resolver=build_trust_zone_resolver(local_zone=local_zone),
    )


def test_trust_zone_blocks_sensitive_target_writes(tmp_path):
    registry = _registry(tmp_path, local_zone="campaign")

    denied = registry.claim_files(
        "agent-a",
        ["core/native/kernel.cpp"],
        access_mode="target_write",
    )

    assert denied.success is False
    assert denied.denied_files[0].reason == "policy.zone.denied:maintainer"


def test_trust_zone_allows_maintainer_for_sensitive_target_writes(tmp_path):
    registry = _registry(tmp_path, local_zone="maintainer")

    allowed = registry.claim_files(
        "agent-a",
        ["core/native/kernel.cpp"],
        access_mode="target_write",
    )

    assert allowed.success is True


def test_trust_zone_allows_readonly_audit_access(tmp_path):
    registry = _registry(tmp_path, local_zone="external")

    allowed = registry.claim_files(
        "agent-a",
        ["shared_kernel/event_store.py"],
        access_mode="audit_readonly",
    )

    assert allowed.success is True
