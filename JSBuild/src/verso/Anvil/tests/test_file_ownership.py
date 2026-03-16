import time

from core.ownership.file_ownership import FileOwnershipRegistry
from saguaro.workset import WorksetManager


class DummyBus:
    def publish(self, topic, sender, payload, priority=None, metadata=None):
        _ = (topic, sender, payload, priority, metadata)


class DummyEventStore:
    def emit(self, event_type, payload, source=None, metadata=None):
        _ = (event_type, payload, source, metadata)


def make_registry(tmp_path, ttl_seconds=1, repo_policy_resolver=None, current_state_getter=None):
    manager = WorksetManager(
        saguaro_dir=str(tmp_path / ".saguaro"),
        repo_path=str(tmp_path),
    )
    return FileOwnershipRegistry(
        workset_manager=manager,
        message_bus=DummyBus(),
        event_store=DummyEventStore(),
        instance_id="local-instance",
        default_ttl_seconds=ttl_seconds,
        repo_policy_resolver=repo_policy_resolver,
        current_state_getter=current_state_getter,
    )


def test_claim_exclusive_success(tmp_path):
    registry = make_registry(tmp_path)
    result = registry.claim_files("agent-a", ["core/a.py"], mode="exclusive")

    assert result.success is True
    assert result.granted_files == ["core/a.py"]
    assert result.denied_files == []


def test_claim_exclusive_conflict(tmp_path):
    registry = make_registry(tmp_path)
    first = registry.claim_files("agent-a", ["core/a.py"], mode="exclusive")
    second = registry.claim_files("agent-b", ["core/a.py"], mode="exclusive")

    assert first.success is True
    assert second.success is False
    assert second.denied_files[0].reason == "exclusive_lock"


def test_claim_shared_read_no_conflict(tmp_path):
    registry = make_registry(tmp_path)
    first = registry.claim_files("agent-a", ["core/a.py"], mode="shared_read")
    second = registry.claim_files("agent-b", ["core/a.py"], mode="shared_read")

    assert first.success is True
    assert second.success is True


def test_release_allows_reclaim(tmp_path):
    registry = make_registry(tmp_path)
    registry.claim_files("agent-a", ["core/a.py"], mode="exclusive")
    registry.release_files("agent-a")

    result = registry.claim_files("agent-b", ["core/a.py"], mode="exclusive")
    assert result.success is True


def test_heartbeat_refresh(tmp_path):
    registry = make_registry(tmp_path, ttl_seconds=1)
    registry.claim_files("agent-a", ["core/a.py"], mode="exclusive")

    time.sleep(0.7)
    registry.heartbeat("agent-a")
    time.sleep(0.6)
    registry.reap_expired_leases()

    status = registry.get_status_snapshot()
    assert "core/a.py" in status["file_owners"]


def test_expired_lease_reap(tmp_path):
    registry = make_registry(tmp_path, ttl_seconds=1)
    registry.claim_files("agent-a", ["core/a.py"], mode="exclusive")

    time.sleep(1.2)
    registry.reap_expired_leases()

    result = registry.claim_files("agent-b", ["core/a.py"], mode="exclusive")
    assert result.success is True


def test_phase_ownership_isolation(tmp_path):
    registry = make_registry(tmp_path)
    first = registry.claim_files(
        "agent-a",
        ["core/a.py"],
        mode="exclusive",
        phase_id="phase-a",
    )
    second = registry.claim_files(
        "agent-b",
        ["core/a.py"],
        mode="exclusive",
        phase_id="phase-b",
    )

    assert first.success is True
    assert second.success is False


def test_collaborative_mode_multiple_writers(tmp_path):
    registry = make_registry(tmp_path)
    first = registry.claim_files("agent-a", ["core/a.py"], mode="collaborative")
    second = registry.claim_files("agent-b", ["core/a.py"], mode="collaborative")

    assert first.success is True
    assert second.success is True
    assert "core/a.py" in registry.get_agent_files("agent-a")
    assert "core/a.py" in registry.get_agent_files("agent-b")


def test_repo_policy_blocks_target_write_outside_allowed_state(tmp_path):
    state = {"value": "RESEARCH"}

    def resolver(**kwargs):
        if kwargs.get("access_mode") == "target_write" and kwargs.get("campaign_state") != "DEVELOPMENT":
            return False, "repo_policy_denied"
        return True, "ok"

    registry = make_registry(
        tmp_path,
        repo_policy_resolver=resolver,
        current_state_getter=lambda: state["value"],
    )
    denied = registry.claim_files(
        "agent-a",
        ["core/a.py"],
        mode="exclusive",
        access_mode="target_write",
    )
    assert denied.success is False
    assert denied.denied_files[0].reason == "repo_policy_denied"

    state["value"] = "DEVELOPMENT"
    allowed = registry.claim_files(
        "agent-a",
        ["core/a.py"],
        mode="exclusive",
        access_mode="target_write",
    )
    assert allowed.success is True
