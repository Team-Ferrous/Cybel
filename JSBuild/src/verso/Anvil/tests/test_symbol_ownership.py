from core.ownership.file_ownership import FileOwnershipRegistry
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
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


def test_symbol_claims_resolve_python_symbols_and_detect_conflicts(tmp_path):
    source = tmp_path / "core" / "sample.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "class Presence:\n    pass\n\n"
        "def build_mesh():\n    return 1\n",
        encoding="utf-8",
    )

    substrate = SaguaroSubstrate(root_dir=str(tmp_path))
    symbols = substrate.resolve_python_symbols("core/sample.py")
    registry = _registry(tmp_path)

    granted = registry.claim_symbols("agent-a", "core/sample.py", ["Presence"])
    denied = registry.claim_symbols("agent-b", "core/sample.py", ["Presence"])

    assert symbols == ["Presence", "build_mesh"]
    assert granted.success is True
    assert denied.success is False
    assert denied.denied_files[0].reason == "symbol_conflict"
    assert registry.query_symbol_ownership("core/sample.py", ["Presence"])["Presence"].owner_agent_id == "agent-a"
