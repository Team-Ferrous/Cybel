from dataclasses import asdict

from cli.history import ConversationHistory
from core.memory.fabric import MemoryFabricStore
from core.task_memory import TaskMemory, TaskMemoryManager


class _Console:
    def print(self, *_args, **_kwargs):
        return None


class _SaguaroTools:
    def __init__(self):
        self.records = {}

    def memory(self, *, action, key=None, value=None, query=None, **_kwargs):
        if action == "store":
            self.records[key] = value
            return {"status": "ok"}
        if action == "recall":
            return {"matches": [{"value": value} for value in self.records.values()]}
        raise ValueError(action)


def test_conversation_and_task_memory_ingest_into_almf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    history = ConversationHistory(
        history_file=str(tmp_path / "history.json"),
        db_path=str(tmp_path / "anvil.db"),
    )
    history.add_message("user", "hello latent fabric")
    fabric = MemoryFabricStore.from_db_path(str(tmp_path / "anvil.db"))
    memories = fabric.list_memories("conversation", session_id=history.session_id)

    assert len(memories) == 1
    assert memories[0]["memory_kind"] == "conversation_turn"

    manager = TaskMemoryManager(_SaguaroTools(), None, _Console())
    task = TaskMemory(
        task_id="task-1",
        task_type="edit",
        description="Implement latent replay",
        files_modified=["core/qsg/continuous_engine.py"],
        tools_used=["apply_patch"],
        success=True,
        execution_time=1.0,
        iterations=1,
        timestamp=1.0,
        approach="patch and verify",
        key_steps=["edit", "test"],
        difficulties=[],
        tests_passed=True,
        syntax_valid=True,
    )
    manager.remember(task)

    task_fabric = MemoryFabricStore.from_db_path(str(tmp_path / ".anvil/memory/almf.db"))
    task_memories = task_fabric.list_memories("task_memory")
    assert len(task_memories) == 1
    assert task_memories[0]["task_packet_id"] == "task-1"
