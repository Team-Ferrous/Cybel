from __future__ import annotations

from core.memory.latent_memory import LatentMemory
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


def test_tool_outcome_can_be_compressed_into_latent_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = TaskMemoryManager(_SaguaroTools(), None, _Console())
    task = TaskMemory(
        task_id="task-42",
        task_type="tool",
        description="Compress tool outcome into latent artifact",
        files_modified=["core/task_memory.py"],
        tools_used=["saguaro", "pytest"],
        success=True,
        execution_time=0.5,
        iterations=1,
        timestamp=1.0,
        approach="store compressed result",
        key_steps=["collect output", "compress summary"],
        difficulties=[],
        tests_passed=True,
        syntax_valid=True,
    )
    manager.remember(task)

    latent_memory = LatentMemory(max_size=4)
    latent_memory.add_state(
        "tool_outcome",
        task.description,
        vector=[0.2, 0.3, 0.5],
    )
    package, tensor = latent_memory.build_package(
        memory_id="tool-memory",
        capture_stage="tool_outcome",
        summary_text=task.description,
    )

    assert package.capture_stage == "tool_outcome"
    assert package.compatibility_json["segment_count"] >= 1
    assert tensor
