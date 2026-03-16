from __future__ import annotations

from core.parallel_executor import ParallelToolExecutor
from saguaro.indexing.auto_scaler import record_runtime_profile


class _Console:
    def print(self, *args, **kwargs) -> None:
        _ = args, kwargs


def test_parallel_executor_uses_saguaro_agent_budget(tmp_path, monkeypatch) -> None:
    record_runtime_profile(
        str(tmp_path),
        profile_kind="runtime_layout",
        metrics={"max_parallel_agents": 3},
    )
    monkeypatch.chdir(tmp_path)

    executor = ParallelToolExecutor(
        registry=None,
        semantic_engine=None,
        console=_Console(),
    )

    assert executor.max_workers == 3
