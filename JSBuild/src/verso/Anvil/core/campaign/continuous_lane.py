"""Continuous execution wrapper around the shared experiment lane."""

from __future__ import annotations

from typing import Any, Iterable, List


class ContinuousExperimentLane:
    """Runs bounded experiment tasks through a shared lane engine."""

    def __init__(self, experiment_runner) -> None:
        self.experiment_runner = experiment_runner

    def execute_once(self, task: dict[str, Any]) -> dict[str, Any]:
        return self.experiment_runner.run_lane(task)

    def execute_queue(
        self,
        tasks: Iterable[dict[str, Any]],
        *,
        max_iterations: int | None = None,
    ) -> List[dict[str, Any]]:
        results: List[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            if max_iterations is not None and index > max_iterations:
                break
            results.append(self.execute_once(task))
        return results
