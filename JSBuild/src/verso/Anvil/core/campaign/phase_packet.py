"""Phase-packet generation for roadmap documents."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


class PhasePacketBuilder:
    """Renders typed phase packets from compiled roadmap tasks."""

    @staticmethod
    def build(
        phase_documents: Iterable[Dict[str, Any]],
        *,
        objective: str = "",
    ) -> List[Dict[str, Any]]:
        packets: List[Dict[str, Any]] = []
        for phase in phase_documents:
            tasks = list(phase.get("tasks") or [])
            repo_scope = sorted(
                {
                    repo
                    for task in tasks
                    for repo in task.get("repo_scope", [])
                    if repo
                }
            )
            owning_specialists = sorted(
                {
                    str(task.get("owner_type") or "")
                    for task in tasks
                    if task.get("owner_type")
                }
            )
            telemetry_contract = {
                "minimum": sorted(
                    {
                        metric
                        for task in tasks
                        for metric in (
                            task.get("telemetry_contract", {}).get("minimum", [])
                        )
                        if metric
                    }
                )
            }
            promotion_gate = {
                "tasks": [task.get("item_id") for task in tasks],
                "required_success_metrics": sorted(
                    {
                        metric
                        for task in tasks
                        for metric in task.get("success_metrics", [])
                        if metric
                    }
                ),
            }
            packets.append(
                {
                    "phase_id": phase.get("phase_id"),
                    "name": phase.get("name"),
                    "objective": objective,
                    "repo_scope": repo_scope,
                    "owning_specialist_type": owning_specialists,
                    "allowed_writes": sorted(
                        {
                            repo
                            for task in tasks
                            for repo in task.get("allowed_writes", [])
                            if repo
                        }
                    ),
                    "telemetry_contract": telemetry_contract,
                    "required_evidence": sorted(
                        {
                            evidence
                            for task in tasks
                            for evidence in task.get("required_evidence", [])
                            if evidence
                        }
                    ),
                    "required_artifacts": sorted(
                        {
                            artifact
                            for task in tasks
                            for artifact in task.get("required_artifacts", [])
                            if artifact
                        }
                    ),
                    "rollback_criteria": sorted(
                        {
                            criterion
                            for task in tasks
                            for criterion in task.get("rollback_criteria", [])
                            if criterion
                        }
                    ),
                    "promotion_gate": promotion_gate,
                    "tasks": tasks,
                    "dependencies": list(phase.get("dependencies") or []),
                    "success_criteria": list(phase.get("success_criteria") or []),
                    "estimated_iterations": phase.get("estimated_iterations", 1),
                    "artifact_folder": phase.get("artifact_folder", ""),
                }
            )
        return packets
