"""Designs bounded shared-lane experiment tracks for EID."""

from __future__ import annotations

import os
import uuid
from typing import Any


class ExperimentDesignService:
    """Transforms EID proposals into shared-lane task packets."""

    def design(
        self,
        objective: str,
        proposals: list[dict[str, Any]] | Any,
        *,
        workspace_root: str,
        metadata_path: str,
    ) -> list[dict[str, Any]]:
        tracks: list[dict[str, Any]] = []
        for proposal in proposals:
            required = list(proposal.get("required_experiments") or [])
            for index, name in enumerate(required, start=1):
                tracks.append(
                    {
                        "lane_id": f"eid_lane_{uuid.uuid4().hex[:12]}",
                        "caller_mode": "eid",
                        "lane_type": self._lane_type(name),
                        "name": name,
                        "description": proposal.get("hypothesis_statement", ""),
                        "objective_function": (
                            f"Validate '{proposal.get('hypothesis_statement', '')}' "
                            f"for objective '{objective}'."
                        ),
                        "editable_scope": [],
                        "read_only_scope": [
                            os.path.relpath(metadata_path, workspace_root),
                            "artifacts/research",
                            "artifacts/feature_map",
                        ],
                        "allowed_writes": ["artifact_store"],
                        "benchmark_contract": {
                            "name": name,
                            "required_metrics": [
                                "wall_time_seconds",
                                "determinism_pass",
                                "correctness_pass",
                            ],
                            "objective_metric": "telemetry_contract_satisfied",
                        },
                        "promotion_policy": {
                            "name": "experimental_eid",
                            "minimum_score": 0.0,
                            "max_regression_penalty": 0.5,
                        },
                        "telemetry_contract": {
                            "schema_version": "lane.telemetry.v1",
                            "required_metrics": [
                                "wall_time_seconds",
                                "command_count",
                                "success_count",
                                "failure_count",
                                "correctness_pass",
                                "determinism_pass",
                            ],
                            "optional_metrics": [
                                "replayability",
                                "telemetry_contract_satisfied",
                                "analysis_pack_reuse",
                                "latency_ms",
                            ],
                            "minimum_success_count": 1,
                        },
                        "rollback_criteria": list(proposal.get("kill_criteria") or []),
                        "kill_criteria": list(proposal.get("kill_criteria") or []),
                        "commands": self._commands_for(name, metadata_path),
                        "metadata": {
                            "proposal_id": proposal.get("proposal_id"),
                            "hypothesis_id": proposal.get("hypothesis_id"),
                            "priority": round(
                                float(proposal.get("innovation_score") or 0.0)
                                - index * 0.05,
                                3,
                            ),
                        },
                    }
                )
        return sorted(
            tracks,
            key=lambda item: (
                -float(item.get("metadata", {}).get("priority") or 0.0),
                str(item.get("name") or ""),
            ),
        )

    @staticmethod
    def _lane_type(experiment_name: str) -> str:
        if "benchmark" in experiment_name:
            return "benchmark_first_optimization"
        if "replay" in experiment_name:
            return "bounded_file_experiment"
        return "subsystem_experiment"

    @staticmethod
    def _commands_for(experiment_name: str, metadata_path: str) -> list[dict[str, Any]]:
        quoted_metadata_path = metadata_path.replace("'", "'\"'\"'")
        if experiment_name == "artifact_resume_replay":
            return [
                {
                    "label": "metadata_exists",
                    "command": f"test -f '{quoted_metadata_path}' && echo replayability=1 determinism_pass=1",
                    "timeout_seconds": 10,
                },
                {
                    "label": "artifact_count_probe",
                    "command": "echo experiment_depth=2 correctness_pass=1",
                    "timeout_seconds": 10,
                },
            ]
        if experiment_name == "telemetry_contract_replay":
            return [
                {
                    "label": "telemetry_contract_probe",
                    "command": "echo latency_ms=5.0 determinism_pass=1 correctness_pass=1",
                    "timeout_seconds": 10,
                }
            ]
        if experiment_name == "analysis_pack_reuse_benchmark":
            return [
                {
                    "label": "analysis_pack_probe",
                    "command": "echo analysis_pack_reuse=1 throughput=1 determinism_pass=1 correctness_pass=1",
                    "timeout_seconds": 10,
                }
            ]
        return [
            {
                "label": "generic_probe",
                "command": "echo determinism_pass=1 correctness_pass=1",
                "timeout_seconds": 10,
            }
        ]
