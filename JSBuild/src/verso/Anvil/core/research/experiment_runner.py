"""Experiment execution and reproducibility records."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import uuid
from typing import Any

from core.campaign.baseline_manager import BaselineManager
from core.campaign.cpu_scorecard import CPUScorecard
from core.memory.fabric import MemoryEdge, MemoryFabricStore, MemoryProjector
from core.campaign.promotion_policy import PromotionPolicyEngine
from core.campaign.state_store import CampaignStateStore
from core.campaign.telemetry_contracts import TelemetryContractRegistry
from core.campaign.worktree_manager import CampaignWorktreeManager
from core.qsg.latent_bridge import QSGLatentBridge
from core.research.simulator_planner import SimulatorPlanner


class ExperimentRunner:
    """Runs stored experiment commands and records reproducible results."""

    METRIC_PATTERN = re.compile(r"([a-zA-Z_][\w.-]*)\s*[:=]\s*(-?\d+(?:\.\d+)?)")

    def __init__(
        self,
        campaign_id: str,
        state_store: CampaignStateStore,
        cwd: str = ".",
    ) -> None:
        self.campaign_id = campaign_id
        self.state_store = state_store
        self.cwd = cwd
        self.baselines = BaselineManager(cwd)
        self.worktrees = CampaignWorktreeManager(cwd)
        self.telemetry_contracts = TelemetryContractRegistry()
        self.scorecard = CPUScorecard()
        self.promotion = PromotionPolicyEngine()
        self.memory_fabric = MemoryFabricStore(state_store)
        self.memory_projector = MemoryProjector()
        self.latent_bridge = QSGLatentBridge(self.memory_fabric, self.memory_projector)
        self.shadow_planner = SimulatorPlanner()

    def run(
        self,
        name: str,
        commands: list[str | dict[str, Any]],
        *,
        default_timeout_seconds: int = 30,
        environment: str = "local_shell",
        cwd: str | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        experiment_id = str(uuid.uuid4())
        now = time.time()
        payload = {
            "experiment_id": experiment_id,
            "name": name,
            "environment": environment,
            "commands": [self._serialize_command(command) for command in commands],
            "status": "running",
        }
        self.state_store.execute(
            """
            INSERT INTO experiments (
                campaign_id, experiment_id, name, status, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.campaign_id,
                experiment_id,
                name,
                "running",
                json.dumps(payload, default=str),
                now,
                now,
            ),
        )
        spec_memory = self.memory_fabric.create_memory(
            memory_kind="experiment_spec",
            payload_json=payload,
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="experiment_runner.run",
            summary_text=name,
            experiment_id=experiment_id,
            lifecycle_state="running",
            importance_score=0.9,
            confidence_score=0.6,
        )
        self.memory_fabric.register_alias(
            spec_memory.memory_id,
            "experiments",
            experiment_id,
            campaign_id=self.campaign_id,
        )
        self.memory_projector.project_memory(self.memory_fabric, spec_memory)

        run_outputs: list[dict[str, Any]] = []
        experiment_started = time.time()
        for command in commands:
            spec = self._normalize_command(command, default_timeout_seconds=default_timeout_seconds)
            started_at = time.time()
            try:
                completed = subprocess.run(
                    spec["command"],
                    shell=spec["shell"],
                    capture_output=True,
                    text=True,
                    cwd=cwd or self.cwd,
                    timeout=spec["timeout_seconds"],
                    env={**spec["env"], **(extra_env or {})},
                )
                stdout = completed.stdout
                stderr = completed.stderr
                returncode = completed.returncode
                status = "completed" if returncode == 0 else "failed"
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                returncode = -1
                status = "timed_out"
            except OSError as exc:
                stdout = ""
                stderr = str(exc)
                returncode = -1
                status = "failed"
            duration_seconds = time.time() - started_at
            metrics = self._extract_metrics(stdout, stderr)
            run_outputs.append(
                {
                    "label": spec["label"],
                    "command": spec["command"],
                    "shell": spec["shell"],
                    "timeout_seconds": spec["timeout_seconds"],
                    "returncode": returncode,
                    "stdout": stdout,
                    "stderr": stderr,
                    "duration_seconds": duration_seconds,
                    "metrics": metrics,
                    "status": status,
                }
            )

        verdict = "passed" if all(item["returncode"] == 0 for item in run_outputs) else "failed"
        summary_metrics = self._summarize_metrics(run_outputs, experiment_started)
        self.state_store.execute(
            """
            INSERT INTO experiment_runs (
                campaign_id,
                run_id,
                experiment_id,
                status,
                command_json,
                result_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.campaign_id,
                str(uuid.uuid4()),
                experiment_id,
                verdict,
                json.dumps(payload["commands"], default=str),
                json.dumps(run_outputs, default=str),
                time.time(),
            ),
        )
        self.state_store.execute(
            """
            INSERT INTO experiment_results (
                campaign_id,
                result_id,
                experiment_id,
                verdict,
                metrics_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                self.campaign_id,
                str(uuid.uuid4()),
                experiment_id,
                verdict,
                json.dumps(summary_metrics, default=str),
                time.time(),
            ),
        )
        self.state_store.execute(
            """
            UPDATE experiments
            SET status = ?, payload_json = ?, updated_at = ?
            WHERE campaign_id = ? AND experiment_id = ?
            """,
            (
                verdict,
                json.dumps({**payload, "status": verdict, "summary_metrics": summary_metrics}, default=str),
                time.time(),
                self.campaign_id,
                experiment_id,
            ),
        )
        run_memory = self.memory_fabric.create_memory(
            memory_kind="experiment_run",
            payload_json={
                "experiment_id": experiment_id,
                "name": name,
                "verdict": verdict,
                "runs": run_outputs,
            },
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="experiment_runner.run",
            summary_text=f"{name} {verdict}",
            experiment_id=experiment_id,
            lifecycle_state=verdict,
            importance_score=0.9,
            confidence_score=1.0 if verdict == "passed" else 0.5,
        )
        result_memory = self.memory_fabric.create_memory(
            memory_kind="experiment_result",
            payload_json={
                "experiment_id": experiment_id,
                "verdict": verdict,
                "summary_metrics": summary_metrics,
            },
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="experiment_runner.run",
            summary_text=f"{name} metrics",
            experiment_id=experiment_id,
            lifecycle_state=verdict,
            importance_score=0.95,
            confidence_score=1.0 if verdict == "passed" else 0.55,
        )
        self.memory_projector.project_memory(
            self.memory_fabric,
            run_memory,
            include_multivector=True,
        )
        self.memory_projector.project_memory(self.memory_fabric, result_memory)
        self.memory_fabric.add_edge(
            MemoryEdge(
                src_memory_id=run_memory.memory_id,
                dst_memory_id=spec_memory.memory_id,
                edge_type="tests",
                evidence_json={"experiment_id": experiment_id},
            )
        )
        self.memory_fabric.add_edge(
            MemoryEdge(
                src_memory_id=result_memory.memory_id,
                dst_memory_id=run_memory.memory_id,
                edge_type="derived_from",
                evidence_json={"experiment_id": experiment_id},
            )
        )
        latent_capture = self.latent_bridge.capture_summary_package(
            memory_id=result_memory.memory_id,
            summary_text=f"{name} {verdict} {summary_metrics}",
            capture_stage="experiment_result_interpretation",
            supporting_memory_ids=[spec_memory.memory_id, run_memory.memory_id],
            creation_reason="experiment result summary",
        )
        return {
            "experiment_id": experiment_id,
            "verdict": verdict,
            "runs": run_outputs,
            "summary_metrics": summary_metrics,
            "almf": {
                "experiment_spec_memory_id": spec_memory.memory_id,
                "experiment_run_memory_id": run_memory.memory_id,
                "experiment_result_memory_id": result_memory.memory_id,
                "latent_capture": latent_capture,
            },
        }

    def run_lane(
        self,
        task: dict[str, Any],
        *,
        default_timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        lane_id = str(task.get("lane_id") or f"lane_{uuid.uuid4().hex[:12]}")
        editable_scope = list(task.get("editable_scope") or [])
        telemetry_contract = dict(
            task.get("telemetry_contract")
            or self.telemetry_contracts.build(
                str(task.get("caller_mode") or "development")
            )
        )
        task_packet = {
            "task_packet_id": lane_id,
            "campaign_id": self.campaign_id,
            "phase_id": task.get("caller_mode") or "development",
            "role": "ExperimentLane",
            "specialist_role": "ExperimentLane",
            "objective": task.get("objective_function") or "",
            "description": task.get("description") or task.get("name") or "",
            "repo_scope": list(task.get("allowed_writes") or ["target"]),
            "artifact_scope": ["experiments", "telemetry"],
            "editable_scope": editable_scope,
            "read_only_scope": list(task.get("read_only_scope") or []),
            "telemetry_contract": telemetry_contract,
            "promotion_policy": dict(task.get("promotion_policy") or {}),
        }
        self.state_store.record_task_packet(task_packet)
        shadow_preflight = self.shadow_preflight(
            task,
            telemetry_contract=telemetry_contract,
            default_timeout_seconds=default_timeout_seconds,
        )
        worktree = self.worktrees.prepare(lane_id, editable_scope)
        baseline = self.baselines.capture(
            editable_scope=editable_scope,
            imported_baseline=task.get("metadata", {}).get("imported_baseline"),
        )

        run_result = self.run(
            name=str(task.get("name") or lane_id),
            commands=list(task.get("commands") or []),
            default_timeout_seconds=default_timeout_seconds,
            environment=str(task.get("caller_mode") or "local_shell"),
            cwd=str(worktree.get("workspace_dir") or self.cwd),
            extra_env={
                "ANVIL_LANE_ID": lane_id,
                "ANVIL_LANE_WORKSPACE": str(worktree.get("workspace_dir") or ""),
                "ANVIL_LANE_SOURCE_ROOT": self.cwd,
            },
        )
        current_metrics = self._metrics_for_score(run_result["summary_metrics"])
        telemetry_check = self.telemetry_contracts.evaluate(
            run_result["summary_metrics"],
            telemetry_contract,
        )
        current_metrics["telemetry_contract_satisfied"] = 1.0 if telemetry_check[
            "contract_satisfied"
        ] else 0.0
        scorecard = self.scorecard.score(
            dict(task.get("metadata", {}).get("imported_baseline_metrics") or {}),
            current_metrics,
            complexity_penalty=float(
                task.get("metadata", {}).get("complexity_penalty", 0.0)
            ),
            instability_penalty=float(
                task.get("metadata", {}).get("instability_penalty", 0.0)
            ),
        )
        decision = self.promotion.evaluate(
            scorecard,
            telemetry_check,
            policy=dict(task.get("promotion_policy") or {}),
        )
        finalized = self.worktrees.finalize(lane_id, keep=decision["verdict"] == "keep")
        branch_metrics = {
            "changed_files": list(finalized.get("changed_files") or []),
            "verify_result": bool(telemetry_check["contract_satisfied"]),
            "test_delta": float(current_metrics.get("success_count", 0.0))
            - float(task.get("metadata", {}).get("baseline_success_count", 0.0)),
            "runtime_cost": float(
                run_result["summary_metrics"]
                .get("aggregate_metrics", {})
                .get("wall_time_seconds", 0.0)
            ),
            "shadow_success_probability": float(
                shadow_preflight.get("shadow_success_probability") or 0.0
            ),
            "predicted_contract_gaps": list(
                shadow_preflight.get("predicted_contract_gaps") or []
            ),
            "predicted_runtime_cost": float(
                shadow_preflight.get("predicted_runtime_cost") or 0.0
            ),
        }
        bundle = {
            "lane_id": lane_id,
            "caller_mode": str(task.get("caller_mode") or "development"),
            "lane_type": str(task.get("lane_type") or "bounded_file_experiment"),
            "task": task,
            "baseline": baseline,
            "worktree": worktree,
            "experiment": run_result,
            "shadow_preflight": shadow_preflight,
            "telemetry_check": telemetry_check,
            "scorecard": scorecard,
            "promotion": decision,
            "finalized": finalized,
            "branch_metrics": branch_metrics,
        }
        lane_memory = self.memory_fabric.create_memory(
            memory_kind="latent_branch",
            payload_json=bundle,
            campaign_id=self.campaign_id,
            workspace_id=self.campaign_id,
            source_system="experiment_runner.run_lane",
            summary_text=str(task.get("name") or lane_id),
            lane_id=lane_id,
            task_packet_id=lane_id,
            lifecycle_state=str(decision["verdict"]),
            importance_score=0.9,
            confidence_score=0.75,
        )
        self.memory_projector.project_memory(self.memory_fabric, lane_memory)
        bundle["development_replay"] = self.latent_bridge.capture_summary_package(
            memory_id=lane_memory.memory_id,
            summary_text=f"{lane_id} {decision['verdict']} {scorecard}",
            capture_stage="experimental_lane_design",
            creation_reason="development lane summary",
        )
        self.state_store.record_task_run(
            lane_id,
            status=str(decision["verdict"]),
            result={
                "campaign_id": self.campaign_id,
                "lane_id": lane_id,
                "promotion": decision,
                "scorecard": scorecard,
                "telemetry_check": telemetry_check,
                "branch_metrics": branch_metrics,
            },
        )
        self.state_store.record_telemetry(
            self.campaign_id,
            telemetry_kind="experiment_lane",
            payload={
                "lane_id": lane_id,
                "caller_mode": bundle["caller_mode"],
                "lane_type": bundle["lane_type"],
                "score": decision["score"],
                "verdict": decision["verdict"],
                "telemetry_contract_satisfied": telemetry_check[
                    "contract_satisfied"
                ],
                "required_metrics": telemetry_check["required_metrics"],
                "missing_metrics": telemetry_check["missing_metrics"],
                "branch_metrics": branch_metrics,
                "shadow_preflight": shadow_preflight,
            },
            task_packet_id=lane_id,
        )
        for metric_name, metric_value in current_metrics.items():
            self.state_store.record_experiment_telemetry(
                self.campaign_id,
                run_result["experiment_id"],
                metric_name,
                float(metric_value),
            )
        self.state_store.insert_json_row(
            "loop_runs",
            campaign_id=self.campaign_id,
            payload=bundle,
            id_field="loop_id",
            id_value=lane_id,
            status=str(decision["verdict"]),
        )
        return bundle

    def shadow_preflight(
        self,
        task: dict[str, Any],
        *,
        telemetry_contract: dict[str, Any] | None = None,
        default_timeout_seconds: int = 30,
    ) -> dict[str, Any]:
        contract = dict(
            telemetry_contract
            or task.get("telemetry_contract")
            or self.telemetry_contracts.build(str(task.get("caller_mode") or "development"))
        )
        commands = [
            self._normalize_command(command, default_timeout_seconds=default_timeout_seconds)
            for command in list(task.get("commands") or [])
        ]
        recent_telemetry = list(self.state_store.list_telemetry(self.campaign_id))[-40:]
        historical_verdicts = [
            item
            for item in recent_telemetry
            if item.get("telemetry_kind") in {"experiment_lane", "verification_lane"}
        ]
        historical_counterexamples = [
            str(counterexample)
            for item in historical_verdicts
            for counterexample in list(item.get("counterexamples") or [])
        ]
        preflight = self.shadow_planner.estimate_shadow_preflight(
            required_metrics=list(contract.get("required_metrics") or []),
            command_count=max(1, len(commands)),
            shell_commands=len([spec for spec in commands if bool(spec.get("shell"))]),
            timeout_budget_seconds=sum(
                float(spec.get("timeout_seconds") or default_timeout_seconds)
                for spec in commands
            ),
            historical_verdicts=historical_verdicts,
            historical_counterexamples=historical_counterexamples,
            predicted_artifacts=list(task.get("artifact_scope") or []),
        )
        payload = {
            "lane_id": str(task.get("lane_id") or ""),
            "telemetry_contract": contract,
            "editable_scope": list(task.get("editable_scope") or []),
            **preflight,
        }
        self.state_store.record_telemetry(
            self.campaign_id,
            telemetry_kind="shadow_preflight",
            payload=payload,
            task_packet_id=str(task.get("lane_id") or "") or None,
        )
        return payload

    def _normalize_command(
        self,
        command: str | dict[str, Any],
        *,
        default_timeout_seconds: int,
    ) -> dict[str, Any]:
        if isinstance(command, str):
            return {
                "label": command.split()[0] if command.strip() else "command",
                "command": command,
                "shell": True,
                "timeout_seconds": default_timeout_seconds,
                "env": None,
            }
        return {
            "label": str(command.get("label") or command.get("name") or "command"),
            "command": command.get("argv") or str(command.get("command") or ""),
            "shell": bool(command.get("shell", not bool(command.get("argv")))),
            "timeout_seconds": int(command.get("timeout_seconds", default_timeout_seconds)),
            "env": {**os.environ, **dict(command.get("env") or {})} or None,
        }

    def _serialize_command(self, command: str | dict[str, Any]) -> dict[str, Any]:
        spec = self._normalize_command(command, default_timeout_seconds=30)
        return {
            "label": spec["label"],
            "command": spec["command"],
            "shell": spec["shell"],
            "timeout_seconds": spec["timeout_seconds"],
        }

    def _extract_metrics(self, stdout: str, stderr: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for stream in (stdout, stderr):
            for key, value in self.METRIC_PATTERN.findall(stream):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
        return metrics

    @staticmethod
    def _summarize_metrics(
        run_outputs: list[dict[str, Any]],
        experiment_started: float,
    ) -> dict[str, Any]:
        aggregate_metrics: dict[str, float] = {}
        for result in run_outputs:
            for key, value in result["metrics"].items():
                aggregate_metrics[key] = value
        return {
            "command_count": len(run_outputs),
            "success_count": len([item for item in run_outputs if item["returncode"] == 0]),
            "failure_count": len([item for item in run_outputs if item["returncode"] != 0]),
            "wall_time_seconds": time.time() - experiment_started,
            "aggregate_metrics": aggregate_metrics,
        }

    @staticmethod
    def _metrics_for_score(summary_metrics: dict[str, Any]) -> dict[str, float]:
        aggregate = {
            key: float(value)
            for key, value in dict(summary_metrics.get("aggregate_metrics") or {}).items()
            if isinstance(value, (int, float))
        }
        aggregate["wall_time_seconds"] = float(
            summary_metrics.get("wall_time_seconds") or 0.0
        )
        aggregate["command_count"] = float(summary_metrics.get("command_count") or 0.0)
        aggregate["success_count"] = float(summary_metrics.get("success_count") or 0.0)
        aggregate["failure_count"] = float(summary_metrics.get("failure_count") or 0.0)
        aggregate["correctness_pass"] = 1.0 if aggregate["failure_count"] == 0 else 0.0
        aggregate["determinism_pass"] = float(
            aggregate.get("determinism_pass", aggregate.get("replayability", 1.0))
        )
        return aggregate
