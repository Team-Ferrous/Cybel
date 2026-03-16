"""Base runtime for deterministic multi-phase campaigns."""

from __future__ import annotations

import ast
import functools
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from core.campaign.audit_engine import CampaignAuditEngine
from core.campaign.completion_engine import CompletionEngine
from core.campaign.gate_engine import GateEngine
from core.campaign.ledger import TheLedger
from core.campaign.loop_scheduler import LoopDefinition, LoopScheduler
from core.campaign.repo_registry import CampaignRepoRegistry
from core.campaign.state_store import CampaignStateStore
from core.campaign.telemetry import CampaignTelemetry
from core.campaign.workspace import CampaignWorkspace
from core.agents.specialists import SpecialistRegistry

try:
    from config.settings import CAMPAIGN_CONFIG as DEFAULT_CAMPAIGN_CONFIG
except Exception:
    DEFAULT_CAMPAIGN_CONFIG = {}


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class AgentResult:
    summary: str
    raw_result: Any = None
    structured_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CampaignReport:
    campaign_id: str
    campaign_name: str
    campaign_version: str
    started_at: Optional[float]
    completed_at: Optional[float]
    duration_seconds: float
    phase_statuses: Dict[str, str]
    metrics: Dict[str, Any]
    gate_verdicts: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "campaign_version": self.campaign_version,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "phase_statuses": dict(self.phase_statuses),
            "metrics": dict(self.metrics),
            "gate_verdicts": list(self.gate_verdicts),
        }


@dataclass
class CampaignState:
    campaign_id: str
    campaign_name: str
    campaign_version: str
    state_path: str
    campaign_path: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    phase_statuses: Dict[str, str] = field(default_factory=dict)
    phase_attempts: Dict[str, int] = field(default_factory=dict)
    phase_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_state: str = "INTAKE"
    active_loop: Optional[str] = None
    completed_loops: List[str] = field(default_factory=list)
    last_error: Optional[str] = None

    def save(self) -> None:
        payload = {
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "campaign_version": self.campaign_version,
            "campaign_path": self.campaign_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "phase_statuses": self.phase_statuses,
            "phase_attempts": self.phase_attempts,
            "phase_results": self.phase_results,
            "current_state": self.current_state,
            "active_loop": self.active_loop,
            "completed_loops": self.completed_loops,
            "last_error": self.last_error,
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as handle:
            import json

            json.dump(payload, handle, indent=2, default=str)

    @classmethod
    def load_or_new(
        cls,
        campaign_id: str,
        campaign_name: str,
        campaign_version: str,
        state_path: str,
    ) -> "CampaignState":
        if os.path.exists(state_path):
            try:
                import json

                with open(state_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                return cls(
                    campaign_id=payload.get("campaign_id", campaign_id),
                    campaign_name=payload.get("campaign_name", campaign_name),
                    campaign_version=payload.get("campaign_version", campaign_version),
                    campaign_path=payload.get("campaign_path"),
                    state_path=state_path,
                    started_at=payload.get("started_at"),
                    completed_at=payload.get("completed_at"),
                    phase_statuses=dict(payload.get("phase_statuses", {})),
                    phase_attempts=dict(payload.get("phase_attempts", {})),
                    phase_results=dict(payload.get("phase_results", {})),
                    current_state=payload.get("current_state", "INTAKE"),
                    active_loop=payload.get("active_loop"),
                    completed_loops=list(payload.get("completed_loops", [])),
                    last_error=payload.get("last_error"),
                )
            except Exception:
                pass
        return cls(
            campaign_id=campaign_id,
            campaign_name=campaign_name,
            campaign_version=campaign_version,
            state_path=state_path,
        )


def phase(
    order: int,
    name: str,
    depends_on: Optional[List[str]] = None,
    files: Optional[List[str]] = None,
    memory_policy: str = "keep_model",
):
    """Decorator to mark a method as a campaign phase.

    Parameters
    ----------
    memory_policy : str
        ``"keep_model"`` (default) — model stays loaded during this phase.
        ``"suspend_model"`` — model weights are evicted before the phase
        runs and reloaded after it completes.  Use for test / build phases
        on CPU-only systems where RAM is tight.
    """

    def decorator(func):
        func._is_phase = True
        func._phase_order = order
        func._phase_name = name
        func._phase_depends_on = list(depends_on or [])
        func._phase_files = list(files or [])
        func._phase_memory_policy = memory_policy

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper._is_phase = True
        wrapper._phase_order = order
        wrapper._phase_name = name
        wrapper._phase_depends_on = list(depends_on or [])
        wrapper._phase_files = list(files or [])
        wrapper._phase_memory_policy = memory_policy
        return wrapper

    return decorator


def gate(phase: str):
    """Decorator to mark a method as a phase gate."""

    def decorator(func):
        func._is_gate = True
        func._gate_for_phase = phase
        return func

    return decorator


class BaseCampaignLoop:
    """Base class for deterministic, code-first campaign execution."""

    campaign_name: str = "Unnamed Campaign"
    campaign_version: str = "1.0"
    max_retries_per_phase: int = 2

    def __init__(
        self,
        root_dir: str = ".",
        brain_factory: Optional[Callable[[], Any]] = None,
        console=None,
        config: Optional[Dict[str, Any]] = None,
        campaign_id: Optional[str] = None,
        ownership_registry=None,
    ):
        merged_config = dict(DEFAULT_CAMPAIGN_CONFIG)
        merged_config.update(config or {})
        self.config = merged_config
        self.root_dir = os.path.abspath(root_dir)
        self.brain_factory = brain_factory
        self.console = console
        self.ownership_registry = ownership_registry
        self.event_store = self.config.get("event_store")

        self.campaign_id = campaign_id or self._make_campaign_id()
        state_dir = self.config.get("state_dir", ".anvil/campaigns/state")
        state_path = os.path.join(state_dir, f"{self.campaign_id}.json")
        ledger_path = self.config.get(
            "ledger_db_path", ".anvil/campaigns/campaign_ledger.db"
        )
        workspace_dir = self.config.get("workspace_dir") or os.path.dirname(
            os.path.abspath(state_dir)
        )

        self.ledger = TheLedger(
            campaign_name=self.campaign_name,
            campaign_id=self.campaign_id,
            db_path=ledger_path,
        )
        self.state = CampaignState.load_or_new(
            campaign_id=self.campaign_id,
            campaign_name=self.campaign_name,
            campaign_version=self.campaign_version,
            state_path=state_path,
        )
        self.workspace = CampaignWorkspace.load(
            self.campaign_id,
            base_dir=workspace_dir,
        )
        self.workspace.save_metadata(
            {
                "campaign_name": self.campaign_name,
                "campaign_version": self.campaign_version,
                "root_dir": self.root_dir,
            }
        )
        self.state_store = CampaignStateStore(self.workspace.db_path)
        self.state_store.initialize_campaign(
            self.campaign_id,
            name=self.campaign_name,
            objective=self.config.get("objective", ""),
            current_state=self.state.current_state,
            metadata={
                "campaign_version": self.campaign_version,
                "root_dir": self.root_dir,
            },
        )
        self.repo_registry = CampaignRepoRegistry(self.workspace, self.state_store)
        self.gate_engine = GateEngine()
        self.loop_scheduler = LoopScheduler()
        self.telemetry = CampaignTelemetry(self.state_store, self.campaign_id)
        self.audit_engine = CampaignAuditEngine(self.state_store, self.campaign_id)
        self.completion_engine = CompletionEngine(
            self.workspace,
            self.state_store,
            self.campaign_id,
        )
        self.specialist_registry = SpecialistRegistry()

        self._phases = self._discover_phases()
        self._gates = self._discover_gates()
        self._phase_name_to_method = {
            phase_info["name"]: phase_info["method_name"] for phase_info in self._phases
        }
        self._register_default_loops()
        self._ensure_registered_target_repo()

        for phase_info in self._phases:
            self.state.phase_statuses.setdefault(
                phase_info["method_name"], PhaseStatus.PENDING.value
            )
            self.state.phase_attempts.setdefault(phase_info["method_name"], 0)

        self.max_retries_per_phase = int(
            self.config.get("max_retries_per_phase", self.max_retries_per_phase)
        )
        self.state.save()

    def _register_default_loops(self) -> None:
        default_loops = [
            LoopDefinition(
                loop_id="repo_ingestion_loop",
                purpose="Register target and analysis repos for the campaign.",
                produced_artifacts=["intake"],
                allowed_repo_roles=["target", "analysis_local", "analysis_external"],
                allowed_tools=["filesystem", "git"],
                stop_conditions=["all_attached_repos_registered"],
                escalation_conditions=["missing_repo_metadata"],
                retry_policy={"max_attempts": 1},
                telemetry_contract={"required_metrics": ["artifact_emission_status"]},
                promotion_effect="REPO_INGESTION",
                controlling_state="INTAKE",
            ),
            LoopDefinition(
                loop_id="development_execution_loop",
                purpose="Execute deterministic phase-oriented development work.",
                produced_artifacts=["roadmap_final", "telemetry"],
                allowed_repo_roles=["target"],
                allowed_tools=["python", "tests", "agents"],
                stop_conditions=["all_phase_gates_pass"],
                escalation_conditions=["phase_failure"],
                retry_policy={"max_attempts": self.max_retries_per_phase + 1},
                telemetry_contract={"required_metrics": ["wall_time", "artifact_emission_status"]},
                promotion_effect="AUDIT",
                controlling_state="DEVELOPMENT",
            ),
            LoopDefinition(
                loop_id="closure_proof_loop",
                purpose="Run final audit and emit closure proof.",
                produced_artifacts=["audits", "closure"],
                allowed_repo_roles=["artifact_store"],
                allowed_tools=["sqlite", "filesystem"],
                stop_conditions=["closure_proof_emitted"],
                escalation_conditions=["material_audit_findings"],
                retry_policy={"max_attempts": 1},
                telemetry_contract={"required_metrics": ["artifact_emission_status"]},
                promotion_effect="CLOSURE",
                controlling_state="AUDIT",
            ),
        ]
        self.loop_scheduler.register_many(default_loops)

    def _ensure_registered_target_repo(self) -> None:
        existing = self.repo_registry.resolve_repo_for_path(self.root_dir)
        if existing is not None:
            return
        self.repo_registry.register_repo(
            name=os.path.basename(self.root_dir.rstrip(os.sep)) or "target",
            local_path=self.root_dir,
            role="target",
            origin=self.root_dir,
            metadata={"registered_by": "BaseCampaignLoop"},
        )

    def _make_campaign_id(self) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.campaign_name).strip("_").lower()
        slug = slug or "campaign"
        return f"{slug}_{int(time.time())}"

    def _discover_phases(self) -> List[Dict[str, Any]]:
        phases: List[Dict[str, Any]] = []
        for name in dir(self):
            method = getattr(self, name, None)
            if callable(method) and getattr(method, "_is_phase", False):
                phases.append(
                    {
                        "method": method,
                        "name": method._phase_name,
                        "order": method._phase_order,
                        "depends_on": method._phase_depends_on,
                        "files": method._phase_files,
                        "method_name": name,
                        "memory_policy": getattr(method, "_phase_memory_policy", "keep_model"),
                    }
                )
        return sorted(phases, key=lambda item: item["order"])

    def _discover_gates(self) -> Dict[str, Callable]:
        gates: Dict[str, Callable] = {}
        for name in dir(self):
            method = getattr(self, name, None)
            if callable(method) and getattr(method, "_is_gate", False):
                gates[method._gate_for_phase] = method
        return gates

    def _resolve_phase_files(self, phase_info: Dict[str, Any]) -> List[str]:
        explicit = list(phase_info.get("files") or [])
        if explicit:
            return explicit

        method = phase_info["method"]
        finder = getattr(method, "_phase_files_resolver", None)
        if callable(finder):
            try:
                output = finder(self)
                return [path for path in (output or []) if isinstance(path, str)]
            except Exception:
                return []
        return []

    def run(self) -> CampaignReport:
        self.log(
            f"[Campaign] Starting: {self.campaign_name} ({len(self._phases)} phases)"
        )

        if self.state.started_at is None:
            self.state.started_at = time.time()
            self.state.save()
        self.transition_state("DEVELOPMENT", cause="campaign_run_started")

        halt_on_failure = bool(self.config.get("halt_on_failure", True))

        for phase_info in self._phases:
            phase_id = phase_info["method_name"]
            current = self.state.phase_statuses.get(phase_id, PhaseStatus.PENDING.value)

            if current == PhaseStatus.PASSED.value:
                self.log(f"[Phase] SKIPPED: {phase_info['name']} (already passed)")
                continue

            if not self._dependencies_met(phase_info):
                self.state.phase_statuses[phase_id] = PhaseStatus.SKIPPED.value
                self.log(f"[Phase] SKIPPED: {phase_info['name']} (dependencies not met)")
                self.state.save()
                continue

            success = False
            retries = self.max_retries_per_phase + 1
            starting_attempt = self.state.phase_attempts.get(phase_id, 0)
            if current in {PhaseStatus.FAILED.value, PhaseStatus.RETRYING.value}:
                starting_attempt = 0

            for attempt in range(starting_attempt, retries):
                attempt_display = f" (retry {attempt})" if attempt > 0 else ""
                self.log(f"[Phase {phase_info['order']}] {phase_info['name']}{attempt_display}")
                self.state.phase_statuses[phase_id] = PhaseStatus.RUNNING.value
                self.state.phase_attempts[phase_id] = attempt
                self.state.active_loop = "development_execution_loop"
                self.state.save()

                claimed_files: List[str] = []
                claim_agent_id = f"Campaign:{self.campaign_id}:{phase_id}"
                phase_span = self.telemetry.start_span(
                    telemetry_kind="phase_execution",
                    task_packet_id=phase_id,
                    metadata={"phase_id": phase_id, "phase_name": phase_info["name"]},
                )

                try:
                    phase_files = self._resolve_phase_files(phase_info)
                    if self.ownership_registry is not None and phase_files:
                        claim = self.ownership_registry.claim_files(
                            agent_id=claim_agent_id,
                            files=phase_files,
                            mode="exclusive",
                            phase_id=phase_id,
                            task_id=phase_id,
                        )
                        claimed_files = list(claim.granted_files)
                        if not claim.success and not claimed_files:
                            reason = (
                                claim.suggested_resolution
                                or f"Ownership claim failed for phase {phase_id}"
                            )
                            raise AssertionError(reason)

                    result = self._run_phase_with_memory_policy(
                        phase_info, phase_id
                    )

                    gate_fn = self._gates.get(phase_id)
                    gate_reason = "No gate defined"
                    if gate_fn is not None:
                        gate_fn(result)
                        gate_reason = "Gate assertions passed"

                    self.state.phase_statuses[phase_id] = PhaseStatus.PASSED.value
                    self.state.phase_results[phase_id] = result or {}
                    self.state.last_error = None
                    self.ledger.record_phase_result(phase_id, "passed", result or {})
                    self.ledger.record_gate_verdict(phase_id, True, gate_reason)
                    self.log(f"[Gate] PASSED - {phase_info['name']}")
                    self.telemetry.finish_span(
                        phase_span,
                        status="passed",
                        metrics={
                            "wall_time": time.time() - float(phase_span["started_at"]),
                            "artifact_emission_status": "present",
                            "retry_count": attempt,
                        },
                    )
                    success = True
                    break

                except AssertionError as exc:
                    reason = str(exc)
                    self.state.last_error = reason
                    self.ledger.record_phase_result(
                        phase_id,
                        "failed",
                        {"error": reason, "attempt": attempt},
                    )
                    self.ledger.record_gate_verdict(phase_id, False, reason)

                    if attempt >= retries - 1:
                        self.state.phase_statuses[phase_id] = PhaseStatus.FAILED.value
                        self.log(f"[Gate] FAILED - {reason}")
                    else:
                        self.state.phase_statuses[phase_id] = PhaseStatus.RETRYING.value
                        self.ledger.record_artifact(
                            f"retry_context_{phase_id}_{attempt}",
                            f"Gate failed: {reason}",
                        )
                        self.log(f"[Gate] RETRYING - {reason}")
                    self.telemetry.finish_span(
                        phase_span,
                        status="failed",
                        metrics={
                            "wall_time": time.time() - float(phase_span["started_at"]),
                            "artifact_emission_status": "missing",
                            "retry_count": attempt,
                            "error": reason,
                        },
                    )

                except Exception as exc:
                    reason = str(exc)
                    self.state.last_error = reason
                    self.state.phase_statuses[phase_id] = PhaseStatus.FAILED.value
                    self.ledger.record_phase_result(
                        phase_id,
                        "error",
                        {"error": reason, "attempt": attempt},
                    )
                    self.ledger.record_gate_verdict(phase_id, False, reason)
                    self.log(f"[Phase] ERROR - {reason}")

                    if attempt < retries - 1:
                        self.state.phase_statuses[phase_id] = PhaseStatus.RETRYING.value
                        self.log(f"[Phase] RETRYING - {phase_info['name']}")
                    else:
                        self.log(f"[Phase] FAILED - {phase_info['name']}")
                    self.telemetry.finish_span(
                        phase_span,
                        status="error",
                        metrics={
                            "wall_time": time.time() - float(phase_span["started_at"]),
                            "artifact_emission_status": "missing",
                            "retry_count": attempt,
                            "error": reason,
                        },
                    )

                finally:
                    if self.ownership_registry is not None and claimed_files:
                        try:
                            self.ownership_registry.release_files(
                                agent_id=claim_agent_id,
                                files=claimed_files,
                            )
                        except Exception:
                            pass
                    self.state.phase_attempts[phase_id] = attempt + 1
                    self.state.save()

            if not success and halt_on_failure:
                self.log(f"[Campaign] Halting after phase failure: {phase_info['name']}")
                self.transition_state(
                    "REMEDIATION",
                    cause="phase_failure",
                    metadata={"phase_id": phase_id},
                )
                break

        if all(
            status == PhaseStatus.PASSED.value
            for status in self.state.phase_statuses.values()
        ):
            self.transition_state("AUDIT", cause="all_phases_passed")
            audit_run = self.audit_engine.run(scope="campaign")
            self.ledger.record_artifact("campaign_audit", audit_run)
            self.transition_state("CLOSURE", cause="audit_complete")
            closure_path = self.completion_engine.persist_proof()
            self.ledger.record_artifact("closure_proof", closure_path)

        self.state.completed_at = time.time()
        self.state.active_loop = None
        self.state.save()
        return self._generate_report()

    def _run_phase_with_memory_policy(
        self,
        phase_info: Dict[str, Any],
        phase_id: str,
    ) -> Any:
        """Execute a phase method, optionally suspending the model first."""
        policy = phase_info.get("memory_policy", "keep_model")

        if policy == "suspend_model":
            return self._run_phase_suspended(phase_info, phase_id)

        # Default: keep_model — run directly
        return phase_info["method"]()

    def _run_phase_suspended(
        self,
        phase_info: Dict[str, Any],
        phase_id: str,
    ) -> Any:
        """Run a phase with model weights evicted and reloaded around it."""
        brain = None
        if callable(self.brain_factory):
            # The brain_factory callable gives us the singleton brain
            try:
                brain = self.brain_factory()
            except Exception:
                pass

        if brain is None:
            self.log(
                f"[Memory] suspend_model policy set for {phase_info['name']} "
                f"but no brain available — running without suspension"
            )
            return phase_info["method"]()

        from core.native.model_suspender import ModelSuspender

        suspender = ModelSuspender(
            brain,
            reason=f"campaign_phase:{phase_id}",
            force=True,
        )
        self.log(
            f"[Memory] Suspending model for phase: {phase_info['name']}"
        )

        with suspender:
            result = phase_info["method"]()

        self.log(
            f"[Memory] Model reloaded after phase: {phase_info['name']} "
            f"(freed ≈ {suspender.memory_freed_mb:.0f} MB, "
            f"reload: {suspender.reload_seconds:.1f}s)"
        )
        return result

    def discover_files(
        self,
        extensions: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[str]:
        extensions = extensions or [".py"]
        exclude = exclude or [".venv", "node_modules", "__pycache__", ".git", ".anvil"]

        files: List[str] = []
        for root, dirs, filenames in os.walk(self.root_dir):
            dirs[:] = [entry for entry in dirs if entry not in exclude]
            for filename in filenames:
                if any(filename.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, filename)
                    files.append(os.path.relpath(full_path, self.root_dir))

        return sorted(files)

    def discover_entry_points(self) -> List[str]:
        entry_points: List[str] = []
        py_files = self.discover_files(extensions=[".py"])

        for rel_path in py_files:
            if rel_path in {"main.py", "setup.py"}:
                module = rel_path[:-3].replace(os.sep, ".")
                entry_points.append(module)
                continue

            abs_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                if "if __name__ == \"__main__\"" in content:
                    module = rel_path[:-3].replace(os.sep, ".")
                    if module.endswith(".__init__"):
                        module = module[: -len(".__init__")]
                    entry_points.append(module)
            except Exception:
                continue

        return sorted(set(entry_points))

    def discover_modules(self) -> List[str]:
        modules: List[str] = []
        for root, _, files in os.walk(self.root_dir):
            if "__init__.py" not in files:
                continue
            rel = os.path.relpath(root, self.root_dir)
            if rel.startswith("."):
                continue
            module = rel.replace(os.sep, ".")
            modules.append(module)
        return sorted(set(modules))

    def build_dependency_graph(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for rel_path in self.discover_files(extensions=[".py"]):
            imports: List[str] = []
            abs_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as handle:
                    tree = ast.parse(handle.read(), filename=rel_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)
            except Exception:
                pass
            graph[rel_path] = sorted(set(imports))
        return graph

    def detect_patterns(self) -> Dict[str, int]:
        counters = {
            "todo_comments": 0,
            "fixme_comments": 0,
            "bare_except": 0,
            "print_statements": 0,
        }

        for rel_path in self.discover_files(extensions=[".py"]):
            abs_path = os.path.join(self.root_dir, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8") as handle:
                    source = handle.read()
                counters["todo_comments"] += source.count("TODO")
                counters["fixme_comments"] += source.count("FIXME")
                counters["print_statements"] += source.count("print(")

                tree = ast.parse(source, filename=rel_path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        counters["bare_except"] += 1
            except Exception:
                continue

        return counters

    def run_shell(self, command: str, timeout: int = 60) -> subprocess.CompletedProcess:
        if command.startswith("python "):
            command = f"{sys.executable} {command[len('python '):]}"
        return subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.root_dir,
        )

    def spawn_agent(
        self,
        objective: str,
        files: Optional[List[str]] = None,
        context_from_ledger: bool = False,
        loop_type: str = "enhanced",
        phase_id: Optional[str] = None,
    ) -> AgentResult:
        del loop_type  # currently reserved for future strategy routing

        objective_text = objective
        if context_from_ledger:
            summary = self.ledger.get_context_summary(
                budget_tokens=int(self.config.get("ledger_max_tokens", 10000))
            )
            objective_text = f"{objective}\n\nLedger Context:\n{summary}"

        if files:
            file_list = "\n".join(f"- {path}" for path in files)
            objective_text = f"{objective_text}\n\nScoped files:\n{file_list}"

        task_packet = self.emit_task_packet(
            objective=objective,
            specialist_role="ImplementationEngineerSubagent",
            roadmap_item_id=phase_id,
            phase_id=phase_id,
            allowed_repos=["target"],
            expected_artifacts=["agent_summary"],
            evidence_bundle={"files": list(files or [])},
        )
        claim_agent_id = f"CampaignAgent:{self.campaign_id}:{phase_id or 'general'}"
        claimed_files: List[str] = []

        if self.ownership_registry is not None and files:
            mode = "shared_read"
            claim = self.ownership_registry.claim_files(
                agent_id=claim_agent_id,
                files=files,
                mode=mode,
                phase_id=phase_id,
                task_id=phase_id,
            )
            claimed_files = list(claim.granted_files)

        try:
            from agents.unified_master import UnifiedMasterAgent

            brain = self.brain_factory() if callable(self.brain_factory) else None
            master = UnifiedMasterAgent(brain=brain, console=self.console)
            raw_result = master.run_mission(objective_text)
            self.state_store.record_task_run(
                task_packet["task_packet_id"],
                status="completed",
                result={"raw_result": raw_result},
            )
            structured = raw_result if isinstance(raw_result, dict) else {}
            summary = self._summarize_agent_result(raw_result)
            return AgentResult(summary=summary, raw_result=raw_result, structured_output=structured)
        except Exception as exc:
            self.state_store.record_task_run(
                task_packet["task_packet_id"],
                status="failed",
                result={"error": str(exc)},
            )
            return AgentResult(
                summary=f"Agent execution unavailable: {exc}",
                raw_result=None,
                structured_output={},
            )
        finally:
            if self.ownership_registry is not None and claimed_files:
                try:
                    self.ownership_registry.release_files(
                        agent_id=claim_agent_id,
                        files=claimed_files,
                    )
                except Exception:
                    pass

    @staticmethod
    def _summarize_agent_result(raw_result: Any) -> str:
        if raw_result is None:
            return "Agent mission executed"
        if isinstance(raw_result, dict):
            if "summary" in raw_result:
                return str(raw_result["summary"])
            return "Agent mission completed with structured output"
        return str(raw_result)

    def get_module_files(self, module_name: str) -> List[str]:
        module_prefix = module_name.replace(".", os.sep)
        return [
            path
            for path in self.discover_files(extensions=[".py"])
            if path.startswith(module_prefix)
        ]

    def save_report(self, content: str) -> str:
        report_dir = self.config.get("reports_dir", ".anvil/campaigns/reports")
        os.makedirs(report_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.campaign_name).strip("_").lower()
        filename = f"{slug or 'campaign'}_{stamp}.md"
        path = os.path.join(report_dir, filename)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def register_repo_resource(
        self,
        name: str,
        path: str,
        role: str = "analysis",
        read_only: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.ledger.record_resource(
            name=name,
            path=path,
            role=role,
            read_only=read_only,
            metadata=metadata,
        )
        mapped_role = {
            "analysis": "analysis_local",
            "target": "target",
            "artifact": "artifact_store",
        }.get(role, role)
        self.repo_registry.register_repo(
            name=name,
            local_path=path,
            role=mapped_role,
            origin=path,
            write_policy="immutable" if read_only else None,
            metadata=metadata,
        )

    def record_evidence(
        self,
        name: str,
        summary: str,
        evidence_type: str = "finding",
        confidence: str = "medium",
        payload: Optional[Dict[str, Any]] = None,
        source_phase: Optional[str] = None,
    ) -> None:
        self.ledger.record_evidence(
            name=name,
            summary=summary,
            evidence_type=evidence_type,
            confidence=confidence,
            payload=payload,
            source_phase=source_phase,
        )

    def log(self, message: str) -> None:
        if self.console is not None and hasattr(self.console, "print"):
            self.console.print(message)
        if self.event_store is not None:
            try:
                self.event_store.emit(
                    event_type="campaign.log",
                    payload={
                        "campaign_id": self.campaign_id,
                        "campaign_name": self.campaign_name,
                        "message": message,
                    },
                    source="campaign",
                )
            except Exception:
                pass

    def transition_state(
        self,
        to_state: str,
        *,
        cause: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        event = self.state_store.transition_state(
            self.campaign_id,
            to_state=to_state,
            cause=cause,
            metadata=metadata,
        )
        self.state.current_state = to_state
        self.state.save()
        return event

    def emit_task_packet(
        self,
        *,
        objective: str,
        specialist_role: str,
        roadmap_item_id: Optional[str] = None,
        phase_id: Optional[str] = None,
        allowed_repos: Optional[List[str]] = None,
        forbidden_repos: Optional[List[str]] = None,
        expected_artifacts: Optional[List[str]] = None,
        evidence_bundle: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        aes_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        packet_model = self.specialist_registry.build_task_packet(
            objective=objective,
            aal=str((aes_metadata or {}).get("aal") or "AAL-3"),
            domains=(aes_metadata or {}).get("domains") or [],
            repo_roles=allowed_repos or [],
            allowed_repos=allowed_repos or [],
            forbidden_repos=forbidden_repos or [],
            required_artifacts=expected_artifacts or [],
            produced_artifacts=expected_artifacts or [],
            allowed_tools=allowed_tools or [],
        )
        packet = packet_model.to_dict()
        packet.update(
            {
                "task_packet_id": packet["task_packet_id"],
                "campaign_id": self.campaign_id,
                "roadmap_item_id": roadmap_item_id,
                "phase_id": phase_id,
                "specialist_role": specialist_role or packet["specialist_role"],
                "objective": objective,
                "allowed_repos": list(allowed_repos or []),
                "forbidden_repos": list(forbidden_repos or []),
                "allowed_tools": list(allowed_tools or []),
                "expected_artifacts": list(expected_artifacts or []),
                "evidence_bundle": dict(evidence_bundle or {}),
                "aes_metadata": dict(aes_metadata or {}),
                "status": "queued",
            }
        )
        self.state_store.record_task_packet(packet)
        return packet

    def control_plane_snapshot(self) -> Dict[str, Any]:
        artifacts = self.state_store.list_artifacts(self.campaign_id)
        questions = self.state_store.list_questions(self.campaign_id)
        features = self.state_store.list_features(self.campaign_id)
        repos = self.repo_registry.list_repos()
        return {
            "campaign_state": self.state.current_state,
            "artifact_families": sorted({row["family"] for row in artifacts}),
            "approved_families": sorted(
                {
                    row["family"]
                    for row in artifacts
                    if row["approval_state"] in {"approved", "accepted"}
                }
            ),
            "blocking_questions": len(
                [
                    row
                    for row in questions
                    if row["current_status"] not in {"answered", "waived"}
                    and row["blocking_level"] in {"high", "critical"}
                ]
            ),
            "pending_feature_confirmation": len(
                [
                    row
                    for row in features
                    if row["requires_user_confirmation"]
                    and row["selection_state"] not in {"approved", "selected"}
                ]
            ),
            "repo_roles": [repo.role for repo in repos],
        }

    def _dependencies_met(self, phase_info: Dict[str, Any]) -> bool:
        dependencies = list(phase_info.get("depends_on") or [])
        if not dependencies:
            return True

        if "all_previous" in dependencies:
            for candidate in self._phases:
                if candidate["order"] >= phase_info["order"]:
                    continue
                status = self.state.phase_statuses.get(candidate["method_name"])
                if status != PhaseStatus.PASSED.value:
                    return False
            return True

        for dependency in dependencies:
            method_name = self._phase_name_to_method.get(dependency, dependency)
            status = self.state.phase_statuses.get(method_name)
            if status != PhaseStatus.PASSED.value:
                return False
        return True

    def _generate_report(self) -> CampaignReport:
        started_at = self.state.started_at
        completed_at = self.state.completed_at
        duration = 0.0
        if started_at is not None and completed_at is not None:
            duration = max(0.0, completed_at - started_at)

        return CampaignReport(
            campaign_id=self.campaign_id,
            campaign_name=self.campaign_name,
            campaign_version=self.campaign_version,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            phase_statuses=dict(self.state.phase_statuses),
            metrics={
                **self.ledger.get_all_metrics(),
                "campaign_state": self.state.current_state,
                "control_plane": self.control_plane_snapshot(),
            },
            gate_verdicts=self.ledger.get_all_gate_verdicts(),
        )
