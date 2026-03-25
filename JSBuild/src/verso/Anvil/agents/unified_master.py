from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from agents.recovery import RecoveryManager
from config.settings import AES_ENFORCEMENT_ENABLED, OWNERSHIP_CONFIG
from core.aes import AALClassifier, DomainDetector
from core.agent import BaseAgent
from core.agents.specialists import SpecialistRegistry
from core.agents.subagent_quality_gate import SubagentQualityGate
from core.ownership.file_ownership import FileOwnershipRegistry
from core.subagent_communication import get_message_bus
from saguaro.workset import WorksetManager
from shared_kernel.event_store import get_event_store


class UnifiedMasterAgent(BaseAgent):
    """
    Unified governance-aware orchestrator.
    """

    def __init__(self, **kwargs):
        super().__init__(name="UnifiedMasterAgent", **kwargs)
        if not hasattr(self, "brain"):
            self.brain = kwargs.get("brain")
        if not hasattr(self, "console"):
            self.console = kwargs.get("console")
        if not hasattr(self, "hook_registry"):
            from infrastructure.hooks.registry import HookRegistry

            self.hook_registry = HookRegistry()
        self.recovery = RecoveryManager(self)
        self.legacy_finalization_allowed = not AES_ENFORCEMENT_ENABLED
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.quality_gate = SubagentQualityGate(
            repo_root=getattr(getattr(self, "semantic_engine", None), "root_dir", "."),
            brain=getattr(self, "brain", None),
        )
        self.specialist_registry = SpecialistRegistry()
        self.message_bus = kwargs.get("message_bus") or get_message_bus()
        self.event_store = get_event_store()
        self.ownership_enabled = bool(OWNERSHIP_CONFIG.get("enabled", False))
        self.workset_manager = None
        self.ownership_registry = None
        if self.ownership_enabled:
            try:
                self.workset_manager = WorksetManager(
                    saguaro_dir=".saguaro",
                    repo_path=getattr(
                        getattr(self, "semantic_engine", None), "root_dir", "."
                    ),
                )
                self.ownership_registry = FileOwnershipRegistry(
                    workset_manager=self.workset_manager,
                    message_bus=self.message_bus,
                    event_store=self.event_store,
                    instance_id="local",
                    default_ttl_seconds=int(
                        OWNERSHIP_CONFIG.get("lease_ttl_seconds", 300)
                    ),
                )
            except Exception:
                self.workset_manager = None
                self.ownership_registry = None
                self.ownership_enabled = False

    def run_mission(self, user_objective: str):
        self.console.print(
            f"[*] UNIFIED MASTER AGENT: Received objective: {user_objective}",
            style="bold green",
        )

        plan_prompt = f"""
        OBJECTIVE: {user_objective}

        You are the Master Architect. Break this into atomic, sequential tasks.
        Each task must specify:
        1. 'id': sequential integer
        2. 'role': a specific role for the task (e.g., researcher, architect, implementer, validator).
        3. 'task': specific and detailed instructions for the task.

        Return ONLY a JSON list of objects.
        """
        self.console.print("[*] Generating execution plan...", style="cyan")
        plan_raw = self.brain.generate(plan_prompt)
        self.console.print("[*] Execution plan generated.", style="cyan")

        try:
            tasks = json.loads(plan_raw)
            if not isinstance(tasks, list):
                raise ValueError("Plan is not a list.")
        except (json.JSONDecodeError, ValueError) as e:
            self.console.print(
                f"[!] FAILED: Could not decode plan. Error: {e}", style="bold red"
            )
            self.console.print(f"Raw response was:\n{plan_raw}")
            return

        prepared_tasks = self._prepare_tasks(
            tasks, self.ownership_registry if self.ownership_enabled else None
        )
        for task_obj in prepared_tasks:
            task_id = task_obj.get("id", "N/A")
            role = task_obj.get("role")
            desc = task_obj.get("task")
            aal = task_obj.get("aal", "AAL-3")
            domains = task_obj.get("domains", [])
            assigned_agent_id = task_obj.get(
                "assigned_agent_id", f"SubAgent:{role}:{task_id}"
            )
            ownership_conflicts = task_obj.get("ownership_conflicts", [])

            if not role or not desc:
                self.console.print(
                    f"[!] SKIPPING invalid task object: {task_obj}", style="yellow"
                )
                continue

            if ownership_conflicts:
                self.console.print(
                    f"[!] Task {task_id} blocked by ownership conflicts: {ownership_conflicts}",
                    style="yellow",
                )
                continue

            self.console.print(
                f"\n[+] Executing Task {task_id} ({role}, {aal}): {desc}",
                style="bold magenta",
            )
            try:
                task_result = self._execute_with_chronicle(
                    task_obj, lambda: self._execute_task_with_retries(task_obj)
                )
                if not task_result.get("success"):
                    self.console.print(
                        f"[!] Task {task_id} FAILED. Reason: {task_result.get('reason')}",
                        style="bold red",
                    )
                    self.console.print(
                        "[!] Mission aborted due to task failure.", style="bold red"
                    )
                    break

                payload = task_result.get("payload") or {}
                crs = int(task_result.get("crs", 0))
                self.console.print(
                    f"[*] Task {task_id} COMPLETED successfully (CRS={crs}, AAL={aal}).",
                    style="green",
                )
                self.history.add_message(
                    "assistant",
                    f"Sub-agent task '{desc}' completed. CRS={crs}, AAL={aal}",
                )
                self.history.add_message("tool", json.dumps(payload, default=str))
            finally:
                if (
                    self.ownership_enabled
                    and self.ownership_registry is not None
                    and OWNERSHIP_CONFIG.get("auto_release_on_completion", True)
                ):
                    self.ownership_registry.release_files(
                        agent_id=assigned_agent_id, files=task_obj.get("owned_files")
                    )
        else:
            self.console.print(
                "\n[SUCCESS] All tasks completed. Mission successful.",
                style="bold green",
            )

    def _prepare_tasks(
        self,
        tasks: Iterable[Dict[str, Any]],
        ownership_registry: Optional[FileOwnershipRegistry] = None,
    ) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for item in tasks:
            if not isinstance(item, dict):
                continue
            task_text = str(item.get("task") or "").strip()
            if not task_text:
                continue
            role = str(item.get("role") or "implementer").strip() or "implementer"
            task_id = str(item.get("id") or len(prepared) + 1)
            phase_id = str(item.get("phase_id") or "").strip() or None

            required_files = self._predict_task_files(item, task_text)
            write_files = self._predict_write_files(task_text, required_files)
            read_files = [path for path in required_files if path not in write_files]
            detected_domains = sorted(
                self.domain_detector.detect_from_description(task_text)
            )
            aal = self.aal_classifier.classify_from_description(task_text)
            routing = self.specialist_registry.route(
                objective=task_text,
                domains=detected_domains,
                repo_roles=["target"] if write_files else ["analysis_local"],
                question_type=str(item.get("question_type") or ""),
                aal=aal,
            )
            assigned_agent_id = f"SubAgent:{routing.primary_role}:{task_id}"
            task_packet = self.specialist_registry.build_task_packet(
                objective=task_text,
                aal=aal,
                domains=detected_domains,
                repo_roles=["target"] if write_files else ["analysis_local"],
                allowed_repos=["target"] if write_files else ["analysis_local"],
                required_artifacts=item.get("required_artifacts") or [],
                produced_artifacts=item.get("produced_artifacts") or [],
                question_type=str(item.get("question_type") or ""),
                allowed_tools=item.get("allowed_tools") or [],
            )

            ownership_conflicts: List[Dict[str, Any]] = []
            granted_write_files: List[str] = list(write_files)
            if ownership_registry is not None and write_files:
                claim = ownership_registry.claim_files(
                    agent_id=assigned_agent_id,
                    files=write_files,
                    mode="exclusive",
                    phase_id=phase_id,
                    task_id=task_id,
                )
                granted_write_files = list(claim.granted_files)
                ownership_conflicts = [
                    denied_file.__dict__ for denied_file in claim.denied_files
                ]
                if not claim.success:
                    self._resolve_ownership_conflict(task_text, claim.suggested_resolution)

            prepared.append(
                {
                    **item,
                    "id": task_id,
                    "task": task_text,
                    "role": routing.primary_role or role,
                    "requested_role": role,
                    "aal": aal,
                    "domains": detected_domains,
                    "reviewer_roles": routing.reviewer_roles,
                    "routing_reasons": routing.reasons,
                    "task_packet": task_packet.to_dict(),
                    "assigned_agent_id": assigned_agent_id,
                    "owned_files": granted_write_files,
                    "read_files": read_files,
                    "phase_id": phase_id,
                    "ownership_workset_id": None,
                    "ownership_conflicts": ownership_conflicts,
                }
            )
        # AAL-2/3 first; AAL-0/1 last.
        rank = {"AAL-3": 0, "AAL-2": 1, "AAL-1": 2, "AAL-0": 3}
        prepared.sort(key=lambda entry: rank.get(str(entry.get("aal")), 1))
        return prepared

    def _predict_task_files(self, task: Dict[str, Any], task_text: str) -> List[str]:
        files = list(task.get("context_files") or [])
        file_pattern = r"[\w./-]+\.(?:py|cc|cpp|c|h|hpp|md|json|toml|yaml|yml|txt)"
        files.extend(re.findall(file_pattern, task_text))
        files.extend(task.get("codebase_files") or [])

        normalized = []
        seen = set()
        for file_path in files:
            if not isinstance(file_path, str):
                continue
            candidate = file_path.strip().replace("\\", "/")
            if not candidate:
                continue
            if os.path.isabs(candidate):
                try:
                    candidate = os.path.relpath(candidate, ".")
                except Exception:
                    pass
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    @staticmethod
    def _predict_write_files(task_text: str, required_files: List[str]) -> List[str]:
        write_intent = re.search(
            r"\b(implement|modify|change|edit|refactor|update|create|delete|fix|patch|rename)\b",
            task_text,
            flags=re.IGNORECASE,
        )
        if write_intent is None:
            return []
        return list(required_files)

    def _resolve_ownership_conflict(
        self, task_text: str, suggested_resolution: Optional[str]
    ) -> None:
        guidance = suggested_resolution or "Task queued until file ownership is released."
        self.console.print(
            f"[yellow]Ownership conflict detected for task '{task_text[:80]}...': {guidance}[/yellow]"
        )

    def _execute_task_with_retries(self, task_obj: Dict[str, Any]) -> Dict[str, Any]:
        role = task_obj.get("role")
        desc = task_obj.get("task")
        aal = task_obj.get("aal", "AAL-3")
        domains = task_obj.get("domains", [])
        compliance = self._ensure_runtime_compliance_context(desc)
        compliance["red_team_required"] = str(aal).upper() in {"AAL-0", "AAL-1"}
        self.current_red_team_required = compliance["red_team_required"]

        current_task = desc
        for attempt in range(1, 4):
            dispatch_context = {
                "trace_id": compliance.get("trace_id"),
                "task": current_task,
                "task_id": task_obj.get("id"),
                "aal": aal,
                "domains": domains,
                "role": role,
            }
            self.hook_registry.execute("pre_dispatch", dispatch_context)

            raw_result = self._execute_tool(
                {
                    "name": "execute_subagent_task",
                    "arguments": {
                        "role": role,
                        "task": current_task,
                        "aal": aal,
                        "domains": domains,
                        "compliance": compliance,
                    },
                }
            )

            payload = self._parse_subagent_payload(
                raw_result,
                current_task,
                aal,
                domains,
                compliance=compliance,
            )
            gate = self.quality_gate.evaluate(
                payload,
                original_query=current_task,
                complexity_score=max(1, min(10, len(current_task.split()) // 4 + 1)),
            )
            payload["subagent_quality"] = gate
            if not gate.get("accepted", False):
                current_task = self.recovery.handle_failure(
                    current_task,
                    f"Subagent quality gate failed: {json.dumps(gate, default=str)}",
                    attempt,
                )
                if not current_task:
                    return {"success": False, "reason": "quality gate failed"}
                continue

            crs = self._compute_crs(task_obj, payload)
            validation = self._review_with_crs(current_task, payload, crs)
            dispatch_context.update(
                {"crs": crs, "quality_gate": gate, "validation": validation}
            )
            self.hook_registry.execute("post_dispatch", dispatch_context)

            if validation.upper().startswith("YES"):
                return {"success": True, "payload": payload, "crs": crs}

            current_task = self.recovery.handle_failure(current_task, validation, attempt)
            if not current_task:
                return {"success": False, "reason": validation}

        return {"success": False, "reason": "max retries exceeded"}

    def _review_with_crs(self, task: str, payload: Dict[str, Any], crs: int) -> str:
        review_mode = "standard review"
        if crs >= 5:
            review_mode = "full red-team review"
        elif crs >= 3:
            review_mode = "FMEA-focused review"

        review_prompt = (
            f"Review mode: {review_mode}\n"
            f"Task: {task}\n"
            f"CRS: {crs}\n"
            f"Sub-agent result:\n{payload.get('subagent_analysis', '')}\n\n"
            "Is this result CORRECT and COMPLETE? Your answer must start with either "
            "'YES' or 'NO:'."
        )
        self.console.print("[*] Reviewing sub-agent result...", style="cyan")
        return self.brain.generate(review_prompt)

    def _compute_crs(self, task: Dict[str, Any], payload: Dict[str, Any]) -> int:
        score = 0
        aal = str(task.get("aal", "AAL-3")).upper()
        text = str(payload.get("subagent_analysis", ""))
        if aal == "AAL-0":
            score += 3
        elif aal == "AAL-1":
            score += 2
        if self._modifies_public_api(text):
            score += 2
        if self._changes_schema(text):
            score += 2
        if self._adds_dependency(text):
            score += 2
        if not self._includes_tests(text):
            score += 2
        return score

    @staticmethod
    def _modifies_public_api(text: str) -> bool:
        return bool(
            re.search(
                r"\b(public\s+api|breaking\s+change|def\s+[A-Za-z_]\w+\(.*\)\s*->)",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _changes_schema(text: str) -> bool:
        return bool(
            re.search(
                r"\b(schema|migration|alter\s+table|ddl|protobuf|json\s*schema)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _adds_dependency(text: str) -> bool:
        return bool(
            re.search(
                r"\b(requirements\.txt|pyproject\.toml|package\.json|poetry\.lock|new dependency)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _includes_tests(text: str) -> bool:
        return bool(
            re.search(r"\b(test_|pytest|unittest|assert )\b|tests?/", text, re.IGNORECASE)
        )

    def _execute_with_chronicle(self, task: Dict[str, Any], executor):
        if str(task.get("aal", "AAL-3")).upper() not in {"AAL-0", "AAL-1"}:
            return executor()
        try:
            self._execute_tool(
                {
                    "name": "run_command",
                    "arguments": {"command": "venv/bin/saguaro chronicle snapshot"},
                }
            )
        except Exception:
            pass
        result = executor()
        try:
            diff = self._execute_tool(
                {
                    "name": "run_command",
                    "arguments": {"command": "venv/bin/saguaro chronicle diff"},
                }
            )
            self.history.add_message(
                "assistant",
                f"Chronicle telemetry for {task.get('task')}: {str(diff)[:500]}",
            )
        except Exception:
            pass
        return result

    @staticmethod
    def _parse_subagent_payload(
        raw_result: str,
        task: str,
        aal: str,
        domains: Optional[Iterable[str]],
        compliance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            payload = json.loads(raw_result)
            if not isinstance(payload, dict):
                payload = {}
        except Exception:
            payload = {}
        payload.setdefault("subagent_analysis", raw_result)
        payload.setdefault("task", task)
        payload.setdefault("aal", aal)
        payload.setdefault("domains", list(domains or []))
        payload.setdefault("codebase_files", [])
        payload.setdefault("compliance", dict(compliance or {}))
        return payload
