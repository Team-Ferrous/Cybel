from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

from agents.recovery import RecoveryManager
from agents.unified_master import UnifiedMasterAgent
from config.settings import AES_ENFORCEMENT_ENABLED, MASTER_MODEL
from core.aes import AALClassifier, DomainDetector
from core.agents.specialists import SpecialistRegistry, build_specialist_subagent, route_specialist
from core.agents.subagent_quality_gate import SubagentQualityGate
from core.ollama_client import DeterministicOllama
from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate


class MasterAgent:
    """
    The 'Persistent Entity'. Holds global context and orchestrates sub-agents.
    """

    def __init__(self, use_unified_adapter=None):
        self.brain = DeterministicOllama(MASTER_MODEL)
        self.saguaro = SaguaroSubstrate()
        self.recovery = RecoveryManager(self)
        self.context_memory = []  # Simulated "Episodic Memory"
        self.aal_classifier = AALClassifier()
        self.domain_detector = DomainDetector()
        self.quality_gate = SubagentQualityGate(
            repo_root=getattr(self.saguaro, "root_dir", "."),
            brain=self.brain,
        )
        self.specialist_registry = SpecialistRegistry()
        if use_unified_adapter is None:
            use_unified_adapter = AES_ENFORCEMENT_ENABLED
        self._delegate = UnifiedMasterAgent() if use_unified_adapter else None

    def run_mission(self, user_objective):
        if self._delegate is not None:
            return self._delegate.run_mission(user_objective)

        print(f"[*] MASTER AGENT: Received objective: {user_objective}")

        # 1. Decomposition Phase
        plan_prompt = f"""
        OBJECTIVE: {user_objective}
        
        You are the Master Architect. Break this into atomic, sequential tasks.
        Each task must specify:
        1. 'id': sequential integer
        2. 'role': researcher, architect, implementer, or validator
        3. 'task': specific instructions
        
        Return ONLY a JSON list of objects. Example:
        [
            {{"id": 1, "role": "researcher", "task": "find auth logic"}},
            {{"id": 2, "role": "implementer", "task": "add logging to auth"}}
        ]
        """
        plan_raw = self.brain.generate(plan_prompt)
        print("[*] MASTER AGENT: Plan generated.")

        tasks = self.recovery.sanitize_json(plan_raw)
        if not tasks:
            print("[!] MASTER AGENT: Failed to decompose objective. Aborting.")
            return

        prepared_tasks = self._prepare_tasks(tasks)
        for task_obj in prepared_tasks:
            task_id = task_obj.get("id")
            role = task_obj.get("role")
            desc = task_obj.get("task")
            aal = task_obj.get("aal", "AAL-3")
            domains = task_obj.get("domains", [])

            print(f"\n[+] ACTIVATING SUB-AGENT for Task {task_id} ({role}): {desc}")

            attempt = 1
            success = False
            current_task = desc

            while not success and attempt <= 3:
                compliance = self._build_compliance_context(task_id, current_task, aal)
                raw_result = self._execute_with_chronicle(
                    task_obj,
                    lambda: self.dispatch_subagent(
                        role,
                        current_task,
                        aal=aal,
                        domains=domains,
                        compliance=compliance,
                    ),
                )
                payload = self._parse_payload(
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
                if not gate.get("accepted", False):
                    current_task = self.recovery.handle_failure(
                        current_task,
                        f"Subagent quality gate failed: {json.dumps(gate, default=str)}",
                        attempt,
                    )
                    if not current_task:
                        break
                    attempt += 1
                    continue

                crs = self._compute_crs(task_obj, payload)
                validation = self._review_with_crs(current_task, payload, crs)

                if validation.upper().startswith("YES"):
                    print(f"[*] MASTER AGENT: Task {task_id} COMPLETED.")
                    self.context_memory.append(
                        {
                            "task": desc,
                            "result": payload.get("subagent_analysis", raw_result),
                            "aal": aal,
                            "domains": domains,
                            "crs": crs,
                            "quality_gate": gate,
                        }
                    )
                    success = True
                else:
                    current_task = self.recovery.handle_failure(
                        desc, validation, attempt
                    )
                    if not current_task:
                        break
                    attempt += 1

    def _prepare_tasks(self, tasks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for item in tasks:
            if not isinstance(item, dict):
                continue
            task_text = str(item.get("task") or "").strip()
            if not task_text:
                continue
            role = str(item.get("role") or "implementer").strip() or "implementer"
            prepared.append(
                {
                    **item,
                    "role": role,
                    "task": task_text,
                    "aal": self.aal_classifier.classify_from_description(task_text),
                    "domains": sorted(
                        self.domain_detector.detect_from_description(task_text)
                    ),
                }
            )
        # AAL-2/3 first, AAL-0/1 last.
        rank = {"AAL-3": 0, "AAL-2": 1, "AAL-1": 2, "AAL-0": 3}
        prepared.sort(key=lambda entry: rank.get(str(entry.get("aal")), 1))
        return prepared

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
            "Is this CORRECT and COMPLETE? Answer with 'YES' or 'NO: <reason>'."
        )
        return self.brain.generate(review_prompt)

    def _compute_crs(self, task: Dict[str, Any], payload: Dict[str, Any]) -> int:
        score = 0
        aal = str(task.get("aal", "AAL-3")).upper()
        analysis = str(payload.get("subagent_analysis", ""))
        if aal == "AAL-0":
            score += 3
        elif aal == "AAL-1":
            score += 2
        if self._modifies_public_api(analysis):
            score += 2
        if self._changes_schema(analysis):
            score += 2
        if self._adds_dependency(analysis):
            score += 2
        if not self._includes_tests(analysis):
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
            re.search(
                r"\b(test_|pytest|unittest|assert )\b|tests?/",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _execute_with_chronicle(self, task: Dict[str, Any], executor):
        if str(task.get("aal", "AAL-3")).upper() not in {"AAL-0", "AAL-1"}:
            return executor()
        try:
            self.saguaro.execute_command("chronicle snapshot")
        except Exception:
            pass
        result = executor()
        try:
            diff = self.saguaro.execute_command("chronicle diff")
            self.context_memory.append(
                {"task": task.get("task"), "chronicle_diff": str(diff), "aal": task.get("aal")}
            )
        except Exception:
            pass
        return result

    @staticmethod
    def _parse_payload(
        raw_result: str,
        task: str,
        aal: str,
        domains: Optional[Iterable[str]],
        compliance: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(raw_result)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}
        parsed.setdefault("subagent_analysis", raw_result)
        parsed.setdefault("task", task)
        parsed.setdefault("aal", aal)
        parsed.setdefault("domains", list(domains or []))
        parsed.setdefault("codebase_files", [])
        parsed.setdefault("compliance", dict(compliance or {}))
        return parsed

    @staticmethod
    def _build_compliance_context(
        task_id: Any, task: str, aal: str
    ) -> Dict[str, Any]:
        task_token = re.sub(r"[^a-zA-Z0-9]+", "-", str(task).strip())[:24] or "task"
        trace_id = f"master::{task_id or 'na'}::{task_token}"
        evidence_bundle_id = f"evidence::{trace_id}"
        return {
            "trace_id": trace_id,
            "evidence_bundle_id": evidence_bundle_id,
            "red_team_required": str(aal).upper() in {"AAL-0", "AAL-1"},
            "waiver_ids": [],
        }

    def dispatch_subagent(
        self,
        role,
        task,
        aal: str = "AAL-2",
        domains: Optional[Iterable[str]] = None,
        compliance: Optional[Dict[str, Any]] = None,
    ):
        """
        Launches a transient sub-agent for a specific task.
        """
        requested_role = str(role or "").strip()
        task_text = str(task or "").strip()
        detected_domains = set(domains or [])
        detected_domains.update(self.domain_detector.detect_from_description(task_text))
        routing = route_specialist(
            registry=self.specialist_registry,
            objective=task_text,
            requested_role=requested_role,
            aal=aal,
            domains=sorted(detected_domains),
            question_type="",
            repo_roles=["analysis_local"],
        )
        selected_role = routing.primary_role
        prompt_key = self.specialist_registry.prompt_key_for_role(selected_role)
        sub_agent = build_specialist_subagent(
            role=selected_role,
            task=task_text,
            parent_name="MasterAgent",
            brain=self.brain,
            message_bus=None,
            prompt_profile="sovereign_build",
            specialist_prompt_key=prompt_key,
            sovereign_build_policy_enabled=True,
            prompt_injection=(
                f"Requested role: {requested_role or 'none'}\n"
                f"Routing reasons: {', '.join(routing.reasons) if routing.reasons else 'none'}"
            ),
        )
        result = sub_agent.run(
            task_text,
            prompt_profile="sovereign_build",
            specialist_prompt_key=prompt_key,
        )
        payload = {
            "requested_role": requested_role,
            "role": selected_role,
            "task": task_text,
            "aal": aal,
            "domains": sorted(detected_domains),
            "routing_reasons": list(routing.reasons),
            "reviewer_roles": list(routing.reviewer_roles),
            "compliance": dict(compliance or {}),
            "subagent_analysis": (
                result.get("summary", "") if isinstance(result, dict) else str(result)
            ),
            "subagent_full_response": (
                result.get("full_response", "")
                if isinstance(result, dict)
                else str(result)
            ),
            "codebase_files": (
                result.get("files_read", []) if isinstance(result, dict) else []
            ),
        }
        return json.dumps(payload, default=str)
