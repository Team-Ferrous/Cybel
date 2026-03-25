import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.agent import BaseAgent


class ExecuteSubagentTaskTool:
    """
    Tool that allows a master agent to dispatch a specific task to a SubAgent.
    """

    schema = {
        "name": "execute_subagent_task",
        "description": "Executes a specific, atomic task using a transient SubAgent with a designated role.",
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "The role for the sub-agent (e.g., researcher, architect, implementer, validator).",
                },
                "task": {
                    "type": "string",
                    "description": "The specific and detailed instructions for the task.",
                },
                "aal": {
                    "type": "string",
                    "description": "Optional AES assurance level assigned to the task.",
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional AES domains associated with the task.",
                },
                "compliance": {
                    "type": "object",
                    "description": "Optional compliance context (trace/evidence/waivers).",
                },
            },
            "required": ["role", "task"],
        },
    }

    def __init__(self, agent: "BaseAgent"):
        """
        Initializes the tool.

        Args:
            agent: The parent agent instance, used to access shared components like the Saguaro substrate.
        """
        self.agent = agent
        from core.agents.specialists import SpecialistRegistry

        self._specialists = SpecialistRegistry()
        self._aal_classifier = None
        self._domain_detector = None
        try:
            from core.aes import AALClassifier, DomainDetector

            self._aal_classifier = AALClassifier()
            self._domain_detector = DomainDetector()
        except Exception:
            self._aal_classifier = None
            self._domain_detector = None

    def execute(
        self,
        role: str,
        task: str,
        aal: str = "AAL-2",
        domains: list[str] | None = None,
        compliance: dict | None = None,
    ) -> str:
        """
        Executes the sub-agent task.

        Args:
            role: The role for the sub-agent.
            task: The specific task description.

        Returns:
            The result or output from the sub-agent's execution.
        """
        from core.agents.specialists import build_specialist_subagent, route_specialist

        requested_role = str(role or "").strip()
        task_text = str(task or "").strip()
        detected_domains = set(domains or [])
        if self._domain_detector is not None:
            detected_domains.update(self._domain_detector.detect_from_description(task_text))
        effective_aal = "AAL-2"
        if aal:
            effective_aal = str(aal)
        elif self._aal_classifier is not None:
            effective_aal = str(self._aal_classifier.classify_from_description(task_text))
        compliance_context = dict(compliance or {})
        compliance_context.setdefault(
            "red_team_required",
            str(effective_aal).upper() in {"AAL-0", "AAL-1"},
        )

        question_hint = ""
        lowered_role = requested_role.lower()
        if "architect" in lowered_role:
            question_hint = "architecture"
        elif "research" in lowered_role:
            question_hint = "research"
        elif "analy" in lowered_role or "investigat" in lowered_role:
            question_hint = "investigation"

        routing = route_specialist(
            registry=self._specialists,
            objective=task_text,
            requested_role=requested_role,
            aal=effective_aal,
            domains=sorted(detected_domains),
            question_type=question_hint,
            repo_roles=["target"] if lowered_role in {"implementer", "implementation"} else ["analysis_local"],
        )
        selected_role = routing.primary_role
        prompt_key = self._specialists.prompt_key_for_role(selected_role)

        sub_agent = build_specialist_subagent(
            role=selected_role,
            task=task_text,
            parent_name=getattr(self.agent, "name", "Master"),
            brain=getattr(self.agent, "brain", None),
            console=getattr(self.agent, "console", None),
            parent_agent=self.agent,
            message_bus=getattr(self.agent, "message_bus", None),
            ownership_registry=getattr(self.agent, "ownership_registry", None),
            prompt_profile="sovereign_build",
            specialist_prompt_key=prompt_key,
            sovereign_build_policy_enabled=True,
            prompt_injection=(
                f"Requested role: {requested_role or 'none'}\n"
                f"Routing reasons: {', '.join(routing.reasons) if routing.reasons else 'none'}\n"
                "Provide evidence-backed delegated output."
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
            "aal": effective_aal,
            "domains": sorted(detected_domains),
            "routing_reasons": list(routing.reasons),
            "reviewer_roles": list(routing.reviewer_roles),
            "compliance": compliance_context,
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
            "latent": result.get("latent", {}) if isinstance(result, dict) else {},
            "evidence_envelope": (
                result.get("evidence_envelope", {}) if isinstance(result, dict) else {}
            ),
        }
        return json.dumps(payload, default=str)
