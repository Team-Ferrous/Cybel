from typing import Any


class DelegateTool:
    """
    Tool that allows the master agent to delegate a specific task to a sub-agent.
    Sub-agents have their own isolated context and history.
    """

    schema = {
        "name": "delegate",
        "description": "Delegate a complex sub-task to a specialized sub-agent. Use this for research, analysis, or isolated implementation steps.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The specific task description for the sub-agent.",
                },
                "quiet": {
                    "type": "boolean",
                    "description": "Whether the sub-agent should run silently without printing to the console.",
                    "default": False,
                },
            },
            "required": ["task"],
        },
    }

    def __init__(self, console: Any = None, brain: Any = None):
        self.console = console
        self.brain = brain
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

    def execute(self, task: str, quiet: bool = False) -> str:
        """
        Execute the delegation.

        Args:
            task: Task description
            quiet: If True, suppress console output

        Returns:
            Summary of the sub-agent's work
        """
        from core.agents.specialists import build_specialist_subagent, route_specialist

        task_text = str(task or "").strip()
        domains = []
        if self._domain_detector is not None:
            domains = sorted(self._domain_detector.detect_from_description(task_text))
        aal = "AAL-2"
        if self._aal_classifier is not None:
            aal = str(self._aal_classifier.classify_from_description(task_text))

        routing = route_specialist(
            registry=self._specialists,
            objective=task_text,
            requested_role="",
            aal=aal,
            domains=domains,
            question_type="",
            repo_roles=["analysis_local"],
        )
        selected_role = routing.primary_role
        prompt_key = self._specialists.prompt_key_for_role(selected_role)
        agent = build_specialist_subagent(
            role=selected_role,
            task=task_text,
            quiet=quiet,
            brain=self.brain,
            console=self.console,
            prompt_profile="sovereign_build",
            specialist_prompt_key=prompt_key,
            sovereign_build_policy_enabled=True,
            prompt_injection=(
                f"Delegated role selected by router: {selected_role}\n"
                f"Routing reasons: {', '.join(routing.reasons) if routing.reasons else 'none'}"
            ),
        )
        result = agent.run(task_text, specialist_prompt_key=prompt_key)
        summary = (
            result.get("summary", "SubAgent completed but produced no summary.")
            if isinstance(result, dict)
            else str(result)
        )
        return f"[{selected_role}] {summary}"
