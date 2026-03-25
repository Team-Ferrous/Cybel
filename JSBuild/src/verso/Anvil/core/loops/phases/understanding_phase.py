from typing import Any, Dict, Optional
from core.loops.phases.base_phase import BasePhase
from core.thinking import ThinkingType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class UnderstandingPhase(BasePhase):
    """
    Phase 1: Understanding - Classify request and check memory.
    """

    def execute(
        self, user_input: str, context: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> Dict[str, Any]:
        logger.info("Executing Understanding Phase")

        if dashboard:
            dashboard.update_phase("Understanding", status="in_progress")

        # Initial Deep Thinking (Understanding)
        if self.loop.enhanced_mode:
            self.loop.thinking_system.start_chain(
                task_id=self.loop.current_task_id,
                compliance_context=getattr(
                    self.loop, "current_compliance_context", None
                ),
            )
            self.loop.thinking_system.think(
                ThinkingType.UNDERSTANDING,
                f"User Input: {user_input}\nContext Analysis: Determining true intent and routing.",
            )

        # Classify the request
        request_type = self.loop._classify_request(user_input)
        logger.info(f"Request classified as: {request_type}")
        classification_meta = getattr(self.loop, "_last_classification_meta", {})
        complexity_profile = context.get("complexity_profile")
        if complexity_profile is None and hasattr(self.loop, "complexity_scorer"):
            complexity_profile = self.loop.complexity_scorer.score_request(user_input)

        if dashboard:
            dashboard.update_phase(
                "Understanding", status="completed", message=f"Type: {request_type}"
            )

        # Check memory (enhanced mode)
        if self.loop.enhanced_mode:
            logger.info("Checking task memory for user input")
            if dashboard:
                dashboard.add_agent("Understanding", "Memory", status="running")

            self._check_memory(user_input)

            if dashboard:
                dashboard.update_agent("Memory", status="completed", progress=1.0)

        return {
            "request_type": request_type,
            "classification_meta": classification_meta,
            "complexity_profile": complexity_profile,
        }

    def _classify_request(self, user_input: str) -> str:
        """
        Classify the user request into categories.
        Uses deterministic keyword matching + heuristics.
        """
        input_lower = user_input.lower()

        # Question patterns
        if any(
            kw in input_lower
            for kw in [
                "how does",
                "how do",
                "explain",
                "describe",
                "what is",
                "where is",
                "why",
                "when",
                "who",
                "which",
                "can you tell",
                "what are",
                "how can",
                "is there",
                "does it",
            ]
        ):
            return "question"

        # Creation patterns
        if any(
            kw in input_lower
            for kw in [
                "create",
                "add",
                "write",
                "implement",
                "build",
                "generate",
                "make a",
                "new file",
                "scaffold",
            ]
        ):
            return "creation"

        # Modification patterns
        if any(
            kw in input_lower
            for kw in [
                "edit",
                "modify",
                "change",
                "update",
                "fix",
                "refactor",
                "rename",
                "move",
                "improve",
                "optimize",
            ]
        ):
            return "modification"

        # Deletion patterns
        if any(
            kw in input_lower
            for kw in ["delete", "remove", "drop", "clear", "clean up"]
        ):
            return "deletion"

        # Investigation patterns
        if any(
            kw in input_lower
            for kw in [
                "search for",
                "find",
                "investigate",
                "analyze",
                "explore",
                "look for",
                "locate",
                "trace",
                "debug",
            ]
        ):
            return "investigation"

        return "conversational"

    def _check_memory(self, task: str):
        """Check task memory for similar past tasks."""
        self.console.print("[dim]Checking task memory...[/dim]")

        similar = self.loop.memory_manager.recall_similar(task, limit=3)

        if similar:
            self.console.print(f"[cyan]Found {len(similar)} similar past tasks[/cyan]")

            # Get best suggestion
            best = similar[0]
            if best.success:
                self.console.print(
                    f"  [green]✓[/green] Similar task succeeded with {best.iterations} iterations"
                )
