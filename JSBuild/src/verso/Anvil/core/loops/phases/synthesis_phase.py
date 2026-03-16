from typing import Any, Dict, Optional
import re
from core.loops.phases.base_phase import BasePhase
from core.response_utils import clean_response
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SynthesisPhase(BasePhase):
    """
    Phase 4: Synthesis - Synthesize final response to user.
    """

    def execute(
        self, user_input: str, context: Dict[str, Any], dashboard: Optional[Any] = None
    ) -> str:
        logger.info("Executing Synthesis Phase")

        if dashboard:
            dashboard.update_phase(
                "Synthesis", status="in_progress", message="Summarizing results..."
            )
        else:
            self.console.print("[cyan]Phase 4: Synthesizing response...[/cyan]")

        # Select synthesis strategy based on request type and previous phase results
        request_type = context.get("request_type")
        evidence = context.get("evidence", {})
        execution_result = context.get("execution_result", {})

        if request_type in ["question", "explanation", "investigation"]:
            response = self.loop._synthesize_answer(
                user_input, evidence, dashboard=dashboard
            )
        elif request_type in ["modification", "creation", "deletion"]:
            action_plan = execution_result.get("action_plan", "")
            response = self.loop._synthesize_action_result(
                user_input, action_plan, execution_result, dashboard=dashboard
            )
        else:
            response = self.loop._handle_conversational(user_input, dashboard=dashboard)

        if dashboard:
            dashboard.update_phase("Synthesis", status="completed")

        response = clean_response(response)
        if self._looks_like_tool_artifact(response):
            logger.warning(
                "Synthesis output resembled tool-call artifact; switching to deterministic grounded synthesis"
            )
            deterministic = ""
            synth_small = getattr(self.loop, "_synthesize_for_small_model", None)
            if callable(synth_small):
                deterministic = synth_small(user_input=user_input, evidence=evidence) or ""
            response = deterministic or (
                "I gathered evidence for your request, but the synthesis stream produced malformed output. "
                "Please retry with the same query."
            )

        # Verify grounding (accuracy check)
        if evidence and isinstance(evidence, dict):
            response = self._verify_response_grounding(response, evidence)

        return response

    def _looks_like_tool_artifact(self, response: str) -> bool:
        if not response:
            return False
        stripped = response.strip()
        if "<tool_call>" in stripped or "Executing Tool:" in stripped:
            return True
        if "[SYSTEM: Streaming loop terminated.]" in stripped:
            return True
        return bool(
            re.fullmatch(
                r'\s*\{\s*"name"\s*:\s*"[a-zA-Z0-9_]+"\s*,\s*"arguments"\s*:\s*\{.*\}\s*\}\s*',
                stripped,
                flags=re.DOTALL,
            )
        )

    def _verify_response_grounding(self, response: str, evidence: dict) -> str:
        """Verify all code references in response exist in evidence."""
        import re

        # Extract mentioned class/method names
        code_refs = re.findall(r"`(\w+(?:\.\w+)*)`", response)
        class_refs = re.findall(r"class\s+(\w+)", response)
        method_refs = re.findall(r"def\s+(\w+)", response)

        all_refs = set(code_refs + class_refs + method_refs)

        # Check against actual file contents
        evidence_text = str(evidence.get("file_contents", {}))

        ungrounded = []
        for ref in all_refs:
            # Filter standard types/keywords
            if ref.lower() in [
                "str",
                "int",
                "bool",
                "dict",
                "list",
                "optional",
                "any",
                "none",
                "true",
                "false",
                "exception",
                "return",
                "import",
            ]:
                continue

            if ref not in evidence_text:
                ungrounded.append(ref)

        if ungrounded:
            # Flag ungrounded claims
            unique_ungrounded = sorted(list(set(ungrounded)))
            response += f"\n\n> **Note:** Some references could not be verified in the loaded context: {', '.join(unique_ungrounded[:5])}"

        return response
