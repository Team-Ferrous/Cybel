from typing import Dict, List, Any, Optional
from domains.verification.coherence_engine import CoherenceEngine


class AutoCorrector:
    """
    Triggers corrections when mathematical coherence falls below threshold.
    Corrects hallucinations and multi-agent drift.
    """

    def __init__(self, agent_loop):
        self.engine = CoherenceEngine()
        self.agent_loop = agent_loop
        self.max_attempts = 3

    async def verify_and_correct(
        self,
        reasoning_trace: List[str],
        interaction_history: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Validates the trace and triggers a correction if necessary.
        """
        results = self.engine.validate_trace(reasoning_trace, interaction_history)

        if results["passed"]:
            return {"status": "success", "results": results}

        # If we failed, generate a correction strategy
        strategy = self._generate_correction_strategy(results)

        return {
            "status": "correction_required",
            "results": results,
            "correction_prompt": strategy,
        }

    def _generate_correction_strategy(self, results: Dict[str, Any]) -> str:
        """
        Generates a specific prompt based on which mathematical check failed.
        """
        critique = []
        if results["sheaf_score"] < 0.7:
            critique.append(
                "- [Contradiction Detected]: Your reasoning steps are diverging. Re-evaluate your last 3 steps for consistency."
            )
        if results["spectral_stability"] < 0.7:
            critique.append(
                "- [Agent Drift]: Multi-agent state is becoming unstable. Focus on the core objective and reduce complexity."
            )
        if results["causal_validity"] < 0.7:
            critique.append(
                "- [Causal Gap]: You are proposing actions without sufficient justification. Explain the expected outcome of each intervention."
            )

        prompt = "CRITICAL COHERENCE VIOLATION DETECTED.\n"
        prompt += "\n".join(critique)
        prompt += (
            "\n\nPlease resolve these issues and provide a corrected thought process."
        )

        return prompt
