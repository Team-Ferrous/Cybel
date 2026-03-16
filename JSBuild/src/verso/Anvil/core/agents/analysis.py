from typing import Any, Dict
from core.agent import BaseAgent
from core.analysis.semantic import SemanticEngine


class AnalysisAgent(BaseAgent):
    """
    Specialized agent for analyzing codebase, tracing dependencies, and understanding logic.
    Supports both log-based debugging and general research.
    """

    def __init__(self, semantic_engine: SemanticEngine, **kwargs):
        super().__init__(name="Analyst", **kwargs)
        self.semantic_engine = semantic_engine
        self.system_prompt_prefix = """
You are a Senior Software Architect and Analysts. Your goal is to deeply understand the codebase, identify patterns, and trace dependencies.

### GUIDELINES
1. **DEEP ANALYSIS**: When researching a topic, look for entry points, core logic, and edge cases.
2. **TOOL USAGE**: Use semantic tools (skeleton, slice, impact) for high-level structure. Use standard tools (grep, list_dir, read_file) for precise evidence gathering.
3. **LOGICAL TRACING**: Trace data flow and control flow between components.
4. **SYNTHESIS**: Connect the dots between different files to form a coherent mental model of the feature or bug.

### OUTPUT FORMAT
You MUST output your findings in a structured way. If you are doing a multi-step loop, provide updates on what you've discovered.
For a final report, use:
---
overview: "High-level summary of the findings"
key_files: ["file1.py", "file2.py"]
logic_flow: "Step-by-step description of how the code works"
dependencies: ["component A", "component B"]
recommendations: "Actionable steps for the next phase"
---
"""

    def analyze_topic(self, topic: str) -> Dict[str, Any]:
        """Performs a targeted research cycle on a specific topic."""
        self.console.print(
            f"[bold cyan]Analyst: Starting deep research on: {topic}[/bold cyan]"
        )

        # This agent typically runs within its own loop (AnalysisLoop)
        # But it can also be used for one-shot analysis if needed.
        response = self.run_loop(
            f"Perform a deep analysis of the following topic in the codebase: {topic}"
        )
        return {
            "analysis": response.get("response", ""),
            "stats": response.get("stats", {}),
        }

    def analyze_log(self, log_content: str) -> Dict[str, Any]:
        """Analyzes logs and identifies root causes."""
        self.console.print(
            "[dim]Analyst: Analyzing error logs and tracing stack trace...[/dim]"
        )

        prompt = f"""
Analyze the following error log. 
1. Identify the primary failure point.
2. List the files and functions in the stack trace.
3. Use Saguaro tools to check for the impact of these components.

Log:
{log_content}
"""
        response = self.run_loop(prompt)
        return {
            "analysis": response.get("response", ""),
            "stats": response.get("stats", {}),
        }
