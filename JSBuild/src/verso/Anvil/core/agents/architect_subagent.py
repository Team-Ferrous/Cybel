"""Architecture-focused specialist for DARE."""

from core.agents.subagent import SubAgent


class ArchitectSubagent(SubAgent):
    """Software architecture specialist."""

    system_prompt = """You are Anvil's Master Software Architect.

Focus on:
- system decomposition
- API surface design
- dependency boundaries
- migration and refactor plans
- architecture tradeoffs with evidence
"""

    tools = [
        "saguaro_query",
        "skeleton",
        "slice",
        "impact",
        "read_file",
        "web_search",
        "web_fetch",
    ]
