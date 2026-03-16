from typing import Any


def analyze_codebase_tool(topic: str, agent_repl: Any = None) -> str:
    """
    Bridge function to trigger an autonomous code analysis subagent.
    """
    if not agent_repl:
        return "Error: Agent REPL instance required for code analysis."

    from core.loops.analysis_loop import AnalysisLoop

    loop = AnalysisLoop(agent_repl)
    result = loop.run(topic)
    return result
