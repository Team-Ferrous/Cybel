"""ML research specialist for DARE."""

from core.agents.subagent import SubAgent


class MLResearchSubagent(SubAgent):
    """Deep learning and model-layer analysis specialist."""

    system_prompt = """You are Anvil's ML Research Scientist.

Focus on:
- layer implementations
- numerical stability
- model architecture research
- quantization and inference tradeoffs
- experiment design for ML claims
"""

    tools = [
        "saguaro_query",
        "skeleton",
        "slice",
        "read_file",
        "run_tests",
        "web_search",
        "web_fetch",
        "search_arxiv",
        "search_scholar",
        "fetch_arxiv_paper",
    ]
