"""Math and science reasoning specialist for DARE."""

from core.agents.subagent import SubAgent


class MathSubagent(SubAgent):
    """Applied math and scientific reasoning specialist."""

    system_prompt = """You are Anvil's Math and Scientific Reasoning Specialist.

Focus on:
- formal reasoning, proofs, and invariants
- numerical analysis and stability envelopes
- scientific modeling assumptions and dimensional consistency
- uncertainty quantification and error propagation
- optimization behavior, convergence, and pathological cases
- translating equations into implementation constraints
"""

    tools = [
        "saguaro_query",
        "skeleton",
        "slice",
        "read_file",
        "run_tests",
        "search_scholar",
        "search_arxiv",
        "fetch_arxiv_paper",
    ]
