"""Competitive intelligence specialist for DARE."""

from core.agents.subagent import SubAgent


class CompetitorSubagent(SubAgent):
    """Competitive intelligence and ecosystem scanning specialist."""

    system_prompt = """You are Anvil's Competitive Intelligence Analyst.

Focus on:
- competing repositories
- market gaps
- user pain points from forums
- docs, ecosystem health, and trend signals
"""

    tools = [
        "web_search",
        "web_fetch",
        "search_arxiv",
        "fetch_arxiv_paper",
        "search_reddit",
        "search_hackernews",
        "search_stackoverflow",
    ]
