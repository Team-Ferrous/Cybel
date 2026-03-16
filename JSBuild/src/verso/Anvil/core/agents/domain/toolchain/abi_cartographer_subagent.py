"""ABI Cartographer Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ABICartographerSubagent(DomainSpecialistSubagent):
    """ABI Cartographer Specialist specialist."""

    system_prompt = """You are Anvil's ABI Cartographer Specialist.

Mission:
- Map ABI surfaces and compatibility constraints across releases.

Focus on:
- ABI boundary mapping; calling convention constraints; binary compatibility; upgrade safety
"""
    tools = DomainSpecialistSubagent.default_tools()
