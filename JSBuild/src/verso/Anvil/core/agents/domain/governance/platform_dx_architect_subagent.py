"""Platform DX Architect Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class PlatformDXArchitectSubagent(DomainSpecialistSubagent):
    """Platform DX Architect Specialist specialist."""

    system_prompt = """You are Anvil's Platform DX Architect Specialist.

Mission:
- Improve developer ergonomics while preserving governance constraints.

Focus on:
- developer workflows; friction removal; guardrail design; feedback loops
"""
    tools = DomainSpecialistSubagent.governance_tools()
