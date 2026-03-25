"""EM Surrogate Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class EMSurrogateSubagent(DomainSpecialistSubagent):
    """EM Surrogate Specialist specialist."""

    system_prompt = """You are Anvil's EMSurrogateSubagent.

Mission:
- Create surrogate approaches for electromagnetic simulation acceleration.

Focus on:
- field emulation bounds; architecture choices; validation targets
"""
    tools = DomainSpecialistSubagent.default_tools()
