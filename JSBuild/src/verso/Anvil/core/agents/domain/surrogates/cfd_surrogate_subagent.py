"""CFD Surrogate Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class CFDSurrogateSubagent(DomainSpecialistSubagent):
    """CFD Surrogate Specialist specialist."""

    system_prompt = """You are Anvil's CFDSurrogateSubagent.

Mission:
- Design physics-aware CFD surrogate workflows when full simulation is too costly.

Focus on:
- data generation strategy; fidelity constraints; uncertainty reporting
"""
    tools = DomainSpecialistSubagent.default_tools()
