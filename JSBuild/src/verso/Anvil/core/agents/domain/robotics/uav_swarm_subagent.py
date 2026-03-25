"""UAV Swarm Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class UAVSwarmSubagent(DomainSpecialistSubagent):
    """UAV Swarm Specialist specialist."""

    system_prompt = """You are Anvil's UAVSwarmSubagent.

Mission:
- Plan multi-agent UAV coordination and communication strategies.

Focus on:
- formation logic; decentralized control; communication resilience
"""
    tools = DomainSpecialistSubagent.default_tools()
