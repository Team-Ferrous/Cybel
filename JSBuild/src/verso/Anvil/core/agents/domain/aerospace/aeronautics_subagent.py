"""Aeronautics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class AeronauticsSubagent(DomainSpecialistSubagent):
    """Aeronautics Specialist specialist."""

    system_prompt = """You are Anvil's Aeronautics Specialist.

Mission:
- Assess aerodynamic configuration and performance envelope decisions.

Focus on:
- aerodynamic performance; stability implications; configuration tradeoffs; flight regime limits

"""
    tools = DomainSpecialistSubagent.default_tools()
