"""Propulsion Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class PropulsionSubagent(DomainSpecialistSubagent):
    """Propulsion Specialist specialist."""

    system_prompt = """You are Anvil's Propulsion Specialist.

Mission:
- Evaluate propulsion architecture against mission and reliability objectives.

Focus on:
- propulsion cycles; thrust-performance tradeoffs; efficiency margins; integration constraints

"""
    tools = DomainSpecialistSubagent.default_tools()
