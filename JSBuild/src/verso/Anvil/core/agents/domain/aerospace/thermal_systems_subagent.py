"""Thermal Systems Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ThermalSystemsSubagent(DomainSpecialistSubagent):
    """Thermal Systems Specialist specialist."""

    system_prompt = """You are Anvil's Thermal Systems Specialist.

Mission:
- Develop thermal control strategies and margin assessments.

Focus on:
- thermal modeling; heat-flow management; margin tracking; environmental loads

"""
    tools = DomainSpecialistSubagent.default_tools()
