"""Electric Propulsion Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ElectricPropulsionSubagent(DomainSpecialistSubagent):
    """Electric Propulsion Specialist specialist."""

    system_prompt = """You are Anvil's ElectricPropulsionSubagent.

Mission:
- Design electric thruster concepts with integrated plasma and power assumptions.

Focus on:
- hall and ion thruster sizing; plume and lifetime risks; ppu constraints
"""
    tools = DomainSpecialistSubagent.default_tools()
