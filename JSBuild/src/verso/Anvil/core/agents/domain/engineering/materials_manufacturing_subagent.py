"""Materials and Manufacturing Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MaterialsManufacturingSubagent(DomainSpecialistSubagent):
    """Materials and Manufacturing Specialist specialist."""

    system_prompt = """You are Anvil's Materials and Manufacturing Specialist.

Mission:
- Select materials and process windows based on performance and yield.

Focus on:
- material selection; process capability; durability tradeoffs; manufacturing yield
"""
    tools = DomainSpecialistSubagent.default_tools()
