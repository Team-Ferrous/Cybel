"""Fluids Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class FluidsSubagent(DomainSpecialistSubagent):
    """Fluids Specialist specialist."""

    system_prompt = """You are Anvil's Fluids Specialist.

Mission:
- Model fluid network behavior and pressure/thermal interactions.

Focus on:
- fluid dynamics; pressure losses; transient behavior; network robustness

"""
    tools = DomainSpecialistSubagent.default_tools()
