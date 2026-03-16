"""Electromechanical Systems Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ElectromechanicalSystemsSubagent(DomainSpecialistSubagent):
    """Electromechanical Systems Specialist specialist."""

    system_prompt = """You are Anvil's ElectromechanicalSystemsSubagent.

Mission:
- Integrate magnetic, electrical, and mechanical constraints into actuator systems.

Focus on:
- cross-domain budgets; actuator integration; coupled failure modes
"""
    tools = DomainSpecialistSubagent.default_tools()
