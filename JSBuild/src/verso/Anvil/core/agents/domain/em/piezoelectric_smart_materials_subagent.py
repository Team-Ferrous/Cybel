"""Piezoelectric and Smart Materials Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class PiezoelectricAndSmartMaterialsSubagent(DomainSpecialistSubagent):
    """Piezoelectric and Smart Materials Specialist specialist."""

    system_prompt = """You are Anvil's PiezoelectricAndSmartMaterialsSubagent.

Mission:
- Integrate piezoelectric and smart-material behavior into system design.

Focus on:
- electromechanical coupling; vibration control; sensing and actuation tradeoffs
"""
    tools = DomainSpecialistSubagent.default_tools()
