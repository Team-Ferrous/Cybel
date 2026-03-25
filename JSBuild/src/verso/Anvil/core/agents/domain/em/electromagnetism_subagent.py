"""Electromagnetism Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ElectromagnetismSubagent(DomainSpecialistSubagent):
    """Electromagnetism Specialist specialist."""

    system_prompt = """You are Anvil's ElectromagnetismSubagent.

Mission:
- Model electromagnetic fields, interference, and shielding decisions.

Focus on:
- maxwell constraints; boundary conditions; emc and emi risk controls
"""
    tools = DomainSpecialistSubagent.default_tools()
