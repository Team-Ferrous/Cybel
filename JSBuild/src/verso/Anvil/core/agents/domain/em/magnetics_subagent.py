"""Magnetics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MagneticsSubagent(DomainSpecialistSubagent):
    """Magnetics Specialist specialist."""

    system_prompt = """You are Anvil's MagneticsSubagent.

Mission:
- Design magnet circuits, materials, and actuator magnetic paths.

Focus on:
- flux density targets; saturation limits; coil and core trade studies
"""
    tools = DomainSpecialistSubagent.default_tools()
