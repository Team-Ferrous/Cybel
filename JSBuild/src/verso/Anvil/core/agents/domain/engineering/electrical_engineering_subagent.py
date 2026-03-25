"""Electrical Engineering Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ElectricalEngineeringSubagent(DomainSpecialistSubagent):
    """Electrical Engineering Specialist specialist."""

    system_prompt = """You are Anvil's Electrical Engineering Specialist.

Mission:
- Develop electrical design guidance from requirements to validation.

Focus on:
- power and signal integrity; interface definition; protection strategy; validation planning
"""
    tools = DomainSpecialistSubagent.default_tools()
