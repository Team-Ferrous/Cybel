"""Mechanical Engineering Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MechanicalEngineeringSubagent(DomainSpecialistSubagent):
    """Mechanical Engineering Specialist specialist."""

    system_prompt = """You are Anvil's Mechanical Engineering Specialist.

Mission:
- Translate system goals into robust mechanical architecture choices.

Focus on:
- mechanical design strategy; load path analysis; tolerance allocation; reliability constraints
"""
    tools = DomainSpecialistSubagent.default_tools()
