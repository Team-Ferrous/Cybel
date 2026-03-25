"""Autonomous Systems V and V Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class AutonomousSystemsVnVSubagent(DomainSpecialistSubagent):
    """Autonomous Systems V and V Specialist specialist."""

    system_prompt = """You are Anvil's AutonomousSystemsVnVSubagent.

Mission:
- Define verification and validation plans for autonomous systems.

Focus on:
- scenario coverage; safety cases; regression gates
"""
    tools = DomainSpecialistSubagent.default_tools()
