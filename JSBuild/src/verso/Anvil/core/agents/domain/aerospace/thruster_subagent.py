"""Thruster Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ThrusterSubagent(DomainSpecialistSubagent):
    """Thruster Specialist specialist."""

    system_prompt = """You are Anvil's Thruster Specialist.

Mission:
- Analyze thruster-level design, controls interfaces, and operational limits.

Focus on:
- thruster configuration; valve/actuation behavior; operating margins; fault sensitivity

"""
    tools = DomainSpecialistSubagent.default_tools()
