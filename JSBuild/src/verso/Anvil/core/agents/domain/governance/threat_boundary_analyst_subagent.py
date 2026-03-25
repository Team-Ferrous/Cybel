"""Threat Boundary Analyst Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ThreatBoundaryAnalystSubagent(DomainSpecialistSubagent):
    """Threat Boundary Analyst Specialist specialist."""

    system_prompt = """You are Anvil's Threat Boundary Analyst Specialist.

Mission:
- Model trust zones and attack surfaces across system boundaries.

Focus on:
- threat modeling; trust boundaries; abuse paths; defense priorities
"""
    tools = DomainSpecialistSubagent.governance_tools()
