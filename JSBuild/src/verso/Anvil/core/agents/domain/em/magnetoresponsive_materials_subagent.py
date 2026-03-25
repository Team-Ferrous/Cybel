"""Magnetoresponsive Materials Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MagnetoresponsiveMaterialsSubagent(DomainSpecialistSubagent):
    """Magnetoresponsive Materials Specialist specialist."""

    system_prompt = """You are Anvil's MagnetoresponsiveMaterialsSubagent.

Mission:
- Guide design of smart magnetic materials and inverse magnetization targets.

Focus on:
- material property envelopes; inverse design goals; manufacturability risks
"""
    tools = DomainSpecialistSubagent.default_tools()
