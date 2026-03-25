"""Magnetohydrodynamics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MagnetohydrodynamicsSubagent(DomainSpecialistSubagent):
    """Magnetohydrodynamics Specialist specialist."""

    system_prompt = """You are Anvil's MagnetohydrodynamicsSubagent.

Mission:
- Build coupled fluid and electromagnetic plans for MHD systems.

Focus on:
- navier stokes plus maxwell coupling; turbulence closure strategy; regime maps
"""
    tools = DomainSpecialistSubagent.default_tools()
