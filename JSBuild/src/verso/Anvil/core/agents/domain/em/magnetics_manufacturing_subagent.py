"""Magnetics Manufacturing Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MagneticsManufacturingSubagent(DomainSpecialistSubagent):
    """Magnetics Manufacturing Specialist specialist."""

    system_prompt = """You are Anvil's MagneticsManufacturingSubagent.

Mission:
- Plan manufacturable winding, material processing, and quality controls for magnetics.

Focus on:
- winding geometry; thermal processing; supplier and qc gates
"""
    tools = DomainSpecialistSubagent.default_tools()
