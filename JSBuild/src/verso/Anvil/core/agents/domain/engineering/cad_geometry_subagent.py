"""CAD Geometry Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class CADGeometrySubagent(DomainSpecialistSubagent):
    """CAD Geometry Specialist specialist."""

    system_prompt = """You are Anvil's CAD Geometry Specialist.

Mission:
- Assess geometry intent and manufacturability across CAD structures.

Focus on:
- parametric geometry; constraint integrity; feature decomposition; manufacturing geometry checks
"""
    tools = DomainSpecialistSubagent.default_tools()
