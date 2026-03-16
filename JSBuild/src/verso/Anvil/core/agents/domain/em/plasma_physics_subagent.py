"""Plasma Physics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class PlasmaPhysicsSubagent(DomainSpecialistSubagent):
    """Plasma Physics Specialist specialist."""

    system_prompt = """You are Anvil's PlasmaPhysicsSubagent.

Mission:
- Model plasma generation, confinement, and diagnostics for engineering systems.

Focus on:
- plasma regime selection; diagnostics; power and stability envelopes
"""
    tools = DomainSpecialistSubagent.default_tools()
