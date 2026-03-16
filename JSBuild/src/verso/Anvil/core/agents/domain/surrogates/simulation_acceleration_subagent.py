"""Simulation Acceleration Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SimulationAccelerationSubagent(DomainSpecialistSubagent):
    """Simulation Acceleration Specialist specialist."""

    system_prompt = """You are Anvil's SimulationAccelerationSubagent.

Mission:
- Plan acceleration paths for expensive numerical workflows.

Focus on:
- solver decomposition; parallel scaling strategy; bottleneck instrumentation
"""
    tools = DomainSpecialistSubagent.default_tools()
