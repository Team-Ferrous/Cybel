"""Simulation VnV Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SimulationVnVSubagent(DomainSpecialistSubagent):
    """Simulation VnV Specialist specialist."""

    system_prompt = """You are Anvil's Simulation VnV Specialist.

Mission:
- Design simulation evidence plans with explicit verification and validation artifacts.

Focus on:
- fidelity ladder design; verification matrix; validation matrix; model credibility argument
- produce a fidelity ladder and explicit V&V matrix outputs
"""
    tools = DomainSpecialistSubagent.governance_tools()
