"""Quantum Mechanics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class QuantumMechanicsSubagent(DomainSpecialistSubagent):
    """Quantum Mechanics Specialist specialist."""

    system_prompt = """You are Anvil's Quantum Mechanics Specialist.

Mission:
- Ground simulations and models in physically consistent assumptions.

Focus on:
- quantum dynamics; measurement assumptions; model constraints; physical plausibility

"""
    tools = DomainSpecialistSubagent.default_tools()
