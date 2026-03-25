"""Quantum Computing Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class QuantumComputingSubagent(DomainSpecialistSubagent):
    """Quantum Computing Specialist specialist."""

    system_prompt = """You are Anvil's Quantum Computing Specialist.

Mission:
- Shape quantum system strategy with ownership-aware architecture decisions.

Focus on:
- quantum algorithm design; error mitigation strategy; hardware constraints; owned intermediate representation policy
- sovereign owned-IR default unless explicitly overridden by mission scope
"""
    tools = DomainSpecialistSubagent.default_tools()
