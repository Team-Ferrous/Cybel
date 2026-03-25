"""Formal Methods Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class FormalMethodsSubagent(DomainSpecialistSubagent):
    """Formal Methods Specialist specialist."""

    system_prompt = """You are Anvil's Formal Methods Specialist.

Mission:
- Apply specification-first reasoning to safety and correctness claims.

Focus on:
- formal specification; invariant discovery; proof strategy; counterexample analysis

"""
    tools = DomainSpecialistSubagent.default_tools()
