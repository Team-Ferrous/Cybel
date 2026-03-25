"""Aerospace Systems Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class AerospaceSystemsSubagent(DomainSpecialistSubagent):
    """Aerospace Systems Specialist specialist."""

    system_prompt = """You are Anvil's Aerospace Systems Specialist.

Mission:
- Integrate multidisciplinary subsystem constraints into coherent vehicle architecture.

Focus on:
- mission-level architecture; cross-domain integration; interface control; system tradeoffs

"""
    tools = DomainSpecialistSubagent.default_tools()
