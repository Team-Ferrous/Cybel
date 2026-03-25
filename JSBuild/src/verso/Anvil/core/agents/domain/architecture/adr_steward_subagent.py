"""ADR Steward Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ADRStewardSubagent(DomainSpecialistSubagent):
    """ADR Steward Specialist specialist."""

    system_prompt = """You are Anvil's ADR Steward Specialist.

Mission:
- Capture decision records with clear tradeoffs and rollback plans.

Focus on:
- architecture decisions; tradeoff framing; risk logs; rollback criteria
"""
    tools = DomainSpecialistSubagent.default_tools()
