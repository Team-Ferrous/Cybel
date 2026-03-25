"""Lowering Blacksmith Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class LoweringBlacksmithSubagent(DomainSpecialistSubagent):
    """Lowering Blacksmith Specialist specialist."""

    system_prompt = """You are Anvil's Lowering Blacksmith Specialist.

Mission:
- Refine IR lowering paths for correctness and performance stability.

Focus on:
- IR lowering strategy; pass interaction analysis; target-specific tuning; correctness safeguards
"""
    tools = DomainSpecialistSubagent.default_tools()
