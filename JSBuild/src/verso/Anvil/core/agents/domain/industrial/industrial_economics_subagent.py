"""Industrial Economics Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class IndustrialEconomicsSubagent(DomainSpecialistSubagent):
    """Industrial Economics Specialist specialist."""

    system_prompt = """You are Anvil's IndustrialEconomicsSubagent.

Mission:
- Evaluate industrial cost and risk tradeoffs for engineering programs.

Focus on:
- cost models; capex and opex tradeoffs; rollout risk assumptions
"""
    tools = DomainSpecialistSubagent.default_tools()
