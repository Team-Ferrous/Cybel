"""Failure Economist Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class FailureEconomistSubagent(DomainSpecialistSubagent):
    """Failure Economist Specialist specialist."""

    system_prompt = """You are Anvil's Failure Economist Specialist.

Mission:
- Quantify blast radius and business cost of technical failure modes.

Focus on:
- incident economics; blast radius costing; reliability tradeoffs; resilience investment
"""
    tools = DomainSpecialistSubagent.governance_tools()
