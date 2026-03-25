"""FinOps Model Economist Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class FinOpsModelEconomistSubagent(DomainSpecialistSubagent):
    """FinOps Model Economist Specialist specialist."""

    system_prompt = """You are Anvil's FinOps Model Economist Specialist.

Mission:
- Optimize model and infrastructure spend with measurable performance outcomes.

Focus on:
- cost-performance tuning; capacity economics; efficiency targets; budget guardrails
"""
    tools = DomainSpecialistSubagent.governance_tools()
