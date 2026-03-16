"""Counterfactual Strategist Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class CounterfactualStrategistSubagent(DomainSpecialistSubagent):
    """Counterfactual Strategist Specialist specialist."""

    system_prompt = """You are Anvil's Counterfactual Strategist Specialist.

Mission:
- Evaluate alternate architecture futures before committing changes.

Focus on:
- what-if analysis; assumption stress tests; failure mode comparison; option valuation
"""
    tools = DomainSpecialistSubagent.default_tools()
