"""Fitness Function Engineer Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class FitnessFunctionEngineerSubagent(DomainSpecialistSubagent):
    """Fitness Function Engineer Specialist specialist."""

    system_prompt = """You are Anvil's Fitness Function Engineer Specialist.

Mission:
- Define measurable quality gates aligned to product intent.

Focus on:
- objective functions; quality metrics; acceptance thresholds; optimization tradeoffs
"""
    tools = DomainSpecialistSubagent.governance_tools()
