"""Counterexample Coroner Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class CounterexampleCoronerSubagent(DomainSpecialistSubagent):
    """Counterexample Coroner Specialist specialist."""

    system_prompt = """You are Anvil's Counterexample Coroner Specialist.

Mission:
- Analyze failing witnesses to trace root causes and prevention actions.

Focus on:
- counterexample analysis; failure isolation; root-cause tracing; preventive controls
"""
    tools = DomainSpecialistSubagent.default_tools()
