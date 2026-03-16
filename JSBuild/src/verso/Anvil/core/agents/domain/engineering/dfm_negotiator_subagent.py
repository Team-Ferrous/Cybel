"""DFM Negotiator Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class DFMNegotiatorSubagent(DomainSpecialistSubagent):
    """DFM Negotiator Specialist specialist."""

    system_prompt = """You are Anvil's DFM Negotiator Specialist.

Mission:
- Reconcile design intent with manufacturability and production constraints.

Focus on:
- design for manufacture; cost-yield tradeoffs; tolerance negotiation; production readiness
"""
    tools = DomainSpecialistSubagent.default_tools()
