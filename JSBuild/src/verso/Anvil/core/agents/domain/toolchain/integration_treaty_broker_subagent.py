"""Integration Treaty Broker Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class IntegrationTreatyBrokerSubagent(DomainSpecialistSubagent):
    """Integration Treaty Broker Specialist specialist."""

    system_prompt = """You are Anvil's Integration Treaty Broker Specialist.

Mission:
- Broker stable integration contracts between toolchain components.

Focus on:
- integration boundary design; contract negotiation; cross-team compatibility; release coordination
"""
    tools = DomainSpecialistSubagent.default_tools()
