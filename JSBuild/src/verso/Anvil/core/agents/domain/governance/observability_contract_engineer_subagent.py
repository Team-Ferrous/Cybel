"""Observability Contract Engineer Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ObservabilityContractEngineerSubagent(DomainSpecialistSubagent):
    """Observability Contract Engineer Specialist specialist."""

    system_prompt = """You are Anvil's Observability Contract Engineer Specialist.

Mission:
- Specify telemetry contracts required for reliable operations.

Focus on:
- signal contracts; SLO observability; event schemas; diagnostic coverage
"""
    tools = DomainSpecialistSubagent.governance_tools()
