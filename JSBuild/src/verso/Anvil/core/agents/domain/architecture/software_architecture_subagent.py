"""Software Architecture Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SoftwareArchitectureSubagent(DomainSpecialistSubagent):
    """Software Architecture Specialist specialist."""

    system_prompt = """You are Anvil's Software Architecture Specialist.

Mission:
- Design durable module boundaries and integration contracts.

Focus on:
- system decomposition; interface contracts; dependency boundaries; migration sequencing
"""
    tools = DomainSpecialistSubagent.default_tools()
