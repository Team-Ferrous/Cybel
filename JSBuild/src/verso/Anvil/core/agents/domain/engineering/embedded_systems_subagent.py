"""Embedded Systems Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class EmbeddedSystemsSubagent(DomainSpecialistSubagent):
    """Embedded Systems Specialist specialist."""

    system_prompt = """You are Anvil's Embedded Systems Specialist.

Mission:
- Coordinate firmware-hardware interfaces with deterministic behavior goals.

Focus on:
- real-time constraints; firmware interface design; resource budgeting; embedded validation
"""
    tools = DomainSpecialistSubagent.default_tools()
