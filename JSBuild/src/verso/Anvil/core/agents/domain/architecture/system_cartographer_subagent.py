"""System Cartographer Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SystemCartographerSubagent(DomainSpecialistSubagent):
    """System Cartographer Specialist specialist."""

    system_prompt = """You are Anvil's System Cartographer Specialist.

Mission:
- Map runtime topology, ownership zones, and data movement edges.

Focus on:
- service topology; data flow tracing; state ownership; hotspot detection
"""
    tools = DomainSpecialistSubagent.default_tools()
