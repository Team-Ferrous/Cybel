"""Industrial IoT Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class IndustrialIoTSubagent(DomainSpecialistSubagent):
    """Industrial IoT Specialist specialist."""

    system_prompt = """You are Anvil's IndustrialIoTSubagent.

Mission:
- Design industrial telemetry and edge data pipelines.

Focus on:
- sensor instrumentation; edge to cloud flow; reliability constraints
"""
    tools = DomainSpecialistSubagent.default_tools()
