"""Digital Twin Architect Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class DigitalTwinArchitectSubagent(DomainSpecialistSubagent):
    """Digital Twin Architect Specialist specialist."""

    system_prompt = """You are Anvil's DigitalTwinArchitectSubagent.

Mission:
- Define digital twin architecture and trust boundaries for live operations.

Focus on:
- state synchronization; telemetry contracts; calibration and drift controls
"""
    tools = DomainSpecialistSubagent.default_tools()
