"""Sensor Fusion Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SensorFusionSubagent(DomainSpecialistSubagent):
    """Sensor Fusion Specialist specialist."""

    system_prompt = """You are Anvil's SensorFusionSubagent.

Mission:
- Design multisensor estimation pipelines for robust autonomy.

Focus on:
- state estimation assumptions; observability gaps; fault handling
"""
    tools = DomainSpecialistSubagent.default_tools()
