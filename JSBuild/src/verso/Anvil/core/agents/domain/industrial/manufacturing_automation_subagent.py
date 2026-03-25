"""Manufacturing Automation Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ManufacturingAutomationSubagent(DomainSpecialistSubagent):
    """Manufacturing Automation Specialist specialist."""

    system_prompt = """You are Anvil's ManufacturingAutomationSubagent.

Mission:
- Design automation flows for manufacturing operations.

Focus on:
- workcell orchestration; plc and scada constraints; throughput and safety
"""
    tools = DomainSpecialistSubagent.default_tools()
