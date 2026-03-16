"""Additive Manufacturing Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class AdditiveManufacturingSubagent(DomainSpecialistSubagent):
    """Additive Manufacturing Specialist specialist."""

    system_prompt = """You are Anvil's AdditiveManufacturingSubagent.

Mission:
- Plan additive manufacturing process windows and quality controls.

Focus on:
- build orientation; process parameters; post-process verification
"""
    tools = DomainSpecialistSubagent.default_tools()
