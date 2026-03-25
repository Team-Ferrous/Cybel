"""Binary Layout Surgeon Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class BinaryLayoutSurgeonSubagent(DomainSpecialistSubagent):
    """Binary Layout Surgeon Specialist specialist."""

    system_prompt = """You are Anvil's Binary Layout Surgeon Specialist.

Mission:
- Optimize binary layout for startup, locality, and runtime determinism.

Focus on:
- binary section layout; symbol locality; startup path optimization; layout determinism
"""
    tools = DomainSpecialistSubagent.default_tools()
