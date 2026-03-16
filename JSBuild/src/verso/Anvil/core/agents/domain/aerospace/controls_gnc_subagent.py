"""Controls and GNC Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ControlsGNCSubagent(DomainSpecialistSubagent):
    """Controls and GNC Specialist specialist."""

    system_prompt = """You are Anvil's Controls and GNC Specialist.

Mission:
- Define guidance-navigation-control architecture and robustness checks.

Focus on:
- guidance strategy; navigation observability; control stability; robustness margins

"""
    tools = DomainSpecialistSubagent.default_tools()
