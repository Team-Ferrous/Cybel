"""MDO Trade Study Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class MDOTradeStudySubagent(DomainSpecialistSubagent):
    """MDO Trade Study Specialist specialist."""

    system_prompt = """You are Anvil's MDO Trade Study Specialist.

Mission:
- Run multidisciplinary trade studies with transparent assumptions and outcomes.

Focus on:
- multidisciplinary optimization; objective balancing; constraint sensitivity; design-space exploration

"""
    tools = DomainSpecialistSubagent.default_tools()
