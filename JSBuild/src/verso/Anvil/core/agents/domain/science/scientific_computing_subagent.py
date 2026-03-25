"""Scientific Computing Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ScientificComputingSubagent(DomainSpecialistSubagent):
    """Scientific Computing Specialist specialist."""

    system_prompt = """You are Anvil's Scientific Computing Specialist.

Mission:
- Design numerically sound computational methods and validation plans.

Focus on:
- numerical stability; algorithm selection; error bounds; reproducibility

"""
    tools = DomainSpecialistSubagent.default_tools()
