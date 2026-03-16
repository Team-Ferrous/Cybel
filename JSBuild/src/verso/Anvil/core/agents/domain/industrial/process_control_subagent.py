"""Process Control Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ProcessControlSubagent(DomainSpecialistSubagent):
    """Process Control Specialist specialist."""

    system_prompt = """You are Anvil's ProcessControlSubagent.

Mission:
- Design process-control loops and control law deployment constraints.

Focus on:
- pid and mpc selection; stability margins; operator safeguards
"""
    tools = DomainSpecialistSubagent.default_tools()
