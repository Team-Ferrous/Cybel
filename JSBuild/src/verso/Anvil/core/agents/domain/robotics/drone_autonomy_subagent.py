"""Drone Autonomy Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class DroneAutonomySubagent(DomainSpecialistSubagent):
    """Drone Autonomy Specialist specialist."""

    system_prompt = """You are Anvil's DroneAutonomySubagent.

Mission:
- Design UAV autonomy policies and simulation-to-reality workflows.

Focus on:
- autonomy stack decomposition; rl transfer strategy; flight envelope constraints
"""
    tools = DomainSpecialistSubagent.default_tools()
