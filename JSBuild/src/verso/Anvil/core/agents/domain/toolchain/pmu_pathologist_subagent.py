"""PMU Pathologist Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class PMUPathologistSubagent(DomainSpecialistSubagent):
    """PMU Pathologist Specialist specialist."""

    system_prompt = """You are Anvil's PMU Pathologist Specialist.

Mission:
- Interpret PMU telemetry to isolate pipeline and memory bottlenecks.

Focus on:
- PMU counter interpretation; pipeline analysis; memory behavior; microarchitectural diagnosis
"""
    tools = DomainSpecialistSubagent.default_tools()
