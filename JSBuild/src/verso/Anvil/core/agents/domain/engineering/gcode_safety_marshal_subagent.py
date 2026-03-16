"""G-code Safety Marshal Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class GCodeSafetyMarshalSubagent(DomainSpecialistSubagent):
    """G-code Safety Marshal Specialist specialist."""

    system_prompt = """You are Anvil's G-code Safety Marshal Specialist.

Mission:
- Review CNC execution plans for safety, repeatability, and quality.

Focus on:
- machine safety envelopes; toolpath sanity checks; fixture assumptions; runtime safeguards
"""
    tools = DomainSpecialistSubagent.default_tools()
