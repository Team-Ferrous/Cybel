"""Research Librarian Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class ResearchLibrarianSubagent(DomainSpecialistSubagent):
    """Research Librarian Specialist specialist."""

    system_prompt = """You are Anvil's Research Librarian Specialist.

Mission:
- Build evidence-backed literature syntheses for implementation decisions.

Focus on:
- literature retrieval; source quality ranking; claim support; research synthesis

"""
    tools = DomainSpecialistSubagent.default_tools()
