"""Heat Transfer Surrogate Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class HeatTransferSurrogateSubagent(DomainSpecialistSubagent):
    """Heat Transfer Surrogate Specialist specialist."""

    system_prompt = """You are Anvil's HeatTransferSurrogateSubagent.

Mission:
- Model thermal behavior with surrogate methods while preserving physical constraints.

Focus on:
- thermal boundary assumptions; transient approximations; error envelopes
"""
    tools = DomainSpecialistSubagent.default_tools()
