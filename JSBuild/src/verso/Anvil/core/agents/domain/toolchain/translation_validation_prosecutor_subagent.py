"""Translation Validation Prosecutor Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class TranslationValidationProsecutorSubagent(DomainSpecialistSubagent):
    """Translation Validation Prosecutor Specialist specialist."""

    system_prompt = """You are Anvil's Translation Validation Prosecutor Specialist.

Mission:
- Cross-check compiler transformations with counterexample-oriented scrutiny.

Focus on:
- translation validation; semantic equivalence checks; counterexample triage; proof obligations
"""
    tools = DomainSpecialistSubagent.default_tools()
