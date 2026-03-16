"""Compatibility Migration Architect Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class CompatibilityMigrationArchitectSubagent(DomainSpecialistSubagent):
    """Compatibility Migration Architect Specialist specialist."""

    system_prompt = """You are Anvil's Compatibility Migration Architect Specialist.

Mission:
- Plan backward-compatible migrations with explicit deprecation strategy.

Focus on:
- compatibility strategy; migration choreography; version policy; deprecation safety
"""
    tools = DomainSpecialistSubagent.governance_tools()
