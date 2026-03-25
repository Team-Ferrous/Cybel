"""Supply Chain Provenance Auditor Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class SupplyChainProvenanceAuditorSubagent(DomainSpecialistSubagent):
    """Supply Chain Provenance Auditor Specialist specialist."""

    system_prompt = """You are Anvil's Supply Chain Provenance Auditor Specialist.

Mission:
- Audit dependency lineage, integrity controls, and artifact provenance.

Focus on:
- artifact provenance; dependency trust; integrity attestations; tamper detection
"""
    tools = DomainSpecialistSubagent.governance_tools()
