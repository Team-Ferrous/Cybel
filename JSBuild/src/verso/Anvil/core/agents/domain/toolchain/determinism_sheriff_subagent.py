"""Determinism Sheriff Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class DeterminismSheriffSubagent(DomainSpecialistSubagent):
    """Determinism Sheriff Specialist specialist."""

    system_prompt = """You are Anvil's Determinism Sheriff Specialist.

Mission:
- Enforce deterministic build and runtime behavior under varied environments.

Focus on:
- deterministic execution; build reproducibility; nondeterminism tracing; stability controls
"""
    tools = DomainSpecialistSubagent.default_tools()
