"""Benchmark Engineer Specialist domain specialist subagent."""

from core.agents.domain.base_domain_subagent import DomainSpecialistSubagent


class BenchmarkEngineerSubagent(DomainSpecialistSubagent):
    """Benchmark Engineer Specialist specialist."""

    system_prompt = """You are Anvil's Benchmark Engineer Specialist.

Mission:
- Construct representative benchmark suites and reproducible measurements.

Focus on:
- benchmark design; measurement rigor; noise control; performance baselines
"""
    tools = DomainSpecialistSubagent.default_tools()
