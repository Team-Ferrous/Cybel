"""CPU optimization specialist for DARE."""

from core.agents.subagent import SubAgent


class HPCSubagent(SubAgent):
    """Legacy alias for CPU optimization specialist behavior."""

    system_prompt = """You are Anvil's CPUOptimizerSubagent.

Focus on:
- ISA-aware optimization strategy (SIMD/AVX2/AVX-512/NEON)
- threading, affinity, and cache-aware memory layout
- ABI and determinism risk checks for low-level changes
- correctness-preserving optimization plans

Benchmark evidence policy:
- Never claim speedups without before/after measurements.
- Require benchmark methodology, dataset/fixture, and run conditions.
- Include variance notes and regression risks before recommending rollout.
"""

    tools = [
        "saguaro_query",
        "skeleton",
        "slice",
        "read_file",
        "verify",
        "run_tests",
        "search_hackernews",
        "search_stackoverflow",
    ]


class CPUOptimizerSubagent(HPCSubagent):
    """Modern role alias for CPU optimization specialist behavior."""
