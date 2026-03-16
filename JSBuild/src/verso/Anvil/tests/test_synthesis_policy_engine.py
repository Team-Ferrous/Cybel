from __future__ import annotations

from saguaro.synthesis.policy import SynthesisPolicyEngine


def test_synthesis_policy_engine_disables_expensive_search_in_degraded_mode() -> None:
    decision = SynthesisPolicyEngine().decide(
        {"degraded": True, "scheduler_queue_wait_ms": 40.0}
    )

    assert decision.strategy == "assembler_only"
    assert decision.eqsat_enabled is False
    assert decision.solver_enabled is False

