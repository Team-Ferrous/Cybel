from core.memory.fabric import MemoryTierPolicy


def test_memory_tier_compiler_prefers_repo_delta_for_incompatible_replay() -> None:
    decision = MemoryTierPolicy().choose(
        purpose="mission_replay",
        runtime_status={"qsg_queue_wait_ms_p95": 4.0},
        latent_package={"latent_package_id": "latent-1"},
        compatibility={"compatible": False, "mismatches": ["capability_digest"]},
        repo_delta_memory={
            "delta_id": "delta-1",
            "changed_paths": ["core/qsg/grover.py"],
        },
    )

    assert decision.selected_tier == "repo_delta_memory"
    assert decision.replay_allowed is False
    assert decision.reason == "delta_scoped_recovery_required"
