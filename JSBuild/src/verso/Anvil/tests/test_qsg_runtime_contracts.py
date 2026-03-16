from core.qsg.runtime_contracts import (
    DeltaWatermark,
    PerformanceTwinModel,
    PerformanceEnvelope,
    RuntimeCapabilityVector,
    evaluate_qsg_runtime_invariants,
)
from core.memory.fabric import MemoryTierPolicy, RepoDeltaMemoryRecord


def test_runtime_capability_vector_builds_digest_and_optional_leaves() -> None:
    vector = RuntimeCapabilityVector.from_status(
        {
            "model": "granite",
            "digest": "abc123",
            "architecture": "granite",
            "native_isa_baseline": "avx2_fma",
            "native_optional_isa_leaves": ["amx", "avx2"],
            "avx2_enabled": True,
            "amx_enabled": True,
            "backend_module": "granite4",
            "backend_module_loaded": True,
            "native_backend_abi_match": True,
            "strict_path_stable": True,
            "hot_path_numpy_detected": False,
            "affinity_policy": "close",
            "l3_domain_count": 2,
        }
    )

    payload = vector.as_dict()

    assert payload["native_isa_baseline"] == "avx2_fma"
    assert payload["optional_isa_leaves"] == ["amx", "avx2"]
    assert payload["strict_path_stable"] is True
    assert len(vector.stable_digest()) == 64


def test_performance_envelope_derives_acceptance_ratio() -> None:
    envelope = PerformanceEnvelope.from_runtime_status(
        {
            "ttft_ms": 42.0,
            "per_token_latency_p95_ms": 9.5,
            "end_to_end_throughput_tps": 12.0,
            "effective_prefill_throughput_tps": 40.0,
            "prompt_cache_hit_ratio": 0.7,
            "prefix_cache_hit_rate": 0.5,
            "drift_overhead_percent": 4.0,
            "proposed_parallel_tokens": 8,
            "accepted_parallel_tokens": 6,
            "scheduler_queue_wait_ms": 1.0,
            "scheduler_iteration_ms": 2.0,
            "python_hot_path_calls": 0,
            "numpy_hot_path_calls": 0,
            "measurement_valid": True,
        },
        capability_digest="digest-1",
        delta_watermark=DeltaWatermark(delta_id="delta-1", logical_clock=3).as_dict(),
    )

    payload = envelope.as_dict()

    assert payload["capability_digest"] == "digest-1"
    assert payload["draft_acceptance_ratio"] == 0.75
    assert payload["delta_watermark"]["delta_id"] == "delta-1"


def test_qsg_runtime_invariants_detect_contract_mismatch() -> None:
    violations = evaluate_qsg_runtime_invariants(
        runtime_status={"qsg_native_runtime_authority": True, "qsg_capability_digest": ""},
        latent_packet={
            "abi_version": 1,
            "execution_capsule_id": "capsule-a",
            "capability_digest": "digest-a",
            "delta_watermark": {"delta_id": "delta-a"},
        },
        execution_capsule={
            "capsule_id": "capsule-b",
            "version": 1,
            "capability_digest": "digest-b",
            "delta_watermark": {"delta_id": "delta-b"},
        },
    )

    codes = {item["code"] for item in violations}

    assert "latent_packet_abi_version" in codes
    assert "execution_capsule_version" in codes
    assert "packet_capsule_mismatch" in codes
    assert "capability_digest_mismatch" in codes
    assert "delta_watermark_mismatch" in codes
    assert "runtime_capability_digest" in codes


def test_memory_tier_policy_prefers_latent_replay_for_mission_replay() -> None:
    decision = MemoryTierPolicy().choose(
        purpose="mission_replay",
        runtime_status={"qsg_queue_wait_ms_p95": 18.0},
        latent_package={"latent_package_id": "latent-1"},
        compatibility={"compatible": True, "mismatches": []},
        repo_delta_memory={"delta_id": "delta-1", "changed_paths": ["core/a.py"]},
    )

    assert decision.selected_tier == "latent_replay"
    assert decision.replay_allowed is True
    assert decision.fallback_tier == "repo_delta_memory"


def test_repo_delta_memory_record_uses_delta_watermark_changed_paths() -> None:
    record = RepoDeltaMemoryRecord.from_delta_watermark(
        DeltaWatermark(
            delta_id="delta-1",
            workspace_id="ws-1",
            changed_paths=["core/a.py", "tests/test_a.py"],
        ).as_dict(),
        capability_digest="cap-1",
        source_stage="capture",
    )

    payload = record.as_dict()

    assert payload["delta_id"] == "delta-1"
    assert payload["path_count"] == 2
    assert payload["capability_digest"] == "cap-1"


def test_performance_twin_predicts_queue_bound_risk() -> None:
    envelope = PerformanceEnvelope.from_runtime_status(
        {
            "ttft_ms": 210.0,
            "per_token_latency_p95_ms": 65.0,
            "scheduler_queue_wait_ms": 55.0,
            "drift_overhead_percent": 3.0,
            "proposed_parallel_tokens": 10,
            "accepted_parallel_tokens": 1,
        },
        capability_digest="digest-queue",
    )
    vector = RuntimeCapabilityVector.from_status(
        {
            "model": "granite",
            "strict_path_stable": True,
            "hot_path_numpy_detected": False,
        }
    )

    prediction = PerformanceTwinModel().predict(
        envelope=envelope,
        capability_vector=vector,
        controller_state={"frontier": {"selected_mode": "medusa"}},
    )

    assert prediction.risk_level in {"medium", "high"}
    assert prediction.predicted_regime == "queue_bound"
    assert "queue_wait_high" in prediction.issues
