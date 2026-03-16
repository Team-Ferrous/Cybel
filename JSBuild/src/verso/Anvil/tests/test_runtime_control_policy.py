from core.runtime_control_policy import RuntimeControlPolicy


def test_runtime_control_policy_prefers_native_ready_when_backend_is_healthy():
    policy = RuntimeControlPolicy()

    decision = policy.decide(
        {
            "backend_module_loaded": True,
            "native_backend_abi_match": True,
            "native_fast_path": True,
            "python_hot_path_calls": 0,
            "drift_overhead_percent": 3.0,
            "hot_path_proof": {"full_qsg": "enabled"},
            "capability_digest": "digest-1",
        }
    )

    assert decision.posture == "native_ready"
    assert decision.planning_depth == "deep"
    assert decision.verification_max_attempts == 2
    assert decision.degraded is False


def test_runtime_control_policy_degrades_when_python_fallback_is_observed():
    policy = RuntimeControlPolicy()

    decision = policy.decide(
        {
            "backend_module_loaded": True,
            "native_backend_abi_match": True,
            "native_fast_path": False,
            "python_hot_path_calls": 4,
            "drift_overhead_percent": 18.0,
            "hot_path_proof": {"full_qsg": "disabled"},
            "capability_digest": "digest-2",
        }
    )

    assert decision.posture in {"degraded_python_fallback", "high_drift_overhead"}
    assert decision.planning_depth in {"compact", "lean"}
    assert decision.verification_max_attempts == 1
    assert decision.degraded is True
    assert decision.reasons
