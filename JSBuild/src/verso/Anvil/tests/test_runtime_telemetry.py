from __future__ import annotations

from core.native.runtime_telemetry import (
    NativeGenerationTelemetry,
    build_runtime_capability_ledger,
)


def test_runtime_telemetry_exposes_prefill_decode_cache_metrics() -> None:
    telemetry = NativeGenerationTelemetry(
        prompt_tokens=100,
        generated_tokens=20,
        total_seconds=5.0,
        prefill_seconds=2.0,
        decode_seconds=3.0,
        first_token_latency_seconds=2.25,
        prompt_format_seconds=0.01,
        tokenize_seconds=0.02,
        embedding_lookup_seconds=0.03,
        graph_prefill_seconds=1.75,
        graph_decode_seconds=2.5,
        sample_seconds=0.04,
        logits_processor_seconds=0.05,
        penalty_seconds=0.06,
        suppression_seconds=0.07,
        logits_processor_calls=2,
        penalty_calls=3,
        suppression_calls=4,
        sanctioned_backend_path="prompt -> native tokenizer -> NativeQSGEngine -> NativeModelGraph -> C++ graph -> C++ QSG postprocess/sampling -> tokens",
        tokenizer_backend="native",
        full_qsg_enabled=True,
        context_stabilizer_enabled=True,
        context_stabilizer_mode="aggressive",
        drift_latest=0.12,
        drift_mean=0.2,
        drift_max=0.55,
        drift_decay_ratio=0.93,
        drift_damped_blocks=3,
        drift_pruned_blocks=1,
        drift_active_tokens=16384,
        stabilizer_seconds=0.45,
        stabilizer_calls=22,
        drift_auto_downgrade_events=2,
        coconut_enabled=True,
        coconut_paths=4,
        coconut_alpha=0.1,
        strict_cpp_only=True,
        strict_native_qsg=True,
        python_hot_path_calls=0,
        numpy_hot_path_calls=0,
        hot_path_proof={"lm_head_layout": "q6_k_r4"},
        graph_stage_seconds={"lm_head": 0.5, "sanitize": 0.1},
        graph_stage_calls={
            "forward_token": 20,
            "forward_token_id": 18,
            "forward_token_ids": 2,
            "attention": 16,
        },
        runtime_thread_switches=2,
        prompt_cache_hits=3,
        prompt_cache_misses=1,
        prompt_cache_reused_tokens=40,
    )

    payload = telemetry.as_dict()

    assert payload["ttft_ms"] == 2250.0
    assert payload["prompt_format_ms"] == 10.0
    assert payload["tokenize_ms"] == 20.0
    assert payload["embedding_lookup_ms"] == 30.0
    assert payload["graph_prefill_ms"] == 1750.0
    assert payload["graph_decode_ms"] == 2500.0
    assert payload["sample_ms"] == 40.0
    assert payload["logits_processor_ms"] == 50.0
    assert payload["penalty_ms"] == 60.0
    assert payload["suppression_ms"] == 70.0
    assert payload["logits_processor_avg_ms"] == 25.0
    assert payload["penalty_avg_ms"] == 20.0
    assert payload["suppression_avg_ms"] == 17.5
    assert payload["prefill_ms"] == 2000.0
    assert payload["decode_ms"] == 3000.0
    assert payload["prefill_throughput_tps"] == 50.0
    assert payload["effective_prefill_tokens"] == 60
    assert payload["effective_prefill_throughput_tps"] == 30.0
    assert payload["decode_throughput_tps"] == 20.0 / 3.0
    assert payload["end_to_end_throughput_tps"] == 4.0
    assert payload["prompt_cache_lookups"] == 4
    assert payload["prompt_cache_hit_ratio"] == 0.75
    assert payload["prompt_cache_reuse_ratio"] == 0.4
    assert payload["runtime_thread_switches"] == 2
    assert payload["sanctioned_backend_path"].startswith("prompt -> native tokenizer")
    assert payload["tokenizer_backend"] == "native"
    assert payload["full_qsg_enabled"] is True
    assert payload["python_hot_path_calls"] == 0
    assert payload["numpy_hot_path_calls"] == 0
    assert payload["hot_path_proof"]["lm_head_layout"] == "q6_k_r4"
    assert payload["graph_stage_ms"]["lm_head"] == 500.0
    assert payload["graph_stage_ms"]["sanitize"] == 100.0
    assert payload["graph_stage_calls"]["forward_token"] == 20
    assert payload["graph_stage_calls"]["forward_token_id"] == 18
    assert payload["graph_stage_calls"]["forward_token_ids"] == 2
    assert payload["graph_stage_avg_ms"]["lm_head"] == 500.0
    assert payload["context_stabilizer_enabled"] is True
    assert payload["context_stabilizer_mode"] == "aggressive"
    assert payload["drift_mean"] == 0.2
    assert payload["stabilizer_calls"] == 22
    assert payload["drift_auto_downgrade_events"] == 2
    assert payload["drift_overhead_percent"] == 15.0
    assert payload["strict_path_stable"] is True
    assert payload["measurement_valid"] is True
    assert payload["measurement_issue_count"] == 0
    assert payload["measurement_issues"] == []
    assert payload["hot_path_proof"]["sanctioned_backend_path"].startswith(
        "prompt -> native tokenizer"
    )
    assert payload["hot_path_proof"]["tokenizer_backend"] == "native"
    assert payload["hot_path_proof"]["executed_cpp_only"] == "true"
    assert payload["hot_path_proof"]["full_qsg"] == "enabled"
    assert payload["hot_path_proof"]["context_stabilizer_enabled"] == "true"
    assert payload["hot_path_proof"]["strict_path_stable"] == "true"
    assert payload["hot_path_proof"]["coconut_enabled"] == "true"


def test_runtime_telemetry_derives_full_qsg_and_hot_path_detection() -> None:
    telemetry = NativeGenerationTelemetry(
        full_graph_enabled=True,
        batched_prefill_native_enabled=True,
        qsg_processors_native_enabled=True,
        sanctioned_backend_path="prompt -> native tokenizer -> engine",
        tokenizer_backend="native",
        python_hot_path_calls=1,
    )

    payload = telemetry.as_dict()

    assert payload["full_qsg_enabled"] is True
    assert payload["hot_path_numpy_detected"] is True
    assert payload["hot_path_proof"]["executed_cpp_only"] == "false"
    assert payload["hot_path_proof"]["python_hot_path_calls"] == "1"


def test_runtime_telemetry_marks_impossible_timing_invalid() -> None:
    telemetry = NativeGenerationTelemetry(
        prompt_tokens=8,
        generated_tokens=4,
        total_seconds=1.0,
        prefill_seconds=0.9,
        decode_seconds=0.8,
        first_token_latency_seconds=1.2,
    )

    payload = telemetry.as_dict()

    assert payload["measurement_valid"] is False
    assert payload["measurement_issue_count"] == 2
    assert "ttft_exceeds_total" in payload["measurement_issues"]
    assert "prefill_plus_decode_exceeds_total" in payload["measurement_issues"]


def test_runtime_telemetry_reports_latency_percentiles_and_negative_duration() -> None:
    telemetry = NativeGenerationTelemetry(
        prompt_tokens=12,
        generated_tokens=3,
        total_seconds=0.75,
        prefill_seconds=-0.1,
        decode_seconds=0.25,
        first_token_latency_seconds=0.2,
        per_token_latencies_seconds=[0.04, 0.05, 0.09],
        prompt_cache_hits=0,
        prompt_cache_misses=0,
        prompt_cache_reused_tokens=0,
    )

    payload = telemetry.as_dict()

    assert payload["measurement_valid"] is False
    assert payload["measurement_issue_count"] == 1
    assert payload["measurement_issues"] == ["negative_duration"]
    assert payload["prompt_cache_lookups"] == 0
    assert payload["prompt_cache_hit_ratio"] == 0.0
    assert payload["prompt_cache_reuse_ratio"] == 0.0
    assert payload["per_token_latency_p50_ms"] == 50.0
    assert payload["per_token_latency_p95_ms"] == 90.0


def test_runtime_telemetry_exposes_backend_selection_audit_fields() -> None:
    telemetry = NativeGenerationTelemetry(
        backend_module_required=True,
        backend_module="qwen35",
        backend_module_loaded=True,
        backend_selection_source="inferred",
        backend_selection_reason="family",
        native_split_layout="kernels/runtime_core/backends/compat",
        native_public_load_target="libanvil_native_ops.so",
        native_runtime_core_target="libanvil_runtime_core.so",
        native_split_abi_version=1,
        backend_module_abi_version=1,
    )

    payload = telemetry.as_dict()

    assert payload["measurement_valid"] is True
    assert payload["measurement_issue_count"] == 0
    assert payload["hot_path_proof"]["backend_selection_source"] == "inferred"
    assert payload["hot_path_proof"]["backend_selection_reason"] == "family"
    assert payload["hot_path_proof"]["native_split_layout"] == (
        "kernels/runtime_core/backends/compat"
    )
    assert payload["hot_path_proof"]["native_public_load_target"] == (
        "libanvil_native_ops.so"
    )
    assert payload["hot_path_proof"]["native_runtime_core_target"] == (
        "libanvil_runtime_core.so"
    )
    assert payload["hot_path_proof"]["native_split_abi_version"] == "1"
    assert payload["hot_path_proof"]["backend_module_abi_version"] == "1"


def test_runtime_telemetry_carries_repo_coupled_runtime_fields() -> None:
    telemetry = NativeGenerationTelemetry(
        controller_state={"frontier": {"selected_mode": "prompt_lookup"}},
        execution_capsule_id="capsule-7",
        delta_watermark={"delta_id": "delta-7", "logical_clock": 3},
        performance_twin={"risk_level": "low"},
        repo_coupled_runtime={"delta_authority": "state_ledger"},
    )

    payload = telemetry.as_dict()

    assert payload["controller_state"]["frontier"]["selected_mode"] == "prompt_lookup"
    assert payload["execution_capsule_id"] == "capsule-7"
    assert payload["delta_watermark"]["delta_id"] == "delta-7"
    assert payload["performance_twin"]["risk_level"] == "low"
    assert payload["repo_coupled_runtime"]["delta_authority"] == "state_ledger"


def test_runtime_telemetry_flags_unresolved_backend_selection() -> None:
    telemetry = NativeGenerationTelemetry(
        backend_module_required=True,
        backend_module="qwen35",
        backend_module_loaded=True,
        backend_selection_source="unresolved",
    )

    payload = telemetry.as_dict()

    assert payload["measurement_valid"] is False
    assert "backend_selection_unresolved" in payload["measurement_issues"]


def test_runtime_capability_ledger_exposes_optional_isa_truth() -> None:
    payload = build_runtime_capability_ledger(
        {
            "model": "granite4:tiny-h",
            "digest": "sha256:test",
            "native_isa_baseline": "avx2",
            "decode_threads": 16,
            "batch_threads": 16,
            "ubatch": 32,
            "native_optional_isa_leaves": ["amx"],
            "native_compiled_with_amx": True,
            "native_runtime_amx_available": False,
            "native_backend_abi_match": True,
            "perf_event_access": True,
        },
        host_fingerprint="host123",
        certification_state="calibration_required",
        source="suite_preflight",
    )

    assert payload["native_isa_baseline"] == "avx2"
    assert payload["native_optional_isa_leaves"] == ["amx"]
    assert payload["native_compiled_with_amx"] is True
    assert payload["native_runtime_amx_available"] is False
    assert payload["optional_isa_leaf_state"]["amx"]["compiled"] is True
    assert payload["optional_isa_leaf_state"]["amx"]["runtime_available"] is False
    assert payload["optional_isa_leaf_state"]["amx"]["readiness_impact"] == "none"
    assert payload["continuous_config"]["scheduler_policy"] == ""
    assert payload["pager_config"]["state_page_rows"] == 0
    assert payload["tuning"]["budget_tier"] == ""
