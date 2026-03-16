from __future__ import annotations

import argparse
import sys

import benchmarks.native_qsg_benchmark as native_bench
import pytest
from benchmarks.native_qsg_benchmark import (
    BENCHMARK_REPORT_SCHEMA,
    BenchmarkResult,
    _build_report,
    _markdown_report,
    _phase_breakdown,
)


def _make_result(**overrides: object) -> BenchmarkResult:
    base = BenchmarkResult(
        model="qwen3.5:9b",
        threads=8,
        max_new_tokens=32,
        run_index=0,
        load_seconds=1.5,
        generate_seconds=4.0,
        new_tokens=16,
        tokens_per_second=4.0,
        digest="sha256:test",
        quantization="Q4_K_M",
        context_length=4096,
        prompt_tokens=100,
        runtime_total_seconds=3.5,
        runtime_prefill_seconds=2.0,
        runtime_decode_seconds=1.5,
        first_token_latency_seconds=2.2,
        prompt_format_ms=8.0,
        tokenize_ms=12.0,
        embedding_lookup_ms=20.0,
        graph_prefill_ms=1600.0,
        graph_decode_ms=900.0,
        sample_ms=35.0,
        logits_processor_ms=25.0,
        penalty_ms=10.0,
        suppression_ms=5.0,
        graph_prefill_calls=2,
        graph_decode_calls=16,
        sample_calls=16,
        logits_processor_calls=5,
        penalty_calls=4,
        suppression_calls=16,
        graph_prefill_avg_ms=800.0,
        graph_decode_avg_ms=56.25,
        sample_avg_ms=35.0 / 16.0,
        logits_processor_avg_ms=5.0,
        penalty_avg_ms=2.5,
        suppression_avg_ms=5.0 / 16.0,
        ttft_ms=2200.0,
        prefill_throughput_tps=50.0,
        effective_prefill_tokens=80,
        effective_prefill_throughput_tps=40.0,
        decode_throughput_tps=10.5,
        end_to_end_throughput_tps=4.57,
        per_token_latency_p50_ms=90.0,
        per_token_latency_p95_ms=120.0,
        per_token_latencies_ms=[80.0, 90.0, 100.0, 120.0],
        per_token_latency_p10_ms=83.0,
        per_token_latency_p25_ms=87.5,
        per_token_latency_p75_ms=105.0,
        per_token_latency_p99_ms=119.4,
        per_token_latency_stddev_ms=14.79,
        per_token_latency_min_ms=80.0,
        per_token_latency_max_ms=120.0,
        runtime_vs_wall_delta_ms=500.0,
        decode_threads=8,
        batch_threads=12,
        ubatch=32,
        requested_decode_threads=8,
        requested_batch_threads=12,
        requested_ubatch=32,
        openmp_enabled=True,
        avx2_enabled=True,
        avx512_enabled=False,
        mmap_enabled=True,
        mapped_model_bytes=1024,
        loader_cache_residency_bytes=2048,
        embedding_materialization_bytes=128,
        kv_used_cells=204,
        kv_fragmentation_ratio=0.1,
        kv_defrag_count=1,
        kv_cache_quantization="q8",
        template_name="chatml",
        granite_moe_mode="dense",
        active_thread_mode="split",
        prefill_chunk_count=2,
        lm_head_layout="q6_k_r4",
        lm_head_qtype=114,
        graph_forward_token_id_calls=16,
        graph_forward_token_ids_calls=2,
        batched_prefill_token_id_calls=2,
        batched_prefill_token_id_path=True,
        runtime_thread_switches=3,
        parallel_decode=False,
        speculative_decode=False,
        generation_mode="parallel_hybrid",
        benchmark_label="parallel_hybrid",
        prompt_category="code",
        temperature_band="deterministic",
        accepted_parallel_tokens=8,
        rejected_parallel_tokens=1,
        proposed_parallel_tokens=9,
        draft_frontier_width=8,
        verify_depth=8,
        parallel_step_latency_ms=3.2,
        draft_confidence_mean=0.84,
        draft_confidence_min=0.61,
        draft_source="medusa_head_native",
        blockwise_blocks=0,
        blockwise_denoise_steps=0,
        blockwise_convergence_rate=0.0,
        prefix_cache_hit_rate=2.0 / 3.0,
        scheduler_queue_wait_ms=1.5,
        scheduler_iteration_ms=2.25,
        quality_guard_triggered=True,
        prompt_cache_hit=True,
        prompt_cache_hits=2,
        prompt_cache_misses=1,
        prompt_cache_lookups=3,
        prompt_cache_hit_ratio=2.0 / 3.0,
        prompt_cache_reused_tokens=20,
        prompt_cache_reuse_ratio=0.2,
        sampling_profile="instruct_deterministic",
        temperature=0.0,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.05,
        rss_before_mb=100.0,
        rss_after_load_mb=250.0,
        rss_after_generate_mb=280.0,
        sample_text="SIMD improves locality and evaluates more lanes per CPU step.",
        printable_ratio=1.0,
        repeated_4gram_ratio=0.01,
        repeated_8gram_ratio=0.0,
        ascii_ratio=1.0,
        word_count=10,
        utf8_valid=True,
        leaked_control_text=False,
        leaked_think_tags=False,
        stop_reason="max_tokens",
        schema_valid=False,
        tool_call_parse_success=False,
        measurement_valid=True,
        measurement_issues=[],
        measurement_issue_count=0,
        measurement_source="runtime",
        native_build_id="20260306T000000Z",
        native_build_sha256="abc123",
        loaded_native_library="/tmp/libanvil_native_ops.so",
        sanctioned_backend_path=native_bench.SANCTIONED_BACKEND_PATH,
        tokenizer_backend="native",
        pmu_observed=True,
        pmu_parse_error="",
        pmu_cycles=3_200_000.0,
        pmu_instructions=6_400_000.0,
        pmu_ipc=2.0,
        pmu_cache_references=400_000.0,
        pmu_cache_misses=20_000.0,
        pmu_cache_miss_rate=0.05,
        pmu_context_switches=12.0,
        pmu_cpu_migrations=4.0,
        pmu_page_faults=128.0,
        sampling_backend="native_simd",
        penalties_backend="native_simd",
        suppression_backend="native_simd",
        logits_backend="native_graph_token_id",
        full_qsg_enabled=True,
        full_graph_enabled=True,
        qsg_processors_native_enabled=True,
        batched_prefill_native_enabled=True,
        context_stabilizer_enabled=True,
        context_stabilizer_mode="aggressive",
        drift_latest=0.31,
        drift_mean=0.22,
        drift_max=0.75,
        drift_decay_ratio=0.88,
        drift_damped_blocks=4,
        drift_pruned_blocks=1,
        drift_active_tokens=16384,
        stabilizer_seconds=0.18,
        stabilizer_calls=24,
        drift_auto_downgrade_events=1,
        drift_overhead_percent=12.0,
        coconut_enabled=True,
        strict_path_stable=True,
        hot_path_numpy_detected=False,
        python_hot_path_calls=0,
        numpy_hot_path_calls=0,
        python_qsg_forward_calls=0,
        python_attention_fallback_calls=0,
        python_ssm_fallback_calls=0,
        python_moe_fallback_calls=0,
        llama_cpp_hot_path_calls=0,
        physical_core_count=8,
        logical_core_count=16,
        p_core_count=8,
        affinity_policy="close",
        affinity_mode=1,
        l3_domain_count=2,
        os_thread_migrations=4,
        os_last_cpu=7,
        omp_places="cores",
        omp_proc_bind="close",
        hot_path_proof={"sampling": "C++ AVX2/OpenMP"},
        graph_stage_ms={"lm_head": 900.0, "attention_decode": 600.0},
        graph_stage_avg_ms={"lm_head": 56.25, "attention_decode": 37.5},
        graph_stage_calls={"forward_token": 16, "attention": 16},
        arch_hidden_dim=4096,
        arch_num_layers=32,
        arch_num_attention_layers=24,
        arch_num_ssm_layers=8,
        arch_num_moe_layers=0,
        arch_num_heads=32,
        arch_num_kv_heads=8,
        arch_head_dim=128,
        arch_intermediate_dim=14336,
        arch_vocab_size=151936,
        arch_ssm_state_dim=0,
        arch_ssm_conv_kernel=0,
        arch_num_experts=0,
        arch_top_k_experts=0,
        arch_rope_dim=128,
        arch_weight_qtype="Q6_K",
        arch_lm_head_qtype="Q6_K_LM",
        kernel_efficiency={"lm_head": {"gflops_per_second": 210.5}},
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_phase_breakdown_exposes_prefill_decode_gaps() -> None:
    result = _make_result()

    payload = _phase_breakdown(result)

    assert payload["runtime_prefill_ms"] == 2000.0
    assert payload["runtime_decode_ms"] == 1500.0
    assert payload["prefill_phase_accounted_ms"] == 1640.0
    assert payload["decode_phase_accounted_ms"] == 975.0
    assert payload["prefill_phase_gap_ms"] == 360.0
    assert payload["decode_phase_gap_ms"] == 525.0
    assert payload["total_phase_gap_ms"] == 885.0
    assert payload["runtime_vs_wall_delta_ms"] == 500.0


def test_apply_suite_target_affinity_expands_process(
    monkeypatch,
) -> None:  # noqa: ANN001
    affinity_state = {"cpus": {0}}

    monkeypatch.setenv("ANVIL_SUITE_TARGET_CPUS", "0-3")
    monkeypatch.delenv("ANVIL_NATIVE_PIN_THREADS", raising=False)
    monkeypatch.setattr(
        native_bench.os,
        "sched_getaffinity",
        lambda _pid: set(affinity_state["cpus"]),
    )

    def _set_affinity(_pid: int, cpus: set[int]) -> None:
        affinity_state["cpus"] = set(cpus)

    monkeypatch.setattr(native_bench.os, "sched_setaffinity", _set_affinity)

    payload = native_bench._apply_suite_target_affinity()

    assert payload["requested"] == [0, 1, 2, 3]
    assert payload["before"] == [0]
    assert payload["after"] == [0, 1, 2, 3]
    assert payload["applied"] is True
    assert native_bench.os.environ["ANVIL_NATIVE_PIN_THREADS"] == "0"


def test_host_facts_preserve_launch_affinity_view(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        native_bench,
        "_AFFINITY_BOOTSTRAP",
        {
            "requested": [0, 1, 2, 3],
            "before": [0],
            "after": [0, 1, 2, 3],
            "applied": True,
            "error": "",
        },
    )
    monkeypatch.setattr(native_bench.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(native_bench, "_current_affinity_cpus", lambda: [0, 1])
    monkeypatch.setattr(
        native_bench, "_perf_event_probe", lambda: (True, "available:test")
    )
    monkeypatch.setattr(native_bench, "_cpu_model", lambda: "Test CPU")
    monkeypatch.setattr(native_bench, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(
        native_bench, "_read_selected_kernel_mode", lambda _path: "always"
    )
    monkeypatch.setattr(native_bench.platform, "node", lambda: "test-host")
    monkeypatch.setattr(native_bench.platform, "platform", lambda: "Linux-test")
    monkeypatch.setattr(native_bench.platform, "machine", lambda: "x86_64")
    if native_bench.psutil is not None:
        monkeypatch.setattr(native_bench.psutil, "cpu_count", lambda logical=False: 4)

    facts = native_bench._host_facts()

    assert facts["affinity_visible_threads"] == 4
    assert facts["runtime_affinity_visible_threads"] == 2
    assert facts["suite_target_cpus"] == [0, 1, 2, 3]
    assert facts["suite_target_applied"] is True


def test_report_host_facts_prefers_runtime_thread_budget(
    monkeypatch,
) -> None:  # noqa: ANN001
    monkeypatch.setattr(
        native_bench,
        "_host_facts",
        lambda: {
            "affinity_visible_threads": 16,
            "runtime_affinity_visible_threads": 2,
            "launch_affinity_cpus": list(range(16)),
            "runtime_affinity_cpus": [0, 8],
        },
    )

    report = native_bench._report_host_facts(
        [
            _make_result(
                logical_core_count=16,
                omp_max_threads=16,
                worker_cpu_mask="",
                orchestrator_cpu_mask="",
            )
        ]
    )

    assert report["runtime_affinity_visible_threads"] == 16
    assert report["runtime_affinity_cpus"] == list(range(16))


def test_parse_perf_stat_artifact_extracts_phase10_metrics(
    tmp_path,
) -> None:  # noqa: ANN001
    perf_stat = tmp_path / "perf_stat.txt"
    perf_stat.write_text(
        "\n".join(
            [
                "1,234,567      cycles",
                "2,469,134      instructions              #    2.00  insn per cycle",
                "98,765         cache-references",
                "4,938          cache-misses              #    5.00% of all cache refs",
                "32             context-switches",
                "7              cpu-migrations",
                "12             page-faults",
            ]
        ),
        encoding="utf-8",
    )

    metrics = native_bench._parse_perf_stat_artifact(str(perf_stat))

    assert metrics["pmu_observed"] is True
    assert metrics["pmu_parse_error"] == ""
    assert metrics["pmu_cycles"] == 1234567.0
    assert metrics["pmu_instructions"] == 2469134.0
    assert metrics["pmu_ipc"] == 2.0
    assert metrics["pmu_cache_miss_rate"] == 0.05
    assert metrics["pmu_context_switches"] == 32.0
    assert metrics["pmu_cpu_migrations"] == 7.0


def test_parse_perf_stat_artifact_marks_missing_file(tmp_path) -> None:  # noqa: ANN001
    metrics = native_bench._parse_perf_stat_artifact(str(tmp_path / "missing.txt"))

    assert metrics["pmu_observed"] is False
    assert metrics["pmu_parse_error"] == "perf_stat_artifact_missing"


def test_parse_perf_stat_artifact_handles_unavailable_counters(
    tmp_path,
) -> None:  # noqa: ANN001
    perf_stat = tmp_path / "perf_stat.txt"
    perf_stat.write_text(
        "\n".join(
            [
                "<not counted> cycles",
                "<not counted> instructions",
                "<not counted> cache-misses",
            ]
        ),
        encoding="utf-8",
    )

    metrics = native_bench._parse_perf_stat_artifact(str(perf_stat))

    assert metrics["pmu_observed"] is False
    assert metrics["pmu_parse_error"] == "perf_stat_counters_unavailable"
    assert metrics["pmu_ipc"] is None


def test_compute_kernel_efficiency_emits_quantized_stage_metrics() -> None:
    efficiency = native_bench._compute_kernel_efficiency(
        graph_stage_ms={"lm_head": 120.0, "ffn_gate_up": 60.0},
        graph_stage_calls={"lm_head": 4, "ffn_gate_up": 8},
        arch={
            "hidden_dim": 4096,
            "intermediate_dim": 14336,
            "vocab_size": 151936,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
    )

    assert "lm_head" in efficiency
    assert "ffn_gate_up" in efficiency
    assert efficiency["lm_head"]["calls"] == 4
    assert float(efficiency["lm_head"]["gbytes_per_second"]) > 0.0
    assert float(efficiency["ffn_gate_up"]["arithmetic_intensity"]) > 0.0


def test_build_report_groups_metrics_for_analysis() -> None:
    results = [
        _make_result(),
        _make_result(
            run_index=1,
            ttft_ms=2100.0,
            effective_prefill_throughput_tps=44.0,
            decode_throughput_tps=12.0,
            prompt_cache_hit_ratio=1.0,
            prompt_cache_hits=1,
            prompt_cache_misses=0,
            prompt_cache_lookups=1,
            runtime_thread_switches=1,
        ),
    ]
    failures = {"qwen3.5:9b#1": ["measurement_valid=false"]}

    report = _build_report(
        results,
        failures,
        prompt="Explain AVX2 prefill.",
        criteria={"require_measurement_valid": True, "max_ttft_ms": 2500.0},
    )

    assert report["schema_version"] == BENCHMARK_REPORT_SCHEMA
    assert report["host"]["logical_cpus"] >= 1
    assert report["failure_count"] == 1
    assert report["pass_count"] == 1
    assert report["criteria"]["max_ttft_ms"] == 2500.0

    first = report["results"][0]
    assert first["identity"]["template_name"] == "chatml"
    assert first["identity"]["prompt_category"] == "code"
    assert first["phases"]["prefill_phase_gap_ms"] == 360.0
    assert first["phases"]["decode_phase_gap_ms"] == 525.0
    assert first["cache"]["prompt_cache_hit_ratio"] == 2.0 / 3.0
    assert first["threading"]["runtime_thread_switches"] == 3
    assert first["threading"]["batched_prefill_token_id_calls"] == 2
    assert first["threading"]["batched_prefill_token_id_path"] is True
    assert first["threading"]["batched_prefill_token_id_tokens"] == 0
    assert first["hotspots"]["suppression_calls"] == 16
    assert first["hotspots"]["penalty_avg_ms"] == 2.5
    assert first["hotspots"]["graph_forward_token_ids_calls"] == 2
    assert first["hotspots"]["graph_stage_ms"]["lm_head"] == 900.0
    assert first["hotspots"]["graph_stage_calls"]["attention"] == 16
    assert first["hotspots"]["drift_mean"] == 0.22
    assert first["kv_cache"]["quantization"] == "q8"
    assert first["hotspots"]["drift_overhead_percent"] == 12.0
    assert first["hardware"]["native_build_id"] == "20260306T000000Z"
    assert first["hardware"]["physical_core_count"] == 8
    assert first["hardware"]["affinity_mode"] == 1
    assert first["hardware"]["l3_domain_count"] == 2
    assert first["threading"]["os_thread_migrations"] == 4
    assert first["threading"]["os_last_cpu"] == 7
    assert first["threading"]["generation_mode"] == "parallel_hybrid"
    assert first["threading"]["benchmark_label"] == "parallel_hybrid"
    assert first["threading"]["prompt_category"] == "code"
    assert first["threading"]["temperature_band"] == "deterministic"
    assert first["threading"]["scheduler_queue_wait_ms"] == 1.5
    assert first["hot_path"]["sampling_backend"] == "native_simd"
    assert (
        first["hot_path"]["sanctioned_backend_path"]
        == native_bench.SANCTIONED_BACKEND_PATH
    )
    assert first["hot_path"]["tokenizer_backend"] == "native"
    assert first["hot_path"]["full_qsg_enabled"] is True
    assert first["hot_path"]["lm_head_layout"] == "q6_k_r4"
    assert first["hot_path"]["lm_head_qtype"] == 114
    assert first["timecrystal"]["context_stabilizer_enabled"] is True
    assert first["timecrystal"]["context_stabilizer_mode"] == "aggressive"
    assert first["timecrystal"]["drift_mean"] == 0.22
    assert first["timecrystal"]["drift_overhead_percent"] == 12.0
    assert first["timecrystal"]["coconut_enabled"] is True
    assert first["timecrystal"]["strict_path_stable"] is True
    assert first["architecture"]["hidden_dim"] == 4096
    assert first["architecture"]["weight_qtype"] == "Q6_K"
    assert first["kernel_efficiency"]["lm_head"]["gflops_per_second"] == 210.5
    assert first["latency_distribution"]["p99_ms"] == 119.4
    assert first["latency_distribution"]["min_ms"] == 80.0
    assert first["latency_distribution"]["max_ms"] == 120.0
    assert first["quality"]["coherence_valid"] is True
    assert first["quality"]["coherence_issue_count"] == 0
    assert first["measurement"]["issue_count"] == 0
    assert first["measurement"]["pmu"]["observed"] is True
    assert first["measurement"]["pmu"]["ipc"] == 2.0
    assert first["measurement"]["pmu"]["context_switches"] == 12.0
    assert first["measurement"]["pmu"]["cpu_migrations"] == 4.0
    assert first["status"]["ok"] is True

    second = report["results"][1]
    assert second["status"]["ok"] is False
    assert second["status"]["issues"] == ["measurement_valid=false"]

    summary = report["summary"][0]
    assert summary["model"] == "qwen3.5:9b"
    assert summary["runs"] == 2
    assert summary["passes"] == 1
    assert summary["failures"] == 1
    assert summary["avg_ttft_ms"] == 2150.0
    assert summary["avg_effective_prefill_tps"] == 42.0
    assert summary["avg_decode_tps"] == 11.25
    assert summary["avg_runtime_thread_switches"] == 2.0
    assert summary["avg_pmu_ipc"] == 2.0
    assert summary["avg_pmu_context_switches"] == 12.0
    assert summary["avg_pmu_cpu_migrations"] == 4.0
    assert summary["avg_pmu_cache_miss_rate"] == 0.05
    assert summary["avg_drift_mean"] == 0.22
    assert summary["avg_drift_overhead_percent"] == 12.0
    speculative = report["speculative_acceptance"][0]
    assert speculative["prompt_category"] == "code"
    assert speculative["temperature_band"] == "deterministic"
    assert speculative["benchmark_label"] == "parallel_hybrid"
    assert speculative["proposed_parallel_tokens"] == 18
    assert speculative["accepted_parallel_tokens"] == 16
    assert speculative["rejected_parallel_tokens"] == 2
    assert speculative["avg_draft_confidence_mean"] == 0.84
    assert speculative["min_draft_confidence"] == 0.61
    assert report["non_ar_decision"][0]["recommended_mode"] == "insufficient_candidate_coverage"
    assert report["non_ar_decision"][0]["decision_scope"] == "research_only"
    assert report["non_ar_decision"][0]["production_ready"] is False


def test_markdown_report_includes_summary_and_phase_columns() -> None:
    result = _make_result()

    report = _markdown_report(
        [result],
        {"qwen3.5:9b#0": ["leaked_control_text=true"]},
    )

    assert "# Native QSG Benchmark Report" in report
    assert "Schema: `native_qsg_benchmark.v7`" in report
    assert "## Summary" in report
    assert "## Speculative Acceptance" in report
    assert "## Non-AR Decision Framework" in report
    assert "## Runs" in report
    assert "Avg Prefill Gap ms" in report
    assert "Benchmark Label" in report
    assert "Avg Draft Conf" in report
    assert "Prod Ready" in report
    assert "Readiness Blockers" in report
    assert "research_only" in report
    assert "LM Head" in report
    assert "Batch TokenIds" in report
    assert "Batch Tokens" in report
    assert "Coherence" in report
    assert "Thread Switches" in report
    assert "q6_k_r4" in report
    assert "`qwen3.5:9b#0`: leaked_control_text=true" in report


def test_build_report_emits_non_ar_decision_framework() -> None:
    ar_result = _make_result(
        run_index=0,
        generation_mode="ar_verify",
        benchmark_label="ar_baseline",
        quality_guard_triggered=False,
        accepted_parallel_tokens=0,
        rejected_parallel_tokens=0,
        proposed_parallel_tokens=0,
        draft_confidence_mean=0.0,
        draft_confidence_min=0.0,
        draft_source="",
        decode_throughput_tps=10.0,
        ttft_ms=120.0,
        coherence_valid=True,
    )
    self_spec_result = _make_result(
        run_index=1,
        generation_mode="parallel_hybrid",
        benchmark_label="parallel_hybrid",
        quality_guard_triggered=False,
        decode_throughput_tps=11.0,
        ttft_ms=125.0,
        coherence_valid=True,
    )
    block_result = _make_result(
        run_index=2,
        generation_mode="block_diffusion",
        benchmark_label="block_diffusion_candidate",
        quality_guard_triggered=False,
        decode_throughput_tps=13.0,
        ttft_ms=128.0,
        coherence_valid=True,
        draft_confidence_mean=0.42,
        draft_confidence_min=0.24,
        draft_source="block_diffusion_native",
        blockwise_blocks=4,
        blockwise_denoise_steps=6,
        blockwise_convergence_rate=0.75,
        python_hot_path_calls=8,
    )

    report = _build_report(
        [ar_result, self_spec_result, block_result],
        {},
        prompt="Complete the repeated code scaffold with structured boilerplate.",
        criteria={"require_measurement_valid": True},
    )

    decision = report["non_ar_decision"][0]
    assert decision["recommended_mode"] == "block_diffusion"
    assert decision["decision_scope"] == "research_only"
    assert decision["production_ready"] is False
    assert decision["native_hot_path_owned"] is False
    assert decision["python_hot_path_calls_max"] == 8
    assert "python_or_numpy_hot_path_owned" in decision["production_blockers"]
    assert decision["decode_speedup_vs_ar"] == pytest.approx(1.3)
    assert decision["decode_speedup_vs_self_spec"] == pytest.approx(13.0 / 11.0)
    assert decision["blockwise_convergence_rate"] == pytest.approx(0.75)
    assert decision["draft_source"] == "block_diffusion_native"
    assert "research-only" in decision["reason"]


def test_non_ar_decision_omits_masked_runtime_blocker_when_runtime_ready() -> None:
    ar_result = _make_result(
        run_index=0,
        generation_mode="ar_verify",
        benchmark_label="ar_baseline",
        decode_throughput_tps=10.0,
        ttft_ms=120.0,
        coherence_valid=True,
    )
    block_result = _make_result(
        run_index=1,
        generation_mode="block_diffusion",
        benchmark_label="block_diffusion_candidate",
        decode_throughput_tps=12.0,
        ttft_ms=121.0,
        coherence_valid=True,
        masked_generation_ready=True,
    )

    report = _build_report(
        [ar_result, block_result],
        {},
        prompt="Emit a candidate continuation from the research substrate.",
        criteria={"require_measurement_valid": True},
    )

    decision = report["non_ar_decision"][0]
    assert "masked_generation_runtime_missing" not in decision["production_blockers"]


def test_build_report_emits_self_spec_runtime_fields() -> None:
    result = _make_result(
        generation_mode="ssd_bridge",
        benchmark_label="ssd_bridge",
        self_spec_native_path=True,
        self_spec_policy="heuristic",
        self_spec_exit_layer=12,
        self_spec_exit_fraction=0.5,
        self_spec_draft_tokens=8,
    )

    report = _build_report(
        [result],
        {},
        prompt="Explain the native early-exit verifier tail.",
        criteria={"require_measurement_valid": True},
    )

    runtime = report["results"][0]["threading"]
    assert runtime["self_spec_native_path"] is True
    assert runtime["self_spec_policy"] == "heuristic"
    assert runtime["self_spec_exit_layer"] == 12
    assert runtime["self_spec_exit_fraction"] == pytest.approx(0.5)
    assert runtime["self_spec_draft_tokens"] == 8


def test_run_once_plumbs_lm_head_layout_and_batched_prefill_runtime_fields(
    monkeypatch,
) -> None:
    perf_counter_values = iter([0.0, 0.1, 0.1, 0.7])
    monkeypatch.setattr(
        native_bench.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(native_bench, "_rss_mb", lambda: 128.0)

    class _StubEngine:
        def __init__(self, model: str, context_length: int):  # noqa: ARG002
            self.num_threads_decode = 8
            self.num_threads_batch = 12
            self.num_ubatch = 32
            self.last_generate_kwargs = None
            self._model_config = type(
                "Cfg",
                (),
                {
                    "hidden_dim": 4096,
                    "num_layers": 32,
                    "num_heads": 32,
                    "num_kv_heads": 8,
                    "head_dim": 128,
                    "intermediate_dim": 14336,
                    "vocab_size": 151936,
                    "num_experts": 0,
                    "top_k_experts": 0,
                    "weight_qtype": "Q6_K",
                    "lm_head_qtype": "Q6_K_LM",
                },
            )()

        def prepare_prompt_tokens(self, prompt: str) -> list[int]:  # noqa: ARG002
            return [11, 22, 33, 44]

        def generate(self, prompt_tokens: list[int], **kwargs) -> list[int]:
            self.last_generate_kwargs = dict(kwargs)
            return list(prompt_tokens) + [55, 66]

        def get_runtime_status(self) -> dict[str, object]:
            return {
                "digest": "sha256:stub",
                "quantization": "Q6_K",
                "kv_cache_quantization": "q8",
                "context_length": 4096,
                "total_seconds": 0.6,
                "prefill_seconds": 0.4,
                "decode_seconds": 0.2,
                "first_token_latency_seconds": 0.45,
                "graph_prefill_ms": 320.0,
                "graph_decode_ms": 180.0,
                "graph_prefill_calls": 2,
                "graph_decode_calls": 2,
                "graph_prefill_avg_ms": 160.0,
                "graph_decode_avg_ms": 90.0,
                "per_token_latencies_ms": [80.0, 90.0],
                "prefill_chunk_count": 2,
                "prefill_throughput_tps": 10.0,
                "effective_prefill_tokens": 4,
                "effective_prefill_throughput_tps": 10.0,
                "decode_throughput_tps": 10.0,
                "end_to_end_throughput_tps": 10.0 / 3.0,
                "decode_threads": 8,
                "batch_threads": 12,
                "ubatch": 32,
                "template_name": "chatml",
                "granite_moe_mode": "dense",
                "active_thread_mode": "batch",
                "runtime_thread_switches": 1,
                "sampling_backend": "native_simd",
                "penalties_backend": "native_simd",
                "suppression_backend": "none",
                "logits_backend": "native_graph_token_id",
                "sanctioned_backend_path": native_bench.SANCTIONED_BACKEND_PATH,
                "tokenizer_backend": "native",
                "full_qsg_enabled": True,
                "full_graph_enabled": True,
                "qsg_processors_native_enabled": True,
                "batched_prefill_native_enabled": True,
                "context_stabilizer_enabled": True,
                "context_stabilizer_mode": "conservative",
                "drift_latest": 0.35,
                "drift_mean": 0.28,
                "drift_max": 0.71,
                "drift_decay_ratio": 0.91,
                "drift_damped_blocks": 3,
                "drift_pruned_blocks": 0,
                "drift_active_tokens": 16384,
                "stabilizer_seconds": 0.08,
                "stabilizer_calls": 12,
                "drift_auto_downgrade_events": 2,
                "drift_overhead_percent": 14.5,
                "coconut_enabled": True,
                "strict_path_stable": True,
                "hot_path_numpy_detected": False,
                "python_hot_path_calls": 0,
                "numpy_hot_path_calls": 0,
                "python_qsg_forward_calls": 0,
                "python_attention_fallback_calls": 0,
                "python_ssm_fallback_calls": 0,
                "python_moe_fallback_calls": 0,
                "llama_cpp_hot_path_calls": 0,
                "physical_core_count": 8,
                "logical_core_count": 16,
                "p_core_count": 8,
                "affinity_policy": "close",
                "affinity_mode": 2,
                "l3_domain_count": 4,
                "os_thread_migrations": 6,
                "os_last_cpu": 11,
                "omp_places": "cores",
                "omp_proc_bind": "close",
                "batched_prefill_token_id_calls": 5,
                "batched_prefill_token_id_tokens": 9,
                "packed_lm_head_calls": 7,
                "lm_head_layout": "q6_k_r4",
                "lm_head_qtype": 114,
                "hot_path_proof": {"lm_head_layout": "q6_k_r4"},
                "graph_stage_calls": {
                    "forward_token": 2,
                    "forward_token_id": 1,
                    "forward_token_ids": 2,
                    "forward_token_ids_token_count": 4,
                    "packed_lm_head": 2,
                    "lm_head": 2,
                    "ffn_gate_up": 2,
                },
                "graph_stage_ms": {"lm_head": 25.0, "ffn_gate_up": 50.0},
            }

        def decode_generated_tokens(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "stub output"

        def close(self) -> None:
            return None

    monkeypatch.setattr(native_bench, "NativeQSGEngine", _StubEngine)

    result = native_bench._run_once(
        model="granite4:tiny-h",
        prompt="Explain batched prefill.",
        max_new_tokens=2,
        temperature=0.0,
        run_index=0,
    )

    assert result.lm_head_layout == "q6_k_r4"
    assert result.lm_head_qtype == 114
    assert result.full_qsg_enabled is True
    assert result.sanctioned_backend_path == native_bench.SANCTIONED_BACKEND_PATH
    assert result.tokenizer_backend == "native"
    assert result.graph_forward_token_id_calls == 1
    assert result.graph_forward_token_ids_calls == 2
    assert result.batched_prefill_token_id_calls == 5
    assert result.batched_prefill_token_id_path is True
    assert result.batched_prefill_token_id_tokens == 9
    assert result.packed_lm_head_calls == 7
    assert result.prefill_chunk_count == 2
    assert result.graph_stage_calls["forward_token_ids"] == 2
    assert result.affinity_mode == 2
    assert result.l3_domain_count == 4
    assert result.os_thread_migrations == 6
    assert result.os_last_cpu == 11
    assert result.context_stabilizer_enabled is True
    assert result.context_stabilizer_mode == "conservative"
    assert result.drift_latest == 0.35
    assert result.drift_mean == 0.28
    assert result.drift_max == 0.71
    assert result.drift_decay_ratio == 0.91
    assert result.drift_damped_blocks == 3
    assert result.drift_pruned_blocks == 0
    assert result.drift_active_tokens == 16384
    assert result.stabilizer_seconds == 0.08
    assert result.stabilizer_calls == 12
    assert result.drift_auto_downgrade_events == 2
    assert result.drift_overhead_percent == 14.5
    assert result.coconut_enabled is True
    assert result.strict_path_stable is True
    assert result.kv_cache_quantization == "q8"
    assert result.arch_hidden_dim == 4096
    assert result.arch_num_layers == 32
    assert result.arch_weight_qtype == "Q6_K"
    assert result.prompt_category == "analysis"
    assert result.temperature_band == "deterministic"
    assert result.per_token_latency_p10_ms > 0.0
    assert result.per_token_latency_p99_ms > 0.0
    assert "lm_head" in result.kernel_efficiency


def test_run_once_passes_configured_seed_to_engine(monkeypatch) -> None:
    perf_counter_values = iter([0.0, 0.1, 0.1, 0.7])
    monkeypatch.setattr(
        native_bench.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )
    monkeypatch.setattr(native_bench, "_rss_mb", lambda: 128.0)

    captured: dict[str, object] = {}

    class _StubEngine:
        def __init__(self, model: str, context_length: int):  # noqa: ARG002
            self.num_threads_decode = 8
            self.num_threads_batch = 8
            self.num_ubatch = 8

        def prepare_prompt_tokens(self, prompt: str) -> list[int]:  # noqa: ARG002
            return [1, 2, 3]

        def generate(self, prompt_tokens: list[int], **kwargs) -> list[int]:
            captured.update(kwargs)
            return list(prompt_tokens) + [4]

        def get_runtime_status(self) -> dict[str, object]:
            return {}

        def decode_generated_tokens(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "seeded output"

        def close(self) -> None:
            return None

    monkeypatch.setattr(native_bench, "NativeQSGEngine", _StubEngine)

    native_bench._run_once(
        model="qwen3.5:9b",
        prompt="Explain sampling.",
        max_new_tokens=1,
        run_index=0,
    )

    assert captured["seed"] == 720720


def test_result_failures_hard_fails_incoherent_output() -> None:
    result = _make_result(
        sample_text="This is a",
        word_count=3,
        stop_reason="eos",
        coherence_valid=False,
        coherence_issues=[
            "coherence_trivial_completion=true",
            "coherence_word_count 3 < 6",
        ],
        coherence_issue_count=2,
    )

    failures = native_bench._result_failures(
        result,
        min_decode_tps=None,
        max_ttft_ms=None,
        min_printable_ratio=0.95,
        max_repeated_4gram_ratio=0.12,
        max_repeated_8gram_ratio=0.05,
        min_ascii_ratio=None,
        min_word_count=None,
        max_rss_growth_mb=None,
        require_utf8=False,
        require_measurement_valid=False,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
    )

    assert "coherence_trivial_completion=true" in failures


def test_result_failures_hard_fail_missing_full_qsg_and_sanctioned_path() -> None:
    result = _make_result(
        full_qsg_enabled=False,
        sanctioned_backend_path="llama.cpp",
        tokenizer_backend="python",
        python_hot_path_calls=1,
    )

    failures = native_bench._result_failures(
        result,
        min_decode_tps=None,
        max_ttft_ms=None,
        min_printable_ratio=0.95,
        max_repeated_4gram_ratio=0.12,
        max_repeated_8gram_ratio=0.05,
        min_ascii_ratio=None,
        min_word_count=None,
        max_rss_growth_mb=None,
        require_utf8=False,
        require_measurement_valid=False,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
    )

    assert "full_qsg_enabled=false" in failures
    assert "sanctioned_backend_path_mismatch" in failures
    assert "tokenizer_backend=python" in failures
    assert "python_hot_path_calls>0" in failures


def test_main_emits_json_payload_on_runtime_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        model=["granite4:tiny-h"],
        runs=1,
        prompt="Explain AVX2 throughput.",
        max_new_tokens=32,
        json=True,
        isolated_child=True,
        json_out=None,
        markdown_out=None,
        context_length=2048,
        temperature=None,
        sampling_profile=None,
        top_p=None,
        top_k=None,
        min_p=None,
        presence_penalty=None,
        repetition_penalty=None,
        min_decode_tps=None,
        max_ttft_ms=None,
        min_printable_ratio=None,
        max_repeated_4gram_ratio=None,
        max_repeated_8gram_ratio=None,
        min_ascii_ratio=None,
        min_word_count=None,
        max_rss_growth_mb=None,
        require_utf8=False,
        require_measurement_valid=True,
        require_openmp=False,
        require_avx2=False,
        require_mmap=False,
        host_access="user",
        collect_hw_counters="auto",
        require_grover=False,
        require_coconut=False,
        autotune="off",
        fast_native=False,
        disable_logits_processors=None,
        disable_token_penalties=None,
        force_parallel_decode=False,
        min_new_tokens_before_eos=None,
        coherence_first=False,
        self_artifact_dir=None,
        capture_self_numa_map=False,
    )

    monkeypatch.setattr(native_bench, "parse_args", lambda: args)
    monkeypatch.setattr(
        native_bench,
        "_model_thread_sweep",
        lambda model, parsed_args: [(16, 16, 32)],
    )

    def _boom(**_kwargs: object) -> BenchmarkResult:
        raise RuntimeError(
            "Strict native non-autoregressive mode forbids autoregressive fallback: "
            "planner_selected_autoregressive_completion (plan_mode=ar_verify)"
        )

    monkeypatch.setattr(native_bench, "_run_once", _boom)

    rc = native_bench.main()

    captured = capsys.readouterr()
    payload = native_bench.json.loads(captured.out)

    assert rc == 1
    assert payload["failure_count"] == 1
    assert payload["failure_keys"] == ["granite4:tiny-h#1"]
    assert payload["flat_results"][0]["measurement_valid"] is False
    assert payload["flat_results"][0]["generation_mode"] == "ar_verify"
    assert "measurement_valid=false" in payload["failures"]["granite4:tiny-h#1"]


def test_deprecated_native_benchmark_entrypoint_rejects_legacy_flags() -> None:
    rc = native_bench._deprecated_entrypoint(
        ["--model", "granite4:tiny-h", "--runs", "2"]
    )
    assert rc == 2


def test_deprecated_native_benchmark_entrypoint_keeps_isolated_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_bench, "main", lambda: 7)
    assert native_bench._deprecated_entrypoint(["--isolated-child"]) == 7


def test_deprecated_native_benchmark_entrypoint_forwards_safe_flags(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _Completed:
        returncode = 0

    def _stub_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setattr(native_bench.subprocess, "run", _stub_run)

    rc = native_bench._deprecated_entrypoint(
        ["--profile", "smoke", "--run-id", "suite-2", "--resume", "--out-root", "audit"]
    )
    assert rc == 0
    cmd = list(captured["cmd"])
    assert cmd[0] == sys.executable
    assert cmd[1].endswith("audit/runner/benchmark_suite.py")
    assert "--profile" in cmd and "smoke" in cmd
    assert "--run-id" in cmd and "suite-2" in cmd
