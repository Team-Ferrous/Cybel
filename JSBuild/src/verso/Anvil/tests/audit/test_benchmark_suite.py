from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit.control_plane.compiler import materialize_control_plane_artifacts
from audit.control_plane.ledger import reduce_runtime_state
from audit.provenance import capture as provenance_capture
from audit.runner import benchmark_suite
from audit.runner import suite_preflight
from audit.runner.suite_profiles import compile_suite_profile, load_suite_profile
from audit.runtime_logging import SuiteEventLogger
from audit.runtime_logging import set_active_logger
from audit.store.suite_layout import (
    ensure_suite_layout,
    required_suite_artifacts,
    resolve_suite_layout,
)


def _fake_accuracy_result(model: str) -> dict[str, object]:
    return {
        "schema_version": "native_qsg_eval.accuracy.v1",
        "model": model,
        "samples": 2,
        "generated_tokens": 4,
        "pass_count": 2,
        "pass_rate": 1.0,
        "exact_match_rate": 1.0,
        "contains_match_rate": 1.0,
        "option_match_rate": 1.0,
        "records": [],
        "documents": [],
    }


def test_default_profile_is_silver() -> None:
    assert benchmark_suite.DEFAULT_PROFILE == "silver"


def test_suite_preflight_run_times_out_cleanly(tmp_path: Path) -> None:
    code, _stdout, stderr = suite_preflight._run(
        [sys.executable, "-c", "import time; time.sleep(1)"],
        cwd=tmp_path,
        telemetry_dir=tmp_path,
        label="timeout_probe",
        timeout=0.05,
    )

    assert code == 124
    assert "timed out" in stderr


def test_load_silver_suite_profile() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    assert spec.profile_name == "silver"
    assert spec.models == ["granite4:tiny-h", "qwen3.5:4b", "qwen3.5:9b"]
    assert spec.max_parallel_models == 1
    assert spec.affinity_policy == "repair_allowed"
    assert spec.preflight_strictness == "audit"
    assert spec.tuning_contract_policy == "optional"
    assert spec.assurance_level == "AAL-2"
    assert spec.evidence_class == "certification"
    assert spec.force_parallel_decode is True
    assert spec.forbid_autoregressive_fallback is True
    assert spec.enabled_lanes == [
        "canonical_all_on",
        "continuous_scheduler",
        "kernel_microbench",
        "quality_eval",
    ]
    assert spec.ablations == []
    assert spec.scenario_pack.measured_runs == 2
    assert spec.scenario_pack.thread_matrix_ubatch == [32]
    assert spec.scenario_pack.continuous_concurrency == [1, 2, 4]
    assert spec.kernel_microbench.iterations == 48
    assert spec.calibration_search.search_profile == "search"


def test_compile_suite_profile_includes_quality_and_memory_policy() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    compiled = compile_suite_profile(spec)

    assert compiled["preflight_strictness"] == "audit"
    assert compiled["quality_policy"]["adaptive_top_k"] == 0
    assert compiled["quality_policy"]["shadow_mode"] is True
    assert compiled["quality_policy"]["accuracy_corpus"].endswith(
        "benchmarks/corpora/accuracy_corpus.jsonl"
    )
    assert compiled["quality_policy"]["max_samples_per_family"] == 0
    assert compiled["memory_replay"]["cases_path"] == ""


def test_load_gold_suite_profile() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))

    assert spec.profile_name == "gold"
    assert spec.models == ["granite4:tiny-h", "qwen3.5:4b", "qwen3.5:9b"]
    assert spec.max_parallel_models == 1
    assert spec.host_contract_id == "auto"
    assert spec.affinity_policy == "certified_exact"
    assert spec.preflight_strictness == "certify"
    assert spec.tuning_contract_policy == "required"
    assert spec.canonical_decode_threads == [None]
    assert spec.canonical_batch_threads == [None]
    assert spec.enabled_lanes == [
        "canonical_all_on",
        "continuous_scheduler",
        "kernel_microbench",
        "quality_eval",
    ]
    assert spec.ablations == []


def test_load_gold_fast_suite_profile() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold-fast.yaml"))

    assert spec.profile_name == "gold-fast"
    assert spec.models == ["granite4:tiny-h", "qwen3.5:4b", "qwen3.5:9b"]
    assert spec.max_parallel_models == 1
    assert spec.affinity_policy == "repair_allowed"
    assert spec.tuning_contract_policy == "optional"
    assert spec.scenario_pack.measured_runs == 2
    assert spec.scenario_pack.thread_matrix_ubatch == [32]
    assert spec.scenario_pack.continuous_concurrency == [1, 2]
    assert spec.kernel_microbench.iterations == 48


def test_quality_lane_envs_are_all_on_only() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    lane_envs = benchmark_suite._quality_lane_envs(spec)

    assert lane_envs == [
        {"lane_id": "canonical_all_on", "ablation_id": "all_on", "env": {}}
    ]


def test_planned_lane_total_respects_enabled_lanes() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    assert benchmark_suite._planned_lane_total(spec) == 4


def test_load_calibrate_suite_profile() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_calibrate.yaml"))

    assert spec.profile_name == "calibrate"
    assert spec.host_contract_id == "auto"
    assert spec.affinity_policy == "certified_exact"
    assert spec.preflight_strictness == "optimize"
    assert spec.tuning_contract_policy == "generate"
    assert spec.calibration_search.search_kernel_iterations == 24


def test_execution_policy_requires_sequential_models() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))

    benchmark_suite._validate_execution_policy(spec)


def test_parse_cpu_list_accepts_ranges_and_values() -> None:
    parsed = benchmark_suite._parse_cpu_list("0-2,4,7-8")

    assert parsed == {0, 1, 2, 4, 7, 8}


def test_ensure_suite_affinity_expands_visible_threads(
    monkeypatch,
) -> None:  # noqa: ANN001
    spec = load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml"))
    affinity_state = {"cpus": {0}}

    monkeypatch.delenv("ANVIL_DISABLE_AUTO_AFFINITY_EXPAND", raising=False)
    monkeypatch.setattr(benchmark_suite, "_cpu_target_candidates", lambda: ["0-3"])
    monkeypatch.setattr(
        benchmark_suite.os,
        "sched_getaffinity",
        lambda _pid: set(affinity_state["cpus"]),
    )

    def _set_affinity(_pid: int, cpus: set[int]) -> None:
        affinity_state["cpus"] = set(cpus)

    monkeypatch.setattr(benchmark_suite.os, "sched_setaffinity", _set_affinity)

    payload = benchmark_suite._ensure_suite_affinity(spec)

    assert payload["attempted"] is True
    assert payload["expanded"] is True
    assert payload["repair_required"] is True
    assert payload["before"] == [0]
    assert payload["after"] == [0, 1, 2, 3]
    assert os.environ["ANVIL_SUITE_TARGET_CPUS"] == "0,1,2,3"


def test_gold_affinity_policy_refuses_repair(monkeypatch) -> None:  # noqa: ANN001
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))
    affinity_state = {"cpus": {0, 8}}

    monkeypatch.delenv("ANVIL_DISABLE_AUTO_AFFINITY_EXPAND", raising=False)
    monkeypatch.setattr(benchmark_suite, "_cpu_target_candidates", lambda: ["0-15"])
    monkeypatch.setattr(
        benchmark_suite.os,
        "sched_getaffinity",
        lambda _pid: set(affinity_state["cpus"]),
    )

    def _set_affinity(_pid: int, cpus: set[int]) -> None:
        affinity_state["cpus"] = set(cpus)

    monkeypatch.setattr(benchmark_suite.os, "sched_setaffinity", _set_affinity)

    payload = benchmark_suite._ensure_suite_affinity(spec)

    assert payload["repair_allowed"] is False
    assert payload["repair_required"] is True
    assert payload["attempted"] is False
    assert payload["before"] == [0, 8]
    assert payload["after"] == [0, 8]


def test_candidate_metrics_collects_phase10_pmu_summary() -> None:
    rows = [
        {
            "attempt_id": "a1",
            "lane_id": "calibration_stage1",
            "model_id": "qwen3.5:4b",
            "warmup": False,
            "thread_config": {"decode_threads": 8, "batch_threads": 8, "ubatch": 32},
            "throughput": {"decode_tps": 100.0},
            "latency": {"ttft_ms": 50.0},
            "runtime": {"graph_stage_ms": {"lm_head": 10.0}},
            "accepted_parallel_tokens": 5,
            "rejected_parallel_tokens": 1,
            "proposed_parallel_tokens": 6,
            "coherence": {"ok": True},
            "measurement": {
                "pmu": {
                    "observed": True,
                    "ipc": 1.9,
                    "cache_miss_rate": 0.04,
                    "context_switches": 9.0,
                    "cpu_migrations": 2.0,
                    "page_faults": 12.0,
                }
            },
        },
        {
            "attempt_id": "a2",
            "lane_id": "calibration_stage1",
            "model_id": "qwen3.5:4b",
            "warmup": False,
            "thread_config": {"decode_threads": 8, "batch_threads": 8, "ubatch": 32},
            "throughput": {"decode_tps": 98.0},
            "latency": {"ttft_ms": 52.0},
            "runtime": {"graph_stage_ms": {"lm_head": 12.0}},
            "accepted_parallel_tokens": 4,
            "rejected_parallel_tokens": 2,
            "proposed_parallel_tokens": 6,
            "coherence": {"ok": True},
            "measurement": {
                "pmu": {
                    "observed": True,
                    "ipc": 2.1,
                    "cache_miss_rate": 0.06,
                    "context_switches": 7.0,
                    "cpu_migrations": 1.0,
                    "page_faults": 14.0,
                }
            },
        },
    ]

    metrics = benchmark_suite._candidate_metrics(
        attempt_rows=rows,
        failure_rows=[],
        lane_id="calibration_stage1",
        model="qwen3.5:4b",
        candidate=(8, 8, 32),
    )

    assert metrics["ok"] is True
    assert metrics["pmu_summary"]["observed_runs"] == 2
    assert metrics["pmu_summary"]["ipc_median"] == 2.0
    assert metrics["pmu_summary"]["context_switches_median"] == 8.0
    assert metrics["pmu_summary"]["cpu_migrations_median"] == 1.5
    assert metrics["pmu_summary"]["cache_miss_rate_median"] == 0.05
    assert metrics["accepted_parallel_tokens_total"] == 9
    assert metrics["rejected_parallel_tokens_total"] == 3
    assert metrics["proposed_parallel_tokens_total"] == 12
    assert metrics["draft_acceptance_ratio"] == pytest.approx(0.75)
    assert metrics["speculative_attempts"] == 2
    assert metrics["speculative_summary"]["draft_acceptance_ratio"] == pytest.approx(
        0.75
    )


def test_rank_calibration_candidates_prefers_better_pmu_tiebreaks() -> None:
    ranked = benchmark_suite._rank_calibration_candidates(
        [
            {
                "ok": True,
                "decode_tps_median": 100.0,
                "decode_tps_cv_pct": 1.0,
                "ttft_ms_median": 50.0,
                "draft_acceptance_ratio": 0.5,
                "accepted_parallel_tokens_total": 8,
                "candidate": {"decode_threads": 8, "batch_threads": 8, "ubatch": 32},
                "pmu_summary": {
                    "observed_runs": 2,
                    "ipc_median": 1.6,
                    "cpu_migrations_median": 4.0,
                    "context_switches_median": 12.0,
                },
            },
            {
                "ok": True,
                "decode_tps_median": 100.0,
                "decode_tps_cv_pct": 1.0,
                "ttft_ms_median": 50.0,
                "draft_acceptance_ratio": 0.5,
                "accepted_parallel_tokens_total": 8,
                "candidate": {"decode_threads": 12, "batch_threads": 12, "ubatch": 32},
                "pmu_summary": {
                    "observed_runs": 2,
                    "ipc_median": 2.0,
                    "cpu_migrations_median": 1.0,
                    "context_switches_median": 6.0,
                },
            },
        ]
    )

    assert ranked[0]["candidate"]["decode_threads"] == 12


def test_rank_calibration_candidates_prefers_stronger_parallel_acceptance() -> None:
    ranked = benchmark_suite._rank_calibration_candidates(
        [
            {
                "ok": True,
                "decode_tps_median": 100.0,
                "decode_tps_cv_pct": 1.0,
                "ttft_ms_median": 50.0,
                "draft_acceptance_ratio": 0.25,
                "accepted_parallel_tokens_total": 4,
                "candidate": {"decode_threads": 8, "batch_threads": 8, "ubatch": 32},
                "pmu_summary": {
                    "observed_runs": 2,
                    "ipc_median": 2.0,
                    "cpu_migrations_median": 1.0,
                    "context_switches_median": 6.0,
                },
            },
            {
                "ok": True,
                "decode_tps_median": 100.0,
                "decode_tps_cv_pct": 1.0,
                "ttft_ms_median": 50.0,
                "draft_acceptance_ratio": 0.75,
                "accepted_parallel_tokens_total": 12,
                "candidate": {"decode_threads": 12, "batch_threads": 12, "ubatch": 32},
                "pmu_summary": {
                    "observed_runs": 2,
                    "ipc_median": 1.5,
                    "cpu_migrations_median": 9.0,
                    "context_switches_median": 14.0,
                },
            },
        ]
    )

    assert ranked[0]["candidate"]["decode_threads"] == 12


def test_ablation_deltas_are_measured_against_canonical_all_on() -> None:
    quality_payload = {
        "confidence": [
            {
                "model": "qwen3.5:4b",
                "ablation_id": "all_on",
                "mean_token_confidence": 0.8,
            },
            {
                "model": "qwen3.5:4b",
                "ablation_id": "timecrystal_off",
                "mean_token_confidence": 0.72,
            },
        ],
        "perplexity": [
            {"model": "qwen3.5:4b", "ablation_id": "all_on", "perplexity": 10.0},
            {
                "model": "qwen3.5:4b",
                "ablation_id": "timecrystal_off",
                "perplexity": 12.0,
            },
        ],
    }
    attempt_rows = [
        {
            "model_id": "qwen3.5:4b",
            "lane_id": "canonical_all_on",
            "ablation_id": "all_on",
            "warmup": False,
            "throughput": {"decode_tps": 100.0},
            "latency": {"ttft_ms": 50.0},
        },
        {
            "model_id": "qwen3.5:4b",
            "lane_id": "native_ablations",
            "ablation_id": "timecrystal_off",
            "warmup": False,
            "throughput": {"decode_tps": 90.0},
            "latency": {"ttft_ms": 60.0},
        },
    ]

    deltas = benchmark_suite._ablation_deltas(quality_payload, attempt_rows)

    assert len(deltas) == 1
    assert deltas[0]["model"] == "qwen3.5:4b"
    assert deltas[0]["ablation_id"] == "timecrystal_off"
    assert round(float(deltas[0]["decode_tps_delta_pct"]), 2) == -10.0
    assert round(float(deltas[0]["perplexity_delta_pct"]), 2) == 20.0


def test_suite_layout_contract_lists_required_artifacts(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "run123")
    ensure_suite_layout(layout)

    required = required_suite_artifacts(layout)

    assert layout.summary_json in required
    assert layout.assurance_plan_json in required
    assert layout.closure_matrix_json in required
    assert layout.change_manifest_json in required
    assert layout.traceability_json in required
    assert layout.evidence_bundle_json in required
    assert layout.runtime_gates_json in required
    assert layout.telemetry_contract_json in required
    assert layout.chronicle_json in required
    assert layout.quality_summary_json in required
    assert layout.kernel_summary_json in required
    assert layout.triage_json in required
    assert layout.report_md in required
    assert layout.executive_summary_md in required
    assert layout.index_json in required
    assert layout.agent_handoff_json in required
    assert layout.metrics_rollup_json in required
    assert layout.run_ledger_json in required
    assert layout.topology_passport_json in required
    assert layout.variance_budget_json in required
    assert layout.baseline_lineage_json in required
    assert layout.closure_result_json in required
    assert layout.traceability_graph_json in required
    assert layout.run_capsule_manifest_json in required
    assert layout.saguaro_verify_json in required
    assert layout.events_ndjson in required
    assert layout.terminal_transcript_log in required


def test_attempt_artifact_paths_include_phase9_replay_refs(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "run123")
    ensure_suite_layout(layout)

    root, artifact_paths = benchmark_suite._attempt_artifact_paths(layout, "attempt-1")

    assert root.endswith("attempt-1")
    assert artifact_paths["evidence_capsule"].endswith("evidence_capsule.json")
    assert artifact_paths["checkpoint_metadata"] == str(layout.checkpoint_json)
    assert artifact_paths["flight_recorder_timeline"] == str(layout.events_ndjson)
    assert artifact_paths["terminal_transcript"] == str(layout.terminal_transcript_log)


def test_capture_runtime_provenance_exposes_phase0_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "12")
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_PLACES", "cores")
    monkeypatch.setattr(
        provenance_capture,
        "_run",
        lambda cmd, cwd=None: "abc123" if cmd[:2] == ["git", "rev-parse"] else "",
    )
    monkeypatch.setattr(provenance_capture, "_cpu_model", lambda: "Unit Test CPU")
    monkeypatch.setattr(
        provenance_capture,
        "_cpuinfo_field",
        lambda field_name: "0x42" if field_name == "microcode" else "",
    )
    monkeypatch.setattr(provenance_capture, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(
        provenance_capture, "_transparent_hugepage_mode", lambda: "madvise"
    )
    monkeypatch.setattr(
        provenance_capture,
        "_memory_snapshot",
        lambda: {
            "mem_total_kb": 1024,
            "mem_available_kb": 512,
            "hugepages_total": 8,
            "hugepages_free": 4,
            "hugepage_size_kb": 2048,
            "memory_speed_mt_s": 3200,
            "memory_speed_source": "dmidecode",
        },
    )
    monkeypatch.setattr(
        provenance_capture,
        "_numa_policy_snapshot",
        lambda: {
            "available": True,
            "policy": "preferred",
            "preferred": "0",
            "physcpubind": "0-7",
            "cpubind": "0",
            "nodebind": "0",
            "membind": "0",
            "interleave": "",
            "source": "numactl",
        },
    )
    monkeypatch.setattr(provenance_capture.platform, "node", lambda: "unit-host")
    monkeypatch.setattr(provenance_capture.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(provenance_capture.platform, "platform", lambda: "Linux-test")
    monkeypatch.setattr(provenance_capture.platform, "release", lambda: "6.8.0-test")
    monkeypatch.setattr(provenance_capture.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(
        provenance_capture.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}
    )

    payload = provenance_capture.capture_runtime_provenance(tmp_path)

    assert payload["git"]["commit"] == "abc123"
    assert payload["host"]["kernel_release"] == "6.8.0-test"
    assert payload["host"]["microcode_version"] == "0x42"
    assert payload["host"]["cpu_governor"] == "performance"
    assert payload["host"]["transparent_hugepage_mode"] == "madvise"
    assert payload["threading"]["omp_num_threads"] == "12"
    assert payload["threading"]["omp_proc_bind"] == "close"
    assert payload["threading"]["omp_places"] == "cores"
    assert payload["threading"]["visible_threads"] == 4
    assert payload["memory"]["memory_speed_mt_s"] == 3200
    assert payload["memory"]["memory_speed_source"] == "dmidecode"
    assert payload["numa"]["policy"] == "preferred"
    assert payload["numa"]["membind"] == "0"


def test_canonical_thread_tuple_uses_host_derived_defaults() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))

    tuples = benchmark_suite._canonical_thread_tuples(
        spec,
        {
            "host": {
                "logical_cpus": 16,
                "visible_threads": 16,
            }
        },
    )

    assert tuples == [(16, 16, 32)]


def test_max_affinity_thread_tuple_uses_visible_threads() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml"))

    thread_tuple = benchmark_suite._max_affinity_thread_tuple(
        spec,
        {
            "host": {
                "logical_cpus": 32,
                "visible_threads": 16,
            }
        },
    )

    assert thread_tuple == (16, 16, 32)


def test_run_continuous_surface_uses_native_thread_contract(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml")),
        models=["granite4:tiny-h"],
    )
    layout = resolve_suite_layout(tmp_path, "continuous-contract")
    ensure_suite_layout(layout)
    commands: list[list[str]] = []

    def _fake_run_logged_subprocess(*, cmd, cwd, env, **kwargs):  # noqa: ANN001, ARG001
        commands.append(list(cmd))
        out_path = Path(cmd[cmd.index("--json-out") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": benchmark_suite.CONTINUOUS_SCHEMA_VERSION,
                    "model": "granite4:tiny-h",
                    "results": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        benchmark_suite, "run_logged_subprocess", _fake_run_logged_subprocess
    )

    payload = benchmark_suite._run_continuous_surface(
        repo_root=Path.cwd(),
        layout=layout,
        spec=spec,
        runtime_payload={"host": {"logical_cpus": 16, "visible_threads": 16}},
        model_thread_overrides={"granite4:tiny-h": (8, 6, 32)},
    )

    assert payload == [
        {
            "schema_version": benchmark_suite.CONTINUOUS_SCHEMA_VERSION,
            "model": "granite4:tiny-h",
            "results": [],
        }
    ]
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--max-new-tokens") + 1] == "24"
    assert command[command.index("--context-length") + 1] == "1024"
    assert command[command.index("--decode-threads") + 1] == "8"
    assert command[command.index("--batch-threads") + 1] == "6"
    assert command[command.index("--ubatch") + 1] == "32"


def test_summary_schema_accepts_suite_fields() -> None:
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "suite-run",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": False,
        "failure_count": 1,
        "failure_counts": {"total": 1, "gate_failure": 1, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 0,
        "planned_attempts": 0,
        "pass": False,
        "quality": {},
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "certification_state": "certified_fail",
        "run_exit_reason": "suite_fail",
        "terminal_state": "completed_fail",
        "last_successful_lane": "quality_eval",
    }

    schema = json.loads(
        Path("audit/schemas/summary.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(summary, schema)


def test_summary_schema_accepts_calibration_bundle() -> None:
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "suite-run",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": False,
        "failure_count": 1,
        "failure_counts": {"total": 1, "gate_failure": 1, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 0,
        "planned_attempts": 0,
        "pass": False,
        "quality": {},
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "calibration": {
            "contracts": {"granite4:tiny-h": {"path": "/tmp/contract.json"}}
        },
        "certification_state": "certified_fail",
        "run_exit_reason": "suite_fail",
        "terminal_state": "completed_fail",
        "last_successful_lane": "calibration",
    }

    schema = json.loads(
        Path("audit/schemas/summary.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(summary, schema)


def test_summary_schema_accepts_quality_governance_bundle() -> None:
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "suite-run",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": True,
        "failure_count": 0,
        "failure_counts": {"total": 0, "gate_failure": 0, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 1,
        "planned_attempts": 1,
        "pass": True,
        "quality": {},
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "quality_governance": {
            "schema_version": "native_qsg_suite.acceptance_governance.v1",
            "passed": True,
            "issues": [],
            "blocked_items": [],
            "artifact_completeness": {"passed": True, "checks": []},
            "quality_evidence": {},
            "mode_coverage": {},
        },
        "certification_state": "certified_pass",
        "run_exit_reason": "suite_pass",
        "terminal_state": "completed_pass",
        "last_successful_lane": "quality_eval",
    }

    schema = json.loads(
        Path("audit/schemas/summary.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(summary, schema)


def test_summary_schema_accepts_assurance_bundle() -> None:
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "suite-run",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": True,
        "failure_count": 0,
        "failure_counts": {"total": 0, "gate_failure": 0, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 1,
        "planned_attempts": 1,
        "pass": True,
        "quality": {},
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "assurance": {
            "strict_native_decode": {
                "required": True,
                "force_parallel_decode": True,
                "forbid_autoregressive_fallback": True,
                "attempt_count": 1,
                "observed_generation_modes": ["block_diffusion"],
                "observed_benchmark_labels": ["block_diffusion_candidate"],
                "observed_non_ar_modes": ["block_diffusion"],
                "observed_ar_modes": [],
                "observed_ar_attempt_ids": [],
                "passed": True,
                "issues": [],
            }
        },
        "certification_state": "certified_pass",
        "run_exit_reason": "suite_pass",
        "terminal_state": "completed_pass",
        "last_successful_lane": "quality_eval",
    }

    schema = json.loads(
        Path("audit/schemas/summary.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(summary, schema)


def test_summary_schema_accepts_control_plane_bundle() -> None:
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "suite-run",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": True,
        "failure_count": 0,
        "failure_counts": {"total": 0, "gate_failure": 0, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 1,
        "planned_attempts": 1,
        "pass": True,
        "quality": {},
        "memory_replay": {"state": "skipped"},
        "comparisons": {"compare_to": "latest_compatible"},
        "baseline_run_id": "",
        "prompt_hash": "abc",
        "prompt_contract_hash": "def",
        "memory_snapshot_hash": "ghi",
        "feature_toggle_hash": "jkl",
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "control_plane": {
            "profile_name": "silver",
            "completed_lanes": ["canonical_all_on"],
            "topology_hash": "abc",
            "cohort_key": "def",
            "variance_within_budget": True,
            "history_aware_comparison": False,
            "closure_pass": True,
        },
        "topology_passport": {"topology_hash": "abc"},
        "variance_budget": {"overall": {"within_budget": True}},
        "baseline_lineage": {"comparable": False},
        "closure_result": {"overall_pass": True},
        "spc_report": {"status": "stable"},
        "traceability_graph": {"question_counts": {"intended": 1}},
        "saguaro_verify": {"status": "ok"},
        "advisory_bundle": {"stage_graph": {"changed": False}},
        "black_box": {"path": "telemetry/black_box_manifest.json"},
        "capsule_archive": {"path": "artifacts/run_capsule.tar.gz"},
        "certification_state": "certified_pass",
        "run_exit_reason": "suite_pass",
        "terminal_state": "completed_pass",
        "last_successful_lane": "quality_eval",
    }

    schema = json.loads(
        Path("audit/schemas/summary.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(summary, schema)


def test_control_plane_materializes_typed_artifacts(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "control-plane")
    ensure_suite_layout(layout)
    (tmp_path / "venv" / "bin").mkdir(parents=True)
    (tmp_path / "venv" / "bin" / "saguaro").write_text("#!/bin/sh\n", encoding="utf-8")

    spec = load_suite_profile(REPO_ROOT / "audit/profiles/native_qsg_silver.yaml")
    manifest = {
        "schema_version": "native_qsg_suite.manifest.v1",
        "run_id": layout.run_id,
        "profile": spec.profile_name,
        "models": list(spec.models),
    }
    launch_runtime = {
        "host": {
            "hostname": "host",
            "machine": "x86_64",
            "platform": "linux",
            "cpu_model": "cpu",
            "logical_cpus": 8,
            "visible_threads": 8,
            "host_fingerprint": "fingerprint",
        }
    }
    preflight_payload = {
        "runtime": launch_runtime,
        "host": dict(launch_runtime["host"]),
        "memory": {
            "meminfo_artifact": "telemetry/meminfo.txt",
            "meminfo": {"MemAvailable": 1024},
        },
        "saguaro": {"ok": True, "artifact": {"path": "telemetry/saguaro_health.json"}},
        "launch_affinity": list(range(8)),
        "post_adjustment_affinity": list(range(8)),
        "repair_allowed": True,
        "repair_attempted": False,
        "repair_required": False,
        "cpu_governor": "performance",
        "thp_mode": "madvise",
        "perf_event_paranoid": "1",
        "perf": {"available": True},
    }
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": layout.run_id,
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [
            {
                "model": "granite4:tiny-h",
                "decode_tps_p50": 10.0,
                "ttft_ms_p95": 100.0,
                "decode_time_accounted_pct": 99.0,
                "run_stability": {
                    "decode_tps_cv_pct": 5.0,
                    "ttft_ms_cv_pct": 4.0,
                },
            }
        ],
        "overall_pass": True,
        "assurance": {"runtime_gates_passed": True, "missing_artifacts": []},
    }
    comparisons = {
        "baseline": {
            "run_id": "baseline-1",
            "models": [
                {
                    "model": "granite4:tiny-h",
                    "decode_tps_p50": 9.0,
                    "ttft_ms_p95": 110.0,
                }
            ],
        }
    }
    assurance_plan = {"required_runtime_gates": []}

    payload = materialize_control_plane_artifacts(
        repo_root=tmp_path,
        layout=layout,
        spec=spec,
        manifest=manifest,
        launch_runtime=launch_runtime,
        preflight_payload=preflight_payload,
        summary=summary,
        comparisons=comparisons,
        assurance_plan=assurance_plan,
        completed_lanes=["canonical_all_on"],
        verify_runner=lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='{"status":"pass","violations":[],"count":0}',
            stderr="",
        ),
    )

    assert layout.topology_passport_json.exists()
    assert layout.variance_budget_json.exists()
    assert layout.baseline_lineage_json.exists()
    assert layout.run_ledger_json.exists()
    assert layout.mission_receipts_json.exists()
    assert layout.closure_result_json.exists()
    assert layout.traceability_graph_json.exists()
    assert layout.advisory_bundle_json.exists()
    assert layout.run_capsule_manifest_json.exists()
    assert layout.saguaro_verify_json.exists()
    assert payload["summary"]["topology_hash"]
    assert payload["spc_report"]["status"] in {"stable", "insufficient_history"}
    assert payload["advisory_bundle"]["stage_graph"]["status"] in {"covered", "blocked"}


def test_reduce_runtime_state_tracks_receipts_and_checkpoint() -> None:
    state = reduce_runtime_state(
        [
            {
                "schema_version": "native_qsg_suite.event.v1",
                "run_id": "run-1",
                "timestamp": "2026-03-11T00:00:00Z",
                "level": "debug",
                "source": "benchmark_suite",
                "event_type": "suite_checkpoint",
                "phase": "canonical_all_on",
                "lane": "canonical_all_on",
                "attempt_id": "",
                "model": "",
                "message": "checkpoint",
                "payload": {
                    "completed_lanes": ["canonical_all_on"],
                    "completed_attempt_ids": ["attempt-1"],
                    "last_successful_lane": "canonical_all_on",
                    "run_exit_reason": "in_progress",
                },
            },
            {
                "schema_version": "native_qsg_suite.event.v1",
                "run_id": "run-1",
                "timestamp": "2026-03-11T00:00:01Z",
                "level": "info",
                "source": "benchmark_suite",
                "event_type": "mission_node_receipt",
                "phase": "native_surface",
                "lane": "canonical_all_on",
                "attempt_id": "",
                "model": "",
                "message": "canonical_all_on:completed",
                "payload": {
                    "node_id": "canonical_all_on",
                    "phase": "native_surface",
                    "kind": "lane",
                    "status": "completed",
                    "blocking": True,
                },
            },
        ]
    )

    assert state["completed_lanes"] == ["canonical_all_on"]
    assert state["completed_attempt_ids"] == ["attempt-1"]
    assert state["node_receipts"][0]["node_id"] == "canonical_all_on"


def test_suite_event_logger_exports_event_store(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    logger = SuiteEventLogger(
        run_id="suite-run",
        run_root=run_root,
        events_path=run_root / "events.ndjson",
        transcript_path=run_root / "terminal.log",
        console_log_path=run_root / "console.log",
    )
    logger.start()
    logger.emit(
        level="info",
        source="benchmark_suite",
        event_type="suite_state",
        message="initialized",
        phase="initialized",
        payload={"state": "initialized"},
    )
    logger.close()

    export_payload = json.loads(
        (run_root / "telemetry" / "event_store_export.json").read_text(encoding="utf-8")
    )
    assert export_payload["run_id"] == "suite-run"
    assert export_payload["events"][0]["event_type"] == "suite_state"


def test_memory_replay_lane_skips_without_cases(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "memory-replay")
    ensure_suite_layout(layout)
    spec = load_suite_profile(REPO_ROOT / "audit/profiles/native_qsg_silver.yaml")

    payload = benchmark_suite._run_memory_replay_lane(
        repo_root=REPO_ROOT,
        layout=layout,
        spec=spec,
    )

    assert payload["state"] == "skipped"
    assert layout.memory_replay_summary_json.exists()


def test_persist_summary_bundle_writes_failed_summary_fallback(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "summary-fallback")
    ensure_suite_layout(layout)
    summary = {
        "schema_version": "native_qsg_audit.v3",
        "run_id": "summary-fallback",
        "generated_at": "2026-03-09T00:00:00Z",
        "models": [],
        "top_stage_hotspots": [],
        "stage_origin_map": {},
        "baseline_floors": {},
        "overall_pass": True,
        "failure_count": 0,
        "failure_counts": {"total": 0, "gate_failure": 0, "execution_failure": 0},
        "failed_attempt_ids": [],
        "completed_attempts": 1,
        "planned_attempts": 1,
        "pass": True,
        "quality": {},
        "ablation_deltas": [],
        "kernel_hotspots": [],
        "host_compliance": {},
        "agent_triage": {},
        "continuous": {},
        "certification_state": "certified_pass",
        "run_exit_reason": "suite_pass",
        "terminal_state": "completed_pass",
        "last_successful_lane": "quality_eval",
        "unexpected": True,
    }

    result = benchmark_suite._persist_summary_bundle(
        layout=layout,
        summary=summary,
        comparisons={"schema_version": "native_qsg_suite.comparisons.v1"},
    )

    persisted_summary = json.loads(layout.summary_json.read_text(encoding="utf-8"))
    failed_bundle = json.loads(layout.summary_failed_json.read_text(encoding="utf-8"))

    assert result["ok"] is False
    assert persisted_summary["terminal_state"] == "internal_error"
    assert persisted_summary["run_exit_reason"] == "summary_persistence_failed"
    assert failed_bundle["errors"]
    assert failed_bundle["raw_summary"]["unexpected"] is True


def test_checkpoint_schema_accepts_suite_checkpoint_fields() -> None:
    checkpoint = {
        "schema_version": "native_qsg_suite.checkpoint.v2",
        "run_id": "suite-run",
        "updated_at": "2026-03-09T00:00:00Z",
        "completed_lanes": ["canonical_all_on"],
        "completed_attempt_ids": ["attempt-1"],
        "run_exit_reason": "in_progress",
        "last_successful_lane": "canonical_all_on",
    }

    schema = json.loads(
        Path("audit/schemas/checkpoint.schema.json").read_text(encoding="utf-8")
    )
    import jsonschema

    jsonschema.validate(checkpoint, schema)


def test_main_writes_failed_preflight_terminal_state(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        benchmark_suite,
        "capture_runtime_provenance",
        lambda _repo_root: {"host": {"host_fingerprint": "fp"}},
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_ensure_suite_affinity",
        lambda _spec: {
            "repair_allowed": False,
            "attempted": False,
            "repair_required": False,
            "before": [],
            "after": [],
            "target": [],
            "error": "",
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "parse_args",
        lambda: argparse.Namespace(
            profile="gold",
            run_id="suite-preflight-fail",
            resume=False,
            out_root=str(tmp_path),
            experiment=None,
            ui_mode="plain",
            log_level="trace",
        ),
    )
    monkeypatch.setattr(
        benchmark_suite,
        "run_preflight",
        lambda **_: SimpleNamespace(
            ok=False,
            payload={
                "ok": False,
                "failures": ["perf_unavailable"],
                "runtime": {},
                "certification_state": "certified_candidate",
                "remediations": {"perf_unavailable": "fix perf"},
            },
        ),
    )

    rc = benchmark_suite.main()
    assert rc == 1

    layout = resolve_suite_layout(tmp_path, "suite-preflight-fail")
    status = json.loads(layout.suite_status_json.read_text(encoding="utf-8"))
    checkpoint = json.loads(layout.checkpoint_json.read_text(encoding="utf-8"))
    summary = json.loads(layout.summary_json.read_text(encoding="utf-8"))

    assert status["state"] == "failed_preflight"
    assert status["run_exit_reason"] == "preflight_failed"
    assert checkpoint["run_exit_reason"] == "preflight_failed"
    assert summary["terminal_state"] == "failed_preflight"
    assert summary["certification_state"] == "certified_fail"
    assert layout.index_json.exists()
    assert layout.events_ndjson.exists()
    assert layout.terminal_transcript_log.exists()


def test_main_environment_snapshot_persists_phase0_benchmark_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        benchmark_suite,
        "capture_runtime_provenance",
        lambda _repo_root: {"host": {"host_fingerprint": "fp"}},
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_ensure_suite_affinity",
        lambda _spec: {
            "repair_allowed": False,
            "attempted": False,
            "repair_required": False,
            "before": [0, 1],
            "after": [0, 1],
            "target": [0, 1],
            "error": "",
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "parse_args",
        lambda: argparse.Namespace(
            profile="gold",
            run_id="suite-phase0-env",
            resume=False,
            out_root=str(tmp_path),
            experiment=None,
            ui_mode="plain",
            log_level="trace",
        ),
    )
    monkeypatch.setattr(
        benchmark_suite,
        "run_preflight",
        lambda **_: SimpleNamespace(
            ok=False,
            payload={
                "ok": False,
                "failures": ["perf_unavailable"],
                "runtime": {"host": {"host_fingerprint": "fp"}},
                "host": {"host_fingerprint_expected": "fp"},
                "models": {
                    "granite4:tiny-h": {
                        "quant_variant": "manifest-pinned",
                        "digest": "sha256:model",
                    }
                },
                "cpu_governor": "performance",
                "thp_mode": "madvise",
                "perf_event_paranoid": "-1",
                "memory": {"meminfo": {"MemAvailable": 123}},
                "lscpu": {"ok": True, "artifact": "telemetry/lscpu.json"},
                "launch_affinity": [0, 1],
                "post_adjustment_affinity": [0, 1],
                "certification_state": "certified_candidate",
                "benchmark_metadata": {
                    "git_sha": "abc123",
                    "models": [
                        {
                            "model": "granite4:tiny-h",
                            "quantization_profile": "manifest-pinned",
                            "kv_cache_quantization": "q8",
                        }
                    ],
                    "host": {
                        "cpu_model": "Unit Test CPU",
                        "kernel_release": "6.8.0-test",
                    },
                },
                "remediations": {"perf_unavailable": "fix perf"},
            },
        ),
    )

    rc = benchmark_suite.main()
    assert rc == 1

    layout = resolve_suite_layout(tmp_path, "suite-phase0-env")
    environment = json.loads(layout.environment_json.read_text(encoding="utf-8"))

    assert environment["cpu_governor"] == "performance"
    assert environment["thp_mode"] == "madvise"
    assert environment["perf_event_paranoid"] == "-1"
    assert environment["benchmark_metadata"]["git_sha"] == "abc123"
    assert environment["benchmark_metadata"]["models"][0]["quantization_profile"] == (
        "manifest-pinned"
    )
    assert environment["benchmark_metadata"]["models"][0]["kv_cache_quantization"] == (
        "q8"
    )
    assert environment["memory"]["meminfo"]["MemAvailable"] == 123
    assert environment["lscpu"]["artifact"] == "telemetry/lscpu.json"


def test_main_marks_interrupted_terminal_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        benchmark_suite,
        "capture_runtime_provenance",
        lambda _repo_root: {"host": {"host_fingerprint": "fp"}},
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_ensure_suite_affinity",
        lambda _spec: {
            "repair_allowed": False,
            "attempted": False,
            "repair_required": False,
            "before": [],
            "after": [],
            "target": [],
            "error": "",
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "parse_args",
        lambda: argparse.Namespace(
            profile="gold",
            run_id="suite-interrupted",
            resume=False,
            out_root=str(tmp_path),
            experiment=None,
            ui_mode="plain",
            log_level="trace",
        ),
    )
    monkeypatch.setattr(
        benchmark_suite,
        "run_preflight",
        lambda **_: SimpleNamespace(
            ok=True,
            payload={
                "ok": True,
                "failures": [],
                "runtime": {"host": {}},
                "certification_state": "certified_candidate",
            },
        ),
    )

    def _raise_interrupt(**_: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark_suite, "_lane_attempts", _raise_interrupt)

    rc = benchmark_suite.main()
    assert rc == 130

    layout = resolve_suite_layout(tmp_path, "suite-interrupted")
    status = json.loads(layout.suite_status_json.read_text(encoding="utf-8"))
    summary = json.loads(layout.summary_json.read_text(encoding="utf-8"))

    assert status["state"] == "interrupted_incomplete"
    assert status["run_exit_reason"] == "keyboard_interrupt"
    assert summary["terminal_state"] == "interrupted_incomplete"
    assert summary["certification_state"] == "certified_fail"


def test_lane_attempts_uses_provided_thread_matrix_for_canonical(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))
    layout = resolve_suite_layout(tmp_path, "lane-thread-check")
    ensure_suite_layout(layout)
    captured: list[tuple[int | None, int | None, int]] = []

    def _stub_record_attempt(**kwargs):  # noqa: ANN001
        attempt_spec = kwargs["spec"]
        captured.append(
            (
                attempt_spec.decode_threads,
                attempt_spec.batch_threads,
                int(attempt_spec.ubatch or 0),
            )
        )
        kwargs["completed_attempt_ids"].add(attempt_spec.attempt_id)

    monkeypatch.setattr(benchmark_suite, "_record_attempt", _stub_record_attempt)

    benchmark_suite._lane_attempts(
        repo_root=Path("."),
        layout=layout,
        spec=spec,
        lane_id="canonical_all_on",
        ablation_id="all_on",
        env_overrides={},
        prompt=spec.scenario_pack.canonical_prompt,
        warmup_runs=0,
        measured_runs=1,
        thread_matrix=[(8, 8, 32)],
        use_thread_matrix=False,
        attempt_rows=[],
        failure_rows=[],
        completed_attempt_ids=set(),
        resume_completed=set(),
    )

    assert captured
    assert captured[0] == (8, 8, 32)


def test_lane_attempts_uses_model_thread_overrides(monkeypatch, tmp_path: Path) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))
    layout = resolve_suite_layout(tmp_path, "lane-thread-overrides")
    ensure_suite_layout(layout)
    captured: list[tuple[str, int | None, int | None, int]] = []

    def _stub_record_attempt(**kwargs):  # noqa: ANN001
        attempt_spec = kwargs["spec"]
        captured.append(
            (
                attempt_spec.model,
                attempt_spec.decode_threads,
                attempt_spec.batch_threads,
                int(attempt_spec.ubatch or 0),
            )
        )
        kwargs["completed_attempt_ids"].add(attempt_spec.attempt_id)

    monkeypatch.setattr(benchmark_suite, "_record_attempt", _stub_record_attempt)

    benchmark_suite._lane_attempts(
        repo_root=Path("."),
        layout=layout,
        spec=spec,
        lane_id="canonical_all_on",
        ablation_id="all_on",
        env_overrides={},
        prompt=spec.scenario_pack.canonical_prompt,
        warmup_runs=0,
        measured_runs=1,
        thread_matrix=[(16, 16, 32)],
        use_thread_matrix=False,
        attempt_rows=[],
        failure_rows=[],
        completed_attempt_ids=set(),
        resume_completed=set(),
        model_thread_overrides={
            "granite4:tiny-h": (12, 12, 32),
            "qwen3.5:4b": (8, 8, 16),
            "qwen3.5:9b": (16, 16, 64),
        },
    )

    assert captured[0] == ("granite4:tiny-h", 12, 12, 32)
    assert captured[1] == ("qwen3.5:4b", 8, 8, 16)
    assert captured[2] == ("qwen3.5:9b", 16, 16, 64)


def test_gold_preflight_fails_when_launch_affinity_is_constrained(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_gold.yaml"))
    launch_runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "0bea3bede3a8874a",
            "logical_cpus": 16,
            "visible_threads": 2,
        },
    }
    runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "0bea3bede3a8874a",
            "logical_cpus": 16,
            "visible_threads": 16,
        },
    }

    monkeypatch.setattr(
        suite_preflight, "capture_runtime_provenance", lambda _repo_root: runtime
    )
    monkeypatch.setattr(
        suite_preflight,
        "capture_model_provenance",
        lambda model: {
            "model": model,
            "digest_validated": True,
            "strict_native_supported": True,
            "digest": f"sha256:{benchmark_suite._safe_name(model)}",
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_saguaro_probe",
        lambda _repo_root, _telemetry_dir, strictness=None: {"ok": True},
    )
    monkeypatch.setattr(
        suite_preflight, "_perf_probe", lambda _tmp_dir: {"available": True}
    )
    monkeypatch.setattr(suite_preflight, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(suite_preflight, "_read_text", lambda _path: "")
    monkeypatch.setattr(
        suite_preflight,
        "_run",
        lambda _cmd, cwd=None, telemetry_dir=None, label=None: (0, "{}", ""),
    )
    monkeypatch.setattr(
        suite_preflight,
        "load_tuning_contract",
        lambda repo_root, host_fingerprint, model: (
            {
                "host_contract_id": "0bea3bede3a8874a",
                "host_fingerprint": host_fingerprint,
                "model": model,
                "model_digest": f"sha256:{benchmark_suite._safe_name(model)}",
                "profile_schema_version": spec.schema_version,
                "benchmark_harness_hash": suite_preflight.benchmark_harness_hash(
                    repo_root
                ),
                "contract_hashes": {
                    "host_contract_sha256": suite_preflight.sha256_file(
                        Path("audit/contracts/hosts/0bea3bede3a8874a.yaml")
                    )
                },
                "thread_config": {
                    "decode_threads": 16,
                    "batch_threads": 16,
                    "ubatch": 32,
                },
            },
            tmp_path / f"{benchmark_suite._safe_name(model)}.json",
        ),
    )

    result = suite_preflight.run_preflight(
        repo_root=Path("."),
        spec=spec,
        telemetry_dir=tmp_path,
        launch_runtime=launch_runtime,
        affinity_adjustment={
            "repair_allowed": False,
            "attempted": False,
            "repair_required": True,
            "before": [0, 8],
            "after": [0, 8],
            "target": list(range(16)),
            "error": "",
        },
    )

    assert result.ok is False
    assert "launch_affinity_constrained" in result.payload["failures"]
    assert result.payload["repair_required"] is True
    assert result.payload["launch_affinity"] == [0, 8]
    assert result.payload["post_adjustment_affinity"] == [0, 8]


def test_preflight_builds_phase0_benchmark_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml")),
        host_contract_id=None,
        tuning_contract_policy="optional",
    )
    launch_runtime = {
        "env": {
            "ANVIL_KV_QUANT": "q8",
            "OMP_NUM_THREADS": "8",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        },
        "threading": {
            "omp_num_threads": "8",
            "omp_proc_bind": "close",
            "omp_places": "cores",
            "visible_threads": 8,
        },
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
        "python": {"virtual_env": "/venv"},
    }
    runtime = {
        "git": {"commit": "abc123"},
        "env": {
            "ANVIL_KV_QUANT": "q8",
            "OMP_NUM_THREADS": "8",
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        },
        "threading": {
            "omp_num_threads": "8",
            "omp_proc_bind": "close",
            "omp_places": "cores",
            "visible_threads": 8,
        },
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
            "cpu_model": "Unit Test CPU",
            "machine": "x86_64",
            "platform": "Linux-test",
            "kernel_release": "6.8.0-test",
            "microcode_version": "0x42",
            "cpu_governor": "performance",
            "transparent_hugepage_mode": "madvise",
        },
        "memory": {
            "memory_speed_mt_s": 3200,
            "memory_speed_source": "dmidecode",
        },
        "numa": {
            "policy": "preferred",
            "membind": "0",
            "cpubind": "0",
            "physcpubind": "0-7",
        },
        "python": {"virtual_env": "/venv"},
    }

    monkeypatch.setattr(
        suite_preflight, "capture_runtime_provenance", lambda _repo_root: runtime
    )
    monkeypatch.setattr(
        suite_preflight,
        "capture_model_provenance",
        lambda model: {
            "model": model,
            "digest": f"sha256:{benchmark_suite._safe_name(model)}",
            "digest_validated": True,
            "quant_variant": "manifest-pinned",
            "strict_native_supported": True,
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_saguaro_probe",
        lambda _repo_root, _telemetry_dir, strictness=None: {"ok": True},
    )
    monkeypatch.setattr(
        suite_preflight, "_perf_probe", lambda _tmp_dir: {"available": True}
    )
    monkeypatch.setattr(suite_preflight, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(suite_preflight, "_read_text", lambda _path: "")
    monkeypatch.setattr(
        suite_preflight,
        "_run",
        lambda _cmd, cwd=None, telemetry_dir=None, label=None: (0, "{}", ""),
    )

    result = suite_preflight.run_preflight(
        repo_root=Path("."),
        spec=spec,
        telemetry_dir=tmp_path,
        launch_runtime=launch_runtime,
        affinity_adjustment={
            "repair_allowed": True,
            "attempted": False,
            "repair_required": False,
            "before": list(range(8)),
            "after": list(range(8)),
            "target": list(range(8)),
            "error": "",
        },
    )

    metadata = result.payload["benchmark_metadata"]

    assert result.ok is True
    assert metadata["git_sha"] == "abc123"
    assert metadata["launch"]["threading"]["omp_num_threads"] == "8"
    assert metadata["runtime"]["threading"]["omp_proc_bind"] == "close"
    assert metadata["host"]["cpu_model"] == "Unit Test CPU"
    assert metadata["host"]["kernel_release"] == "6.8.0-test"
    assert metadata["host"]["microcode_version"] == "0x42"
    assert metadata["host"]["huge_page_mode"] == "madvise"
    assert metadata["host"]["numa_policy"] == "preferred"
    assert metadata["host"]["memory_speed_mt_s"] == 3200
    assert metadata["models"][0]["quantization_profile"] == "manifest-pinned"
    assert metadata["models"][0]["kv_cache_quantization"] == "q8"


def test_preflight_emits_live_events(monkeypatch, tmp_path: Path) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml"))
    host_contract_path = tmp_path / "host123.yaml"
    host_contract_path.write_text(
        "contract_id: host123\nhost_fingerprint: host123\nrequired_visible_threads: 8\n",
        encoding="utf-8",
    )
    launch_runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
    }
    runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
    }
    monkeypatch.setattr(
        suite_preflight, "capture_runtime_provenance", lambda _repo_root: runtime
    )
    monkeypatch.setattr(
        suite_preflight,
        "capture_model_provenance",
        lambda model: {
            "model": model,
            "digest_validated": True,
            "strict_native_supported": True,
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_saguaro_probe",
        lambda _repo_root, _telemetry_dir, strictness=None: {
            "ok": True,
            "return_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_perf_probe",
        lambda _tmp_dir: {"available": True, "reason": "ok"},
    )
    monkeypatch.setattr(suite_preflight, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(suite_preflight, "_read_text", lambda _path: "")
    monkeypatch.setattr(
        suite_preflight,
        "ensure_host_contract",
        lambda *_args, **_kwargs: (
            {
                "contract_id": "host123",
                "host_fingerprint": "host123",
                "required_visible_threads": 8,
            },
            host_contract_path,
            False,
        ),
    )
    monkeypatch.setattr(
        suite_preflight, "validate_host_contract", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        suite_preflight,
        "_run",
        lambda _cmd, cwd=None, telemetry_dir=None, label=None: (0, "{}", ""),
    )

    logger = SuiteEventLogger(
        run_id="preflight-events",
        run_root=tmp_path,
        events_path=tmp_path / "events.ndjson",
        transcript_path=tmp_path / "terminal_transcript.log",
        console_log_path=tmp_path / "console.log",
        ui_mode="plain",
        log_level="trace",
    )
    logger.start()
    set_active_logger(logger)
    try:
        result = suite_preflight.run_preflight(
            repo_root=Path("."),
            spec=spec,
            telemetry_dir=tmp_path / "telemetry",
            launch_runtime=launch_runtime,
            affinity_adjustment={
                "repair_allowed": True,
                "attempted": False,
                "repair_required": False,
                "before": list(range(8)),
                "after": list(range(8)),
                "target": list(range(8)),
                "error": "",
            },
        )
    finally:
        set_active_logger(None)
        logger.close()

    assert result.ok is True
    events = [
        json.loads(line)
        for line in (tmp_path / "events.ndjson")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    event_types = {event["event_type"] for event in events}
    assert "preflight_start" in event_types
    assert "host_state" in event_types
    assert "preflight_complete" in event_types


def test_preflight_fails_when_cpu_scan_is_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml"))
    launch_runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
    }
    runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
    }
    monkeypatch.setattr(
        suite_preflight, "capture_runtime_provenance", lambda _repo_root: runtime
    )
    monkeypatch.setattr(
        suite_preflight,
        "capture_model_provenance",
        lambda model: {
            "model": model,
            "digest_validated": True,
            "strict_native_supported": True,
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_saguaro_probe",
        lambda _repo_root, _telemetry_dir, strictness=None: {
            "ok": False,
            "health": {"ok": True},
            "cpu_scan_ok": False,
        },
    )
    monkeypatch.setattr(
        suite_preflight, "_perf_probe", lambda _tmp_dir: {"available": True}
    )
    monkeypatch.setattr(suite_preflight, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(suite_preflight, "_read_text", lambda _path: "")
    monkeypatch.setattr(
        suite_preflight,
        "_run",
        lambda _cmd, cwd=None, telemetry_dir=None, label=None, timeout=None: (
            0,
            "{}",
            "",
        ),
    )

    result = suite_preflight.run_preflight(
        repo_root=Path("."),
        spec=spec,
        telemetry_dir=tmp_path / "telemetry",
        launch_runtime=launch_runtime,
        affinity_adjustment={
            "repair_allowed": True,
            "attempted": False,
            "repair_required": False,
            "before": list(range(8)),
            "after": list(range(8)),
            "target": list(range(8)),
            "error": "",
        },
    )

    assert result.ok is False
    assert "saguaro_cpu_scan_failed" in result.payload["failures"]


def test_silver_preflight_warns_instead_of_failing_for_stale_tuning_and_graph(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))
    host_contract_path = tmp_path / "host123.yaml"
    host_contract_path.write_text(
        "contract_id: host123\nhost_fingerprint: host123\nrequired_visible_threads: 8\n",
        encoding="utf-8",
    )
    launch_runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
    }
    runtime = {
        "python": {"virtual_env": "/venv"},
        "host": {
            "host_fingerprint": "host123",
            "logical_cpus": 8,
            "visible_threads": 8,
        },
        "native_library": {
            "native_isa_baseline": "avx2",
            "native_optional_isa_leaves": ["amx"],
            "native_compiled_with_amx": True,
            "native_runtime_amx_available": False,
        },
    }
    monkeypatch.setattr(
        suite_preflight, "capture_runtime_provenance", lambda _repo_root: runtime
    )
    monkeypatch.setattr(
        suite_preflight,
        "capture_model_provenance",
        lambda model: {
            "model": model,
            "digest_validated": True,
            "strict_native_supported": True,
            "digest": f"sha256:{benchmark_suite._safe_name(model)}",
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_saguaro_probe",
        lambda _repo_root, _telemetry_dir, strictness=None: {
            "ok": True,
            "health": {"ok": True},
            "cpu_scan_ok": True,
        },
    )
    monkeypatch.setattr(
        suite_preflight, "_perf_probe", lambda _tmp_dir: {"available": True}
    )
    monkeypatch.setattr(
        suite_preflight,
        "_dependency_report",
        lambda _repo_root: {
            "dev_metadata": {"hwloc_pkg_config": {"present": False}},
            "binaries": {},
            "libraries": {},
        },
    )
    monkeypatch.setattr(suite_preflight, "_cpu_governor", lambda: "performance")
    monkeypatch.setattr(suite_preflight, "_read_text", lambda _path: "")
    monkeypatch.setattr(
        suite_preflight,
        "ensure_host_contract",
        lambda *_args, **_kwargs: (
            {
                "contract_id": "host123",
                "host_fingerprint": "host123",
                "required_visible_threads": 8,
            },
            host_contract_path,
            False,
        ),
    )
    monkeypatch.setattr(
        suite_preflight, "validate_host_contract", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        suite_preflight,
        "load_tuning_contract",
        lambda _repo_root, _fingerprint, model: (
            {
                "host_contract_id": "host123",
                "host_fingerprint": "host123",
                "model": model,
                "model_digest": f"sha256:{benchmark_suite._safe_name(model)}",
                "profile_schema_version": spec.schema_version,
                "benchmark_harness_hash": "sha256:stale",
                "thread_config": {
                    "decode_threads": 8,
                    "batch_threads": 8,
                    "ubatch": 32,
                },
            },
            tmp_path / f"{benchmark_suite._safe_name(model)}.json",
        ),
    )
    monkeypatch.setattr(
        suite_preflight,
        "validate_tuning_contract",
        lambda *_args, **_kwargs: ["benchmark_harness_hash_mismatch:sha256:new"],
    )
    monkeypatch.setattr(
        suite_preflight,
        "_graph_preflight",
        lambda **_kwargs: {
            "passed": False,
            "status": "covered",
            "unresolved_boundaries": [{"file": "core/native/model_graph_wrapper.py"}],
            "resolved_by_manifest": [],
            "artifact": str(tmp_path / "graph_preflight.json"),
        },
    )
    monkeypatch.setattr(
        suite_preflight,
        "_run",
        lambda _cmd, cwd=None, telemetry_dir=None, label=None, timeout=None: (
            0,
            "{}",
            "",
        ),
    )

    result = suite_preflight.run_preflight(
        repo_root=Path("."),
        spec=spec,
        telemetry_dir=tmp_path / "telemetry",
        launch_runtime=launch_runtime,
        affinity_adjustment={
            "repair_allowed": True,
            "attempted": False,
            "repair_required": False,
            "before": list(range(8)),
            "after": list(range(8)),
            "target": list(range(8)),
            "error": "",
        },
    )

    assert result.ok is True
    assert result.payload["preflight_strictness"] == "audit"
    assert result.payload["tuning_state"] == "stale"
    assert "graph_preflight_warning" in result.payload["warnings"]
    assert any(
        item.startswith("tuning_contract:") for item in result.payload["warnings"]
    )
    assert "graph_preflight_failed" not in result.payload["failures"]


def test_model_thread_contracts_ignore_stale_contracts() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    resolved = benchmark_suite._model_thread_contracts(
        repo_root=Path("."),
        spec=spec,
        preflight_payload={
            "host_contract": {"host_fingerprint": "host123"},
            "tuning_contracts": {
                "granite4:tiny-h": {
                    "readiness_state": "stale",
                    "thread_config": {
                        "decode_threads": 8,
                        "batch_threads": 8,
                        "ubatch": 32,
                    },
                }
            },
        },
    )

    assert resolved == {}


def test_calibration_seed_candidates_are_deterministic() -> None:
    assert benchmark_suite._calibration_seed_candidates(16, [16, 32]) == [
        (8, 8, 16),
        (8, 8, 32),
        (12, 12, 16),
        (12, 12, 32),
        (16, 16, 16),
        (16, 16, 32),
        (16, 12, 32),
        (12, 16, 32),
    ]


def test_run_calibration_writes_tuning_contracts_for_each_model(
    monkeypatch, tmp_path: Path
) -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_calibrate.yaml"))
    layout = resolve_suite_layout(tmp_path, "calibration-run")
    ensure_suite_layout(layout)
    attempt_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    written: dict[str, dict[str, object]] = {}

    def _stub_lane_attempts(**kwargs):  # noqa: ANN001
        lane_id = kwargs["lane_id"]
        model = kwargs["spec"].models[0]
        for decode_threads, batch_threads, ubatch in kwargs["thread_matrix"]:
            decode_score = (
                100.0
                - abs(int(decode_threads) - (16 if model == "granite4:tiny-h" else 12))
                * 3.0
            )
            decode_score -= (
                abs(int(batch_threads) - (16 if model == "granite4:tiny-h" else 12))
                * 2.0
            )
            decode_score -= abs(int(ubatch) - 32) * 0.05
            for run_index in range(kwargs["measured_runs"]):
                attempt_rows.append(
                    {
                        "attempt_id": f"{lane_id}-{model}-{decode_threads}-{batch_threads}-{ubatch}-{run_index}",
                        "lane_id": lane_id,
                        "ablation_id": "all_on",
                        "model_id": model,
                        "warmup": False,
                        "thread_config": {
                            "decode_threads": decode_threads,
                            "batch_threads": batch_threads,
                            "ubatch": ubatch,
                        },
                        "throughput": {"decode_tps": decode_score - run_index},
                        "latency": {
                            "ttft_ms": 50.0 + abs(16 - int(decode_threads)) + run_index
                        },
                        "runtime": {
                            "graph_stage_ms": {
                                "lm_head": 10.0 + abs(16 - int(batch_threads))
                            }
                        },
                        "accepted_parallel_tokens": max(
                            int(decode_threads) // 2 - run_index, 0
                        ),
                        "rejected_parallel_tokens": int(run_index),
                        "proposed_parallel_tokens": max(
                            int(decode_threads) // 2 + 2, 0
                        ),
                        "coherence": {"ok": True},
                    }
                )

    monkeypatch.setattr(benchmark_suite, "_lane_attempts", _stub_lane_attempts)
    monkeypatch.setattr(
        benchmark_suite,
        "_run_continuous_surface",
        lambda **kwargs: [
            {
                "model": kwargs["spec"].models[0],
                "results": [
                    {
                        "concurrency": 1,
                        "max_active_requests": 2,
                        "prompt_class": "short",
                        "scheduler_policy": "fcfs",
                        "batch_wait_timeout_ms": 2,
                        "state_page_rows": 128,
                        "max_prefill_rows_per_iteration": 1024,
                        "continuous_interleaved_streams": False,
                        "ttft_ms_p95": 40.0,
                        "tpot_ms_p50": 5.0,
                        "queue_wait_ms_p95": 5.0,
                        "scheduler_iteration_ms_p95": 1.0,
                        "decode_tps_global": 24.0,
                        "decode_goodput_tps": 24.0,
                        "fairness": 1.0,
                        "state_fragmentation_ratio": 0.05,
                        "drift_overhead_percent": 0.01,
                        "continuous_metrics": {},
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_run_quality_evals",
        lambda layout, spec: {
            "perplexity": [
                {
                    "model": model,
                    "ablation_id": "all_on",
                    "perplexity": 10.0,
                    "tokens_scored": 32,
                }
                for model in spec.models
            ],
            "confidence": [
                {
                    "model": model,
                    "ablation_id": "all_on",
                    "tokens_scored": 8,
                    "mean_token_confidence": 0.8,
                    "p95_token_confidence": 0.9,
                    "expected_calibration_error": 0.1,
                }
                for model in spec.models
            ],
            "coherence": [
                {
                    "model": model,
                    "ablation_id": "all_on",
                    "pass_rate": 1.0,
                    "records": [{"generated_text": "valid output"}],
                }
                for model in spec.models
            ],
            "accuracy": [
                {
                    "model": model,
                    "ablation_id": "all_on",
                    "samples": 2,
                    "pass_rate": 1.0,
                    "exact_match_rate": 1.0,
                }
                for model in spec.models
            ],
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_run_kernel_harness",
        lambda **kwargs: {
            "schema_version": "native_qsg_suite.kernel_summary.v1",
            "models": [
                {
                    "model": model,
                    "runs": [
                        {
                            "kernel": "lm_head",
                            "estimated_recoverable_gain_pct": 4.0,
                            "cv_pct": 2.0,
                        }
                    ],
                }
                for model in kwargs["spec"].models
            ],
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "write_tuning_contract",
        lambda repo_root, host_fingerprint, model, payload: (
            written.__setitem__(model, payload),
            tmp_path / f"{benchmark_suite._safe_name(model)}.json",
        )[1],
    )

    result = benchmark_suite._run_calibration(
        repo_root=Path("."),
        layout=layout,
        spec=spec,
        preflight_payload={
            "host_contract": {
                "contract_id": "0bea3bede3a8874a",
                "host_fingerprint": "0bea3bede3a8874a",
                "required_visible_threads": 16,
            },
            "host_contract_sha256": "sha256:host",
            "models": {
                model: {"digest": f"sha256:{benchmark_suite._safe_name(model)}"}
                for model in spec.models
            },
        },
        attempt_rows=attempt_rows,
        failure_rows=failure_rows,
        completed_attempt_ids=set(),
        resume_completed=set(),
        calibration_mode="search",
        calibration_source="campaign",
        calibration_target_profile="gold",
    )

    assert set(written) == set(spec.models)
    assert (
        result["contracts"]["granite4:tiny-h"]["thread_config"]["decode_threads"] == 16
    )
    assert result["contracts"]["qwen3.5:4b"]["thread_config"]["decode_threads"] == 12
    assert written["granite4:tiny-h"]["thread_config"]["batch_threads"] == 16
    assert written["qwen3.5:9b"]["thread_config"]["ubatch"] == 32
    assert written["granite4:tiny-h"]["quantization_profile"] == {
        "weights": "",
        "kv_cache": "fp32",
    }
    assert written["granite4:tiny-h"]["schema_version"] == (
        "native_qsg_suite.tuning_contract.v2"
    )
    assert written["granite4:tiny-h"]["readiness_state"] == "ready"
    assert written["granite4:tiny-h"]["source_phase"] == "gold"
    assert written["granite4:tiny-h"]["quality_gate_version"] == (
        "native_qsg_suite.quality_gate.v1"
    )
    assert written["granite4:tiny-h"]["continuous_config"]["scheduler_policy"] == "fcfs"
    assert written["granite4:tiny-h"]["pager_config"]["state_page_rows"] > 0
    assert written["granite4:tiny-h"]["admission"]["budget_tier"] == "search"
    assert written["granite4:tiny-h"]["admission"]["invocation_source"] == "campaign"
    assert written["granite4:tiny-h"]["objective_vector"]["decode_goodput_tps"] >= 0.0
    assert written["granite4:tiny-h"]["safe_envelope"]["quality_regression_policy"] == (
        "fail_closed"
    )
    assert (
        written["granite4:tiny-h"]["calibration_stats"][
            "accepted_parallel_tokens_total"
        ]
        > 0
    )
    assert (
        written["granite4:tiny-h"]["calibration_stats"][
            "proposed_parallel_tokens_total"
        ]
        >= written["granite4:tiny-h"]["calibration_stats"][
            "accepted_parallel_tokens_total"
        ]
    )
    assert (
        written["granite4:tiny-h"]["calibration_stats"]["draft_acceptance_ratio"] > 0.0
    )
    assert result["admission"]["target_profile"] == "gold"
    assert result["admission"]["budget_tier"] == "search"


def test_run_calibration_rejects_bad_quality_candidates(
    monkeypatch, tmp_path: Path
) -> None:
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_calibrate.yaml")),
        models=["granite4:tiny-h"],
    )
    layout = resolve_suite_layout(tmp_path, "calibration-bad-quality")
    ensure_suite_layout(layout)
    attempt_rows: list[dict[str, object]] = []

    def _stub_lane_attempts(**kwargs):  # noqa: ANN001
        lane_id = kwargs["lane_id"]
        for decode_threads, batch_threads, ubatch in kwargs["thread_matrix"]:
            attempt_rows.append(
                {
                    "attempt_id": f"{lane_id}-{decode_threads}-{batch_threads}-{ubatch}",
                    "lane_id": lane_id,
                    "ablation_id": "all_on",
                    "model_id": "granite4:tiny-h",
                    "warmup": False,
                    "thread_config": {
                        "decode_threads": decode_threads,
                        "batch_threads": batch_threads,
                        "ubatch": ubatch,
                    },
                    "throughput": {"decode_tps": 12.0},
                    "latency": {"ttft_ms": 50.0},
                    "runtime": {"graph_stage_ms": {"lm_head": 10.0}},
                    "coherence": {"ok": True},
                }
            )

    monkeypatch.setattr(benchmark_suite, "_lane_attempts", _stub_lane_attempts)
    monkeypatch.setattr(
        benchmark_suite,
        "_run_continuous_surface",
        lambda **kwargs: [
            {
                "model": kwargs["spec"].models[0],
                "results": [
                    {
                        "concurrency": 1,
                        "max_active_requests": 2,
                        "prompt_class": "short",
                        "scheduler_policy": "fcfs",
                        "batch_wait_timeout_ms": 2,
                        "state_page_rows": 128,
                        "max_prefill_rows_per_iteration": 1024,
                        "continuous_interleaved_streams": False,
                        "ttft_ms_p95": 50.0,
                        "tpot_ms_p50": 6.0,
                        "queue_wait_ms_p95": 5.0,
                        "scheduler_iteration_ms_p95": 1.0,
                        "decode_tps_global": 12.0,
                        "decode_goodput_tps": 12.0,
                        "fairness": 1.0,
                        "state_fragmentation_ratio": 0.05,
                        "drift_overhead_percent": 0.01,
                        "continuous_metrics": {},
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_run_quality_evals",
        lambda layout, spec: {
            "perplexity": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "perplexity": 12.0,
                    "tokens_scored": 32,
                }
            ],
            "confidence": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "mean_token_confidence": 0.0,
                    "p95_token_confidence": 0.0,
                    "tokens_scored": 0,
                    "expected_calibration_error": 0.0,
                }
            ],
            "coherence": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "pass_rate": 0.0,
                    "records": [{"generated_text": ""}],
                }
            ],
            "accuracy": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "samples": 0,
                    "pass_rate": 0.0,
                    "exact_match_rate": 0.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "_run_kernel_harness",
        lambda **kwargs: {
            "schema_version": "native_qsg_suite.kernel_summary.v1",
            "models": [{"model": "granite4:tiny-h", "runs": []}],
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "write_tuning_contract",
        lambda *args, **kwargs: tmp_path / "should-not-write.json",
    )

    with pytest.raises(RuntimeError, match="Calibration quality gate rejected"):
        benchmark_suite._run_calibration(
            repo_root=Path("."),
            layout=layout,
            spec=spec,
            preflight_payload={
                "host_contract": {
                    "contract_id": "0bea3bede3a8874a",
                    "host_fingerprint": "0bea3bede3a8874a",
                    "required_visible_threads": 16,
                },
                "host_contract_sha256": "sha256:host",
                "models": {
                    "granite4:tiny-h": {
                        "digest": "sha256:granite4_tiny-h",
                    }
                },
            },
            attempt_rows=attempt_rows,
            failure_rows=[],
            completed_attempt_ids=set(),
            resume_completed=set(),
            calibration_mode="search",
            calibration_source="campaign",
            calibration_target_profile="gold",
        )


def test_quality_gate_rejects_high_expected_calibration_error() -> None:
    gate = benchmark_suite._quality_gate_for_model(
        {
            "perplexity": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "perplexity": 10.0,
                    "tokens_scored": 32,
                }
            ],
            "confidence": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "tokens_scored": 16,
                    "mean_token_confidence": 0.8,
                    "p95_token_confidence": 0.9,
                    "expected_calibration_error": 0.35,
                }
            ],
            "coherence": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "pass_rate": 1.0,
                    "records": [{"generated_text": "valid output"}],
                }
            ],
            "accuracy": [
                {
                    "model": "granite4:tiny-h",
                    "ablation_id": "all_on",
                    "samples": 2,
                    "pass_rate": 1.0,
                    "exact_match_rate": 1.0,
                }
            ],
        },
        model="granite4:tiny-h",
    )

    assert gate["passed"] is False
    assert "expected_calibration_error=0.35" in gate["issues"]


def test_run_quality_evals_uses_finalized_decode_path(
    monkeypatch, tmp_path: Path
) -> None:
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml")),
        models=["granite4:tiny-h"],
    )
    layout = resolve_suite_layout(tmp_path, "quality-finalized")
    ensure_suite_layout(layout)
    rubric_path = tmp_path / "rubric.jsonl"
    rubric_path.write_text(
        json.dumps(
            {
                "sample_id": "rubric-1",
                "prompt": "Explain AVX2 and OpenMP.",
                "must_include": ["AVX2", "OpenMP"],
                "must_not_include": ["<think>", "</think>"],
                "min_words": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = replace(
        spec,
        quality_eval=replace(spec.quality_eval, rubric_corpus=str(rubric_path)),
    )

    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_perplexity",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.perplexity.v1",
            "model": kwargs["model"],
            "tokens_scored": 4,
            "perplexity": 10.0,
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_confidence",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.confidence.v1",
            "model": kwargs["model"],
            "tokens_scored": 4,
            "mean_token_confidence": 0.5,
            "p95_token_confidence": 0.6,
            "expected_calibration_error": 0.1,
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_accuracy",
        lambda **kwargs: _fake_accuracy_result(kwargs["model"]),
    )

    class _FakeEngine:
        def prepare_prompt_tokens(self, prompt: str) -> list[int]:  # noqa: ARG002
            return [1, 2]

        def generate(self, *, prompt_tokens, **kwargs):  # noqa: ANN001, ARG002
            return list(prompt_tokens) + [3, 4]

        def detokenize(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "<think>hidden</think> AVX2 OpenMP"

        def decode_generated_tokens(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "AVX2 and OpenMP improve throughput clearly."

    class _FakeScorer:
        def __init__(
            self, model: str, context_length: int, env_overrides=None
        ):  # noqa: ARG002
            self.engine = _FakeEngine()

        def close(self) -> None:
            return None

    monkeypatch.setattr(benchmark_suite, "NativeLogitScorer", _FakeScorer)

    payload = benchmark_suite._run_quality_evals(layout=layout, spec=spec)

    record = payload["coherence"][0]["records"][0]
    assert record["generated_text"] == "AVX2 and OpenMP improve throughput clearly."
    assert record["raw_generated_text"].startswith("<think>")
    assert record["finalized_differs_from_raw"] is True
    assert record["ok"] is True


def test_run_quality_evals_expands_budget_to_match_rubric(
    monkeypatch, tmp_path: Path
) -> None:
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml")),
        models=["granite4:tiny-h"],
    )
    layout = resolve_suite_layout(tmp_path, "quality-budget")
    ensure_suite_layout(layout)
    rubric_path = tmp_path / "rubric.jsonl"
    rubric_path.write_text(
        json.dumps(
            {
                "sample_id": "rubric-1",
                "prompt": "Explain native CPU inference throughput clearly.",
                "must_include": ["native"],
                "must_not_include": [],
                "min_words": 24,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = replace(
        spec,
        quality_eval=replace(spec.quality_eval, rubric_corpus=str(rubric_path)),
    )

    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_perplexity",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.perplexity.v1",
            "model": kwargs["model"],
            "tokens_scored": 4,
            "perplexity": 10.0,
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_confidence",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.confidence.v1",
            "model": kwargs["model"],
            "tokens_scored": 4,
            "mean_token_confidence": 0.5,
            "p95_token_confidence": 0.6,
            "expected_calibration_error": 0.1,
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_accuracy",
        lambda **kwargs: _fake_accuracy_result(kwargs["model"]),
    )

    captured: dict[str, int] = {}

    class _FakeEngine:
        def prepare_prompt_tokens(self, prompt: str) -> list[int]:  # noqa: ARG002
            return [1, 2]

        def generate(
            self, *, prompt_tokens, max_new_tokens, **kwargs
        ):  # noqa: ANN001, ARG002
            captured["max_new_tokens"] = int(max_new_tokens)
            return list(prompt_tokens) + [3, 4]

        def detokenize(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "native throughput"

        def decode_generated_tokens(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "native throughput"

    class _FakeScorer:
        def __init__(
            self, model: str, context_length: int, env_overrides=None
        ):  # noqa: ARG002
            self.engine = _FakeEngine()

        def close(self) -> None:
            return None

    monkeypatch.setattr(benchmark_suite, "NativeLogitScorer", _FakeScorer)

    payload = benchmark_suite._run_quality_evals(layout=layout, spec=spec)

    assert captured["max_new_tokens"] > spec.scenario_pack.canonical_max_new_tokens
    assert (
        payload["coherence"][0]["records"][0]["max_new_tokens"]
        == captured["max_new_tokens"]
    )


def test_run_quality_evals_emits_governance_summary(
    monkeypatch, tmp_path: Path
) -> None:
    spec = replace(
        load_suite_profile(Path("audit/profiles/native_qsg_bronze.yaml")),
        models=["granite4:tiny-h"],
    )
    layout = resolve_suite_layout(tmp_path, "quality-governance")
    ensure_suite_layout(layout)
    rubric_path = tmp_path / "rubric.jsonl"
    rubric_path.write_text(
        json.dumps(
            {
                "sample_id": "rubric-1",
                "prompt": "Explain why build digests matter.",
                "must_include": ["digest"],
                "must_not_include": ["<think>", "</think>"],
                "min_words": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    spec = replace(
        spec,
        quality_eval=replace(spec.quality_eval, rubric_corpus=str(rubric_path)),
    )

    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_perplexity",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.perplexity.v1",
            "model": kwargs["model"],
            "tokens_scored": 8,
            "perplexity": 9.0,
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_confidence",
        lambda **kwargs: {
            "schema_version": "native_qsg_eval.confidence.v1",
            "model": kwargs["model"],
            "tokens_scored": 4,
            "mean_token_confidence": 0.6,
            "p95_token_confidence": 0.8,
            "expected_calibration_error": 0.1,
            "mean_entropy": 0.7,
            "records": [
                {
                    "sample_id": "conf-1",
                    "mean_confidence": 0.6,
                    "mean_entropy": 0.7,
                }
            ],
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "evaluate_accuracy",
        lambda **kwargs: _fake_accuracy_result(kwargs["model"]),
    )

    class _FakeEngine:
        def prepare_prompt_tokens(self, prompt: str) -> list[int]:  # noqa: ARG002
            return [1, 2]

        def generate(self, *, prompt_tokens, **kwargs):  # noqa: ANN001, ARG002
            return list(prompt_tokens) + [3, 4]

        def detokenize(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "digest build reproducibility"

        def decode_generated_tokens(self, tokens: list[int]) -> str:  # noqa: ARG002
            return "digest and build identifiers preserve reproducibility."

    class _FakeScorer:
        def __init__(
            self, model: str, context_length: int, env_overrides=None
        ):  # noqa: ARG002
            self.engine = _FakeEngine()

        def close(self) -> None:
            return None

    monkeypatch.setattr(benchmark_suite, "NativeLogitScorer", _FakeScorer)

    payload = benchmark_suite._run_quality_evals(layout=layout, spec=spec)

    governance = payload["governance"]
    assert governance["passed"] is True
    assert (
        governance["benchmark_families"]["held_out_perplexity"]["status"] == "covered"
    )
    assert (
        governance["benchmark_families"]["structural_validity"]["status"] == "covered"
    )
    assert (
        governance["benchmark_families"]["latent_to_text_faithfulness"]["status"]
        == "covered"
    )
    assert governance["benchmark_families"]["task_accuracy"]["status"] == "covered"
    assert (
        governance["benchmark_families"]["evidence_capsule_fidelity"]["status"]
        == "covered"
    )
    assert governance["entropy_buckets"]["medium"]["samples"] == 1
    assert governance["coherence"]["control_tag_leak_records"] == 0
    assert governance["latent_projection"]["finalized_decode_records"] == 1
    assert governance["latent_projection"]["faithfulness_failures"] == 0
    assert governance["latent_projection"]["evidence_missing_records"] == 0


def test_empty_quality_summary_reports_pending_governance() -> None:
    payload = benchmark_suite._empty_quality_summary()

    assert payload["state"] == "pending"
    governance = payload["governance"]
    assert governance["status"] == "pending"
    assert governance["issues"] == []
    assert (
        governance["benchmark_families"]["held_out_perplexity"]["status"] == "pending"
    )
    assert governance["benchmark_families"]["coherence_rubric"]["status"] == "pending"
    assert (
        governance["benchmark_families"]["latent_to_text_faithfulness"]["status"]
        == "pending"
    )
    assert governance["benchmark_families"]["task_accuracy"]["status"] == "pending"
    assert governance["latent_projection"]["status"] == "pending"


def test_quality_governance_rejects_latent_projection_leaks_control_tags() -> None:
    quality_payload = {
        "schema_version": "native_qsg_suite.quality.v1",
        "perplexity": [
            {
                "model": "granite4:tiny-h",
                "tokens_scored": 8,
                "perplexity": 9.0,
                "ablation_id": "all_on",
            }
        ],
        "confidence": [
            {
                "model": "granite4:tiny-h",
                "tokens_scored": 4,
                "expected_calibration_error": 0.1,
                "mean_entropy": 0.4,
                "ablation_id": "all_on",
                "records": [{"mean_confidence": 0.7, "mean_entropy": 0.4}],
            }
        ],
        "coherence": [
            {
                "model": "granite4:tiny-h",
                "pass_rate": 0.0,
                "ablation_id": "all_on",
                "records": [
                    {
                        "generated_text": "<think>unsafe</think> final answer",
                        "raw_generated_text": "<think>unsafe</think> final answer",
                        "finalized_decode_used": True,
                        "finalized_differs_from_raw": False,
                        "utf8_valid": True,
                        "printable_ratio": 1.0,
                        "repeated_8gram_ratio": 0.0,
                        "leaked_control_tags": ["<think>", "</think>"],
                        "raw_control_tags": ["<think>", "</think>"],
                    }
                ],
            }
        ],
        "accuracy": [
            {
                "model": "granite4:tiny-h",
                "samples": 2,
                "pass_rate": 1.0,
                "exact_match_rate": 1.0,
                "ablation_id": "all_on",
            }
        ],
    }

    governance = benchmark_suite._quality_governance_report(quality_payload)

    assert governance["passed"] is False
    assert "latent_faithfulness_failures:1" in governance["issues"]
    assert governance["latent_projection"]["faithfulness_failures"] == 1


def test_acceptance_governance_marks_prerequisite_blockers() -> None:
    quality_payload = {
        "schema_version": "native_qsg_suite.quality.v1",
        "perplexity": [
            {"model": "granite4:tiny-h", "tokens_scored": 8, "ablation_id": "all_on"}
        ],
        "confidence": [
            {
                "model": "granite4:tiny-h",
                "tokens_scored": 4,
                "expected_calibration_error": 0.1,
                "mean_entropy": 0.4,
                "ablation_id": "all_on",
                "records": [{"mean_confidence": 0.7, "mean_entropy": 0.4}],
            }
        ],
        "coherence": [
            {
                "model": "granite4:tiny-h",
                "pass_rate": 1.0,
                "ablation_id": "all_on",
                "records": [
                    {
                        "generated_text": "valid output",
                        "raw_generated_text": "valid output",
                        "finalized_decode_used": True,
                        "finalized_differs_from_raw": False,
                        "utf8_valid": True,
                        "printable_ratio": 1.0,
                        "repeated_8gram_ratio": 0.0,
                        "leaked_control_tags": [],
                        "raw_control_tags": [],
                    }
                ],
            }
        ],
        "accuracy": [
            {
                "model": "granite4:tiny-h",
                "samples": 2,
                "pass_rate": 1.0,
                "exact_match_rate": 1.0,
                "ablation_id": "all_on",
            }
        ],
    }
    quality_payload["governance"] = benchmark_suite._quality_governance_report(
        quality_payload
    )

    governance = benchmark_suite._acceptance_governance_report(
        summary={"models": [{"model": "granite4:tiny-h"}]},
        quality_payload=quality_payload,
        attempt_rows=[
            {
                "generation_mode": "ar_verify",
                "prompt_category": "analysis",
                "temperature_band": "deterministic",
                "drift_mean": 0.0,
                "drift_max": 0.0,
                "drift_overhead_percent": 0.0,
                "drift_auto_downgrade_events": 0,
            }
        ],
    )

    assert governance["passed"] is True
    assert governance["artifact_completeness"]["passed"] is True
    assert governance["hidden_drift"]["status"] == "covered"
    assert governance["resume_quality"]["status"] == "not_requested"
    assert (
        governance["mode_coverage"]["speculative"]["status"]
        == "blocked_by_prerequisite"
    )
    assert governance["mode_coverage"]["non_ar"]["status"] == "blocked_by_prerequisite"
    assert [item["prerequisite_phase"] for item in governance["blocked_items"]] == [
        "Phase 5",
        "Phase 8",
    ]


def test_acceptance_governance_reports_resume_quality_and_hidden_drift() -> None:
    quality_payload = {
        "schema_version": "native_qsg_suite.quality.v1",
        "perplexity": [
            {
                "model": "granite4:tiny-h",
                "tokens_scored": 8,
                "perplexity": 9.0,
                "ablation_id": "all_on",
            }
        ],
        "confidence": [
            {
                "model": "granite4:tiny-h",
                "tokens_scored": 4,
                "expected_calibration_error": 0.1,
                "mean_entropy": 0.4,
                "ablation_id": "all_on",
                "records": [{"mean_confidence": 0.7, "mean_entropy": 0.4}],
            }
        ],
        "coherence": [
            {
                "model": "granite4:tiny-h",
                "pass_rate": 1.0,
                "ablation_id": "all_on",
                "records": [
                    {
                        "generated_text": "valid output",
                        "raw_generated_text": "<think>scratch</think> valid output",
                        "finalized_decode_used": True,
                        "finalized_differs_from_raw": True,
                        "utf8_valid": True,
                        "printable_ratio": 1.0,
                        "repeated_8gram_ratio": 0.0,
                        "leaked_control_tags": [],
                        "raw_control_tags": ["<think>", "</think>"],
                    }
                ],
            }
        ],
        "accuracy": [
            {
                "model": "granite4:tiny-h",
                "samples": 2,
                "pass_rate": 1.0,
                "exact_match_rate": 1.0,
                "ablation_id": "all_on",
            }
        ],
    }
    quality_payload["governance"] = benchmark_suite._quality_governance_report(
        quality_payload
    )

    governance = benchmark_suite._acceptance_governance_report(
        summary={"models": [{"model": "granite4:tiny-h"}], "completed_attempts": 2},
        quality_payload=quality_payload,
        attempt_rows=[
            {
                "generation_mode": "ar_verify",
                "drift_mean": 0.0,
                "drift_max": 0.0,
                "drift_overhead_percent": 0.0,
                "drift_auto_downgrade_events": 0,
            }
        ],
        resume_context={
            "requested": True,
            "resumed_attempt_ids": ["attempt-1"],
            "completed_lanes": ["calibration", "quality_eval"],
            "checkpoint_artifact_present": True,
            "quality_artifact_present": True,
        },
    )

    assert governance["passed"] is True
    assert governance["hidden_drift"]["status"] == "covered"
    assert governance["hidden_drift"]["passed"] is True
    assert governance["resume_quality"]["status"] == "covered"
    assert governance["resume_quality"]["passed"] is True


def test_acceptance_governance_rejects_missing_resume_quality_artifact() -> None:
    quality_payload = benchmark_suite._empty_quality_summary()

    governance = benchmark_suite._acceptance_governance_report(
        summary={"models": [], "completed_attempts": 0},
        quality_payload=quality_payload,
        attempt_rows=[],
        resume_context={
            "requested": True,
            "resumed_attempt_ids": ["attempt-1"],
            "completed_lanes": ["quality_eval"],
            "checkpoint_artifact_present": True,
            "quality_artifact_present": False,
        },
    )

    assert governance["passed"] is False
    assert "resume_completed_attempts_truncated:0<1" in governance["issues"]
    assert "resume_quality_artifact_missing" in governance["issues"]


def test_strict_native_decode_receipt_rejects_autoregressive_modes() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    receipt = benchmark_suite._strict_native_decode_receipt(
        spec,
        [
            {
                "attempt_id": "attempt-1",
                "warmup": False,
                "generation_mode": "ar_verify",
                "benchmark_label": "autoregressive_verify",
                "provenance": {
                    "env_overrides": {
                        "ANVIL_FORCE_PARALLEL_DECODE": "1",
                        "ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK": "1",
                        "ANVIL_PARALLEL_AR_RECOVERY_ENABLED": "0",
                    }
                },
            }
        ],
    )

    assert receipt["required"] is True
    assert receipt["passed"] is False
    assert receipt["observed_ar_modes"] == ["ar_verify"]
    assert receipt["observed_ar_attempt_ids"] == ["attempt-1"]
    assert "observed_autoregressive_modes:ar_verify" in receipt["issues"]


def test_strict_native_decode_receipt_accepts_block_diffusion_rows() -> None:
    spec = load_suite_profile(Path("audit/profiles/native_qsg_silver.yaml"))

    receipt = benchmark_suite._strict_native_decode_receipt(
        spec,
        [
            {
                "attempt_id": "attempt-1",
                "warmup": False,
                "generation_mode": "block_diffusion",
                "benchmark_label": "block_diffusion_candidate",
                "provenance": {
                    "env_overrides": {
                        "ANVIL_FORCE_PARALLEL_DECODE": "1",
                        "ANVIL_FORBID_AUTOREGRESSIVE_FALLBACK": "1",
                        "ANVIL_PARALLEL_AR_RECOVERY_ENABLED": "0",
                    }
                },
            }
        ],
    )

    assert receipt["required"] is True
    assert receipt["passed"] is True
    assert receipt["observed_non_ar_modes"] == ["block_diffusion"]
    assert receipt["issues"] == []


def test_write_reports_persists_platinum_publication_manifest(tmp_path: Path) -> None:
    layout = resolve_suite_layout(tmp_path, "platinum-run")
    ensure_suite_layout(layout)
    summary = {
        "run_id": "platinum-run",
        "profile_name": "platinum",
        "overall_pass": True,
        "certification_state": "certified_pass",
        "failure_count": 0,
        "agent_triage": {
            "top_hotspot": {"model": "granite4:tiny-h", "kernel": "lm_head"},
            "next_action": "Inspect model_graph.cpp::lm_head",
        },
        "baseline_lineage": {"comparator_mode": "latest_compatible"},
        "spc_report": {"status": "pass"},
        "models": [
            {
                "model": "granite4:tiny-h",
                "decode_tps_p50": 12.0,
                "ttft_ms_p95": 45.0,
                "continuous_decode_tps_global_p50": 10.0,
                "accuracy_pass_rate": 1.0,
                "coherence_pass_rate": 1.0,
            }
        ],
        "quality": benchmark_suite._empty_quality_summary(),
        "kernel_hotspots": [],
        "comparisons": {"compare_to": "latest_compatible"},
        "baseline_run_id": "baseline-1",
    }
    summary["publication_manifest"] = benchmark_suite._build_publication_manifest(
        layout=layout,
        summary=summary,
    )

    benchmark_suite._write_reports(layout, summary)

    manifest_path = layout.reports_dir / "publication_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["publishable"] is True
    assert payload["profile_name"] == "platinum"
    assert payload["models"][0]["accuracy_pass_rate"] == 1.0
