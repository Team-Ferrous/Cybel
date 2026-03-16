#!/usr/bin/env python3
"""Benchmark native QSG generation throughput, coherence, and memory telemetry."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any


def _parse_cpu_list(raw: str) -> list[int]:
    cpus: set[int] = set()
    for token in str(raw or "").replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            start_text, _, end_text = token.partition("-")
            try:
                start = int(start_text)
                end = int(end_text)
            except Exception:
                continue
            if end < start:
                start, end = end, start
            cpus.update(range(start, end + 1))
            continue
        try:
            cpus.add(int(token))
        except Exception:
            continue
    return sorted(cpu for cpu in cpus if cpu >= 0)


def _current_affinity_cpus() -> list[int]:
    if not hasattr(os, "sched_getaffinity"):
        return []
    try:
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    except Exception:
        return []


def _apply_suite_target_affinity() -> dict[str, object]:
    requested = _parse_cpu_list(os.getenv("ANVIL_SUITE_TARGET_CPUS", ""))
    before = _current_affinity_cpus()
    after = list(before)
    applied = False
    error = ""
    if requested and hasattr(os, "sched_setaffinity"):
        if os.getenv("ANVIL_NATIVE_PIN_THREADS") is None:
            # The suite already selected the allowed CPU set. Let OpenMP schedule
            # within that set instead of collapsing the process back to a tiny mask.
            os.environ["ANVIL_NATIVE_PIN_THREADS"] = "0"
        try:
            os.sched_setaffinity(0, set(requested))
            after = _current_affinity_cpus()
            applied = after == requested
        except Exception as exc:
            error = f"{type(exc).__name__}:{exc}"
            after = _current_affinity_cpus()
    return {
        "requested": requested,
        "before": before,
        "after": after,
        "applied": applied,
        "error": error,
    }


_AFFINITY_BOOTSTRAP = _apply_suite_target_affinity()


def _maybe_reexec_with_omp_affinity() -> None:
    if __name__ != "__main__":
        return
    if os.getenv("ANVIL_BENCHMARK_OMP_BOOTSTRAPPED") == "1":
        return
    if os.getenv("ANVIL_DISABLE_OMP_AFFINITY_DEFAULTS", "0") == "1":
        return
    model_args = [
        sys.argv[idx + 1].strip().lower()
        for idx, value in enumerate(sys.argv[:-1])
        if value == "--model"
    ]
    default_proc_bind = "close"
    default_places = "cores"
    if any("granite" in model for model in model_args):
        default_proc_bind = "false"
        default_places = "threads"
    env_updates: dict[str, str] = {}
    if os.getenv("OMP_PROC_BIND") is None:
        env_updates["OMP_PROC_BIND"] = str(
            os.getenv("ANVIL_OMP_PROC_BIND", default_proc_bind)
        )
    if os.getenv("OMP_PLACES") is None:
        env_updates["OMP_PLACES"] = str(os.getenv("ANVIL_OMP_PLACES", default_places))
    if not env_updates:
        return
    env = os.environ.copy()
    env.update(env_updates)
    env["ANVIL_BENCHMARK_OMP_BOOTSTRAPPED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_reexec_with_omp_affinity()

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_PERF_STAT_EVENT_MAP = {
    "cycles": "pmu_cycles",
    "instructions": "pmu_instructions",
    "cache-references": "pmu_cache_references",
    "cache-misses": "pmu_cache_misses",
    "context-switches": "pmu_context_switches",
    "cpu-migrations": "pmu_cpu_migrations",
    "page-faults": "pmu_page_faults",
}


def _pmu_metric_defaults() -> dict[str, Any]:
    return {
        "pmu_observed": False,
        "pmu_parse_error": "",
        "pmu_cycles": None,
        "pmu_instructions": None,
        "pmu_ipc": None,
        "pmu_cache_references": None,
        "pmu_cache_misses": None,
        "pmu_cache_miss_rate": None,
        "pmu_context_switches": None,
        "pmu_cpu_migrations": None,
        "pmu_page_faults": None,
    }


def _parse_perf_stat_number(raw: object) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if "not counted" in lowered or "not supported" in lowered or "<scaled>" in lowered:
        return None
    cleaned = text.replace(",", "")
    try:
        return float(cleaned)
    except Exception:
        return None


def _parse_perf_stat_text(raw: str) -> dict[str, Any]:
    metrics = _pmu_metric_defaults()
    text = str(raw or "")
    if not text.strip():
        metrics["pmu_parse_error"] = "empty_perf_stat_artifact"
        return metrics

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = re.split(r"\s{2,}", stripped)
        if len(parts) < 2:
            continue
        field_name = str(parts[1]).split()[0]
        target = _PERF_STAT_EVENT_MAP.get(field_name)
        if target is None:
            continue
        value = _parse_perf_stat_number(parts[0])
        if value is None:
            continue
        metrics[target] = value
        metrics["pmu_observed"] = True

        ipc_match = re.search(r"#\s*([0-9][0-9.,]*)\s+insn per cycle", stripped)
        if ipc_match is not None:
            metrics["pmu_ipc"] = _parse_perf_stat_number(ipc_match.group(1))
        miss_rate_match = re.search(
            r"#\s*([0-9][0-9.,]*)%\s+of all cache refs", stripped
        )
        if miss_rate_match is not None:
            miss_rate_pct = _parse_perf_stat_number(miss_rate_match.group(1))
            if miss_rate_pct is not None:
                metrics["pmu_cache_miss_rate"] = miss_rate_pct / 100.0

    cycles = metrics["pmu_cycles"]
    instructions = metrics["pmu_instructions"]
    if metrics["pmu_ipc"] is None and cycles and instructions:
        metrics["pmu_ipc"] = instructions / cycles if cycles > 0.0 else None
    cache_refs = metrics["pmu_cache_references"]
    cache_misses = metrics["pmu_cache_misses"]
    if (
        metrics["pmu_cache_miss_rate"] is None
        and cache_refs
        and cache_misses is not None
        and cache_refs > 0.0
    ):
        metrics["pmu_cache_miss_rate"] = cache_misses / cache_refs
    if not metrics["pmu_observed"] and not metrics["pmu_parse_error"]:
        metrics["pmu_parse_error"] = "perf_stat_counters_unavailable"
    return metrics


def _parse_perf_stat_artifact(path: str) -> dict[str, Any]:
    metrics = _pmu_metric_defaults()
    text = str(path or "").strip()
    if not text:
        return metrics
    perf_path = pathlib.Path(text)
    if not perf_path.exists():
        metrics["pmu_parse_error"] = "perf_stat_artifact_missing"
        return metrics
    try:
        raw = perf_path.read_text(encoding="utf-8")
    except Exception as exc:
        metrics["pmu_parse_error"] = f"perf_stat_read_failed:{type(exc).__name__}"
        return metrics
    return _parse_perf_stat_text(raw)


from config.settings import (
    GENERATION_PARAMS,
    GRANITE4_SAMPLING_PROFILES,
    QWEN35_SAMPLING_PROFILES,
)
from core.model.chat_templates import postprocess_strict_native_response
from core.native.native_qsg_engine import (
    NativeQSGEngine,
    SANCTIONED_BACKEND_PATH,
)
from core.native.parallel_generation import benchmark_label_for_mode
from core.qsg.runtime_contracts import PerformanceEnvelope

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None
BENCHMARK_REPORT_SCHEMA = "native_qsg_benchmark.v7"
COHERENCE_MIN_PRINTABLE_RATIO = 0.95
COHERENCE_MAX_REPEATED_4GRAM_RATIO = 0.12
COHERENCE_MAX_REPEATED_8GRAM_RATIO = 0.05
COHERENCE_MIN_WORD_COUNT = 6
NON_AR_DECODE_SPEEDUP_MIN = 1.05
NON_AR_TTFT_REGRESSION_MAX = 1.10
NON_AR_MIN_BLOCK_CONVERGENCE_RATE = 0.50
AR_DECISION_MODES = frozenset({"ar_verify", "ar_recovery"})
SELF_SPEC_DECISION_MODES = frozenset(
    {
        "parallel_hybrid",
        "prompt_lookup",
        "replacement",
        "ssd_bridge",
        "medusa_head",
        "hydra_head",
    }
)
PHASE8_DECISION_SCOPE = "research_only"
PHASE8_PRODUCTION_BLOCKERS = (
    "ar_owned_verification_recovery",
    "diffusion_checkpoint_missing",
)


def _trace_enabled() -> bool:
    return str(
        os.getenv("ANVIL_SUITE_LOG_LEVEL", "trace") or "trace"
    ).strip().lower() in {
        "trace",
        "debug",
    }


def _trace(message: str) -> None:
    if _trace_enabled():
        print(f"[native_qsg_benchmark] {message}", file=sys.stderr, flush=True)


def _prompt_category(prompt: str) -> str:
    normalized = str(prompt or "").strip().lower()
    if not normalized:
        return "unknown"
    if any(
        marker in normalized
        for marker in (
            "```",
            "def ",
            "class ",
            "function",
            "refactor",
            "stack trace",
            "bug",
            "python",
            "javascript",
            "typescript",
            "c++",
            "code",
        )
    ):
        return "code"
    if any(
        marker in normalized
        for marker in ("json", "yaml", "schema", "table", "csv", "xml")
    ):
        return "structured"
    if any(
        marker in normalized
        for marker in ("explain", "analyze", "compare", "reason", "why")
    ):
        return "analysis"
    if "summarize" in normalized or "summary" in normalized:
        return "summarization"
    return "general"


def _temperature_band(value: float) -> str:
    temperature = float(value)
    if temperature <= 0.2:
        return "deterministic"
    if temperature <= 0.7:
        return "low"
    if temperature <= 1.0:
        return "medium"
    return "high"


@dataclass
class BenchmarkResult:
    model: str
    threads: int
    max_new_tokens: int
    run_index: int
    load_seconds: float
    generate_seconds: float
    new_tokens: int
    tokens_per_second: float
    digest: str = ""
    quantization: str = ""
    context_length: int = 0
    prompt_tokens: int = 0
    runtime_total_seconds: float = 0.0
    runtime_prefill_seconds: float = 0.0
    runtime_decode_seconds: float = 0.0
    first_token_latency_seconds: float = 0.0
    prompt_format_ms: float = 0.0
    tokenize_ms: float = 0.0
    embedding_lookup_ms: float = 0.0
    graph_prefill_ms: float = 0.0
    graph_decode_ms: float = 0.0
    sample_ms: float = 0.0
    logits_processor_ms: float = 0.0
    penalty_ms: float = 0.0
    suppression_ms: float = 0.0
    graph_prefill_calls: int = 0
    graph_decode_calls: int = 0
    sample_calls: int = 0
    logits_processor_calls: int = 0
    penalty_calls: int = 0
    suppression_calls: int = 0
    graph_prefill_avg_ms: float = 0.0
    graph_decode_avg_ms: float = 0.0
    sample_avg_ms: float = 0.0
    logits_processor_avg_ms: float = 0.0
    penalty_avg_ms: float = 0.0
    suppression_avg_ms: float = 0.0
    ttft_ms: float = 0.0
    prefill_throughput_tps: float = 0.0
    effective_prefill_tokens: int = 0
    effective_prefill_throughput_tps: float = 0.0
    decode_throughput_tps: float = 0.0
    end_to_end_throughput_tps: float = 0.0
    per_token_latency_p50_ms: float = 0.0
    per_token_latency_p95_ms: float = 0.0
    per_token_latencies_ms: list[float] = field(default_factory=list)
    per_token_latency_p10_ms: float = 0.0
    per_token_latency_p25_ms: float = 0.0
    per_token_latency_p75_ms: float = 0.0
    per_token_latency_p99_ms: float = 0.0
    per_token_latency_stddev_ms: float = 0.0
    per_token_latency_min_ms: float = 0.0
    per_token_latency_max_ms: float = 0.0
    runtime_vs_wall_delta_ms: float = 0.0
    prefill_phase_gap_ms: float = 0.0
    decode_phase_gap_ms: float = 0.0
    decode_threads: int = 0
    batch_threads: int = 0
    ubatch: int = 0
    requested_decode_threads: int = 0
    requested_batch_threads: int = 0
    requested_ubatch: int = 0
    openmp_enabled: bool = False
    avx2_enabled: bool = False
    avx512_enabled: bool = False
    mmap_enabled: bool = False
    mapped_model_bytes: int = 0
    loader_cache_residency_bytes: int = 0
    embedding_materialization_bytes: int = 0
    kv_used_cells: int = 0
    kv_fragmentation_ratio: float = 0.0
    kv_defrag_count: int = 0
    kv_cache_quantization: str = ""
    template_name: str = ""
    granite_moe_mode: str = ""
    active_thread_mode: str = ""
    prefill_chunk_count: int = 0
    lm_head_layout: str = ""
    lm_head_qtype: int = 0
    graph_forward_token_id_calls: int = 0
    graph_forward_token_ids_calls: int = 0
    batched_prefill_token_id_calls: int = 0
    batched_prefill_token_id_path: bool = False
    batched_prefill_token_id_tokens: int = 0
    batch_token_fallback_count: int = 0
    packed_lm_head_calls: int = 0
    runtime_thread_switches: int = 0
    parallel_decode: bool = False
    speculative_decode: bool = False
    generation_mode: str = ""
    benchmark_label: str = ""
    prompt_category: str = ""
    temperature_band: str = ""
    accepted_parallel_tokens: int = 0
    rejected_parallel_tokens: int = 0
    proposed_parallel_tokens: int = 0
    draft_frontier_width: int = 0
    verify_depth: int = 0
    parallel_step_latency_ms: float = 0.0
    draft_confidence_mean: float = 0.0
    draft_confidence_min: float = 0.0
    draft_source: str = ""
    blockwise_blocks: int = 0
    blockwise_denoise_steps: int = 0
    blockwise_convergence_rate: float = 0.0
    masked_generation_ready: bool = False
    masked_generation_steps: int = 0
    masked_generation_proposed_tokens: int = 0
    masked_generation_accepted_tokens: int = 0
    masked_generation_density: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    scheduler_queue_wait_ms: float = 0.0
    scheduler_iteration_ms: float = 0.0
    quality_guard_triggered: bool = False
    speculative_accept_count: int = 0
    speculative_reject_count: int = 0
    self_spec_native_path: bool = False
    self_spec_policy: str = ""
    self_spec_exit_layer: int = 0
    self_spec_exit_fraction: float = 0.0
    self_spec_draft_tokens: int = 0
    min_new_tokens_before_eos: int = 0
    coherence_guard_events: int = 0
    prompt_cache_hit: bool = False
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0
    prompt_cache_lookups: int = 0
    prompt_cache_hit_ratio: float = 0.0
    prompt_cache_reused_tokens: int = 0
    prompt_cache_reuse_ratio: float = 0.0
    sampling_profile: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    rss_before_mb: float = 0.0
    rss_after_load_mb: float = 0.0
    rss_after_generate_mb: float = 0.0
    sample_text: str = ""
    printable_ratio: float = 1.0
    repeated_4gram_ratio: float = 0.0
    repeated_8gram_ratio: float = 0.0
    ascii_ratio: float = 1.0
    word_count: int = 0
    utf8_valid: bool = True
    leaked_control_text: bool = False
    leaked_think_tags: bool = False
    trivial_completion_detected: bool = False
    coherence_valid: bool = True
    coherence_issues: list[str] = field(default_factory=list)
    coherence_issue_count: int = 0
    stop_reason: str = ""
    native_fast_path: bool = False
    parallel_decode_disable_reason: str = ""
    schema_valid: bool = False
    tool_call_parse_success: bool = False
    measurement_valid: bool = True
    measurement_issues: list[str] = field(default_factory=list)
    measurement_issue_count: int = 0
    measurement_source: str = ""
    native_build_id: str = ""
    native_build_sha256: str = ""
    loaded_native_library: str = ""
    native_isa_baseline: str = ""
    native_backend_abi_match: bool = False
    sanctioned_backend_path: str = ""
    tokenizer_backend: str = ""
    backend_module: str = ""
    backend_module_library: str = ""
    backend_module_loaded: bool = False
    backend_module_marker_symbol: str = ""
    backend_module_marker: int = 0
    perf_stat_artifact: str = ""
    perf_c2c_artifact: str = ""
    numa_maps_artifact: str = ""
    pmu_observed: bool = False
    pmu_parse_error: str = ""
    pmu_cycles: float | None = None
    pmu_instructions: float | None = None
    pmu_ipc: float | None = None
    pmu_cache_references: float | None = None
    pmu_cache_misses: float | None = None
    pmu_cache_miss_rate: float | None = None
    pmu_context_switches: float | None = None
    pmu_cpu_migrations: float | None = None
    pmu_page_faults: float | None = None
    sampling_backend: str = ""
    penalties_backend: str = ""
    suppression_backend: str = ""
    logits_backend: str = ""
    full_qsg_enabled: bool = False
    full_graph_enabled: bool = False
    qsg_processors_native_enabled: bool = False
    batched_prefill_native_enabled: bool = False
    context_stabilizer_enabled: bool = False
    context_stabilizer_mode: str = ""
    drift_latest: float = 0.0
    drift_mean: float = 0.0
    drift_max: float = 0.0
    drift_decay_ratio: float = 1.0
    drift_damped_blocks: int = 0
    drift_pruned_blocks: int = 0
    drift_active_tokens: int = 0
    stabilizer_seconds: float = 0.0
    stabilizer_calls: int = 0
    drift_auto_downgrade_events: int = 0
    drift_overhead_percent: float = 0.0
    coconut_enabled: bool = False
    coconut_paths: int = 0
    coconut_alpha: float = 0.0
    coconut_seconds: float = 0.0
    coconut_candidate_count: int = 0
    coconut_entropy_mean: float = 0.0
    coconut_amplitude_mean: float = 0.0
    coconut_consistency_rejects: int = 0
    grover_enabled: bool = False
    grover_top_k: int = 0
    grover_damping: float = 0.0
    grover_calls: int = 0
    grover_seconds: float = 0.0
    grover_candidate_count: int = 0
    grover_rescore_delta_mean: float = 0.0
    grover_timeout_events: int = 0
    strict_path_stable: bool = False
    hot_path_numpy_detected: bool = False
    python_hot_path_calls: int = 0
    numpy_hot_path_calls: int = 0
    python_qsg_forward_calls: int = 0
    python_attention_fallback_calls: int = 0
    python_ssm_fallback_calls: int = 0
    python_moe_fallback_calls: int = 0
    llama_cpp_hot_path_calls: int = 0
    physical_core_count: int = 0
    logical_core_count: int = 0
    p_core_count: int = 0
    affinity_policy: str = ""
    affinity_mode: int = 1
    l3_domain_count: int = 0
    numa_strict: bool = False
    numa_affinity_mode: str = ""
    numa_hugepage: str = ""
    numa_bind_policy: str = ""
    numa_first_touch: bool = False
    topology_json: str = ""
    os_thread_migrations: int = 0
    os_last_cpu: int = -1
    omp_places: str = ""
    omp_proc_bind: str = ""
    omp_max_threads: int = 0
    omp_dynamic: bool = False
    omp_active_levels: int = 0
    perf_event_access: bool = False
    perf_event_access_reason: str = ""
    cpu_governor: str = ""
    thp_mode: str = ""
    perf_counter_source: str = ""
    worker_cpu_mask: str = ""
    orchestrator_cpu_mask: str = ""
    l3_domain_ids_active: list[int] = field(default_factory=list)
    autotune_profile_id: str = ""
    autotune_source: str = ""
    autotune_score: float = 0.0
    autotune_exploration_count: int = 0
    hot_path_proof: dict[str, str] = field(default_factory=dict)
    performance_envelope: dict[str, Any] = field(default_factory=dict)
    performance_twin: dict[str, Any] = field(default_factory=dict)
    repo_coupled_runtime: dict[str, Any] = field(default_factory=dict)
    graph_stage_ms: dict[str, float] = field(default_factory=dict)
    graph_stage_avg_ms: dict[str, float] = field(default_factory=dict)
    graph_stage_calls: dict[str, int] = field(default_factory=dict)
    arch_hidden_dim: int = 0
    arch_num_layers: int = 0
    arch_num_attention_layers: int = 0
    arch_num_ssm_layers: int = 0
    arch_num_moe_layers: int = 0
    arch_num_heads: int = 0
    arch_num_kv_heads: int = 0
    arch_head_dim: int = 0
    arch_intermediate_dim: int = 0
    arch_vocab_size: int = 0
    arch_ssm_state_dim: int = 0
    arch_ssm_conv_kernel: int = 0
    arch_num_experts: int = 0
    arch_top_k_experts: int = 0
    arch_rope_dim: int = 0
    arch_weight_qtype: str = ""
    arch_lm_head_qtype: str = ""
    kernel_efficiency: dict[str, dict[str, float | int]] = field(default_factory=dict)
    raw_sample_text: str = ""
    raw_printable_ratio: float = 1.0
    raw_repeated_4gram_ratio: float = 0.0
    raw_repeated_8gram_ratio: float = 0.0
    raw_word_count: int = 0
    raw_utf8_valid: bool = True
    raw_leaked_control_text: bool = False
    raw_leaked_think_tags: bool = False
    raw_trivial_completion_detected: bool = False
    raw_coherence_valid: bool = True
    raw_coherence_issues: list[str] = field(default_factory=list)
    raw_coherence_issue_count: int = 0


def _rss_mb() -> float:
    if _PROCESS is None:
        return 0.0
    try:
        return float(_PROCESS.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _cpu_model() -> str:
    try:
        if pathlib.Path("/proc/cpuinfo").exists():
            for line in (
                pathlib.Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            ):
                if line.lower().startswith("model name"):
                    _, _, value = line.partition(":")
                    return value.strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _read_text_file(path: str) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_selected_kernel_mode(path: str) -> str:
    raw = _read_text_file(path)
    if not raw:
        return ""
    for token in raw.split("["):
        if "]" in token:
            selected, _, _ = token.partition("]")
            if selected.strip():
                return selected.strip()
    return raw.splitlines()[0].strip()


def _cpu_governor() -> str:
    governor = _read_text_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor:
        return governor
    return _read_selected_kernel_mode(
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"
    )


def _perf_event_probe() -> tuple[bool, str]:
    paranoid = _read_text_file("/proc/sys/kernel/perf_event_paranoid")
    if paranoid:
        try:
            value = int(paranoid)
        except Exception:
            value = None
        else:
            if value >= 3:
                return False, f"blocked:perf_event_paranoid={value}"
            return True, f"available:perf_event_paranoid={value}"
    perf_present = shutil.which("perf") is not None
    return (
        perf_present,
        "available:unknown" if perf_present else "blocked:perf_missing",
    )


def _host_fingerprint() -> str:
    facts = _host_facts()
    fields = (
        str(facts.get("hostname", "")),
        str(facts.get("machine", "")),
        str(facts.get("platform", "")),
        str(facts.get("cpu_model", "")),
        str(facts.get("affinity_visible_threads", "")),
    )
    return "|".join(fields)


def _safe_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _autotune_cache_path(model: str) -> pathlib.Path:
    return (
        REPO_ROOT
        / ".anvil"
        / "benchmarks"
        / "autotune"
        / _safe_key(_host_fingerprint())
        / f"{_safe_key(model)}.json"
    )


def _load_autotune_profile(model: str) -> dict[str, object]:
    path = _autotune_cache_path(model)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_autotune_profile(model: str, payload: dict[str, object]) -> None:
    path = _autotune_cache_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _host_facts() -> dict[str, object]:
    logical = int(os.cpu_count() or 1)
    physical = 0
    if psutil is not None:
        try:
            physical = int(psutil.cpu_count(logical=False) or 0)
        except Exception:
            physical = 0
    runtime_affinity = _current_affinity_cpus()
    launch_affinity = list(
        (_AFFINITY_BOOTSTRAP.get("after") or _AFFINITY_BOOTSTRAP.get("before") or [])
    )
    affinity_visible = int(len(launch_affinity or runtime_affinity)) or logical
    runtime_affinity_visible = int(len(runtime_affinity)) or affinity_visible
    perf_event_access, perf_event_access_reason = _perf_event_probe()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "logical_cpus": logical,
        "physical_cores": physical,
        "affinity_visible_threads": affinity_visible,
        "runtime_affinity_visible_threads": runtime_affinity_visible,
        "launch_affinity_cpus": launch_affinity,
        "runtime_affinity_cpus": runtime_affinity,
        "suite_target_cpus": list(_AFFINITY_BOOTSTRAP.get("requested") or []),
        "suite_target_applied": bool(_AFFINITY_BOOTSTRAP.get("applied", False)),
        "suite_target_error": str(_AFFINITY_BOOTSTRAP.get("error") or ""),
        "cpu_governor": _cpu_governor(),
        "thp_mode": _read_selected_kernel_mode(
            "/sys/kernel/mm/transparent_hugepage/enabled"
        ),
        "perf_event_access": perf_event_access,
        "perf_event_access_reason": perf_event_access_reason,
    }


def _report_host_facts(results: list["BenchmarkResult"]) -> dict[str, object]:
    facts = dict(_host_facts())
    runtime_visible = int(facts.get("runtime_affinity_visible_threads", 0) or 0)
    runtime_cpus = {
        int(cpu)
        for cpu in list(facts.get("runtime_affinity_cpus") or [])
        if isinstance(cpu, int)
    }
    for result in results:
        runtime_visible = max(
            runtime_visible,
            int(getattr(result, "logical_core_count", 0) or 0),
            int(getattr(result, "omp_max_threads", 0) or 0),
        )
        runtime_cpus.update(_parse_cpu_list(getattr(result, "worker_cpu_mask", "")))
        runtime_cpus.update(
            _parse_cpu_list(getattr(result, "orchestrator_cpu_mask", ""))
        )
    launch_cpus = [
        int(cpu)
        for cpu in list(facts.get("launch_affinity_cpus") or [])
        if isinstance(cpu, int)
    ]
    if runtime_cpus:
        if runtime_visible > len(runtime_cpus) and len(launch_cpus) >= runtime_visible:
            facts["runtime_affinity_cpus"] = list(launch_cpus)
            facts["runtime_affinity_visible_threads"] = runtime_visible
            return facts
        facts["runtime_affinity_cpus"] = sorted(runtime_cpus)
        facts["runtime_affinity_visible_threads"] = max(
            len(runtime_cpus), runtime_visible
        )
        return facts
    if runtime_visible > int(facts.get("runtime_affinity_visible_threads", 0) or 0):
        facts["runtime_affinity_visible_threads"] = runtime_visible
        if launch_cpus and len(launch_cpus) >= runtime_visible:
            facts["runtime_affinity_cpus"] = list(launch_cpus)
    return facts


def _printable_ratio(text: str) -> float:
    if not text:
        return 1.0
    printable = sum(1 for ch in text if ch.isprintable() and ch not in {"\x0b", "\x0c"})
    return printable / max(len(text), 1)


def _ascii_ratio(text: str) -> float:
    if not text:
        return 1.0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(len(text), 1)


def _utf8_valid(text: str) -> bool:
    try:
        text.encode("utf-8")
    except UnicodeError:
        return False
    return True


def _coherence_word_count(text: str) -> int:
    if not text:
        return 0
    words = [part for part in text.split() if part.strip()]
    if words:
        return len(words)
    cjk_words = re.findall(r"[\u4e00-\u9fff]", text)
    alpha_words = re.findall(r"[A-Za-z0-9]+", text)
    fallback = len(cjk_words) + len(alpha_words)
    return fallback


def _repeated_ngram_ratio(text: str, n: int = 4) -> float:
    if not text or len(text) < n:
        return 0.0
    ngrams = [text[i : i + n] for i in range(0, len(text) - n + 1)]
    if not ngrams:
        return 0.0
    repeats = len(ngrams) - len(set(ngrams))
    return float(repeats) / float(len(ngrams))


def _trivial_completion_detected(text: str, stop_reason: str) -> bool:
    normalized = " ".join(text.strip().lower().split())
    if not normalized:
        return True
    incomplete_prefixes = (
        "this is a",
        "this is an",
        "this is the",
        "the answer is",
        "here is a",
        "here's a",
        "i am",
        "it is",
        "the user wants",
        "the previous",
        "the user is",
    )
    if normalized in incomplete_prefixes:
        return True
    if any(normalized.startswith(prefix) for prefix in incomplete_prefixes):
        if len(normalized) < 32:
            return True
    odd_prefixes = (
        "the user wants me",
        "in the previous",
        "the previous domain",
    )
    if any(prefix in normalized for prefix in odd_prefixes):
        return len(normalized) < 64
    short_stop = stop_reason in {
        "eos",
        "speculative_complete",
        "parallel_decode_complete",
    }
    return short_stop and len(normalized) < 32 and len(normalized.split()) <= 4


def _coherence_issues(
    *,
    text: str,
    printable_ratio: float,
    repeated_4gram_ratio: float,
    repeated_8gram_ratio: float,
    word_count: int,
    utf8_valid: bool,
    leaked_control_text: bool,
    leaked_think_tags: bool,
    stop_reason: str,
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    if not utf8_valid:
        issues.append("coherence_utf8_valid=false")
    if printable_ratio < COHERENCE_MIN_PRINTABLE_RATIO:
        issues.append(
            f"coherence_printable_ratio {printable_ratio:.3f} < "
            f"{COHERENCE_MIN_PRINTABLE_RATIO:.3f}"
        )
    if repeated_4gram_ratio > COHERENCE_MAX_REPEATED_4GRAM_RATIO:
        issues.append(
            "coherence_repeated_4gram_ratio "
            f"{repeated_4gram_ratio:.3f} > {COHERENCE_MAX_REPEATED_4GRAM_RATIO:.3f}"
        )
    if repeated_8gram_ratio > COHERENCE_MAX_REPEATED_8GRAM_RATIO:
        issues.append(
            "coherence_repeated_8gram_ratio "
            f"{repeated_8gram_ratio:.3f} > {COHERENCE_MAX_REPEATED_8GRAM_RATIO:.3f}"
        )
    if leaked_control_text:
        issues.append("coherence_leaked_control_text=true")
    if leaked_think_tags:
        issues.append("coherence_leaked_think_tags=true")
    if _trivial_completion_detected(text, stop_reason):
        issues.append("coherence_trivial_completion=true")
    if word_count < COHERENCE_MIN_WORD_COUNT and len(text.strip()) < 48:
        issues.append(f"coherence_word_count {word_count} < {COHERENCE_MIN_WORD_COUNT}")
    return (not issues), issues


def _result_failures(
    result: BenchmarkResult,
    *,
    min_decode_tps: float | None,
    max_ttft_ms: float | None,
    min_printable_ratio: float | None,
    max_repeated_4gram_ratio: float | None,
    max_repeated_8gram_ratio: float | None,
    min_ascii_ratio: float | None,
    min_word_count: int | None,
    max_rss_growth_mb: float | None,
    require_utf8: bool,
    require_measurement_valid: bool,
    require_openmp: bool,
    require_avx2: bool,
    require_mmap: bool,
    host_access: str = "user",
    collect_hw_counters: str = "auto",
    require_grover: bool = False,
    require_coconut: bool = False,
    strict_native_gates: bool = True,
) -> list[str]:
    failures: list[str] = []
    if min_decode_tps is not None and result.decode_throughput_tps < min_decode_tps:
        failures.append(
            f"decode_tps {result.decode_throughput_tps:.2f} < {min_decode_tps:.2f}"
        )
    ttft_ms = result.first_token_latency_seconds * 1000.0
    if max_ttft_ms is not None and ttft_ms > max_ttft_ms:
        failures.append(f"ttft_ms {ttft_ms:.2f} > {max_ttft_ms:.2f}")
    if min_printable_ratio is not None and result.printable_ratio < min_printable_ratio:
        failures.append(
            f"printable_ratio {result.printable_ratio:.3f} < {min_printable_ratio:.3f}"
        )
    if (
        max_repeated_4gram_ratio is not None
        and result.repeated_4gram_ratio > max_repeated_4gram_ratio
    ):
        failures.append(
            "repeated_4gram_ratio "
            f"{result.repeated_4gram_ratio:.3f} > {max_repeated_4gram_ratio:.3f}"
        )
    if (
        max_repeated_8gram_ratio is not None
        and result.repeated_8gram_ratio > max_repeated_8gram_ratio
    ):
        failures.append(
            "repeated_8gram_ratio "
            f"{result.repeated_8gram_ratio:.3f} > {max_repeated_8gram_ratio:.3f}"
        )
    if min_ascii_ratio is not None and result.ascii_ratio < min_ascii_ratio:
        failures.append(f"ascii_ratio {result.ascii_ratio:.3f} < {min_ascii_ratio:.3f}")
    if min_word_count is not None and result.word_count < min_word_count:
        failures.append(f"word_count {result.word_count} < {min_word_count}")
    if max_rss_growth_mb is not None:
        rss_growth = result.rss_after_generate_mb - result.rss_before_mb
        if rss_growth > max_rss_growth_mb:
            failures.append(f"rss_growth_mb {rss_growth:.2f} > {max_rss_growth_mb:.2f}")
    if require_utf8 and not result.utf8_valid:
        failures.append("utf8_valid=false")
    if require_measurement_valid and not result.measurement_valid:
        failures.append("measurement_valid=false")
    if require_openmp and not result.openmp_enabled:
        failures.append("openmp_enabled=false")
    if require_avx2 and not result.avx2_enabled:
        failures.append("avx2_enabled=false")
    if require_mmap and not result.mmap_enabled:
        failures.append("mmap_enabled=false")
    if require_grover and not result.grover_enabled:
        failures.append("grover_enabled=false")
    if require_coconut and not result.coconut_enabled:
        failures.append("coconut_enabled=false")
    if (
        str(host_access).strip().lower() == "privileged"
        and str(collect_hw_counters).strip().lower() == "required"
        and not result.perf_event_access
    ):
        failures.append(
            f"perf_event_access=false({result.perf_event_access_reason or 'unknown'})"
        )
    if strict_native_gates:
        if not result.full_graph_enabled:
            failures.append("full_graph_enabled=false")
        if not result.full_qsg_enabled:
            failures.append("full_qsg_enabled=false")
        if not result.qsg_processors_native_enabled:
            failures.append("qsg_processors_native_enabled=false")
        if not result.batched_prefill_native_enabled:
            failures.append("batched_prefill_native_enabled=false")
        if result.hot_path_numpy_detected:
            failures.append("hot_path_numpy_detected=true")
        if result.python_hot_path_calls > 0:
            failures.append("python_hot_path_calls>0")
        if result.numpy_hot_path_calls > 0:
            failures.append("numpy_hot_path_calls>0")
        if result.python_qsg_forward_calls > 0:
            failures.append("python_qsg_forward_calls>0")
        if result.python_attention_fallback_calls > 0:
            failures.append("python_attention_fallback_calls>0")
        if result.python_ssm_fallback_calls > 0:
            failures.append("python_ssm_fallback_calls>0")
        if result.python_moe_fallback_calls > 0:
            failures.append("python_moe_fallback_calls>0")
        if result.llama_cpp_hot_path_calls > 0:
            failures.append("llama_cpp_hot_path_calls>0")
        if result.tokenizer_backend != "native":
            failures.append(
                f"tokenizer_backend={result.tokenizer_backend or 'unknown'}"
            )
        if result.sanctioned_backend_path != SANCTIONED_BACKEND_PATH:
            failures.append("sanctioned_backend_path_mismatch")
        if not result.strict_path_stable:
            failures.append("strict_path_stable=false")
        if result.batch_token_fallback_count > 0:
            failures.append(
                f"batch_token_fallback_count={result.batch_token_fallback_count}"
            )
        if not result.native_backend_abi_match:
            failures.append("native_backend_abi_match=false")
        if result.native_isa_baseline != "avx2":
            failures.append(
                f"native_isa_baseline={result.native_isa_baseline or 'unknown'}"
            )
    if not result.sample_text.strip():
        failures.append("empty_sample_text")
    if result.leaked_control_text:
        failures.append("leaked_control_text=true")
    if result.leaked_think_tags:
        failures.append("leaked_think_tags=true")
    if not result.coherence_valid:
        failures.extend(list(result.coherence_issues))
    if not result.raw_coherence_valid:
        failures.extend(f"raw_{issue}" for issue in result.raw_coherence_issues)
    return failures


def _markdown_report(
    results: list[BenchmarkResult],
    failures: dict[str, list[str]],
) -> str:
    summary_rows = _summary_rows(results, failures)
    speculative_rows = _speculative_acceptance_rows(results)
    lines = [
        "# Native QSG Benchmark Report",
        "",
        f"Schema: `{BENCHMARK_REPORT_SCHEMA}`",
        "",
        "## Summary",
        "",
        "| Model | Profile | Template | Runs | Pass | Avg TTFT ms | Avg Prefill TPS | Avg Decode TPS | Avg Cache Hit% | Avg Prefill Gap ms | Avg Decode Gap ms | Avg Thread Switches | Avg Drift Mean | Avg Drift Ovhd% |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['sampling_profile'] or '-'} | "
            f"{row['template_name'] or '-'} | {row['runs']} | {row['passes']} | "
            f"{row['avg_ttft_ms']:.2f} | {row['avg_effective_prefill_tps']:.2f} | "
            f"{row['avg_decode_tps']:.2f} | {row['avg_prompt_cache_hit_ratio'] * 100.0:.1f} | "
            f"{row['avg_prefill_phase_gap_ms']:.2f} | {row['avg_decode_phase_gap_ms']:.2f} | "
            f"{row['avg_runtime_thread_switches']:.2f} | {row['avg_drift_mean']:.4f} | "
            f"{row['avg_drift_overhead_percent']:.2f} |"
        )
    if speculative_rows:
        lines.extend(
            [
                "",
                "## Speculative Acceptance",
                "",
                "| Prompt Category | Temp Band | Mode | Benchmark Label | Runs | Proposed | Accepted | Rejected | Acceptance % | Avg Draft Conf | Min Draft Conf | Avg Decode TPS | Avg E2E TPS |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in speculative_rows:
            lines.append(
                f"| {row['prompt_category']} | {row['temperature_band']} | "
                f"{row['generation_mode']} | {row['benchmark_label']} | {row['runs']} | "
                f"{row['proposed_parallel_tokens']} | "
                f"{row['accepted_parallel_tokens']} | {row['rejected_parallel_tokens']} | "
                f"{row['acceptance_rate'] * 100.0:.1f} | "
                f"{row['avg_draft_confidence_mean']:.3f} | {row['min_draft_confidence']:.3f} | "
                f"{row['avg_decode_tps']:.2f} | "
                f"{row['avg_end_to_end_tps']:.2f} |"
            )
    non_ar_rows = _non_ar_decision_rows(results)
    if non_ar_rows:
        lines.extend(
            [
                "",
                "## Non-AR Decision Framework",
                "",
                "| Model | Prompt Category | Temp Band | Candidate Runs | Decision | Scope | Prod Ready | Native Hot Path | Decode vs AR | Decode vs Self-Spec | Block Conv % | Draft Source | Readiness Blockers |",
                "| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in non_ar_rows:
            lines.append(
                f"| {row['model']} | {row['prompt_category']} | {row['temperature_band']} | "
                f"{row['candidate_runs']} | {row['recommended_mode']} | "
                f"{row['decision_scope']} | {row['production_ready']} | "
                f"{row['native_hot_path_owned']} | "
                f"{row['decode_speedup_vs_ar']:.2f}x | "
                f"{row['decode_speedup_vs_self_spec']:.2f}x | "
                f"{row['blockwise_convergence_rate'] * 100.0:.1f} | "
                f"{row['draft_source'] or '-'} | "
                f"{','.join(row['production_blockers']) or '-'} |"
            )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Model | Run | Profile | Template | Mode | Thread Mode | Fast Path | LM Head | TC Mode | Batch TokenIds | Batch Tokens | Coherence | Eos Guard | Decode Skip Reason | Prefill TPS | Decode TPS | TTFT ms | GraphD Avg ms | Drift Mean | Drift Max | Drift Ovhd% | TC Downgrades | TC Calls | Sample Avg ms | Prefill Gap ms | Decode Gap ms | Thread Switches | Cache Hit% | Status | Sample |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in results:
        key = f"{result.model}#{result.run_index}"
        status = "PASS" if not failures.get(key) else "FAIL"
        sample = result.sample_text.replace("\n", " ").strip()
        if len(sample) > 96:
            sample = sample[:93] + "..."
        phases = _phase_breakdown(result)
        lines.append(
            f"| {result.model} | {result.run_index} | {result.sampling_profile or '-'} | "
            f"{result.template_name or '-'} | {result.granite_moe_mode or '-'} | "
            f"{result.active_thread_mode or '-'} | {'yes' if result.native_fast_path else 'no'} | "
            f"{result.lm_head_layout or '-'} | {result.context_stabilizer_mode or '-'} | {result.batched_prefill_token_id_calls} | "
            f"{result.batched_prefill_token_id_tokens} | "
            f"{'PASS' if result.coherence_valid else 'FAIL'} | "
            f"{result.coherence_guard_events} | "
            f"{result.parallel_decode_disable_reason or '-'} | "
            f"{result.effective_prefill_throughput_tps:.2f} | {result.decode_throughput_tps:.2f} | "
            f"{result.ttft_ms:.2f} | {result.graph_decode_avg_ms:.2f} | "
            f"{result.drift_mean:.4f} | {result.drift_max:.4f} | "
            f"{result.drift_overhead_percent:.2f} | {result.drift_auto_downgrade_events} | "
            f"{result.stabilizer_calls} | {result.sample_avg_ms:.2f} | "
            f"{phases['prefill_phase_gap_ms']:.2f} | {phases['decode_phase_gap_ms']:.2f} | "
            f"{result.runtime_thread_switches} | {result.prompt_cache_hit_ratio * 100.0:.1f} | "
            f"{status} | {sample} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for key, issues in failures.items():
            lines.append(f"- `{key}`: {', '.join(issues)}")
    return "\n".join(lines) + "\n"


def _phase_breakdown(result: BenchmarkResult) -> dict[str, float]:
    prefill_phase_accounted_ms = (
        float(result.prompt_format_ms)
        + float(result.tokenize_ms)
        + float(result.embedding_lookup_ms)
        + float(result.graph_prefill_ms)
    )
    decode_phase_accounted_ms = (
        float(result.graph_decode_ms)
        + float(result.sample_ms)
        + float(result.logits_processor_ms)
        + float(result.penalty_ms)
        + float(result.suppression_ms)
    )
    runtime_prefill_ms = float(result.runtime_prefill_seconds) * 1000.0
    runtime_decode_ms = float(result.runtime_decode_seconds) * 1000.0
    runtime_total_ms = float(result.runtime_total_seconds) * 1000.0
    total_phase_accounted_ms = prefill_phase_accounted_ms + decode_phase_accounted_ms
    prefill_phase_gap_ms = float(result.prefill_phase_gap_ms)
    decode_phase_gap_ms = float(result.decode_phase_gap_ms)
    if prefill_phase_gap_ms == 0.0 and decode_phase_gap_ms == 0.0:
        prefill_phase_gap_ms = runtime_prefill_ms - prefill_phase_accounted_ms
        decode_phase_gap_ms = runtime_decode_ms - decode_phase_accounted_ms
    return {
        "wall_load_ms": float(result.load_seconds) * 1000.0,
        "wall_generate_ms": float(result.generate_seconds) * 1000.0,
        "runtime_total_ms": runtime_total_ms,
        "runtime_prefill_ms": runtime_prefill_ms,
        "runtime_decode_ms": runtime_decode_ms,
        "ttft_ms": float(result.ttft_ms),
        "prompt_format_ms": float(result.prompt_format_ms),
        "tokenize_ms": float(result.tokenize_ms),
        "embedding_lookup_ms": float(result.embedding_lookup_ms),
        "graph_prefill_ms": float(result.graph_prefill_ms),
        "graph_decode_ms": float(result.graph_decode_ms),
        "sample_ms": float(result.sample_ms),
        "logits_processor_ms": float(result.logits_processor_ms),
        "penalty_ms": float(result.penalty_ms),
        "suppression_ms": float(result.suppression_ms),
        "prefill_phase_accounted_ms": prefill_phase_accounted_ms,
        "decode_phase_accounted_ms": decode_phase_accounted_ms,
        "total_phase_accounted_ms": total_phase_accounted_ms,
        "prefill_phase_gap_ms": prefill_phase_gap_ms,
        "decode_phase_gap_ms": decode_phase_gap_ms,
        "total_phase_gap_ms": runtime_total_ms - total_phase_accounted_ms,
        "runtime_vs_wall_delta_ms": float(result.runtime_vs_wall_delta_ms),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * max(0.0, min(100.0, float(percentile))) / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _compute_kernel_efficiency(
    graph_stage_ms: dict[str, float],
    graph_stage_calls: dict[str, int],
    arch: dict[str, int | str],
) -> dict[str, dict[str, float | int]]:
    """Estimate stage-level FLOP/s and bandwidth from timing + model dimensions."""
    efficiency: dict[str, dict[str, float | int]] = {}
    hidden = int(arch.get("hidden_dim", 0) or 0)
    intermediate = int(arch.get("intermediate_dim", 0) or 0)
    vocab = int(arch.get("vocab_size", 0) or 0)
    num_heads = int(arch.get("num_heads", 0) or 0)
    num_kv_heads = int(arch.get("num_kv_heads", 0) or 0)
    head_dim = int(arch.get("head_dim", 0) or 0)

    q6k_bytes_per_element = 210.0 / 256.0
    q6k_lm_bytes_per_element = 276.0 / 256.0

    stage_specs = {
        "ffn_gate_up": {
            "rows": 2 * intermediate,
            "cols": hidden,
            "bytes_per_elem": q6k_bytes_per_element,
        },
        "ffn_down": {
            "rows": hidden,
            "cols": intermediate,
            "bytes_per_elem": q6k_bytes_per_element,
        },
        "attention_proj": {
            "rows": (num_heads + 2 * num_kv_heads) * head_dim,
            "cols": hidden,
            "bytes_per_elem": q6k_bytes_per_element,
        },
        "attention_out_proj": {
            "rows": hidden,
            "cols": num_heads * head_dim,
            "bytes_per_elem": q6k_bytes_per_element,
        },
        "lm_head": {
            "rows": vocab,
            "cols": hidden,
            "bytes_per_elem": q6k_lm_bytes_per_element,
        },
    }

    for stage, spec in stage_specs.items():
        total_ms = float(graph_stage_ms.get(stage, 0.0) or 0.0)
        calls = int(graph_stage_calls.get(stage, 0) or 0)
        rows = int(spec["rows"])
        cols = int(spec["cols"])
        if total_ms <= 0.0 or calls <= 0 or rows <= 0 or cols <= 0:
            continue

        total_seconds = total_ms / 1000.0
        flops_per_call = 2.0 * rows * cols
        bytes_per_call = rows * cols * float(spec["bytes_per_elem"])
        bytes_per_call += cols * 4.0 + rows * 4.0

        total_flops = flops_per_call * calls
        total_bytes = bytes_per_call * calls
        efficiency[stage] = {
            "total_flops": total_flops,
            "total_bytes": total_bytes,
            "gflops_per_second": round(total_flops / total_seconds / 1e9, 2),
            "gbytes_per_second": round(total_bytes / total_seconds / 1e9, 2),
            "flops_per_call": flops_per_call,
            "bytes_per_call": bytes_per_call,
            "calls": calls,
            "total_ms": total_ms,
            "avg_ms_per_call": round(total_ms / calls, 4),
            "arithmetic_intensity": (
                round(flops_per_call / bytes_per_call, 2)
                if bytes_per_call > 0.0
                else 0.0
            ),
        }

    return efficiency


def _result_record(
    result: BenchmarkResult,
    *,
    issues: list[str] | None = None,
) -> dict[str, object]:
    failures = list(issues or [])
    return {
        "run_id": f"{result.model}#{result.run_index}",
        "model": result.model,
        "run_index": result.run_index,
        "identity": {
            "model": result.model,
            "digest": result.digest,
            "quantization": result.quantization,
            "template_name": result.template_name,
            "granite_moe_mode": result.granite_moe_mode,
            "context_length": result.context_length,
            "prompt_category": result.prompt_category,
        },
        "sampling": {
            "profile": result.sampling_profile,
            "temperature": result.temperature,
            "temperature_band": result.temperature_band,
            "top_p": result.top_p,
            "top_k": result.top_k,
            "min_p": result.min_p,
            "presence_penalty": result.presence_penalty,
            "repetition_penalty": result.repetition_penalty,
        },
        "tokens": {
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.new_tokens,
            "max_new_tokens": result.max_new_tokens,
        },
        "throughput": {
            "wall_tokens_per_second": result.tokens_per_second,
            "prefill_throughput_tps": result.prefill_throughput_tps,
            "effective_prefill_tokens": result.effective_prefill_tokens,
            "effective_prefill_throughput_tps": result.effective_prefill_throughput_tps,
            "decode_throughput_tps": result.decode_throughput_tps,
            "end_to_end_throughput_tps": result.end_to_end_throughput_tps,
            "per_token_latency_p50_ms": result.per_token_latency_p50_ms,
            "per_token_latency_p95_ms": result.per_token_latency_p95_ms,
        },
        "phases": _phase_breakdown(result),
        "hotspots": {
            "graph_prefill_calls": result.graph_prefill_calls,
            "graph_decode_calls": result.graph_decode_calls,
            "graph_forward_token_id_calls": result.graph_forward_token_id_calls,
            "graph_forward_token_ids_calls": result.graph_forward_token_ids_calls,
            "sample_calls": result.sample_calls,
            "logits_processor_calls": result.logits_processor_calls,
            "penalty_calls": result.penalty_calls,
            "suppression_calls": result.suppression_calls,
            "graph_prefill_avg_ms": result.graph_prefill_avg_ms,
            "graph_decode_avg_ms": result.graph_decode_avg_ms,
            "sample_avg_ms": result.sample_avg_ms,
            "logits_processor_avg_ms": result.logits_processor_avg_ms,
            "penalty_avg_ms": result.penalty_avg_ms,
            "suppression_avg_ms": result.suppression_avg_ms,
            "graph_stage_ms": dict(result.graph_stage_ms),
            "graph_stage_avg_ms": dict(result.graph_stage_avg_ms),
            "graph_stage_calls": dict(result.graph_stage_calls),
            "drift_latest": result.drift_latest,
            "drift_mean": result.drift_mean,
            "drift_max": result.drift_max,
            "drift_decay_ratio": result.drift_decay_ratio,
            "drift_overhead_percent": result.drift_overhead_percent,
            "stabilizer_seconds": result.stabilizer_seconds,
            "stabilizer_calls": result.stabilizer_calls,
        },
        "cache": {
            "prompt_cache_hit": result.prompt_cache_hit,
            "prompt_cache_hits": result.prompt_cache_hits,
            "prompt_cache_misses": result.prompt_cache_misses,
            "prompt_cache_lookups": result.prompt_cache_lookups,
            "prompt_cache_hit_ratio": result.prompt_cache_hit_ratio,
            "prompt_cache_reused_tokens": result.prompt_cache_reused_tokens,
            "prompt_cache_reuse_ratio": result.prompt_cache_reuse_ratio,
        },
        "threading": {
            "effective_threads": result.threads,
            "decode_threads": result.decode_threads,
            "batch_threads": result.batch_threads,
            "ubatch": result.ubatch,
            "requested_decode_threads": result.requested_decode_threads,
            "requested_batch_threads": result.requested_batch_threads,
            "requested_ubatch": result.requested_ubatch,
            "active_thread_mode": result.active_thread_mode,
            "prefill_chunk_count": result.prefill_chunk_count,
            "batched_prefill_token_id_calls": result.batched_prefill_token_id_calls,
            "batched_prefill_token_id_path": result.batched_prefill_token_id_path,
            "batched_prefill_token_id_tokens": result.batched_prefill_token_id_tokens,
            "batch_token_fallback_count": result.batch_token_fallback_count,
            "runtime_thread_switches": result.runtime_thread_switches,
            "os_thread_migrations": result.os_thread_migrations,
            "os_last_cpu": result.os_last_cpu,
            "parallel_decode": result.parallel_decode,
            "speculative_decode": result.speculative_decode,
            "generation_mode": result.generation_mode,
            "benchmark_label": result.benchmark_label,
            "prompt_category": result.prompt_category,
            "temperature_band": result.temperature_band,
            "accepted_parallel_tokens": result.accepted_parallel_tokens,
            "rejected_parallel_tokens": result.rejected_parallel_tokens,
            "proposed_parallel_tokens": result.proposed_parallel_tokens,
            "draft_frontier_width": result.draft_frontier_width,
            "verify_depth": result.verify_depth,
            "parallel_step_latency_ms": result.parallel_step_latency_ms,
            "draft_confidence_mean": result.draft_confidence_mean,
            "draft_confidence_min": result.draft_confidence_min,
            "draft_source": result.draft_source,
            "blockwise_blocks": result.blockwise_blocks,
            "blockwise_denoise_steps": result.blockwise_denoise_steps,
            "blockwise_convergence_rate": result.blockwise_convergence_rate,
            "prefix_cache_hit_rate": result.prefix_cache_hit_rate,
            "scheduler_queue_wait_ms": result.scheduler_queue_wait_ms,
            "scheduler_iteration_ms": result.scheduler_iteration_ms,
            "quality_guard_triggered": result.quality_guard_triggered,
            "speculative_accept_count": result.speculative_accept_count,
            "speculative_reject_count": result.speculative_reject_count,
            "self_spec_native_path": result.self_spec_native_path,
            "self_spec_policy": result.self_spec_policy,
            "self_spec_exit_layer": result.self_spec_exit_layer,
            "self_spec_exit_fraction": result.self_spec_exit_fraction,
            "self_spec_draft_tokens": result.self_spec_draft_tokens,
            "coherence_guard_events": result.coherence_guard_events,
            "native_fast_path": result.native_fast_path,
            "parallel_decode_disable_reason": result.parallel_decode_disable_reason,
        },
        "hardware": {
            "openmp_enabled": result.openmp_enabled,
            "avx2_enabled": result.avx2_enabled,
            "avx512_enabled": result.avx512_enabled,
            "mmap_enabled": result.mmap_enabled,
            "native_isa_baseline": result.native_isa_baseline,
            "native_backend_abi_match": result.native_backend_abi_match,
            "loaded_native_library": result.loaded_native_library,
            "native_build_id": result.native_build_id,
            "native_build_sha256": result.native_build_sha256,
            "physical_core_count": result.physical_core_count,
            "logical_core_count": result.logical_core_count,
            "p_core_count": result.p_core_count,
            "affinity_policy": result.affinity_policy,
            "affinity_mode": result.affinity_mode,
            "l3_domain_count": result.l3_domain_count,
            "numa_strict": result.numa_strict,
            "numa_affinity_mode": result.numa_affinity_mode,
            "numa_hugepage": result.numa_hugepage,
            "numa_bind_policy": result.numa_bind_policy,
            "numa_first_touch": result.numa_first_touch,
            "topology_json": result.topology_json,
            "omp_places": result.omp_places,
            "omp_proc_bind": result.omp_proc_bind,
            "omp_max_threads": result.omp_max_threads,
            "omp_dynamic": result.omp_dynamic,
            "omp_active_levels": result.omp_active_levels,
            "perf_event_access": result.perf_event_access,
            "perf_event_access_reason": result.perf_event_access_reason,
            "cpu_governor": result.cpu_governor,
            "thp_mode": result.thp_mode,
            "perf_counter_source": result.perf_counter_source,
            "worker_cpu_mask": result.worker_cpu_mask,
            "orchestrator_cpu_mask": result.orchestrator_cpu_mask,
            "l3_domain_ids_active": list(result.l3_domain_ids_active),
            "autotune_profile_id": result.autotune_profile_id,
            "autotune_source": result.autotune_source,
            "autotune_score": result.autotune_score,
            "autotune_exploration_count": result.autotune_exploration_count,
        },
        "memory": {
            "rss_before_mb": result.rss_before_mb,
            "rss_after_load_mb": result.rss_after_load_mb,
            "rss_after_generate_mb": result.rss_after_generate_mb,
            "mapped_model_bytes": result.mapped_model_bytes,
            "loader_cache_residency_bytes": result.loader_cache_residency_bytes,
            "embedding_materialization_bytes": result.embedding_materialization_bytes,
        },
        "kv_cache": {
            "quantization": result.kv_cache_quantization,
            "kv_used_cells": result.kv_used_cells,
            "kv_fragmentation_ratio": result.kv_fragmentation_ratio,
            "kv_defrag_count": result.kv_defrag_count,
        },
        "quality": {
            "raw_coherence_valid": result.raw_coherence_valid,
            "raw_coherence_issue_count": result.raw_coherence_issue_count,
            "raw_coherence_issues": list(result.raw_coherence_issues),
            "min_new_tokens_before_eos": result.min_new_tokens_before_eos,
            "printable_ratio": result.printable_ratio,
            "repeated_4gram_ratio": result.repeated_4gram_ratio,
            "repeated_8gram_ratio": result.repeated_8gram_ratio,
            "ascii_ratio": result.ascii_ratio,
            "word_count": result.word_count,
            "utf8_valid": result.utf8_valid,
            "leaked_control_text": result.leaked_control_text,
            "leaked_think_tags": result.leaked_think_tags,
            "trivial_completion_detected": result.trivial_completion_detected,
            "coherence_valid": result.coherence_valid,
            "coherence_issue_count": result.coherence_issue_count,
            "coherence_issues": list(result.coherence_issues),
        },
        "measurement": {
            "source": result.measurement_source,
            "valid": result.measurement_valid,
            "issue_count": result.measurement_issue_count,
            "issues": list(result.measurement_issues),
            "perf_stat_artifact": result.perf_stat_artifact,
            "perf_c2c_artifact": result.perf_c2c_artifact,
            "numa_maps_artifact": result.numa_maps_artifact,
            "pmu": {
                "observed": result.pmu_observed,
                "parse_error": result.pmu_parse_error,
                "cycles": result.pmu_cycles,
                "instructions": result.pmu_instructions,
                "ipc": result.pmu_ipc,
                "cache_references": result.pmu_cache_references,
                "cache_misses": result.pmu_cache_misses,
                "cache_miss_rate": result.pmu_cache_miss_rate,
                "context_switches": result.pmu_context_switches,
                "cpu_migrations": result.pmu_cpu_migrations,
                "page_faults": result.pmu_page_faults,
            },
        },
        "hot_path": {
            "sanctioned_backend_path": result.sanctioned_backend_path,
            "tokenizer_backend": result.tokenizer_backend,
            "backend_module": result.backend_module,
            "backend_module_library": result.backend_module_library,
            "backend_module_loaded": result.backend_module_loaded,
            "backend_module_marker_symbol": result.backend_module_marker_symbol,
            "backend_module_marker": result.backend_module_marker,
            "sampling_backend": result.sampling_backend,
            "penalties_backend": result.penalties_backend,
            "suppression_backend": result.suppression_backend,
            "logits_backend": result.logits_backend,
            "full_qsg_enabled": result.full_qsg_enabled,
            "full_graph_enabled": result.full_graph_enabled,
            "qsg_processors_native_enabled": result.qsg_processors_native_enabled,
            "batched_prefill_native_enabled": result.batched_prefill_native_enabled,
            "lm_head_layout": result.lm_head_layout,
            "lm_head_qtype": result.lm_head_qtype,
            "packed_lm_head_calls": result.packed_lm_head_calls,
            "hot_path_numpy_detected": result.hot_path_numpy_detected,
            "python_hot_path_calls": result.python_hot_path_calls,
            "numpy_hot_path_calls": result.numpy_hot_path_calls,
            "python_qsg_forward_calls": result.python_qsg_forward_calls,
            "python_attention_fallback_calls": result.python_attention_fallback_calls,
            "python_ssm_fallback_calls": result.python_ssm_fallback_calls,
            "python_moe_fallback_calls": result.python_moe_fallback_calls,
            "llama_cpp_hot_path_calls": result.llama_cpp_hot_path_calls,
            "proof": dict(result.hot_path_proof),
        },
        "timecrystal": {
            "context_stabilizer_enabled": result.context_stabilizer_enabled,
            "context_stabilizer_mode": result.context_stabilizer_mode,
            "drift_latest": result.drift_latest,
            "drift_mean": result.drift_mean,
            "drift_max": result.drift_max,
            "drift_decay_ratio": result.drift_decay_ratio,
            "drift_damped_blocks": result.drift_damped_blocks,
            "drift_pruned_blocks": result.drift_pruned_blocks,
            "drift_active_tokens": result.drift_active_tokens,
            "drift_overhead_percent": result.drift_overhead_percent,
            "stabilizer_seconds": result.stabilizer_seconds,
            "stabilizer_calls": result.stabilizer_calls,
            "drift_auto_downgrade_events": result.drift_auto_downgrade_events,
            "coconut_enabled": result.coconut_enabled,
            "coconut_paths": result.coconut_paths,
            "coconut_alpha": result.coconut_alpha,
            "coconut_seconds": result.coconut_seconds,
            "coconut_candidate_count": result.coconut_candidate_count,
            "coconut_entropy_mean": result.coconut_entropy_mean,
            "coconut_amplitude_mean": result.coconut_amplitude_mean,
            "coconut_consistency_rejects": result.coconut_consistency_rejects,
            "grover_enabled": result.grover_enabled,
            "grover_top_k": result.grover_top_k,
            "grover_damping": result.grover_damping,
            "grover_calls": result.grover_calls,
            "grover_seconds": result.grover_seconds,
            "grover_candidate_count": result.grover_candidate_count,
            "grover_rescore_delta_mean": result.grover_rescore_delta_mean,
            "grover_timeout_events": result.grover_timeout_events,
            "strict_path_stable": result.strict_path_stable,
        },
        "performance_envelope": dict(result.performance_envelope),
        "performance_twin": dict(result.performance_twin),
        "repo_coupled_runtime": dict(result.repo_coupled_runtime),
        "architecture": {
            "hidden_dim": result.arch_hidden_dim,
            "num_layers": result.arch_num_layers,
            "num_attention_layers": result.arch_num_attention_layers,
            "num_ssm_layers": result.arch_num_ssm_layers,
            "num_moe_layers": result.arch_num_moe_layers,
            "num_heads": result.arch_num_heads,
            "num_kv_heads": result.arch_num_kv_heads,
            "head_dim": result.arch_head_dim,
            "intermediate_dim": result.arch_intermediate_dim,
            "vocab_size": result.arch_vocab_size,
            "ssm_state_dim": result.arch_ssm_state_dim,
            "ssm_conv_kernel": result.arch_ssm_conv_kernel,
            "num_experts": result.arch_num_experts,
            "top_k_experts": result.arch_top_k_experts,
            "rope_dim": result.arch_rope_dim,
            "weight_qtype": result.arch_weight_qtype,
            "lm_head_qtype": result.arch_lm_head_qtype,
        },
        "kernel_efficiency": dict(result.kernel_efficiency),
        "latency_distribution": {
            "p10_ms": result.per_token_latency_p10_ms,
            "p25_ms": result.per_token_latency_p25_ms,
            "p50_ms": result.per_token_latency_p50_ms,
            "p75_ms": result.per_token_latency_p75_ms,
            "p95_ms": result.per_token_latency_p95_ms,
            "p99_ms": result.per_token_latency_p99_ms,
            "min_ms": result.per_token_latency_min_ms,
            "max_ms": result.per_token_latency_max_ms,
            "stddev_ms": result.per_token_latency_stddev_ms,
        },
        "sample": {
            "raw_text": result.raw_sample_text,
            "text": result.sample_text,
            "stop_reason": result.stop_reason,
            "schema_valid": result.schema_valid,
            "tool_call_parse_success": result.tool_call_parse_success,
        },
        "status": {
            "ok": not failures,
            "failure_count": len(failures),
            "issues": failures,
        },
    }


def _result_flat_record(result: BenchmarkResult) -> dict[str, object]:
    payload = dataclasses.asdict(result)
    payload["coherence_issues"] = list(result.coherence_issues)
    payload["measurement_issues"] = list(result.measurement_issues)
    payload["hot_path_proof"] = dict(result.hot_path_proof)
    payload["performance_envelope"] = dict(result.performance_envelope)
    payload["performance_twin"] = dict(result.performance_twin)
    payload["repo_coupled_runtime"] = dict(result.repo_coupled_runtime)
    payload["graph_stage_ms"] = dict(result.graph_stage_ms)
    payload["graph_stage_avg_ms"] = dict(result.graph_stage_avg_ms)
    payload["graph_stage_calls"] = dict(result.graph_stage_calls)
    payload["raw_coherence_issues"] = list(result.raw_coherence_issues)
    return payload


def _summary_rows(
    results: list[BenchmarkResult],
    failures: dict[str, list[str]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(
            (
                result.model,
                result.sampling_profile,
                result.template_name,
                result.granite_moe_mode,
            ),
            [],
        ).append(result)

    rows: list[dict[str, object]] = []
    for (model, profile, template_name, granite_moe_mode), values in sorted(
        grouped.items()
    ):
        pass_count = sum(
            1
            for value in values
            if not failures.get(f"{value.model}#{value.run_index}")
        )
        prefill_gaps = [
            _phase_breakdown(value)["prefill_phase_gap_ms"] for value in values
        ]
        decode_gaps = [
            _phase_breakdown(value)["decode_phase_gap_ms"] for value in values
        ]
        pmu_ipc = [
            float(value.pmu_ipc) for value in values if value.pmu_ipc is not None
        ]
        pmu_context_switches = [
            float(value.pmu_context_switches)
            for value in values
            if value.pmu_context_switches is not None
        ]
        pmu_cpu_migrations = [
            float(value.pmu_cpu_migrations)
            for value in values
            if value.pmu_cpu_migrations is not None
        ]
        pmu_cache_miss_rate = [
            float(value.pmu_cache_miss_rate)
            for value in values
            if value.pmu_cache_miss_rate is not None
        ]
        rows.append(
            {
                "model": model,
                "sampling_profile": profile,
                "template_name": template_name,
                "granite_moe_mode": granite_moe_mode,
                "runs": len(values),
                "passes": pass_count,
                "failures": len(values) - pass_count,
                "avg_ttft_ms": statistics.mean(value.ttft_ms for value in values),
                "avg_effective_prefill_tps": statistics.mean(
                    value.effective_prefill_throughput_tps for value in values
                ),
                "avg_decode_tps": statistics.mean(
                    value.decode_throughput_tps for value in values
                ),
                "avg_end_to_end_tps": statistics.mean(
                    value.end_to_end_throughput_tps for value in values
                ),
                "avg_prompt_cache_hit_ratio": statistics.mean(
                    value.prompt_cache_hit_ratio for value in values
                ),
                "avg_prompt_cache_reuse_ratio": statistics.mean(
                    value.prompt_cache_reuse_ratio for value in values
                ),
                "avg_prefill_phase_gap_ms": statistics.mean(prefill_gaps),
                "avg_decode_phase_gap_ms": statistics.mean(decode_gaps),
                "avg_runtime_vs_wall_delta_ms": statistics.mean(
                    value.runtime_vs_wall_delta_ms for value in values
                ),
                "avg_runtime_thread_switches": statistics.mean(
                    value.runtime_thread_switches for value in values
                ),
                "avg_pmu_ipc": statistics.mean(pmu_ipc) if pmu_ipc else None,
                "avg_pmu_context_switches": (
                    statistics.mean(pmu_context_switches)
                    if pmu_context_switches
                    else None
                ),
                "avg_pmu_cpu_migrations": (
                    statistics.mean(pmu_cpu_migrations) if pmu_cpu_migrations else None
                ),
                "avg_pmu_cache_miss_rate": (
                    statistics.mean(pmu_cache_miss_rate)
                    if pmu_cache_miss_rate
                    else None
                ),
                "avg_drift_mean": statistics.mean(value.drift_mean for value in values),
                "avg_drift_overhead_percent": statistics.mean(
                    value.drift_overhead_percent for value in values
                ),
            }
        )
    return rows


def _speculative_acceptance_rows(
    results: list[BenchmarkResult],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(
            (
                str(result.prompt_category or "unknown"),
                str(result.temperature_band or _temperature_band(result.temperature)),
                str(result.generation_mode or "ar_verify"),
                str(
                    result.benchmark_label
                    or benchmark_label_for_mode(result.generation_mode).value
                ),
            ),
            [],
        ).append(result)

    rows: list[dict[str, object]] = []
    for (
        prompt_category,
        temperature_band,
        generation_mode,
        benchmark_label,
    ), values in sorted(grouped.items()):
        accepted = sum(int(value.accepted_parallel_tokens) for value in values)
        rejected = sum(int(value.rejected_parallel_tokens) for value in values)
        proposed = sum(int(value.proposed_parallel_tokens) for value in values)
        speculative_attempts = accepted + rejected
        rows.append(
            {
                "prompt_category": prompt_category,
                "temperature_band": temperature_band,
                "generation_mode": generation_mode,
                "benchmark_label": benchmark_label,
                "runs": len(values),
                "proposed_parallel_tokens": proposed,
                "accepted_parallel_tokens": accepted,
                "rejected_parallel_tokens": rejected,
                "speculative_attempts": speculative_attempts,
                "acceptance_rate": (
                    float(accepted) / float(speculative_attempts)
                    if speculative_attempts > 0
                    else 0.0
                ),
                "avg_draft_confidence_mean": statistics.mean(
                    float(value.draft_confidence_mean) for value in values
                ),
                "min_draft_confidence": (
                    min(
                        float(value.draft_confidence_min)
                        for value in values
                        if float(value.draft_confidence_min) > 0.0
                    )
                    if any(float(value.draft_confidence_min) > 0.0 for value in values)
                    else 0.0
                ),
                "avg_decode_tps": statistics.mean(
                    value.decode_throughput_tps for value in values
                ),
                "avg_end_to_end_tps": statistics.mean(
                    value.end_to_end_throughput_tps for value in values
                ),
            }
        )
    return rows


def _aggregate_non_ar_metrics(
    values: list[BenchmarkResult],
) -> dict[str, object] | None:
    if not values:
        return None
    return {
        "runs": len(values),
        "decode_tps": statistics.mean(value.decode_throughput_tps for value in values),
        "ttft_ms": statistics.mean(value.ttft_ms for value in values),
        "coherence_pass_rate": statistics.mean(
            1.0 if value.coherence_valid else 0.0 for value in values
        ),
        "quality_guard_rate": statistics.mean(
            1.0 if value.quality_guard_triggered else 0.0 for value in values
        ),
        "blockwise_convergence_rate": statistics.mean(
            float(value.blockwise_convergence_rate) for value in values
        ),
        "draft_source": str(
            next((value.draft_source for value in values if value.draft_source), "")
        ),
        "native_hot_path_owned": all(
            int(value.python_hot_path_calls) <= 0
            and int(value.numpy_hot_path_calls) <= 0
            for value in values
        ),
        "python_hot_path_calls_max": max(
            int(value.python_hot_path_calls) for value in values
        ),
        "numpy_hot_path_calls_max": max(
            int(value.numpy_hot_path_calls) for value in values
        ),
    }


def _non_ar_decision_rows(results: list[BenchmarkResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(
            (
                str(result.model or "unknown"),
                str(result.prompt_category or "unknown"),
                str(result.temperature_band or _temperature_band(result.temperature)),
            ),
            [],
        ).append(result)

    rows: list[dict[str, object]] = []
    for (model, prompt_category, temperature_band), values in sorted(grouped.items()):
        candidate_rows = [
            value
            for value in values
            if str(value.generation_mode or "") == "block_diffusion"
        ]
        ar_rows = [
            value
            for value in values
            if str(value.generation_mode or "") in AR_DECISION_MODES
        ]
        self_spec_rows = [
            value
            for value in values
            if str(value.generation_mode or "") in SELF_SPEC_DECISION_MODES
        ]
        candidate = _aggregate_non_ar_metrics(candidate_rows)
        ar = _aggregate_non_ar_metrics(ar_rows)
        self_spec = _aggregate_non_ar_metrics(self_spec_rows)
        decode_speedup_vs_ar = 0.0
        decode_speedup_vs_self_spec = 0.0
        recommended_mode = "insufficient_candidate_coverage"
        reason = "no measured block_diffusion runs"
        candidate_native_hot_path_owned = False
        python_hot_path_calls_max = 0
        numpy_hot_path_calls_max = 0
        production_blockers = list(PHASE8_PRODUCTION_BLOCKERS)
        masked_runtime_ready = any(
            bool(getattr(value, "masked_generation_ready", False)) for value in values
        ) or any(
            str(value.generation_mode or "").strip().lower() == "masked_diffusion"
            for value in values
        )
        if not masked_runtime_ready:
            production_blockers.append("masked_generation_runtime_missing")
        if candidate is not None:
            candidate_native_hot_path_owned = bool(
                candidate.get("native_hot_path_owned", False)
            )
            python_hot_path_calls_max = int(
                candidate.get("python_hot_path_calls_max", 0) or 0
            )
            numpy_hot_path_calls_max = int(
                candidate.get("numpy_hot_path_calls_max", 0) or 0
            )
            if not candidate_native_hot_path_owned:
                production_blockers.append("python_or_numpy_hot_path_owned")
            ar_decode = float((ar or {}).get("decode_tps", 0.0) or 0.0)
            self_spec_decode = float((self_spec or {}).get("decode_tps", 0.0) or 0.0)
            decode_speedup_vs_ar = (
                float(candidate["decode_tps"]) / ar_decode if ar_decode > 0.0 else 0.0
            )
            decode_speedup_vs_self_spec = (
                float(candidate["decode_tps"]) / self_spec_decode
                if self_spec_decode > 0.0
                else 0.0
            )
            if not ar_rows and not self_spec_rows:
                recommended_mode = "insufficient_baseline_coverage"
                reason = "no AR or self-spec comparison rows"
            else:
                best_baseline_decode = max(ar_decode, self_spec_decode)
                ttft_reference = min(
                    [
                        value
                        for value in (
                            float((ar or {}).get("ttft_ms", 0.0) or 0.0),
                            float((self_spec or {}).get("ttft_ms", 0.0) or 0.0),
                        )
                        if value > 0.0
                    ]
                    or [0.0]
                )
                coherence_reference = max(
                    float((ar or {}).get("coherence_pass_rate", 0.0) or 0.0),
                    float((self_spec or {}).get("coherence_pass_rate", 0.0) or 0.0),
                )
                quality_reference = min(
                    [
                        value
                        for value in (
                            float((ar or {}).get("quality_guard_rate", 0.0) or 0.0),
                            float(
                                (self_spec or {}).get("quality_guard_rate", 0.0) or 0.0
                            ),
                        )
                        if value > 0.0
                    ]
                    or [0.0]
                )
                decode_gate = best_baseline_decode <= 0.0 or float(
                    candidate["decode_tps"]
                ) >= (best_baseline_decode * NON_AR_DECODE_SPEEDUP_MIN)
                ttft_gate = ttft_reference <= 0.0 or float(candidate["ttft_ms"]) <= (
                    ttft_reference * NON_AR_TTFT_REGRESSION_MAX
                )
                coherence_gate = (
                    float(candidate["coherence_pass_rate"]) >= coherence_reference
                )
                quality_gate = float(candidate["quality_guard_rate"]) <= (
                    quality_reference + 0.05
                )
                convergence_gate = float(candidate["blockwise_convergence_rate"]) >= (
                    NON_AR_MIN_BLOCK_CONVERGENCE_RATE
                )
                if (
                    decode_gate
                    and ttft_gate
                    and coherence_gate
                    and quality_gate
                    and convergence_gate
                ):
                    recommended_mode = "block_diffusion"
                    reason = (
                        "throughput, latency, and quality gates passed; "
                        "Phase 8 remains research-only"
                    )
                else:
                    recommended_mode = (
                        "parallel_hybrid"
                        if self_spec_decode >= ar_decode
                        else "ar_verify"
                    )
                    failed = [
                        name
                        for name, ok in (
                            ("decode", decode_gate),
                            ("ttft", ttft_gate),
                            ("coherence", coherence_gate),
                            ("quality_guard", quality_gate),
                            ("convergence", convergence_gate),
                        )
                        if not ok
                    ]
                    reason = "failed gates: " + ",".join(failed)

        rows.append(
            {
                "model": model,
                "prompt_category": prompt_category,
                "temperature_band": temperature_band,
                "candidate_runs": len(candidate_rows),
                "recommended_mode": recommended_mode,
                "reason": reason,
                "decision_scope": PHASE8_DECISION_SCOPE,
                "production_ready": False,
                "production_blockers": production_blockers,
                "native_hot_path_owned": candidate_native_hot_path_owned,
                "python_hot_path_calls_max": python_hot_path_calls_max,
                "numpy_hot_path_calls_max": numpy_hot_path_calls_max,
                "decode_speedup_vs_ar": decode_speedup_vs_ar,
                "decode_speedup_vs_self_spec": decode_speedup_vs_self_spec,
                "blockwise_convergence_rate": float(
                    (candidate or {}).get("blockwise_convergence_rate", 0.0) or 0.0
                ),
                "draft_source": str((candidate or {}).get("draft_source", "") or ""),
            }
        )
    return rows


def _build_report(
    results: list[BenchmarkResult],
    failures: dict[str, list[str]],
    *,
    prompt: str,
    criteria: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_REPORT_SCHEMA,
        "host": _report_host_facts(results),
        "prompt": prompt,
        "criteria": criteria,
        "flat_results": [_result_flat_record(result) for result in results],
        "results": [
            _result_record(
                result, issues=failures.get(f"{result.model}#{result.run_index}", [])
            )
            for result in results
        ],
        "summary": _summary_rows(results, failures),
        "speculative_acceptance": _speculative_acceptance_rows(results),
        "non_ar_decision": _non_ar_decision_rows(results),
        "failures": failures,
        "failure_keys": sorted(failures),
        "failure_count": len(failures),
        "pass_count": len(results) - len(failures),
    }


def _runtime_failure_issue_key(exc: BaseException) -> str:
    detail = str(exc or "").strip().lower()
    if not detail:
        detail = exc.__class__.__name__.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", detail).strip("_")
    return normalized[:96] or "runtime_exception"


def _runtime_failure_plan_mode(exc: BaseException) -> str:
    match = re.search(r"plan_mode=([a-z0-9_]+)", str(exc or "").lower())
    if match is None:
        return ""
    return str(match.group(1) or "").strip().lower()


def _runtime_failure_result(
    *,
    model: str,
    prompt: str,
    max_new_tokens: int,
    run_index: int,
    context_length: int,
    decode_threads: int | None,
    batch_threads: int | None,
    ubatch: int | None,
    profile_name: str,
    params: dict[str, Any],
    prompt_tokens: list[int],
    runtime: dict[str, Any],
    load_seconds: float,
    generate_seconds: float,
    rss_before: float,
    rss_after_load: float,
    rss_after_generate: float,
    exc: BaseException,
) -> BenchmarkResult:
    issue_key = _runtime_failure_issue_key(exc)
    plan_mode = _runtime_failure_plan_mode(exc)
    benchmark_label = ""
    if plan_mode:
        try:
            benchmark_label = benchmark_label_for_mode(plan_mode).value
        except Exception:
            benchmark_label = plan_mode
    sanctioned_backend_path = str(
        runtime.get(
            "sanctioned_backend_path",
            dict(runtime.get("hot_path_proof", {})).get("sanctioned_backend_path", ""),
        )
    )
    tokenizer_backend = str(
        runtime.get(
            "tokenizer_backend",
            dict(runtime.get("hot_path_proof", {})).get("tokenizer_backend", ""),
        )
    )
    measurement_issues = [
        "benchmark_runtime_exception",
        f"runtime_error_type={exc.__class__.__name__}",
        f"runtime_error_key={issue_key}",
    ]
    if plan_mode:
        measurement_issues.append(f"planner_mode={plan_mode}")
    coherence_issues = [
        "generation_failed_runtime_exception",
        f"runtime_error_type={exc.__class__.__name__}",
    ]
    return BenchmarkResult(
        model=model,
        threads=int(decode_threads or max(1, os.cpu_count() or 1)),
        max_new_tokens=max_new_tokens,
        run_index=run_index,
        load_seconds=load_seconds,
        generate_seconds=generate_seconds,
        new_tokens=0,
        tokens_per_second=0.0,
        context_length=int(context_length),
        prompt_tokens=len(prompt_tokens),
        runtime_total_seconds=float(runtime.get("total_seconds", 0.0) or 0.0),
        runtime_prefill_seconds=float(runtime.get("prefill_seconds", 0.0) or 0.0),
        runtime_decode_seconds=float(runtime.get("decode_seconds", 0.0) or 0.0),
        first_token_latency_seconds=float(
            runtime.get("ttft_seconds", runtime.get("first_token_latency_seconds", 0.0))
            or 0.0
        ),
        decode_threads=int(decode_threads or 0),
        batch_threads=int(batch_threads or 0),
        ubatch=int(ubatch or 0),
        requested_decode_threads=int(decode_threads or 0),
        requested_batch_threads=int(batch_threads or 0),
        requested_ubatch=int(ubatch or 0),
        openmp_enabled=bool(runtime.get("openmp_enabled", False)),
        avx2_enabled=bool(runtime.get("avx2_enabled", False)),
        avx512_enabled=bool(runtime.get("avx512_enabled", False)),
        mmap_enabled=bool(runtime.get("mmap_enabled", False)),
        parallel_decode=bool(runtime.get("parallel_decode", False)),
        speculative_decode=bool(runtime.get("speculative_decode", False)),
        generation_mode=plan_mode,
        benchmark_label=benchmark_label,
        prompt_category=_prompt_category(prompt),
        temperature_band=_temperature_band(float(params.get("temperature", 0.0) or 0.0)),
        prompt_cache_hit=bool(runtime.get("prompt_cache_hit", False)),
        prompt_cache_hits=int(runtime.get("prompt_cache_hits", 0) or 0),
        prompt_cache_misses=int(runtime.get("prompt_cache_misses", 0) or 0),
        prompt_cache_lookups=int(runtime.get("prompt_cache_lookups", 0) or 0),
        prompt_cache_hit_ratio=float(runtime.get("prompt_cache_hit_ratio", 0.0) or 0.0),
        prompt_cache_reused_tokens=int(
            runtime.get("prompt_cache_reused_tokens", 0) or 0
        ),
        prompt_cache_reuse_ratio=float(
            runtime.get("prompt_cache_reuse_ratio", 0.0) or 0.0
        ),
        sampling_profile=profile_name,
        temperature=float(params.get("temperature", 0.0) or 0.0),
        top_p=float(params.get("top_p", 1.0) or 1.0),
        top_k=int(params.get("top_k", 0) or 0),
        min_p=float(params.get("min_p", 0.0) or 0.0),
        presence_penalty=float(params.get("presence_penalty", 0.0) or 0.0),
        repetition_penalty=float(params.get("repetition_penalty", 1.0) or 1.0),
        rss_before_mb=rss_before,
        rss_after_load_mb=rss_after_load,
        rss_after_generate_mb=rss_after_generate,
        sample_text="",
        printable_ratio=0.0,
        ascii_ratio=0.0,
        word_count=0,
        utf8_valid=True,
        coherence_valid=False,
        coherence_issues=coherence_issues,
        coherence_issue_count=len(coherence_issues),
        stop_reason="runtime_error",
        measurement_valid=False,
        measurement_issues=measurement_issues,
        measurement_issue_count=len(measurement_issues),
        measurement_source="runtime_exception",
        native_build_id=str(runtime.get("native_build_id", "")),
        native_build_sha256=str(runtime.get("native_build_sha256", "")),
        loaded_native_library=str(runtime.get("loaded_native_library", "")),
        native_isa_baseline=str(runtime.get("native_isa_baseline", "")),
        native_backend_abi_match=bool(runtime.get("native_backend_abi_match", False)),
        sanctioned_backend_path=sanctioned_backend_path,
        tokenizer_backend=tokenizer_backend,
        backend_module=str(runtime.get("backend_module", "")),
        backend_module_library=str(runtime.get("backend_module_library", "")),
        backend_module_loaded=bool(runtime.get("backend_module_loaded", False)),
        sampling_backend=str(runtime.get("sampling_backend", "")),
        penalties_backend=str(runtime.get("penalties_backend", "")),
        suppression_backend=str(runtime.get("suppression_backend", "")),
        logits_backend=str(runtime.get("logits_backend", "")),
        full_qsg_enabled=bool(runtime.get("full_qsg_enabled", False)),
        full_graph_enabled=bool(runtime.get("full_graph_enabled", False)),
        qsg_processors_native_enabled=bool(
            runtime.get("qsg_processors_native_enabled", False)
        ),
        batched_prefill_native_enabled=bool(
            runtime.get("batched_prefill_native_enabled", False)
        ),
        context_stabilizer_enabled=bool(
            runtime.get("context_stabilizer_enabled", False)
        ),
        context_stabilizer_mode=str(runtime.get("context_stabilizer_mode", "")),
        coconut_enabled=bool(runtime.get("coconut_enabled", False)),
        grover_enabled=bool(runtime.get("grover_enabled", False)),
        strict_path_stable=bool(runtime.get("strict_path_stable", False)),
        hot_path_proof=dict(runtime.get("hot_path_proof", {})),
        performance_envelope=dict(runtime.get("performance_envelope", {})),
        performance_twin=dict(runtime.get("performance_twin", {})),
        repo_coupled_runtime=dict(runtime.get("repo_coupled_runtime", {})),
        graph_stage_ms=dict(runtime.get("graph_stage_ms", {})),
        graph_stage_avg_ms=dict(runtime.get("graph_stage_avg_ms", {})),
        graph_stage_calls=dict(runtime.get("graph_stage_calls", {})),
        raw_sample_text=f"{exc.__class__.__name__}: {exc}",
    )


def _model_sampling_defaults(
    model: str,
    requested_profile: str | None,
    *,
    coherence_first: bool = False,
) -> tuple[str, dict]:
    lower = model.lower()
    base = {
        "temperature": float(GENERATION_PARAMS.get("temperature", 0.0)),
        "top_p": float(GENERATION_PARAMS.get("top_p", 0.9)),
        "top_k": int(GENERATION_PARAMS.get("top_k", 40)),
        "min_p": float(GENERATION_PARAMS.get("min_p", 0.0)),
        "presence_penalty": float(GENERATION_PARAMS.get("presence_penalty", 0.0)),
        "repetition_penalty": float(GENERATION_PARAMS.get("repetition_penalty", 1.0)),
    }

    if "qwen3.5" in lower or "qwen35" in lower:
        default_profile = str(
            GENERATION_PARAMS.get("qwen35_sampling_profile", "instruct_deterministic")
        )
        if coherence_first:
            default_profile = "instruct_deterministic"
        profile_name = requested_profile or default_profile
        profile = QWEN35_SAMPLING_PROFILES.get(profile_name)
        if profile is None:
            profile_name = default_profile
            profile = QWEN35_SAMPLING_PROFILES[profile_name]
        merged = dict(base)
        merged.update(profile)
        return profile_name, merged

    if "granite4" in lower:
        profile_name = requested_profile or str(
            GENERATION_PARAMS.get("granite4_sampling_profile", "coding_deterministic")
        )
        profile = GRANITE4_SAMPLING_PROFILES.get(profile_name)
        if profile is None:
            profile_name = "coding_deterministic"
            profile = GRANITE4_SAMPLING_PROFILES[profile_name]
        merged = dict(base)
        merged.update(profile)
        return profile_name, merged

    return requested_profile or "generation_params", base


def _run_once(
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float | None = None,
    sampling_profile: str | None = None,
    override_params: dict | None = None,
    run_index: int = 0,
    context_length: int = 2048,
    decode_threads: int | None = None,
    batch_threads: int | None = None,
    ubatch: int | None = None,
    native_fast_path: bool = False,
    disable_logits_processors: bool | None = None,
    disable_token_penalties: bool | None = None,
    force_parallel_decode: bool = False,
    min_new_tokens_before_eos: int | None = None,
    coherence_first: bool = False,
    self_artifact_dir: str | None = None,
    capture_self_numa_map: bool = False,
) -> BenchmarkResult:
    _trace(
        f"starting run_index={run_index} model={model} "
        f"threads={decode_threads or 'auto'}x{batch_threads or 'auto'}x{ubatch or 'auto'}"
    )
    restore_env = {
        "ANVIL_NUM_THREADS": os.environ.get("ANVIL_NUM_THREADS"),
        "ANVIL_NUM_THREADS_DECODE": os.environ.get("ANVIL_NUM_THREADS_DECODE"),
        "ANVIL_NUM_THREADS_BATCH": os.environ.get("ANVIL_NUM_THREADS_BATCH"),
        "ANVIL_NUM_UBATCH": os.environ.get("ANVIL_NUM_UBATCH"),
        "ANVIL_NATIVE_FAST_PATH": os.environ.get("ANVIL_NATIVE_FAST_PATH"),
        "ANVIL_MIN_NEW_TOKENS_BEFORE_EOS": os.environ.get(
            "ANVIL_MIN_NEW_TOKENS_BEFORE_EOS"
        ),
        "ANVIL_DISABLE_LOGITS_PROCESSORS": os.environ.get(
            "ANVIL_DISABLE_LOGITS_PROCESSORS"
        ),
        "ANVIL_DISABLE_TOKEN_PENALTIES": os.environ.get(
            "ANVIL_DISABLE_TOKEN_PENALTIES"
        ),
        "ANVIL_FORCE_PARALLEL_DECODE": os.environ.get("ANVIL_FORCE_PARALLEL_DECODE"),
    }
    os.environ["ANVIL_NUM_THREADS"] = "0"
    if decode_threads is None:
        os.environ.pop("ANVIL_NUM_THREADS_DECODE", None)
    else:
        os.environ["ANVIL_NUM_THREADS_DECODE"] = str(int(decode_threads))
    if batch_threads is None:
        os.environ.pop("ANVIL_NUM_THREADS_BATCH", None)
    else:
        os.environ["ANVIL_NUM_THREADS_BATCH"] = str(int(batch_threads))
    if ubatch is None:
        os.environ.pop("ANVIL_NUM_UBATCH", None)
    else:
        os.environ["ANVIL_NUM_UBATCH"] = str(int(ubatch))
    if native_fast_path:
        os.environ["ANVIL_NATIVE_FAST_PATH"] = "1"
    elif (
        "ANVIL_NATIVE_FAST_PATH" in restore_env
        and restore_env["ANVIL_NATIVE_FAST_PATH"] is None
    ):
        os.environ.pop("ANVIL_NATIVE_FAST_PATH", None)
    if min_new_tokens_before_eos is None:
        if restore_env["ANVIL_MIN_NEW_TOKENS_BEFORE_EOS"] is None:
            os.environ.pop("ANVIL_MIN_NEW_TOKENS_BEFORE_EOS", None)
    else:
        os.environ["ANVIL_MIN_NEW_TOKENS_BEFORE_EOS"] = str(
            int(min_new_tokens_before_eos)
        )
    if disable_logits_processors is True:
        os.environ["ANVIL_DISABLE_LOGITS_PROCESSORS"] = "1"
    elif disable_logits_processors is False:
        os.environ["ANVIL_DISABLE_LOGITS_PROCESSORS"] = "0"
    if disable_token_penalties is True:
        os.environ["ANVIL_DISABLE_TOKEN_PENALTIES"] = "1"
    elif disable_token_penalties is False:
        os.environ["ANVIL_DISABLE_TOKEN_PENALTIES"] = "0"
    if force_parallel_decode:
        os.environ["ANVIL_FORCE_PARALLEL_DECODE"] = "1"

    profile_name, params = _model_sampling_defaults(
        model,
        sampling_profile,
        coherence_first=coherence_first,
    )
    merged_overrides = dict(override_params or {})
    if temperature is not None:
        merged_overrides["temperature"] = temperature
    for key, value in merged_overrides.items():
        if value is not None:
            params[key] = value
    sample_seed = int(params.get("seed", GENERATION_PARAMS.get("seed", 0)) or 0)

    engine = None
    runtime: dict = {}
    rss_before = _rss_mb()
    rss_after_load = rss_before
    rss_after_generate = rss_before
    load_seconds = 0.0
    generate_seconds = 0.0
    active_threads = max(1, os.cpu_count() or 1)
    prompt_tokens: list[int] = []
    out_tokens: list[int] = []
    text = ""
    raw_text = ""
    arch_config: dict[str, int | str] = {}
    numa_maps_artifact = ""
    runtime_exception: Exception | None = None
    try:
        start = time.perf_counter()
        _trace(
            f"initializing NativeQSGEngine model={model} context_length={context_length}"
        )
        engine = NativeQSGEngine(model, context_length=int(context_length))
        load_seconds = time.perf_counter() - start
        rss_after_load = _rss_mb()
        _trace(
            f"engine loaded in {load_seconds:.3f}s rss_after_load_mb={rss_after_load:.2f}"
        )

        active_threads = int(
            getattr(
                engine,
                "num_threads_decode",
                getattr(engine, "num_threads", max(1, os.cpu_count() or 1)),
            )
        )

        def _pick_int(obj, *names: str) -> int:
            for name in names:
                value = getattr(obj, name, 0)
                if value:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        continue
            return 0

        def _pick_str(obj, *names: str) -> str:
            for name in names:
                value = getattr(obj, name, "")
                if value:
                    return str(value)
            return ""

        def _metadata_int(
            metadata: dict[str, object], suffixes: tuple[str, ...]
        ) -> int:
            for key, raw in metadata.items():
                key_str = str(key)
                if any(key_str.endswith(suffix) for suffix in suffixes):
                    try:
                        value = int(raw)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        return value
            return 0

        cfg = getattr(engine, "_model_config", None)
        if cfg is None:
            cfg = getattr(engine, "profile", None)
        if cfg is not None:
            arch_config = {
                "hidden_dim": _pick_int(cfg, "hidden_dim", "embedding_dim", "dim"),
                "num_layers": _pick_int(cfg, "num_layers", "n_layers"),
                "num_attention_layers": _pick_int(
                    cfg,
                    "num_attention_layers",
                    "n_attention_layers",
                ),
                "num_ssm_layers": _pick_int(cfg, "num_ssm_layers"),
                "num_moe_layers": _pick_int(cfg, "num_moe_layers"),
                "num_heads": _pick_int(cfg, "num_heads", "n_heads"),
                "num_kv_heads": _pick_int(cfg, "num_kv_heads", "n_kv_heads"),
                "head_dim": _pick_int(cfg, "head_dim"),
                "intermediate_dim": _pick_int(cfg, "intermediate_dim", "ffn_dim"),
                "vocab_size": _pick_int(cfg, "vocab_size"),
                "ssm_state_dim": _pick_int(cfg, "ssm_state_dim"),
                "ssm_conv_kernel": _pick_int(
                    cfg,
                    "ssm_conv_kernel",
                    "ssm_conv_kernel_size",
                ),
                "num_experts": _pick_int(cfg, "num_experts"),
                "top_k_experts": _pick_int(
                    cfg,
                    "top_k_experts",
                    "num_experts_per_tok",
                ),
                "rope_dim": _pick_int(cfg, "rope_dim"),
                "weight_qtype": _pick_str(cfg, "weight_qtype", "qtype"),
                "lm_head_qtype": _pick_str(cfg, "lm_head_qtype"),
            }
            if (
                arch_config["head_dim"] <= 0
                and arch_config["hidden_dim"] > 0
                and arch_config["num_heads"] > 0
            ):
                arch_config["head_dim"] = int(arch_config["hidden_dim"]) // int(
                    arch_config["num_heads"]
                )
            if (
                arch_config["num_moe_layers"] <= 0
                and bool(getattr(cfg, "has_moe", False))
                and arch_config["num_layers"] > 0
            ):
                arch_config["num_moe_layers"] = int(arch_config["num_layers"])

        loader = getattr(engine, "loader", None)
        metadata = {}
        if loader is not None and hasattr(loader, "get_metadata"):
            meta_candidate = loader.get_metadata()
            if isinstance(meta_candidate, dict):
                metadata = meta_candidate

        if arch_config.get("intermediate_dim", 0) <= 0 and metadata:
            arch_config["intermediate_dim"] = _metadata_int(
                metadata,
                (
                    "feed_forward_length",
                    "ffn_length",
                    "ffn_dim",
                    "intermediate_size",
                ),
            )
        if arch_config.get("num_experts", 0) <= 0 and metadata:
            arch_config["num_experts"] = _metadata_int(
                metadata,
                ("expert_count", "num_experts"),
            )
        if arch_config.get("top_k_experts", 0) <= 0 and metadata:
            arch_config["top_k_experts"] = _metadata_int(
                metadata,
                ("expert_used_count", "num_experts_used", "top_k_experts"),
            )

        prompt_tokens = (
            engine.prepare_prompt_tokens(prompt)
            if hasattr(engine, "prepare_prompt_tokens")
            else engine.tokenize(prompt)
        )
        _trace(f"prepared prompt tokens={len(prompt_tokens)}")
        t0 = time.perf_counter()
        _trace(
            f"starting generation max_new_tokens={max_new_tokens} "
            f"profile={profile_name} seed={sample_seed}"
        )
        out_tokens = engine.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=float(params["temperature"]),
            top_p=float(params["top_p"]),
            top_k=int(params["top_k"]),
            min_p=float(params["min_p"]),
            presence_penalty=float(params["presence_penalty"]),
            repetition_penalty=float(params["repetition_penalty"]),
            seed=sample_seed,
        )
        generate_seconds = time.perf_counter() - t0
        rss_after_generate = _rss_mb()
        _trace(
            f"generation finished in {generate_seconds:.3f}s "
            f"rss_after_generate_mb={rss_after_generate:.2f}"
        )
        runtime = (
            engine.get_runtime_status() if hasattr(engine, "get_runtime_status") else {}
        )
        _trace(
            "captured runtime telemetry "
            f"keys={len(runtime)} new_tokens={max(0, len(out_tokens) - len(prompt_tokens))}"
        )
        new_tokens = max(0, len(out_tokens) - len(prompt_tokens))
        decode_generated_tokens = getattr(engine, "decode_generated_tokens", None)
        if callable(decode_generated_tokens):
            raw_text = str(decode_generated_tokens(out_tokens[len(prompt_tokens) :]))
        else:
            raw_text = engine.detokenize(out_tokens[len(prompt_tokens) :])
        text = postprocess_strict_native_response(raw_text, model_name=model)
        if capture_self_numa_map:
            numa_maps_artifact = _capture_self_numa_artifacts(
                pathlib.Path(self_artifact_dir) if self_artifact_dir else None
            )
            _trace(f"captured self NUMA artifacts -> {numa_maps_artifact or 'none'}")
    except Exception as exc:
        runtime_exception = exc
        rss_after_generate = _rss_mb()
        if engine is not None and hasattr(engine, "get_runtime_status"):
            try:
                runtime = engine.get_runtime_status()
            except Exception:
                runtime = runtime or {}
        print(
            f"[native_qsg_benchmark] run failed model={model} run_index={run_index}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
    finally:
        if engine is not None:
            try:
                engine.close()
            except Exception:
                pass
        for key, value in restore_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if runtime_exception is not None:
        return _runtime_failure_result(
            model=model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            run_index=run_index,
            context_length=context_length,
            decode_threads=decode_threads,
            batch_threads=batch_threads,
            ubatch=ubatch,
            profile_name=profile_name,
            params=params,
            prompt_tokens=prompt_tokens,
            runtime=runtime,
            load_seconds=load_seconds,
            generate_seconds=generate_seconds,
            rss_before=rss_before,
            rss_after_load=rss_after_load,
            rss_after_generate=rss_after_generate,
            exc=runtime_exception,
        )
    new_tokens = max(0, len(out_tokens) - len(prompt_tokens))
    runtime_total_seconds = float(runtime.get("total_seconds", 0.0))
    runtime_prefill_seconds = float(runtime.get("prefill_seconds", 0.0))
    runtime_decode_seconds = float(runtime.get("decode_seconds", 0.0))
    ttft_seconds = float(
        runtime.get(
            "ttft_seconds",
            runtime.get("first_token_latency_seconds", 0.0),
        )
    )
    prompt_cache_hits = int(runtime.get("prompt_cache_hits", 0))
    prompt_cache_misses = int(runtime.get("prompt_cache_misses", 0))
    prompt_cache_lookups = int(
        runtime.get("prompt_cache_lookups", prompt_cache_hits + prompt_cache_misses)
    )
    prompt_cache_reused_tokens = int(runtime.get("prompt_cache_reused_tokens", 0))
    effective_prefill_tokens = int(
        runtime.get(
            "effective_prefill_tokens",
            max(0, len(prompt_tokens) - prompt_cache_reused_tokens),
        )
    )
    prefill_tps = float(runtime.get("prefill_throughput_tps", 0.0))
    if prefill_tps <= 0.0 and runtime_prefill_seconds > 0.0 and prompt_tokens:
        prefill_tps = float(len(prompt_tokens)) / runtime_prefill_seconds
    effective_prefill_tps = float(runtime.get("effective_prefill_throughput_tps", 0.0))
    if (
        effective_prefill_tps <= 0.0
        and runtime_prefill_seconds > 0.0
        and effective_prefill_tokens > 0
    ):
        effective_prefill_tps = (
            float(effective_prefill_tokens) / runtime_prefill_seconds
        )
    decode_tps = float(runtime.get("decode_throughput_tps", 0.0))
    if decode_tps <= 0.0 and runtime_decode_seconds > 0.0 and new_tokens > 0:
        decode_tps = float(new_tokens) / runtime_decode_seconds
    end_to_end_tps = float(runtime.get("end_to_end_throughput_tps", 0.0))
    if end_to_end_tps <= 0.0 and runtime_total_seconds > 0.0 and new_tokens > 0:
        end_to_end_tps = float(new_tokens) / runtime_total_seconds
    measurement_source = "runtime"
    if not runtime:
        measurement_source = "wallclock_only"
    elif runtime_total_seconds <= 0.0:
        measurement_source = "runtime_partial"
    measurement_issues = list(runtime.get("measurement_issues", []))
    runtime_measurement_valid = runtime.get("measurement_valid")
    if runtime_measurement_valid is None:
        measurement_issues.append("measurement_valid_missing")
    if measurement_source != "runtime":
        measurement_issues.append(f"measurement_source={measurement_source}")
    measurement_issues = sorted(
        set(str(issue) for issue in measurement_issues if issue)
    )
    measurement_valid = bool(runtime_measurement_valid) and not measurement_issues
    runtime_vs_wall_delta_ms = 0.0
    if runtime_total_seconds > 0.0 and generate_seconds > 0.0:
        runtime_vs_wall_delta_ms = (
            abs(generate_seconds - runtime_total_seconds) * 1000.0
        )
    graph_stage_ms = {
        str(name): float(value)
        for name, value in dict(runtime.get("graph_stage_ms", {})).items()
    }
    graph_stage_avg_ms = {
        str(name): float(value)
        for name, value in dict(runtime.get("graph_stage_avg_ms", {})).items()
    }
    graph_stage_calls = {
        str(name): int(value)
        for name, value in dict(runtime.get("graph_stage_calls", {})).items()
    }
    if not str(arch_config.get("weight_qtype", "") or "").strip():
        arch_config["weight_qtype"] = str(
            runtime.get("quantization", "") or runtime.get("qtype", "") or ""
        )
    if not str(arch_config.get("lm_head_qtype", "") or "").strip():
        arch_config["lm_head_qtype"] = str(runtime.get("lm_head_qtype", "") or "")
    per_token_latencies_ms = [
        float(value)
        for value in list(runtime.get("per_token_latencies_ms") or [])
        if isinstance(value, (int, float))
    ]
    runtime_p50 = float(runtime.get("per_token_latency_p50_ms", 0.0) or 0.0)
    runtime_p95 = float(runtime.get("per_token_latency_p95_ms", 0.0) or 0.0)
    if per_token_latencies_ms:
        per_token_latency_p10_ms = _percentile(per_token_latencies_ms, 10.0)
        per_token_latency_p25_ms = _percentile(per_token_latencies_ms, 25.0)
        per_token_latency_p50_ms = _percentile(per_token_latencies_ms, 50.0)
        per_token_latency_p75_ms = _percentile(per_token_latencies_ms, 75.0)
        per_token_latency_p95_ms = _percentile(per_token_latencies_ms, 95.0)
        per_token_latency_p99_ms = _percentile(per_token_latencies_ms, 99.0)
        per_token_latency_min_ms = float(min(per_token_latencies_ms))
        per_token_latency_max_ms = float(max(per_token_latencies_ms))
        per_token_latency_stddev_ms = (
            float(statistics.pstdev(per_token_latencies_ms))
            if len(per_token_latencies_ms) >= 2
            else 0.0
        )
    else:
        per_token_latency_p50_ms = runtime_p50
        per_token_latency_p95_ms = runtime_p95
        per_token_latency_p10_ms = float(
            runtime.get("per_token_latency_p10_ms", per_token_latency_p50_ms)
            or per_token_latency_p50_ms
        )
        per_token_latency_p25_ms = float(
            runtime.get("per_token_latency_p25_ms", per_token_latency_p50_ms)
            or per_token_latency_p50_ms
        )
        per_token_latency_p75_ms = float(
            runtime.get("per_token_latency_p75_ms", per_token_latency_p95_ms)
            or per_token_latency_p95_ms
        )
        per_token_latency_p99_ms = float(
            runtime.get("per_token_latency_p99_ms", per_token_latency_p95_ms)
            or per_token_latency_p95_ms
        )
        per_token_latency_min_ms = float(
            runtime.get(
                "per_token_latency_min_ms",
                min(per_token_latency_p50_ms, per_token_latency_p95_ms),
            )
            or 0.0
        )
        per_token_latency_max_ms = float(
            runtime.get(
                "per_token_latency_max_ms",
                max(per_token_latency_p50_ms, per_token_latency_p95_ms),
            )
            or 0.0
        )
        per_token_latency_stddev_ms = float(
            runtime.get("per_token_latency_stddev_ms", 0.0) or 0.0
        )
    kernel_efficiency = _compute_kernel_efficiency(
        graph_stage_ms=graph_stage_ms,
        graph_stage_calls=graph_stage_calls,
        arch=arch_config,
    )
    graph_forward_token_id_calls = int(graph_stage_calls.get("forward_token_id", 0))
    graph_forward_token_ids_calls = int(graph_stage_calls.get("forward_token_ids", 0))
    batched_prefill_token_id_calls = int(
        runtime.get("batched_prefill_token_id_calls", graph_forward_token_ids_calls)
    )
    batched_prefill_token_id_tokens = int(
        runtime.get(
            "batched_prefill_token_id_tokens",
            graph_stage_calls.get("forward_token_ids_token_count", 0),
        )
    )
    packed_lm_head_calls = int(
        runtime.get("packed_lm_head_calls", graph_stage_calls.get("packed_lm_head", 0))
    )
    sanctioned_backend_path = str(
        runtime.get(
            "sanctioned_backend_path",
            dict(runtime.get("hot_path_proof", {})).get("sanctioned_backend_path", ""),
        )
    )
    tokenizer_backend = str(
        runtime.get(
            "tokenizer_backend",
            dict(runtime.get("hot_path_proof", {})).get("tokenizer_backend", ""),
        )
    )
    python_hot_path_calls = int(runtime.get("python_hot_path_calls", 0))
    numpy_hot_path_calls = int(runtime.get("numpy_hot_path_calls", 0))
    python_qsg_forward_calls = int(runtime.get("python_qsg_forward_calls", 0))
    python_attention_fallback_calls = int(
        runtime.get("python_attention_fallback_calls", 0)
    )
    python_ssm_fallback_calls = int(runtime.get("python_ssm_fallback_calls", 0))
    python_moe_fallback_calls = int(runtime.get("python_moe_fallback_calls", 0))
    llama_cpp_hot_path_calls = int(runtime.get("llama_cpp_hot_path_calls", 0))
    hot_path_numpy_detected = bool(
        runtime.get("hot_path_numpy_detected", False)
    ) or any(
        (
            python_hot_path_calls,
            numpy_hot_path_calls,
            python_qsg_forward_calls,
            python_attention_fallback_calls,
            python_ssm_fallback_calls,
            python_moe_fallback_calls,
            llama_cpp_hot_path_calls,
        )
    )
    full_qsg_enabled = bool(runtime.get("full_qsg_enabled", False)) or (
        bool(runtime.get("full_graph_enabled", False))
        and bool(runtime.get("qsg_processors_native_enabled", False))
        and bool(runtime.get("batched_prefill_native_enabled", False))
        and tokenizer_backend == "native"
        and sanctioned_backend_path == SANCTIONED_BACKEND_PATH
    )
    context_stabilizer_enabled = bool(runtime.get("context_stabilizer_enabled", False))
    context_stabilizer_mode = str(runtime.get("context_stabilizer_mode", ""))
    drift_latest = float(runtime.get("drift_latest", 0.0))
    drift_mean = float(runtime.get("drift_mean", 0.0))
    drift_max = float(runtime.get("drift_max", 0.0))
    drift_decay_ratio = float(runtime.get("drift_decay_ratio", 1.0))
    drift_damped_blocks = int(runtime.get("drift_damped_blocks", 0))
    drift_pruned_blocks = int(runtime.get("drift_pruned_blocks", 0))
    drift_active_tokens = int(runtime.get("drift_active_tokens", 0))
    stabilizer_seconds = float(runtime.get("stabilizer_seconds", 0.0))
    stabilizer_calls = int(runtime.get("stabilizer_calls", 0))
    drift_auto_downgrade_events = int(runtime.get("drift_auto_downgrade_events", 0))
    drift_overhead_percent = float(runtime.get("drift_overhead_percent", 0.0))
    coconut_enabled = bool(runtime.get("coconut_enabled", False))
    strict_path_stable = bool(runtime.get("strict_path_stable", False))
    printable_ratio = _printable_ratio(text)
    repeated_4gram_ratio = _repeated_ngram_ratio(text, n=4)
    repeated_8gram_ratio = _repeated_ngram_ratio(text, n=8)
    word_count = _coherence_word_count(text)
    utf8_valid = _utf8_valid(text)
    raw_printable_ratio = _printable_ratio(raw_text)
    raw_repeated_4gram_ratio = _repeated_ngram_ratio(raw_text, n=4)
    raw_repeated_8gram_ratio = _repeated_ngram_ratio(raw_text, n=8)
    raw_word_count = _coherence_word_count(raw_text)
    raw_utf8_valid = _utf8_valid(raw_text)
    leaked_control_text = (
        "<|im_start|>" in text
        or "<|im_end|>" in text
        or "<|start_of_role|>" in text
        or "<|end_of_role|>" in text
    )
    raw_leaked_control_text = (
        "<|im_start|>" in raw_text
        or "<|im_end|>" in raw_text
        or "<|start_of_role|>" in raw_text
        or "<|end_of_role|>" in raw_text
    )
    leaked_think_tags = "<think>" in text.lower() or "</think>" in text.lower()
    raw_leaked_think_tags = (
        "<think>" in raw_text.lower() or "</think>" in raw_text.lower()
    )
    stop_reason = str(runtime.get("stop_reason", ""))
    coherence_valid, coherence_issues = _coherence_issues(
        text=text,
        printable_ratio=printable_ratio,
        repeated_4gram_ratio=repeated_4gram_ratio,
        repeated_8gram_ratio=repeated_8gram_ratio,
        word_count=word_count,
        utf8_valid=utf8_valid,
        leaked_control_text=leaked_control_text,
        leaked_think_tags=leaked_think_tags,
        stop_reason=stop_reason,
    )
    raw_coherence_valid, raw_coherence_issues = _coherence_issues(
        text=raw_text,
        printable_ratio=raw_printable_ratio,
        repeated_4gram_ratio=raw_repeated_4gram_ratio,
        repeated_8gram_ratio=raw_repeated_8gram_ratio,
        word_count=raw_word_count,
        utf8_valid=raw_utf8_valid,
        leaked_control_text=raw_leaked_control_text,
        leaked_think_tags=raw_leaked_think_tags,
        stop_reason=stop_reason,
    )
    generation_mode = str(runtime.get("generation_mode", ""))
    benchmark_label = str(runtime.get("benchmark_label", "")).strip()
    if not benchmark_label:
        benchmark_label = benchmark_label_for_mode(generation_mode).value
    prompt_category = str(
        runtime.get("prompt_category", "")
    ).strip() or _prompt_category(prompt)
    temperature_band = str(
        runtime.get("temperature_band", "")
    ).strip() or _temperature_band(float(params["temperature"]))

    result = BenchmarkResult(
        model=model,
        threads=active_threads,
        max_new_tokens=max_new_tokens,
        run_index=run_index,
        digest=str(runtime.get("digest", "")),
        quantization=str(runtime.get("quantization", "")),
        context_length=int(runtime.get("context_length", 0)),
        prompt_tokens=len(prompt_tokens),
        runtime_total_seconds=runtime_total_seconds,
        runtime_prefill_seconds=runtime_prefill_seconds,
        runtime_decode_seconds=runtime_decode_seconds,
        load_seconds=load_seconds,
        generate_seconds=generate_seconds,
        first_token_latency_seconds=ttft_seconds,
        prompt_format_ms=float(runtime.get("prompt_format_ms", 0.0)),
        tokenize_ms=float(runtime.get("tokenize_ms", 0.0)),
        embedding_lookup_ms=float(runtime.get("embedding_lookup_ms", 0.0)),
        graph_prefill_ms=float(runtime.get("graph_prefill_ms", 0.0)),
        graph_decode_ms=float(runtime.get("graph_decode_ms", 0.0)),
        sample_ms=float(runtime.get("sample_ms", 0.0)),
        logits_processor_ms=float(runtime.get("logits_processor_ms", 0.0)),
        penalty_ms=float(runtime.get("penalty_ms", 0.0)),
        suppression_ms=float(runtime.get("suppression_ms", 0.0)),
        graph_prefill_calls=int(runtime.get("graph_prefill_calls", 0)),
        graph_decode_calls=int(runtime.get("graph_decode_calls", 0)),
        sample_calls=int(runtime.get("sample_calls", 0)),
        logits_processor_calls=int(runtime.get("logits_processor_calls", 0)),
        penalty_calls=int(runtime.get("penalty_calls", 0)),
        suppression_calls=int(runtime.get("suppression_calls", 0)),
        graph_prefill_avg_ms=float(runtime.get("graph_prefill_avg_ms", 0.0)),
        graph_decode_avg_ms=float(runtime.get("graph_decode_avg_ms", 0.0)),
        sample_avg_ms=float(runtime.get("sample_avg_ms", 0.0)),
        logits_processor_avg_ms=float(runtime.get("logits_processor_avg_ms", 0.0)),
        penalty_avg_ms=float(runtime.get("penalty_avg_ms", 0.0)),
        suppression_avg_ms=float(runtime.get("suppression_avg_ms", 0.0)),
        ttft_ms=ttft_seconds * 1000.0,
        prefill_throughput_tps=prefill_tps,
        effective_prefill_tokens=effective_prefill_tokens,
        effective_prefill_throughput_tps=effective_prefill_tps,
        decode_throughput_tps=decode_tps,
        end_to_end_throughput_tps=end_to_end_tps,
        per_token_latency_p50_ms=per_token_latency_p50_ms,
        per_token_latency_p95_ms=per_token_latency_p95_ms,
        per_token_latencies_ms=per_token_latencies_ms,
        per_token_latency_p10_ms=per_token_latency_p10_ms,
        per_token_latency_p25_ms=per_token_latency_p25_ms,
        per_token_latency_p75_ms=per_token_latency_p75_ms,
        per_token_latency_p99_ms=per_token_latency_p99_ms,
        per_token_latency_stddev_ms=per_token_latency_stddev_ms,
        per_token_latency_min_ms=per_token_latency_min_ms,
        per_token_latency_max_ms=per_token_latency_max_ms,
        runtime_vs_wall_delta_ms=runtime_vs_wall_delta_ms,
        prefill_phase_gap_ms=float(runtime.get("prefill_phase_gap_ms", 0.0)),
        decode_phase_gap_ms=float(runtime.get("decode_phase_gap_ms", 0.0)),
        new_tokens=new_tokens,
        tokens_per_second=(
            (new_tokens / generate_seconds) if generate_seconds > 0 else 0.0
        ),
        decode_threads=int(runtime.get("decode_threads", active_threads)),
        batch_threads=int(runtime.get("batch_threads", active_threads)),
        ubatch=int(runtime.get("ubatch", 0)),
        requested_decode_threads=int(decode_threads or 0),
        requested_batch_threads=int(batch_threads or 0),
        requested_ubatch=int(ubatch or 0),
        openmp_enabled=bool(runtime.get("openmp_enabled", False)),
        avx2_enabled=bool(runtime.get("avx2_enabled", False)),
        avx512_enabled=bool(runtime.get("avx512_enabled", False)),
        mmap_enabled=bool(runtime.get("mmap_enabled", False)),
        mapped_model_bytes=int(runtime.get("mapped_model_bytes", 0)),
        loader_cache_residency_bytes=int(
            runtime.get("loader_cache_residency_bytes", 0)
        ),
        embedding_materialization_bytes=int(
            runtime.get("embedding_materialization_bytes", 0)
        ),
        kv_used_cells=int(runtime.get("kv_used_cells", 0)),
        kv_fragmentation_ratio=float(runtime.get("kv_fragmentation_ratio", 0.0)),
        kv_defrag_count=int(runtime.get("kv_defrag_count", 0)),
        kv_cache_quantization=str(runtime.get("kv_cache_quantization", "")),
        template_name=str(
            runtime.get("template_name", runtime.get("chat_template", ""))
        ),
        granite_moe_mode=str(runtime.get("granite_moe_mode", "")),
        active_thread_mode=str(runtime.get("active_thread_mode", "")),
        prefill_chunk_count=int(runtime.get("prefill_chunk_count", 0)),
        lm_head_layout=str(runtime.get("lm_head_layout", "")),
        lm_head_qtype=int(runtime.get("lm_head_qtype", 0)),
        graph_forward_token_id_calls=graph_forward_token_id_calls,
        graph_forward_token_ids_calls=graph_forward_token_ids_calls,
        batched_prefill_token_id_calls=batched_prefill_token_id_calls,
        batched_prefill_token_id_path=batched_prefill_token_id_calls > 0,
        batched_prefill_token_id_tokens=batched_prefill_token_id_tokens,
        batch_token_fallback_count=int(runtime.get("batch_token_fallback_count", 0)),
        packed_lm_head_calls=packed_lm_head_calls,
        runtime_thread_switches=int(runtime.get("runtime_thread_switches", 0)),
        parallel_decode=bool(runtime.get("parallel_decode", False)),
        speculative_decode=bool(runtime.get("speculative_decode", False)),
        generation_mode=generation_mode,
        benchmark_label=benchmark_label,
        prompt_category=prompt_category,
        temperature_band=temperature_band,
        accepted_parallel_tokens=int(runtime.get("accepted_parallel_tokens", 0)),
        rejected_parallel_tokens=int(runtime.get("rejected_parallel_tokens", 0)),
        proposed_parallel_tokens=int(
            runtime.get(
                "proposed_parallel_tokens",
                int(runtime.get("accepted_parallel_tokens", 0))
                + int(runtime.get("rejected_parallel_tokens", 0)),
            )
        ),
        draft_frontier_width=int(runtime.get("draft_frontier_width", 0)),
        verify_depth=int(runtime.get("verify_depth", 0)),
        parallel_step_latency_ms=float(runtime.get("parallel_step_latency_ms", 0.0)),
        draft_confidence_mean=float(runtime.get("draft_confidence_mean", 0.0)),
        draft_confidence_min=float(runtime.get("draft_confidence_min", 0.0)),
        draft_source=str(runtime.get("draft_source", "")),
        blockwise_blocks=int(runtime.get("blockwise_blocks", 0)),
        blockwise_denoise_steps=int(runtime.get("blockwise_denoise_steps", 0)),
        blockwise_convergence_rate=float(
            runtime.get("blockwise_convergence_rate", 0.0)
        ),
        masked_generation_ready=bool(runtime.get("masked_generation_ready", False)),
        masked_generation_steps=int(runtime.get("masked_generation_steps", 0)),
        masked_generation_proposed_tokens=int(
            runtime.get("masked_generation_proposed_tokens", 0)
        ),
        masked_generation_accepted_tokens=int(
            runtime.get("masked_generation_accepted_tokens", 0)
        ),
        masked_generation_density=float(runtime.get("masked_generation_density", 0.0)),
        prefix_cache_hit_rate=float(
            runtime.get(
                "prefix_cache_hit_rate",
                runtime.get("prompt_cache_hit_ratio", 0.0),
            )
        ),
        scheduler_queue_wait_ms=float(runtime.get("scheduler_queue_wait_ms", 0.0)),
        scheduler_iteration_ms=float(runtime.get("scheduler_iteration_ms", 0.0)),
        quality_guard_triggered=bool(runtime.get("quality_guard_triggered", False)),
        speculative_accept_count=int(runtime.get("speculative_accept_count", 0)),
        speculative_reject_count=int(runtime.get("speculative_reject_count", 0)),
        self_spec_native_path=bool(runtime.get("self_spec_native_path", False)),
        self_spec_policy=str(runtime.get("self_spec_policy", "")),
        self_spec_exit_layer=int(runtime.get("self_spec_exit_layer", 0)),
        self_spec_exit_fraction=float(runtime.get("self_spec_exit_fraction", 0.0)),
        self_spec_draft_tokens=int(runtime.get("self_spec_draft_tokens", 0)),
        min_new_tokens_before_eos=int(runtime.get("min_new_tokens_before_eos", 0)),
        coherence_guard_events=int(runtime.get("coherence_guard_events", 0)),
        prompt_cache_hit=bool(runtime.get("prompt_cache_hit", False)),
        prompt_cache_hits=prompt_cache_hits,
        prompt_cache_misses=prompt_cache_misses,
        prompt_cache_lookups=prompt_cache_lookups,
        prompt_cache_hit_ratio=(
            float(prompt_cache_hits) / float(prompt_cache_lookups)
            if prompt_cache_lookups > 0
            else 0.0
        ),
        prompt_cache_reused_tokens=prompt_cache_reused_tokens,
        prompt_cache_reuse_ratio=(
            float(prompt_cache_reused_tokens) / float(len(prompt_tokens))
            if prompt_tokens
            else 0.0
        ),
        sampling_profile=profile_name,
        temperature=float(params["temperature"]),
        top_p=float(params["top_p"]),
        top_k=int(params["top_k"]),
        min_p=float(params["min_p"]),
        presence_penalty=float(params["presence_penalty"]),
        repetition_penalty=float(params["repetition_penalty"]),
        rss_before_mb=rss_before,
        rss_after_load_mb=rss_after_load,
        rss_after_generate_mb=rss_after_generate,
        sample_text=text,
        printable_ratio=printable_ratio,
        repeated_4gram_ratio=repeated_4gram_ratio,
        repeated_8gram_ratio=repeated_8gram_ratio,
        ascii_ratio=_ascii_ratio(text),
        word_count=word_count,
        utf8_valid=utf8_valid,
        native_fast_path=bool(runtime.get("native_fast_path", False)),
        parallel_decode_disable_reason=str(
            runtime.get("parallel_decode_disable_reason", "")
        ),
        leaked_control_text=leaked_control_text,
        leaked_think_tags=leaked_think_tags,
        trivial_completion_detected="coherence_trivial_completion=true"
        in coherence_issues,
        coherence_valid=coherence_valid,
        coherence_issues=coherence_issues,
        coherence_issue_count=len(coherence_issues),
        stop_reason=stop_reason,
        schema_valid=bool(runtime.get("schema_valid", False)),
        tool_call_parse_success=bool(runtime.get("tool_call_parse_success", False)),
        measurement_valid=measurement_valid,
        measurement_issues=measurement_issues,
        measurement_issue_count=len(measurement_issues),
        measurement_source=measurement_source,
        native_build_id=str(runtime.get("native_build_id", "")),
        native_build_sha256=str(runtime.get("native_build_sha256", "")),
        loaded_native_library=str(runtime.get("loaded_native_library", "")),
        sanctioned_backend_path=sanctioned_backend_path,
        tokenizer_backend=tokenizer_backend,
        backend_module=str(runtime.get("backend_module", "")),
        backend_module_library=str(runtime.get("backend_module_library", "")),
        backend_module_loaded=bool(runtime.get("backend_module_loaded", False)),
        backend_module_marker_symbol=str(
            runtime.get("backend_module_marker_symbol", "")
        ),
        backend_module_marker=int(runtime.get("backend_module_marker", 0)),
        perf_stat_artifact=str(runtime.get("perf_stat_artifact", "")),
        perf_c2c_artifact=str(runtime.get("perf_c2c_artifact", "")),
        numa_maps_artifact=numa_maps_artifact,
        sampling_backend=str(runtime.get("sampling_backend", "")),
        penalties_backend=str(runtime.get("penalties_backend", "")),
        suppression_backend=str(runtime.get("suppression_backend", "")),
        logits_backend=str(runtime.get("logits_backend", "")),
        full_qsg_enabled=full_qsg_enabled,
        full_graph_enabled=bool(runtime.get("full_graph_enabled", False)),
        qsg_processors_native_enabled=bool(
            runtime.get("qsg_processors_native_enabled", False)
        ),
        batched_prefill_native_enabled=bool(
            runtime.get("batched_prefill_native_enabled", False)
        ),
        context_stabilizer_enabled=context_stabilizer_enabled,
        context_stabilizer_mode=context_stabilizer_mode,
        drift_latest=drift_latest,
        drift_mean=drift_mean,
        drift_max=drift_max,
        drift_decay_ratio=drift_decay_ratio,
        drift_damped_blocks=drift_damped_blocks,
        drift_pruned_blocks=drift_pruned_blocks,
        drift_active_tokens=drift_active_tokens,
        stabilizer_seconds=stabilizer_seconds,
        stabilizer_calls=stabilizer_calls,
        drift_auto_downgrade_events=drift_auto_downgrade_events,
        drift_overhead_percent=drift_overhead_percent,
        coconut_enabled=coconut_enabled,
        coconut_paths=int(runtime.get("coconut_paths", 0)),
        coconut_alpha=float(runtime.get("coconut_alpha", 0.0)),
        coconut_seconds=float(runtime.get("coconut_seconds", 0.0)),
        coconut_candidate_count=int(runtime.get("coconut_candidate_count", 0)),
        coconut_entropy_mean=float(runtime.get("coconut_entropy_mean", 0.0)),
        coconut_amplitude_mean=float(runtime.get("coconut_amplitude_mean", 0.0)),
        coconut_consistency_rejects=int(runtime.get("coconut_consistency_rejects", 0)),
        grover_enabled=bool(runtime.get("grover_enabled", False)),
        grover_top_k=int(runtime.get("grover_top_k", 0)),
        grover_damping=float(runtime.get("grover_damping", 0.0)),
        grover_calls=int(runtime.get("grover_calls", 0)),
        grover_seconds=float(runtime.get("grover_seconds", 0.0)),
        grover_candidate_count=int(runtime.get("grover_candidate_count", 0)),
        grover_rescore_delta_mean=float(runtime.get("grover_rescore_delta_mean", 0.0)),
        grover_timeout_events=int(runtime.get("grover_timeout_events", 0)),
        strict_path_stable=strict_path_stable,
        hot_path_numpy_detected=hot_path_numpy_detected,
        python_hot_path_calls=python_hot_path_calls,
        numpy_hot_path_calls=numpy_hot_path_calls,
        python_qsg_forward_calls=python_qsg_forward_calls,
        python_attention_fallback_calls=python_attention_fallback_calls,
        python_ssm_fallback_calls=python_ssm_fallback_calls,
        python_moe_fallback_calls=python_moe_fallback_calls,
        llama_cpp_hot_path_calls=llama_cpp_hot_path_calls,
        physical_core_count=int(runtime.get("physical_core_count", 0)),
        logical_core_count=int(runtime.get("logical_core_count", 0)),
        p_core_count=int(runtime.get("p_core_count", 0)),
        affinity_policy=str(runtime.get("affinity_policy", "")),
        affinity_mode=int(runtime.get("affinity_mode", 1)),
        l3_domain_count=int(runtime.get("l3_domain_count", 0)),
        numa_strict=bool(runtime.get("numa_strict", False)),
        numa_affinity_mode=str(runtime.get("numa_affinity_mode", "")),
        numa_hugepage=str(runtime.get("numa_hugepage", "")),
        numa_bind_policy=str(runtime.get("numa_bind_policy", "")),
        numa_first_touch=bool(runtime.get("numa_first_touch", False)),
        topology_json=str(runtime.get("topology_json", "")),
        os_thread_migrations=int(runtime.get("os_thread_migrations", 0)),
        os_last_cpu=int(runtime.get("os_last_cpu", -1)),
        omp_places=str(runtime.get("omp_places", "")),
        omp_proc_bind=str(runtime.get("omp_proc_bind", "")),
        omp_max_threads=int(runtime.get("omp_max_threads", 0)),
        omp_dynamic=bool(runtime.get("omp_dynamic", False)),
        omp_active_levels=int(runtime.get("omp_active_levels", 0)),
        native_isa_baseline=str(runtime.get("native_isa_baseline", "")),
        native_backend_abi_match=bool(runtime.get("native_backend_abi_match", False)),
        perf_event_access=bool(runtime.get("perf_event_access", False)),
        perf_event_access_reason=str(runtime.get("perf_event_access_reason", "")),
        cpu_governor=str(runtime.get("cpu_governor", "")),
        thp_mode=str(runtime.get("thp_mode", "")),
        perf_counter_source=str(runtime.get("perf_counter_source", "")),
        worker_cpu_mask=str(runtime.get("worker_cpu_mask", "")),
        orchestrator_cpu_mask=str(runtime.get("orchestrator_cpu_mask", "")),
        l3_domain_ids_active=[
            int(value) for value in (runtime.get("l3_domain_ids_active") or [])
        ],
        autotune_profile_id=str(runtime.get("autotune_profile_id", "")),
        autotune_source=str(runtime.get("autotune_source", "")),
        autotune_score=float(runtime.get("autotune_score", 0.0)),
        autotune_exploration_count=int(runtime.get("autotune_exploration_count", 0)),
        hot_path_proof=dict(runtime.get("hot_path_proof", {})),
        performance_envelope=dict(
            runtime.get("performance_envelope")
            or PerformanceEnvelope.from_runtime_status(
                runtime,
                capability_digest=str(runtime.get("capability_digest", "")),
                delta_watermark=dict(runtime.get("delta_watermark") or {}),
            ).as_dict()
        ),
        performance_twin=dict(runtime.get("performance_twin") or {}),
        repo_coupled_runtime=dict(runtime.get("repo_coupled_runtime") or {}),
        graph_stage_ms=graph_stage_ms,
        graph_stage_avg_ms=graph_stage_avg_ms,
        graph_stage_calls=graph_stage_calls,
        arch_hidden_dim=int(arch_config.get("hidden_dim", 0) or 0),
        arch_num_layers=int(arch_config.get("num_layers", 0) or 0),
        arch_num_attention_layers=int(arch_config.get("num_attention_layers", 0) or 0),
        arch_num_ssm_layers=int(arch_config.get("num_ssm_layers", 0) or 0),
        arch_num_moe_layers=int(arch_config.get("num_moe_layers", 0) or 0),
        arch_num_heads=int(arch_config.get("num_heads", 0) or 0),
        arch_num_kv_heads=int(arch_config.get("num_kv_heads", 0) or 0),
        arch_head_dim=int(arch_config.get("head_dim", 0) or 0),
        arch_intermediate_dim=int(arch_config.get("intermediate_dim", 0) or 0),
        arch_vocab_size=int(arch_config.get("vocab_size", 0) or 0),
        arch_ssm_state_dim=int(arch_config.get("ssm_state_dim", 0) or 0),
        arch_ssm_conv_kernel=int(arch_config.get("ssm_conv_kernel", 0) or 0),
        arch_num_experts=int(arch_config.get("num_experts", 0) or 0),
        arch_top_k_experts=int(arch_config.get("top_k_experts", 0) or 0),
        arch_rope_dim=int(arch_config.get("rope_dim", 0) or 0),
        arch_weight_qtype=str(arch_config.get("weight_qtype", "") or ""),
        arch_lm_head_qtype=str(arch_config.get("lm_head_qtype", "") or ""),
        kernel_efficiency=kernel_efficiency,
        raw_sample_text=raw_text,
        raw_printable_ratio=raw_printable_ratio,
        raw_repeated_4gram_ratio=raw_repeated_4gram_ratio,
        raw_repeated_8gram_ratio=raw_repeated_8gram_ratio,
        raw_word_count=raw_word_count,
        raw_utf8_valid=raw_utf8_valid,
        raw_leaked_control_text=raw_leaked_control_text,
        raw_leaked_think_tags=raw_leaked_think_tags,
        raw_trivial_completion_detected=_trivial_completion_detected(
            raw_text, stop_reason
        ),
        raw_coherence_valid=raw_coherence_valid,
        raw_coherence_issues=list(raw_coherence_issues),
        raw_coherence_issue_count=len(raw_coherence_issues),
    )
    _trace(
        f"completed run_index={run_index} model={model} "
        f"decode_tps={result.decode_throughput_tps:.3f} ttft_ms={result.ttft_ms:.3f}"
    )
    return result


def _parse_thread_values(raw: object | None, fallback: list[int]) -> list[int]:
    if raw is None:
        return list(fallback)
    text = str(raw).strip()
    if not text:
        return list(fallback)
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except Exception:
            continue
        if value > 0:
            values.append(value)
    if not values:
        return list(fallback)
    return list(dict.fromkeys(values))


def _model_thread_sweep(
    model: str,
    args: argparse.Namespace,
) -> list[tuple[int | None, int | None, int | None]]:
    lower = str(model).lower()
    host = _host_facts()
    logical = max(
        1, int(host.get("affinity_visible_threads", os.cpu_count() or 1) or 1)
    )
    physical = int(host.get("physical_cores", 0) or 0)
    if physical <= 0:
        physical = (
            max(1, logical // 2) if logical >= 8 and logical % 2 == 0 else logical
        )
    per_l3 = max(1, physical // 2) if physical >= 4 else physical
    default_decode = list(
        dict.fromkeys(
            value
            for value in (
                per_l3,
                max(1, physical - 2),
                physical,
                logical,
            )
            if value > 0
        )
    )
    default_batch = list(
        dict.fromkeys(value for value in (per_l3, physical, logical) if value > 0)
    )
    if "qwen3.5" in lower or "qwen35" in lower:
        default_ubatch = [4, 8, 16, 32]
    elif "granite4" in lower:
        default_ubatch = [8, 16, 32]
    else:
        default_ubatch = [4, 8, 16, 32]

    autotune_mode = str(getattr(args, "autotune", "off") or "off").lower()
    cached = _load_autotune_profile(model) if autotune_mode == "locked" else {}
    if cached:
        decode_value = int(cached.get("decode_threads", 0) or 0)
        batch_value = int(cached.get("batch_threads", 0) or 0)
        ubatch_value = int(cached.get("ubatch", 0) or 0)
        if decode_value > 0 and batch_value > 0 and ubatch_value > 0:
            return [(decode_value, batch_value, ubatch_value)]

    if not bool(getattr(args, "thread_sweep", False)):
        return [
            (
                getattr(args, "decode_threads", None),
                getattr(args, "batch_threads", None),
                getattr(args, "ubatch", None),
            )
        ]

    decode_candidates = _parse_thread_values(
        getattr(args, "decode_thread_sweep", None),
        default_decode,
    )
    batch_candidates = _parse_thread_values(
        getattr(args, "batch_thread_sweep", None),
        default_batch,
    )
    ubatch_candidates = _parse_thread_values(
        getattr(args, "ubatch_sweep", None),
        default_ubatch,
    )
    decode_explicit = getattr(args, "decode_threads", None)
    batch_explicit = getattr(args, "batch_threads", None)
    ubatch_explicit = getattr(args, "ubatch", None)
    if decode_explicit is not None:
        decode_candidates = [int(decode_explicit)]
    if batch_explicit is not None:
        batch_candidates = [int(batch_explicit)]
    if ubatch_explicit is not None:
        ubatch_candidates = [int(ubatch_explicit)]

    combos: list[tuple[int | None, int | None, int | None]] = []
    for decode_threads in decode_candidates:
        for batch_threads in batch_candidates:
            for ubatch in ubatch_candidates:
                combos.append((decode_threads, batch_threads, ubatch))
    return combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model name/path. Repeat for multiple models.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "In 3-4 sentences, explain how AVX2 and OpenMP improve CPU LLM inference "
            "throughput and latency, using concrete mechanisms and avoiding repetition."
        ),
        help="Prompt text to benchmark.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--sampling-profile", type=str, default=None)
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--batch-threads", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument(
        "--thread-sweep",
        action="store_true",
        help="Benchmark multiple decode/batch/ubatch combinations.",
    )
    parser.add_argument(
        "--decode-thread-sweep",
        type=str,
        default=None,
        help="Comma-separated decode thread candidates for sweep mode.",
    )
    parser.add_argument(
        "--batch-thread-sweep",
        type=str,
        default=None,
        help="Comma-separated batch thread candidates for sweep mode.",
    )
    parser.add_argument(
        "--ubatch-sweep",
        type=str,
        default=None,
        help="Comma-separated ubatch candidates for sweep mode.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of table output.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    parser.add_argument(
        "--fast-native",
        action="store_true",
        help="Enable native fast-path controls (disable Python logits post-processing).",
    )
    parser.add_argument(
        "--disable-logits-processors",
        action="store_true",
        default=None,
        help="Disable Python logits processor on benchmark runs.",
    )
    parser.add_argument(
        "--disable-token-penalties",
        action="store_true",
        default=None,
        help="Disable Python repetition/presence penalties on benchmark runs.",
    )
    parser.add_argument(
        "--force-parallel-decode",
        action="store_true",
        help="Force native parallel decode when safe.",
    )
    parser.add_argument(
        "--min-new-tokens-before-eos",
        type=int,
        default=None,
        help="Prevent EOS before this many generated tokens.",
    )
    parser.add_argument(
        "--coherence-first",
        action="store_true",
        help="Prefer deterministic coherent sampling profiles for benchmark defaults.",
    )
    parser.add_argument(
        "--collect-perf-stat",
        action="store_true",
        help="Wrap isolated benchmark execution with `perf stat` and persist artifact text.",
    )
    parser.add_argument(
        "--collect-perf-c2c",
        action="store_true",
        help="Wrap isolated benchmark execution with `perf c2c record/report` artifacts.",
    )
    parser.add_argument(
        "--collect-numa-page-map",
        action="store_true",
        help="Persist NUMA map snapshots (`/proc/<pid>/numa_maps`, `numastat -p`).",
    )
    parser.add_argument(
        "--host-access",
        choices=("user", "privileged", "mixed"),
        default="user",
    )
    parser.add_argument(
        "--collect-hw-counters",
        choices=("off", "auto", "required"),
        default="auto",
    )
    parser.add_argument("--require-grover", action="store_true")
    parser.add_argument("--require-coconut", action="store_true")
    parser.add_argument(
        "--autotune",
        choices=("off", "explore", "locked"),
        default="off",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory for perf/NUMA artifacts when collection flags are enabled.",
    )
    parser.add_argument("--min-decode-tps", type=float, default=None)
    parser.add_argument("--max-ttft-ms", type=float, default=None)
    parser.add_argument("--min-printable-ratio", type=float, default=0.95)
    parser.add_argument("--max-repeated-4gram-ratio", type=float, default=0.12)
    parser.add_argument("--max-repeated-8gram-ratio", type=float, default=0.05)
    parser.add_argument("--min-ascii-ratio", type=float, default=None)
    parser.add_argument("--min-word-count", type=int, default=None)
    parser.add_argument("--max-rss-growth-mb", type=float, default=None)
    parser.add_argument("--require-utf8", action="store_true")
    parser.add_argument("--require-measurement-valid", action="store_true")
    parser.add_argument("--require-openmp", action="store_true")
    parser.add_argument("--require-avx2", action="store_true")
    parser.add_argument("--require-mmap", action="store_true")
    parser.add_argument(
        "--isolated-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--self-artifact-dir", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--capture-self-numa-map",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _append_optional_arg(cmd: list[str], name: str, value) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def _artifact_run_dir(
    args: argparse.Namespace, run_index: int, model: str
) -> pathlib.Path | None:
    collect_any = bool(
        getattr(args, "collect_perf_stat", False)
        or getattr(args, "collect_perf_c2c", False)
        or getattr(args, "collect_numa_page_map", False)
    )
    if not collect_any:
        return None
    root = pathlib.Path(
        str(
            getattr(args, "artifacts_dir", "")
            or (REPO_ROOT / ".anvil" / "benchmarks" / "native_qsg_artifacts")
        )
    )
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(model))
    run_dir = root / f"run_{run_index:04d}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _capture_self_numa_artifacts(artifact_dir: pathlib.Path | None) -> str:
    if artifact_dir is None:
        return ""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    numa_maps_artifact = ""
    numa_maps_path = artifact_dir / "numa_maps_snapshot.txt"
    numastat_path = artifact_dir / "numastat_snapshot.txt"
    try:
        numa_maps_path.write_text(
            pathlib.Path(f"/proc/{os.getpid()}/numa_maps").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        numa_maps_artifact = str(numa_maps_path)
    except Exception:
        pass
    try:
        numastat_completed = subprocess.run(
            ["numastat", "-p", str(os.getpid())],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
        numastat_path.write_text(
            (numastat_completed.stdout or "") + (numastat_completed.stderr or ""),
            encoding="utf-8",
        )
        if not numa_maps_artifact:
            numa_maps_artifact = str(numastat_path)
    except Exception:
        pass
    return numa_maps_artifact


def _run_once_isolated(
    *,
    model: str,
    args: argparse.Namespace,
    override_params: dict,
    run_index: int,
    decode_threads: int | None,
    batch_threads: int | None,
    ubatch: int | None,
) -> BenchmarkResult:
    _trace(f"starting isolated wrapper run_index={run_index} model={model}")
    artifact_dir = _artifact_run_dir(args, run_index=run_index, model=model)
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--isolated-child",
        "--model",
        model,
        "--prompt",
        str(args.prompt),
        "--max-new-tokens",
        str(int(args.max_new_tokens)),
        "--runs",
        "1",
        "--context-length",
        str(int(getattr(args, "context_length", 2048))),
        "--json",
    ]
    _append_optional_arg(
        cmd, "--sampling-profile", getattr(args, "sampling_profile", None)
    )
    _append_optional_arg(cmd, "--decode-threads", decode_threads)
    _append_optional_arg(cmd, "--batch-threads", batch_threads)
    _append_optional_arg(cmd, "--ubatch", ubatch)
    _append_optional_arg(cmd, "--temperature", override_params.get("temperature"))
    _append_optional_arg(cmd, "--top-p", override_params.get("top_p"))
    _append_optional_arg(cmd, "--top-k", override_params.get("top_k"))
    _append_optional_arg(cmd, "--min-p", override_params.get("min_p"))
    _append_optional_arg(
        cmd, "--presence-penalty", override_params.get("presence_penalty")
    )
    _append_optional_arg(
        cmd, "--repetition-penalty", override_params.get("repetition_penalty")
    )
    _append_optional_arg(
        cmd,
        "--min-new-tokens-before-eos",
        getattr(args, "min_new_tokens_before_eos", None),
    )
    if bool(getattr(args, "fast_native", False)):
        cmd.append("--fast-native")
    if bool(getattr(args, "disable_logits_processors", False)):
        cmd.append("--disable-logits-processors")
    if bool(getattr(args, "disable_token_penalties", False)):
        cmd.append("--disable-token-penalties")
    if bool(getattr(args, "force_parallel_decode", False)):
        cmd.append("--force-parallel-decode")
    if bool(getattr(args, "coherence_first", False)):
        cmd.append("--coherence-first")
    if bool(getattr(args, "require_grover", False)):
        cmd.append("--require-grover")
    if bool(getattr(args, "require_coconut", False)):
        cmd.append("--require-coconut")
    _append_optional_arg(cmd, "--host-access", getattr(args, "host_access", None))
    _append_optional_arg(
        cmd,
        "--collect-hw-counters",
        getattr(args, "collect_hw_counters", None),
    )
    _append_optional_arg(cmd, "--autotune", getattr(args, "autotune", None))
    if bool(getattr(args, "collect_numa_page_map", False)) and artifact_dir is not None:
        cmd.extend(
            ["--self-artifact-dir", str(artifact_dir), "--capture-self-numa-map"]
        )

    perf_stat_artifact = ""
    perf_c2c_artifact = ""
    wrapped_cmd = list(cmd)
    if bool(getattr(args, "collect_perf_c2c", False)) and artifact_dir is not None:
        perf_data_path = artifact_dir / "perf_c2c.data"
        perf_c2c_report = artifact_dir / "perf_c2c_report.txt"
        wrapped_cmd = [
            "perf",
            "c2c",
            "record",
            "-o",
            str(perf_data_path),
            "--",
            *cmd,
        ]
        perf_c2c_artifact = str(perf_c2c_report)
    elif bool(getattr(args, "collect_perf_stat", False)) and artifact_dir is not None:
        perf_stat_path = artifact_dir / "perf_stat.txt"
        wrapped_cmd = [
            "perf",
            "stat",
            "-d",
            "-d",
            "-d",
            "-e",
            (
                "cycles,instructions,cache-references,cache-misses,"
                "context-switches,cpu-migrations,page-faults"
            ),
            "-o",
            str(perf_stat_path),
            "--",
            *cmd,
        ]
        perf_stat_artifact = str(perf_stat_path)

    completed = subprocess.run(
        wrapped_cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    _trace(
        f"isolated child returned rc={completed.returncode} "
        f"stdout_bytes={len(completed.stdout or '')} stderr_bytes={len(completed.stderr or '')}"
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            "Isolated benchmark subprocess failed:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    if bool(getattr(args, "collect_perf_c2c", False)) and artifact_dir is not None:
        perf_data_path = artifact_dir / "perf_c2c.data"
        perf_c2c_report = artifact_dir / "perf_c2c_report.txt"
        if perf_data_path.exists():
            report_completed = subprocess.run(
                [
                    "perf",
                    "c2c",
                    "report",
                    "--stdio",
                    "-i",
                    str(perf_data_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
                env=os.environ.copy(),
            )
            perf_c2c_report.write_text(
                (report_completed.stdout or "") + (report_completed.stderr or ""),
                encoding="utf-8",
            )
            perf_c2c_artifact = str(perf_c2c_report)
    stdout = completed.stdout.strip()
    payload: dict[str, object] | list[object] | None = None
    for start_char, end_char in (("[", "]"), ("{", "}")):
        json_start = stdout.find(start_char)
        json_end = stdout.rfind(end_char)
        if json_start < 0 or json_end < json_start:
            continue
        candidate = stdout[json_start : json_end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            payload = parsed
            break

    if payload is None:
        raise RuntimeError(
            "Isolated benchmark subprocess did not emit parseable JSON:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    flat_results: list[dict[str, object]] = []
    if isinstance(payload, list):
        # Legacy adapter contract: isolated mode may emit a raw list payload.
        flat_results = [dict(item) for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        flat_results = [
            dict(item)
            for item in (payload.get("flat_results") or [])
            if isinstance(item, dict)
        ]
        if not flat_results:
            # Compatibility path for older structured reports without flat_results.
            for item in payload.get("results") or []:
                if not isinstance(item, dict):
                    continue
                merged: dict[str, object] = {}
                for section in (
                    "identity",
                    "sampling",
                    "quality",
                    "measurement",
                    "hot_path",
                ):
                    value = item.get(section)
                    if isinstance(value, dict):
                        merged.update(value)
                metrics = item.get("throughput")
                if isinstance(metrics, dict):
                    merged.update(metrics)
                merged["run_index"] = item.get("run_index")
                merged["model"] = item.get("model")
                flat_results.append(merged)
    if len(flat_results) != 1:
        raise RuntimeError(
            f"Expected exactly 1 flat result from isolated child, got {len(flat_results)}"
        )
    result = BenchmarkResult(**flat_results[0])
    result.run_index = run_index
    result.perf_stat_artifact = perf_stat_artifact
    result.perf_c2c_artifact = perf_c2c_artifact
    if perf_stat_artifact:
        for key, value in _parse_perf_stat_artifact(perf_stat_artifact).items():
            setattr(result, key, value)
    decode_tps = float(getattr(result, "decode_throughput_tps", 0.0) or 0.0)
    _trace(
        f"isolated wrapper parsed result model={model} " f"decode_tps={decode_tps:.3f}"
    )
    return result


def main() -> int:
    args = parse_args()
    _trace(
        f"main start models={','.join(args.model)} runs={args.runs} "
        f"json={bool(args.json)} isolated_child={bool(getattr(args, 'isolated_child', False))}"
    )
    legacy_harness_mode = not hasattr(args, "isolated_child")
    override_params = {
        "temperature": getattr(args, "temperature", None),
        "top_p": getattr(args, "top_p", None),
        "top_k": getattr(args, "top_k", None),
        "min_p": getattr(args, "min_p", None),
        "presence_penalty": getattr(args, "presence_penalty", None),
        "repetition_penalty": getattr(args, "repetition_penalty", None),
    }

    results: list[BenchmarkResult] = []
    run_index = 0
    for model in args.model:
        thread_candidates = _model_thread_sweep(model, args)
        for run_idx in range(args.runs):
            for decode_threads, batch_threads, ubatch in thread_candidates:
                run_index += 1
                try:
                    if bool(getattr(args, "isolated_child", True)):
                        result = _run_once(
                            model=model,
                            prompt=args.prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=getattr(args, "temperature", None),
                            sampling_profile=getattr(args, "sampling_profile", None),
                            override_params=override_params,
                            run_index=run_index,
                            context_length=int(getattr(args, "context_length", 2048)),
                            decode_threads=decode_threads,
                            batch_threads=batch_threads,
                            ubatch=ubatch,
                            native_fast_path=bool(getattr(args, "fast_native", False)),
                            disable_logits_processors=getattr(
                                args,
                                "disable_logits_processors",
                                None,
                            ),
                            disable_token_penalties=getattr(
                                args,
                                "disable_token_penalties",
                                None,
                            ),
                            force_parallel_decode=bool(
                                getattr(args, "force_parallel_decode", False)
                            ),
                            min_new_tokens_before_eos=getattr(
                                args,
                                "min_new_tokens_before_eos",
                                None,
                            ),
                            coherence_first=bool(
                                getattr(args, "coherence_first", False)
                            ),
                            self_artifact_dir=getattr(args, "self_artifact_dir", None),
                            capture_self_numa_map=bool(
                                getattr(args, "capture_self_numa_map", False)
                            ),
                        )
                    else:
                        result = _run_once_isolated(
                            model=model,
                            args=args,
                            override_params=override_params,
                            run_index=run_index,
                            decode_threads=decode_threads,
                            batch_threads=batch_threads,
                            ubatch=ubatch,
                        )
                except TypeError:
                    # Backward compatibility for monkeypatched harnesses.
                    try:
                        result = _run_once(
                            model=model,
                            prompt=args.prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=getattr(args, "temperature", None),
                            run_index=run_index,
                        )
                    except Exception as exc:
                        print(
                            f"[native_qsg_benchmark] compat run failed model={model} run_index={run_index}: {exc}",
                            file=sys.stderr,
                        )
                        traceback.print_exc(file=sys.stderr)
                        result = _runtime_failure_result(
                            model=model,
                            prompt=args.prompt,
                            max_new_tokens=args.max_new_tokens,
                            run_index=run_index,
                            context_length=int(getattr(args, "context_length", 2048)),
                            decode_threads=decode_threads,
                            batch_threads=batch_threads,
                            ubatch=ubatch,
                            profile_name=str(
                                getattr(args, "sampling_profile", None)
                                or "compat_fallback"
                            ),
                            params=override_params,
                            prompt_tokens=[],
                            runtime={},
                            load_seconds=0.0,
                            generate_seconds=0.0,
                            rss_before=_rss_mb(),
                            rss_after_load=_rss_mb(),
                            rss_after_generate=_rss_mb(),
                            exc=exc,
                        )
                except Exception as exc:
                    print(
                        f"[native_qsg_benchmark] main loop failed model={model} run_index={run_index}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    result = _runtime_failure_result(
                        model=model,
                        prompt=args.prompt,
                        max_new_tokens=args.max_new_tokens,
                        run_index=run_index,
                        context_length=int(getattr(args, "context_length", 2048)),
                        decode_threads=decode_threads,
                        batch_threads=batch_threads,
                        ubatch=ubatch,
                        profile_name=str(
                            getattr(args, "sampling_profile", None)
                            or getattr(args, "autotune", "off")
                            or "runtime_exception"
                        ),
                        params=override_params,
                        prompt_tokens=[],
                        runtime={},
                        load_seconds=0.0,
                        generate_seconds=0.0,
                        rss_before=_rss_mb(),
                        rss_after_load=_rss_mb(),
                        rss_after_generate=_rss_mb(),
                        exc=exc,
                    )
                result.run_index = run_index
                results.append(result)

    failures: dict[str, list[str]] = {}
    for result in results:
        issues = _result_failures(
            result,
            min_decode_tps=getattr(args, "min_decode_tps", None),
            max_ttft_ms=getattr(args, "max_ttft_ms", None),
            min_printable_ratio=getattr(args, "min_printable_ratio", None),
            max_repeated_4gram_ratio=getattr(args, "max_repeated_4gram_ratio", None),
            max_repeated_8gram_ratio=getattr(args, "max_repeated_8gram_ratio", None),
            min_ascii_ratio=getattr(args, "min_ascii_ratio", None),
            min_word_count=getattr(args, "min_word_count", None),
            max_rss_growth_mb=getattr(args, "max_rss_growth_mb", None),
            require_utf8=bool(getattr(args, "require_utf8", False)),
            require_measurement_valid=bool(
                getattr(args, "require_measurement_valid", False)
            ),
            require_openmp=bool(getattr(args, "require_openmp", False)),
            require_avx2=bool(getattr(args, "require_avx2", False)),
            require_mmap=bool(getattr(args, "require_mmap", False)),
            host_access=str(getattr(args, "host_access", "user")),
            collect_hw_counters=str(getattr(args, "collect_hw_counters", "auto")),
            require_grover=bool(getattr(args, "require_grover", False)),
            require_coconut=bool(getattr(args, "require_coconut", False)),
            strict_native_gates=not legacy_harness_mode,
        )
        if issues:
            failures[f"{result.model}#{result.run_index}"] = issues

    if str(getattr(args, "autotune", "off")).lower() == "explore":
        best_by_model: dict[str, BenchmarkResult] = {}
        for result in results:
            key = f"{result.model}#{result.run_index}"
            if failures.get(key):
                continue
            current = best_by_model.get(result.model)
            if (
                current is None
                or result.decode_throughput_tps > current.decode_throughput_tps
            ):
                best_by_model[result.model] = result
        for model, best in best_by_model.items():
            _write_autotune_profile(
                model,
                {
                    "schema_version": BENCHMARK_REPORT_SCHEMA,
                    "host_fingerprint": _host_fingerprint(),
                    "profile_id": f"{_safe_key(model)}:{best.decode_threads}x{best.batch_threads}x{best.ubatch}",
                    "decode_threads": int(best.decode_threads),
                    "batch_threads": int(best.batch_threads),
                    "ubatch": int(best.ubatch),
                    "score": float(best.decode_throughput_tps),
                    "exploration_count": len([r for r in results if r.model == model]),
                    "native_isa_baseline": str(best.native_isa_baseline),
                    "native_backend_abi_match": bool(best.native_backend_abi_match),
                },
            )

    criteria = {
        "min_decode_tps": getattr(args, "min_decode_tps", None),
        "max_ttft_ms": getattr(args, "max_ttft_ms", None),
        "min_printable_ratio": getattr(args, "min_printable_ratio", None),
        "max_repeated_4gram_ratio": getattr(args, "max_repeated_4gram_ratio", None),
        "max_repeated_8gram_ratio": getattr(args, "max_repeated_8gram_ratio", None),
        "min_ascii_ratio": getattr(args, "min_ascii_ratio", None),
        "min_word_count": getattr(args, "min_word_count", None),
        "max_rss_growth_mb": getattr(args, "max_rss_growth_mb", None),
        "require_utf8": bool(getattr(args, "require_utf8", False)),
        "require_measurement_valid": bool(
            getattr(args, "require_measurement_valid", False)
        ),
        "require_openmp": bool(getattr(args, "require_openmp", False)),
        "require_avx2": bool(getattr(args, "require_avx2", False)),
        "require_mmap": bool(getattr(args, "require_mmap", False)),
        "host_access": str(getattr(args, "host_access", "user")),
        "collect_hw_counters": str(getattr(args, "collect_hw_counters", "auto")),
        "require_grover": bool(getattr(args, "require_grover", False)),
        "require_coconut": bool(getattr(args, "require_coconut", False)),
        "autotune": str(getattr(args, "autotune", "off")),
    }
    report = _build_report(
        results,
        failures,
        prompt=args.prompt,
        criteria=criteria,
    )
    legacy_payload: list[dict[str, object]] = []
    for legacy_index, result in enumerate(results):
        row = _result_flat_record(result)
        row["run_index"] = legacy_index
        legacy_payload.append(row)

    if getattr(args, "json_out", None):
        json_path = pathlib.Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(legacy_payload if legacy_harness_mode else report, indent=2),
            encoding="utf-8",
        )
        _trace(f"wrote json report -> {json_path}")

    if getattr(args, "markdown_out", None):
        markdown_path = pathlib.Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            _markdown_report(results, failures),
            encoding="utf-8",
        )
        _trace(f"wrote markdown report -> {markdown_path}")

    if args.json:
        _trace("emitting json payload to stdout")
        print(json.dumps(legacy_payload if legacy_harness_mode else report, indent=2))
        return 1 if failures else 0

    print(
        "model\tthreads\tbatch_threads\tubatch\trun\tprofile\ttemplate\tmode\tthread_mode\tload_s\tgen_s\tttft_ms\tprefill_tps\tdecode_tps\tgraph_decode_avg_ms\tsample_avg_ms\tcache_hit%\tcoh_eos\tcoh_guard\tcoherence\tnew_tok\ttok/s\tkv_used\tprintable\trep4\tsample"
    )
    for r in results:
        sample = r.sample_text.replace("\n", " ").strip()
        if len(sample) > 90:
            sample = sample[:87] + "..."
        print(
            f"{r.model}\t{r.decode_threads}\t{r.batch_threads}\t{r.ubatch}\t{r.run_index}\t{r.sampling_profile}\t"
            f"{r.template_name or '-'}\t{r.granite_moe_mode or '-'}\t{r.active_thread_mode or '-'}\t"
            f"{r.load_seconds:.2f}\t{r.generate_seconds:.2f}\t{r.first_token_latency_seconds * 1000.0:.2f}\t"
            f"{r.effective_prefill_throughput_tps:.2f}\t{r.decode_throughput_tps:.2f}\t"
            f"{r.graph_decode_avg_ms:.2f}\t{r.sample_avg_ms:.2f}\t"
            f"{r.prompt_cache_hit_ratio * 100.0:.1f}\t{r.min_new_tokens_before_eos}\t"
            f"{r.coherence_guard_events}\t"
            f"{'PASS' if r.coherence_valid else 'FAIL'}\t{r.new_tokens}\t"
            f"{r.tokens_per_second:.2f}\t{r.kv_used_cells}\t"
            f"{r.printable_ratio:.2f}\t{r.repeated_4gram_ratio:.2f}\t{sample}"
        )

    print("\nsummary:")
    grouped: dict[tuple[str, int, int, int, str], list[BenchmarkResult]] = {}
    for r in results:
        grouped.setdefault(
            (r.model, r.decode_threads, r.batch_threads, r.ubatch, r.sampling_profile),
            [],
        ).append(r)
    for (
        model,
        decode_threads,
        batch_threads,
        ubatch,
        profile,
    ), values in grouped.items():
        print(
            f"{model} dec={decode_threads} batch={batch_threads} ubatch={ubatch} profile={profile}: "
            f"tok/s={statistics.mean(v.tokens_per_second for v in values):.2f} "
            f"decode_tps={statistics.mean(v.decode_throughput_tps for v in values):.2f} "
            f"prefill_tps={statistics.mean(v.effective_prefill_throughput_tps for v in values):.2f} "
            f"ttft_ms={statistics.mean(v.ttft_ms for v in values):.2f}"
        )

    if failures:
        print("\nfailures:")
        for key, issues in failures.items():
            print(f"{key}: {', '.join(issues)}")

    return 1 if failures else 0


def _deprecated_entrypoint(argv: list[str]) -> int:
    if "--isolated-child" in argv:
        return main()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    known, unknown = parser.parse_known_args(argv)

    legacy_flags = [token for token in unknown if str(token).startswith("-")]
    if legacy_flags:
        print(
            "ERROR: direct native_qsg_benchmark flags are deprecated for operator runs.\n"
            "Use `./scripts/run_native_qsg_suite.sh` and profile defaults.\n"
            f"Rejected flags: {', '.join(sorted(set(legacy_flags)))}",
            file=sys.stderr,
        )
        print(
            "Allowed compatibility flags: --profile, --run-id, --resume, --out-root, --experiment",
            file=sys.stderr,
        )
        return 2

    forwarded: list[str] = []
    if known.profile:
        forwarded.extend(["--profile", str(known.profile)])
    if known.run_id:
        forwarded.extend(["--run-id", str(known.run_id)])
    if known.resume:
        forwarded.append("--resume")
    if known.out_root:
        forwarded.extend(["--out-root", str(known.out_root)])
    if known.experiment:
        forwarded.extend(["--experiment", str(known.experiment)])

    suite_script = REPO_ROOT / "audit" / "runner" / "benchmark_suite.py"
    print(
        "DEPRECATED: `python benchmarks/native_qsg_benchmark.py` is not an operator entrypoint.\n"
        "Use: ./scripts/run_native_qsg_suite.sh",
        file=sys.stderr,
    )
    completed = subprocess.run(
        [sys.executable, str(suite_script), *forwarded],
        cwd=str(REPO_ROOT),
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(_deprecated_entrypoint(sys.argv[1:]))
