"""Runtime telemetry structures for native QSG."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any


def _safe_rate(units: int | float, seconds: float) -> float:
    if seconds <= 0.0 or units <= 0:
        return 0.0
    return float(units) / float(seconds)


def _percentile_ms(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = min(
        len(ordered) - 1, max(0, int(round((percentile / 100.0) * (len(ordered) - 1))))
    )
    return ordered[index] * 1000.0


def _benchmark_label_for_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "prompt_lookup":
        return "prompt_lookup"
    if normalized == "block_diffusion":
        return "block_diffusion_candidate"
    if normalized == "masked_diffusion":
        return "masked_diffusion_candidate"
    if normalized == "ssd_bridge":
        return "ssd_bridge"
    if normalized == "parallel_hybrid":
        return "parallel_hybrid"
    if normalized == "medusa_head":
        return "medusa_head_candidate"
    if normalized == "hydra_head":
        return "hydra_head_candidate"
    if normalized == "replacement":
        return "replacement_candidate"
    return "ar_baseline"


def build_runtime_capability_ledger(
    status: dict[str, Any],
    *,
    host_fingerprint: str = "",
    certification_state: str = "",
    source: str = "runtime_status",
) -> dict[str, Any]:
    """Normalize a stable runtime capability contract for audit surfaces."""
    normalized = dict(status or {})
    feature_flags = {
        "full_qsg_enabled": bool(normalized.get("full_qsg_enabled", False)),
        "full_graph_enabled": bool(normalized.get("full_graph_enabled", False)),
        "qsg_processors_native_enabled": bool(
            normalized.get("qsg_processors_native_enabled", False)
        ),
        "batched_prefill_native_enabled": bool(
            normalized.get("batched_prefill_native_enabled", False)
        ),
        "parallel_decode_allowed": bool(
            normalized.get("parallel_decode_allowed", False)
        ),
        "native_backend_abi_match": bool(
            normalized.get("native_backend_abi_match", False)
        ),
        "perf_event_access": bool(normalized.get("perf_event_access", False)),
    }
    degraded = [
        key
        for key, enabled in feature_flags.items()
        if key in {"native_backend_abi_match", "perf_event_access"} and not enabled
    ]
    optional_isa_leaves = [
        str(item).strip()
        for item in list(normalized.get("native_optional_isa_leaves", []) or [])
        if str(item).strip()
    ]
    amx_compiled = bool(
        normalized.get("native_compiled_with_amx", False)
        or normalized.get("amx_compiled", False)
        or "amx" in optional_isa_leaves
    )
    amx_runtime_available = bool(
        normalized.get("native_runtime_amx_available", False)
        or normalized.get("amx_runtime_available", False)
    )
    optional_isa_leaf_state = {
        "amx": {
            "compiled": amx_compiled,
            "runtime_available": amx_runtime_available,
            "readiness_impact": "none",
        }
    }
    digest_payload = {
        "model": str(normalized.get("model", "")),
        "digest": str(normalized.get("digest", "")),
        "native_isa_baseline": str(normalized.get("native_isa_baseline", "")),
        "decode_threads": int(normalized.get("decode_threads", 0) or 0),
        "batch_threads": int(normalized.get("batch_threads", 0) or 0),
        "ubatch": int(normalized.get("ubatch", 0) or 0),
        "scheduler_policy": str(normalized.get("scheduler_policy", "")),
        "max_active_requests": int(normalized.get("max_active_requests", 0) or 0),
        "batch_wait_timeout_ms": int(normalized.get("batch_wait_timeout_ms", 0) or 0),
        "max_prefill_rows_per_iteration": int(
            normalized.get("max_prefill_rows_per_iteration", 0) or 0
        ),
        "continuous_interleaved_streams": bool(
            normalized.get("continuous_interleaved_streams", False)
        ),
        "state_page_rows": int(normalized.get("state_page_rows", 0) or 0),
        "state_compaction_soft_threshold": float(
            normalized.get("state_compaction_soft_threshold", 0.0) or 0.0
        ),
        "state_compaction_hard_threshold": float(
            normalized.get("state_compaction_hard_threshold", 0.0) or 0.0
        ),
        "workload_digest": str(normalized.get("workload_digest", "")),
        "budget_tier": str(normalized.get("budget_tier", "")),
        "admission_decision": str(normalized.get("admission_decision", "")),
        "backend_module": str(normalized.get("backend_module", "")),
        "backend_module_loaded": bool(normalized.get("backend_module_loaded", False)),
        "affinity_policy": str(normalized.get("affinity_policy", "")),
        "worker_cpu_mask": str(normalized.get("worker_cpu_mask", "")),
        "orchestrator_cpu_mask": str(normalized.get("orchestrator_cpu_mask", "")),
        "l3_domain_ids_active": list(normalized.get("l3_domain_ids_active", []) or []),
        "native_optional_isa_leaves": optional_isa_leaves,
        "optional_isa_leaf_state": optional_isa_leaf_state,
        "feature_flags": feature_flags,
    }
    capability_digest = str(normalized.get("capability_digest") or "").strip()
    if not capability_digest:
        capability_digest = hashlib.sha1(
            json.dumps(digest_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    return {
        "schema_version": "native_qsg_suite.capability_ledger.v1",
        "source": source,
        "host_fingerprint": str(
            host_fingerprint or normalized.get("host_fingerprint") or ""
        ),
        "certification_state": str(certification_state or "").strip(),
        "capability_digest": capability_digest,
        "model": str(normalized.get("model", "")),
        "model_digest": str(normalized.get("digest", "")),
        "native_isa_baseline": str(normalized.get("native_isa_baseline", "")),
        "native_optional_isa_leaves": optional_isa_leaves,
        "optional_isa_leaf_state": optional_isa_leaf_state,
        "native_compiled_with_amx": amx_compiled,
        "native_runtime_amx_available": amx_runtime_available,
        "selected_kernel_mode": str(
            normalized.get("generation_mode")
            or normalized.get("active_thread_mode")
            or ""
        ),
        "thread_config": {
            "decode_threads": int(normalized.get("decode_threads", 0) or 0),
            "batch_threads": int(normalized.get("batch_threads", 0) or 0),
            "ubatch": int(normalized.get("ubatch", 0) or 0),
        },
        "continuous_config": {
            "scheduler_policy": str(normalized.get("scheduler_policy", "")),
            "max_active_requests": int(normalized.get("max_active_requests", 0) or 0),
            "batch_wait_timeout_ms": int(
                normalized.get("batch_wait_timeout_ms", 0) or 0
            ),
            "max_prefill_rows_per_iteration": int(
                normalized.get("max_prefill_rows_per_iteration", 0) or 0
            ),
            "continuous_interleaved_streams": bool(
                normalized.get("continuous_interleaved_streams", False)
            ),
        },
        "pager_config": {
            "state_page_rows": int(normalized.get("state_page_rows", 0) or 0),
            "state_compaction_soft_threshold": float(
                normalized.get("state_compaction_soft_threshold", 0.0) or 0.0
            ),
            "state_compaction_hard_threshold": float(
                normalized.get("state_compaction_hard_threshold", 0.0) or 0.0
            ),
        },
        "tuning": {
            "workload_digest": str(normalized.get("workload_digest", "")),
            "budget_tier": str(normalized.get("budget_tier", "")),
            "admission_decision": str(normalized.get("admission_decision", "")),
        },
        "affinity": {
            "policy": str(normalized.get("affinity_policy", "")),
            "worker_cpu_mask": str(normalized.get("worker_cpu_mask", "")),
            "orchestrator_cpu_mask": str(normalized.get("orchestrator_cpu_mask", "")),
            "l3_domain_ids_active": [
                int(value)
                for value in list(normalized.get("l3_domain_ids_active", []) or [])
            ],
        },
        "feature_flags": feature_flags,
        "degraded_capabilities": degraded,
    }


@dataclass
class NativeGenerationTelemetry:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    first_token_latency_seconds: float = 0.0
    prompt_format_seconds: float = 0.0
    tokenize_seconds: float = 0.0
    embedding_lookup_seconds: float = 0.0
    graph_prefill_seconds: float = 0.0
    graph_decode_seconds: float = 0.0
    sample_seconds: float = 0.0
    logits_processor_seconds: float = 0.0
    penalty_seconds: float = 0.0
    suppression_seconds: float = 0.0
    graph_prefill_calls: int = 0
    graph_decode_calls: int = 0
    sample_calls: int = 0
    logits_processor_calls: int = 0
    penalty_calls: int = 0
    suppression_calls: int = 0
    grammar_fastlane_calls: int = 0
    native_fast_path: bool = False
    parallel_decode_disable_reason: str = ""
    per_token_latencies_seconds: list[float] = field(default_factory=list)
    stop_reason: str = "unknown"
    coherence_guard_events: int = 0
    kv_used_cells: int = 0
    kv_fragmentation_ratio: float = 0.0
    kv_defrag_count: int = 0
    kv_cache_quantization: str = ""
    template_name: str = ""
    granite_moe_mode: str = ""
    active_thread_mode: str = ""
    prefill_chunk_count: int = 0
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
    jacobi_frontier_width: int = 0
    jacobi_branch_survival_rate: float = 0.0
    jacobi_verify_cost_ms: float = 0.0
    jacobi_branch_entropy: float = 0.0
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
    prompt_cache_hit: bool = False
    prompt_cache_hits: int = 0
    prompt_cache_misses: int = 0
    prompt_cache_reused_tokens: int = 0
    schema_valid: bool = False
    tool_call_parse_success: bool = False
    measurement_valid: bool = True
    native_build_id: str = ""
    native_build_sha256: str = ""
    loaded_native_library: str = ""
    native_split_layout: str = ""
    native_public_load_target: str = ""
    native_runtime_core_target: str = ""
    native_split_abi_version: int = 0
    native_isa_baseline: str = ""
    native_compat_aliases: list[str] = field(default_factory=list)
    sanctioned_backend_path: str = ""
    tokenizer_backend: str = ""
    backend_module: str = ""
    backend_module_requested: str = ""
    backend_module_library: str = ""
    backend_module_loaded: bool = False
    backend_module_candidates: list[str] = field(default_factory=list)
    backend_selection_source: str = ""
    backend_selection_reason: str = ""
    backend_selection_model_name: str = ""
    backend_selection_architecture: str = ""
    backend_selection_family: str = ""
    backend_module_marker_symbol: str = ""
    backend_module_marker: int = 0
    backend_module_name_symbol: str = ""
    backend_module_name: str = ""
    backend_module_build_id_symbol: str = ""
    backend_module_build_id: str = ""
    backend_module_abi_symbol: str = ""
    backend_module_abi_version: int = 0
    backend_module_required: bool = False
    omp_max_threads: int = 0
    omp_dynamic: bool = False
    omp_active_levels: int = 0
    sampling_backend: str = ""
    penalties_backend: str = ""
    suppression_backend: str = ""
    logits_backend: str = ""
    full_qsg_enabled: bool = False
    full_graph_enabled: bool = False
    batched_prefill_native_enabled: bool = False
    qsg_processors_native_enabled: bool = False
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
    strict_cpp_only: bool = False
    strict_native_qsg: bool = False
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
    native_backend_abi_match: bool = False
    perf_event_access: bool = False
    perf_event_access_reason: str = ""
    cpu_governor: str = ""
    thp_mode: str = ""
    perf_counter_source: str = ""
    l3_domain_ids_active: list[int] = field(default_factory=list)
    worker_cpu_mask: str = ""
    orchestrator_cpu_mask: str = ""
    autotune_profile_id: str = ""
    autotune_source: str = ""
    autotune_score: float = 0.0
    autotune_exploration_count: int = 0
    hot_path_proof: dict[str, str] = field(default_factory=dict)
    controller_state: dict[str, Any] = field(default_factory=dict)
    execution_capsule_id: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    performance_twin: dict[str, Any] = field(default_factory=dict)
    repo_coupled_runtime: dict[str, Any] = field(default_factory=dict)
    runtime_policy_id: str = ""
    runtime_context_band: str = ""
    runtime_model_family: str = ""
    runtime_task_hint: str = ""
    runtime_cache_mode: str = ""
    runtime_shortlist_size: int = 0
    runtime_prompt_tokens_estimate: int = 0
    graph_stage_seconds: dict[str, float] = field(default_factory=dict)
    graph_stage_calls: dict[str, int] = field(default_factory=dict)
    batched_prefill_token_id_calls: int = 0
    batched_prefill_token_id_tokens: int = 0
    batch_token_fallback_count: int = 0
    packed_lm_head_calls: int = 0
    speculative_accept_count: int = 0
    speculative_reject_count: int = 0
    self_spec_native_path: bool = False
    self_spec_policy: str = ""
    self_spec_exit_layer: int = 0
    self_spec_exit_fraction: float = 0.0
    self_spec_draft_tokens: int = 0
    prefill_phase_gap_ms: float = 0.0
    decode_phase_gap_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        prompt_cache_lookups = max(
            0,
            int(self.prompt_cache_hits) + int(self.prompt_cache_misses),
        )
        effective_prefill_tokens = max(
            0,
            int(self.prompt_tokens) - int(self.prompt_cache_reused_tokens),
        )
        measurement_issues: list[str] = []
        total_seconds = float(self.total_seconds)
        prefill_seconds = float(self.prefill_seconds)
        decode_seconds = float(self.decode_seconds)
        ttft_seconds = float(self.first_token_latency_seconds)
        tolerance_seconds = 0.05
        if min(total_seconds, prefill_seconds, decode_seconds, ttft_seconds) < 0.0:
            measurement_issues.append("negative_duration")
        if total_seconds > 0.0 and ttft_seconds > total_seconds + tolerance_seconds:
            measurement_issues.append("ttft_exceeds_total")
        if total_seconds > 0.0 and prefill_seconds > total_seconds + tolerance_seconds:
            measurement_issues.append("prefill_exceeds_total")
        if total_seconds > 0.0 and decode_seconds > total_seconds + tolerance_seconds:
            measurement_issues.append("decode_exceeds_total")
        if (
            total_seconds > 0.0
            and (prefill_seconds + decode_seconds) > total_seconds + tolerance_seconds
        ):
            measurement_issues.append("prefill_plus_decode_exceeds_total")
        if bool(self.backend_module_required):
            if not str(self.backend_module).strip():
                measurement_issues.append("backend_module_missing")
            if not bool(self.backend_module_loaded):
                measurement_issues.append("backend_module_not_loaded")
            if str(self.backend_selection_source).strip() in {"", "unresolved"}:
                measurement_issues.append("backend_selection_unresolved")

        payload = asdict(self)
        payload["native_split_abi_version"] = int(
            payload.get("native_split_abi_version", 0)
        )
        payload["backend_module_abi_version"] = int(
            payload.get("backend_module_abi_version", 0)
        )
        payload["prompt_category"] = str(payload.get("prompt_category", "")).strip()
        payload["temperature_band"] = str(payload.get("temperature_band", "")).strip()
        payload["draft_source"] = str(payload.get("draft_source", "")).strip()
        payload["native_compat_aliases"] = [
            str(alias).strip()
            for alias in payload.get("native_compat_aliases", [])
            if str(alias).strip()
        ]
        payload["controller_state"] = dict(payload.get("controller_state") or {})
        payload["execution_capsule_id"] = str(
            payload.get("execution_capsule_id", "")
        ).strip()
        payload["delta_watermark"] = dict(payload.get("delta_watermark") or {})
        payload["performance_twin"] = dict(payload.get("performance_twin") or {})
        payload["repo_coupled_runtime"] = dict(
            payload.get("repo_coupled_runtime") or {}
        )
        payload["runtime_policy_id"] = str(payload.get("runtime_policy_id", "")).strip()
        payload["runtime_context_band"] = str(
            payload.get("runtime_context_band", "")
        ).strip()
        payload["runtime_model_family"] = str(
            payload.get("runtime_model_family", "")
        ).strip()
        payload["runtime_task_hint"] = str(payload.get("runtime_task_hint", "")).strip()
        payload["runtime_cache_mode"] = str(
            payload.get("runtime_cache_mode", "")
        ).strip()
        payload["runtime_shortlist_size"] = int(
            payload.get("runtime_shortlist_size", 0)
        )
        payload["runtime_prompt_tokens_estimate"] = int(
            payload.get("runtime_prompt_tokens_estimate", 0)
        )
        payload["backend_module_candidates"] = [
            str(candidate).strip()
            for candidate in payload.get("backend_module_candidates", [])
            if str(candidate).strip()
        ]
        hot_path_counter_names = (
            "python_hot_path_calls",
            "numpy_hot_path_calls",
            "python_qsg_forward_calls",
            "python_attention_fallback_calls",
            "python_ssm_fallback_calls",
            "python_moe_fallback_calls",
            "llama_cpp_hot_path_calls",
        )
        hot_path_detected = bool(self.hot_path_numpy_detected)
        for name in hot_path_counter_names:
            payload[name] = int(payload.get(name, 0))
            hot_path_detected = hot_path_detected or payload[name] > 0
        payload["hot_path_numpy_detected"] = hot_path_detected
        payload["full_qsg_enabled"] = bool(payload.get("full_qsg_enabled", False)) or (
            bool(payload.get("full_graph_enabled", False))
            and bool(payload.get("qsg_processors_native_enabled", False))
            and bool(payload.get("batched_prefill_native_enabled", False))
            and str(payload.get("tokenizer_backend", "")) == "native"
            and bool(str(payload.get("sanctioned_backend_path", "")).strip())
        )
        payload["context_stabilizer_enabled"] = bool(
            payload.get("context_stabilizer_enabled", False)
        )
        payload["context_stabilizer_mode"] = str(
            payload.get("context_stabilizer_mode", "")
        )
        payload["drift_latest"] = float(payload.get("drift_latest", 0.0))
        payload["drift_mean"] = float(payload.get("drift_mean", 0.0))
        payload["drift_max"] = float(payload.get("drift_max", 0.0))
        payload["drift_decay_ratio"] = float(payload.get("drift_decay_ratio", 1.0))
        payload["drift_damped_blocks"] = int(payload.get("drift_damped_blocks", 0))
        payload["drift_pruned_blocks"] = int(payload.get("drift_pruned_blocks", 0))
        payload["drift_active_tokens"] = int(payload.get("drift_active_tokens", 0))
        payload["stabilizer_seconds"] = float(payload.get("stabilizer_seconds", 0.0))
        payload["stabilizer_calls"] = int(payload.get("stabilizer_calls", 0))
        payload["drift_auto_downgrade_events"] = int(
            payload.get("drift_auto_downgrade_events", 0)
        )
        payload["coconut_enabled"] = bool(payload.get("coconut_enabled", False))
        payload["coconut_paths"] = int(payload.get("coconut_paths", 0))
        payload["coconut_alpha"] = float(payload.get("coconut_alpha", 0.0))
        payload["coconut_seconds"] = float(payload.get("coconut_seconds", 0.0))
        payload["coconut_candidate_count"] = int(
            payload.get("coconut_candidate_count", 0)
        )
        payload["coconut_entropy_mean"] = float(
            payload.get("coconut_entropy_mean", 0.0)
        )
        payload["coconut_amplitude_mean"] = float(
            payload.get("coconut_amplitude_mean", 0.0)
        )
        payload["coconut_consistency_rejects"] = int(
            payload.get("coconut_consistency_rejects", 0)
        )
        payload["grover_enabled"] = bool(payload.get("grover_enabled", False))
        payload["grover_top_k"] = int(payload.get("grover_top_k", 0))
        payload["grover_damping"] = float(payload.get("grover_damping", 0.0))
        payload["grover_calls"] = int(payload.get("grover_calls", 0))
        payload["grover_seconds"] = float(payload.get("grover_seconds", 0.0))
        payload["grover_candidate_count"] = int(
            payload.get("grover_candidate_count", 0)
        )
        payload["grover_rescore_delta_mean"] = float(
            payload.get("grover_rescore_delta_mean", 0.0)
        )
        payload["accepted_parallel_tokens"] = int(
            payload.get("accepted_parallel_tokens", 0)
        )
        payload["rejected_parallel_tokens"] = int(
            payload.get("rejected_parallel_tokens", 0)
        )
        payload["proposed_parallel_tokens"] = int(
            payload.get(
                "proposed_parallel_tokens",
                payload["accepted_parallel_tokens"]
                + payload["rejected_parallel_tokens"],
            )
        )
        payload["draft_frontier_width"] = int(payload.get("draft_frontier_width", 0))
        payload["verify_depth"] = int(payload.get("verify_depth", 0))
        payload["jacobi_frontier_width"] = int(
            payload.get("jacobi_frontier_width", payload["draft_frontier_width"])
        )
        payload["jacobi_branch_survival_rate"] = float(
            payload.get("jacobi_branch_survival_rate", 0.0)
        )
        payload["jacobi_verify_cost_ms"] = float(
            payload.get("jacobi_verify_cost_ms", 0.0)
        )
        payload["jacobi_branch_entropy"] = float(
            payload.get("jacobi_branch_entropy", 0.0)
        )
        payload["parallel_step_latency_ms"] = float(
            payload.get("parallel_step_latency_ms", 0.0)
        )
        payload["draft_confidence_mean"] = float(
            payload.get("draft_confidence_mean", 0.0)
        )
        payload["draft_confidence_min"] = float(
            payload.get("draft_confidence_min", 0.0)
        )
        payload["grover_timeout_events"] = int(payload.get("grover_timeout_events", 0))
        payload["strict_cpp_only"] = bool(payload.get("strict_cpp_only", False))
        payload["strict_native_qsg"] = bool(payload.get("strict_native_qsg", False))
        payload["affinity_mode"] = int(payload.get("affinity_mode", 1))
        payload["l3_domain_count"] = int(payload.get("l3_domain_count", 0))
        payload["numa_strict"] = bool(payload.get("numa_strict", False))
        payload["numa_affinity_mode"] = str(payload.get("numa_affinity_mode", ""))
        payload["numa_hugepage"] = str(payload.get("numa_hugepage", ""))
        payload["numa_bind_policy"] = str(payload.get("numa_bind_policy", ""))
        payload["numa_first_touch"] = bool(payload.get("numa_first_touch", False))
        payload["topology_json"] = str(payload.get("topology_json", ""))
        payload["os_thread_migrations"] = int(payload.get("os_thread_migrations", 0))
        payload["os_last_cpu"] = int(payload.get("os_last_cpu", -1))
        payload["native_isa_baseline"] = str(payload.get("native_isa_baseline", ""))
        payload["native_backend_abi_match"] = bool(
            payload.get("native_backend_abi_match", False)
        )
        payload["perf_event_access"] = bool(payload.get("perf_event_access", False))
        payload["perf_event_access_reason"] = str(
            payload.get("perf_event_access_reason", "")
        )
        payload["cpu_governor"] = str(payload.get("cpu_governor", ""))
        payload["thp_mode"] = str(payload.get("thp_mode", ""))
        payload["perf_counter_source"] = str(payload.get("perf_counter_source", ""))
        payload["l3_domain_ids_active"] = [
            int(value) for value in payload.get("l3_domain_ids_active", []) or []
        ]
        payload["worker_cpu_mask"] = str(payload.get("worker_cpu_mask", ""))
        payload["orchestrator_cpu_mask"] = str(payload.get("orchestrator_cpu_mask", ""))
        payload["autotune_profile_id"] = str(payload.get("autotune_profile_id", ""))
        payload["autotune_source"] = str(payload.get("autotune_source", ""))
        payload["autotune_score"] = float(payload.get("autotune_score", 0.0))
        payload["autotune_exploration_count"] = int(
            payload.get("autotune_exploration_count", 0)
        )
        if decode_seconds > 0.0 and payload["stabilizer_seconds"] > 0.0:
            payload["drift_overhead_percent"] = (
                payload["stabilizer_seconds"] / decode_seconds
            ) * 100.0
        else:
            payload["drift_overhead_percent"] = max(
                0.0,
                float(payload.get("drift_overhead_percent", 0.0)),
            )
        payload["strict_path_stable"] = bool(
            payload.get("strict_path_stable", False)
            or (
                payload["full_qsg_enabled"]
                and not bool(payload.get("hot_path_numpy_detected", False))
                and (payload["strict_cpp_only"] or payload["strict_native_qsg"])
            )
        )
        payload["measurement_valid"] = (
            bool(self.measurement_valid) and not measurement_issues
        )
        payload["measurement_issues"] = measurement_issues
        payload["measurement_issue_count"] = len(measurement_issues)
        payload["total_ms"] = total_seconds * 1000.0
        payload["prefill_ms"] = prefill_seconds * 1000.0
        payload["decode_ms"] = decode_seconds * 1000.0
        payload["ttft_seconds"] = ttft_seconds
        payload["ttft_ms"] = ttft_seconds * 1000.0
        payload["prompt_format_ms"] = float(self.prompt_format_seconds) * 1000.0
        payload["tokenize_ms"] = float(self.tokenize_seconds) * 1000.0
        payload["embedding_lookup_ms"] = float(self.embedding_lookup_seconds) * 1000.0
        payload["graph_prefill_ms"] = float(self.graph_prefill_seconds) * 1000.0
        payload["graph_decode_ms"] = float(self.graph_decode_seconds) * 1000.0
        payload["sample_ms"] = float(self.sample_seconds) * 1000.0
        payload["logits_processor_ms"] = float(self.logits_processor_seconds) * 1000.0
        payload["penalty_ms"] = float(self.penalty_seconds) * 1000.0
        payload["suppression_ms"] = float(self.suppression_seconds) * 1000.0
        payload["graph_prefill_calls"] = int(self.graph_prefill_calls)
        payload["graph_decode_calls"] = int(self.graph_decode_calls)
        payload["sample_calls"] = int(self.sample_calls)
        payload["logits_processor_calls"] = int(self.logits_processor_calls)
        payload["penalty_calls"] = int(self.penalty_calls)
        payload["suppression_calls"] = int(self.suppression_calls)
        payload["graph_prefill_avg_ms"] = (
            payload["graph_prefill_ms"] / float(self.graph_prefill_calls)
            if self.graph_prefill_calls > 0
            else 0.0
        )
        payload["graph_decode_avg_ms"] = (
            payload["graph_decode_ms"] / float(self.graph_decode_calls)
            if self.graph_decode_calls > 0
            else 0.0
        )
        payload["sample_avg_ms"] = (
            payload["sample_ms"] / float(self.sample_calls)
            if self.sample_calls > 0
            else 0.0
        )
        payload["logits_processor_avg_ms"] = (
            payload["logits_processor_ms"] / float(self.logits_processor_calls)
            if self.logits_processor_calls > 0
            else 0.0
        )
        payload["penalty_avg_ms"] = (
            payload["penalty_ms"] / float(self.penalty_calls)
            if self.penalty_calls > 0
            else 0.0
        )
        payload["suppression_avg_ms"] = (
            payload["suppression_ms"] / float(self.suppression_calls)
            if self.suppression_calls > 0
            else 0.0
        )
        payload["prefill_phase_gap_ms"] = payload["prefill_ms"] - (
            payload["prompt_format_ms"]
            + payload["tokenize_ms"]
            + payload["embedding_lookup_ms"]
            + payload["graph_prefill_ms"]
        )
        payload["decode_phase_gap_ms"] = payload["decode_ms"] - (
            payload["graph_decode_ms"]
            + payload["sample_ms"]
            + payload["logits_processor_ms"]
            + payload["penalty_ms"]
            + payload["suppression_ms"]
        )
        payload["prefill_throughput_tps"] = _safe_rate(
            self.prompt_tokens, prefill_seconds
        )
        payload["effective_prefill_tokens"] = effective_prefill_tokens
        payload["effective_prefill_throughput_tps"] = _safe_rate(
            effective_prefill_tokens,
            prefill_seconds,
        )
        payload["decode_throughput_tps"] = _safe_rate(
            self.generated_tokens, decode_seconds
        )
        payload["end_to_end_throughput_tps"] = _safe_rate(
            self.generated_tokens, total_seconds
        )
        payload["prompt_cache_lookups"] = prompt_cache_lookups
        payload["prompt_cache_hit_ratio"] = (
            float(self.prompt_cache_hits) / float(prompt_cache_lookups)
            if prompt_cache_lookups > 0
            else 0.0
        )
        payload["prompt_cache_reuse_ratio"] = (
            float(self.prompt_cache_reused_tokens) / float(self.prompt_tokens)
            if self.prompt_tokens > 0
            else 0.0
        )
        payload["prefix_cache_hit_rate"] = (
            float(self.prefix_cache_hit_rate)
            if float(self.prefix_cache_hit_rate) > 0.0
            else float(payload["prompt_cache_hit_ratio"])
        )
        payload["benchmark_label"] = str(
            payload.get("benchmark_label") or ""
        ).strip() or (
            _benchmark_label_for_mode(str(payload.get("generation_mode", "")))
        )
        payload["per_token_latency_p50_ms"] = (
            median(self.per_token_latencies_seconds) * 1000.0
            if self.per_token_latencies_seconds
            else 0.0
        )
        payload["per_token_latency_p95_ms"] = _percentile_ms(
            self.per_token_latencies_seconds, 95.0
        )
        payload["graph_stage_ms"] = {
            str(name): float(seconds) * 1000.0
            for name, seconds in self.graph_stage_seconds.items()
        }
        payload["graph_stage_avg_ms"] = {
            str(name): (
                (float(self.graph_stage_seconds.get(name, 0.0)) * 1000.0)
                / float(max(1, int(self.graph_stage_calls.get(name, 0))))
            )
            for name in self.graph_stage_seconds
        }
        payload["graph_stage_calls"] = {
            str(name): int(count) for name, count in self.graph_stage_calls.items()
        }
        hot_path_proof = dict(payload.get("hot_path_proof", {}))
        hot_path_proof["sanctioned_backend_path"] = str(
            payload.get("sanctioned_backend_path", "")
        )
        hot_path_proof["tokenizer_backend"] = str(payload.get("tokenizer_backend", ""))
        hot_path_proof["backend_module"] = str(payload.get("backend_module", ""))
        hot_path_proof["backend_selection_source"] = str(
            payload.get("backend_selection_source", "")
        )
        hot_path_proof["backend_selection_reason"] = str(
            payload.get("backend_selection_reason", "")
        )
        hot_path_proof["backend_module_loaded"] = (
            "true" if bool(payload.get("backend_module_loaded", False)) else "false"
        )
        hot_path_proof["native_split_layout"] = str(
            payload.get("native_split_layout", "")
        )
        hot_path_proof["native_public_load_target"] = str(
            payload.get("native_public_load_target", "")
        )
        hot_path_proof["native_runtime_core_target"] = str(
            payload.get("native_runtime_core_target", "")
        )
        hot_path_proof["native_isa_baseline"] = str(
            payload.get("native_isa_baseline", "")
        )
        hot_path_proof["native_split_abi_version"] = str(
            payload.get("native_split_abi_version", 0)
        )
        hot_path_proof["backend_module_abi_version"] = str(
            payload.get("backend_module_abi_version", 0)
        )
        hot_path_proof["native_backend_abi_match"] = (
            "true" if bool(payload.get("native_backend_abi_match", False)) else "false"
        )
        hot_path_proof["context_stabilizer_enabled"] = (
            "true" if payload["context_stabilizer_enabled"] else "false"
        )
        hot_path_proof["context_stabilizer_mode"] = str(
            payload.get("context_stabilizer_mode", "")
        )
        hot_path_proof["drift_overhead_percent"] = (
            f"{float(payload.get('drift_overhead_percent', 0.0)):.3f}"
        )
        hot_path_proof["coconut_enabled"] = (
            "true" if bool(payload.get("coconut_enabled", False)) else "false"
        )
        hot_path_proof["grover_enabled"] = (
            "true" if bool(payload.get("grover_enabled", False)) else "false"
        )
        hot_path_proof["perf_event_access"] = (
            "true" if bool(payload.get("perf_event_access", False)) else "false"
        )
        hot_path_proof["perf_event_access_reason"] = str(
            payload.get("perf_event_access_reason", "")
        )
        hot_path_proof["cpu_governor"] = str(payload.get("cpu_governor", ""))
        hot_path_proof["thp_mode"] = str(payload.get("thp_mode", ""))
        hot_path_proof["perf_counter_source"] = str(
            payload.get("perf_counter_source", "")
        )
        hot_path_proof["worker_cpu_mask"] = str(payload.get("worker_cpu_mask", ""))
        hot_path_proof["orchestrator_cpu_mask"] = str(
            payload.get("orchestrator_cpu_mask", "")
        )
        hot_path_proof["l3_domain_ids_active"] = ",".join(
            str(value) for value in payload.get("l3_domain_ids_active", [])
        )
        hot_path_proof["autotune_profile_id"] = str(
            payload.get("autotune_profile_id", "")
        )
        hot_path_proof["autotune_source"] = str(payload.get("autotune_source", ""))
        hot_path_proof["strict_cpp_only"] = (
            "true" if bool(payload.get("strict_cpp_only", False)) else "false"
        )
        hot_path_proof["strict_native_qsg"] = (
            "true" if bool(payload.get("strict_native_qsg", False)) else "false"
        )
        hot_path_proof["strict_path_stable"] = (
            "true" if bool(payload.get("strict_path_stable", False)) else "false"
        )
        hot_path_proof["runtime_policy_id"] = str(payload.get("runtime_policy_id", ""))
        hot_path_proof["runtime_context_band"] = str(
            payload.get("runtime_context_band", "")
        )
        hot_path_proof["runtime_model_family"] = str(
            payload.get("runtime_model_family", "")
        )
        hot_path_proof["runtime_task_hint"] = str(payload.get("runtime_task_hint", ""))
        hot_path_proof["runtime_cache_mode"] = str(
            payload.get("runtime_cache_mode", "")
        )
        hot_path_proof["runtime_shortlist_size"] = str(
            payload.get("runtime_shortlist_size", 0)
        )
        hot_path_proof["executed_cpp_only"] = (
            "true" if not payload["hot_path_numpy_detected"] else "false"
        )
        hot_path_proof["python_or_numpy_hot_path"] = (
            "not_detected" if not payload["hot_path_numpy_detected"] else "detected"
        )
        hot_path_proof["full_qsg"] = (
            "enabled" if payload["full_qsg_enabled"] else "disabled"
        )
        for name in hot_path_counter_names:
            hot_path_proof[name] = str(payload[name])
        payload["hot_path_proof"] = hot_path_proof
        return payload
