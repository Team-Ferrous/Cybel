from array import array
import json
import time

import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import Mock

from config.settings import PERFORMANCE_CONFIG
from core.native import native_qsg_engine as native_qsg_engine_module
from core.native import parallel_generation as parallel_generation_module
from core.native import simd_ops_wrapper
from core.native.parallel_generation import (
    DraftCandidateBundle,
    GenerationEvidence,
    GenerationMode,
    ParallelDecodePlanner,
)
from core.native.parallel_decode import JacobiDecoder
from core.native.native_qsg_engine import NativeQSGEngine
from core.native.runtime_telemetry import NativeGenerationTelemetry


class _EngineStub:
    def __init__(self):
        self._step = 0
        self.sample_calls = 0

    def token_eos(self):
        return 99

    def _get_logits_for_tokens(self, token_ids):
        logits = np.full((8,), -100.0, dtype=np.float32)
        next_token = min(3 + len(token_ids), 7)
        logits[next_token] = 10.0
        return logits

    @staticmethod
    def _apply_logits_processors(tokens, logits, processors):
        for processor in processors:
            logits = processor(tokens, logits)
        return logits

    def _sample(self, logits, **kwargs):
        self.sample_calls += 1
        return int(np.argmax(logits))


def test_jacobi_decoder_accepts_matching_prefix():
    engine = _EngineStub()
    decoder = JacobiDecoder(width=3)
    result = decoder.decode(
        engine=engine,
        prompt_tokens=[1, 2],
        max_tokens=3,
        temperature=0.7,
    )

    assert result.accepted >= 1
    assert result.tokens
    assert engine.sample_calls == 0


def test_parallel_decode_gate_prefers_sequential_for_short_deterministic_runs():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._force_parallel_decode = False
    engine._parallel_decode_allowed = True
    engine._parallel_decode_min_new_tokens = 32
    engine._parallel_decode_min_prompt_tokens = 64

    prev = PERFORMANCE_CONFIG.get("parallel_decode", False)
    PERFORMANCE_CONFIG["parallel_decode"] = True
    try:
        assert not engine._should_parallel_decode(
            [1, 2, 3], max_new_tokens=8, temperature=0.0
        )
        assert not engine._should_parallel_decode(
            [1] * 16, max_new_tokens=64, temperature=0.7
        )
        assert engine._should_parallel_decode(
            [1] * 128, max_new_tokens=64, temperature=0.7
        )
    finally:
        PERFORMANCE_CONFIG["parallel_decode"] = prev


def test_parallel_decode_is_no_longer_architecture_gated():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._force_parallel_decode = False
    engine._parallel_decode_allowed = True
    engine._parallel_decode_min_new_tokens = 32
    engine._parallel_decode_min_prompt_tokens = 64

    prev = PERFORMANCE_CONFIG.get("parallel_decode", False)
    PERFORMANCE_CONFIG["parallel_decode"] = True
    try:
        assert engine._should_parallel_decode(
            [1] * 128, max_new_tokens=64, temperature=0.7
        )
    finally:
        PERFORMANCE_CONFIG["parallel_decode"] = prev


def test_parallel_decode_force_override_still_allows_parallel_entry():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._force_parallel_decode = True
    engine._parallel_decode_allowed = True
    engine._parallel_decode_min_new_tokens = 32
    engine._parallel_decode_min_prompt_tokens = 64

    prev = PERFORMANCE_CONFIG.get("parallel_decode", False)
    PERFORMANCE_CONFIG["parallel_decode"] = True
    try:
        assert engine._should_parallel_decode(
            [1] * 128, max_new_tokens=64, temperature=0.7
        )
    finally:
        PERFORMANCE_CONFIG["parallel_decode"] = prev


def test_parallel_planner_accepts_prompt_lookup_prefix():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.05
    engine._parallel_prompt_lookup_window = 4
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._ssm_spec_decoder = None
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine.token_eos = lambda: 99
    engine._apply_logits_processors = NativeQSGEngine._apply_logits_processors

    def _lookup_logits(tokens):
        next_token = 1 if len(tokens) == 5 else 2
        logits = np.full((4,), -10.0, dtype=np.float32)
        logits[next_token] = 10.0
        return logits

    engine._get_logits_for_tokens = _lookup_logits

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[0, 1, 2, 1, 2],
        max_new_tokens=4,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.PROMPT_LOOKUP
    assert plan.accepted_prefix_tokens == [1, 2]
    assert plan.evidence.generation_mode == GenerationMode.PROMPT_LOOKUP.value
    assert plan.evidence.benchmark_label == "prompt_lookup"
    assert plan.evidence.accepted_parallel_tokens == 2
    assert plan.evidence.draft_frontier_width >= 2


def test_parallel_planner_selects_block_diffusion_candidate():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_enabled = True
    engine._block_diffusion_native_ready = True
    engine._block_diffusion_force = True
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = False
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 3
    engine._prefix_cache_misses = 1
    engine._scheduler_queue_wait_ms = 1.5
    engine._scheduler_iteration_ms = 2.5
    engine._scheduler_kv_fragmentation_ratio = 0.2
    engine._ssm_spec_decoder = None
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1] * 300,
        max_new_tokens=128,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.BLOCK_DIFFUSION
    assert plan.evidence.benchmark_label == "block_diffusion_candidate"
    assert plan.evidence.kv_fragmentation_ratio == 0.2


def test_parallel_planner_selects_masked_diffusion_candidate():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._masked_diffusion_enabled = True
    engine._masked_diffusion_native_ready = True
    engine._masked_diffusion_force = True
    engine._block_diffusion_enabled = False
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = False
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 1
    engine._prefix_cache_misses = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._scheduler_kv_fragmentation_ratio = 0.0
    engine._ssm_spec_decoder = None
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1] * 300,
        max_new_tokens=128,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.MASKED_DIFFUSION
    assert plan.evidence.benchmark_label == "masked_diffusion_candidate"


def test_parallel_planner_selects_replacement_candidate(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_enabled = False
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = True
    engine._parallel_replacement_top_k = 8
    engine._parallel_replacement_max_tree_width = 4
    engine._parallel_replacement_acceptance_floor = 0.01
    engine._parallel_replacement_max_draft_tokens = 4
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._scheduler_kv_fragmentation_ratio = 0.0
    engine._ssm_spec_decoder = None
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine.token_eos = lambda: 99

    def _logits(_tokens):
        logits = np.full((6,), -10.0, dtype=np.float32)
        logits[3] = 10.0
        return logits

    engine._get_logits_for_tokens = _logits
    engine._apply_logits_processors = NativeQSGEngine._apply_logits_processors
    engine._should_parallel_decode = lambda *args, **kwargs: True  # noqa: ARG005

    monkeypatch.setattr(
        "core.native.parallel_generation.qsg_eagle_replacement_draft",
        lambda *args, **kwargs: ([3], [0.95]),  # noqa: ARG005
    )

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1, 2, 3],
        max_new_tokens=8,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.REPLACEMENT
    assert plan.accepted_prefix_tokens == [3]
    assert plan.evidence.benchmark_label == "replacement_candidate"


def test_parallel_planner_selects_medusa_head_candidate(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_enabled = False
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = False
    engine._medusa_head_enabled = True
    engine._hydra_head_enabled = False
    engine._medusa_head_max_draft_tokens = 4
    engine._medusa_head_top_k = 8
    engine._medusa_head_acceptance_floor = 0.01
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._scheduler_kv_fragmentation_ratio = 0.0
    engine._ssm_spec_decoder = None
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine.token_eos = lambda: 99
    engine._draft_model_head_candidates = lambda **kwargs: [3, 4]  # noqa: ARG005

    def _logits(_tokens):
        logits = np.full((6,), -10.0, dtype=np.float32)
        logits[3] = 10.0
        logits[4] = 9.0
        return logits

    engine._get_logits_for_tokens = _logits
    engine._apply_logits_processors = NativeQSGEngine._apply_logits_processors
    engine._should_parallel_decode = lambda *args, **kwargs: True  # noqa: ARG005
    monkeypatch.setattr(
        parallel_generation_module.simd_ops,
        "postprocess_and_score",
        lambda logits, *, token_id, **kwargs: (  # noqa: ARG005
            int(token_id),
            0.95 if int(token_id) == 3 else 0.9,
        ),
    )

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1, 2, 3],
        max_new_tokens=8,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.MEDUSA_HEAD
    assert plan.accepted_prefix_tokens == [3, 4]
    assert plan.evidence.benchmark_label == "medusa_head_candidate"
    assert plan.evidence.proposed_parallel_tokens == 2
    assert plan.evidence.draft_source == "medusa_head_native"


def test_parallel_planner_selects_hydra_head_candidate(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_enabled = False
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = False
    engine._medusa_head_enabled = False
    engine._hydra_head_enabled = True
    engine._hydra_head_max_draft_tokens = 4
    engine._hydra_head_top_k = 8
    engine._hydra_head_acceptance_floor = 0.01
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._scheduler_kv_fragmentation_ratio = 0.0
    engine._ssm_spec_decoder = None
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine.token_eos = lambda: 99
    engine._draft_model_head_candidates = lambda **kwargs: [3]  # noqa: ARG005

    def _logits(_tokens):
        logits = np.full((6,), -10.0, dtype=np.float32)
        logits[3] = 10.0
        return logits

    engine._get_logits_for_tokens = _logits
    engine._apply_logits_processors = NativeQSGEngine._apply_logits_processors
    engine._should_parallel_decode = lambda *args, **kwargs: True  # noqa: ARG005
    monkeypatch.setattr(
        parallel_generation_module.simd_ops,
        "postprocess_and_score",
        lambda logits, *, token_id, **kwargs: (  # noqa: ARG005
            int(token_id),
            0.95,
        ),
    )

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1, 2, 3],
        max_new_tokens=8,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.HYDRA_HEAD
    assert plan.accepted_prefix_tokens == [3]
    assert plan.evidence.benchmark_label == "hydra_head_candidate"
    assert plan.evidence.proposed_parallel_tokens == 1
    assert plan.evidence.draft_source == "hydra_head_native"


def test_parallel_planner_preserves_rejected_medusa_bundle_evidence(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_enabled = False
    engine._parallel_prompt_lookup_enabled = False
    engine._parallel_jacobi_lookahead_enabled = False
    engine._parallel_replacement_enabled = False
    engine._medusa_head_enabled = True
    engine._hydra_head_enabled = False
    engine._medusa_head_max_draft_tokens = 4
    engine._medusa_head_top_k = 8
    engine._medusa_head_acceptance_floor = 0.5
    engine._parallel_ssd_bridge_enabled = False
    engine._parallel_ar_recovery_enabled = True
    engine._parallel_prompt_lookup_min_ngram = 2
    engine._parallel_prompt_lookup_max_ngram = 8
    engine._parallel_prompt_lookup_accept_prob = 0.1
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._scheduler_kv_fragmentation_ratio = 0.0
    engine._ssm_spec_decoder = None
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine.token_eos = lambda: 99
    engine._draft_model_head_bundle = (
        lambda **kwargs: DraftCandidateBundle(  # noqa: ARG005
            tokens=[3, 4],
            probabilities=[0.81, 0.63],
            source="medusa_head_native",
        )
    )

    def _logits(_tokens):
        logits = np.full((6,), -10.0, dtype=np.float32)
        logits[3] = 10.0
        return logits

    engine._get_logits_for_tokens = _logits
    engine._apply_logits_processors = NativeQSGEngine._apply_logits_processors
    engine._should_parallel_decode = lambda *args, **kwargs: True  # noqa: ARG005
    monkeypatch.setattr(
        parallel_generation_module.simd_ops,
        "postprocess_and_score",
        lambda logits, *, token_id, **kwargs: (  # noqa: ARG005
            int(token_id) + 1,
            0.0,
        ),
    )

    planner = ParallelDecodePlanner(engine)
    plan = planner.plan(
        prompt_tokens=[1, 2, 3],
        max_new_tokens=8,
        temperature=0.7,
    )

    assert plan.mode == GenerationMode.PARALLEL_HYBRID
    assert plan.evidence.benchmark_label == "parallel_hybrid"
    assert plan.evidence.accepted_parallel_tokens == 0
    assert plan.evidence.rejected_parallel_tokens == 2
    assert plan.evidence.proposed_parallel_tokens == 2
    assert plan.evidence.draft_frontier_width == 2
    assert plan.evidence.draft_confidence_mean == pytest.approx(0.72)
    assert plan.evidence.draft_confidence_min == pytest.approx(0.63)
    assert plan.evidence.draft_source == "medusa_head_native"


def test_draft_model_head_bundle_preserves_native_probabilities(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(vocab_size=6)
    engine._medusa_head_ready = True
    engine._hydra_head_ready = False
    engine._medusa_head_top_k = 8
    engine._medusa_head_acceptance_floor = 0.2
    engine._medusa_head_config = SimpleNamespace(
        num_heads=2,
        hidden_dim=4,
        vocab_size=6,
        weights=np.ones((2 * 6 * 4,), dtype=np.float32),
        bias=np.zeros((2 * 6,), dtype=np.float32),
    )
    engine._get_hidden_and_logits_for_tokens = lambda tokens: (  # noqa: ARG005
        np.ones((4,), dtype=np.float32),
        np.zeros((6,), dtype=np.float32),
    )

    monkeypatch.setattr(
        native_qsg_engine_module,
        "qsg_medusa_head_draft",
        lambda *args, **kwargs: ([2, 3], [0.9, 0.8]),  # noqa: ARG005
    )

    bundle = engine._draft_model_head_bundle(
        head_type="medusa",
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        temperature=0.7,
        top_k=8,
    )

    assert bundle.tokens == [2, 3]
    assert bundle.probabilities == [0.9, 0.8]
    assert bundle.source == "medusa_head_native"


def test_draft_model_head_candidates_calls_native_medusa(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(vocab_size=6)
    engine._medusa_head_ready = True
    engine._hydra_head_ready = False
    engine._medusa_head_top_k = 8
    engine._medusa_head_acceptance_floor = 0.2
    engine._medusa_head_config = SimpleNamespace(
        num_heads=2,
        hidden_dim=4,
        vocab_size=6,
        weights=np.ones((2 * 6 * 4,), dtype=np.float32),
        bias=np.zeros((2 * 6,), dtype=np.float32),
    )
    engine._get_hidden_and_logits_for_tokens = lambda tokens: (  # noqa: ARG005
        np.ones((4,), dtype=np.float32),
        np.zeros((6,), dtype=np.float32),
    )

    monkeypatch.setattr(
        native_qsg_engine_module,
        "qsg_medusa_head_draft",
        lambda *args, **kwargs: ([2, 3], [0.9, 0.8]),  # noqa: ARG005
    )

    drafted = engine._draft_model_head_candidates(
        head_type="medusa",
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        temperature=0.7,
        top_k=8,
    )

    assert drafted == [2, 3]


def test_draft_model_head_candidates_calls_native_hydra(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(vocab_size=6)
    engine._medusa_head_ready = False
    engine._hydra_head_ready = True
    engine._hydra_head_top_k = 8
    engine._hydra_head_acceptance_floor = 0.22
    engine._hydra_head_blend_alpha = 0.55
    engine._hydra_head_config = SimpleNamespace(
        num_heads=2,
        hidden_dim=4,
        vocab_size=6,
        weights=np.ones((2 * 6 * 4,), dtype=np.float32),
        bias=np.zeros((2 * 6,), dtype=np.float32),
    )
    engine._get_hidden_and_logits_for_tokens = lambda tokens: (  # noqa: ARG005
        np.ones((4,), dtype=np.float32),
        np.zeros((6,), dtype=np.float32),
    )

    monkeypatch.setattr(
        native_qsg_engine_module,
        "qsg_hydra_head_draft",
        lambda *args, **kwargs: ([4], [0.91]),  # noqa: ARG005
    )

    drafted = engine._draft_model_head_candidates(
        head_type="hydra",
        prompt_tokens=[1, 2],
        max_new_tokens=4,
        temperature=0.7,
        top_k=8,
    )

    assert drafted == [4]


def test_apply_generation_evidence_propagates_benchmark_and_scheduler_fields():
    telemetry = NativeGenerationTelemetry(
        prompt_cache_hits=4,
        prompt_cache_misses=1,
        kv_fragmentation_ratio=0.1,
    )
    evidence = GenerationEvidence(
        generation_mode=GenerationMode.AR_RECOVERY.value,
        benchmark_label="",
        accepted_parallel_tokens=2,
        rejected_parallel_tokens=1,
        proposed_parallel_tokens=3,
        jacobi_frontier_width=3,
        jacobi_branch_survival_rate=0.67,
        jacobi_verify_cost_ms=1.8,
        jacobi_branch_entropy=0.42,
        draft_confidence_mean=0.75,
        draft_confidence_min=0.55,
        draft_source="medusa_head_native",
        scheduler_queue_wait_ms=1.25,
        scheduler_iteration_ms=2.75,
        kv_fragmentation_ratio=0.3,
    )

    updated = NativeQSGEngine._apply_generation_evidence(telemetry, evidence)

    assert updated.generation_mode == GenerationMode.AR_RECOVERY.value
    assert updated.benchmark_label == "ar_baseline"
    assert updated.prefix_cache_hit_rate == 0.8
    assert updated.proposed_parallel_tokens == 3
    assert updated.jacobi_frontier_width == 3
    assert updated.jacobi_branch_survival_rate == pytest.approx(0.67)
    assert updated.jacobi_verify_cost_ms == pytest.approx(1.8)
    assert updated.jacobi_branch_entropy == pytest.approx(0.42)
    assert updated.draft_confidence_mean == pytest.approx(0.75)
    assert updated.draft_confidence_min == pytest.approx(0.55)
    assert updated.draft_source == "medusa_head_native"
    assert updated.scheduler_queue_wait_ms == 1.25
    assert updated.scheduler_iteration_ms == 2.75
    assert updated.kv_fragmentation_ratio == 0.3


def test_annotate_telemetry_propagates_prompt_category_and_temperature_band():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.contract = {"model": "qwen3.5:9b"}
    engine.profile = SimpleNamespace(model_name="qwen3.5:9b")
    engine._refresh_os_thread_telemetry = lambda: None
    engine._granite_moe_mode = "dense"
    engine._active_thread_mode = "decode"
    engine._prefill_chunk_count = 0
    engine._runtime_thread_switches = 0
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._prompt_cache_reused_tokens = 0
    engine._scheduler_queue_wait_ms = 0.0
    engine._scheduler_iteration_ms = 0.0
    engine._last_prompt_format_seconds = 0.0
    engine._last_tokenize_seconds = 0.0
    engine._last_embedding_lookup_seconds = 0.0
    engine._last_graph_prefill_seconds = 0.0
    engine._last_graph_decode_seconds = 0.0
    engine._last_sample_seconds = 0.0
    engine._last_logits_processor_seconds = 0.0
    engine._last_penalty_seconds = 0.0
    engine._last_suppression_seconds = 0.0
    engine._last_graph_prefill_calls = 0
    engine._last_graph_decode_calls = 0
    engine._last_sample_calls = 0
    engine._last_logits_processor_calls = 0
    engine._last_penalty_calls = 0
    engine._last_suppression_calls = 0
    engine._tc_enabled = False
    engine._tc_graph_drift_available = False
    engine._tc_last_snapshot_valid = False
    engine._tc_last_snapshot = {}
    engine._tc_stabilizer_seconds_total = 0.0
    engine._tc_stabilizer_calls_total = 0
    engine._tc_auto_downgrade_events = 0
    engine._tc_last_overhead_percent = 0.0
    engine._native_qsg_use_coconut = False
    engine._native_qsg_coconut_paths = 0
    engine._native_qsg_coconut_alpha = 0.0
    engine._native_qsg_use_grover = False
    engine._native_qsg_grover_top_k = 0
    engine._native_qsg_grover_damping = 0.0
    engine._strict_cpp_only = False
    engine._native_fast_path = True
    engine._parallel_decode_disable_reason = ""
    engine._last_speculative_accept_count = 0
    engine._last_speculative_reject_count = 0
    engine._last_self_spec_native_path = True
    engine._last_self_spec_policy = "heuristic"
    engine._last_self_spec_exit_layer = 12
    engine._last_self_spec_exit_fraction = 0.5
    engine._last_self_spec_draft_tokens = 6
    engine._native_library_info = {}
    engine._affinity_plan_json = ""
    engine._topology_json = ""
    engine._logical_core_count = 8
    engine._physical_core_count = 4
    engine._p_core_count = 4
    engine._affinity_policy = "close"
    engine._affinity_mode = 1
    engine._l3_domain_count = 1
    engine._numa_strict = False
    engine._numa_affinity_mode = "legacy"
    engine._numa_hugepage = "off"
    engine._numa_bind_policy = "none"
    engine._numa_first_touch = False
    engine._os_thread_migrations = 0
    engine._os_last_cpu = 0
    engine._omp_places = ""
    engine._omp_proc_bind = ""
    engine._omp_max_threads = 8
    engine._omp_dynamic = False
    engine._omp_active_levels = 1
    engine._perf_event_access = False
    engine._perf_event_access_reason = "denied"
    engine._cpu_governor = "performance"
    engine._thp_mode = "always"
    engine._perf_counter_source = "telemetry_only"
    engine._autotune_profile_id = ""
    engine._autotune_source = "heuristic"
    engine._autotune_score = 0.0
    engine._autotune_exploration_count = 0
    engine._last_prompt_category = "code"
    engine._last_temperature_band = "low"

    telemetry = NativeGenerationTelemetry(
        generation_mode=GenerationMode.PROMPT_LOOKUP.value
    )
    updated = NativeQSGEngine._annotate_telemetry(engine, telemetry)

    assert updated.prompt_category == "code"
    assert updated.temperature_band == "low"
    assert updated.benchmark_label == "prompt_lookup"
    assert updated.self_spec_native_path is True
    assert updated.self_spec_policy == "heuristic"
    assert updated.self_spec_exit_layer == 12
    assert updated.self_spec_exit_fraction == pytest.approx(0.5)
    assert updated.self_spec_draft_tokens == 6


def test_generate_ssd_bridge_prefers_native_exit_continuation_path(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)

    class _GraphStub:
        def reset(self):
            return True

        @staticmethod
        def forward_token_id_to_exit(token_id, exit_layer, position):
            return np.asarray(
                [float(token_id), float(exit_layer), float(position), 1.0],
                dtype=np.float32,
            )

        @staticmethod
        def continue_from_hidden(hidden, *, start_layer, position):
            logits = np.full((6,), -10.0, dtype=np.float32)
            logits[4] = 7.0 + float(position) + float(start_layer) * 0.01
            return logits

        @staticmethod
        def forward_head(hidden):
            _ = hidden
            logits = np.full((6,), -10.0, dtype=np.float32)
            logits[4] = 9.0
            return logits

    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "qsg_postprocess_and_score",
        lambda logits, *args, token_id, **kwargs: (  # noqa: ARG005
            int(token_id),
            0.9 if float(np.max(logits)) > 8.0 else 0.95,
        ),
    )

    engine._model_graph = _GraphStub()
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._self_spec_native_supported = True
    engine._self_spec_force_exit_layer = 3
    engine._self_spec_exit_layer_min = 2
    engine._self_spec_exit_layer_max = 6
    engine._self_spec_acceptance_threshold = 0.7
    engine._self_spec_max_draft_length = 4
    engine._prefix_logits_cache = {}
    engine._cached_logits_tokens = []
    engine._cached_logits = None
    engine._last_speculative_accept_count = 0
    engine._last_speculative_reject_count = 0
    engine._last_temperature_band = "low"
    engine._last_prompt_category = "code"
    engine._reset_generation_counters = lambda: None
    engine._clear_logits_cache = lambda: None
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry  # noqa: ARG005
    engine._sample = lambda logits, *args, **kwargs: int(
        np.argmax(np.asarray(logits))
    )  # noqa: ARG005
    engine.token_eos = lambda: 5
    engine.n_layer = 8

    output = engine._generate_ssd_bridge(
        prompt_tokens=[1],
        max_new_tokens=2,
        temperature=0.7,
        logits_processor=None,
        generation_started=0.0,
    )

    assert output == [1, 4, 4]
    assert engine._last_self_spec_native_path is True
    assert engine._last_self_spec_exit_layer == 3
    assert engine._last_self_spec_policy == "forced"
    assert engine._last_self_spec_draft_tokens == 2
    assert engine._last_speculative_accept_count == 2
    assert engine._last_speculative_reject_count == 0


def test_generate_routes_ssd_bridge_through_native_self_spec_without_python_decoder():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    bridge_calls: list[dict[str, object]] = []

    engine._parallel_decode_disable_reason = ""
    engine._last_temperature_band = ""
    engine._parallel_planner = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            mode=GenerationMode.SSD_BRIDGE,
            accepted_prefix_tokens=[],
            evidence=GenerationEvidence(
                generation_mode=GenerationMode.SSD_BRIDGE.value,
                benchmark_label="ssd_bridge",
            ),
        )
    )
    engine._supports_native_self_spec = lambda: True
    engine._ssm_spec_decoder = None
    engine._generate_ssd_bridge = lambda **kwargs: bridge_calls.append(kwargs) or [1, 7]
    engine._apply_generation_evidence = (
        lambda telemetry, evidence: telemetry
    )  # noqa: ARG005
    engine._generate_autoregressive = Mock(
        side_effect=AssertionError("autoregressive fallback should not run")
    )
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry  # noqa: ARG005
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._last_generation = SimpleNamespace()
    engine.token_eos = lambda: 99

    output = engine.generate(
        prompt_tokens=[1],
        max_new_tokens=1,
        temperature=0.7,
    )

    assert output == [1, 7]
    assert len(bridge_calls) == 1
    assert bridge_calls[0]["prompt_tokens"] == [1]
    assert bridge_calls[0]["max_new_tokens"] == 1


def test_generate_raises_when_strict_non_ar_mode_would_fallback_to_autoregressive():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._parallel_decode_disable_reason = ""
    engine._last_temperature_band = ""
    engine._forbid_autoregressive_fallback = True
    engine._parallel_planner = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            mode=GenerationMode.PROMPT_LOOKUP,
            accepted_prefix_tokens=[],
            evidence=GenerationEvidence(
                generation_mode=GenerationMode.PROMPT_LOOKUP.value,
                benchmark_label="prompt_lookup",
            ),
        )
    )
    engine._block_diffusion_native_ready = False
    engine._supports_native_self_spec = lambda: False
    engine._ssm_spec_decoder = None
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._generate_autoregressive = Mock(
        side_effect=AssertionError("autoregressive fallback should not run")
    )
    engine.token_eos = lambda: 99

    with pytest.raises(RuntimeError, match="forbids autoregressive fallback"):
        engine.generate(
            prompt_tokens=[1, 2, 3],
            max_new_tokens=4,
            temperature=0.7,
        )


def test_get_logits_full_graph_multi_token_prefers_batch_token_id_api():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    graph_calls: list[tuple[list[int], int]] = []

    def _forward_token_ids(token_ids, start_pos):
        graph_calls.append((list(token_ids), int(start_pos)))
        final_pos = int(start_pos) + len(token_ids) - 1
        return [float(final_pos), 33.0]

    token_ids = [11, 22, 33]
    start_pos = 5

    engine._model_graph = SimpleNamespace(
        has_full_graph=True,
        has_hybrid_mode=False,
        forward_token_ids=Mock(side_effect=_forward_token_ids),
        forward_token_id=Mock(
            side_effect=AssertionError("single-token path should not run")
        ),
    )
    engine.forward_pass = SimpleNamespace(
        forward=Mock(side_effect=AssertionError("fallback should not run"))
    )
    engine.architecture = "qwen35"
    engine.n_layer = 1
    engine._hybrid_mode = False

    logits = engine._get_logits(token_ids, start_pos=start_pos)

    assert engine.forward_pass.forward.call_count == 0
    assert graph_calls == [([11, 22, 33], 5)]
    assert logits == [7.0, 33.0]


def test_get_logits_switches_runtime_thread_mode_by_token_count():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    modes: list[str] = []
    engine._set_runtime_thread_mode = lambda mode: modes.append(mode)
    engine._model_graph = SimpleNamespace(
        has_full_graph=True,
        has_hybrid_mode=False,
        forward_token_id=lambda token_id, position: [float(position)],  # noqa: ARG005
    )
    engine.architecture = "qwen35"
    engine.n_layer = 1
    engine._hybrid_mode = False

    _ = engine._get_logits([11, 12], start_pos=0)
    _ = engine._get_logits([13], start_pos=2)

    assert modes == ["batch", "decode"]


def test_get_logits_pulls_timecrystal_snapshot_and_auto_downgrades_mode():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    mode_updates: list[dict[str, int | float]] = []
    snapshot_state = {"seconds": 0.0, "calls": 0}

    def _forward_token_id(token_id, position):  # noqa: ARG001
        return np.asarray([float(position), 1.0], dtype=np.float32)

    def _get_snapshot():
        snapshot_state["seconds"] += 0.05
        snapshot_state["calls"] += 1
        return {
            "latest_drift": 0.6,
            "mean_drift": 0.4,
            "max_drift": 0.8,
            "decay_ratio": 0.9,
            "active_token_count": 16384,
            "damped_block_count": 5,
            "pruned_block_count": 1,
            "stabilizer_seconds": snapshot_state["seconds"],
            "stabilizer_calls": snapshot_state["calls"],
            "mode": 2,
        }

    engine._model_graph = SimpleNamespace(
        has_full_graph=True,
        has_hybrid_mode=False,
        forward_token_id=_forward_token_id,
        get_last_drift_snapshot=_get_snapshot,
        set_drift_config=lambda config: mode_updates.append(dict(config)) or True,
    )
    engine._set_runtime_thread_mode = lambda mode: None  # noqa: ARG005
    engine._graph_token_id_enabled = True
    engine._graph_batch_token_id_enabled = True
    engine._native_fast_path = False
    engine._disable_logits_processors = False
    engine._disable_token_penalties = False
    engine.architecture = "qwen35"
    engine.n_layer = 1
    engine._hybrid_mode = False
    engine._last_graph_prefill_seconds = 0.0
    engine._last_graph_decode_seconds = 0.0
    engine._last_graph_prefill_calls = 0
    engine._last_graph_decode_calls = 0
    engine._tc_enabled = True
    engine._tc_base_mode_native = 2
    engine._tc_current_mode_native = 2
    engine._tc_current_mode = "aggressive"
    engine._tc_graph_drift_available = True
    engine._tc_control_interval_tokens = 4
    engine._tc_overhead_window = 4
    engine._tc_recovery_interval_tokens = 8
    engine._tc_overhead_target_pct = 15.0
    engine._tc_overhead_max_pct = 20.0
    engine._tc_block_size_tokens = 128
    engine._tc_update_interval_tokens = 64
    engine._tc_prune_interval_tokens = 128
    engine._tc_preserve_head_tokens = 256
    engine._tc_preserve_recent_tokens = 8192
    engine._tc_min_active_tokens = 16384
    engine._tc_damp_threshold = 0.35
    engine._tc_prune_threshold = 0.72
    engine._tc_damping_strength = 1.2
    engine._tc_hysteresis = 0.05
    engine._tc_decode_steps = 0
    engine._tc_recovery_steps = 0
    engine._tc_overhead_samples = []
    engine._tc_last_overhead_percent = 0.0
    engine._tc_prev_stabilizer_seconds = 0.0
    engine._tc_prev_stabilizer_calls = 0
    engine._tc_stabilizer_seconds_total = 0.0
    engine._tc_stabilizer_calls_total = 0
    engine._tc_last_snapshot = {}
    engine._tc_last_snapshot_valid = False
    engine._tc_auto_downgrade_events = 0

    for position in range(4):
        _ = engine._get_logits([11], start_pos=position)

    assert engine._tc_last_snapshot_valid is True
    assert engine._tc_last_snapshot["stabilizer_calls"] == 4
    assert engine._tc_current_mode_native == 1
    assert engine._tc_auto_downgrade_events == 1
    assert mode_updates
    assert mode_updates[-1]["mode"] == 1


def test_get_logits_raises_when_token_id_graph_path_unavailable():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._set_runtime_thread_mode = lambda mode: None  # noqa: ARG005
    engine._graph_token_id_enabled = True
    engine._graph_batch_token_id_enabled = True
    engine._last_graph_prefill_seconds = 0.0
    engine._last_graph_decode_seconds = 0.0

    engine._model_graph = SimpleNamespace(
        has_full_graph=True,
        has_hybrid_mode=False,
        forward_token_ids=lambda token_ids, start_pos: None,  # noqa: ARG005
        forward_token_id=lambda token_id, position: None,  # noqa: ARG005
    )
    engine.architecture = "qwen35"
    engine.n_layer = 1
    engine._hybrid_mode = False

    with pytest.raises(RuntimeError, match="fallback is disabled"):
        engine._get_logits([11, 12], start_pos=4)

    assert engine._graph_token_id_enabled is False
    assert engine._graph_batch_token_id_enabled is False


def test_get_logits_raises_when_graph_unavailable():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._set_runtime_thread_mode = lambda mode: None  # noqa: ARG005
    engine._model_graph = None
    engine.architecture = "qwen35"
    engine.profile = SimpleNamespace(vocab_size=8)
    engine._hybrid_mode = False

    with np.testing.assert_raises(RuntimeError):
        engine._get_logits([11], start_pos=0)


def test_format_prompt_uses_shared_template_for_granite():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(chat_template="granite")
    engine.contract = {"model": "granite4:tiny-h"}
    engine._strict_prompt_contract = SimpleNamespace(
        template_name="granite",
        system_prompt=None,
        assistant_prefix=None,
    )

    prompt = engine.format_prompt("Hello")

    assert "<|start_of_role|>system<|end_of_role|>" in prompt
    assert "professional, accurate, and safe." in prompt
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_generate_prefill_honors_ubatch_chunking():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    calls: list[tuple[list[int], int]] = []
    engine.num_ubatch = 2
    engine.profile = SimpleNamespace(
        vocab_size=8, chat_template="chatml", model_name="qwen3.5:9b"
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._granite_moe_mode = "not_applicable"
    engine._ssm_spec_decoder = None
    engine._model_graph = None
    engine.forward_pass = SimpleNamespace(reset=lambda: None)
    engine._cached_logits_tokens = []
    engine._cached_logits = None
    engine._prefix_logits_cache = {}
    engine._prefix_cache_hits = 0
    engine._prefix_cache_misses = 0
    engine._prompt_cache_reused_tokens = 0
    engine._prefill_chunk_count = 0
    engine._active_thread_mode = "batch"
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._get_logits = lambda tokens, start_pos=0: (  # noqa: ARG005
        calls.append((list(tokens), int(start_pos))) or np.zeros((8,), dtype=np.float32)
    )
    engine.token_eos = lambda: 2

    out = engine.generate([1, 2, 3, 4, 5], max_new_tokens=0, temperature=0.8)

    assert out == [1, 2, 3, 4, 5]
    assert calls == [([1, 2], 0), ([3, 4], 2), ([5], 4)]
    metrics = engine.get_last_run_metrics()
    assert metrics["template_name"] == "chatml"
    assert metrics["prefill_chunk_count"] == 3
    assert metrics["active_thread_mode"] == "batch"


def test_generate_resets_graph_perf_stats_after_prefill():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    reset_events: list[str] = []
    engine._ssm_spec_decoder = None
    engine._model_graph = SimpleNamespace(
        reset=lambda: reset_events.append("reset"),
        reset_perf_stats=lambda: reset_events.append("perf") or True,
    )
    engine._clear_logits_cache = lambda: None
    engine._reset_generation_counters = lambda: None
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._parallel_decode_disable_reason = ""
    engine.num_ubatch = 1
    engine.profile = SimpleNamespace(
        vocab_size=8, chat_template="chatml", model_name="qwen3.5:9b"
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._granite_moe_mode = "not_applicable"
    engine._native_fast_path = False
    engine._disable_logits_processors = False
    engine._disable_token_penalties = False
    engine._get_logits = lambda tokens, start_pos=0: np.zeros(
        (8,), dtype=np.float32
    )  # noqa: ARG005
    engine.token_eos = lambda: 2
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry

    out = engine.generate([1, 2, 3], max_new_tokens=0, temperature=0.8)

    assert out == [1, 2, 3]
    assert reset_events == ["reset", "perf"]


def test_generate_stream_resets_graph_perf_stats_after_prefill():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    reset_events: list[str] = []
    engine._ssm_spec_decoder = None
    engine._model_graph = SimpleNamespace(
        reset=lambda: reset_events.append("reset"),
        reset_perf_stats=lambda: reset_events.append("perf") or True,
    )
    engine._clear_logits_cache = lambda: None
    engine._reset_generation_counters = lambda: None
    engine._parallel_decode_disable_reason = ""
    engine.num_ubatch = 1
    engine.profile = SimpleNamespace(
        vocab_size=8, chat_template="chatml", model_name="qwen3.5:9b"
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._granite_moe_mode = "not_applicable"
    engine._native_fast_path = False
    engine._disable_logits_processors = False
    engine._disable_token_penalties = False
    engine._get_logits = lambda tokens, start_pos=0: np.zeros(
        (8,), dtype=np.float32
    )  # noqa: ARG005
    engine.token_eos = lambda: 2
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry

    assert (
        list(engine.generate_stream([1, 2, 3], max_new_tokens=0, temperature=0.8)) == []
    )
    assert reset_events == ["reset", "perf"]


def test_generate_autoregressive_marks_python_hot_path_loop():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._model_graph = None
    engine._clear_logits_cache = lambda: None
    engine._reset_generation_counters = lambda: None
    engine._reset_graph_perf_stats_for_decode_window = lambda: None
    engine._sample_os_thread_cpu = lambda: None
    engine._should_block_eos = lambda generated_tokens: False  # noqa: ARG005
    engine._get_logits = lambda tokens, start_pos=0: np.array(  # noqa: ARG005
        [0.0, 0.0, 9.0, -1.0], dtype=np.float32
    )
    engine._sample = lambda *args, **kwargs: 2
    engine.profile = SimpleNamespace(vocab_size=4)
    engine.num_ubatch = 1
    engine.token_eos = lambda: 7

    output, telemetry = engine._generate_autoregressive(
        prompt_tokens=[1, 2],
        max_new_tokens=1,
        generation_started=0.0,
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logits_processor=None,
    )

    assert output == [1, 2, 2]
    assert telemetry.generated_tokens == 1
    assert telemetry.python_hot_path_calls == 1


def test_generate_autoregressive_uses_native_runtime_when_available():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine.profile = SimpleNamespace(vocab_size=32)
    engine.token_eos = lambda: 99

    class _Runtime:
        def __init__(self):
            self._events = [
                SimpleNamespace(token_id=4, done=False, error=None),
                SimpleNamespace(token_id=None, done=True, error=None),
            ]

        def submit(self, request_id, **kwargs):
            del request_id, kwargs

        def poll(self, request_id):
            del request_id
            if not self._events:
                return None
            return self._events.pop(0)

        def metrics(self):
            return SimpleNamespace(
                prefill_batches=1,
                runtime_prefill_tokens=2,
                runtime_decode_steps=1,
            )

        def close(self):
            return None

    engine._build_decode_runtime = lambda: _Runtime()

    output, telemetry = engine._generate_autoregressive(
        prompt_tokens=[1, 2],
        max_new_tokens=1,
        generation_started=time.perf_counter(),
        temperature=0.8,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logits_processor=None,
    )

    assert output == [1, 2, 4]
    assert telemetry.generated_tokens == 1
    assert telemetry.python_hot_path_calls == 0
    assert telemetry.native_fast_path is True


def test_generate_block_diffusion_verifies_tokens_with_native_score(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._block_diffusion_native_ready = True
    engine._block_diffusion_block_size_tokens = 2
    engine._block_diffusion_denoise_iterations = 1
    engine._block_diffusion_acceptance_floor = 0.2
    engine._model_graph = SimpleNamespace(_handle=1)
    engine.profile = SimpleNamespace(vocab_size=4)
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._should_block_eos = lambda generated_tokens: False  # noqa: ARG005
    engine._sample = lambda *args, **kwargs: 1  # noqa: ARG005
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine._get_logits_for_tokens = lambda _tokens: np.asarray(  # noqa: ARG005
        [0.0, 0.0, 0.0, 5.0],
        dtype=np.float32,
    )
    engine.token_eos = lambda: 99

    score_calls: list[int] = []
    monkeypatch.setattr(
        native_qsg_engine_module,
        "qsg_block_diffusion_draft",
        lambda *args, **kwargs: ([3], [0.95]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "_native_verify_draft_tokens",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            accepted_count=1,
            probabilities=[0.95],
            recovery_token=None,
        ),
    )

    evidence = GenerationEvidence()
    out = engine._generate_block_diffusion(
        prompt_tokens=[1, 2],
        max_new_tokens=1,
        temperature=0.7,
        top_p=1.0,
        top_k=8,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logits_processor=None,
        generation_started=0.0,
        evidence=evidence,
    )

    assert out == [1, 2, 3]
    assert score_calls == []
    assert engine._last_generation.python_hot_path_calls == 0
    assert engine._last_generation.numpy_hot_path_calls == 0
    assert engine._last_generation.draft_source == "block_diffusion_native"
    assert engine._last_generation.blockwise_blocks == 1
    assert engine._last_generation.blockwise_denoise_steps == 1
    assert engine._last_generation.blockwise_convergence_rate == pytest.approx(1.0)


def test_generate_model_head_verifies_tokens_with_native_score(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._medusa_head_ready = True
    engine._hydra_head_ready = False
    engine._medusa_head_max_draft_tokens = 2
    engine._medusa_head_acceptance_floor = 0.2
    engine._model_graph = SimpleNamespace(_handle=1)
    engine.profile = SimpleNamespace(vocab_size=4)
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._should_block_eos = lambda generated_tokens: False  # noqa: ARG005
    engine._sample = lambda *args, **kwargs: 1  # noqa: ARG005
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine._draft_model_head_candidates = lambda **kwargs: [3]  # noqa: ARG005
    engine._get_logits_for_tokens = lambda _tokens: np.asarray(  # noqa: ARG005
        [0.0, 0.0, 0.0, 5.0],
        dtype=np.float32,
    )
    engine.token_eos = lambda: 99

    score_calls: list[int] = []
    monkeypatch.setattr(
        native_qsg_engine_module,
        "_native_verify_draft_tokens",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            accepted_count=1,
            probabilities=[0.95],
            recovery_token=None,
        ),
    )

    evidence = GenerationEvidence()
    out = engine._generate_model_head(
        head_type="medusa",
        prompt_tokens=[1, 2],
        max_new_tokens=1,
        temperature=0.7,
        top_p=1.0,
        top_k=8,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logits_processor=None,
        generation_started=0.0,
        evidence=evidence,
    )

    assert out == [1, 2, 3]
    assert score_calls == []
    assert engine._last_generation.python_hot_path_calls == 0
    assert engine._last_generation.numpy_hot_path_calls == 0


def test_generate_masked_diffusion_uses_native_verifier(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._masked_diffusion_native_ready = True
    engine._masked_diffusion_block_size_tokens = 2
    engine._masked_diffusion_denoise_iterations = 1
    engine._masked_diffusion_mask_stride = 2
    engine._masked_diffusion_acceptance_floor = 0.2
    engine._model_graph = SimpleNamespace(_handle=1)
    engine.profile = SimpleNamespace(vocab_size=4)
    engine._native_fast_path = True
    engine._disable_logits_processors = True
    engine._should_block_eos = lambda generated_tokens: False  # noqa: ARG005
    engine._sample = lambda *args, **kwargs: 1  # noqa: ARG005
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine._get_logits_for_tokens = lambda _tokens: np.asarray(  # noqa: ARG005
        [0.0, 0.0, 0.0, 5.0],
        dtype=np.float32,
    )
    engine.token_eos = lambda: 99

    monkeypatch.setattr(
        native_qsg_engine_module,
        "qsg_masked_diffusion_draft",
        lambda *args, **kwargs: ([3], [0.95], [0]),  # noqa: ARG005
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "_native_verify_draft_tokens",
        lambda *args, **kwargs: SimpleNamespace(  # noqa: ARG005
            accepted_count=1,
            probabilities=[0.95],
            recovery_token=None,
        ),
    )

    evidence = GenerationEvidence()
    out = engine._generate_masked_diffusion(
        prompt_tokens=[1, 2],
        max_new_tokens=1,
        temperature=0.7,
        top_p=1.0,
        top_k=8,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        logits_processor=None,
        generation_started=0.0,
        evidence=evidence,
    )

    assert out == [1, 2, 3]
    assert engine._last_generation.python_hot_path_calls == 0
    assert engine._last_generation.masked_generation_ready is True
    assert engine._last_generation.masked_generation_steps == 1
    assert engine._last_generation.masked_generation_proposed_tokens == 1
    assert engine._last_generation.masked_generation_accepted_tokens == 1


def test_generate_stream_marks_python_hot_path_loop():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._ssm_spec_decoder = None
    engine._model_graph = None
    engine._clear_logits_cache = lambda: None
    engine._reset_generation_counters = lambda: None
    engine._reset_graph_perf_stats_for_decode_window = lambda: None
    engine._sample_os_thread_cpu = lambda: None
    engine._should_block_eos = lambda generated_tokens: False  # noqa: ARG005
    engine._parallel_decode_disable_reason = ""
    engine._native_fast_path = False
    engine._disable_logits_processors = False
    engine._get_logits = lambda tokens, start_pos=0: np.array(  # noqa: ARG005
        [0.0, 0.0, 9.0, -1.0], dtype=np.float32
    )
    engine._sample = lambda *args, **kwargs: 2
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine.profile = SimpleNamespace(vocab_size=4)
    engine.num_ubatch = 1
    engine.token_eos = lambda: 7

    assert list(engine.generate_stream([1, 2], max_new_tokens=1, temperature=0.8)) == [
        2
    ]
    assert engine._last_generation.generated_tokens == 1
    assert engine._last_generation.python_hot_path_calls == 1


def test_generate_seeds_native_sampler_when_requested(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._ssm_spec_decoder = None
    engine._model_graph = None
    engine._native_fast_path = False
    engine._disable_logits_processors = False
    engine._disable_token_penalties = False
    engine._force_parallel_decode = False
    engine._parallel_decode_min_new_tokens = 32
    engine._parallel_decode_min_prompt_tokens = 64
    engine._granite_moe_mode = "not_applicable"
    engine._prefill_chunk_count = 0
    engine._active_thread_mode = "decode"
    engine._last_embedding_lookup_seconds = 0.0
    engine._last_graph_prefill_seconds = 0.0
    engine._last_graph_decode_seconds = 0.0
    engine._last_sample_seconds = 0.0
    engine._last_logits_processor_seconds = 0.0
    engine._last_penalty_seconds = 0.0
    engine._last_suppression_seconds = 0.0
    engine._last_graph_prefill_calls = 0
    engine._last_graph_decode_calls = 0
    engine._last_sample_calls = 0
    engine._last_logits_processor_calls = 0
    engine._last_penalty_calls = 0
    engine._last_suppression_calls = 0
    engine._prompt_cache_reused_tokens = 0
    engine._profile = None
    engine.profile = SimpleNamespace(
        vocab_size=8, chat_template="chatml", model_name="qwen3.5:9b"
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine.forward_pass = SimpleNamespace(reset=lambda: None)
    engine._clear_logits_cache = lambda: None
    engine._set_runtime_thread_mode = lambda mode: None  # noqa: ARG005
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._get_logits = lambda tokens, start_pos=0: np.array(  # noqa: ARG005
        [0.0, 0.0, 9.0, -1.0], dtype=np.float32
    )
    engine.token_eos = lambda: 7
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry

    seeded = []
    monkeypatch.setattr(
        simd_ops_wrapper,
        "seed_rng",
        lambda value: seeded.append(int(value)),
    )
    monkeypatch.setattr(
        simd_ops_wrapper,
        "sample_token",
        lambda logits, temperature, eos_token, top_p=1.0, top_k=0, min_p=0.0: 2,  # noqa: ARG005
    )

    out = engine.generate([1, 2], max_new_tokens=1, temperature=0.7, seed=1234)

    assert out == [1, 2, 2]
    assert seeded == [1234]


def test_generate_continues_parallel_decode_after_verified_prefix_in_strict_mode():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._model_graph = None
    engine._parallel_planner = SimpleNamespace(
        plan=lambda **kwargs: SimpleNamespace(
            mode=GenerationMode.PARALLEL_HYBRID,
            accepted_prefix_tokens=[4, 5, 6, 8],
            evidence=GenerationEvidence(
                generation_mode=GenerationMode.PARALLEL_HYBRID.value,
                benchmark_label="parallel_hybrid",
            ),
        )
    )
    engine._forbid_autoregressive_fallback = True
    engine._parallel_decode_disable_reason = ""
    engine._last_temperature_band = ""
    engine._granite_moe_mode = "not_applicable"
    engine._prefill_chunk_count = 0
    engine._active_thread_mode = "decode"
    engine._last_embedding_lookup_seconds = 0.0
    engine._last_graph_prefill_seconds = 0.0
    engine._last_graph_decode_seconds = 0.0
    engine._last_sample_seconds = 0.0
    engine._last_logits_processor_seconds = 0.0
    engine._last_penalty_seconds = 0.0
    engine._last_suppression_seconds = 0.0
    engine._last_graph_prefill_calls = 0
    engine._last_graph_decode_calls = 0
    engine._last_sample_calls = 0
    engine._last_logits_processor_calls = 0
    engine._last_penalty_calls = 0
    engine._last_suppression_calls = 0
    engine._prompt_cache_reused_tokens = 0
    engine._profile = None
    engine.profile = SimpleNamespace(
        vocab_size=8, chat_template="chatml", model_name="qwen3.5:9b"
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._last_generation = NativeGenerationTelemetry(prompt_tokens=2)
    engine.token_eos = lambda: 99
    engine._annotate_telemetry = lambda telemetry, **kwargs: telemetry
    engine._apply_generation_evidence = lambda telemetry, evidence: telemetry
    engine._should_parallel_decode = lambda *args, **kwargs: False  # noqa: ARG005
    engine._generate_autoregressive = Mock(
        side_effect=AssertionError("autoregressive fallback should not run")
    )
    engine.generate_parallel = Mock(return_value=[1, 2, 4, 5, 6, 8, 9])

    out = engine.generate([1, 2], max_new_tokens=6, temperature=0.7)

    engine.generate_parallel.assert_called_once()
    assert out == [1, 2, 4, 5, 6, 8, 9]


def test_annotate_telemetry_omits_composite_ssm_stage_but_keeps_substages():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.contract = {"model": "qwen3.5:9b"}
    engine.profile = SimpleNamespace(chat_template="chatml", model_name="qwen3.5:9b")
    engine._model_graph = SimpleNamespace(
        _lm_head_layout="q6_k_r4",
        _has_batch_token_id_api=True,
        get_perf_stats=lambda: {
            "embedding_lookup_seconds": 0.01,
            "attention_proj_seconds": 0.02,
            "attention_rope_kv_seconds": 0.03,
            "attention_decode_seconds": 0.04,
            "attention_out_proj_seconds": 0.05,
            "ffn_norm_seconds": 0.06,
            "ffn_gate_up_seconds": 0.07,
            "ffn_down_seconds": 0.08,
            "ssm_projection_seconds": 0.11,
            "ssm_conv_seconds": 0.12,
            "ssm_recurrent_seconds": 0.13,
            "ssm_output_seconds": 0.14,
            "ssm_seconds": 0.50,
            "moe_seconds": 0.09,
            "final_norm_seconds": 0.15,
            "lm_head_seconds": 0.16,
            "sanitize_seconds": 0.17,
            "forward_token_calls": 2,
            "forward_token_id_calls": 2,
            "forward_token_ids_calls": 1,
            "forward_token_ids_token_count": 4,
            "attention_calls": 2,
            "ffn_calls": 2,
            "ssm_calls": 2,
            "moe_calls": 0,
            "packed_lm_head_calls": 1,
        },
    )
    engine._native_library_info = {}
    engine._native_backend_info = {}
    engine._granite_moe_mode = "not_applicable"
    engine._active_thread_mode = "decode"
    engine._prefill_chunk_count = 1
    engine._runtime_thread_switches = 0
    engine._qsg_processors_native_enabled = True
    engine._native_qsg_use_coconut = False
    engine._native_qsg_coconut_paths = 1
    engine._native_qsg_coconut_alpha = 0.0
    engine._strict_cpp_only = True
    engine._native_fast_path = True
    engine._parallel_decode_disable_reason = ""
    engine._suppressed_token_ids_sorted = ()
    engine._native_backend_required = False
    engine._tc_enabled = False
    engine._tc_graph_drift_available = False
    engine._tc_current_mode = "telemetry"
    engine._tc_last_snapshot_valid = False

    telemetry = engine._annotate_telemetry(NativeGenerationTelemetry())

    assert "ssm" not in telemetry.graph_stage_seconds
    assert telemetry.graph_stage_seconds["ssm_projection"] == pytest.approx(0.11)
    assert telemetry.graph_stage_seconds["ssm_conv"] == pytest.approx(0.12)
    assert telemetry.graph_stage_seconds["ssm_recurrent"] == pytest.approx(0.13)
    assert telemetry.graph_stage_seconds["ssm_output"] == pytest.approx(0.14)
    assert "ssm" not in telemetry.graph_stage_calls
    assert telemetry.graph_stage_calls["ssm_projection"] == 2
    assert telemetry.graph_stage_calls["ssm_conv"] == 2
    assert telemetry.graph_stage_calls["ssm_recurrent"] == 2
    assert telemetry.graph_stage_calls["ssm_output"] == 2


def test_get_runtime_status_exposes_lm_head_layout_and_batched_prefill_calls():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._runtime_capabilities = {
        "lm_head_layout": "q6_k_r4",
        "lm_head_qtype": 114,
        "kv_cache_quantization": "q8",
        "qsg_state_pages_total": 6,
        "qsg_state_pages_in_use": 6,
        "qsg_state_shared_page_slots": 4,
        "qsg_state_snapshot_count": 2,
        "qsg_state_cow_events": 3,
        "qsg_state_prefix_share_events": 5,
        "full_qsg_enabled": True,
        "graph_batched_token_id_api": True,
        "parallel_decode_allowed": False,
        "hot_path_proof": {"full_qsg": "enabled"},
        "decode_threads": 8,
        "batch_threads": 12,
        "ubatch": 32,
        "affinity_mode": 2,
        "l3_domain_count": 4,
        "os_thread_migrations": 1,
        "os_last_cpu": 3,
    }
    engine._last_generation = NativeGenerationTelemetry(
        prompt_tokens=5,
        generated_tokens=0,
        total_seconds=0.5,
        prefill_seconds=0.5,
        decode_seconds=0.0,
        first_token_latency_seconds=0.5,
        prefill_chunk_count=3,
        affinity_mode=3,
        l3_domain_count=4,
        os_thread_migrations=6,
        os_last_cpu=11,
        hot_path_proof={"lm_head_layout": "q6_k_r4"},
        graph_stage_calls={"forward_token_id": 1, "forward_token_ids": 3},
    )

    status = engine.get_runtime_status()

    assert status["lm_head_layout"] == "q6_k_r4"
    assert status["lm_head_qtype"] == 114
    assert status["kv_cache_quantization"] == "q8"
    assert status["qsg_state_pages_total"] == 6
    assert status["qsg_state_pages_in_use"] == 6
    assert status["qsg_state_shared_page_slots"] == 4
    assert status["qsg_state_snapshot_count"] == 2
    assert status["qsg_state_cow_events"] == 3
    assert status["qsg_state_prefix_share_events"] == 5
    assert status["full_qsg_enabled"] is True
    assert status["graph_batched_token_id_api"] is True
    assert status["parallel_decode_allowed"] is False
    assert status["prefill_chunk_count"] == 3
    assert status["affinity_mode"] == 3
    assert status["l3_domain_count"] == 4
    assert status["os_thread_migrations"] == 6
    assert status["os_last_cpu"] == 11
    assert status["hot_path_proof"]["lm_head_layout"] == "q6_k_r4"
    assert status["hot_path_proof"]["full_qsg"] == "enabled"
    assert status["graph_stage_calls"]["forward_token_id"] == 1
    assert status["graph_stage_calls"]["forward_token_ids"] == 3
    assert status["performance_twin"]["risk_level"] in {"low", "medium", "high"}
    assert status["repo_coupled_runtime"]["delta_authority"] == "state_ledger"
    assert (
        status["repo_coupled_runtime"]["performance_twin"]["risk_level"]
        == status["performance_twin"]["risk_level"]
    )


def test_get_runtime_status_preserves_capability_backends_when_telemetry_is_blank():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._runtime_capabilities = {
        "logits_backend": "native_graph_token_id",
        "sampling_backend": "native_simd",
        "lm_head_layout": "q6_k_r4",
    }
    engine._last_generation = NativeGenerationTelemetry()

    status = engine.get_runtime_status()

    assert status["logits_backend"] == "native_graph_token_id"
    assert status["sampling_backend"] == "native_simd"
    assert status["lm_head_layout"] == "q6_k_r4"


def test_get_runtime_status_exposes_controller_memory_and_twin() -> None:
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._runtime_capabilities = {
        "context_stabilizer_enabled": True,
        "context_stabilizer_mode": "aggressive",
        "generation_mode": "medusa_head",
        "draft_frontier_width": 4,
        "accepted_parallel_tokens": 6,
        "rejected_parallel_tokens": 2,
        "proposed_parallel_tokens": 8,
        "prompt_cache_hit_ratio": 0.65,
        "prefix_cache_hit_rate": 0.5,
        "prompt_category": "structured",
        "drift_overhead_percent": 3.5,
        "stabilizer_calls": 2,
        "backend_module_loaded": True,
        "strict_path_stable": True,
    }
    engine._last_generation = NativeGenerationTelemetry(
        prompt_tokens=128,
        generated_tokens=16,
        total_seconds=1.0,
        first_token_latency_seconds=0.08,
        generation_mode="medusa_head",
        prompt_category="structured",
        accepted_parallel_tokens=6,
        rejected_parallel_tokens=2,
        proposed_parallel_tokens=8,
        draft_frontier_width=4,
        prompt_cache_hit=True,
    )
    engine._delta_watermark = {"delta_id": "delta-1", "changed_paths": ["core/a.py"]}

    status = engine.get_runtime_status()

    assert status["controller_state"]["frontier"]["controller"] == "frontier"
    assert status["controller_state"]["drift"]["controller"] == "drift"
    assert status["controller_state"]["memory_tier"]["controller"] == "memory_tier"
    assert status["controller_state"]["memory_tier"]["selected_mode"] in {
        "prompt_cache",
        "repo_delta_memory",
    }
    assert status["performance_twin"]["risk_level"] in {"low", "medium", "high"}
    assert status["repo_coupled_runtime"]["delta_watermark"]["delta_id"] == "delta-1"


def test_build_runtime_capabilities_exposes_optional_isa_leaves_and_l3_reservation(
    monkeypatch: pytest.MonkeyPatch,
):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.loader = SimpleNamespace(
        get_metadata=lambda: {},
        get_quantization_label=lambda: "q4_k",
    )
    engine.contract = {"model": "granite4:tiny-h", "digest": "abc123"}
    engine.profile = SimpleNamespace(model_name="granite4:tiny-h")
    engine.backend = "native"
    engine.architecture = "granite"
    engine.context_length = 4096
    engine.load_seconds = 0.0
    engine.num_threads_decode = 8
    engine.num_threads_batch = 12
    engine.num_ubatch = 32
    engine._use_mmap_weights = True
    engine._strict_prompt_contract = SimpleNamespace(template_name="")
    engine._refresh_os_thread_telemetry = lambda: None
    engine._suppressed_token_ids_sorted = ()
    engine._affinity_plan_json = json.dumps(
        {
            "decode_worker_cpus": [2, 3],
            "orchestrator_cpus": [0],
            "visible_l3_domains": [0, 1, 2],
            "batch_l3_domains": [0, 2],
            "decode_primary_l3_domain": 1,
            "preferred_decode_l3_domain": 1,
            "decode_domain_reserved": True,
        }
    )
    engine._topology_json = json.dumps({"l3_domains": []})
    engine._native_library_info = {
        "native_isa_baseline": "avx2",
        "native_optional_isa_leaves": ["amx"],
        "native_optional_isa_leaves_csv": "amx",
        "native_split_abi_version": 1,
        "native_compiled_with_amx": True,
        "native_runtime_amx_available": True,
    }
    engine._native_backend_info = {
        "backend_module": "granite4",
        "backend_module_loaded": False,
        "backend_module_abi_version": 0,
    }

    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "get_qsg_sampling_stats",
        lambda: {},
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "get_omp_max_threads",
        lambda: 16,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "get_omp_dynamic",
        lambda: False,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "get_omp_active_levels",
        lambda: 1,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "openmp_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "compiled_with_avx2",
        lambda: True,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "compiled_with_avx512",
        lambda: False,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "compiled_with_amx",
        lambda: True,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "runtime_amx_available",
        lambda: True,
    )
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops,
        "get_num_procs",
        lambda: 16,
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "native_parallel_kernels_available",
        lambda: False,
    )

    capabilities = engine._build_runtime_capabilities()

    assert capabilities["native_optional_isa_leaves"] == ["amx"]
    assert capabilities["amx_enabled"] is True
    assert capabilities["amx_compiled"] is True
    assert capabilities["amx_kernel_enabled"] is True
    assert capabilities["amx_runtime_available"] is True
    assert capabilities["decode_primary_l3_domain"] == 1
    assert capabilities["preferred_decode_l3_domain"] == 1
    assert capabilities["batch_l3_domains"] == [0, 2]
    assert capabilities["decode_domain_reserved"] is True
    assert capabilities["hot_path_proof"]["native_optional_isa_leaves"] == "amx"
    assert capabilities["hot_path_proof"]["amx_leaf"] == "enabled"


def test_update_scheduler_metrics_snapshot_tracks_native_kv_fields():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine._runtime_capabilities = {}

    NativeQSGEngine._update_scheduler_metrics_snapshot(
        engine,
        {
            "scheduler_queue_wait_ms": 1.25,
            "scheduler_iteration_ms": 2.75,
            "kv_fragmentation_ratio": 0.3,
            "qsg_state_pages_total": 6,
            "qsg_state_pages_in_use": 6,
            "qsg_state_active_page_slots": 9,
            "qsg_state_shared_page_slots": 4,
            "qsg_state_snapshot_count": 2,
            "qsg_state_cow_events": 3,
            "qsg_state_prefix_share_events": 5,
            "qsg_state_active_tokens": 17,
            "qsg_state_committed_token_capacity": 64,
            "qsg_state_page_tokens": 8,
        },
    )

    assert engine._scheduler_queue_wait_ms == 1.25
    assert engine._scheduler_iteration_ms == 2.75
    assert engine._scheduler_kv_fragmentation_ratio == 0.3
    assert engine._scheduler_kv_pages_total == 6
    assert engine._scheduler_kv_pages_in_use == 6
    assert engine._scheduler_kv_active_page_slots == 9
    assert engine._scheduler_kv_shared_page_slots == 4
    assert engine._scheduler_kv_snapshot_count == 2
    assert engine._scheduler_kv_cow_events == 3
    assert engine._scheduler_kv_prefix_share_events == 5
    assert engine._scheduler_kv_active_tokens == 17
    assert engine._scheduler_kv_committed_token_capacity == 64
    assert engine._scheduler_kv_page_tokens == 8
    assert engine._runtime_capabilities["qsg_state_shared_page_slots"] == 4
    assert engine._runtime_capabilities["qsg_state_cow_events"] == 3


def test_enforce_native_split_backend_abi_accepts_matching_versions():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.model_name = "qwen3.5:4b"
    engine._native_library_info = {"native_split_abi_version": 1}
    engine._native_backend_info = {
        "backend_module": "qwen35",
        "backend_module_loaded": True,
        "backend_module_abi_version": 1,
    }

    engine._enforce_native_split_backend_abi()


def test_enforce_native_split_backend_abi_rejects_mismatch():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.model_name = "qwen3.5:4b"
    engine._native_library_info = {"native_split_abi_version": 1}
    engine._native_backend_info = {
        "backend_module": "qwen35",
        "backend_module_loaded": True,
        "backend_module_abi_version": 2,
    }

    with pytest.raises(RuntimeError, match="ABI mismatch"):
        engine._enforce_native_split_backend_abi()


def test_sample_prefers_finite_argmax_when_non_finite_logits_present():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 7

    token = engine._sample(
        np.asarray([np.nan, -np.inf, 2.5], dtype=np.float32),
        temperature=0.0,
    )

    assert token == 2


def test_format_prompt_uses_shared_chatml_template():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(
        chat_template="chatml",
        family="qwen",
        model_name="qwen3.5:9b",
    )
    engine.contract = {"model": "qwen3.5:9b"}
    engine._strict_prompt_contract = SimpleNamespace(
        template_name="chatml",
        system_prompt="You are a helpful assistant.",
        assistant_prefix=None,
    )

    prompt = engine.format_prompt("Hello")

    assert prompt.startswith("<|im_start|>user\nHello<|im_end|>\n")
    assert prompt.endswith("<|im_start|>assistant\n")


def test_decode_generated_tokens_postprocesses_qwen_think_stub():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.contract = {"model": "qwen3.5:9b"}
    engine.profile = SimpleNamespace(model_name="qwen3.5:9b", chat_template="chatml")
    engine._strict_prompt_contract = SimpleNamespace(template_name="chatml")
    engine.detokenize = lambda tokens: "<think>\n\n</think>\n\nanswer"  # noqa: ARG005

    assert engine.decode_generated_tokens([1, 2, 3]) == "answer"


def test_format_prompt_uses_shared_granite_template():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.profile = SimpleNamespace(
        chat_template="granite",
        family="granite",
        model_name="granite4:tiny-h",
    )
    engine.contract = {"model": "granite4:tiny-h"}
    engine._strict_prompt_contract = SimpleNamespace(
        template_name="granite",
        system_prompt="You are a helpful assistant. Please ensure responses are professional, accurate, and safe.",
        assistant_prefix=None,
    )

    prompt = engine.format_prompt("Hello")

    assert prompt.startswith("<|start_of_role|>system<|end_of_role|>")
    assert "professional, accurate, and safe." in prompt
    assert prompt.endswith("<|start_of_role|>assistant<|end_of_role|>")


def test_sample_returns_eos_when_all_logits_non_finite():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 5

    token = engine._sample(
        np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32),
        temperature=0.8,
    )

    assert token == 5


def test_sample_suppresses_control_tokens_before_argmax():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 7
    engine._suppressed_token_ids = {1}

    token = engine._sample(
        np.asarray([0.0, 10.0, 9.0, 8.0], dtype=np.float32),
        temperature=0.0,
    )

    assert token == 2


def test_sample_suppresses_leading_disallowed_prefix_tokens_before_argmax():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 7
    engine._suppressed_token_ids = set()
    engine._leading_disallowed_token_ids = (1,)

    token = engine._sample(
        np.asarray([0.0, 10.0, 9.0, 8.0], dtype=np.float32),
        temperature=0.0,
        leading_response=True,
    )

    assert token == 2


def test_sample_applies_repetition_penalty_before_deterministic_argmax():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 7
    engine._suppressed_token_ids = set()
    engine.architecture = "qwen35"

    token = engine._sample(
        np.asarray([0.0, 10.0, 9.0, 1.0], dtype=np.float32),
        token_history=array("i", [1, 1]),
        temperature=0.0,
        repetition_penalty=2.0,
    )

    assert token == 2


def test_sample_uses_json_grammar_fastlane_for_structured_prompts(monkeypatch):
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    engine.token_eos = lambda: 7
    engine._suppressed_token_ids = set()
    engine._last_prompt_category = "structured"
    engine._json_fastlane_open_ids = array("i", [1])
    engine._json_fastlane_object_follow_ids = array("i", [2, 3])
    engine.finalize_response_text = lambda text: text
    engine.detokenize = lambda tokens: ""

    captured: dict[str, list[int]] = {}

    def _fake_sample(logits, **kwargs):
        del logits
        captured["grammar_allowed_ids"] = list(kwargs.get("grammar_allowed_ids") or [])
        return captured["grammar_allowed_ids"][0]

    monkeypatch.setattr(simd_ops_wrapper, "qsg_postprocess_and_sample", _fake_sample)

    token = engine._sample(
        np.asarray([0.0, 10.0, 9.0, 8.0], dtype=np.float32),
        temperature=0.0,
    )

    assert token == 1
    assert captured["grammar_allowed_ids"] == [1]


def test_native_penalty_wrapper_does_not_pin_array_history_buffer():
    logits = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    history = array("i", [1, 2, 3])

    simd_ops_wrapper.apply_token_penalties_inplace(
        logits,
        history,
        presence_penalty=0.0,
        repetition_penalty=1.05,
    )
    history.append(4)

    assert list(history) == [1, 2, 3, 4]


def test_native_score_token_returns_greedy_and_probability():
    logits = np.asarray([0.0, 1.0, 2.0], dtype=np.float32)

    greedy_token, token_prob = simd_ops_wrapper.score_token(
        logits,
        token_id=2,
        temperature=1.0,
    )

    expected = float(np.exp(2.0) / np.sum(np.exp(np.asarray([0.0, 1.0, 2.0]))))
    assert greedy_token == 2
    assert token_prob == pytest.approx(expected, rel=2e-2, abs=1e-3)


def test_verify_draft_tokens_uses_native_score_token(monkeypatch):
    calls: list[int] = []

    class _DraftEngine:
        _native_fast_path = True
        _disable_logits_processors = True

        @staticmethod
        def _get_logits_for_tokens(_tokens):
            return [0.0, 1.0, 2.0, 3.0]

        @staticmethod
        def token_eos():
            return 99

    monkeypatch.setattr(
        parallel_generation_module.simd_ops,
        "score_token",
        lambda logits, *, token_id, temperature: (  # noqa: ARG005
            calls.append(int(token_id)) or int(token_id),
            0.95,
        ),
    )

    accepted, probabilities = parallel_generation_module._verify_draft_tokens(
        _DraftEngine(),
        prompt_tokens=[1, 2],
        draft_tokens=[3, 4],
        temperature=0.7,
        min_accept_prob=0.2,
    )

    assert accepted == 2
    assert probabilities == [0.95, 0.95]
    assert calls == [3, 4]


def test_apply_logits_processors_sanitizes_non_finite_output():
    logits = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def _broken_processor(_input_ids, _scores):
        return np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)

    updated = NativeQSGEngine._apply_logits_processors(
        [1, 2, 3], logits, [_broken_processor]
    )

    assert updated.shape == logits.shape
    assert np.all(np.isfinite(updated))


def test_build_weight_store_defaults_to_mmap(monkeypatch):
    calls: list[str] = []

    class _MmapStore:
        def __init__(self, _loader, _profile):
            calls.append("mmap")

    class _DenseStore:
        def __init__(self, _loader, _profile):
            calls.append("dense")

    monkeypatch.delenv("ANVIL_NATIVE_USE_MMAP_WEIGHTS", raising=False)
    monkeypatch.setattr(native_qsg_engine_module, "MMapWeightStore", _MmapStore)
    monkeypatch.setattr(native_qsg_engine_module, "WeightStore", _DenseStore)

    _, use_mmap = native_qsg_engine_module._build_weight_store(object(), object())

    assert calls == ["mmap"]
    assert use_mmap is True


def test_build_weight_store_opt_out_uses_dense_store(monkeypatch):
    calls: list[str] = []

    class _MmapStore:
        def __init__(self, _loader, _profile):
            calls.append("mmap")

    class _DenseStore:
        def __init__(self, _loader, _profile):
            calls.append("dense")

    monkeypatch.setenv("ANVIL_NATIVE_USE_MMAP_WEIGHTS", "0")
    monkeypatch.setattr(native_qsg_engine_module, "MMapWeightStore", _MmapStore)
    monkeypatch.setattr(native_qsg_engine_module, "WeightStore", _DenseStore)

    _, use_mmap = native_qsg_engine_module._build_weight_store(object(), object())

    assert calls == ["dense"]
    assert use_mmap is False


def test_build_weight_store_enabled_without_mmap_store_raises(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_USE_MMAP_WEIGHTS", raising=False)
    monkeypatch.setattr(native_qsg_engine_module, "MMapWeightStore", None)

    with np.testing.assert_raises(RuntimeError):
        native_qsg_engine_module._build_weight_store(object(), object())


def test_allow_full_graph_for_architecture_always_true():
    assert native_qsg_engine_module._allow_full_graph_for_architecture("granitehybrid")
    assert native_qsg_engine_module._allow_full_graph_for_architecture("qwen35")


def test_auto_num_threads_uses_affinity_count_when_available(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_USE_LOGICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NUM_THREADS_HEADROOM", raising=False)
    monkeypatch.setattr(
        native_qsg_engine_module.os,
        "sched_getaffinity",
        lambda _pid: {0, 1, 2, 3, 4, 5},
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 16)
    assert native_qsg_engine_module._auto_num_threads() == 6


def test_auto_num_threads_falls_back_to_cpu_count_when_affinity_unavailable(
    monkeypatch,
):
    monkeypatch.delenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_USE_LOGICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NUM_THREADS_HEADROOM", raising=False)

    def _raise_affinity(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(
        native_qsg_engine_module.os, "sched_getaffinity", _raise_affinity
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 14)
    assert native_qsg_engine_module._auto_num_threads() == 14


def test_auto_num_threads_can_opt_into_physical_core_policy(monkeypatch):
    monkeypatch.setenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", "1")
    monkeypatch.delenv("ANVIL_NUM_THREADS_HEADROOM", raising=False)

    def _raise_affinity(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(
        native_qsg_engine_module.os, "sched_getaffinity", _raise_affinity
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 14)

    assert native_qsg_engine_module._auto_num_threads() == 7


def test_auto_num_threads_defaults_granite_to_physical_hint(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_USE_LOGICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NUM_THREADS_HEADROOM", raising=False)

    def _raise_affinity(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(
        native_qsg_engine_module.os, "sched_getaffinity", _raise_affinity
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 16)

    assert native_qsg_engine_module._auto_num_threads("granitehybrid") == 12
    assert native_qsg_engine_module._auto_num_threads("qwen35") == 12


def test_auto_num_threads_respects_headroom_env_override(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_USE_LOGICAL_THREADS", raising=False)
    monkeypatch.setenv("ANVIL_NUM_THREADS_HEADROOM", "3")

    def _raise_affinity(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(
        native_qsg_engine_module.os, "sched_getaffinity", _raise_affinity
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 16)

    assert native_qsg_engine_module._auto_num_threads("granitehybrid") == 13
    assert (
        native_qsg_engine_module._auto_num_threads(
            "qwen35",
            n_layers=64,
            embedding_dim=4096,
        )
        == 13
    )
    assert native_qsg_engine_module._auto_num_threads() == 13


def test_auto_num_threads_headroom_respects_auto_min_floor(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_USE_PHYSICAL_THREADS", raising=False)
    monkeypatch.delenv("ANVIL_NATIVE_USE_LOGICAL_THREADS", raising=False)
    monkeypatch.setenv("ANVIL_NUM_THREADS_HEADROOM", "99")
    monkeypatch.setenv("ANVIL_AUTO_MIN_THREADS", "6")

    def _raise_affinity(_pid):
        raise OSError("no affinity")

    monkeypatch.setattr(
        native_qsg_engine_module.os, "sched_getaffinity", _raise_affinity
    )
    monkeypatch.setattr(native_qsg_engine_module.os, "cpu_count", lambda: 16)

    assert native_qsg_engine_module._auto_num_threads("granitehybrid") == 6
    assert native_qsg_engine_module._auto_num_threads("qwen35") == 6
    assert native_qsg_engine_module._auto_num_threads() == 6


def test_auto_batch_threads_clamps_to_available_threads(monkeypatch):
    monkeypatch.setattr(
        native_qsg_engine_module, "_visible_logical_threads", lambda: 12
    )
    assert native_qsg_engine_module._auto_batch_threads(64) == 12
    assert native_qsg_engine_module._auto_batch_threads(6) == 6


def test_auto_batch_threads_granite_matches_decode_pool(monkeypatch):
    monkeypatch.setattr(
        native_qsg_engine_module, "_visible_logical_threads", lambda: 16
    )

    assert native_qsg_engine_module._auto_batch_threads(8, "granitehybrid") == 8
    assert native_qsg_engine_module._auto_batch_threads(6, "granitehybrid") == 6
    assert native_qsg_engine_module._auto_batch_threads(8, "qwen35") == 8
    assert native_qsg_engine_module._auto_batch_threads(6, "qwen35") == 6


def test_auto_num_ubatch_uses_architecture_specific_defaults():
    assert native_qsg_engine_module._auto_num_ubatch(12, "granitehybrid") == 32
    assert native_qsg_engine_module._auto_num_ubatch(8, "granitehybrid") == 32
    assert native_qsg_engine_module._auto_num_ubatch(4, "granitehybrid") == 8
    assert native_qsg_engine_module._auto_num_ubatch(16, "qwen35") == 32
    assert native_qsg_engine_module._auto_num_ubatch(8, "qwen35") == 16
    assert native_qsg_engine_module._auto_num_ubatch(4, "qwen35") == 8


def test_configure_openmp_affinity_sets_granite_defaults(monkeypatch):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    monkeypatch.delenv("OMP_PLACES", raising=False)
    monkeypatch.delenv("ANVIL_OMP_PROC_BIND", raising=False)
    monkeypatch.delenv("ANVIL_OMP_PLACES", raising=False)

    native_qsg_engine_module._configure_openmp_affinity("granitehybrid")

    assert native_qsg_engine_module.os.environ["OMP_PROC_BIND"] == "false"
    assert native_qsg_engine_module.os.environ["OMP_PLACES"] == "threads"


def test_configure_openmp_affinity_respects_existing_env(monkeypatch):
    monkeypatch.setenv("OMP_PROC_BIND", "spread")
    monkeypatch.setenv("OMP_PLACES", "threads")

    native_qsg_engine_module._configure_openmp_affinity("granitehybrid")

    assert native_qsg_engine_module.os.environ["OMP_PROC_BIND"] == "spread"
    assert native_qsg_engine_module.os.environ["OMP_PLACES"] == "threads"


def test_resolve_native_context_length_uses_400k_cap_by_default(monkeypatch):
    monkeypatch.delenv("ANVIL_NATIVE_CTX_CAP", raising=False)
    profile = SimpleNamespace(family="qwen", architecture="qwen35")
    loader = SimpleNamespace(get_context_length=lambda: 131072)

    resolved, requested, model_limit, cap = (
        native_qsg_engine_module._resolve_native_context_length(
            requested_ctx=400000,
            profile=profile,
            loader=loader,
        )
    )

    assert requested == 400000
    assert model_limit == 131072
    assert cap == 400000
    assert resolved == 131072


def test_resolve_native_context_length_respects_env_cap(monkeypatch):
    monkeypatch.setenv("ANVIL_NATIVE_CTX_CAP", "4096")
    profile = SimpleNamespace(family="qwen", architecture="qwen35")
    loader = SimpleNamespace(get_context_length=lambda: 131072)

    resolved, _, _, cap = native_qsg_engine_module._resolve_native_context_length(
        requested_ctx=400000,
        profile=profile,
        loader=loader,
    )

    assert cap == 4096
    assert resolved == 4096


def test_get_logits_hybrid_is_disabled_in_strict_cpp_mode():
    engine = NativeQSGEngine.__new__(NativeQSGEngine)
    with np.testing.assert_raises(RuntimeError):
        engine._get_logits_hybrid([42], start_pos=7)


def test_engine_strict_cpp_requires_full_layer_coverage(monkeypatch):
    close_calls: list[int] = []

    class _FakeLoader:
        def __init__(self, _model_name: str):
            self.model_path = "/tmp/fake.gguf"

        @staticmethod
        def get_special_tokens():
            return {"eos": 2, "bos": 1}

        @staticmethod
        def get_metadata():
            return {
                "tokenizer.ggml.add_bos_token": False,
                "tokenizer.ggml.model": "bpe",
            }

        @staticmethod
        def get_vocab_tokens():
            return ["<pad>", "<bos>", "<eos>", "hello"]

        @staticmethod
        def get_tokenizer_merges():
            return None

    class _FakeGraph:
        drift_configs: list[dict[str, int | float]] = []

        def __init__(self, **_kwargs):
            self.available = True
            self._has_full_graph = True

        @property
        def has_full_graph(self):
            return self._has_full_graph and self.available

        def can_layer_cpp(self, layer_idx):
            return False

        def close(self):
            close_calls.append(1)

    class _FakeForwardPass:
        def __init__(self, *_args, **_kwargs):
            self.embedding_scale = 0.0

        @staticmethod
        def reset():
            return None

        @staticmethod
        def forward(token_ids, start_pos=0):
            del token_ids
            del start_pos
            return np.zeros((4,), dtype=np.float32)

        @staticmethod
        def get_hidden_states(token_ids, target_layer=0):
            del target_layer
            return np.zeros((len(token_ids), 4), dtype=np.float32)

    class _FakeTokenizer:
        eos_id = 2

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def encode(_text, add_bos=True):
            return [1] if add_bos else []

        @staticmethod
        def decode(tokens, skip_special=True):
            del skip_special
            return " ".join(str(t) for t in tokens)

    profile = SimpleNamespace(
        architecture="granitehybrid",
        n_layers=1,
        embedding_dim=4,
        vocab_size=4,
        n_heads=1,
        n_kv_heads=1,
    )

    monkeypatch.setenv("ANVIL_NATIVE_GRAPH_MODE", "1")
    monkeypatch.setenv("ANVIL_NATIVE_STRICT_CPP_ONLY", "1")
    monkeypatch.setattr(native_qsg_engine_module, "GGUFModelLoader", _FakeLoader)
    monkeypatch.setattr(
        native_qsg_engine_module.ModelProfile,
        "from_loader",
        lambda model_name, loader: profile,
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "resolve_model_contract",
        lambda model_name: SimpleNamespace(
            canonical_name="granite4:tiny-h",
            template_name="granite",
            strict_native_supported=True,
            manifest_path="manifest",
            blob_path="blob",
            manifest_sha256="sha256:manifest",
            expected_manifest_digest="sha256:manifest",
            manifest_digest="sha256:model",
            expected_model_digest="sha256:model",
            expected_digest="sha256:model",
            blob_size=1024,
            digest_validated=True,
            local_sha256=None,
            quant_variant="test",
        ),
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "model_contract_snapshot",
        lambda contract: {
            "model": contract.canonical_name,
            "digest": contract.expected_model_digest,
            "blob_size": contract.blob_size,
            "template_name": contract.template_name,
        },
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "_build_weight_store",
        lambda loader, profile: (
            SimpleNamespace(
                lookup_embeddings=lambda token_ids: np.zeros(
                    (len(token_ids), 4), dtype=np.float32
                )
            ),
            True,
        ),
    )
    monkeypatch.setattr(native_qsg_engine_module, "QSGForwardPass", _FakeForwardPass)
    monkeypatch.setattr(native_qsg_engine_module, "NativeTokenizer", _FakeTokenizer)
    monkeypatch.setattr(native_qsg_engine_module, "NativeModelGraph", _FakeGraph)
    monkeypatch.setattr(native_qsg_engine_module, "NativeKVCacheWrapper", None)
    monkeypatch.setattr(native_qsg_engine_module, "SSMSelfSpeculativeDecoder", None)
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops, "set_num_threads", lambda n: None
    )
    monkeypatch.setattr(native_qsg_engine_module.simd_ops, "get_num_threads", lambda: 1)

    with np.testing.assert_raises(RuntimeError):
        NativeQSGEngine("granite4:tiny-h", context_length=32)
    assert close_calls == [1]


def test_engine_allows_partial_cpp_when_strict_mode_disabled(monkeypatch):
    class _FakeLoader:
        def __init__(self, _model_name: str):
            self.model_path = "/tmp/fake.gguf"

        @staticmethod
        def get_special_tokens():
            return {"eos": 2, "bos": 1}

        @staticmethod
        def get_metadata():
            return {
                "tokenizer.ggml.add_bos_token": False,
                "tokenizer.ggml.model": "bpe",
            }

        @staticmethod
        def get_vocab_tokens():
            return ["<pad>", "<bos>", "<eos>", "hello"]

        @staticmethod
        def get_tokenizer_merges():
            return None

    class _FakeGraph:
        drift_configs: list[dict[str, int | float]] = []

        def __init__(self, **_kwargs):
            self.available = True
            self._has_full_graph = True

        @property
        def has_full_graph(self):
            return self._has_full_graph and self.available

        @staticmethod
        def can_layer_cpp(_layer_idx):
            return False

        @staticmethod
        def close():
            return None

        @classmethod
        def set_drift_config(cls, config):
            cls.drift_configs.append(dict(config))
            return True

    class _FakeForwardPass:
        def __init__(self, *_args, **_kwargs):
            self.embedding_scale = 0.0
            self.enable_residual_stabilizer = False

        @staticmethod
        def reset():
            return None

        @staticmethod
        def forward(token_ids, start_pos=0):
            del token_ids
            del start_pos
            return np.zeros((4,), dtype=np.float32)

        @staticmethod
        def get_hidden_states(token_ids, target_layer=0):
            del target_layer
            return np.zeros((len(token_ids), 4), dtype=np.float32)

    class _FakeTokenizer:
        eos_id = 2

        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def encode(_text, add_bos=True):
            return [1] if add_bos else []

        @staticmethod
        def decode(tokens, skip_special=True):
            del skip_special
            return " ".join(str(t) for t in tokens)

    profile = SimpleNamespace(
        architecture="granitehybrid",
        n_layers=1,
        embedding_dim=4,
        vocab_size=4,
        n_heads=1,
        n_kv_heads=1,
    )

    monkeypatch.setenv("ANVIL_NATIVE_GRAPH_MODE", "1")
    monkeypatch.setenv("ANVIL_NATIVE_STRICT_CPP_ONLY", "0")
    monkeypatch.setenv("ANVIL_TC_STABILIZER", "1")
    monkeypatch.setenv("ANVIL_TC_MODE", "conservative")
    monkeypatch.setenv("ANVIL_TC_BLOCK_SIZE", "256")
    monkeypatch.setenv("ANVIL_TC_UPDATE_INTERVAL", "32")
    monkeypatch.setenv("ANVIL_TC_DAMP_THRESHOLD", "0.4")
    monkeypatch.setenv("ANVIL_TC_PRUNE_THRESHOLD", "0.8")
    monkeypatch.setenv("ANVIL_TC_PRESERVE_RECENT", "4096")
    monkeypatch.setattr(native_qsg_engine_module, "GGUFModelLoader", _FakeLoader)
    monkeypatch.setattr(
        native_qsg_engine_module.ModelProfile,
        "from_loader",
        lambda model_name, loader: profile,
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "resolve_model_contract",
        lambda model_name: SimpleNamespace(
            canonical_name="granite4:tiny-h",
            template_name="granite",
            strict_native_supported=True,
            manifest_path="manifest",
            blob_path="blob",
            manifest_sha256="sha256:manifest",
            expected_manifest_digest="sha256:manifest",
            manifest_digest="sha256:model",
            expected_model_digest="sha256:model",
            expected_digest="sha256:model",
            blob_size=1024,
            digest_validated=True,
            local_sha256=None,
            quant_variant="test",
        ),
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "model_contract_snapshot",
        lambda contract: {
            "model": contract.canonical_name,
            "digest": contract.expected_model_digest,
            "blob_size": contract.blob_size,
            "template_name": contract.template_name,
        },
    )
    monkeypatch.setattr(
        native_qsg_engine_module,
        "_build_weight_store",
        lambda loader, profile: (
            SimpleNamespace(
                lookup_embeddings=lambda token_ids: np.zeros(
                    (len(token_ids), 4), dtype=np.float32
                ),
                get_layer_weights=lambda layer_idx: {},
            ),
            True,
        ),
    )
    monkeypatch.setattr(native_qsg_engine_module, "QSGForwardPass", _FakeForwardPass)
    monkeypatch.setattr(native_qsg_engine_module, "NativeTokenizer", _FakeTokenizer)
    monkeypatch.setattr(native_qsg_engine_module, "NativeModelGraph", _FakeGraph)
    monkeypatch.setattr(native_qsg_engine_module, "NativeKVCacheWrapper", None)
    monkeypatch.setattr(native_qsg_engine_module, "SSMSelfSpeculativeDecoder", None)
    monkeypatch.setattr(
        native_qsg_engine_module.simd_ops, "set_num_threads", lambda n: None
    )
    monkeypatch.setattr(native_qsg_engine_module.simd_ops, "get_num_threads", lambda: 1)

    engine = NativeQSGEngine("granite4:tiny-h", context_length=32)

    assert engine._model_graph is not None
    assert engine._hybrid_mode is False
    assert _FakeGraph.drift_configs
    config = _FakeGraph.drift_configs[-1]
    assert config["enabled"] == 1
    assert config["mode"] == 1
    assert config["block_size_tokens"] == 256
    assert config["update_interval_tokens"] == 32
    assert config["preserve_recent_tokens"] == 4096
    assert config["damp_threshold"] == pytest.approx(0.4)
    assert config["prune_threshold"] == pytest.approx(0.8)
