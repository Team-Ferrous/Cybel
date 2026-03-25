"""Parallel-first generation planner and native scheduler wrappers for QSG."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import threading
import time
from typing import Any, Callable, Iterator, Protocol
import uuid

import numpy as np

from core.native import simd_ops_wrapper as simd_ops
from core.native.qsg_parallel_kernels_wrapper import (
    NativeQSGRuntime,
    NativeQSGScheduler,
    NativeQSGRequestState,
    qsg_eagle_replacement_draft,
    qsg_prompt_lookup_draft,
    qsg_verify_draft_tokens,
)
from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGChunk, QSGRequest
from core.qsg.runtime_contracts import (
    DeltaWatermark,
    ExecutionCapsule as RuntimeExecutionCapsule,
    SpeculativeFrontierPolicy,
)
from shared_kernel.event_store import get_event_store


class GenerationMode(str, Enum):
    PROMPT_LOOKUP = "prompt_lookup"
    PARALLEL_HYBRID = "parallel_hybrid"
    MEDUSA_HEAD = "medusa_head"
    HYDRA_HEAD = "hydra_head"
    SSD_BRIDGE = "ssd_bridge"
    BLOCK_DIFFUSION = "block_diffusion"
    MASKED_DIFFUSION = "masked_diffusion"
    REPLACEMENT = "replacement"
    AR_VERIFY = "ar_verify"
    AR_RECOVERY = "ar_recovery"


class BenchmarkLabel(str, Enum):
    AR_BASELINE = "ar_baseline"
    PROMPT_LOOKUP = "prompt_lookup"
    SSD_BRIDGE = "ssd_bridge"
    PARALLEL_HYBRID = "parallel_hybrid"
    MEDUSA_HEAD_CANDIDATE = "medusa_head_candidate"
    HYDRA_HEAD_CANDIDATE = "hydra_head_candidate"
    BLOCK_DIFFUSION_CANDIDATE = "block_diffusion_candidate"
    MASKED_DIFFUSION_CANDIDATE = "masked_diffusion_candidate"
    REPLACEMENT_CANDIDATE = "replacement_candidate"


class SequenceLifecycle(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    AWAITING_TOOL = "awaiting_tool"
    RESUME_PENDING = "resume_pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class SequenceExecutionMode(str, Enum):
    TEXT = "text"
    LATENT = "latent"
    HYBRID = "hybrid"


class ParkLevel(str, Enum):
    HOT_PAUSE = "hot_pause"
    WARM_PARK = "warm_park"
    COLD_HIBERNATE = "cold_hibernate"


def generation_mode_from_value(value: GenerationMode | str) -> GenerationMode:
    if isinstance(value, GenerationMode):
        return value
    normalized = str(value or "").strip().lower()
    for mode in GenerationMode:
        if mode.value == normalized:
            return mode
    return GenerationMode.AR_VERIFY


def benchmark_label_for_mode(value: GenerationMode | str) -> BenchmarkLabel:
    mode = generation_mode_from_value(value)
    if mode == GenerationMode.PROMPT_LOOKUP:
        return BenchmarkLabel.PROMPT_LOOKUP
    if mode == GenerationMode.BLOCK_DIFFUSION:
        return BenchmarkLabel.BLOCK_DIFFUSION_CANDIDATE
    if mode == GenerationMode.MASKED_DIFFUSION:
        return BenchmarkLabel.MASKED_DIFFUSION_CANDIDATE
    if mode == GenerationMode.MEDUSA_HEAD:
        return BenchmarkLabel.MEDUSA_HEAD_CANDIDATE
    if mode == GenerationMode.HYDRA_HEAD:
        return BenchmarkLabel.HYDRA_HEAD_CANDIDATE
    if mode == GenerationMode.SSD_BRIDGE:
        return BenchmarkLabel.SSD_BRIDGE
    if mode == GenerationMode.PARALLEL_HYBRID:
        return BenchmarkLabel.PARALLEL_HYBRID
    if mode == GenerationMode.REPLACEMENT:
        return BenchmarkLabel.REPLACEMENT_CANDIDATE
    return BenchmarkLabel.AR_BASELINE


def supported_benchmark_labels() -> list[str]:
    return [label.value for label in BenchmarkLabel]


class DraftPlanner(Protocol):
    def draft(self, prompt_tokens: list[int], max_draft_tokens: int) -> list[int]: ...

    def verify(
        self,
        engine: Any,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        *,
        temperature: float,
        logits_processor: Any = None,
    ) -> tuple[int, list[float]]: ...


class Verifier(Protocol):
    def verify(
        self,
        engine: Any,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        *,
        temperature: float,
        logits_processor: Any = None,
    ) -> tuple[int, list[float]]: ...


class ParallelGenerator(Protocol):
    def generate(
        self,
        *,
        engine: Any,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        presence_penalty: float,
        repetition_penalty: float,
        logits_processor: Any = None,
    ) -> Any: ...


@dataclass(slots=True)
class DraftCandidateBundle:
    tokens: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    source: str = ""


@dataclass(slots=True)
class GenerationEvidence:
    generation_mode: str = GenerationMode.PARALLEL_HYBRID.value
    benchmark_label: str = BenchmarkLabel.PARALLEL_HYBRID.value
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
    prefix_cache_hit_rate: float = 0.0
    scheduler_queue_wait_ms: float = 0.0
    scheduler_iteration_ms: float = 0.0
    kv_fragmentation_ratio: float = 0.0
    quality_guard_triggered: bool = False


@dataclass(slots=True)
class LatentPacket:
    abi_version: int = 2
    packet_version: int = 1
    sequence_id: str = ""
    checkpoint_id: str = ""
    hidden_dimension: int = 0
    tensor: list[list[float]] = field(default_factory=list)
    norm_summary: dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    branch_score: float = 0.0
    stop_policy: str = ""
    capability_digest: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    execution_capsule_id: str = ""
    created_at_ns: int = field(default_factory=time.time_ns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "abi_version": int(self.abi_version),
            "packet_version": int(self.packet_version),
            "sequence_id": str(self.sequence_id),
            "checkpoint_id": str(self.checkpoint_id),
            "hidden_dimension": int(self.hidden_dimension),
            "hidden_dim": int(self.hidden_dimension),
            "tensor": list(self.tensor),
            "norm_summary": dict(self.norm_summary),
            "checksum": str(self.checksum),
            "branch_score": float(self.branch_score),
            "stop_policy": str(self.stop_policy),
            "capability_digest": str(self.capability_digest),
            "delta_watermark": dict(self.delta_watermark),
            "execution_capsule_id": str(self.execution_capsule_id),
            "created_at_ns": int(self.created_at_ns),
        }


@dataclass(slots=True)
class ToolEvidenceRecord:
    evidence_id: str
    tool_name: str = ""
    capsule_path: str = ""
    result_digest: str = ""
    replay: dict[str, Any] = field(default_factory=dict)
    latent_feedback: dict[str, Any] = field(default_factory=dict)
    created_at_ns: int = field(default_factory=time.time_ns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": str(self.evidence_id),
            "tool_name": str(self.tool_name),
            "capsule_path": str(self.capsule_path),
            "result_digest": str(self.result_digest),
            "replay": dict(self.replay),
            "latent_feedback": dict(self.latent_feedback),
            "created_at_ns": int(self.created_at_ns),
        }


@dataclass(slots=True)
class SuspendCheckpoint:
    checkpoint_id: str
    sequence_id: str
    park_level: str
    lifecycle: str
    mode: str
    reason: str
    emitted_text: str
    generated_tokens: int
    sampler_snapshot: dict[str, Any] = field(default_factory=dict)
    rng_snapshot: dict[str, Any] = field(default_factory=dict)
    kv_table_handle: str = ""
    latent_packets: list[dict[str, Any]] = field(default_factory=list)
    evidence_capsules: list[dict[str, Any]] = field(default_factory=list)
    execution_capsule: dict[str, Any] = field(default_factory=dict)
    captured_at_ns: int = field(default_factory=time.time_ns)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": str(self.checkpoint_id),
            "sequence_id": str(self.sequence_id),
            "park_level": str(self.park_level),
            "lifecycle": str(self.lifecycle),
            "mode": str(self.mode),
            "reason": str(self.reason),
            "emitted_text": str(self.emitted_text),
            "generated_tokens": int(self.generated_tokens),
            "sampler_snapshot": dict(self.sampler_snapshot),
            "rng_snapshot": dict(self.rng_snapshot),
            "kv_table_handle": str(self.kv_table_handle),
            "latent_packets": list(self.latent_packets),
            "evidence_capsules": list(self.evidence_capsules),
            "execution_capsule": dict(self.execution_capsule),
            "captured_at_ns": int(self.captured_at_ns),
        }


@dataclass(slots=True)
class SequenceLedger:
    sequence_id: str
    lifecycle: str = SequenceLifecycle.PENDING.value
    mode: str = SequenceExecutionMode.TEXT.value
    park_level: str = ""
    last_text_projection_id: str = ""
    last_tool_run_id: str = ""
    last_evidence_capsule_id: str = ""
    sampler_snapshot: dict[str, Any] = field(default_factory=dict)
    rng_snapshot: dict[str, Any] = field(default_factory=dict)
    kv_table_handle: str = ""
    latent_packets: list[LatentPacket] = field(default_factory=list)
    evidence_capsules: list[ToolEvidenceRecord] = field(default_factory=list)
    checkpoint_id: str = ""
    emitted_text: str = ""
    generated_tokens: int = 0
    resume_count: int = 0
    suspended_reason: str = ""
    lineage_id: str = ""
    prefix_reuse_hit: bool = False
    capability_digest: str = ""
    delta_watermark: dict[str, Any] = field(default_factory=dict)
    execution_capsule_id: str = ""
    execution_capsule_version: int = 2
    reasoning_lane: str = ""
    verified_prefix_hashes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": str(self.sequence_id),
            "lifecycle": str(self.lifecycle),
            "mode": str(self.mode),
            "park_level": str(self.park_level),
            "last_text_projection_id": str(self.last_text_projection_id),
            "last_tool_run_id": str(self.last_tool_run_id),
            "last_evidence_capsule_id": str(self.last_evidence_capsule_id),
            "sampler_snapshot": dict(self.sampler_snapshot),
            "rng_snapshot": dict(self.rng_snapshot),
            "kv_table_handle": str(self.kv_table_handle),
            "latent_packets": [packet.as_dict() for packet in self.latent_packets],
            "evidence_capsules": [
                capsule.as_dict() for capsule in self.evidence_capsules
            ],
            "checkpoint_id": str(self.checkpoint_id),
            "emitted_text": str(self.emitted_text),
            "generated_tokens": int(self.generated_tokens),
            "resume_count": int(self.resume_count),
            "suspended_reason": str(self.suspended_reason),
            "lineage_id": str(self.lineage_id),
            "prefix_reuse_hit": bool(self.prefix_reuse_hit),
            "capability_digest": str(self.capability_digest),
            "delta_watermark": dict(self.delta_watermark),
            "execution_capsule_id": str(self.execution_capsule_id),
            "execution_capsule_version": int(self.execution_capsule_version),
            "reasoning_lane": str(self.reasoning_lane),
            "verified_prefix_hashes": list(self.verified_prefix_hashes),
        }


@dataclass(slots=True)
class ParallelGenerationPlan:
    mode: GenerationMode
    accepted_prefix_tokens: list[int] = field(default_factory=list)
    evidence: GenerationEvidence = field(default_factory=GenerationEvidence)


def _verify_draft_tokens(
    engine: Any,
    *,
    prompt_tokens: list[int],
    draft_tokens: list[int],
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_accept_prob: float,
    logits_processor: Any = None,
) -> tuple[int, list[float]]:
    native_result = _native_verify_draft_tokens(
        engine,
        prompt_tokens=prompt_tokens,
        draft_tokens=draft_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        min_accept_prob=min_accept_prob,
        logits_processor=logits_processor,
    )
    if native_result is not None:
        return native_result.accepted_count, list(native_result.probabilities or [])

    accepted = 0
    probabilities: list[float] = []
    context = [int(token) for token in prompt_tokens]
    native_fast_path = bool(
        getattr(engine, "_native_fast_path", False)
        and getattr(engine, "_disable_logits_processors", False)
    )
    use_processors = bool(logits_processor) and not native_fast_path

    for token in draft_tokens:
        logits = engine._get_logits_for_tokens(context)
        if use_processors:
            logits = engine._apply_logits_processors(
                context,
                logits,
                logits_processor,
            )
        if native_fast_path:
            greedy_token, token_prob = simd_ops.score_token(
                logits,
                token_id=int(token),
                temperature=max(float(temperature), 1.0e-6),
            )
            if 0 <= int(token) < len(logits):
                gated_token, gated_prob = simd_ops.postprocess_and_score(
                    logits,
                    token_id=int(token),
                    suppressed_ids=(),
                    token_history=context,
                    presence_penalty=0.0,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=0,
                    temperature=max(float(temperature), 1.0e-6),
                    eos_token=engine.token_eos(),
                    top_p=float(top_p),
                    top_k=int(top_k),
                    min_p=float(min_p),
                )
                if gated_token != int(token) and gated_prob < float(min_accept_prob):
                    greedy_token = int(gated_token)
                    token_prob = float(gated_prob)
        else:
            greedy_token, token_prob = simd_ops.postprocess_and_score(
                logits,
                token_id=int(token),
                suppressed_ids=(),
                token_history=context,
                presence_penalty=0.0,
                repetition_penalty=1.0,
                no_repeat_ngram_size=0,
                temperature=max(float(temperature), 1.0e-6),
                eos_token=engine.token_eos(),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
            )
        probabilities.append(token_prob)
        if greedy_token != int(token) and token_prob < float(min_accept_prob):
            break
        accepted += 1
        context.append(int(token))
        if int(token) == int(engine.token_eos()):
            break
    return accepted, probabilities


def _native_graph_handle(engine: Any) -> int | None:
    graph = getattr(engine, "_model_graph", None)
    handle = getattr(graph, "_handle", None)
    if handle:
        return int(handle)
    handle = getattr(engine, "_graph_handle", None)
    if handle:
        return int(handle)
    return None


def _native_verify_draft_tokens(
    engine: Any,
    *,
    prompt_tokens: list[int],
    draft_tokens: list[int],
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    min_accept_prob: float,
    logits_processor: Any = None,
    generated_prefix_count: int = 0,
    presence_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    min_new_tokens_before_eos: int = 0,
    sample_recovery_token: bool = False,
) -> Any | None:
    if not draft_tokens:
        return None
    native_fast_path = bool(
        getattr(engine, "_native_fast_path", False)
        and getattr(engine, "_disable_logits_processors", False)
    )
    if bool(logits_processor) and not native_fast_path:
        return None
    handle = _native_graph_handle(engine)
    if handle is None:
        return None
    profile = getattr(engine, "profile", None)
    vocab_size = int(getattr(profile, "vocab_size", 0) or 0)
    eos_token = int(engine.token_eos())
    if vocab_size <= 0:
        return None
    try:
        return qsg_verify_draft_tokens(
            model_graph_handle=handle,
            prompt_tokens=[int(token) for token in prompt_tokens],
            draft_tokens=[int(token) for token in draft_tokens],
            generated_prefix_count=int(max(0, generated_prefix_count)),
            vocab_size=vocab_size,
            eos_token=eos_token,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(max(0, top_k)),
            min_p=float(min_p),
            presence_penalty=float(presence_penalty),
            repetition_penalty=float(repetition_penalty),
            no_repeat_ngram_size=int(max(0, no_repeat_ngram_size)),
            min_new_tokens_before_eos=int(max(0, min_new_tokens_before_eos)),
            min_accept_probability=float(min_accept_prob),
            sample_recovery_token=bool(sample_recovery_token),
        )
    except Exception:
        return None


class PromptLookupDraftPlanner:
    """Draft continuation tokens by reusing repeated prompt suffixes."""

    def __init__(
        self,
        *,
        min_ngram: int = 2,
        max_ngram: int = 24,
        min_accept_prob: float = 0.18,
    ) -> None:
        self._min_ngram = max(1, int(min_ngram))
        self._max_ngram = max(self._min_ngram, int(max_ngram))
        self._min_accept_prob = float(max(0.0, min(1.0, min_accept_prob)))

    def draft(self, prompt_tokens: list[int], max_draft_tokens: int) -> list[int]:
        tokens = [int(token) for token in prompt_tokens]
        limit = max(0, int(max_draft_tokens))
        if len(tokens) < (self._min_ngram * 2) or limit <= 0:
            return []
        return qsg_prompt_lookup_draft(
            tokens,
            min_ngram=self._min_ngram,
            max_ngram=self._max_ngram,
            max_draft_tokens=limit,
        )

    def verify(
        self,
        engine: Any,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        *,
        temperature: float,
        logits_processor: Any = None,
    ) -> tuple[int, list[float]]:
        return _verify_draft_tokens(
            engine,
            prompt_tokens=list(prompt_tokens),
            draft_tokens=list(draft_tokens),
            temperature=float(temperature),
            min_accept_prob=self._min_accept_prob,
            logits_processor=logits_processor,
        )


class ReplacementDraftPlanner:
    """EAGLE-style native replacement candidate drafts."""

    def __init__(
        self,
        *,
        max_tree_width: int = 4,
        acceptance_floor: float = 0.20,
        max_draft_tokens: int = 6,
    ) -> None:
        self._max_tree_width = max(1, int(max_tree_width))
        self._acceptance_floor = float(max(0.0, min(1.0, acceptance_floor)))
        self._max_draft_tokens = max(1, int(max_draft_tokens))

    def draft(
        self,
        engine: Any,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[int]:
        limit = min(max(0, int(max_new_tokens)), self._max_draft_tokens)
        if limit <= 0 or not prompt_tokens:
            return []
        logits = np.asarray(
            engine._get_logits_for_tokens(list(prompt_tokens)), dtype=np.float32
        )
        if logits.ndim != 1 or logits.size <= 1:
            return []
        seed = (
            (len(prompt_tokens) << 32)
            ^ max(0, int(max_new_tokens))
            ^ (int(time.time_ns()) & 0xFFFFFFFF)
        )
        draft_tokens, _ = qsg_eagle_replacement_draft(
            logits,
            logits,
            draft_tokens=limit,
            temperature=float(max(temperature, 1.0e-6)),
            max_tree_width=(
                max(1, int(top_k)) if int(top_k) > 0 else self._max_tree_width
            ),
            acceptance_threshold=self._acceptance_floor,
            seed=int(seed),
        )
        return [int(token) for token in draft_tokens]

    def verify(
        self,
        engine: Any,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        *,
        temperature: float,
        logits_processor: Any = None,
    ) -> tuple[int, list[float]]:
        return _verify_draft_tokens(
            engine,
            prompt_tokens=list(prompt_tokens),
            draft_tokens=list(draft_tokens),
            temperature=float(temperature),
            min_accept_prob=self._acceptance_floor,
            logits_processor=logits_processor,
        )


class ModelHeadDraftPlanner:
    """Native Medusa/Hydra draft heads projected from graph hidden state."""

    def __init__(self, *, min_accept_prob: float = 0.20) -> None:
        self._min_accept_prob = float(max(0.0, min(1.0, min_accept_prob)))

    def draft_bundle(
        self,
        engine: Any,
        *,
        head_type: str,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> DraftCandidateBundle:
        bundle_fn = getattr(engine, "_draft_model_head_bundle", None)
        if callable(bundle_fn):
            bundle = bundle_fn(
                head_type=str(head_type),
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
            )
            if isinstance(bundle, DraftCandidateBundle) and (
                bundle.tokens
                or bundle.probabilities
                or not callable(engine.__dict__.get("_draft_model_head_candidates"))
            ):
                return bundle
        draft_fn = getattr(engine, "_draft_model_head_candidates", None)
        if not callable(draft_fn):
            return DraftCandidateBundle()
        tokens = draft_fn(
            head_type=str(head_type),
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )
        return DraftCandidateBundle(
            tokens=[int(token) for token in list(tokens or ())],
            source=f"{str(head_type or '').strip().lower()}_head_native",
        )

    def draft(
        self,
        engine: Any,
        *,
        head_type: str,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[int]:
        bundle = self.draft_bundle(
            engine,
            head_type=str(head_type),
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
        )
        return [int(token) for token in list(bundle.tokens or ())]

    def verify(
        self,
        engine: Any,
        prompt_tokens: list[int],
        draft_tokens: list[int],
        *,
        temperature: float,
        logits_processor: Any = None,
    ) -> tuple[int, list[float]]:
        return _verify_draft_tokens(
            engine,
            prompt_tokens=list(prompt_tokens),
            draft_tokens=list(draft_tokens),
            temperature=float(temperature),
            min_accept_prob=self._min_accept_prob,
            logits_processor=logits_processor,
        )


class ParallelDecodePlanner:
    """Always-enter planner that chooses the active native generation mode."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        self._prompt_lookup_enabled = bool(
            getattr(engine, "_parallel_prompt_lookup_enabled", True)
        )
        self._medusa_head_enabled = bool(getattr(engine, "_medusa_head_enabled", False))
        self._hydra_head_enabled = bool(getattr(engine, "_hydra_head_enabled", False))
        self._jacobi_lookahead_enabled = bool(
            getattr(engine, "_parallel_jacobi_lookahead_enabled", True)
        )
        self._ssd_bridge_enabled = bool(
            getattr(engine, "_parallel_ssd_bridge_enabled", True)
        )
        self._replacement_enabled = bool(
            getattr(engine, "_parallel_replacement_enabled", True)
        )
        self._ar_recovery_enabled = bool(
            getattr(engine, "_parallel_ar_recovery_enabled", True)
        )
        self._prompt_lookup = PromptLookupDraftPlanner(
            min_ngram=int(getattr(engine, "_parallel_prompt_lookup_min_ngram", 2)),
            max_ngram=int(getattr(engine, "_parallel_prompt_lookup_max_ngram", 24)),
            min_accept_prob=float(
                getattr(engine, "_parallel_prompt_lookup_accept_prob", 0.18)
            ),
        )
        self._replacement = ReplacementDraftPlanner(
            max_tree_width=int(
                getattr(engine, "_parallel_replacement_max_tree_width", 4)
            ),
            acceptance_floor=float(
                getattr(engine, "_parallel_replacement_acceptance_floor", 0.20)
            ),
            max_draft_tokens=int(
                getattr(engine, "_parallel_replacement_max_draft_tokens", 6)
            ),
        )
        self._medusa_head = ModelHeadDraftPlanner(
            min_accept_prob=float(
                getattr(engine, "_medusa_head_acceptance_floor", 0.20)
            ),
        )
        self._hydra_head = ModelHeadDraftPlanner(
            min_accept_prob=float(
                getattr(engine, "_hydra_head_acceptance_floor", 0.22)
            ),
        )
        self._frontier_policy = SpeculativeFrontierPolicy(
            prompt_lookup_window=int(
                getattr(engine, "_parallel_prompt_lookup_window", 6)
            ),
            max_frontier_width=max(
                int(getattr(engine, "_parallel_prompt_lookup_window", 6)),
                int(getattr(engine, "_medusa_head_max_draft_tokens", 4)),
                int(getattr(engine, "_hydra_head_max_draft_tokens", 4)),
                int(getattr(engine, "_parallel_replacement_max_draft_tokens", 6)),
            ),
            verify_depth_cap=max(
                1,
                int(getattr(engine, "_parallel_prompt_lookup_window", 6)),
            ),
            temperature_ceiling=float(
                getattr(engine, "_parallel_decode_temp_max", 0.85)
            ),
        )
        setattr(engine, "_frontier_policy", self._frontier_policy.as_dict())

    def _populate_shared_evidence(self, evidence: GenerationEvidence) -> None:
        prompt_cache_hits = int(getattr(self._engine, "_prefix_cache_hits", 0))
        prompt_cache_misses = int(getattr(self._engine, "_prefix_cache_misses", 0))
        lookups = max(0, prompt_cache_hits + prompt_cache_misses)
        evidence.prefix_cache_hit_rate = (
            float(prompt_cache_hits) / float(lookups) if lookups > 0 else 0.0
        )
        evidence.scheduler_queue_wait_ms = float(
            getattr(self._engine, "_scheduler_queue_wait_ms", 0.0)
        )
        evidence.scheduler_iteration_ms = float(
            getattr(self._engine, "_scheduler_iteration_ms", 0.0)
        )
        evidence.kv_fragmentation_ratio = float(
            getattr(self._engine, "_scheduler_kv_fragmentation_ratio", 0.0)
        )

    def _should_use_block_diffusion(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> bool:
        if not bool(getattr(self._engine, "_block_diffusion_enabled", False)):
            return False
        if not bool(getattr(self._engine, "_block_diffusion_native_ready", False)):
            return False
        if bool(getattr(self._engine, "_block_diffusion_force", False)):
            return True
        if temperature <= 0.0:
            return False
        min_new_tokens = int(
            getattr(self._engine, "_block_diffusion_min_new_tokens", 96)
        )
        min_prompt_tokens = int(
            getattr(self._engine, "_block_diffusion_min_prompt_tokens", 256)
        )
        return (
            int(max_new_tokens) >= min_new_tokens
            and len(prompt_tokens) >= min_prompt_tokens
        )

    def _should_use_masked_diffusion(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> bool:
        if not bool(getattr(self._engine, "_masked_diffusion_enabled", False)):
            return False
        if not bool(getattr(self._engine, "_masked_diffusion_native_ready", False)):
            return False
        if bool(getattr(self._engine, "_masked_diffusion_force", False)):
            return True
        if temperature <= 0.0:
            return False
        min_new_tokens = int(
            getattr(
                self._engine,
                "_block_diffusion_min_new_tokens",
                96,
            )
        )
        min_prompt_tokens = int(
            getattr(
                self._engine,
                "_block_diffusion_min_prompt_tokens",
                256,
            )
        )
        return (
            int(max_new_tokens) >= min_new_tokens
            and len(prompt_tokens) >= min_prompt_tokens
        )

    def _jacobi_lookahead_draft(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        logits_processor: Any,
    ) -> list[int]:
        jacobi = getattr(self._engine, "_jacobi", None)
        decode = getattr(jacobi, "decode", None)
        if not callable(decode):
            return []
        width = max(1, int(getattr(jacobi, "width", 4)))
        result = decode(
            engine=self._engine,
            prompt_tokens=list(prompt_tokens),
            max_tokens=max(1, min(int(max_new_tokens), width)),
            temperature=float(temperature),
            logits_processor=logits_processor,
        )
        tokens = list(getattr(result, "tokens", ()))
        return [int(token) for token in tokens]

    def _supports_parallel_decode(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
    ) -> bool:
        gate = getattr(self._engine, "_should_parallel_decode", None)
        if not callable(gate):
            return False
        return bool(gate(list(prompt_tokens), int(max_new_tokens), float(temperature)))

    def plan(
        self,
        *,
        prompt_tokens: list[int],
        max_new_tokens: int,
        temperature: float,
        logits_processor: Any = None,
    ) -> ParallelGenerationPlan:
        policy = self._frontier_policy
        evidence = GenerationEvidence(
            generation_mode=GenerationMode.PARALLEL_HYBRID.value,
            benchmark_label=BenchmarkLabel.PARALLEL_HYBRID.value,
        )
        self._populate_shared_evidence(evidence)

        def _record_frontier_decision(reason: str, mode: GenerationMode) -> None:
            setattr(
                self._engine,
                "_last_frontier_decision",
                policy.decision(
                    selected_mode=mode.value,
                    reason=reason,
                    prompt_tokens=len(prompt_tokens),
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    accepted_parallel_tokens=int(evidence.accepted_parallel_tokens),
                    rejected_parallel_tokens=int(evidence.rejected_parallel_tokens),
                    draft_frontier_width=int(evidence.draft_frontier_width),
                    verify_depth=int(evidence.verify_depth),
                ).as_dict(),
            )

        if max_new_tokens <= 0:
            evidence.generation_mode = GenerationMode.AR_VERIFY.value
            evidence.benchmark_label = BenchmarkLabel.AR_BASELINE.value
            _record_frontier_decision(
                "no_generation_requested", GenerationMode.AR_VERIFY
            )
            return ParallelGenerationPlan(
                mode=GenerationMode.AR_VERIFY, evidence=evidence
            )

        if self._should_use_masked_diffusion(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            evidence.generation_mode = GenerationMode.MASKED_DIFFUSION.value
            evidence.benchmark_label = BenchmarkLabel.MASKED_DIFFUSION_CANDIDATE.value
            _record_frontier_decision(
                "masked_diffusion_heuristic",
                GenerationMode.MASKED_DIFFUSION,
            )
            return ParallelGenerationPlan(
                mode=GenerationMode.MASKED_DIFFUSION,
                evidence=evidence,
            )

        if self._should_use_block_diffusion(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ):
            evidence.generation_mode = GenerationMode.BLOCK_DIFFUSION.value
            evidence.benchmark_label = BenchmarkLabel.BLOCK_DIFFUSION_CANDIDATE.value
            _record_frontier_decision(
                "block_diffusion_heuristic",
                GenerationMode.BLOCK_DIFFUSION,
            )
            return ParallelGenerationPlan(
                mode=GenerationMode.BLOCK_DIFFUSION,
                evidence=evidence,
            )

        drafted: list[int] = []
        prompt_lookup_attempted = False
        if self._prompt_lookup_enabled:
            prompt_lookup_limit = int(policy.prompt_lookup_limit(int(max_new_tokens)))
            if prompt_lookup_limit > 0:
                prompt_lookup_attempted = True
                verify_started = time.perf_counter()
                drafted = self._prompt_lookup.draft(
                    prompt_tokens=list(prompt_tokens),
                    max_draft_tokens=prompt_lookup_limit,
                )
                evidence.draft_frontier_width = len(drafted)
                if drafted:
                    accepted, probs = self._prompt_lookup.verify(
                        self._engine,
                        prompt_tokens=list(prompt_tokens),
                        draft_tokens=drafted,
                        temperature=float(temperature),
                        logits_processor=logits_processor,
                    )
                    evidence.accepted_parallel_tokens = int(accepted)
                    evidence.rejected_parallel_tokens = max(
                        0, len(drafted) - int(accepted)
                    )
                    evidence.verify_depth = len(probs)
                    evidence.parallel_step_latency_ms = (
                        time.perf_counter() - verify_started
                    ) * 1000.0
                    if accepted > 0:
                        evidence.generation_mode = GenerationMode.PROMPT_LOOKUP.value
                        evidence.benchmark_label = BenchmarkLabel.PROMPT_LOOKUP.value
                        _record_frontier_decision(
                            "verified_prompt_lookup_prefix",
                            GenerationMode.PROMPT_LOOKUP,
                        )
                        return ParallelGenerationPlan(
                            mode=GenerationMode.PROMPT_LOOKUP,
                            accepted_prefix_tokens=list(drafted[:accepted]),
                            evidence=evidence,
                        )

        parallel_decode_ok = self._supports_parallel_decode(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        if self._medusa_head_enabled and parallel_decode_ok:
            medusa_started = time.perf_counter()
            medusa_bundle = self._medusa_head.draft_bundle(
                self._engine,
                head_type="medusa",
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=min(
                    int(max_new_tokens),
                    int(getattr(self._engine, "_medusa_head_max_draft_tokens", 4)),
                ),
                temperature=float(temperature),
                top_k=int(
                    getattr(
                        self._engine, "_medusa_head_top_k", max(1, int(max_new_tokens))
                    )
                ),
            )
            medusa_tokens = list(medusa_bundle.tokens)
            evidence.parallel_step_latency_ms = max(
                evidence.parallel_step_latency_ms,
                (time.perf_counter() - medusa_started) * 1000.0,
            )
            if medusa_tokens:
                accepted, probs = self._medusa_head.verify(
                    self._engine,
                    prompt_tokens=list(prompt_tokens),
                    draft_tokens=list(medusa_tokens),
                    temperature=float(temperature),
                    logits_processor=logits_processor,
                )
                evidence.accepted_parallel_tokens = max(
                    int(evidence.accepted_parallel_tokens),
                    int(accepted),
                )
                evidence.rejected_parallel_tokens = max(
                    int(evidence.rejected_parallel_tokens),
                    max(0, len(medusa_tokens) - int(accepted)),
                )
                evidence.verify_depth = max(int(evidence.verify_depth), len(probs))
                evidence.draft_frontier_width = max(
                    int(evidence.draft_frontier_width),
                    len(medusa_tokens),
                )
                evidence.proposed_parallel_tokens = max(
                    int(evidence.proposed_parallel_tokens),
                    len(medusa_tokens),
                )
                if medusa_bundle.probabilities:
                    evidence.draft_confidence_mean = float(
                        sum(float(value) for value in medusa_bundle.probabilities)
                        / len(medusa_bundle.probabilities)
                    )
                    evidence.draft_confidence_min = float(
                        min(float(value) for value in medusa_bundle.probabilities)
                    )
                evidence.draft_source = str(
                    medusa_bundle.source or "medusa_head_native"
                )
                if accepted > 0:
                    evidence.generation_mode = GenerationMode.MEDUSA_HEAD.value
                    evidence.benchmark_label = (
                        BenchmarkLabel.MEDUSA_HEAD_CANDIDATE.value
                    )
                    _record_frontier_decision(
                        "medusa_head_verified",
                        GenerationMode.MEDUSA_HEAD,
                    )
                    return ParallelGenerationPlan(
                        mode=GenerationMode.MEDUSA_HEAD,
                        accepted_prefix_tokens=list(medusa_tokens[:accepted]),
                        evidence=evidence,
                    )

        if self._hydra_head_enabled and parallel_decode_ok:
            hydra_started = time.perf_counter()
            hydra_bundle = self._hydra_head.draft_bundle(
                self._engine,
                head_type="hydra",
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=min(
                    int(max_new_tokens),
                    int(getattr(self._engine, "_hydra_head_max_draft_tokens", 4)),
                ),
                temperature=float(temperature),
                top_k=int(
                    getattr(
                        self._engine, "_hydra_head_top_k", max(1, int(max_new_tokens))
                    )
                ),
            )
            hydra_tokens = list(hydra_bundle.tokens)
            evidence.parallel_step_latency_ms = max(
                evidence.parallel_step_latency_ms,
                (time.perf_counter() - hydra_started) * 1000.0,
            )
            if hydra_tokens:
                accepted, probs = self._hydra_head.verify(
                    self._engine,
                    prompt_tokens=list(prompt_tokens),
                    draft_tokens=list(hydra_tokens),
                    temperature=float(temperature),
                    logits_processor=logits_processor,
                )
                evidence.accepted_parallel_tokens = max(
                    int(evidence.accepted_parallel_tokens),
                    int(accepted),
                )
                evidence.rejected_parallel_tokens = max(
                    int(evidence.rejected_parallel_tokens),
                    max(0, len(hydra_tokens) - int(accepted)),
                )
                evidence.verify_depth = max(int(evidence.verify_depth), len(probs))
                evidence.draft_frontier_width = max(
                    int(evidence.draft_frontier_width),
                    len(hydra_tokens),
                )
                evidence.proposed_parallel_tokens = max(
                    int(evidence.proposed_parallel_tokens),
                    len(hydra_tokens),
                )
                if hydra_bundle.probabilities:
                    evidence.draft_confidence_mean = float(
                        sum(float(value) for value in hydra_bundle.probabilities)
                        / len(hydra_bundle.probabilities)
                    )
                    evidence.draft_confidence_min = float(
                        min(float(value) for value in hydra_bundle.probabilities)
                    )
                evidence.draft_source = str(hydra_bundle.source or "hydra_head_native")
                if accepted > 0:
                    evidence.generation_mode = GenerationMode.HYDRA_HEAD.value
                    evidence.benchmark_label = BenchmarkLabel.HYDRA_HEAD_CANDIDATE.value
                    _record_frontier_decision(
                        "hydra_head_verified",
                        GenerationMode.HYDRA_HEAD,
                    )
                    return ParallelGenerationPlan(
                        mode=GenerationMode.HYDRA_HEAD,
                        accepted_prefix_tokens=list(hydra_tokens[:accepted]),
                        evidence=evidence,
                    )

        if self._jacobi_lookahead_enabled and parallel_decode_ok:
            jacobi_started = time.perf_counter()
            jacobi_tokens = self._jacobi_lookahead_draft(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                logits_processor=logits_processor,
            )
            evidence.parallel_step_latency_ms = max(
                evidence.parallel_step_latency_ms,
                (time.perf_counter() - jacobi_started) * 1000.0,
            )
            if jacobi_tokens:
                evidence.accepted_parallel_tokens = max(
                    int(evidence.accepted_parallel_tokens),
                    len(jacobi_tokens),
                )
                evidence.rejected_parallel_tokens = max(
                    int(evidence.rejected_parallel_tokens),
                    0,
                )
                evidence.draft_frontier_width = max(
                    int(evidence.draft_frontier_width),
                    len(jacobi_tokens),
                )
                evidence.verify_depth = max(
                    int(evidence.verify_depth),
                    len(jacobi_tokens),
                )
                evidence.benchmark_label = BenchmarkLabel.PARALLEL_HYBRID.value
                _record_frontier_decision(
                    "jacobi_lookahead_window",
                    GenerationMode.PARALLEL_HYBRID,
                )
                return ParallelGenerationPlan(
                    mode=GenerationMode.PARALLEL_HYBRID,
                    accepted_prefix_tokens=list(jacobi_tokens),
                    evidence=evidence,
                )

        if (
            self._replacement_enabled
            and parallel_decode_ok
            and int(max_new_tokens) > 0
            and float(temperature) > 0.0
        ):
            replacement_started = time.perf_counter()
            replacement_tokens = self._replacement.draft(
                self._engine,
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(getattr(self._engine, "_parallel_replacement_top_k", 8)),
            )
            evidence.parallel_step_latency_ms = max(
                evidence.parallel_step_latency_ms,
                (time.perf_counter() - replacement_started) * 1000.0,
            )
            if replacement_tokens:
                accepted, probs = self._replacement.verify(
                    self._engine,
                    prompt_tokens=list(prompt_tokens),
                    draft_tokens=list(replacement_tokens),
                    temperature=float(temperature),
                    logits_processor=logits_processor,
                )
                evidence.accepted_parallel_tokens = max(
                    int(evidence.accepted_parallel_tokens),
                    int(accepted),
                )
                evidence.rejected_parallel_tokens = max(
                    int(evidence.rejected_parallel_tokens),
                    max(0, len(replacement_tokens) - int(accepted)),
                )
                evidence.verify_depth = max(int(evidence.verify_depth), len(probs))
                evidence.draft_frontier_width = max(
                    int(evidence.draft_frontier_width),
                    len(replacement_tokens),
                )
                if accepted > 0:
                    evidence.generation_mode = GenerationMode.REPLACEMENT.value
                    evidence.benchmark_label = (
                        BenchmarkLabel.REPLACEMENT_CANDIDATE.value
                    )
                    _record_frontier_decision(
                        "replacement_frontier_verified",
                        GenerationMode.REPLACEMENT,
                    )
                    return ParallelGenerationPlan(
                        mode=GenerationMode.REPLACEMENT,
                        accepted_prefix_tokens=list(replacement_tokens[:accepted]),
                        evidence=evidence,
                    )

        if parallel_decode_ok:
            evidence.generation_mode = GenerationMode.PARALLEL_HYBRID.value
            evidence.benchmark_label = BenchmarkLabel.PARALLEL_HYBRID.value
            if drafted:
                evidence.rejected_parallel_tokens = max(
                    int(evidence.rejected_parallel_tokens),
                    len(drafted),
                )
                evidence.verify_depth = max(int(evidence.verify_depth), len(drafted))
            _record_frontier_decision(
                "parallel_decode_available",
                GenerationMode.PARALLEL_HYBRID,
            )
            return ParallelGenerationPlan(
                mode=GenerationMode.PARALLEL_HYBRID,
                evidence=evidence,
            )

        if (
            self._ssd_bridge_enabled
            and getattr(self._engine, "_ssm_spec_decoder", None) is not None
            and int(max_new_tokens) > 0
            and float(temperature) > 0.0
        ):
            evidence.generation_mode = GenerationMode.SSD_BRIDGE.value
            evidence.benchmark_label = BenchmarkLabel.SSD_BRIDGE.value
            _record_frontier_decision("ssd_bridge_available", GenerationMode.SSD_BRIDGE)
            return ParallelGenerationPlan(
                mode=GenerationMode.SSD_BRIDGE,
                evidence=evidence,
            )

        if self._ar_recovery_enabled and (prompt_lookup_attempted or bool(drafted)):
            evidence.generation_mode = GenerationMode.AR_RECOVERY.value
            evidence.benchmark_label = BenchmarkLabel.AR_BASELINE.value
            evidence.quality_guard_triggered = bool(
                evidence.rejected_parallel_tokens > 0
            )
            _record_frontier_decision(
                "speculative_recovery_required",
                GenerationMode.AR_RECOVERY,
            )
            return ParallelGenerationPlan(
                mode=GenerationMode.AR_RECOVERY,
                evidence=evidence,
            )

        evidence.generation_mode = GenerationMode.AR_VERIFY.value
        evidence.benchmark_label = BenchmarkLabel.AR_BASELINE.value
        _record_frontier_decision(
            "strict_autoregressive_baseline",
            GenerationMode.AR_VERIFY,
        )
        return ParallelGenerationPlan(
            mode=GenerationMode.AR_VERIFY,
            evidence=evidence,
        )


def _stable_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prefix_hashes(
    prompt_tokens: list[int] | tuple[int, ...],
    *,
    max_hashes: int = 4,
) -> list[str]:
    values = list(prompt_tokens or ())
    if not values:
        return []
    prefixes: list[str] = []
    prefix_lengths = {
        min(len(values), max(1, len(values) // 4)),
        min(len(values), max(1, len(values) // 2)),
        min(len(values), max(1, (3 * len(values)) // 4)),
        len(values),
    }
    for prefix_len in sorted(prefix_lengths):
        prefix = values[:prefix_len]
        prefixes.append(_stable_digest("prefix", prefix_len, *prefix))
        if len(prefixes) >= int(max_hashes):
            break
    return prefixes


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_field(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = float(raw)
        except (TypeError, ValueError):
            continue
    return result


def _normalize_latent_packet(
    sequence_id: str,
    value: Any,
    *,
    checkpoint_id: str = "",
    abi_version: int = 2,
    capability_digest: str = "",
    delta_watermark: dict[str, Any] | None = None,
    execution_capsule_id: str = "",
) -> LatentPacket:
    payload = _mapping(value)
    hidden_dimension = int(
        payload.get("hidden_dimension", payload.get("hidden_dim", 0)) or 0
    )
    branch_score = float(payload.get("branch_score", 0.0) or 0.0)
    stop_policy = _string_field(payload.get("stop_policy", ""))
    checksum = _string_field(payload.get("checksum", ""))
    if not checksum:
        checksum = _stable_digest(
            sequence_id,
            checkpoint_id,
            hidden_dimension,
            branch_score,
            stop_policy,
        )
    normalized_delta = DeltaWatermark.from_dict(
        payload.get("delta_watermark") or delta_watermark
    ).as_dict()
    return LatentPacket(
        abi_version=int(payload.get("abi_version", abi_version) or abi_version),
        packet_version=int(payload.get("packet_version", 1) or 1),
        sequence_id=_string_field(payload.get("sequence_id", sequence_id), sequence_id),
        checkpoint_id=_string_field(payload.get("checkpoint_id", checkpoint_id)),
        hidden_dimension=hidden_dimension,
        tensor=[
            [float(cell) for cell in row]
            for row in list(payload.get("tensor") or [])
            if isinstance(row, (list, tuple))
        ],
        norm_summary=_float_mapping(payload.get("norm_summary", {})),
        checksum=checksum,
        branch_score=branch_score,
        stop_policy=stop_policy,
        capability_digest=_string_field(
            payload.get("capability_digest", capability_digest), capability_digest
        ),
        delta_watermark=normalized_delta,
        execution_capsule_id=_string_field(
            payload.get("execution_capsule_id", execution_capsule_id),
            execution_capsule_id,
        ),
        created_at_ns=int(
            payload.get("created_at_ns", time.time_ns()) or time.time_ns()
        ),
    )


def _normalize_tool_evidence(value: Any) -> ToolEvidenceRecord:
    payload = _mapping(value)
    evidence_id = _string_field(payload.get("evidence_id", "")) or uuid.uuid4().hex
    tool_name = _string_field(payload.get("tool_name", ""))
    capsule_path = _string_field(payload.get("capsule_path", ""))
    result_digest = _string_field(payload.get("result_digest", ""))
    if not result_digest:
        result_digest = _stable_digest(
            evidence_id,
            tool_name,
            capsule_path,
            payload.get("replay", {}),
            payload.get("latent_feedback", {}),
        )
    return ToolEvidenceRecord(
        evidence_id=evidence_id,
        tool_name=tool_name,
        capsule_path=capsule_path,
        result_digest=result_digest,
        replay=_mapping(payload.get("replay", {})),
        latent_feedback=_mapping(payload.get("latent_feedback", {})),
        created_at_ns=int(
            payload.get("created_at_ns", time.time_ns()) or time.time_ns()
        ),
    )


@dataclass(slots=True)
class _NativeRequestState:
    request: QSGRequest
    stream: Iterator[str] | None = None
    chunks: deque[QSGChunk] = field(default_factory=deque)
    completed: bool = False
    cancelled: bool = False
    first_scheduled_ts_ns: int | None = None
    queue_wait_ms: float = 0.0
    generated_tokens: int = 0
    lifecycle: SequenceLifecycle = SequenceLifecycle.PENDING
    mode: SequenceExecutionMode = SequenceExecutionMode.TEXT
    park_level: ParkLevel | None = None
    ledger: SequenceLedger | None = None
    pending_checkpoint: SuspendCheckpoint | None = None
    emitted_text: list[str] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)


class NativeParallelGenerationEngine:
    """Native-owned façade for continuous batching with a native scheduler core."""

    def __init__(
        self,
        *,
        native_engine: Any,
        config: QSGConfig,
        stream_producer: Callable[[QSGRequest], Iterator[str]],
    ) -> None:
        self._native_engine = native_engine
        self._config = config
        self._stream_producer = stream_producer
        self._scheduler_policy = (
            str(getattr(config, "scheduler_policy", "fcfs")).strip().lower()
        )
        if self._scheduler_policy not in {"fcfs", "priority"}:
            self._scheduler_policy = "fcfs"
        self._interleaved_streams = bool(
            getattr(config, "continuous_interleaved_streams", False)
        )
        self._batch_wait_timeout_s = (
            max(1, int(getattr(config, "batch_wait_timeout_ms", 2))) / 1000.0
        )
        self._prefill_chunk_size = max(
            1,
            int(
                getattr(
                    native_engine,
                    "num_ubatch",
                    getattr(config, "max_prefill_rows_per_iteration", 1024),
                )
            ),
        )
        self._native_runtime = self._build_native_runtime()
        self._scheduler = None
        if self._native_runtime is None:
            self._scheduler = NativeQSGScheduler(
                max_active_requests=int(getattr(config, "max_active_requests", 64)),
                max_pending_requests=int(getattr(config, "max_pending_requests", 4096)),
                priority_policy=(self._scheduler_policy == "priority"),
                interleaved_streams=self._interleaved_streams,
            )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._states: dict[str, _NativeRequestState] = {}
        self._shutdown_requested = False
        self._runner_thread: threading.Thread | None = None
        self._start_monotonic = time.perf_counter()
        self._event_store = get_event_store()

        self._ttft_ms: deque[float] = deque(maxlen=2048)
        self._tpot_ms: deque[float] = deque(maxlen=2048)
        self._last_emit_ts: dict[str, float] = {}
        self._generated_token_total = 0
        self._suspend_events = 0
        self._resume_events = 0
        self._lineage_prefix_reuse_enabled = bool(
            getattr(config, "lineage_prefix_reuse_enabled", True)
        )
        self._lineage_prefix_registry: dict[str, set[str]] = {}
        self._prefix_reuse_hits = 0
        self._prefix_reuse_misses = 0

    def _capability_digest(self) -> str:
        return str(getattr(self._config, "capability_digest", "") or "")

    def _delta_watermark(self) -> dict[str, Any]:
        return DeltaWatermark.from_dict(
            getattr(self._config, "delta_watermark", None)
        ).as_dict()

    def _register_lineage_prefixes(
        self,
        request: QSGRequest,
    ) -> tuple[str, list[str], bool]:
        options = _mapping(request.options)
        lineage_id = _string_field(
            options.get("lineage_id", ""), request.request_id or ""
        )
        prefixes = _prefix_hashes(list(request.prompt_tokens or ()))
        if not self._lineage_prefix_reuse_enabled or not lineage_id or not prefixes:
            return lineage_id, prefixes, False
        known = self._lineage_prefix_registry.setdefault(lineage_id, set())
        reuse_hit = any(prefix in known for prefix in prefixes)
        known.update(prefixes)
        if reuse_hit:
            self._prefix_reuse_hits += 1
        else:
            self._prefix_reuse_misses += 1
        return lineage_id, prefixes, reuse_hit

    def _build_execution_capsule_locked(
        self,
        *,
        request_id: str,
        state: _NativeRequestState,
        checkpoint_id: str = "",
    ) -> dict[str, Any]:
        ledger = state.ledger
        hidden_dim = max(
            (
                int(packet.hidden_dimension)
                for packet in (ledger.latent_packets if ledger is not None else [])
            ),
            default=0,
        )
        capability_digest = (
            str(ledger.capability_digest)
            if ledger is not None and ledger.capability_digest
            else self._capability_digest()
        )
        delta_watermark = (
            dict(ledger.delta_watermark)
            if ledger is not None and ledger.delta_watermark
            else self._delta_watermark()
        )
        capsule = RuntimeExecutionCapsule(
            capsule_id=str(checkpoint_id or f"capsule_{uuid.uuid4().hex}"),
            request_id=request_id,
            version=int(getattr(self._config, "execution_capsule_version", 2)),
            model_family="qsg-native",
            capability_digest=capability_digest,
            delta_watermark=delta_watermark,
            generated_tokens=int(state.generated_tokens),
            phase_state=0.0,
            hidden_dim=hidden_dim,
            latent_packet_abi_version=int(
                getattr(self._config, "latent_packet_abi_version", 2)
            ),
            metadata={
                "lifecycle": state.lifecycle.value,
                "mode": state.mode.value,
                "native_runtime_authority": True,
            },
        )
        return capsule.as_dict()

    def _emit_replay_tape(
        self,
        request_id: str,
        *,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        try:
            self._event_store.record_qsg_replay_event(
                request_id=request_id,
                stage=stage,
                payload=payload,
                metadata={
                    "capability_digest": self._capability_digest(),
                    "delta_watermark": self._delta_watermark(),
                },
            )
        except Exception:
            return

    def _build_native_runtime(self) -> NativeQSGRuntime | None:
        model_graph = getattr(self._native_engine, "_model_graph", None)
        graph_handle = int(getattr(model_graph, "_handle", 0) or 0)
        if graph_handle <= 0:
            return None
        try:
            return NativeQSGRuntime(
                model_graph_handle=graph_handle,
                vocab_size=int(getattr(self._native_engine.profile, "vocab_size", 0)),
                eos_token=int(self._native_engine.token_eos()),
                ubatch=int(self._prefill_chunk_size),
                max_active_requests=int(
                    getattr(self._config, "max_active_requests", 64)
                ),
                max_pending_requests=int(
                    getattr(self._config, "max_pending_requests", 4096)
                ),
                priority_policy=(self._scheduler_policy == "priority"),
                interleaved_streams=self._interleaved_streams,
            )
        except Exception:
            return None

    def _scheduler_handle(self) -> NativeQSGScheduler:
        if self._scheduler is None:
            raise RuntimeError("Native scheduler compatibility shim is unavailable")
        return self._scheduler

    def _request_state_bits(self, request_id: str) -> NativeQSGRequestState:
        if self._native_runtime is not None:
            return self._native_runtime.request_state(request_id)
        return self._scheduler_handle().request_state(request_id)

    def _first_scheduled_ns(self, request_id: str) -> int:
        if self._native_runtime is not None:
            return self._native_runtime.first_scheduled_ns(request_id)
        return self._scheduler_handle().first_scheduled_ns(request_id)

    def _mark_request_latent(self, request_id: str, is_latent: bool) -> None:
        if self._native_runtime is not None:
            self._native_runtime.mark_request_latent(request_id, is_latent)
            return
        self._scheduler_handle().mark_request_latent(request_id, is_latent)

    def _mark_request_suspended(self, request_id: str, is_suspended: bool) -> None:
        if self._native_runtime is not None:
            self._native_runtime.mark_request_suspended(request_id, is_suspended)
            return
        self._scheduler_handle().mark_request_suspended(request_id, is_suspended)

    def submit(self, request: QSGRequest) -> str:
        with self._cv:
            if self._shutdown_requested:
                raise RuntimeError("QSGInferenceEngine is shut down")
            request_id = request.request_id or uuid.uuid4().hex
            if request_id in self._states:
                raise ValueError(f"Duplicate request_id '{request_id}'")
            request.request_id = request_id
            lifecycle, mode, park_level = self._initial_request_lifecycle(request)
            lineage_id, prefix_hashes, prefix_reuse_hit = (
                self._register_lineage_prefixes(request)
            )
            state = _NativeRequestState(
                request=request,
                lifecycle=lifecycle,
                mode=mode,
                park_level=park_level,
                ledger=self._build_sequence_ledger(
                    request_id=request_id,
                    request=request,
                    lifecycle=lifecycle,
                    mode=mode,
                    park_level=park_level,
                ),
            )
            if state.ledger is not None:
                state.ledger.lineage_id = lineage_id
                state.ledger.prefix_reuse_hit = bool(prefix_reuse_hit)
                if prefix_hashes:
                    state.ledger.verified_prefix_hashes = list(
                        dict.fromkeys(
                            prefix_hashes + list(state.ledger.verified_prefix_hashes)
                        )
                    )
            latent, suspended = self._extract_request_state_flags(request)
            if self._native_runtime is not None:
                sampling = _mapping(request.sampling)
                self._native_runtime.submit(
                    request_id,
                    priority=int(request.priority),
                    arrival_ts_ns=int(request.arrival_ts_ns),
                    prompt_tokens=list(request.prompt_tokens or ()),
                    max_new_tokens=int(max(0, request.max_new_tokens)),
                    temperature=float(sampling.get("temperature", 0.8) or 0.8),
                    top_p=float(sampling.get("top_p", 1.0) or 1.0),
                    top_k=int(sampling.get("top_k", 0) or 0),
                    min_p=float(sampling.get("min_p", 0.0) or 0.0),
                    presence_penalty=float(
                        sampling.get("presence_penalty", 0.0) or 0.0
                    ),
                    repetition_penalty=float(
                        sampling.get("repetition_penalty", 1.0) or 1.0
                    ),
                    no_repeat_ngram_size=int(
                        getattr(self._native_engine, "_no_repeat_ngram_size", 0)
                    ),
                    min_new_tokens_before_eos=int(
                        getattr(self._native_engine, "_min_new_tokens_before_eos", 0)
                    ),
                    seed=(
                        int(sampling["seed"])
                        if sampling.get("seed") is not None
                        else None
                    ),
                    latent=latent,
                    suspended=suspended,
                )
            else:
                self._scheduler_handle().submit_with_metadata(
                    request_id=request_id,
                    priority=int(request.priority),
                    arrival_ts_ns=int(request.arrival_ts_ns),
                    prompt_token_count=len(request.prompt_tokens or ()),
                    max_new_tokens=int(max(0, request.max_new_tokens)),
                    prefill_chunk_size=self._prefill_chunk_size,
                )
                if latent:
                    self._mark_request_latent(request_id, True)
                if suspended:
                    self._mark_request_suspended(request_id, True)
            self._states[request_id] = state
            self._emit_replay_tape(
                request_id,
                stage="submitted",
                payload={
                    "prompt": request.prompt,
                    "lifecycle": lifecycle.value,
                    "mode": mode.value,
                    "lineage_id": lineage_id,
                    "prefix_reuse_hit": bool(prefix_reuse_hit),
                    "reasoning_lane": state.ledger.reasoning_lane,
                    "verified_prefix_hashes": list(state.ledger.verified_prefix_hashes),
                },
            )
            self._cv.notify_all()
            return request_id

    @staticmethod
    def _extract_request_state_flags(request: QSGRequest) -> tuple[bool, bool]:
        options = request.options
        if not isinstance(options, dict):
            return False, False
        is_latent = (
            bool(options.get("latent", False))
            or bool(options.get("is_latent", False))
            or bool(options.get("latent_mode", False))
        )
        is_suspended = (
            bool(options.get("suspended", False))
            or bool(options.get("is_suspended", False))
            or bool(options.get("parked", False))
        )
        return is_latent, is_suspended

    @staticmethod
    def _initial_request_lifecycle(
        request: QSGRequest,
    ) -> tuple[SequenceLifecycle, SequenceExecutionMode, ParkLevel | None]:
        options = _mapping(request.options)
        latent, suspended = NativeParallelGenerationEngine._extract_request_state_flags(
            request
        )
        mode = SequenceExecutionMode.TEXT
        if latent and options.get("text_projection", False):
            mode = SequenceExecutionMode.HYBRID
        elif latent:
            mode = SequenceExecutionMode.LATENT
        park_level_value = _string_field(options.get("park_level", ""))
        park_level = None
        if park_level_value:
            try:
                park_level = ParkLevel(park_level_value)
            except ValueError:
                park_level = ParkLevel.HOT_PAUSE
        lifecycle = SequenceLifecycle.PENDING
        if suspended:
            lifecycle = (
                SequenceLifecycle.AWAITING_TOOL
                if bool(options.get("awaiting_tool", False))
                else SequenceLifecycle.SUSPENDED
            )
            if park_level is None:
                park_level = ParkLevel.HOT_PAUSE
        return lifecycle, mode, park_level

    def _build_sequence_ledger(
        self,
        *,
        request_id: str,
        request: QSGRequest,
        lifecycle: SequenceLifecycle,
        mode: SequenceExecutionMode,
        park_level: ParkLevel | None,
    ) -> SequenceLedger:
        options = _mapping(request.options)
        checkpoint_id = _string_field(options.get("checkpoint_id", ""))
        reasoning_lane = _string_field(
            options.get("reasoning_lane", ""),
            (
                "tool_wait"
                if lifecycle == SequenceLifecycle.AWAITING_TOOL
                else ("latent" if mode == SequenceExecutionMode.LATENT else "strict")
            ),
        )
        ledger = SequenceLedger(
            sequence_id=request_id,
            lifecycle=lifecycle.value,
            mode=mode.value,
            park_level=park_level.value if park_level is not None else "",
            last_text_projection_id=_string_field(
                options.get("last_text_projection_id", "")
            ),
            last_tool_run_id=_string_field(options.get("last_tool_run_id", "")),
            last_evidence_capsule_id=_string_field(
                options.get("last_evidence_capsule_id", "")
            ),
            sampler_snapshot=_mapping(options.get("sampler_snapshot", {})),
            rng_snapshot=_mapping(options.get("rng_snapshot", {})),
            kv_table_handle=_string_field(options.get("kv_table_handle", "")),
            checkpoint_id=checkpoint_id,
            resume_count=int(options.get("resume_count", 0) or 0),
            suspended_reason=_string_field(options.get("suspended_reason", "")),
            lineage_id=_string_field(options.get("lineage_id", ""), request_id),
            capability_digest=_string_field(
                options.get("capability_digest", self._capability_digest()),
                self._capability_digest(),
            ),
            delta_watermark=DeltaWatermark.from_dict(
                options.get("delta_watermark") or self._delta_watermark()
            ).as_dict(),
            execution_capsule_id=_string_field(
                options.get("execution_capsule_id", checkpoint_id)
            ),
            execution_capsule_version=int(
                options.get(
                    "execution_capsule_version",
                    getattr(self._config, "execution_capsule_version", 2),
                )
                or getattr(self._config, "execution_capsule_version", 2)
            ),
            reasoning_lane=reasoning_lane,
            verified_prefix_hashes=_prefix_hashes(
                list(request.prompt_tokens or ()),
            ),
        )
        for raw_packet in list(options.get("latent_packets", ()) or ()):
            ledger.latent_packets.append(
                _normalize_latent_packet(
                    request_id,
                    raw_packet,
                    checkpoint_id=checkpoint_id,
                    abi_version=int(
                        getattr(self._config, "latent_packet_abi_version", 2)
                    ),
                    capability_digest=ledger.capability_digest,
                    delta_watermark=ledger.delta_watermark,
                    execution_capsule_id=ledger.execution_capsule_id,
                )
            )
        for raw_capsule in list(options.get("evidence_capsules", ()) or ()):
            record = _normalize_tool_evidence(raw_capsule)
            ledger.evidence_capsules.append(record)
            ledger.last_evidence_capsule_id = record.evidence_id
            ledger.last_tool_run_id = record.tool_name or ledger.last_tool_run_id
        return ledger

    @staticmethod
    def _lifecycle_event_for(lifecycle: SequenceLifecycle) -> str:
        if lifecycle == SequenceLifecycle.AWAITING_TOOL:
            return "awaiting_tool"
        if lifecycle == SequenceLifecycle.SUSPENDED:
            return "suspended"
        if lifecycle == SequenceLifecycle.RESUME_PENDING:
            return "resumed"
        return lifecycle.value

    def _capture_checkpoint_locked(
        self,
        request_id: str,
        state: _NativeRequestState,
        *,
        reason: str,
        park_level: ParkLevel,
    ) -> SuspendCheckpoint:
        ledger = state.ledger
        checkpoint_id = uuid.uuid4().hex
        emitted_text = "".join(state.emitted_text)
        checkpoint = SuspendCheckpoint(
            checkpoint_id=checkpoint_id,
            sequence_id=request_id,
            park_level=park_level.value,
            lifecycle=state.lifecycle.value,
            mode=state.mode.value,
            reason=str(reason),
            emitted_text=emitted_text,
            generated_tokens=int(state.generated_tokens),
            sampler_snapshot=dict(
                ledger.sampler_snapshot if ledger is not None else {}
            ),
            rng_snapshot=dict(ledger.rng_snapshot if ledger is not None else {}),
            kv_table_handle=str(ledger.kv_table_handle if ledger is not None else ""),
            latent_packets=[
                packet.as_dict()
                for packet in (ledger.latent_packets if ledger is not None else [])
            ],
            evidence_capsules=[
                capsule.as_dict()
                for capsule in (ledger.evidence_capsules if ledger is not None else [])
            ],
            execution_capsule=self._build_execution_capsule_locked(
                request_id=request_id,
                state=state,
                checkpoint_id=checkpoint_id,
            ),
        )
        if ledger is not None:
            ledger.checkpoint_id = checkpoint_id
            ledger.park_level = park_level.value
            ledger.suspended_reason = str(reason)
            ledger.emitted_text = emitted_text
            ledger.generated_tokens = int(state.generated_tokens)
            ledger.execution_capsule_id = str(
                checkpoint.execution_capsule.get("capsule_id") or checkpoint_id
            )
            ledger.execution_capsule_version = int(
                checkpoint.execution_capsule.get("version")
                or getattr(self._config, "execution_capsule_version", 2)
            )
        return checkpoint

    def record_latent_packet(
        self,
        request_id: str,
        packet: dict[str, Any] | LatentPacket,
    ) -> dict[str, Any]:
        with self._cv:
            state = self._states.get(request_id)
            if state is None:
                raise KeyError(f"Unknown request_id '{request_id}'")
            normalized = (
                packet
                if isinstance(packet, LatentPacket)
                else _normalize_latent_packet(
                    request_id,
                    packet,
                    checkpoint_id=(
                        state.pending_checkpoint.checkpoint_id
                        if state.pending_checkpoint is not None
                        else ""
                    ),
                    abi_version=int(
                        getattr(self._config, "latent_packet_abi_version", 2)
                    ),
                    capability_digest=(
                        state.ledger.capability_digest
                        if state.ledger is not None
                        else ""
                    ),
                    delta_watermark=(
                        state.ledger.delta_watermark if state.ledger is not None else {}
                    ),
                    execution_capsule_id=(
                        (
                            state.pending_checkpoint.execution_capsule.get("capsule_id")
                            if state.pending_checkpoint is not None
                            else ""
                        )
                        or (
                            state.ledger.execution_capsule_id
                            if state.ledger is not None
                            else ""
                        )
                    ),
                )
            )
            if state.ledger is None:
                raise RuntimeError("Sequence ledger is unavailable")
            state.ledger.latent_packets.append(normalized)
            state.ledger.mode = SequenceExecutionMode.LATENT.value
            if normalized.capability_digest:
                state.ledger.capability_digest = normalized.capability_digest
            if normalized.delta_watermark:
                state.ledger.delta_watermark = dict(normalized.delta_watermark)
            if normalized.execution_capsule_id:
                state.ledger.execution_capsule_id = normalized.execution_capsule_id
            state.mode = SequenceExecutionMode.LATENT
            self._mark_request_latent(request_id, True)
            self._emit_replay_tape(
                request_id,
                stage="latent_packet_recorded",
                payload=normalized.as_dict(),
            )
            return normalized.as_dict()

    def record_tool_evidence(
        self,
        request_id: str,
        evidence: dict[str, Any] | ToolEvidenceRecord,
    ) -> dict[str, Any]:
        with self._cv:
            state = self._states.get(request_id)
            if state is None:
                raise KeyError(f"Unknown request_id '{request_id}'")
            normalized = (
                evidence
                if isinstance(evidence, ToolEvidenceRecord)
                else _normalize_tool_evidence(evidence)
            )
            if state.ledger is None:
                raise RuntimeError("Sequence ledger is unavailable")
            state.ledger.evidence_capsules.append(normalized)
            state.ledger.last_evidence_capsule_id = normalized.evidence_id
            state.ledger.last_tool_run_id = (
                normalized.tool_name or state.ledger.last_tool_run_id
            )
            state.chunks.append(
                QSGChunk(
                    request_id=request_id,
                    event="tool_result",
                    metadata=normalized.as_dict(),
                )
            )
            self._cv.notify_all()
            return normalized.as_dict()

    def suspend_request(
        self,
        request_id: str,
        *,
        reason: str = "manual_suspend",
        park_level: ParkLevel | str = ParkLevel.HOT_PAUSE,
        awaiting_tool: bool = False,
    ) -> dict[str, Any]:
        with self._cv:
            state = self._states.get(request_id)
            if state is None or state.completed:
                raise KeyError(f"Unknown active request_id '{request_id}'")
            resolved_park_level = (
                park_level
                if isinstance(park_level, ParkLevel)
                else ParkLevel(_string_field(park_level, ParkLevel.HOT_PAUSE.value))
            )
            state.park_level = resolved_park_level
            state.lifecycle = (
                SequenceLifecycle.AWAITING_TOOL
                if awaiting_tool
                else SequenceLifecycle.SUSPENDED
            )
            checkpoint = self._capture_checkpoint_locked(
                request_id,
                state,
                reason=reason,
                park_level=resolved_park_level,
            )
            state.pending_checkpoint = checkpoint
            if state.ledger is not None:
                state.ledger.lifecycle = state.lifecycle.value
                if awaiting_tool:
                    state.ledger.reasoning_lane = "tool_wait"
            self._mark_request_suspended(request_id, True)
            self._suspend_events += 1
            self._last_emit_ts.pop(request_id, None)
            self._emit_replay_tape(
                request_id,
                stage="checkpoint_captured",
                payload=checkpoint.as_dict(),
            )
            state.chunks.append(
                QSGChunk(
                    request_id=request_id,
                    event=self._lifecycle_event_for(state.lifecycle),
                    metadata=checkpoint.as_dict(),
                )
            )
            self._cv.notify_all()
            return checkpoint.as_dict()

    def resume_request(
        self,
        request_id: str,
        *,
        evidence: dict[str, Any] | ToolEvidenceRecord | None = None,
        latent_packet: dict[str, Any] | LatentPacket | None = None,
        mark_latent: bool | None = None,
    ) -> dict[str, Any]:
        with self._cv:
            state = self._states.get(request_id)
            if state is None or state.completed:
                raise KeyError(f"Unknown active request_id '{request_id}'")
            if evidence is not None:
                self.record_tool_evidence(request_id, evidence)
            if latent_packet is not None:
                self.record_latent_packet(request_id, latent_packet)
            state.lifecycle = SequenceLifecycle.RESUME_PENDING
            state.park_level = None
            if state.ledger is not None:
                state.ledger.lifecycle = state.lifecycle.value
                state.ledger.park_level = ""
                state.ledger.resume_count += 1
                state.ledger.reasoning_lane = (
                    "latent"
                    if bool(mark_latent)
                    else _string_field(
                        _mapping(state.request.options).get("reasoning_lane", ""),
                        "strict",
                    )
                )
            pending_checkpoint = state.pending_checkpoint
            state.pending_checkpoint = None
            self._mark_request_suspended(request_id, False)
            if mark_latent is not None:
                self._mark_request_latent(request_id, bool(mark_latent))
                if state.ledger is not None:
                    state.ledger.mode = (
                        SequenceExecutionMode.LATENT.value
                        if mark_latent
                        else SequenceExecutionMode.TEXT.value
                    )
                state.mode = (
                    SequenceExecutionMode.LATENT
                    if mark_latent
                    else SequenceExecutionMode.TEXT
                )
            self._resume_events += 1
            metadata = {
                "request_id": request_id,
                "checkpoint_id": (
                    pending_checkpoint.checkpoint_id
                    if pending_checkpoint is not None
                    else ""
                ),
                "resume_count": (
                    state.ledger.resume_count if state.ledger is not None else 0
                ),
            }
            state.chunks.append(
                QSGChunk(
                    request_id=request_id,
                    event="resumed",
                    metadata=metadata,
                )
            )
            self._emit_replay_tape(
                request_id,
                stage="resumed",
                payload=metadata,
            )
            self._cv.notify_all()
            return self.request_status(request_id) or metadata

    def request_status(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return None
            scheduler_state = self._request_state_bits(request_id)
            payload = dict(state.ledger.as_dict() if state.ledger is not None else {})
            payload.update(
                {
                    "request_id": request_id,
                    "completed": bool(state.completed),
                    "cancelled": bool(state.cancelled),
                    "generated_tokens": int(state.generated_tokens),
                    "queue_wait_ms": float(state.queue_wait_ms),
                    "scheduler_state": int(scheduler_state),
                    "scheduler_state_flags": [
                        flag.name
                        for flag in NativeQSGRequestState
                        if flag is not NativeQSGRequestState.KNOWN
                        and bool(scheduler_state & flag)
                    ],
                    "pending_checkpoint": (
                        state.pending_checkpoint.as_dict()
                        if state.pending_checkpoint is not None
                        else None
                    ),
                }
            )
            return payload

    def capture_latent_state(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.ledger is None:
                return None
            ledger = state.ledger
            execution_capsule = (
                dict(state.pending_checkpoint.execution_capsule)
                if state.pending_checkpoint is not None
                and state.pending_checkpoint.execution_capsule
                else self._build_execution_capsule_locked(
                    request_id=request_id,
                    state=state,
                    checkpoint_id=ledger.execution_capsule_id,
                )
            )
            latent_packet = (
                ledger.latent_packets[-1]
                if ledger.latent_packets
                else _normalize_latent_packet(
                    request_id,
                    {
                        "hidden_dimension": 0,
                        "generated_tokens": state.generated_tokens,
                    },
                    checkpoint_id=ledger.checkpoint_id,
                    abi_version=int(
                        getattr(self._config, "latent_packet_abi_version", 2)
                    ),
                    capability_digest=ledger.capability_digest,
                    delta_watermark=ledger.delta_watermark,
                    execution_capsule_id=str(
                        execution_capsule.get("capsule_id")
                        or ledger.execution_capsule_id
                    ),
                )
            )
            latent_payload = latent_packet.as_dict()
            latent_payload["execution_capsule_id"] = str(
                latent_payload.get("execution_capsule_id")
                or execution_capsule.get("capsule_id")
                or ""
            )
            return {
                "request_id": request_id,
                "prompt": state.request.prompt,
                "options": dict(state.request.options or {}),
                "tensor": latent_payload.get("tensor") or [[0.0]],
                "generated_tokens": int(state.generated_tokens),
                "phase_state": 0.0,
                "created_at": time.time(),
                "hidden_dim": int(latent_payload.get("hidden_dim") or 0),
                "latent_packet": latent_payload,
                "execution_capsule": execution_capsule,
            }

    def restore_latent_state(
        self,
        package: dict[str, Any],
        *,
        target_request_id: str | None = None,
    ) -> str:
        latent_packet = _mapping(package.get("latent_packet"))
        execution_capsule = _mapping(package.get("execution_capsule"))
        request_id = str(
            target_request_id or package.get("request_id") or uuid.uuid4().hex
        )
        options = dict(_mapping(package.get("options")))
        options["latent"] = True
        options.setdefault(
            "checkpoint_id",
            _string_field(
                execution_capsule.get("capsule_id")
                or latent_packet.get("execution_capsule_id")
            ),
        )
        options.setdefault(
            "execution_capsule_id",
            _string_field(
                execution_capsule.get("capsule_id")
                or latent_packet.get("execution_capsule_id")
            ),
        )
        options.setdefault(
            "execution_capsule_version",
            int(
                execution_capsule.get("version")
                or getattr(self._config, "execution_capsule_version", 2)
            ),
        )
        options.setdefault(
            "capability_digest",
            _string_field(
                latent_packet.get("capability_digest")
                or execution_capsule.get("capability_digest"),
                self._capability_digest(),
            ),
        )
        options.setdefault(
            "delta_watermark",
            DeltaWatermark.from_dict(
                latent_packet.get("delta_watermark")
                or execution_capsule.get("delta_watermark")
                or self._delta_watermark()
            ).as_dict(),
        )
        options.setdefault("latent_packets", [latent_packet])
        options.setdefault("reasoning_lane", "latent")
        request = QSGRequest(
            prompt=str(package.get("prompt") or "latent replay"),
            options=options,
            request_id=request_id,
        )
        restored_id = self.submit(request)
        with self._cv:
            state = self._states[restored_id]
            state.chunks.append(
                QSGChunk(
                    request_id=restored_id,
                    text="",
                    done=False,
                    event="latent_restored",
                    metadata={
                        "source": "almf",
                        "execution_capsule": execution_capsule,
                        "latent_packet_abi_version": int(
                            latent_packet.get(
                                "abi_version",
                                getattr(self._config, "latent_packet_abi_version", 2),
                            )
                        ),
                    },
                )
            )
            self._emit_replay_tape(
                restored_id,
                stage="latent_restored",
                payload={
                    "execution_capsule": execution_capsule,
                    "latent_packet": latent_packet,
                },
            )
        return restored_id

    def _activate_state_locked(self, state: _NativeRequestState) -> bool:
        if state.lifecycle in {
            SequenceLifecycle.SUSPENDED,
            SequenceLifecycle.AWAITING_TOOL,
        }:
            return False
        if state.lifecycle == SequenceLifecycle.PENDING:
            state.lifecycle = SequenceLifecycle.ACTIVE
            if state.ledger is not None:
                state.ledger.lifecycle = state.lifecycle.value
        elif state.lifecycle == SequenceLifecycle.RESUME_PENDING:
            state.lifecycle = SequenceLifecycle.ACTIVE
            if state.ledger is not None:
                state.ledger.lifecycle = state.lifecycle.value
                options = state.request.options
                if not isinstance(options, dict):
                    options = {}
                    state.request.options = options
                options["resume_checkpoint"] = state.ledger.checkpoint_id
        return True

    def _record_emit_locked(
        self,
        *,
        request_id: str,
        state: _NativeRequestState,
        text: str,
        token_estimate: int,
    ) -> None:
        now = time.perf_counter()
        if request_id not in self._last_emit_ts and state.first_scheduled_ts_ns:
            ttft = (time.time_ns() - int(state.first_scheduled_ts_ns)) / 1_000_000.0
            self._ttft_ms.append(float(max(0.0, ttft)))
        if request_id in self._last_emit_ts:
            self._tpot_ms.append(
                float(max(0.0, (now - self._last_emit_ts[request_id]) * 1000.0))
            )
        self._last_emit_ts[request_id] = now
        if text:
            state.emitted_text.append(text)
            state.chunks.append(QSGChunk(request_id=request_id, text=text, done=False))
        state.generated_tokens += int(max(0, token_estimate))
        self._generated_token_total += int(max(0, token_estimate))
        if state.ledger is not None:
            state.ledger.emitted_text = "".join(state.emitted_text)
            state.ledger.generated_tokens = int(state.generated_tokens)

    def poll(self, request_id: str) -> QSGChunk | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None:
                return None
            if state.chunks:
                chunk = state.chunks.popleft()
                if chunk.done:
                    self._states.pop(request_id, None)
                return chunk
            if self._native_runtime is None:
                return None
            runtime_event = self._native_runtime.poll(request_id)
            if runtime_event is None:
                return None
            if state.first_scheduled_ts_ns is None:
                first_scheduled_ns = self._first_scheduled_ns(request_id)
                if first_scheduled_ns > 0:
                    state.first_scheduled_ts_ns = first_scheduled_ns
                    state.queue_wait_ms = (
                        float(first_scheduled_ns - state.request.arrival_ts_ns)
                        / 1_000_000.0
                    )
            scheduler_state = self._request_state_bits(request_id)
            if scheduler_state & NativeQSGRequestState.SUSPENDED:
                if state.lifecycle not in {
                    SequenceLifecycle.SUSPENDED,
                    SequenceLifecycle.AWAITING_TOOL,
                }:
                    state.lifecycle = SequenceLifecycle.SUSPENDED
                    if state.ledger is not None:
                        state.ledger.lifecycle = state.lifecycle.value
                return None
            if runtime_event.token_id is not None and self._activate_state_locked(
                state
            ):
                token_id = int(runtime_event.token_id)
                state.generated_token_ids.append(token_id)
                full_text = str(
                    getattr(self._native_engine, "decode_generated_tokens")(
                        state.generated_token_ids
                    )
                    if callable(
                        getattr(self._native_engine, "decode_generated_tokens", None)
                    )
                    else self._native_engine.detokenize(state.generated_token_ids)
                )
                previous_text = "".join(state.emitted_text)
                text = (
                    full_text[len(previous_text) :]
                    if full_text.startswith(previous_text)
                    else full_text
                )
                self._record_emit_locked(
                    request_id=request_id,
                    state=state,
                    text=text,
                    token_estimate=1,
                )
            if runtime_event.done and not state.completed:
                self._complete_locked(
                    request_id=request_id,
                    state=state,
                    error=runtime_event.error,
                    cancelled=bool(state.cancelled),
                )
            if not state.chunks:
                return None
            chunk = state.chunks.popleft()
            if chunk.done:
                self._states.pop(request_id, None)
            return chunk

    def cancel(self, request_id: str) -> None:
        with self._cv:
            state = self._states.get(request_id)
            if state is None or state.completed:
                return
            state.cancelled = True
            if self._native_runtime is not None:
                self._native_runtime.cancel(request_id)
            else:
                self._scheduler_handle().cancel(request_id)
                self._complete_locked(
                    request_id=request_id,
                    state=state,
                    error="cancelled",
                    cancelled=True,
                )
            self._cv.notify_all()

    def run_forever(self) -> None:
        with self._cv:
            self._runner_thread = threading.current_thread()
        if self._native_runtime is not None:
            try:
                self._native_runtime.run_forever()
            finally:
                with self._cv:
                    self._runner_thread = None
                    self._cv.notify_all()
            return

        while True:
            with self._cv:
                self._scheduler_handle().promote()
                active_ids = self._scheduler_handle().active_ids()
                while not active_ids and not self._shutdown_requested:
                    self._cv.wait(timeout=self._batch_wait_timeout_s)
                    self._scheduler_handle().promote()
                    active_ids = self._scheduler_handle().active_ids()

                has_unfinished = any(
                    not state.completed for state in self._states.values()
                )
                if self._shutdown_requested and not active_ids and not has_unfinished:
                    self._runner_thread = None
                    return

                if not active_ids:
                    continue

            iteration_start = time.perf_counter()
            for request_id in active_ids:
                with self._cv:
                    state = self._states.get(request_id)
                    if state is None or state.completed:
                        continue
                    scheduler_state = self._request_state_bits(request_id)
                    if state.first_scheduled_ts_ns is None:
                        first_scheduled_ns = int(self._first_scheduled_ns(request_id))
                        if first_scheduled_ns > 0:
                            state.first_scheduled_ts_ns = first_scheduled_ns
                            state.queue_wait_ms = (
                                float(first_scheduled_ns - state.request.arrival_ts_ns)
                                / 1_000_000.0
                            )
                    if state.cancelled:
                        self._complete_locked(
                            request_id=request_id,
                            state=state,
                            error="cancelled",
                            cancelled=True,
                        )
                        continue
                    if scheduler_state & NativeQSGRequestState.SUSPENDED:
                        if state.lifecycle not in {
                            SequenceLifecycle.SUSPENDED,
                            SequenceLifecycle.AWAITING_TOOL,
                        }:
                            state.lifecycle = SequenceLifecycle.SUSPENDED
                            if state.ledger is not None:
                                state.ledger.lifecycle = state.lifecycle.value
                        continue
                    if state.lifecycle in {
                        SequenceLifecycle.SUSPENDED,
                        SequenceLifecycle.AWAITING_TOOL,
                    }:
                        continue
                    if state.lifecycle == SequenceLifecycle.PENDING:
                        state.lifecycle = SequenceLifecycle.ACTIVE
                        if state.ledger is not None:
                            state.ledger.lifecycle = state.lifecycle.value
                    elif state.lifecycle == SequenceLifecycle.RESUME_PENDING:
                        state.lifecycle = SequenceLifecycle.ACTIVE
                        if state.ledger is not None:
                            state.ledger.lifecycle = state.lifecycle.value
                            options = state.request.options
                            if not isinstance(options, dict):
                                options = {}
                                state.request.options = options
                            options["resume_checkpoint"] = state.ledger.checkpoint_id
                    stream = state.stream
                    if stream is None:
                        try:
                            stream = iter(self._stream_producer(state.request))
                        except Exception as exc:
                            self._complete_locked(
                                request_id=request_id,
                                state=state,
                                error=str(exc),
                            )
                            continue
                        state.stream = stream

                try:
                    value = next(stream)
                except StopIteration:
                    with self._cv:
                        latest = self._states.get(request_id)
                        if latest is not None and not latest.completed:
                            self._complete_locked(request_id=request_id, state=latest)
                    continue
                except Exception as exc:
                    with self._cv:
                        latest = self._states.get(request_id)
                        if latest is not None and not latest.completed:
                            self._complete_locked(
                                request_id=request_id,
                                state=latest,
                                error=str(exc),
                            )
                    continue

                text = "" if value is None else str(value)
                if not text:
                    continue

                token_estimate = max(1, len(text.split()))
                with self._cv:
                    latest = self._states.get(request_id)
                    if latest is None or latest.completed:
                        continue
                    self._record_emit_locked(
                        request_id=request_id,
                        state=latest,
                        text=text,
                        token_estimate=token_estimate,
                    )
                    self._scheduler_handle().record_decode_emit(
                        request_id, token_estimate
                    )

            elapsed_ms = (time.perf_counter() - iteration_start) * 1000.0
            with self._cv:
                self._scheduler_handle().record_iteration(elapsed_ms)
                self._scheduler_handle().rotate_active()
                self._scheduler_handle().promote()
                self._cv.notify_all()

    def shutdown(self, graceful_timeout_s: float = 1.0) -> None:
        with self._cv:
            self._shutdown_requested = True
            if self._native_runtime is not None:
                self._native_runtime.shutdown()
            self._cv.notify_all()
            runner = self._runner_thread

        if runner is not None and runner is not threading.current_thread():
            runner.join(timeout=max(0.0, float(graceful_timeout_s)))

        with self._cv:
            for request_id, state in list(self._states.items()):
                if state.completed:
                    continue
                self._complete_locked(
                    request_id=request_id,
                    state=state,
                    error="shutdown",
                    cancelled=True,
                )
            self._runner_thread = None
            self._cv.notify_all()
        if self._native_runtime is not None:
            self._native_runtime.close()
        elif self._scheduler is not None:
            self._scheduler.close()

    def _complete_locked(
        self,
        *,
        request_id: str,
        state: _NativeRequestState,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        if state.completed:
            return
        state.completed = True
        if cancelled:
            state.cancelled = True
            state.lifecycle = SequenceLifecycle.CANCELLED
        elif error:
            state.lifecycle = SequenceLifecycle.FAILED
        else:
            state.lifecycle = SequenceLifecycle.COMPLETED
        if state.ledger is not None:
            state.ledger.lifecycle = state.lifecycle.value
            state.ledger.emitted_text = "".join(state.emitted_text)
            state.ledger.generated_tokens = int(state.generated_tokens)
        if self._native_runtime is None:
            self._scheduler_handle().complete(request_id, cancelled=cancelled)
        self._last_emit_ts.pop(request_id, None)
        state.chunks.append(
            QSGChunk(
                request_id=request_id,
                text="",
                done=True,
                error=error,
                event=state.lifecycle.value,
                metadata=(
                    {
                        "lifecycle": state.lifecycle.value,
                        "generated_tokens": int(state.generated_tokens),
                    }
                ),
            )
        )

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = int((len(ordered) - 1) * min(1.0, max(0.0, q)))
        return float(ordered[idx])

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            runtime_metrics = (
                self._native_runtime.metrics()
                if self._native_runtime is not None
                else None
            )
            scheduler = (
                runtime_metrics.scheduler
                if runtime_metrics is not None
                else self._scheduler_handle().metrics()
            )
            uptime_s = max(1.0e-6, time.perf_counter() - self._start_monotonic)
            native_kv_metrics: dict[str, Any] = {}
            native_kv_cache = getattr(self._native_engine, "_native_kv_cache", None)
            if native_kv_cache is not None:
                get_native_kv_metrics = getattr(
                    native_kv_cache, "metrics_snapshot", None
                )
                if callable(get_native_kv_metrics):
                    try:
                        native_kv_metrics = dict(get_native_kv_metrics())
                    except Exception:
                        native_kv_metrics = {}
            decode_tps_per_agent: dict[str, float] = {}
            lifecycle_counts = {lifecycle.value: 0 for lifecycle in SequenceLifecycle}
            mode_counts = {mode.value: 0 for mode in SequenceExecutionMode}
            reasoning_lane_counts: dict[str, int] = {}
            checkpoint_count = 0
            tool_wait_requests = 0
            latent_packet_count = 0
            evidence_capsule_count = 0
            for request_id, state in self._states.items():
                lifecycle_counts[state.lifecycle.value] = (
                    int(lifecycle_counts.get(state.lifecycle.value, 0)) + 1
                )
                mode_counts[state.mode.value] = (
                    int(mode_counts.get(state.mode.value, 0)) + 1
                )
                if state.pending_checkpoint is not None:
                    checkpoint_count += 1
                if state.lifecycle == SequenceLifecycle.AWAITING_TOOL:
                    tool_wait_requests += 1
                if state.ledger is not None:
                    reasoning_lane = str(state.ledger.reasoning_lane or "strict")
                    reasoning_lane_counts[reasoning_lane] = (
                        int(reasoning_lane_counts.get(reasoning_lane, 0)) + 1
                    )
                    latent_packet_count += len(state.ledger.latent_packets)
                    evidence_capsule_count += len(state.ledger.evidence_capsules)
                if state.generated_tokens <= 0:
                    continue
                started_ns = state.first_scheduled_ts_ns or state.request.arrival_ts_ns
                elapsed_s = max(
                    1.0e-6,
                    (time.time_ns() - int(started_ns)) / 1_000_000_000.0,
                )
                decode_tps_per_agent[request_id] = (
                    float(state.generated_tokens) / elapsed_s
                )
            metrics = {
                "scheduler_policy": self._scheduler_policy,
                "execution_mode": (
                    "interleaved" if self._interleaved_streams else "single_stream"
                ),
                "queue_depth": int(scheduler.queue_depth),
                "active_requests": int(scheduler.active_requests),
                "inflight_requests": int(scheduler.inflight_requests),
                "prefill_active_requests": int(scheduler.prefill_active_requests),
                "decode_active_requests": int(scheduler.decode_active_requests),
                "admitted_requests": int(scheduler.admitted_requests),
                "completed_requests": int(scheduler.completed_requests),
                "cancelled_requests": int(scheduler.cancelled_requests),
                "iterations": int(scheduler.iterations),
                "iteration_latency_ms": {
                    "count": int(scheduler.iterations),
                    "last": float(scheduler.iteration_last_ms),
                    "avg": float(scheduler.iteration_avg_ms),
                    "max": float(scheduler.iteration_p95_ms),
                    "p95": float(scheduler.iteration_p95_ms),
                },
                "qsg_queue_depth": int(scheduler.queue_depth),
                "qsg_active_requests": int(scheduler.active_requests),
                "qsg_prefill_active_requests": int(scheduler.prefill_active_requests),
                "qsg_decode_active_requests": int(scheduler.decode_active_requests),
                "qsg_request_admit_rate_rps": float(scheduler.admitted_requests)
                / uptime_s,
                "qsg_request_evict_rate_rps": float(scheduler.evicted_requests)
                / uptime_s,
                "qsg_queue_wait_ms_p50": float(scheduler.queue_wait_p50_ms),
                "qsg_queue_wait_ms_p95": float(scheduler.queue_wait_p95_ms),
                "qsg_queue_wait_ms_p99": float(scheduler.queue_wait_p99_ms),
                "qsg_scheduler_iteration_ms_p50": float(scheduler.iteration_avg_ms),
                "qsg_scheduler_iteration_ms_p95": float(scheduler.iteration_p95_ms),
                "qsg_prefill_request_count": int(scheduler.prefill_request_count),
                "qsg_prefill_tokens_scheduled": int(scheduler.prefill_tokens_scheduled),
                "qsg_decode_tokens_emitted": int(scheduler.decode_tokens_emitted),
                "qsg_chunked_prefill_requests": int(scheduler.chunked_prefill_requests),
                "qsg_chunked_prefill_chunks": int(scheduler.chunked_prefill_chunks),
                "qsg_decode_tps_global": (
                    float(
                        max(
                            self._generated_token_total,
                            int(scheduler.decode_tokens_emitted),
                        )
                    )
                    / uptime_s
                ),
                "qsg_decode_tps_per_agent": decode_tps_per_agent,
                "qsg_ttft_ms_p50": self._percentile(list(self._ttft_ms), 0.50),
                "qsg_ttft_ms_p95": self._percentile(list(self._ttft_ms), 0.95),
                "qsg_tpot_ms_p50": self._percentile(list(self._tpot_ms), 0.50),
                "qsg_tpot_ms_p95": self._percentile(list(self._tpot_ms), 0.95),
                "qsg_state_pages_total": int(
                    native_kv_metrics.get(
                        "resident_page_count",
                        native_kv_metrics.get("pages_total", 0),
                    )
                ),
                "qsg_state_pages_in_use": int(
                    native_kv_metrics.get(
                        "resident_page_count",
                        native_kv_metrics.get("pages_in_use", 0),
                    )
                ),
                "qsg_state_active_page_slots": int(
                    native_kv_metrics.get("active_page_slots", 0)
                ),
                "qsg_state_shared_page_slots": int(
                    native_kv_metrics.get("shared_page_slots", 0)
                ),
                "qsg_state_snapshot_count": int(
                    native_kv_metrics.get("snapshot_count", 0)
                ),
                "qsg_state_fragmentation_ratio": float(
                    native_kv_metrics.get(
                        "fragmentation_ratio",
                        getattr(
                            self._native_engine,
                            "_scheduler_kv_fragmentation_ratio",
                            0.0,
                        ),
                    )
                ),
                "qsg_state_compaction_count": 0,
                "qsg_state_cow_events": int(
                    native_kv_metrics.get("copy_on_write_events", 0)
                ),
                "qsg_state_prefix_share_events": int(
                    native_kv_metrics.get("prefix_share_events", 0)
                ),
                "qsg_state_active_tokens": int(
                    native_kv_metrics.get("active_tokens", 0)
                ),
                "qsg_state_committed_token_capacity": int(
                    native_kv_metrics.get("committed_token_capacity", 0)
                ),
                "qsg_state_page_tokens": int(native_kv_metrics.get("page_tokens", 0)),
                "qsg_state_allocator_failures": 0,
                "qsg_latent_requests": int(scheduler.latent_requests),
                "qsg_suspended_requests": int(scheduler.suspended_requests),
                "qsg_sequence_state_counts": lifecycle_counts,
                "qsg_sequence_mode_counts": mode_counts,
                "qsg_reasoning_lane_counts": reasoning_lane_counts,
                "qsg_sequence_checkpoint_count": int(checkpoint_count),
                "qsg_tool_wait_requests": int(tool_wait_requests),
                "qsg_latent_packet_count": int(latent_packet_count),
                "qsg_evidence_capsule_count": int(evidence_capsule_count),
                "qsg_suspend_events": int(self._suspend_events),
                "qsg_resume_events": int(self._resume_events),
                "qsg_lineage_prefix_registry_size": int(
                    sum(
                        len(prefixes)
                        for prefixes in self._lineage_prefix_registry.values()
                    )
                ),
                "qsg_prefix_reuse_hits": int(self._prefix_reuse_hits),
                "qsg_prefix_reuse_misses": int(self._prefix_reuse_misses),
                "prefix_cache_hit_rate": (
                    float(self._prefix_reuse_hits)
                    / float(
                        max(
                            1,
                            int(self._prefix_reuse_hits)
                            + int(self._prefix_reuse_misses),
                        )
                    )
                ),
                "qsg_coconut_active_paths": 0,
                "qsg_coconut_entropy_mean": 0.0,
                "qsg_phase_confidence_mean": 0.0,
                "qsg_drift_overhead_percent": 0.0,
                "qsg_python_hot_path_calls": (
                    0 if self._native_runtime is not None else int(scheduler.iterations)
                ),
                "qsg_numpy_hot_path_calls": 0,
                "qsg_batched_prefill_token_id_calls": int(
                    scheduler.prefill_request_count
                ),
                "qsg_batched_prefill_token_id_tokens": int(
                    scheduler.prefill_tokens_scheduled
                ),
            }
            if runtime_metrics is not None:
                metrics["qsg_runtime_worker_iterations"] = int(
                    runtime_metrics.worker_iterations
                )
                metrics["qsg_runtime_emitted_events"] = int(
                    runtime_metrics.emitted_events
                )
                metrics["qsg_runtime_prefill_batches"] = int(
                    runtime_metrics.prefill_batches
                )
                metrics["qsg_runtime_decode_steps"] = int(
                    runtime_metrics.runtime_decode_steps
                )
        runtime_status: dict[str, Any] = {}
        get_status = getattr(self._native_engine, "get_runtime_status", None)
        if callable(get_status):
            try:
                runtime_status = dict(get_status())
            except Exception:
                runtime_status = {}
        if runtime_metrics is not None:
            runtime_status["native_runtime_abi_ready"] = True
            runtime_status["python_hot_path_calls"] = 0
            runtime_status["numpy_hot_path_calls"] = 0
            runtime_status["hot_path_numpy_detected"] = False
        generation_mode = str(
            runtime_status.get("generation_mode", GenerationMode.PARALLEL_HYBRID.value)
        )
        runtime_benchmark_label = str(runtime_status.get("benchmark_label", "")).strip()
        prompt_category = str(runtime_status.get("prompt_category", "")).strip()
        temperature_band = str(runtime_status.get("temperature_band", "")).strip()
        scheduler_queue_wait_ms = float(
            runtime_status.get(
                "scheduler_queue_wait_ms",
                metrics.get("qsg_queue_wait_ms_p95", 0.0),
            )
        )
        scheduler_iteration_ms = float(
            runtime_status.get(
                "scheduler_iteration_ms",
                metrics.get("qsg_scheduler_iteration_ms_p95", 0.0),
            )
        )
        kv_fragmentation_ratio = float(
            runtime_status.get(
                "kv_fragmentation_ratio",
                metrics.get("qsg_state_fragmentation_ratio", 0.0),
            )
        )
        runtime_python_hot_path_calls = int(
            runtime_status.get("python_hot_path_calls", 0)
        )
        runtime_numpy_hot_path_calls = int(
            runtime_status.get("numpy_hot_path_calls", 0)
        )
        hot_path_numpy_detected = bool(
            runtime_status.get("hot_path_numpy_detected", False)
        )
        combined_python_hot_path_calls = max(
            int(metrics.get("qsg_python_hot_path_calls", 0)),
            runtime_python_hot_path_calls,
        )
        combined_numpy_hot_path_calls = max(
            int(metrics.get("qsg_numpy_hot_path_calls", 0)),
            runtime_numpy_hot_path_calls,
        )
        hot_path_numpy_detected = bool(
            hot_path_numpy_detected
            or combined_python_hot_path_calls > 0
            or combined_numpy_hot_path_calls > 0
        )
        native_runtime_abi_ready = bool(
            runtime_status.get("native_runtime_abi_ready", False)
            or self._native_runtime is not None
        )
        continuous_runtime_owner = (
            "native_runtime"
            if native_runtime_abi_ready
            else "python_compatibility_shim"
        )
        phase1_blockers = []
        if not native_runtime_abi_ready:
            phase1_blockers.append("missing_native_serve_runtime_abi")
        if self._native_runtime is None and int(scheduler.iterations) > 0:
            phase1_blockers.append("python_scheduler_loop_active")
        if runtime_python_hot_path_calls > 0:
            phase1_blockers.append("python_token_loop_active")
        if runtime_numpy_hot_path_calls > 0:
            phase1_blockers.append("numpy_hot_path_active")
        if self._native_runtime is None and not phase1_blockers:
            phase1_blockers.append("python_stream_producer_active")
        hot_path_proof = dict(runtime_status.get("hot_path_proof", {}))
        hot_path_proof.update(
            {
                "continuous_runtime_owner": continuous_runtime_owner,
                "continuous_runtime_native_abi": (
                    "true" if native_runtime_abi_ready else "false"
                ),
                "continuous_runtime_phase1_ready": (
                    "true" if not phase1_blockers else "false"
                ),
                "python_or_numpy_hot_path": (
                    "detected" if hot_path_numpy_detected else "not_detected"
                ),
                "executed_cpp_only": "true" if not phase1_blockers else "false",
                "python_hot_path_calls": str(combined_python_hot_path_calls),
                "numpy_hot_path_calls": str(combined_numpy_hot_path_calls),
                "phase1_blockers": ",".join(phase1_blockers),
            }
        )
        metrics["qsg_python_hot_path_calls"] = combined_python_hot_path_calls
        metrics["qsg_numpy_hot_path_calls"] = combined_numpy_hot_path_calls
        metrics["hot_path_numpy_detected"] = hot_path_numpy_detected
        metrics["native_runtime_abi_ready"] = native_runtime_abi_ready
        metrics["continuous_runtime_owner"] = continuous_runtime_owner
        metrics["phase1_ready"] = not phase1_blockers
        metrics["phase1_blockers"] = phase1_blockers
        metrics["hot_path_proof"] = hot_path_proof
        metrics.setdefault("generation_mode", generation_mode)
        metrics.setdefault(
            "benchmark_label",
            runtime_benchmark_label or benchmark_label_for_mode(generation_mode).value,
        )
        metrics.setdefault("prompt_category", prompt_category)
        metrics.setdefault("temperature_band", temperature_band)
        metrics.setdefault(
            "supported_benchmark_labels",
            list(runtime_status.get("supported_benchmark_labels", ()))
            or supported_benchmark_labels(),
        )
        metrics.setdefault(
            "parallel_decode_allowed",
            bool(runtime_status.get("parallel_decode_allowed", True)),
        )
        metrics.setdefault("scheduler_queue_wait_ms", scheduler_queue_wait_ms)
        metrics.setdefault("scheduler_iteration_ms", scheduler_iteration_ms)
        metrics.setdefault("kv_fragmentation_ratio", kv_fragmentation_ratio)
        metrics.setdefault(
            "qsg_state_pages_total",
            int(
                runtime_status.get(
                    "qsg_state_pages_total", metrics["qsg_state_pages_total"]
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_pages_in_use",
            int(
                runtime_status.get(
                    "qsg_state_pages_in_use",
                    metrics["qsg_state_pages_in_use"],
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_active_page_slots",
            int(
                runtime_status.get(
                    "qsg_state_active_page_slots",
                    metrics["qsg_state_active_page_slots"],
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_shared_page_slots",
            int(
                runtime_status.get(
                    "qsg_state_shared_page_slots",
                    metrics["qsg_state_shared_page_slots"],
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_snapshot_count",
            int(
                runtime_status.get(
                    "qsg_state_snapshot_count",
                    metrics["qsg_state_snapshot_count"],
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_cow_events",
            int(
                runtime_status.get(
                    "qsg_state_cow_events",
                    metrics["qsg_state_cow_events"],
                )
            ),
        )
        metrics.setdefault(
            "qsg_state_prefix_share_events",
            int(
                runtime_status.get(
                    "qsg_state_prefix_share_events",
                    metrics["qsg_state_prefix_share_events"],
                )
            ),
        )
        metrics.setdefault(
            "accepted_parallel_tokens",
            int(runtime_status.get("accepted_parallel_tokens", 0)),
        )
        metrics.setdefault(
            "rejected_parallel_tokens",
            int(runtime_status.get("rejected_parallel_tokens", 0)),
        )
        metrics.setdefault(
            "proposed_parallel_tokens",
            int(
                runtime_status.get(
                    "proposed_parallel_tokens",
                    int(runtime_status.get("accepted_parallel_tokens", 0))
                    + int(runtime_status.get("rejected_parallel_tokens", 0)),
                )
            ),
        )
        metrics.setdefault(
            "draft_frontier_width",
            int(runtime_status.get("draft_frontier_width", 0)),
        )
        metrics.setdefault("verify_depth", int(runtime_status.get("verify_depth", 0)))
        metrics.setdefault(
            "parallel_step_latency_ms",
            float(runtime_status.get("parallel_step_latency_ms", 0.0)),
        )
        metrics.setdefault(
            "draft_confidence_mean",
            float(runtime_status.get("draft_confidence_mean", 0.0)),
        )
        metrics.setdefault(
            "draft_confidence_min",
            float(runtime_status.get("draft_confidence_min", 0.0)),
        )
        metrics.setdefault("draft_source", str(runtime_status.get("draft_source", "")))
        metrics.setdefault(
            "quality_guard_triggered",
            bool(runtime_status.get("quality_guard_triggered", False)),
        )
        metrics.setdefault(
            "self_spec_native_path",
            bool(runtime_status.get("self_spec_native_path", False)),
        )
        metrics.setdefault(
            "self_spec_policy",
            str(runtime_status.get("self_spec_policy", "")),
        )
        metrics.setdefault(
            "self_spec_exit_layer",
            int(runtime_status.get("self_spec_exit_layer", 0)),
        )
        metrics.setdefault(
            "self_spec_exit_fraction",
            float(runtime_status.get("self_spec_exit_fraction", 0.0)),
        )
        metrics.setdefault(
            "self_spec_draft_tokens",
            int(runtime_status.get("self_spec_draft_tokens", 0)),
        )
        update_scheduler_metrics = getattr(
            self._native_engine,
            "_update_scheduler_metrics_snapshot",
            None,
        )
        if callable(update_scheduler_metrics):
            try:
                update_scheduler_metrics(metrics)
            except Exception:
                pass
        return metrics
