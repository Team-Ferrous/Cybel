from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any, Callable, Iterator

from core.qsg.config import QSGConfig
from core.qsg.runtime_contracts import (
    CapsuleSegmentIndex,
    ControllerDecisionRecord,
    DeltaWatermark,
    ExecutionCapsule,
    LatentPacketABI,
    TypedLatentSegment,
)
from core.qsg.state_pager import QSGStatePager, RowRef


@dataclass(slots=True)
class QSGRequest:
    prompt: str
    options: dict[str, Any] | None = None
    request_id: str | None = None
    prompt_tokens: list[int] | None = None
    max_new_tokens: int = 0
    sampling: dict[str, Any] | None = None
    priority: int = 0
    arrival_ts_ns: int = field(default_factory=time.time_ns)
    submitted_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class QSGChunk:
    request_id: str
    text: str = ""
    done: bool = False
    error: str | None = None
    emitted_at: float = field(default_factory=time.time)
    event: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class ActiveStateHandle:
    slot_id: int
    state_page_chain: list[int] = field(default_factory=list)
    state_row_offsets: list[int] = field(default_factory=list)
    phase_state: float = 0.0
    coconut_state_ref: int = -1
    generated_tokens: int = 0
    finished: bool = False


@dataclass(slots=True)
class _RequestState:
    request: QSGRequest
    stream: Iterator[str] | None = None
    chunks: deque[QSGChunk] = field(default_factory=deque)
    completed: bool = False
    cancelled: bool = False
    first_scheduled_ts_ns: int | None = None
    queue_wait_ms: float = 0.0
    state_handle: ActiveStateHandle | None = None


class QSGInferenceEngine:
    """Event-driven continuous batching scheduler with paged request state."""

    def __init__(
        self,
        *,
        config: QSGConfig,
        stream_producer: Callable[[QSGRequest], Iterator[str]],
    ) -> None:
        self._config = config
        self._stream_producer = stream_producer
        self._scheduler_policy = (
            str(getattr(config, "scheduler_policy", "fcfs")).strip().lower()
        )
        if self._scheduler_policy not in {"fcfs", "priority"}:
            self._scheduler_policy = "fcfs"
        self._max_active_requests = int(getattr(config, "max_active_requests", 64))
        self._max_pending_requests = int(getattr(config, "max_pending_requests", 4096))
        self._batch_wait_timeout_s = (
            max(1, int(getattr(config, "batch_wait_timeout_ms", 2))) / 1000.0
        )
        self._interleaved_streams = bool(
            getattr(config, "continuous_interleaved_streams", False)
        )

        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._states: dict[str, _RequestState] = {}
        self._pending: deque[str] = deque()
        self._active: deque[str] = deque()
        self._shutdown_requested = False
        self._runner_thread: threading.Thread | None = None

        self._pager = QSGStatePager(
            dim=1,
            state_page_rows=int(getattr(config, "state_page_rows", 128)),
            soft_compaction_threshold=float(
                getattr(config, "state_compaction_soft_threshold", 0.18)
            ),
            hard_compaction_threshold=float(
                getattr(config, "state_compaction_hard_threshold", 0.30)
            ),
        )
        self._next_slot_id = 0

        self._start_monotonic = time.perf_counter()
        self._admitted_count = 0
        self._completed_count = 0
        self._cancelled_count = 0
        self._evicted_count = 0
        self._generated_token_total = 0
        self._iteration_count = 0
        self._iteration_latency_ms: deque[float] = deque(maxlen=1024)
        self._queue_wait_ms: deque[float] = deque(maxlen=2048)
        self._ttft_ms: deque[float] = deque(maxlen=2048)
        self._tpot_ms: deque[float] = deque(maxlen=2048)
        self._last_emit_ts: dict[str, float] = {}
        self._latent_capture_count = 0
        self._latent_restore_count = 0
        self._suspended_count = 0

    @staticmethod
    def _matrix_from_tensor(values: Any) -> list[list[float]]:
        if values is None:
            return []
        if isinstance(values, (bytes, bytearray)):
            return []
        rows = list(values)
        if not rows:
            return []
        first = rows[0]
        if isinstance(first, (int, float)):
            return [[float(value) for value in rows]]
        matrix: list[list[float]] = []
        for row in rows:
            matrix.append([float(value) for value in list(row)])
        return matrix

    @staticmethod
    def _any_nonzero(values: list[list[float]]) -> bool:
        for row in values:
            for value in row:
                if value != 0.0:
                    return True
        return False

    @staticmethod
    def _repo_delta_row(
        delta_watermark: DeltaWatermark, hidden_dim: int
    ) -> list[float]:
        if hidden_dim <= 0:
            return []
        signal = [
            float(delta_watermark.logical_clock or 0),
            float(len(delta_watermark.changed_paths)),
            float(delta_watermark.created_at or 0.0) % 997.0,
            float(len(str(delta_watermark.git_head or ""))),
        ]
        row = [0.0] * hidden_dim
        for idx in range(hidden_dim):
            base = signal[idx % len(signal)] if signal else 0.0
            row[idx] = float(base / max(1.0, signal[1] + 1.0))
        return row

    @staticmethod
    def _select_segment_rows(
        tensor: list[list[float]],
        segments: list[dict[str, Any]],
        requested_kinds: list[str] | None,
    ) -> list[list[float]]:
        if not requested_kinds or not segments:
            return tensor
        allowed = {str(kind) for kind in requested_kinds}
        selected: list[list[float]] = []
        for payload in segments:
            segment = TypedLatentSegment.from_dict(payload)
            if segment.segment_kind not in allowed:
                continue
            start = max(0, int(segment.row_start))
            end = max(start, start + int(segment.row_count))
            selected.extend(tensor[start:end])
        return selected or tensor

    def submit(self, request: QSGRequest) -> str:
        with self._cv:
            if self._shutdown_requested:
                raise RuntimeError("QSGInferenceEngine is shut down")
            if len(self._pending) >= self._max_pending_requests:
                raise RuntimeError("QSGInferenceEngine pending queue is full")

            request_id = request.request_id or uuid.uuid4().hex
            if request_id in self._states:
                raise ValueError(f"Duplicate request_id '{request_id}'")

            request.request_id = request_id
            state = _RequestState(request=request)
            self._states[request_id] = state
            self._enqueue_pending_locked(request_id)
            self._admitted_count += 1
            self._cv.notify_all()
            return request_id

    def poll(self, request_id: str) -> QSGChunk | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None or not state.chunks:
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

        while True:
            with self._cv:
                self._promote_pending_locked()
                while not self._active and not self._shutdown_requested:
                    self._cv.wait(timeout=self._batch_wait_timeout_s)
                    self._promote_pending_locked()

                if self._shutdown_requested and not self._active and not self._pending:
                    self._runner_thread = None
                    return

                active_ids = [
                    request_id
                    for request_id in self._active
                    if request_id in self._states
                    and not self._states[request_id].completed
                ]
                if not active_ids:
                    continue
                if not self._interleaved_streams:
                    # The native QSG stream path is not reentrant across multiple
                    # in-flight iterators on one engine instance. Keep a single
                    # request active at a time unless the caller opts into
                    # interleaving explicitly.
                    active_ids = active_ids[:1]

                slot_refs: list[RowRef] = []
                for request_id in active_ids:
                    state = self._states[request_id]
                    if state.state_handle is None:
                        self._attach_state_handle_locked(request_id, state)
                    if (
                        state.state_handle is not None
                        and state.state_handle.state_page_chain
                        and state.state_handle.state_row_offsets
                    ):
                        slot_refs.append(
                            RowRef(
                                page_id=state.state_handle.state_page_chain[0],
                                row_idx=state.state_handle.state_row_offsets[0],
                            )
                        )
                    else:
                        slot_refs.append(RowRef(page_id=0, row_idx=0))

            iteration_start = time.perf_counter()
            if slot_refs and self._pager.metrics_snapshot()["pages_total"] > 0:
                gathered = self._matrix_from_tensor(
                    self._pager.gather_active(slot_refs)
                )
            else:
                gathered = [[0.0] for _ in active_ids]
            deltas = [[0.0] for _ in active_ids]

            for slot_idx, request_id in enumerate(active_ids):
                with self._cv:
                    state = self._states.get(request_id)
                    if state is None or state.completed:
                        continue
                    if state.cancelled:
                        self._complete_locked(
                            request_id=request_id,
                            state=state,
                            error="cancelled",
                            cancelled=True,
                        )
                        continue
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
                            self._cv.notify_all()
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
                            self._cv.notify_all()
                    continue

                text = "" if value is None else str(value)
                if not text:
                    continue

                now = time.perf_counter()
                token_estimate = max(1, len(text.split()))
                deltas[slot_idx][0] += float(token_estimate)

                with self._cv:
                    latest = self._states.get(request_id)
                    if latest is None or latest.completed:
                        continue
                    if latest.cancelled:
                        self._complete_locked(
                            request_id=request_id,
                            state=latest,
                            error="cancelled",
                            cancelled=True,
                        )
                        continue
                    if (
                        request_id not in self._last_emit_ts
                        and latest.first_scheduled_ts_ns
                    ):
                        ttft = (
                            time.time_ns() - int(latest.first_scheduled_ts_ns)
                        ) / 1_000_000.0
                        self._ttft_ms.append(float(max(0.0, ttft)))
                    if request_id in self._last_emit_ts:
                        self._tpot_ms.append(
                            float(
                                max(
                                    0.0, (now - self._last_emit_ts[request_id]) * 1000.0
                                )
                            )
                        )
                    self._last_emit_ts[request_id] = now

                    latest.chunks.append(
                        QSGChunk(request_id=request_id, text=text, done=False)
                    )
                    if latest.state_handle is not None:
                        latest.state_handle.phase_state += float(token_estimate) * 0.01
                        latest.state_handle.generated_tokens += token_estimate
                    self._generated_token_total += token_estimate
                    self._cv.notify_all()

            if slot_refs and self._any_nonzero(deltas):
                updated = [
                    [
                        float(base_value) + float(delta_value)
                        for base_value, delta_value in zip(base_row, delta_row)
                    ]
                    for base_row, delta_row in zip(gathered, deltas)
                ]
                self._pager.scatter_updates(slot_refs, updated)

            self._pager.soft_compact_if_needed()

            elapsed_ms = (time.perf_counter() - iteration_start) * 1000.0
            with self._cv:
                self._iteration_count += 1
                self._iteration_latency_ms.append(elapsed_ms)
                if self._active and self._interleaved_streams:
                    self._active.rotate(-1)
                self._promote_pending_locked()

    def shutdown(self, graceful_timeout_s: float = 1.0) -> None:
        with self._cv:
            self._shutdown_requested = True
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

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            pager_metrics = self._pager.metrics_snapshot()
            iteration_latencies = list(self._iteration_latency_ms)
            uptime_s = max(1.0e-6, time.perf_counter() - self._start_monotonic)
            active_states = [
                state
                for state in self._states.values()
                if state.state_handle is not None and not state.completed
            ]
            phase_mean = (
                sum(state.state_handle.phase_state for state in active_states)
                / len(active_states)
                if active_states
                else 0.0
            )
            decode_tps_per_agent: dict[str, float] = {}
            coconut_active_paths = 0
            coconut_entropy_samples: list[float] = []
            batched_prefill_calls = 0
            batched_prefill_tokens = 0
            for request_id, state in self._states.items():
                handle = state.state_handle
                if handle is None:
                    continue
                if handle.coconut_state_ref >= 0:
                    coconut_active_paths += 1
                    phase = float(handle.phase_state)
                    coconut_entropy_samples.append(
                        max(0.0, min(1.0, 1.0 - abs(phase - 0.5) * 2.0))
                    )
                if state.request.prompt_tokens:
                    batched_prefill_calls += 1
                    batched_prefill_tokens += len(state.request.prompt_tokens)
                if handle.generated_tokens <= 0:
                    continue
                started_ns = state.first_scheduled_ts_ns or state.request.arrival_ts_ns
                elapsed_s = max(
                    1.0e-6,
                    (time.time_ns() - int(started_ns)) / 1_000_000_000.0,
                )
                decode_tps_per_agent[request_id] = (
                    float(handle.generated_tokens) / elapsed_s
                )

            compaction_count = int(pager_metrics["compaction_count"])
            cow_events = int(pager_metrics["cow_events"])
            drift_overhead_percent = 0.0
            if self._iteration_count > 0:
                drift_overhead_percent = (
                    float(compaction_count + cow_events) / float(self._iteration_count)
                ) * 100.0

            metrics = {
                "scheduler_policy": self._scheduler_policy,
                "execution_mode": (
                    "interleaved" if self._interleaved_streams else "single_stream"
                ),
                "queue_depth": len(self._pending),
                "active_requests": len(self._active),
                "inflight_requests": len(self._states),
                "admitted_requests": self._admitted_count,
                "completed_requests": self._completed_count,
                "cancelled_requests": self._cancelled_count,
                "iterations": self._iteration_count,
                "iteration_latency_ms": {
                    "count": len(iteration_latencies),
                    "last": (
                        float(iteration_latencies[-1]) if iteration_latencies else 0.0
                    ),
                    "avg": (
                        sum(iteration_latencies) / len(iteration_latencies)
                        if iteration_latencies
                        else 0.0
                    ),
                    "max": max(iteration_latencies) if iteration_latencies else 0.0,
                    "p95": self._percentile(iteration_latencies, 0.95),
                },
                "qsg_queue_depth": len(self._pending),
                "qsg_active_requests": len(self._active),
                "qsg_request_admit_rate_rps": float(self._admitted_count) / uptime_s,
                "qsg_request_evict_rate_rps": float(self._evicted_count) / uptime_s,
                "qsg_queue_wait_ms_p50": self._percentile(
                    list(self._queue_wait_ms), 0.50
                ),
                "qsg_queue_wait_ms_p95": self._percentile(
                    list(self._queue_wait_ms), 0.95
                ),
                "qsg_queue_wait_ms_p99": self._percentile(
                    list(self._queue_wait_ms), 0.99
                ),
                "qsg_scheduler_iteration_ms_p50": self._percentile(
                    iteration_latencies, 0.50
                ),
                "qsg_scheduler_iteration_ms_p95": self._percentile(
                    iteration_latencies, 0.95
                ),
                "qsg_decode_tps_global": float(self._generated_token_total) / uptime_s,
                "qsg_decode_tps_per_agent": decode_tps_per_agent,
                "qsg_ttft_ms_p50": self._percentile(list(self._ttft_ms), 0.50),
                "qsg_ttft_ms_p95": self._percentile(list(self._ttft_ms), 0.95),
                "qsg_tpot_ms_p50": self._percentile(list(self._tpot_ms), 0.50),
                "qsg_tpot_ms_p95": self._percentile(list(self._tpot_ms), 0.95),
                "qsg_state_pages_total": pager_metrics["pages_total"],
                "qsg_state_pages_in_use": pager_metrics["pages_in_use"],
                "qsg_state_fragmentation_ratio": pager_metrics["fragmentation_ratio"],
                "qsg_state_compaction_count": pager_metrics["compaction_count"],
                "qsg_state_cow_events": pager_metrics["cow_events"],
                "qsg_state_allocator_failures": pager_metrics["allocator_failures"],
                "qsg_latent_requests": self._latent_restore_count,
                "qsg_suspended_requests": self._suspended_count,
                "qsg_coconut_active_paths": coconut_active_paths,
                "qsg_coconut_entropy_mean": (
                    sum(coconut_entropy_samples) / len(coconut_entropy_samples)
                    if coconut_entropy_samples
                    else 0.0
                ),
                "qsg_phase_confidence_mean": float(phase_mean),
                "qsg_drift_overhead_percent": drift_overhead_percent,
                "qsg_python_hot_path_calls": pager_metrics["python_hot_path_calls"],
                "qsg_numpy_hot_path_calls": pager_metrics["numpy_hot_path_calls"],
                "qsg_batched_prefill_token_id_calls": batched_prefill_calls,
                "qsg_batched_prefill_token_id_tokens": batched_prefill_tokens,
                "qsg_native_runtime_authority": bool(
                    getattr(self._config, "native_runtime_authority", True)
                ),
                "qsg_capability_digest": str(
                    getattr(self._config, "capability_digest", "") or ""
                ),
                "qsg_delta_watermark": dict(
                    getattr(self._config, "delta_watermark", None) or {}
                ),
            }
            return metrics

    def capture_latent_state(self, request_id: str) -> dict[str, Any] | None:
        with self._lock:
            state = self._states.get(request_id)
            if state is None or state.state_handle is None:
                return None
            tensor = self._matrix_from_tensor(
                self._pager.export_request_state(request_id)
            )
            self._latent_capture_count += 1
            delta_watermark = DeltaWatermark.from_dict(
                getattr(self._config, "delta_watermark", None)
            )
            capability_digest = str(
                getattr(self._config, "capability_digest", "") or ""
            )
            hidden_dim = int(self._pager.dim)
            if not tensor:
                tensor = [[0.0] * max(1, hidden_dim)]
            branch_segment = TypedLatentSegment(
                segment_id=f"{request_id}:branch_state",
                segment_kind="branch_state",
                row_start=0,
                row_count=len(tensor),
                hidden_dim=hidden_dim,
                codec="float16",
                importance=1.0,
                metadata={"capture_stage": "continuous_engine"},
            )
            delta_segment = TypedLatentSegment(
                segment_id=f"{request_id}:repo_delta",
                segment_kind="repo_delta",
                row_start=len(tensor),
                row_count=1,
                hidden_dim=hidden_dim,
                codec="float16",
                importance=0.7,
                provenance=delta_watermark.as_dict(),
            )
            full_tensor = list(tensor)
            full_tensor.append(self._repo_delta_row(delta_watermark, hidden_dim))
            segment_index = CapsuleSegmentIndex(
                segments=[branch_segment, delta_segment]
            )
            capsule = ExecutionCapsule(
                capsule_id=f"capsule_{uuid.uuid4().hex}",
                request_id=request_id,
                version=max(
                    3, int(getattr(self._config, "execution_capsule_version", 2))
                ),
                model_family="qsg-continuous",
                capability_digest=capability_digest,
                delta_watermark=delta_watermark.as_dict(),
                generated_tokens=int(state.state_handle.generated_tokens),
                phase_state=float(state.state_handle.phase_state),
                hidden_dim=hidden_dim,
                latent_packet_abi_version=max(
                    3, int(getattr(self._config, "latent_packet_abi_version", 2))
                ),
                segment_count=2,
                segment_kinds=["branch_state", "repo_delta"],
                segment_index=segment_index.as_dict()["segments"],
                controller_decisions=[
                    ControllerDecisionRecord(
                        controller="frontier",
                        selected_mode=(
                            "interleaved"
                            if self._interleaved_streams
                            else "single_stream"
                        ),
                        reason="continuous_engine_scheduler_mode",
                        telemetry={"scheduler_policy": self._scheduler_policy},
                    ).as_dict(),
                ],
                metadata={
                    "native_runtime_authority": bool(
                        getattr(self._config, "native_runtime_authority", True)
                    )
                },
            )
            latent_packet = LatentPacketABI(
                abi_version=max(
                    3, int(getattr(self._config, "latent_packet_abi_version", 2))
                ),
                tensor=full_tensor,
                tensor_format="float32",
                tensor_codec="float16",
                hidden_dim=hidden_dim,
                generated_tokens=int(state.state_handle.generated_tokens),
                phase_state=float(state.state_handle.phase_state),
                capability_digest=capability_digest,
                delta_watermark=delta_watermark.as_dict(),
                execution_capsule_id=capsule.capsule_id,
                segments=segment_index.as_dict()["segments"],
                segment_count=2,
                compatibility_score=1.0,
            )
            return {
                "request_id": request_id,
                "prompt": state.request.prompt,
                "options": dict(state.request.options or {}),
                "tensor": full_tensor,
                "generated_tokens": state.state_handle.generated_tokens,
                "phase_state": state.state_handle.phase_state,
                "created_at": time.time(),
                "hidden_dim": hidden_dim,
                "latent_packet": latent_packet.as_dict(),
                "execution_capsule": capsule.as_dict(),
            }

    def restore_latent_state(
        self,
        package: dict[str, Any],
        *,
        target_request_id: str | None = None,
    ) -> str:
        latent_packet = dict(package.get("latent_packet") or {})
        tensor = self._matrix_from_tensor(
            latent_packet.get("tensor", package.get("tensor"))
        )
        tensor = self._select_segment_rows(
            tensor,
            list(latent_packet.get("segments") or []),
            list(
                latent_packet.get("restore_segment_kinds")
                or package.get("restore_segment_kinds")
                or []
            ),
        )
        with self._cv:
            request_id = target_request_id or str(
                package.get("request_id") or uuid.uuid4().hex
            )
            if request_id in self._states:
                raise ValueError(f"Duplicate request_id '{request_id}'")
            request = QSGRequest(
                prompt=str(package.get("prompt") or "latent-replay"),
                options=dict(package.get("options") or {}),
                request_id=request_id,
            )
            state = _RequestState(request=request)
            self._states[request_id] = state
            refs = self._pager.import_request_state(request_id, tensor)
            if refs:
                ref = refs[0]
                state.state_handle = ActiveStateHandle(
                    slot_id=self._next_slot_id,
                    state_page_chain=[ref.page_id],
                    state_row_offsets=[ref.row_idx],
                    phase_state=float(
                        latent_packet.get(
                            "phase_state", package.get("phase_state") or 0.0
                        )
                    ),
                    generated_tokens=int(
                        latent_packet.get(
                            "generated_tokens",
                            package.get("generated_tokens") or 0,
                        )
                    ),
                )
                self._next_slot_id += 1
            state.chunks.append(
                QSGChunk(
                    request_id=request_id,
                    text="",
                    done=False,
                    event="latent_restored",
                    metadata={
                        "source": "almf",
                        "execution_capsule": dict(
                            package.get("execution_capsule") or {}
                        ),
                        "latent_packet_abi_version": int(
                            latent_packet.get(
                                "abi_version",
                                getattr(self._config, "latent_packet_abi_version", 2),
                            )
                        ),
                    },
                )
            )
            self._enqueue_pending_locked(request_id)
            self._latent_restore_count += 1
            self._cv.notify_all()
            return request_id

    def _enqueue_pending_locked(self, request_id: str) -> None:
        if self._scheduler_policy != "priority":
            self._pending.append(request_id)
            return
        state = self._states[request_id]
        inserted = False
        for idx, pending_request_id in enumerate(self._pending):
            other = self._states.get(pending_request_id)
            if other is None:
                continue
            if state.request.priority > other.request.priority:
                self._pending.insert(idx, request_id)
                inserted = True
                break
            if (
                state.request.priority == other.request.priority
                and state.request.arrival_ts_ns < other.request.arrival_ts_ns
            ):
                self._pending.insert(idx, request_id)
                inserted = True
                break
        if not inserted:
            self._pending.append(request_id)

    def _promote_pending_locked(self) -> None:
        active_limit = self._max_active_requests if self._interleaved_streams else 1
        if not self._interleaved_streams and any(
            state.chunks for state in self._states.values()
        ):
            return
        while self._pending and len(self._active) < active_limit:
            request_id = self._pending.popleft()
            state = self._states.get(request_id)
            if state is None or state.completed:
                continue
            if state.first_scheduled_ts_ns is None:
                state.first_scheduled_ts_ns = time.time_ns()
                state.queue_wait_ms = (
                    float(state.first_scheduled_ts_ns - state.request.arrival_ts_ns)
                    / 1_000_000.0
                )
                self._queue_wait_ms.append(float(max(0.0, state.queue_wait_ms)))
            if state.state_handle is None:
                self._attach_state_handle_locked(request_id, state)
            self._active.append(request_id)

    def _attach_state_handle_locked(
        self, request_id: str, state: _RequestState
    ) -> None:
        refs = self._pager.alloc_rows(request_id, 1)
        if not refs:
            return
        ref = refs[0]
        state.state_handle = ActiveStateHandle(
            slot_id=self._next_slot_id,
            state_page_chain=[ref.page_id],
            state_row_offsets=[ref.row_idx],
        )
        self._next_slot_id += 1

    def _complete_locked(
        self,
        *,
        request_id: str,
        state: _RequestState,
        error: str | None = None,
        cancelled: bool = False,
    ) -> None:
        if state.completed:
            return
        state.completed = True
        if state.state_handle is not None:
            state.state_handle.finished = True

        self._discard_from_deque(self._pending, request_id)
        self._discard_from_deque(self._active, request_id)
        self._pager.release_request(request_id)
        self._last_emit_ts.pop(request_id, None)

        if cancelled:
            self._cancelled_count += 1
            self._evicted_count += 1
        else:
            self._completed_count += 1

        state.chunks.append(
            QSGChunk(request_id=request_id, text="", done=True, error=error)
        )

    @staticmethod
    def _discard_from_deque(values: deque[str], item: str) -> None:
        try:
            values.remove(item)
        except ValueError:
            pass

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = int((len(sorted_values) - 1) * min(1.0, max(0.0, q)))
        return float(sorted_values[idx])
