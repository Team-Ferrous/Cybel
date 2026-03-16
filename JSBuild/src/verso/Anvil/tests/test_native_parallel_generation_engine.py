from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import core.native.parallel_generation as parallel_generation
from core.native.parallel_generation import NativeParallelGenerationEngine
from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGRequest
from shared_kernel.event_store import EventStore


class _NativeEngineStub:
    def __init__(
        self,
        runtime_status: dict[str, object] | None = None,
        native_kv_metrics: dict[str, object] | None = None,
    ):
        self.snapshots = []
        self.runtime_status = dict(runtime_status or {})
        self.num_ubatch = 2
        self._native_kv_cache = (
            _NativeKVCacheStub(native_kv_metrics)
            if native_kv_metrics is not None
            else None
        )

    def get_runtime_status(self):
        return dict(self.runtime_status)

    def _update_scheduler_metrics_snapshot(self, metrics):
        self.snapshots.append(dict(metrics))


class _NativeKVCacheStub:
    def __init__(self, metrics: dict[str, object]):
        self._metrics = dict(metrics)

    def metrics_snapshot(self):
        return dict(self._metrics)


def _collect(engine, request_ids: list[str], timeout_s: float):
    deadline = time.time() + timeout_s
    done = set()
    events: list[tuple[str, str, bool, str | None]] = []
    while time.time() < deadline and len(done) < len(request_ids):
        for request_id in request_ids:
            chunk = engine.poll(request_id)
            if chunk is None:
                continue
            events.append((chunk.request_id, chunk.text, chunk.done, chunk.error))
            if chunk.done:
                done.add(chunk.request_id)
        time.sleep(0.001)
    return events, done


def _collect_detailed(engine, request_ids: list[str], timeout_s: float):
    deadline = time.time() + timeout_s
    done = set()
    events: list[dict[str, object]] = []
    while time.time() < deadline and len(done) < len(request_ids):
        for request_id in request_ids:
            chunk = engine.poll(request_id)
            if chunk is None:
                continue
            events.append(
                {
                    "request_id": chunk.request_id,
                    "text": chunk.text,
                    "done": chunk.done,
                    "error": chunk.error,
                    "event": chunk.event,
                    "metadata": chunk.metadata,
                }
            )
            if chunk.done:
                done.add(chunk.request_id)
        time.sleep(0.001)
    return events, done


def test_native_scheduler_single_stream_ordering():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=4,
            max_pending_requests=8,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    rid_a = engine.submit(
        QSGRequest(
            prompt="a",
            options={"chunks": ["a1", "a2"]},
            prompt_tokens=[11, 12, 13, 14, 15],
            max_new_tokens=4,
        )
    )
    rid_b = engine.submit(
        QSGRequest(
            prompt="b",
            options={"chunks": ["b1", "b2"]},
            prompt_tokens=[21, 22],
            max_new_tokens=4,
        )
    )
    events, done = _collect(engine, [rid_a, rid_b], timeout_s=1.0)
    by_request = {rid_a: [], rid_b: []}
    for rid, text, chunk_done, _ in events:
        if chunk_done or not text:
            continue
        by_request[rid].append(text)

    assert by_request[rid_a] == ["a1", "a2"]
    assert by_request[rid_b] == ["b1", "b2"]
    assert done == {rid_a, rid_b}

    metrics = engine.metrics_snapshot()
    assert metrics["execution_mode"] == "single_stream"
    assert "qsg_queue_wait_ms_p95" in metrics
    assert "qsg_scheduler_iteration_ms_p95" in metrics
    assert metrics["qsg_latent_requests"] == 0
    assert metrics["qsg_suspended_requests"] == 0
    assert metrics["qsg_prefill_request_count"] == 2
    assert metrics["qsg_prefill_tokens_scheduled"] == 7
    assert metrics["qsg_batched_prefill_token_id_calls"] == 2
    assert metrics["qsg_batched_prefill_token_id_tokens"] == 7
    assert metrics["qsg_chunked_prefill_requests"] == 1
    assert metrics["qsg_chunked_prefill_chunks"] == 3
    assert metrics["qsg_decode_tokens_emitted"] >= 4
    assert metrics["continuous_runtime_owner"] == "python_compatibility_shim"
    assert metrics["native_runtime_abi_ready"] is False
    assert metrics["phase1_ready"] is False
    assert "missing_native_serve_runtime_abi" in metrics["phase1_blockers"]
    assert "python_scheduler_loop_active" in metrics["phase1_blockers"]
    assert metrics["qsg_python_hot_path_calls"] >= 1
    assert metrics["hot_path_proof"]["executed_cpp_only"] == "false"

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_interleaved_ordering():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=4,
            max_pending_requests=8,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=True,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    rid_a = engine.submit(
        QSGRequest(
            prompt="a",
            options={"chunks": ["a1", "a2"]},
            prompt_tokens=[1, 2, 3],
            max_new_tokens=4,
        )
    )
    rid_b = engine.submit(
        QSGRequest(
            prompt="b",
            options={"chunks": ["b1", "b2"]},
            prompt_tokens=[4, 5, 6, 7],
            max_new_tokens=4,
        )
    )
    events, done = _collect(engine, [rid_a, rid_b], timeout_s=1.0)
    by_request = {rid_a: [], rid_b: []}
    for rid, text, chunk_done, _ in events:
        if chunk_done or not text:
            continue
        by_request[rid].append(text)

    assert by_request[rid_a] == ["a1", "a2"]
    assert by_request[rid_b] == ["b1", "b2"]
    assert done == {rid_a, rid_b}

    metrics = engine.metrics_snapshot()
    assert metrics["execution_mode"] == "interleaved"
    assert metrics["benchmark_label"] == "parallel_hybrid"
    assert metrics["qsg_prefill_request_count"] == 2
    assert metrics["qsg_prefill_tokens_scheduled"] == 7
    assert metrics["qsg_chunked_prefill_requests"] == 2
    assert metrics["qsg_chunked_prefill_chunks"] == 4
    assert metrics["continuous_runtime_owner"] == "python_compatibility_shim"
    assert metrics["phase1_ready"] is False

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_metrics_merge_runtime_hot_path_status():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(
            runtime_status={
                "python_hot_path_calls": 7,
                "numpy_hot_path_calls": 3,
                "hot_path_numpy_detected": True,
                "hot_path_proof": {"full_qsg": "enabled"},
            }
        ),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=2,
            max_pending_requests=4,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=True,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(QSGRequest(prompt="a", options={"chunks": ["a1"]}))
    _, done = _collect(engine, [request_id], timeout_s=1.0)

    metrics = engine.metrics_snapshot()

    assert done == {request_id}
    assert metrics["qsg_python_hot_path_calls"] >= 7
    assert metrics["qsg_numpy_hot_path_calls"] == 3
    assert metrics["hot_path_numpy_detected"] is True
    assert "python_token_loop_active" in metrics["phase1_blockers"]
    assert "numpy_hot_path_active" in metrics["phase1_blockers"]
    assert (
        metrics["hot_path_proof"]["continuous_runtime_owner"]
        == "python_compatibility_shim"
    )
    assert metrics["hot_path_proof"]["phase1_blockers"]

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_metrics_include_native_kv_cache_evidence():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(
            native_kv_metrics={
                "resident_page_count": 6,
                "active_page_slots": 9,
                "shared_page_slots": 4,
                "snapshot_count": 2,
                "copy_on_write_events": 3,
                "prefix_share_events": 5,
                "active_tokens": 17,
                "committed_token_capacity": 64,
                "page_tokens": 8,
                "fragmentation_ratio": 0.375,
            }
        ),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=2,
            max_pending_requests=4,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(QSGRequest(prompt="a", options={"chunks": ["a1"]}))
    _, done = _collect(engine, [request_id], timeout_s=1.0)

    metrics = engine.metrics_snapshot()

    assert done == {request_id}
    assert metrics["qsg_state_pages_total"] == 6
    assert metrics["qsg_state_pages_in_use"] == 6
    assert metrics["qsg_state_active_page_slots"] == 9
    assert metrics["qsg_state_shared_page_slots"] == 4
    assert metrics["qsg_state_snapshot_count"] == 2
    assert metrics["qsg_state_cow_events"] == 3
    assert metrics["qsg_state_prefix_share_events"] == 5
    assert metrics["qsg_state_active_tokens"] == 17
    assert metrics["qsg_state_committed_token_capacity"] == 64
    assert metrics["qsg_state_page_tokens"] == 8
    assert metrics["qsg_state_fragmentation_ratio"] == 0.375
    assert engine._native_engine.snapshots[-1]["qsg_state_shared_page_slots"] == 4
    assert engine._native_engine.snapshots[-1]["qsg_state_cow_events"] == 3

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_metrics_propagate_phase5_runtime_fields():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(
            runtime_status={
                "generation_mode": "prompt_lookup",
                "benchmark_label": "prompt_lookup",
                "prompt_category": "code",
                "temperature_band": "low",
                "accepted_parallel_tokens": 4,
                "rejected_parallel_tokens": 1,
                "proposed_parallel_tokens": 5,
                "draft_frontier_width": 5,
                "verify_depth": 4,
                "parallel_step_latency_ms": 2.5,
                "draft_confidence_mean": 0.72,
                "draft_confidence_min": 0.51,
                "draft_source": "prompt_lookup_native",
                "quality_guard_triggered": True,
                "self_spec_native_path": True,
                "self_spec_policy": "heuristic",
                "self_spec_exit_layer": 12,
                "self_spec_exit_fraction": 0.5,
                "self_spec_draft_tokens": 4,
                "supported_benchmark_labels": ["ar_baseline", "prompt_lookup"],
            }
        ),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=2,
            max_pending_requests=4,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(
        QSGRequest(prompt="explain prompt lookup", options={"chunks": ["a1"]})
    )
    _, done = _collect(engine, [request_id], timeout_s=1.0)

    metrics = engine.metrics_snapshot()

    assert done == {request_id}
    assert metrics["generation_mode"] == "prompt_lookup"
    assert metrics["benchmark_label"] == "prompt_lookup"
    assert metrics["prompt_category"] == "code"
    assert metrics["temperature_band"] == "low"
    assert metrics["accepted_parallel_tokens"] == 4
    assert metrics["rejected_parallel_tokens"] == 1
    assert metrics["proposed_parallel_tokens"] == 5
    assert metrics["draft_frontier_width"] == 5
    assert metrics["verify_depth"] == 4
    assert metrics["parallel_step_latency_ms"] == 2.5
    assert metrics["draft_confidence_mean"] == 0.72
    assert metrics["draft_confidence_min"] == 0.51
    assert metrics["draft_source"] == "prompt_lookup_native"
    assert metrics["quality_guard_triggered"] is True
    assert metrics["self_spec_native_path"] is True
    assert metrics["self_spec_policy"] == "heuristic"
    assert metrics["self_spec_exit_layer"] == 12
    assert metrics["self_spec_exit_fraction"] == 0.5
    assert metrics["self_spec_draft_tokens"] == 4
    assert metrics["supported_benchmark_labels"] == [
        "ar_baseline",
        "prompt_lookup",
    ]

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_native_runtime_abi_marks_owner():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(
            runtime_status={
                "native_runtime_abi_ready": True,
                "python_hot_path_calls": 0,
                "numpy_hot_path_calls": 0,
                "hot_path_numpy_detected": False,
            }
        ),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=2,
            max_pending_requests=4,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(
        QSGRequest(prompt="a", options={"chunks": ["a1", "a2"]}, max_new_tokens=4)
    )
    _, done = _collect(engine, [request_id], timeout_s=1.0)

    metrics = engine.metrics_snapshot()

    assert done == {request_id}
    assert metrics["native_runtime_abi_ready"] is True
    assert metrics["continuous_runtime_owner"] == "native_runtime"
    assert "python_scheduler_loop_active" in metrics["phase1_blockers"]
    assert "missing_native_serve_runtime_abi" not in metrics["phase1_blockers"]

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_scheduler_marks_request_states_from_options(monkeypatch):
    class _FakeScheduler:
        def __init__(self):
            self.submissions: list[dict[str, int | str]] = []
            self.latent_by_id: dict[str, bool] = {}
            self.suspended_by_id: dict[str, bool] = {}
            self.completed: list[tuple[str, bool]] = []
            self.closed = False

        def submit_with_metadata(
            self,
            request_id: str,
            *,
            priority: int,
            arrival_ts_ns: int,
            prompt_token_count: int,
            max_new_tokens: int,
            prefill_chunk_size: int,
        ) -> None:
            self.submissions.append(
                {
                    "request_id": request_id,
                    "priority": int(priority),
                    "arrival_ts_ns": int(arrival_ts_ns),
                    "prompt_token_count": int(prompt_token_count),
                    "max_new_tokens": int(max_new_tokens),
                    "prefill_chunk_size": int(prefill_chunk_size),
                }
            )

        def mark_request_latent(self, request_id: str, is_latent: bool) -> None:
            self.latent_by_id[str(request_id)] = bool(is_latent)

        def mark_request_suspended(self, request_id: str, is_suspended: bool) -> None:
            self.suspended_by_id[str(request_id)] = bool(is_suspended)

        def complete(self, request_id: str, *, cancelled: bool = False) -> None:
            self.completed.append((str(request_id), bool(cancelled)))

        def close(self) -> None:
            self.closed = True

        def record_iteration(self, iteration_ms: float) -> None:
            pass

        def rotate_active(self) -> None:
            pass

        def promote(self) -> None:
            pass

        def active_ids(self) -> list[str]:
            return []

        def first_scheduled_ns(self, request_id: str) -> int:
            return 0

        def metrics(self):
            class _Metrics:
                queue_depth = 0
                active_requests = 0
                inflight_requests = 0
                prefill_active_requests = 0
                decode_active_requests = 0
                admitted_requests = 0
                completed_requests = 0
                cancelled_requests = 0
                evicted_requests = 0
                iterations = 0
                prefill_request_count = 0
                prefill_tokens_scheduled = 0
                decode_tokens_emitted = 0
                chunked_prefill_requests = 0
                chunked_prefill_chunks = 0
                iteration_last_ms = 0.0
                iteration_avg_ms = 0.0
                iteration_p95_ms = 0.0
                queue_wait_p50_ms = 0.0
                queue_wait_p95_ms = 0.0
                queue_wait_p99_ms = 0.0
                latent_requests = 0
                suspended_requests = 0

            return _Metrics()

    fake_scheduler = _FakeScheduler()

    monkeypatch.setattr(
        parallel_generation,
        "NativeQSGScheduler",
        lambda *args, **kwargs: fake_scheduler,
    )

    engine = parallel_generation.NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=4,
            max_pending_requests=8,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=lambda request: iter([]),
    )

    request_id = engine.submit(
        QSGRequest(
            prompt="stateful",
            options={"latent_mode": True, "parked": True},
            prompt_tokens=[1, 2, 3],
            max_new_tokens=4,
        )
    )

    assert fake_scheduler.latent_by_id.get(request_id) is True
    assert fake_scheduler.suspended_by_id.get(request_id) is True
    assert fake_scheduler.submissions[0]["prefill_chunk_size"] == 2
    assert fake_scheduler.submissions[0]["prompt_token_count"] == 3
    assert fake_scheduler.submissions[0]["max_new_tokens"] == 4

    engine.shutdown()
    assert fake_scheduler.closed is True
    assert fake_scheduler.completed == [(request_id, True)]


def test_native_scheduler_parked_request_resumes_with_tool_evidence():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=4,
            max_pending_requests=8,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=_producer,
    )
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    parked_id = engine.submit(
        QSGRequest(
            prompt="latent task",
            options={
                "chunks": ["p1"],
                "parked": True,
                "awaiting_tool": True,
                "latent_mode": True,
            },
            prompt_tokens=[1, 2, 3],
            max_new_tokens=2,
        )
    )
    ready_id = engine.submit(
        QSGRequest(
            prompt="ready",
            options={"chunks": ["r1"]},
            prompt_tokens=[4],
            max_new_tokens=1,
        )
    )

    early_events, early_done = _collect_detailed(
        engine, [parked_id, ready_id], timeout_s=0.5
    )
    ready_text = [
        event["text"]
        for event in early_events
        if event["request_id"] == ready_id
        and isinstance(event["text"], str)
        and event["text"]
    ]
    parked_text = [
        event["text"]
        for event in early_events
        if event["request_id"] == parked_id
        and isinstance(event["text"], str)
        and event["text"]
    ]

    assert ready_text == ["r1"]
    assert ready_id in early_done
    assert parked_id not in early_done
    assert parked_text == []

    parked_status = engine.request_status(parked_id)
    assert parked_status is not None
    assert parked_status["lifecycle"] == "awaiting_tool"
    assert "SUSPENDED" in parked_status["scheduler_state_flags"]
    assert "LATENT" in parked_status["scheduler_state_flags"]

    latent_packet = engine.record_latent_packet(
        parked_id,
        {
            "hidden_dimension": 128,
            "norm_summary": {"l2": 1.5},
            "branch_score": 0.82,
            "stop_policy": "tool_intercept",
        },
    )
    tool_evidence = engine.record_tool_evidence(
        parked_id,
        {
            "tool_name": "pytest",
            "capsule_path": "/tmp/evidence_capsule.json",
            "latent_feedback": {"status": "passed"},
        },
    )
    resumed_status = engine.resume_request(parked_id, mark_latent=False)

    metrics = engine.metrics_snapshot()
    assert latent_packet["hidden_dimension"] == 128
    assert tool_evidence["tool_name"] == "pytest"
    assert resumed_status["resume_count"] == 1
    assert metrics["qsg_latent_packet_count"] >= 1
    assert metrics["qsg_evidence_capsule_count"] >= 1
    assert metrics["qsg_resume_events"] >= 1
    assert metrics["qsg_sequence_mode_counts"]["text"] >= 1

    resumed_events, resumed_done = _collect_detailed(engine, [parked_id], timeout_s=1.0)
    parked_events = [
        event for event in resumed_events if event["request_id"] == parked_id
    ]
    parked_event_names = [event["event"] for event in parked_events if event["event"]]
    parked_text = [
        event["text"]
        for event in parked_events
        if isinstance(event["text"], str) and event["text"]
    ]

    assert parked_id in resumed_done
    assert "tool_result" in parked_event_names
    assert "resumed" in parked_event_names
    assert parked_text == ["p1"]

    engine.shutdown()
    runner.join(timeout=1.0)


def test_native_runtime_path_marks_phase1_ready(monkeypatch):
    class _Runtime:
        def __init__(self, **kwargs):
            del kwargs
            self._events = {}
            self._states = {}

        def submit(self, request_id, **kwargs):
            arrival_ts_ns = int(kwargs["arrival_ts_ns"])
            self._states[request_id] = {
                "first_scheduled_ns": arrival_ts_ns + 1_000_000,
            }
            self._events[request_id] = [
                SimpleNamespace(token_id=7, done=False, error=None),
                SimpleNamespace(token_id=None, done=True, error=None),
            ]

        def poll(self, request_id):
            queue = self._events.get(request_id, [])
            if not queue:
                return None
            return queue.pop(0)

        def cancel(self, request_id):
            self._states.pop(request_id, None)

        def mark_request_latent(self, request_id, is_latent):
            del request_id, is_latent

        def mark_request_suspended(self, request_id, is_suspended):
            del request_id, is_suspended

        def first_scheduled_ns(self, request_id):
            return int(self._states[request_id]["first_scheduled_ns"])

        def request_state(self, request_id):
            del request_id
            return parallel_generation.NativeQSGRequestState.ACTIVE

        def shutdown(self):
            return None

        def close(self):
            return None

        def metrics(self):
            scheduler = SimpleNamespace(
                queue_depth=0,
                active_requests=0,
                inflight_requests=0,
                prefill_active_requests=0,
                decode_active_requests=0,
                admitted_requests=1,
                completed_requests=1,
                cancelled_requests=0,
                evicted_requests=0,
                iterations=0,
                prefill_request_count=1,
                prefill_tokens_scheduled=3,
                decode_tokens_emitted=1,
                chunked_prefill_requests=1,
                chunked_prefill_chunks=2,
                iteration_last_ms=0.0,
                iteration_avg_ms=0.0,
                iteration_p95_ms=0.0,
                queue_wait_p50_ms=1.0,
                queue_wait_p95_ms=1.0,
                queue_wait_p99_ms=1.0,
                latent_requests=0,
                suspended_requests=0,
            )
            return SimpleNamespace(
                scheduler=scheduler,
                worker_iterations=1,
                emitted_events=2,
                prefill_batches=2,
                runtime_prefill_tokens=3,
                runtime_decode_steps=1,
                worker_running=True,
                native_runtime_abi_ready=True,
            )

    class _RuntimeEngine(_NativeEngineStub):
        def __init__(self):
            super().__init__(runtime_status={"hot_path_proof": {}})
            self._model_graph = SimpleNamespace(_handle=1)
            self.profile = SimpleNamespace(vocab_size=32)

        def token_eos(self):
            return 99

        def detokenize(self, ids):
            return "x" if ids == [7] else ""

    monkeypatch.setattr(parallel_generation, "NativeQSGRuntime", _Runtime)

    engine = NativeParallelGenerationEngine(
        native_engine=_RuntimeEngine(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            max_active_requests=2,
            max_pending_requests=4,
            batch_wait_timeout_ms=1,
            semantic_resonance_timeout_ms=1,
            continuous_interleaved_streams=False,
        ),
        stream_producer=lambda request: iter(()),
    )

    request_id = engine.submit(
        QSGRequest(prompt="rt", prompt_tokens=[1, 2, 3], max_new_tokens=1)
    )
    events, done = _collect(engine, [request_id], timeout_s=1.0)
    metrics = engine.metrics_snapshot()

    assert done == {request_id}
    assert [text for rid, text, chunk_done, _ in events if rid == request_id and not chunk_done] == ["x"]
    assert metrics["native_runtime_abi_ready"] is True
    assert metrics["continuous_runtime_owner"] == "native_runtime"
    assert metrics["phase1_ready"] is True
    assert metrics["phase1_blockers"] == []
    assert metrics["qsg_python_hot_path_calls"] == 0
    assert metrics["hot_path_proof"]["executed_cpp_only"] == "true"

    engine.shutdown()


def test_native_engine_capture_latent_state_uses_runtime_contract_abi():
    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            batch_wait_timeout_ms=1,
            capability_digest="cap-native",
            delta_watermark={
                "delta_id": "delta-1",
                "logical_clock": 7,
                "workspace_id": "main",
            },
        ),
        stream_producer=lambda request: iter((request.prompt,)),
    )

    request_id = engine.submit(
        QSGRequest(
            prompt="latent",
            options={
                "latent": True,
                "latent_packets": [
                    {"hidden_dimension": 64, "norm_summary": {"l2": 1.0}}
                ],
            },
        )
    )

    captured = engine.capture_latent_state(request_id)

    assert captured is not None
    assert captured["execution_capsule"]["version"] == 2
    assert captured["execution_capsule"]["capability_digest"] == "cap-native"
    assert captured["latent_packet"]["abi_version"] == 2
    assert captured["latent_packet"]["hidden_dim"] == 64
    assert (
        captured["latent_packet"]["execution_capsule_id"]
        == captured["execution_capsule"]["capsule_id"]
    )
    assert captured["latent_packet"]["delta_watermark"]["delta_id"] == "delta-1"


def test_native_engine_restore_latent_state_emits_replay_event():
    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            batch_wait_timeout_ms=1,
        ),
        stream_producer=lambda request: iter(("resume",)),
    )

    restored_id = engine.restore_latent_state(
        {
            "request_id": "replay-native",
            "prompt": "latent replay",
            "options": {"replay_memory_id": "mem-1"},
            "latent_packet": {
                "abi_version": 2,
                "hidden_dim": 8,
                "capability_digest": "cap-replay",
                "delta_watermark": {"delta_id": "delta-replay", "logical_clock": 3},
                "execution_capsule_id": "capsule-replay",
            },
            "execution_capsule": {
                "capsule_id": "capsule-replay",
                "version": 2,
                "delta_watermark": {"delta_id": "delta-replay", "logical_clock": 3},
            },
        }
    )

    status = engine.request_status(restored_id)
    chunk = engine.poll(restored_id)

    assert restored_id == "replay-native"
    assert status is not None
    assert status["capability_digest"] == "cap-replay"
    assert status["execution_capsule_id"] == "capsule-replay"
    assert status["delta_watermark"]["delta_id"] == "delta-replay"
    assert chunk is not None
    assert chunk.event == "latent_restored"
    assert chunk.metadata["latent_packet_abi_version"] == 2
    assert chunk.metadata["execution_capsule"]["capsule_id"] == "capsule-replay"


def test_native_engine_records_replay_tape_events(tmp_path):
    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            batch_wait_timeout_ms=1,
        ),
        stream_producer=lambda request: iter(()),
    )
    engine._event_store = EventStore(str(tmp_path / "events.db"))

    request_id = engine.submit(QSGRequest(prompt="tape"))
    engine.record_latent_packet(request_id, {"hidden_dimension": 4})
    engine.suspend_request(request_id, reason="await_tool", awaiting_tool=True)
    engine.resume_request(request_id, mark_latent=False)

    tape = engine._event_store.export_replay_tape(
        request_id, output_path=str(tmp_path / "tape.json")
    )
    stages = [event["payload"]["stage"] for event in tape["events"]]

    assert "submitted" in stages
    assert "latent_packet_recorded" in stages
    assert "checkpoint_captured" in stages
    assert "resumed" in stages


def test_native_engine_tracks_lineage_prefix_reuse_and_reasoning_lanes():
    engine = NativeParallelGenerationEngine(
        native_engine=_NativeEngineStub(),
        config=QSGConfig(
            continuous_batching_enabled=True,
            batch_wait_timeout_ms=1,
            lineage_prefix_reuse_enabled=True,
        ),
        stream_producer=lambda request: iter(()),
    )

    first_id = engine.submit(
        QSGRequest(
            prompt="first",
            prompt_tokens=[1, 2, 3, 4],
            options={"lineage_id": "lineage-a", "reasoning_lane": "verify_heavy"},
        )
    )
    second_id = engine.submit(
        QSGRequest(
            prompt="second",
            prompt_tokens=[1, 2, 3, 4],
            options={"lineage_id": "lineage-a", "awaiting_tool": True, "suspended": True},
        )
    )

    first_status = engine.request_status(first_id)
    second_status = engine.request_status(second_id)
    metrics = engine.metrics_snapshot()

    assert first_status is not None
    assert second_status is not None
    assert first_status["reasoning_lane"] == "verify_heavy"
    assert second_status["reasoning_lane"] == "tool_wait"
    assert second_status["prefix_reuse_hit"] is True
    assert metrics["qsg_prefix_reuse_hits"] >= 1
    assert metrics["prefix_cache_hit_rate"] > 0.0
    assert metrics["qsg_reasoning_lane_counts"]["tool_wait"] >= 1

    engine.shutdown()
