import threading
import time

import pytest

from core.qsg.config import QSGConfig
from core.qsg.continuous_engine import QSGInferenceEngine, QSGRequest


def _make_engine() -> QSGInferenceEngine:
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    config = QSGConfig(
        continuous_batching_enabled=True,
        max_active_requests=4,
        max_pending_requests=8,
        batch_wait_timeout_ms=1,
        semantic_resonance_timeout_ms=1,
    )
    return QSGInferenceEngine(config=config, stream_producer=_producer)


def _collect_events(
    engine: QSGInferenceEngine, request_ids: list[str], timeout_s: float
):
    deadline = time.time() + timeout_s
    done_ids: set[str] = set()
    events: list[tuple[str, str, bool, str | None]] = []
    while time.time() < deadline and len(done_ids) < len(request_ids):
        for request_id in request_ids:
            chunk = engine.poll(request_id)
            if chunk is None:
                continue
            events.append((chunk.request_id, chunk.text, chunk.done, chunk.error))
            if chunk.done:
                done_ids.add(chunk.request_id)
        time.sleep(0.001)
    return events, done_ids


def test_queue_admission_respects_pending_limit():
    config = QSGConfig(
        continuous_batching_enabled=True,
        max_active_requests=1,
        max_pending_requests=1,
    )
    engine = QSGInferenceEngine(
        config=config,
        stream_producer=lambda request: iter((request.options or {}).get("chunks", [])),
    )

    engine.submit(QSGRequest(prompt="r1", options={"chunks": ["a"]}))
    with pytest.raises(RuntimeError, match="pending queue is full"):
        engine.submit(QSGRequest(prompt="r2", options={"chunks": ["b"]}))


def test_round_robin_emits_one_chunk_per_request_per_iteration():
    def _producer(request: QSGRequest):
        for chunk in list((request.options or {}).get("chunks", [])):
            yield chunk

    config = QSGConfig(
        continuous_batching_enabled=True,
        continuous_interleaved_streams=True,
        max_active_requests=4,
        max_pending_requests=8,
        batch_wait_timeout_ms=1,
        semantic_resonance_timeout_ms=1,
    )
    engine = QSGInferenceEngine(config=config, stream_producer=_producer)
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    rid_a = engine.submit(QSGRequest(prompt="a", options={"chunks": ["a1", "a2"]}))
    rid_b = engine.submit(QSGRequest(prompt="b", options={"chunks": ["b1", "b2"]}))

    events, done_ids = _collect_events(engine, [rid_a, rid_b], timeout_s=1.0)

    payload = [(rid, text) for rid, text, done, _ in events if text and not done]
    assert payload[:4] == [
        (rid_a, "a1"),
        (rid_b, "b1"),
        (rid_a, "a2"),
        (rid_b, "b2"),
    ]
    assert done_ids == {rid_a, rid_b}

    metrics = engine.metrics_snapshot()
    assert metrics["admitted_requests"] == 2
    assert metrics["completed_requests"] == 2
    assert metrics["cancelled_requests"] == 0
    assert metrics["execution_mode"] == "interleaved"

    engine.shutdown()
    runner.join(timeout=1.0)


def test_default_scheduler_drains_one_stream_before_advancing():
    engine = _make_engine()
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    rid_a = engine.submit(QSGRequest(prompt="a", options={"chunks": ["a1", "a2"]}))
    rid_b = engine.submit(QSGRequest(prompt="b", options={"chunks": ["b1", "b2"]}))

    events, done_ids = _collect_events(engine, [rid_a, rid_b], timeout_s=1.0)

    payload = [(rid, text) for rid, text, done, _ in events if text and not done]
    assert payload[:4] == [
        (rid_a, "a1"),
        (rid_a, "a2"),
        (rid_b, "b1"),
        (rid_b, "b2"),
    ]
    assert done_ids == {rid_a, rid_b}
    assert engine.metrics_snapshot()["execution_mode"] == "single_stream"

    engine.shutdown()
    runner.join(timeout=1.0)


def test_cancel_marks_request_done_and_tracks_metrics():
    engine = _make_engine()
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(
        QSGRequest(prompt="cancel", options={"chunks": [str(i) for i in range(10)]})
    )
    engine.cancel(request_id)

    deadline = time.time() + 1.0
    terminal = None
    while time.time() < deadline:
        chunk = engine.poll(request_id)
        if chunk is None:
            time.sleep(0.001)
            continue
        if chunk.done:
            terminal = chunk
            break

    assert terminal is not None
    assert terminal.done is True
    assert terminal.error == "cancelled"

    metrics = engine.metrics_snapshot()
    assert metrics["cancelled_requests"] == 1

    engine.shutdown()
    runner.join(timeout=1.0)


def test_shutdown_stops_runner_and_rejects_new_work():
    engine = _make_engine()
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    engine.shutdown(graceful_timeout_s=0.5)
    runner.join(timeout=1.0)
    assert runner.is_alive() is False

    with pytest.raises(RuntimeError, match="is shut down"):
        engine.submit(QSGRequest(prompt="late", options={"chunks": ["x"]}))


def test_priority_scheduler_promotes_high_priority_first():
    config = QSGConfig(
        continuous_batching_enabled=True,
        max_active_requests=1,
        max_pending_requests=8,
        scheduler_policy="priority",
        batch_wait_timeout_ms=1,
    )
    engine = QSGInferenceEngine(
        config=config,
        stream_producer=lambda request: iter((request.options or {}).get("chunks", [])),
    )

    low_id = engine.submit(
        QSGRequest(prompt="low", options={"chunks": ["low"]}, priority=1)
    )
    high_id = engine.submit(
        QSGRequest(prompt="high", options={"chunks": ["high"]}, priority=10)
    )

    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    events, done_ids = _collect_events(engine, [high_id, low_id], timeout_s=1.0)
    payload = [(rid, text) for rid, text, done, _ in events if text and not done]
    assert payload[0] == (high_id, "high")
    assert done_ids == {high_id, low_id}

    engine.shutdown()
    runner.join(timeout=1.0)


def test_metrics_include_state_pager_fields():
    engine = _make_engine()
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(QSGRequest(prompt="metrics", options={"chunks": ["x"]}))
    _collect_events(engine, [request_id], timeout_s=1.0)

    metrics = engine.metrics_snapshot()
    assert "qsg_state_pages_total" in metrics
    assert "qsg_state_fragmentation_ratio" in metrics
    assert "qsg_queue_wait_ms_p95" in metrics
    assert "qsg_scheduler_iteration_ms_p95" in metrics
    assert "qsg_python_hot_path_calls" in metrics
    assert metrics["qsg_latent_requests"] == 0
    assert metrics["qsg_suspended_requests"] == 0
    assert metrics["qsg_native_runtime_authority"] is True

    engine.shutdown()
    runner.join(timeout=1.0)


def test_capture_latent_state_emits_execution_capsule_and_packet():
    engine = _make_engine()
    runner = threading.Thread(target=engine.run_forever, daemon=True)
    runner.start()

    request_id = engine.submit(QSGRequest(prompt="capture", options={"chunks": ["x"]}))
    deadline = time.time() + 1.0
    while time.time() < deadline:
        chunk = engine.poll(request_id)
        if chunk is None:
            time.sleep(0.001)
            continue
        if chunk.text:
            break

    captured = engine.capture_latent_state(request_id)

    assert captured is not None
    assert captured["execution_capsule"]["version"] == 3
    assert captured["execution_capsule"]["request_id"] == request_id
    assert captured["execution_capsule"]["segment_count"] == 2
    assert captured["latent_packet"]["abi_version"] == 3
    assert captured["latent_packet"]["segment_count"] == 2
    assert {item["segment_kind"] for item in captured["latent_packet"]["segments"]} == {
        "branch_state",
        "repo_delta",
    }
    assert (
        captured["latent_packet"]["execution_capsule_id"]
        == captured["execution_capsule"]["capsule_id"]
    )

    engine.shutdown()
    runner.join(timeout=1.0)
