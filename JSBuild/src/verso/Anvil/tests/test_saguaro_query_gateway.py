from __future__ import annotations

import time

from saguaro.query.gateway import (
    SessionGovernor,
    ensure_gateway_started,
    read_gateway_state,
    request_gateway,
    stop_gateway,
)


def test_session_governor_enforces_queue_limit() -> None:
    governor = SessionGovernor(active_limit=1, queue_limit=1)
    ok, reason = governor.acquire(timeout_seconds=0.1)
    assert ok is True
    assert reason is None

    ok, reason = governor.acquire(timeout_seconds=0.0)
    assert ok is False
    assert reason in {"queue_full", "queue_timeout"}

    governor.release()


def test_query_gateway_lifecycle_supports_status_and_shutdown(tmp_path) -> None:
    state = ensure_gateway_started(str(tmp_path), wait_seconds=10.0)
    assert state["status"] == "running"

    status_payload = request_gateway(
        str(tmp_path),
        {"action": "status", "timeout_seconds": 2.0},
        start_if_missing=False,
    )
    assert status_payload["status"] == "ok"
    assert status_payload["gateway"]["status"] == "running"
    assert status_payload["gateway"]["metrics"]["prewarm_state"] in {
        "pending",
        "warming",
        "ready",
        "error",
    }

    stop_payload = stop_gateway(str(tmp_path))
    assert stop_payload["status"] == "ok"

    deadline = time.time() + 5.0
    while time.time() < deadline:
        stopped = read_gateway_state(str(tmp_path))
        if stopped["status"] == "stopped":
            break
        time.sleep(0.05)
    assert read_gateway_state(str(tmp_path))["status"] == "stopped"
