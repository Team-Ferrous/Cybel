import json
import logging

from rich.console import Console

from core.agent import BaseAgent


class _LoopBrain:
    def stream_chat(self, messages, assistant_prefix=""):
        for _ in range(20):
            yield "loop"


class _DummyAgent:
    output_format = "text"

    def __init__(self):
        self.brain = _LoopBrain()
        self.console = Console(quiet=True)
        self.current_mission_id = "test-mission"


def _extract_json_messages(records, event_name):
    payloads = []
    for record in records:
        try:
            payload = json.loads(record.message)
        except Exception:
            continue
        if payload.get("event") == event_name:
            payloads.append(payload)
    return payloads


def test_stream_loop_logs_trigger_context(caplog, monkeypatch):
    monkeypatch.setenv("ANVIL_STREAM_TRACE", "1")
    agent = _DummyAgent()

    with caplog.at_level(logging.INFO):
        response = BaseAgent._stream_response(
            agent,
            messages=[{"role": "user", "content": "test"}],
            is_streaming_ui=True,
            callback=lambda _event: None,
        )

    loop_events = _extract_json_messages(caplog.records, "stream.loop.detected")
    assert loop_events, "expected stream.loop.detected event"
    loop_payload = loop_events[0]
    assert loop_payload["metrics"]["detector"] in {
        "single_word_repeat",
        "sequence_repeat",
    }
    assert loop_payload["metrics"]["history"]

    complete_events = _extract_json_messages(caplog.records, "stream.complete")
    assert complete_events, "expected stream.complete event"
    assert complete_events[-1]["metrics"]["loop_break"] is True
    assert "[SYSTEM: Streaming loop terminated.]" not in response
    assert "<tool_call>" not in response
    assert response.strip()
