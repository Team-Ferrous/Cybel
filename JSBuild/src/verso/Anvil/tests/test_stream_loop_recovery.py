from rich.console import Console

from core.agent import BaseAgent


class _LoopBrain:
    def stream_chat(self, messages, assistant_prefix=""):
        # Repetitive stream to trigger loop detector.
        for _ in range(20):
            yield "loop"


class _DummyAgent:
    output_format = "text"

    def __init__(self):
        self.brain = _LoopBrain()
        self.console = Console(quiet=True)


def test_stream_loop_recovery_returns_valid_prefix():
    agent = _DummyAgent()
    response = BaseAgent._stream_response(
        agent,
        messages=[{"role": "user", "content": "test"}],
        is_streaming_ui=True,
        callback=lambda _event: None,
    )
    assert "[SYSTEM: Streaming loop terminated.]" in response
