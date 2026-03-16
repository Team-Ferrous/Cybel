from core.pipelines import PipelineManager


class _BrainStub:
    def __init__(self) -> None:
        self.chat_calls = []
        self.stream_calls = []

    def chat(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return "ok"

    def stream_chat(self, messages, **kwargs):
        self.stream_calls.append((messages, kwargs))
        yield "ok"


def test_pipeline_manager_routes_request_types() -> None:
    manager = PipelineManager(_BrainStub())

    assert manager.resolve_pipeline_name("question") == "review"
    assert manager.resolve_pipeline_name("investigation") == "review"
    assert manager.resolve_pipeline_name("modification") == "deterministic"
    assert manager.resolve_pipeline_name("deletion") == "deterministic"
    assert manager.resolve_pipeline_name("creation", user_input="Create a new module") == "generation"
    assert manager.resolve_pipeline_name(
        "creation",
        user_input="Create a poem about telemetry",
    ) == "creative"
    assert manager.resolve_pipeline_name(
        "conversational",
        user_input="Brainstorm a tagline",
    ) == "creative"


def test_pipeline_manager_passes_resolved_kwargs_to_brain() -> None:
    brain = _BrainStub()
    manager = PipelineManager(brain)
    messages = [{"role": "user", "content": "Explain the architecture"}]

    assert manager.chat(messages, request_type="question") == "ok"
    list(manager.stream_chat(messages, request_type="modification"))

    assert brain.chat_calls[0][1]["seed"] == 720720
    assert brain.chat_calls[0][1]["temperature"] == 0.0
    assert brain.stream_calls[0][1]["seed"] == 42
    assert brain.stream_calls[0][1]["temperature"] == 0.0
