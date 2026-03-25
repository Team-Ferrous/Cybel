from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from core.chat_loop_enterprise import EnterpriseChatLoop


def test_enterprise_direct_answer_uses_pipeline_manager() -> None:
    pipeline_manager = MagicMock()
    pipeline_manager.stream_chat.return_value = iter(["ok"])
    history = SimpleNamespace(
        add_message=lambda *args, **kwargs: None,
        get_messages=lambda: [],
    )
    agent = SimpleNamespace(
        console=SimpleNamespace(print=lambda *args, **kwargs: None),
        brain=MagicMock(),
        history=history,
        registry=None,
        semantic_engine=None,
        approval_manager=MagicMock(),
        pipeline_manager=pipeline_manager,
        name="test-agent",
    )

    loop = EnterpriseChatLoop(agent)
    response = loop._direct_answer("Brainstorm a tagline")

    pipeline_manager.stream_chat.assert_called_once()
    _, kwargs = pipeline_manager.stream_chat.call_args
    assert kwargs["request_type"] == "conversational"
    assert kwargs["user_input"] == "Brainstorm a tagline"
    assert response == "ok"
