from unittest.mock import MagicMock, patch

import numpy as np

from core.agents.subagent import SubAgent
from core.reasoning.tool_intent_classifier import ToolIntentSignal


class _BrainStub:
    model_name = "granite4:tiny-h"

    def embeddings(self, text: str):
        size = 24
        scale = float((len(text) % 5) + 1)
        return np.ones((size,), dtype=np.float32) * scale


def test_subagent_latent_tool_intent_short_circuits_and_reinjects():
    mock_registry = MagicMock()
    mock_registry.get_schemas.return_value = {
        "tools": [
            {"name": "saguaro_query"},
            {"name": "read_file"},
        ]
    }

    with patch("core.agent.ToolRegistry", return_value=mock_registry):
        agent = SubAgent(
            task="Find auth flow in core/auth.py",
            parent_name="Master",
            brain=_BrainStub(),
            console=MagicMock(),
            quiet=True,
        )

    agent.max_autonomous_steps = 1
    agent.coconut_enabled = True
    agent._build_specialized_system_prompt = lambda: "system"
    agent._build_oneshot_messages = lambda: []
    agent._consume_master_guidance = lambda: ""
    agent._publish_progress = lambda *args, **kwargs: None
    agent._record_tool_result_message = lambda *args, **kwargs: None
    agent._post_shared_finding = lambda *args, **kwargs: None
    agent._stream_response = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("stream path should be bypassed by latent short-circuit")
    )

    executed = []
    agent._execute_tool = (
        lambda tool_call, *a, **k: executed.append(tool_call["name"]) or "ok"
    )

    class _Classifier:
        @staticmethod
        def detect(*_args, **_kwargs):
            return ToolIntentSignal(
                tool_name="saguaro_query",
                confidence=0.97,
                scores={"saguaro_query": 0.97},
            )

    class _Adapter:
        calls = 0

        @staticmethod
        def project_text(_text: str):
            return np.ones((1, 24), dtype=np.float32)

        @classmethod
        def inject(cls, tool_result, pre_tool_state, tool_name):
            cls.calls += 1
            if pre_tool_state is None:
                return np.ones((1, 24), dtype=np.float32)
            return np.asarray(pre_tool_state, dtype=np.float32)

    agent._tool_intent_classifier = _Classifier()
    agent._result_adapter = _Adapter()

    result = agent._isolated_inference("Find auth flow in core/auth.py")
    assert result["error"] is None
    assert "saguaro_query" in executed
    assert _Adapter.calls >= 1
    assert isinstance(result.get("latent"), dict)
    assert result["latent"].get("state_dim") == 24
    assert result["latent"].get("reinjections", 0) >= 1
    assert result.get("latent_state") is not None
    assert len(result.get("latent_tool_signals", [])) >= 1


def test_subagent_can_seed_latent_state_from_master_vector():
    mock_registry = MagicMock()
    mock_registry.get_schemas.return_value = {"tools": [{"name": "saguaro_query"}]}

    with patch("core.agent.ToolRegistry", return_value=mock_registry):
        agent = SubAgent(
            task="Inspect auth flow",
            parent_name="Master",
            brain=_BrainStub(),
            console=MagicMock(),
            quiet=True,
            coconut_context_vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            coconut_depth=5,
        )

    agent._result_adapter = None
    seeded = agent._compute_latent_state("Inspect auth flow")
    assert seeded is not None
    assert seeded.shape == (1, 24)
    assert agent._dynamic_coconut_depth == 5
