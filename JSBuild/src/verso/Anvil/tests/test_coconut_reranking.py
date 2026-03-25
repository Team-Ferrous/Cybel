from unittest.mock import MagicMock

import numpy as np

from core.agent import BaseAgent
from core.unified_chat_loop import UnifiedChatLoop


def _build_mock_loop():
    mock_agent = MagicMock(spec=BaseAgent)
    mock_agent.console = MagicMock()
    mock_agent.brain = MagicMock()
    mock_agent.registry = MagicMock()
    mock_agent.history = MagicMock()
    mock_agent.history.get_messages.return_value = []
    mock_agent.semantic_engine = MagicMock()
    mock_agent.approval_manager = MagicMock()
    mock_agent.name = "TestAgent"
    mock_agent._stream_response.return_value = "ok"
    return UnifiedChatLoop(mock_agent)


def test_coconut_reranking_reorders_evidence():
    loop = UnifiedChatLoop.__new__(UnifiedChatLoop)
    loop.thinking_system = MagicMock()
    loop.thinking_system.rank_evidence.return_value = [(1, 0.9), (0, 0.2)]
    evidence = {"file_contents": {"a.py": "alpha", "b.py": "beta"}}

    ordered = UnifiedChatLoop._rerank_evidence_with_coconut(
        loop, evidence, np.array([1.0, 0.0], dtype=np.float32)
    )

    assert ordered == ["b.py", "a.py"]
    assert list(evidence["file_contents"].keys()) == ["b.py", "a.py"]


def test_synthesis_prompt_contains_coconut_insight_block():
    loop = _build_mock_loop()
    loop.enhanced_mode = False
    loop._format_evidence = MagicMock(return_value="evidence\n" * 50)

    captured = {}

    def fake_grounded_prompt(**kwargs):
        captured["subagent_context"] = kwargs.get("subagent_context", "")
        return "grounded prompt"

    loop._build_grounded_synthesis_prompt = fake_grounded_prompt

    evidence = {
        "request_type": "question",
        "question_type": "architecture",
        "file_contents": {"core/x.py": "x = 1\n" * 40},
        "codebase_files": ["core/x.py"],
        "coconut_amplitudes": [0.8, 0.2],
        "coconut_reranked_files": ["core/x.py"],
    }

    loop._synthesize_answer_core("Explain architecture", evidence)

    assert "[COCONUT REASONING INSIGHT]" in captured["subagent_context"]
