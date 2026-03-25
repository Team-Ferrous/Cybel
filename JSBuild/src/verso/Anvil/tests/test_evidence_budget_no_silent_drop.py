from unittest.mock import MagicMock

from core.agent import BaseAgent
from core.unified_chat_loop import UnifiedChatLoop


def _make_loop():
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


def test_subagent_analysis_truncates_not_drops():
    loop = _make_loop()
    analysis = "important finding\n" * 3000
    evidence = {
        "subagent_analysis": analysis,
        "codebase_files": ["core/agent.py"],
        "file_contents": {},
    }

    formatted = loop._format_evidence(evidence, token_budget=300)
    assert "## File Analysis Subagent Findings" in formatted
    assert "important finding" in formatted


def test_coconut_strategy_selects_tool_mode():
    loop = _make_loop()
    structure = loop._select_tool_strategy([0.9, 0.05, 0.05], "architecture", {})
    implementation = loop._select_tool_strategy([0.05, 0.9, 0.05], "simple", {})
    integration = loop._select_tool_strategy([0.05, 0.05, 0.9], "investigation", {})

    assert structure["name"] == "structure"
    assert implementation["name"] == "implementation"
    assert integration["name"] == "integration"
