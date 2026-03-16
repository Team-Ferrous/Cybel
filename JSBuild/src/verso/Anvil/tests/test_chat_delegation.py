import unittest
from unittest.mock import MagicMock, patch
from core.agent import BaseAgent
from rich.console import Console


class TestChatDelegation(unittest.TestCase):
    def setUp(self):
        self.console = Console(quiet=True)
        self.mock_brain = MagicMock()

        # Mock tool schemas including the new ones
        self.mock_tool_schemas = [
            {"name": "search_web", "parameters": {}},
            {"name": "delegate", "parameters": {}},
            {"name": "analyze_codebase", "parameters": {}},
            {"name": "update_memory_bank", "parameters": {}},
            {"name": "list_dir", "parameters": {}},
        ]

        with patch("core.agent.DeterministicOllama", return_value=self.mock_brain):
            with patch("core.agent.SemanticEngine") as mock_engine_cls:
                self.agent = BaseAgent(name="TestAgent", console=self.console)
                self.agent.tool_schemas = self.mock_tool_schemas
                self.mock_engine = mock_engine_cls.return_value
                self.agent.semantic_engine = self.mock_engine

    def test_simple_chat_whitelist_includes_delegation(self):
        self.mock_brain.stream_chat.return_value = iter(["Hello!"])

        self.agent.simple_chat("Hi")

        # Verify brain was called with whitelisted tools including delegation
        args, kwargs = self.mock_brain.stream_chat.call_args
        messages = args[0]
        system_msg = next(m for m in messages if m["role"] == "system")

        self.assertIn("delegate", system_msg["content"])
        self.assertIn("analyze_codebase", system_msg["content"])
        self.assertIn("update_memory_bank", system_msg["content"])

    def test_simple_chat_contains_evidence_mandate(self):
        self.mock_brain.stream_chat.return_value = iter(["Hello!"])
        self.agent.simple_chat("Hi")

        args, kwargs = self.mock_brain.stream_chat.call_args
        messages = args[0]
        system_msg = next(m for m in messages if m["role"] == "system")

        self.assertIn("VERIFIED EVIDENCE", system_msg["content"])

    @patch("core.task_state.TaskStateManager.start_task")
    @patch("core.task_state.TaskStateManager.end_task")
    def test_simple_chat_updates_task_state(self, mock_end, mock_start):
        self.mock_brain.stream_chat.return_value = iter(["Hello!"])
        self.agent.simple_chat("Hi")

        mock_start.assert_called_once()
        mock_end.assert_called_once()


if __name__ == "__main__":
    unittest.main()
