import unittest
from unittest.mock import MagicMock, patch
from core.agent import BaseAgent
from rich.console import Console


class TestChatEnhanced(unittest.TestCase):
    def setUp(self):
        self.console = Console(quiet=True)
        # Mock brain to avoid actual inference
        self.mock_brain = MagicMock()

        # Mock tool schemas
        self.mock_tool_schemas = [
            {"name": "web_search", "parameters": {}},
            {"name": "write_file", "parameters": {}},  # Blacklisted
            {"name": "web_fetch", "parameters": {}},
        ]

        with patch("core.agent.DeterministicOllama", return_value=self.mock_brain):
            with patch("core.agent.SemanticEngine") as mock_engine_cls:
                self.agent = BaseAgent(name="TestAgent", console=self.console)
                self.agent.tool_schemas = self.mock_tool_schemas
                self.mock_engine = mock_engine_cls.return_value
                self.agent.semantic_engine = self.mock_engine

    def test_simple_chat_whitelists_tools(self):
        self.mock_brain.stream_chat.return_value = iter(["Hello!"])

        self.agent.simple_chat("Hi")

        # Verify brain was called with whitelisted tools only
        args, kwargs = self.mock_brain.stream_chat.call_args
        messages = args[0]
        system_msg = next(m for m in messages if m["role"] == "system")

        self.assertIn("web_search", system_msg["content"])
        self.assertIn("web_fetch", system_msg["content"])
        self.assertNotIn("write_file", system_msg["content"])

    def test_simple_chat_handles_tool_loop(self):
        # Step 1: Return a tool call
        tool_call = '<tool_call>\n{"name": "web_search", "arguments": {"query": "weather"}}\n</tool_call>'
        self.mock_brain.stream_chat.side_effect = [
            iter([tool_call]),
            iter(["The weather is sunny."]),
        ]

        # Mock tool execution
        with patch.object(
            self.agent, "_execute_tool", return_value='{"results": "sunny"}'
        ):
            response = self.agent.simple_chat("What is the weather?")

        self.assertEqual(response, "The weather is sunny.")
        self.assertEqual(self.mock_brain.stream_chat.call_count, 2)

        # Verify tool result was added to history
        # History: [user, assistant (tool_call), tool, assistant (final)]
        history_msgs = self.agent.history.get_messages()
        self.assertEqual(history_msgs[-2]["role"], "tool")
        self.assertIn("sunny", history_msgs[-2]["content"])


if __name__ == "__main__":
    unittest.main()
