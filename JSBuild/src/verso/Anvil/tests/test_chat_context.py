import unittest
from unittest.mock import MagicMock, patch
from core.agent import BaseAgent
from rich.console import Console


class TestChatContext(unittest.TestCase):
    def setUp(self):
        self.console = Console(quiet=True)
        # Mock brain to avoid actual inference
        self.mock_brain = MagicMock()
        self.mock_brain.stream_chat.return_value = iter(["Hello!"])

        with patch("core.agent.DeterministicOllama", return_value=self.mock_brain):
            with patch("core.agent.SemanticEngine") as mock_engine_cls:
                self.agent = BaseAgent(name="TestAgent", console=self.console)
                self.mock_engine = mock_engine_cls.return_value
                self.agent.semantic_engine = self.mock_engine

    def test_simple_chat_injects_context(self):
        # Setup mock semantic engine
        self.mock_engine.get_context_for_objective.return_value = [
            "file1.py",
            "file2.py",
        ]
        self.mock_engine.get_skeleton.side_effect = lambda x: f"Skeleton of {x}"

        # Call simple_chat
        user_input = "Tell me about file1"
        self.agent.simple_chat(user_input)

        # Verify brain was called with context-enriched messages
        args, kwargs = self.mock_brain.stream_chat.call_args
        messages = args[0]
        system_msg = next(m for m in messages if m["role"] == "system")

        self.assertIn("# RELEVANT WORKSPACE CONTEXT", system_msg["content"])
        self.assertIn("File: file1.py", system_msg["content"])
        self.assertIn("Skeleton of file1.py", system_msg["content"])
        self.assertIn("File: file2.py", system_msg["content"])
        self.assertIn("Skeleton of file2.py", system_msg["content"])


if __name__ == "__main__":
    unittest.main()
