import unittest
from unittest.mock import MagicMock, patch
from core.delegation import SubAgent
from rich.console import Console


class TestDelegationFix(unittest.TestCase):
    def test_console_restoration(self):
        # 1. Setup mock console
        mock_console = MagicMock(spec=Console)
        mock_console.quiet = False

        # 2. Setup SubAgent
        # We need to mock run_loop to avoid actual inference
        with patch("core.agent.BaseAgent.run_loop"):
            agent = SubAgent("test task", quiet=True, console=mock_console)

            # Verify initial state within SubAgent
            self.assertTrue(agent.quiet)

            # 3. Execute (should silence and then restore)
            # Mock history to have a response
            agent.history.add_message("assistant", "<thinking>Thought</thinking>Result")

            agent.execute()

            # 4. Verify console.quiet was restored to False
            self.assertFalse(mock_console.quiet)

    def test_thinking_fallback(self):
        mock_console = MagicMock(spec=Console)
        mock_console.quiet = False

        with patch("core.agent.BaseAgent.run_loop"):
            agent = SubAgent("test task", quiet=True, console=mock_console)

            # Case: Response ONLY has thinking
            agent.history.add_message(
                "assistant", '<thinking type="planning">This is the plan</thinking>'
            )

            summary = agent.execute()

            # Should fallback to the thinking content
            self.assertEqual(summary, "This is the plan")


if __name__ == "__main__":
    unittest.main()
