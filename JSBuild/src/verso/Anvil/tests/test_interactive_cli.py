import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add root to pythonpath
sys.path.append(os.getcwd())

from cli.repl import AgentREPL


class TestInteractiveCLI(unittest.TestCase):
    def setUp(self):
        # Mock dependencies to avoid actual network/screen calls
        with patch("core.agent.DeterministicOllama"), patch(
            "core.agent.ConversationHistory"
        ), patch("cli.repl.SaguaroSubstrate"), patch("cli.repl.PromptSession"):

            self.repl = AgentREPL()
            self.repl.console = MagicMock()
            self.repl.history = MagicMock()

    def test_command_registration(self):
        # specific commands should be registered
        cmd = self.repl.command_registry.get_command("help")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.name, "help")

        cmd = self.repl.command_registry.get_command("agent")
        self.assertIsNotNone(cmd)

    def test_help_command(self):
        # Dispatch /help
        res = self.repl.command_registry.dispatch("/help", self.repl)
        self.assertTrue(res)
        self.repl.console.print.assert_called()

    @patch("cli.repl.AgentREPL.run_mission")
    def test_agent_command(self, mock_run_mission):
        # Dispatch /agent
        res = self.repl.command_registry.dispatch("/agent write code", self.repl)
        self.assertTrue(res)

        # Should verify mission started
        mock_run_mission.assert_called_with("write code")

    def test_unknown_command(self):
        res = self.repl.command_registry.dispatch("/unknown", self.repl)
        self.assertTrue(res)  # It consumed the input
        self.repl.console.print.assert_called()

    def test_unwired_command(self):
        self.repl.saguaro.execute_command.return_value = "{\"status\":\"ok\"}"
        res = self.repl.command_registry.dispatch("/unwired --format json", self.repl)
        self.assertTrue(res)
        self.repl.saguaro.execute_command.assert_called_with("unwired --format json")

    @patch("cli.commands.swarm.SequentialSwarmCoordinator")
    def test_swarm_command(self, mock_coordinator_cls):
        coordinator = mock_coordinator_cls.return_value
        coordinator.execute_swarm = AsyncMock(
            return_value={
                "status": "completed",
                "final_summary": "Swarm objective completed.",
                "results": [],
            }
        )

        res = self.repl.command_registry.dispatch(
            "/swarm --topology mesh --agents planner,coder Analyze auth boundaries",
            self.repl,
        )
        self.assertTrue(res)
        coordinator.execute_swarm.assert_called_once()

    def test_thinking_toggle(self):
        self.repl.show_thinking = True
        self.repl.command_registry.dispatch("/thinking off", self.repl)
        self.assertFalse(self.repl.show_thinking)

        self.repl.command_registry.dispatch("/thinking on", self.repl)
        self.assertTrue(self.repl.show_thinking)


if __name__ == "__main__":
    unittest.main()
