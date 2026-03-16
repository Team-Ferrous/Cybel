import unittest
from unittest.mock import MagicMock, patch
import json

from agents.unified_master import UnifiedMasterAgent


class TestUnifiedMasterAgent(unittest.TestCase):

    def setUp(self):
        # Mock BaseAgent dependencies
        self.mock_console = MagicMock()
        self.mock_brain = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_history = MagicMock()

        # UnifiedMasterAgent inherits from BaseAgent, so we can mock its __init__ to inject mocks
        # Or, we can create a mock BaseAgent instance and pass it during UnifiedMasterAgent init
        # For simplicity, let's directly mock the attributes UnifiedMasterAgent will try to access from BaseAgent

        # Patch BaseAgent's __init__ to avoid full initialization and inject our mocks
        with patch("core.agent.BaseAgent.__init__", return_value=None):
            self.agent = UnifiedMasterAgent(
                console=self.mock_console,
                brain=self.mock_brain,
                # These would typically be initialized by BaseAgent's __init__
                # We are setting them directly as we mock BaseAgent.__init__
                # In a real scenario, UnifiedMasterAgent would inherit these.
            )
            self.agent.console = self.mock_console
            self.agent.brain = self.mock_brain
            self.agent.registry = self.mock_registry
            self.agent.history = self.mock_history
            self.agent.config = MagicMock()  # Mock config if accessed
            self.agent._execute_tool = MagicMock(
                return_value="Sub-agent task completed successfully."
            )
            self.agent.quality_gate = MagicMock()
            self.agent.quality_gate.evaluate.return_value = {"accepted": True}
            self.agent._ensure_runtime_compliance_context = MagicMock(
                return_value={
                    "trace_id": "trace-1",
                    "evidence_bundle_id": "bundle-1",
                    "waiver_ids": [],
                    "waiver_id": None,
                    "red_team_required": False,
                }
            )
            self.agent.hook_registry = MagicMock()

    def test_run_mission_successful_execution(self):
        objective = "Test objective"

        # Mock plan decomposition
        mock_plan = [
            {"id": 1, "role": "researcher", "task": "Research topic A."},
            {"id": 2, "role": "implementer", "task": "Implement feature B."},
        ]
        self.mock_brain.generate.side_effect = [
            json.dumps(mock_plan),  # For plan decomposition
            "YES",  # For validation of task 1
            "YES",  # For validation of task 2
        ]

        # Mock sub-agent tool execution
        # Run the mission
        self.agent.run_mission(objective)

        # Assertions
        self.mock_brain.generate.assert_any_call(
            unittest.mock.ANY
        )  # Check plan generation
        self.agent._execute_tool.assert_any_call(
            {
                "name": "execute_subagent_task",
                "arguments": {
                    "role": "researcher",
                    "task": "Research topic A.",
                    "aal": unittest.mock.ANY,
                    "domains": unittest.mock.ANY,
                    "compliance": unittest.mock.ANY,
                },
            }
        )
        self.agent._execute_tool.assert_any_call(
            {
                "name": "execute_subagent_task",
                "arguments": {
                    "role": "implementer",
                    "task": "Implement feature B.",
                    "aal": unittest.mock.ANY,
                    "domains": unittest.mock.ANY,
                    "compliance": unittest.mock.ANY,
                },
            }
        )

        # Check that validation was called for each task
        self.assertGreaterEqual(self.mock_brain.generate.call_count, 3)

        # Check console output for success messages
        self.mock_console.print.assert_any_call(
            unittest.mock.ANY, style="bold green"
        )  # Mission complete
        self.mock_console.print.assert_any_call(
            unittest.mock.ANY, style="green"
        )  # Task completed

    def test_run_mission_failed_plan_decomposition(self):
        objective = "Test objective with bad plan"
        self.mock_brain.generate.return_value = "NOT A JSON STRING"  # Invalid plan

        self.agent.run_mission(objective)

        self.mock_console.print.assert_any_call(
            unittest.mock.ANY, style="bold red"
        )  # Mission aborted

    def test_run_mission_task_validation_failure(self):
        objective = "Test objective with failing task"

        mock_plan = [
            {"id": 1, "role": "researcher", "task": "Research topic A."},
            {"id": 2, "role": "implementer", "task": "Implement feature B."},
        ]
        self.mock_brain.generate.side_effect = [
            json.dumps(mock_plan),  # For plan decomposition
            "NO: This task was incomplete.",  # Validation fails for task 1
        ]

        self.agent._execute_tool.return_value = "Sub-agent produced faulty output."
        self.agent.recovery.handle_failure = MagicMock(return_value=None)

        # Run the mission
        self.agent.run_mission(objective)

        # Assertions
        self.agent._execute_tool.assert_called_once_with(
            {
                "name": "execute_subagent_task",
                "arguments": {
                    "role": "researcher",
                    "task": "Research topic A.",
                    "aal": unittest.mock.ANY,
                    "domains": unittest.mock.ANY,
                    "compliance": unittest.mock.ANY,
                },
            }
        )
        self.mock_console.print.assert_any_call(
            unittest.mock.ANY, style="bold red"
        )  # Task FAILED validation
        self.mock_console.print.assert_any_call(
            "[!] Mission aborted due to task failure.", style="bold red"
        )  # Mission aborted
        # Ensure second task was not attempted
        self.agent._execute_tool.assert_called_once()


if __name__ == "__main__":
    unittest.main()
