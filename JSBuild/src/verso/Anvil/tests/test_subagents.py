import unittest
from unittest.mock import MagicMock, patch
from core.agents.researcher import ResearchSubagent

# Fix import for wizard
from core.wizard import SubagentWorkflow


class TestSubAgents(unittest.TestCase):
    def setUp(self):
        self.mock_brain = MagicMock()
        self.mock_console = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_registry.get_schemas.return_value = {
            "tools": [{"name": "web_search"}]
        }

        # Patch BaseAgent's registry initialization to avoid real file ops
        with patch("core.agent.ToolRegistry", return_value=self.mock_registry):
            self.researcher = ResearchSubagent(
                "query", "Parent", self.mock_brain, self.mock_console
            )

    def test_subagent_init(self):
        self.assertEqual(self.researcher.name, "Parent:ResearchSubagent")
        # Check tool filtering
        # ResearchSubagent tools include "web_search", valid
        # self.researcher.tool_schemas should contain web_search
        pass

    def test_workflow_execution(self):
        # Mock class for runtime context
        class MockContext:
            brain = MagicMock()
            console = MagicMock()

        runtime_ctx = MockContext()

        wf = SubagentWorkflow("TestFlow")

        # Mock subagent class
        mock_agent_cls = MagicMock()
        mock_instance = MagicMock()
        mock_instance.run.return_value = {"response": "result content"}
        mock_agent_cls.return_value = mock_instance
        mock_agent_cls.__name__ = "MockAgent"

        wf.add_step(mock_agent_cls, {"input_val": "{val}"}, ["out"])

        ctx = {"val": "test_input"}
        result_ctx = wf.execute(ctx, runtime_ctx)

        self.assertEqual(result_ctx["out"], "result content")
        mock_agent_cls.assert_called()  # Check instantiation
        mock_instance.run.assert_called_with(input_val="test_input")


if __name__ == "__main__":
    unittest.main()
