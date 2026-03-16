import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to sys.path
sys.path.append(os.getcwd())

from core.wizard import WorkflowWizard
from core.loops.analysis_loop import AnalysisLoop


class TestAnalysisFlows(unittest.TestCase):
    def setUp(self):
        self.mock_agent = MagicMock()
        self.mock_agent.console = MagicMock()
        self.mock_agent.orchestrator = MagicMock()
        self.mock_agent.brain = MagicMock()
        self.mock_agent.registry = MagicMock()

    def test_wizard_code_analysis(self):
        wizard = WorkflowWizard(self.mock_agent)

        # Test the structural logic of the wizard
        with patch("rich.prompt.Prompt.ask", return_value="1"), patch(
            "rich.prompt.Confirm.ask", return_value=False
        ):
            # We need to mock the TEMPLATES access if we want to force choice 1
            # But run() handles the full loop. Let's just test run_code_analysis directly.
            result = wizard.run_code_analysis()
            self.assertEqual(result, "Code Analysis workflow completed.")

    def test_analysis_loop_init(self):
        # Just check if it initializes and has an analyst
        loop = AnalysisLoop(self.mock_agent)
        self.assertIsNotNone(loop.analyst)
        self.assertEqual(loop.analyst.name, "Analyst")


if __name__ == "__main__":
    unittest.main()
