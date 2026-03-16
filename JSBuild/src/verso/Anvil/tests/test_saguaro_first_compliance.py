import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.agent_mode import AgentMode
from core.agents.repo_analyzer import RepoAnalysisSubagent
from core.agents.researcher import ResearchSubagent
from core.agents.subagent import SubAgent
from core.chat_loop_enterprise import EnterpriseChatLoop
from core.subagents.file_analyst import FileAnalysisSubagent
from saguaro.indexing import backends
from tools.grep import grep


class DummySubAgent(SubAgent):
    tools = ["saguaro_query", "skeleton", "slice", "read_file"]


class TestSaguaroFirstCompliance(unittest.TestCase):
    def _make_subagent(self, cls=DummySubAgent):
        mock_brain = MagicMock()
        mock_console = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = {
            "tools": [
                {"name": "saguaro_query"},
                {"name": "skeleton"},
                {"name": "slice"},
                {"name": "read_file"},
            ]
        }
        with patch("core.agent.ToolRegistry", return_value=mock_registry):
            return cls("investigate authentication flow", "Parent", mock_brain, mock_console)

    def test_repo_analyzer_tools_start_with_saguaro(self):
        self.assertEqual(RepoAnalysisSubagent.tools[0], "saguaro_query")
        self.assertNotIn("grep", RepoAnalysisSubagent.tools)
        self.assertNotIn("list_dir", RepoAnalysisSubagent.tools)
        self.assertNotIn("semantic_search", RepoAnalysisSubagent.tools)

    def test_research_subagent_tools_start_with_saguaro(self):
        self.assertIn("saguaro_query", ResearchSubagent.tools)
        self.assertNotIn("grep", ResearchSubagent.tools)
        self.assertNotIn("semantic_search", ResearchSubagent.tools)

    def test_subagent_default_first_tool_is_saguaro_query(self):
        agent = self._make_subagent()
        tool_call = agent._default_first_tool_call()
        self.assertEqual(tool_call["name"], "saguaro_query")
        self.assertIn("query", tool_call["arguments"])

    def test_subagent_prompt_requires_saguaro_query(self):
        agent = self._make_subagent()
        prompt = agent._build_specialized_system_prompt()
        self.assertIn("ALWAYS use first for repository discovery", prompt)
        self.assertIn("Do NOT use grep, semantic_search, glob, or list_dir", prompt)

    def test_oneshot_demonstrates_saguaro_query(self):
        agent = self._make_subagent()
        oneshot = agent._build_oneshot_messages()
        assistant_messages = [m["content"] for m in oneshot if m["role"] == "assistant"]
        self.assertTrue(any("saguaro_query" in msg for msg in assistant_messages))

    def test_grep_disabled_in_strict_mode(self):
        result = grep("AuthManager")
        self.assertIn("disabled in strict grounding mode", result)

    def test_saguaro_backend_is_strict(self):
        if backends._HAS_NATIVE:
            backend = backends.get_backend()
            self.assertEqual(backends.backend_name(backend), "NativeIndexerBackend")
        else:
            with self.assertRaises(RuntimeError):
                backends.get_backend()

    def test_agent_mode_planning_prioritizes_saguaro_only(self):
        tools = AgentMode.PLANNING.prioritized_tools
        self.assertEqual(tools[0], "saguaro_query")
        self.assertNotIn("grep", tools)
        self.assertNotIn("glob", tools)
        self.assertNotIn("query", tools)

    def test_prompt_templates_use_saguaro_query_hierarchy(self):
        root = Path(__file__).resolve().parents[1]
        for relative in [
            "core/prompts/templates/base_agent_core.md",
            "core/prompts/templates/cognitive.md",
            "core/prompts/templates/general.md",
            "core/prompts/templates/action.md",
        ]:
            content = (root / relative).read_text()
            self.assertIn("saguaro_query", content, relative)
            self.assertNotIn("semantic_search", content, relative)

    def test_enterprise_chat_loop_uses_saguaro_query(self):
        source = Path(EnterpriseChatLoop.__module__.replace(".", "/") + ".py").read_text()
        self.assertIn('dispatch("saguaro_query"', source)
        self.assertNotIn('dispatch("grep"', source)

    def test_file_analyst_prompt_requires_saguaro_query(self):
        mock_parent = MagicMock()
        mock_parent.name = "Parent"
        mock_parent.brain = MagicMock()
        mock_parent.console = MagicMock()
        mock_parent.registry = MagicMock()
        mock_parent.history = MagicMock()
        mock_parent.approval_manager = MagicMock()
        mock_parent.semantic_engine = MagicMock()
        with patch("core.agent.ToolRegistry") as registry_cls:
            registry = MagicMock()
            registry.get_schemas.return_value = {
                "tools": [
                    {"name": "saguaro_query"},
                    {"name": "skeleton"},
                    {"name": "slice"},
                    {"name": "read_file"},
                ]
            }
            registry_cls.return_value = registry
            agent = FileAnalysisSubagent(mock_parent, ["core/agent.py"], "find agent loop")

        self.assertIn("saguaro_query", agent.system_prompt)
        self.assertNotIn("grep", agent.system_prompt)


if __name__ == "__main__":
    unittest.main()
