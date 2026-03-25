import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import os
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_chat_loop import UnifiedChatLoop
from core.agent import BaseAgent


class TestUnifiedChatLoop(unittest.TestCase):

    def setUp(self):
        # Mock BaseAgent dependencies
        self.mock_console = MagicMock()
        self.mock_brain = MagicMock()
        self.mock_registry = MagicMock()
        self.mock_history = MagicMock()
        self.mock_semantic_engine = MagicMock()
        self.mock_approval_manager = MagicMock()

        # Create a mock agent object
        self.mock_agent = MagicMock(spec=BaseAgent)
        self.mock_agent.console = self.mock_console
        self.mock_agent.brain = self.mock_brain
        self.mock_agent.registry = self.mock_registry
        self.mock_agent.history = self.mock_history
        self.mock_agent.semantic_engine = self.mock_semantic_engine
        self.mock_agent.approval_manager = self.mock_approval_manager
        self.mock_agent.pipeline_manager = MagicMock()
        self.mock_agent.pipeline_manager.resolve_generation_kwargs.return_value = {
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        self.mock_agent.name = "TestAgent"

        # IMPORTANT: Mock _stream_response to return a string, as UnifiedChatLoop calls it directly
        self.mock_agent._stream_response.return_value = "Mocked successful response"

        # Instantiate UnifiedChatLoop with the mock agent
        self.chat_loop = UnifiedChatLoop(self.mock_agent)

        # Mock response_cache to avoid disk access and ensure no cache hits
        self.chat_loop.response_cache = MagicMock()
        self.chat_loop.response_cache.get_cached_response.return_value = None

    def test_synthesize_answer_without_subagent_analysis(self):
        """
        Test that _synthesize_answer works correctly when evidence does not
        contain 'subagent_analysis', preventing UnboundLocalError for system_prompt.
        """
        user_input = "Test question"
        # Ensure evidence is substantial enough to pass length checks (>100 chars)
        evidence_content = "code line\n" * 20
        evidence = {
            "codebase_files": ["test.py"],
            "file_contents": {"test.py": evidence_content},
            "request_type": "question",
        }

        # Mock the streaming response (though agent._stream_response mock takes we check agent calls)
        self.mock_brain.stream_chat.return_value = iter(["This is a test response."])

        # We also need to mock context_loader.get_context_summary to avoid errors
        self.chat_loop._context_loader = MagicMock()
        self.chat_loop._context_loader.get_context_summary.return_value = "summary"

        # Call the method
        response = self.chat_loop._synthesize_answer(user_input, evidence)

        # Assertions
        # Verify agent._stream_response was called
        self.mock_agent._stream_response.assert_called()
        system_prompt = self.mock_agent._stream_response.call_args[0][0][0]["content"]
        self.assertIn("## AES Runtime Contract", system_prompt)
        self.assertIn("## Compliance Context", system_prompt)
        self.assertEqual(response, "Mocked successful response")

    def test_generate_synthesis_response_passes_pipeline_kwargs(self):
        payload = {
            "messages": [{"role": "user", "content": "question"}],
            "capability_tier": "tier_2_standard",
            "request_type": "question",
        }
        self.mock_agent._stream_response.reset_mock()
        self.mock_agent.pipeline_manager.resolve_generation_kwargs.return_value = {
            "temperature": 0.0,
            "seed": 720720,
        }

        response = self.chat_loop._generate_synthesis_response(
            "Explain the architecture",
            {"request_type": "question"},
            payload,
            dashboard=None,
        )

        self.mock_agent.pipeline_manager.resolve_generation_kwargs.assert_called_once()
        self.mock_agent._stream_response.assert_called_once_with(
            payload["messages"],
            generation_kwargs={
                "temperature": 0.0,
                "seed": 720720,
                "context_text": "Explain the architecture",
            },
        )
        self.assertEqual(response, "Mocked successful response")

    def test_synthesize_answer_with_coconut(self):
        """
        Test synthesis path when COCONUT is enabled and available.
        """
        # Inject a mock COCONUT instance into the initialized thinking system.
        mock_coconut_instance = MagicMock()
        mock_coconut_instance.explore.return_value = []
        mock_coconut_instance.explore_adaptive.return_value = (
            np.asarray([[0.2] * 512], dtype=np.float32),
            SimpleNamespace(path_amplitudes=[0.75, 0.25], to_dict=lambda: {}),
        )
        mock_coconut_instance.amplitudes = [0.75, 0.25]
        mock_coconut_instance.get_device_info.return_value = {"backend": "native"}
        mock_coconut_instance.config = {"embedding_dim": 512}
        self.chat_loop.thinking_system._coconut = mock_coconut_instance

        user_input = "Test question"
        evidence_content = "code line\n" * 20
        evidence = {
            "codebase_files": ["test.py"],
            "file_contents": {"test.py": evidence_content},
            "coconut_paths": None,
            "coconut_amplitudes": None,
            "complexity_score": 20.0,  # High enough to trigger coconut (>= 10)
            "request_type": "question",  # Not simple/conversational
            "question_type": "architecture",  # Triggers COCONUT for architecture questions
        }

        # Enable COCONUT
        self.chat_loop.thinking_system.coconut_enabled = True

        # Mock embeddings for COCONUT input
        self.mock_brain.embeddings.return_value = [0.1] * 512

        # Mock context checks
        self.chat_loop._context_loader = MagicMock()
        self.chat_loop._context_loader.get_context_summary.return_value = "summary"

        # Call the method
        response = self.chat_loop._synthesize_answer(user_input, evidence)

        # Assertions
        self.mock_agent._stream_response.assert_called()
        self.assertEqual(response, "Mocked successful response")

        # COCONUT instance should be used during synthesis.
        mock_coconut_instance.explore_adaptive.assert_called()

    def test_small_model_fallback_includes_compliance_context(self):
        self.chat_loop.current_compliance_context = {
            "trace_id": "trace-1",
            "evidence_bundle_id": "bundle-1",
            "waiver_id": None,
        }
        evidence = {
            "codebase_files": ["test.py"],
            "file_contents": {"test.py": '"""module doc"""\n\ndef foo():\n    return 1\n'},
            "primary_file": "test.py",
            "skeletons": {"test.py": "def foo()"},
        }

        response = self.chat_loop._synthesize_for_small_model("question", evidence)

        self.assertIn("## Compliance Context", response)
        self.assertIn("trace-1", response)
        self.assertIn("bundle-1", response)

    @patch("core.agents.researcher.ResearchSubagent")
    def test_delegate_research_subagent_carries_latent_payload(self, mock_subagent_cls):
        mock_subagent = mock_subagent_cls.return_value
        mock_subagent.name = "TestResearchSubagent"
        mock_subagent.run.return_value = {
            "summary": "A" * 120,
            "full_response": "full response",
            "files_read": ["core/auth.py"],
            "latent": {
                "state": [0.5, 0.5, 0.0, 0.0],
                "reinjections": 3,
                "tool_signals": [{"tool": "saguaro_query", "confidence": 0.9}],
                "depth_used": 4,
                "seeded_from_master": True,
            },
        }
        self.chat_loop._apply_subagent_quality_gate = lambda evidence, _query: evidence
        self.chat_loop._seed_subagent_guidance = lambda *_args, **_kwargs: None

        evidence = self.chat_loop._delegate_to_research_subagent("Investigate auth routing")

        self.assertIn("subagent_latent_signals", evidence)
        self.assertEqual(evidence.get("subagent_latent_signal_count"), 1)
        self.assertEqual(evidence.get("subagent_latent_reinjections"), 3)
        self.assertTrue(evidence.get("subagent_latent_merged"))

    def test_apply_reinjected_latent_to_embeddings_blends_prior(self):
        evidence = {
            "subagent_latent_signals": [
                {"state": [1.0, 0.0, 0.0, 0.0], "reinjections": 2}
            ]
        }
        embeddings = np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        blended, applied = self.chat_loop._apply_reinjected_latent_to_embeddings(
            embeddings=embeddings,
            evidence=evidence,
            embedding_dim=4,
        )

        self.assertTrue(applied)
        self.assertTrue(evidence.get("latent_reinjection_applied"))
        self.assertGreater(float(blended[0, 0]), 0.0)

    def test_complexity_slot_and_depth_resolution_prefers_adaptive_fields(self):
        profile = SimpleNamespace(score=2, coconut_steps=3, subagent_count=5)
        self.assertEqual(self.chat_loop._complexity_subagent_slots(profile), 4)

        self.chat_loop.current_adaptive_complexity = {"subagent_slots": 4, "coconut_depth": 7}
        self.assertEqual(self.chat_loop._resolve_subagent_slot_count(profile), 4)
        self.assertEqual(
            self.chat_loop._resolve_coconut_depth(profile, self.chat_loop.current_adaptive_complexity),
            7,
        )

    def test_route_question_specialist_architecture_hint(self):
        routing, domains, aal = self.chat_loop._route_question_specialist(
            "Explain the architecture boundaries and module interactions",
            "architecture",
        )

        self.assertEqual(routing.primary_role, "SoftwareArchitectureSubagent")
        self.assertIsInstance(domains, list)
        self.assertTrue(str(aal).startswith("AAL-"))

    def test_verify_modified_files_uses_runtime_control_attempt_budget(self):
        self.chat_loop.current_runtime_control = {
            "verification_max_attempts": 1,
            "posture": "degraded_python_fallback",
        }
        self.chat_loop.verification_loop = MagicMock()
        self.chat_loop.verification_loop.verify_with_retry.return_value = True

        results = self.chat_loop._create_action_results()
        self.chat_loop._verify_modified_files(["src/app.py"], results, dashboard=None)

        self.chat_loop.verification_loop.verify_with_retry.assert_called_once_with(
            ["src/app.py"], max_attempts=1
        )
        self.assertEqual(results["verification"]["max_attempts"], 1)
        self.assertEqual(
            results["verification"]["runtime_posture"], "degraded_python_fallback"
        )

    def test_execute_action_enhanced_persists_compiled_plan(self):
        self.chat_loop.current_compliance_context["trace_id"] = "trace-compiled"
        self.chat_loop.current_runtime_control = {
            "posture": "native_ready",
            "planning_depth": "deep",
            "verification_max_attempts": 2,
        }
        self.chat_loop.black_box.event_store.record_checkpoint = MagicMock()
        self.chat_loop._record_reality_event = MagicMock()
        self.chat_loop._parse_action_plan = MagicMock(
            return_value=[
                {
                    "tool": "edit_file",
                    "args": {"file_path": "src/app.py", "_context_updates": []},
                }
            ]
        )
        self.chat_loop._pre_action_tool_checkpoint = MagicMock(
            return_value={"allowed": True}
        )
        self.chat_loop._apply_pre_action_block = MagicMock(return_value=False)
        self.chat_loop._is_high_assurance_change = MagicMock(return_value=False)
        self.chat_loop._run_parallel_tool_calls = MagicMock(return_value=[])
        self.chat_loop._process_executed_tool_results = MagicMock()
        self.chat_loop._verify_modified_files = MagicMock()
        self.chat_loop._reflect_on_action_execution = MagicMock()
        self.chat_loop.saguaro.active_mission_id = "ws-123"
        self.chat_loop.saguaro._load_sync_receipt = MagicMock(
            return_value={"receipt_id": "sync-456"}
        )
        self.chat_loop.saguaro._sync_receipt_path = MagicMock(
            return_value=".saguaro/workspace_sync.json"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            self.chat_loop.saguaro.root_dir = tmpdir
            results = self.chat_loop._execute_action_enhanced(
                "1. Files to modify\n2. Changes to make\n3. Tools to use\n4. Verification steps",
                "Update src/app.py",
            )

            compiled = results["compiled_plan"]
            self.assertTrue(os.path.exists(compiled["path"]))
            payload = json.loads(Path(compiled["path"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["trace_id"], "trace-compiled")
            self.assertEqual(payload["thread_context"]["workset_id"], "ws-123")
            self.assertEqual(payload["thread_context"]["sync_receipt_id"], "sync-456")
            self.assertEqual(payload["steps"][0]["tool"], "edit_file")

    def test_gather_question_evidence_delegates_non_simple_through_router(self):
        evidence = {"question_type": "architecture"}
        self.chat_loop._delegate_to_question_specialist = MagicMock(
            return_value={
                "subagent_type": "SoftwareArchitectureSubagent",
                "subagent_analysis": "architecture findings",
            }
        )

        self.chat_loop._gather_question_evidence(
            "architecture",
            "Explain how components integrate",
            evidence,
            dashboard=None,
        )

        self.chat_loop._delegate_to_question_specialist.assert_called_once()
        self.assertEqual(
            evidence.get("subagent_type"),
            "SoftwareArchitectureSubagent",
        )

    @patch("core.subagents.file_analyst.FileAnalysisSubagent")
    def test_augment_repo_analysis_with_file_subagent_for_large_sets(
        self, mock_file_analyst_cls
    ):
        analyst = mock_file_analyst_cls.return_value
        analyst.analyze.return_value = {
            "summary": "Deep file analyst summary.",
            "key_files": ["core/agent.py", "core/unified_chat_loop.py"],
            "entities_found": {"Agent": "core/agent.py"},
            "token_usage": 321,
        }

        response, files, payload = self.chat_loop._augment_repo_analysis_with_file_subagent(
            query="Analyze architecture boundaries",
            unique_files=[f"core/file_{i}.py" for i in range(11)],
            response_text="Base repo analysis.",
        )

        self.assertIn("File Analyst Synthesis", response)
        self.assertEqual(files[0], "core/agent.py")
        self.assertEqual(payload.get("token_usage"), 321)
        self.assertEqual(payload.get("entity_count"), 1)

    @patch("core.subagents.file_analyst.FileAnalysisSubagent")
    def test_augment_repo_analysis_with_file_subagent_skips_small_sets(
        self, mock_file_analyst_cls
    ):
        response, files, payload = self.chat_loop._augment_repo_analysis_with_file_subagent(
            query="Analyze architecture boundaries",
            unique_files=[f"core/file_{i}.py" for i in range(6)],
            response_text="Base repo analysis.",
        )

        self.assertEqual(response, "Base repo analysis.")
        self.assertEqual(files, [f"core/file_{i}.py" for i in range(6)])
        self.assertEqual(payload, {})
        mock_file_analyst_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
