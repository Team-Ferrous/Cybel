"""
Tests for the production LoopOrchestrator and AgentOrchestrator.

The production LoopOrchestrator (core.orchestrator.loop_orchestrator) always
routes to UnifiedChatLoop. The legacy complexity-based router
(core.loops.orchestrator) is deprecated and removed.
"""
import unittest
from unittest.mock import MagicMock, patch


class TestLoopOrchestrator(unittest.TestCase):
    """Tests for the production LoopOrchestrator."""
    
    def test_orchestrator_imports(self):
        """Verify the production LoopOrchestrator can be imported."""
        from core.orchestrator.loop_orchestrator import LoopOrchestrator
        self.assertIsNotNone(LoopOrchestrator)
    
    def test_orchestrator_has_unified_loop(self):
        """Verify orchestrator initializes with UnifiedChatLoop."""
        from core.orchestrator.loop_orchestrator import LoopOrchestrator
        
        # Create mocks
        mock_agent = MagicMock()
        mock_agent.enhanced_loop_enabled = True
        mock_saguaro = MagicMock()
        mock_token_manager = MagicMock()
        mock_renderer = MagicMock()
        mock_renderer.console = MagicMock()
        
        with patch("core.orchestrator.loop_orchestrator.UnifiedChatLoop") as mock_loop:
            orchestrator = LoopOrchestrator(
                agent=mock_agent,
                saguaro=mock_saguaro,
                token_manager=mock_token_manager,
                renderer=mock_renderer,
            )
            
            # UnifiedChatLoop should be instantiated
            mock_loop.assert_called_once()
            self.assertIsNotNone(orchestrator.unified_loop)


class TestAgentOrchestrator(unittest.TestCase):
    """Tests for the AgentOrchestrator (task graph executor)."""
    
    def test_orchestrator_imports(self):
        """Verify the AgentOrchestrator can be imported."""
        from core.orchestrator.scheduler import AgentOrchestrator
        self.assertIsNotNone(AgentOrchestrator)
    
    def test_orchestrator_constructor(self):
        """Verify AgentOrchestrator can be constructed with mocks."""
        from core.orchestrator.scheduler import AgentOrchestrator
        
        mock_brain = MagicMock()
        mock_console = MagicMock()
        mock_semantic_engine = MagicMock()
        
        orchestrator = AgentOrchestrator(
            brain=mock_brain,
            semantic_engine=mock_semantic_engine,
            console=mock_console,
        )
        
        self.assertIsNotNone(orchestrator)
        self.assertEqual(orchestrator.brain, mock_brain)


class TestLegacyOrchestratorRemoved(unittest.TestCase):
    """Ensure the legacy orchestrator has been removed."""
    
    def test_legacy_orchestrator_not_importable(self):
        """The legacy core.loops.orchestrator should not exist."""
        with self.assertRaises(ImportError):
            from core.loops.orchestrator import LoopOrchestrator  # noqa: F401


if __name__ == "__main__":
    unittest.main()
