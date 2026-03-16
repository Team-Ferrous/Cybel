import sys
import os
import numpy as np
import unittest

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.native.coconut_bridge import CoconutNativeBridge
from core.qsg.grover import GroverAmplifier


class TestCoconutGrover(unittest.TestCase):
    def test_bridge_loading(self):
        """Verify that the native bridge can be loaded and functions called."""
        try:
            bridge = CoconutNativeBridge()
            self.assertIsNotNone(bridge)
            print("Successfully loaded CoconutNativeBridge")
        except Exception as e:
            self.fail(f"Failed to load CoconutNativeBridge: {e}")

    def test_expand_and_aggregate(self):
        """Verify round-trip expand and aggregate."""
        bridge = CoconutNativeBridge()
        dim = 4096
        batch_size = 1
        num_paths = 4

        hidden_state = np.random.normal(0, 1, (batch_size, dim)).astype(np.float32)
        paths = bridge.expand_paths(hidden_state, num_paths)

        self.assertEqual(paths.shape, (batch_size, num_paths, dim))

        # Identity amplitudes
        amplitudes = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
        output = bridge.aggregate_paths(paths, amplitudes)

        self.assertEqual(output.shape, (batch_size, dim))
        # Since we added noise, it won't be EXACTLY same, but should be close
        # Actually coconut_expand_paths adds noise based on path index
        np.testing.assert_allclose(output[0], paths[0, 0], atol=1e-5)

    def test_evolve(self):
        """Verify evolution kernel."""
        bridge = CoconutNativeBridge()
        dim = 4096
        hidden_dim = 11008
        num_paths = 2

        paths = np.random.normal(0, 1, (1, num_paths, dim)).astype(np.float32)
        gamma = np.ones(dim, dtype=np.float32)
        beta = np.zeros(dim, dtype=np.float32)
        w1 = np.random.normal(0, 0.01, (dim, hidden_dim)).astype(np.float32)
        b1 = np.zeros(hidden_dim, dtype=np.float32)
        w2 = np.random.normal(0, 0.01, (hidden_dim, dim)).astype(np.float32)
        b2 = np.zeros(dim, dtype=np.float32)

        original_paths = paths.copy()
        bridge.evolve_paths(paths, gamma, beta, w1, b1, w2, b2, hidden_dim)

        # Paths should have changed
        self.assertFalse(np.allclose(paths, original_paths))

    def test_grover_resonance(self):
        """Verify Grover resonance scoring (mocked)."""

        # Simplified mock of semantic engine
        class MockSemanticEngine:
            def get_context_for_objective(self, obj):
                return ["engine.py", "coconut.py"]

        engine = MockSemanticEngine()
        amplifier = GroverAmplifier(semantic_engine=engine)

        logits = np.array([[0.1, 0.1, 1.0, 1.0]], dtype=np.float32)
        tokens = ["engine", "coconut", "banana", "apple"]

        # Without resonance
        probs_std = amplifier.amplify_logits(logits)

        # With resonance
        probs_res = amplifier.amplify_with_resonance(
            logits, tokens, "reason about coconut"
        )

        # "coconut" (idx 1) should have higher prob in probs_res than probs_std
        print(f"Standard Probs: {probs_std}")
        print(f"Resonance Probs: {probs_res}")

        self.assertTrue(probs_res[0, 1] > probs_std[0, 1])


if __name__ == "__main__":
    unittest.main()
