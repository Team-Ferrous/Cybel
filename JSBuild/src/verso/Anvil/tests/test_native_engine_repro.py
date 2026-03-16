import unittest
from unittest.mock import patch
import sys
import os

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from core.native.engine import NativeInferenceEngine


class TestNativeEngineRepro(unittest.TestCase):
    @patch("core.native.engine.Llama")
    @patch("core.native.engine.GGUFModelLoader")
    def test_loop_detection_and_retry(self, MockLoader, MockLlama):
        print("Running test_loop_detection_and_retry...")
        # Setup mock LLM
        mock_llm = MockLlama.return_value

        # Track calls to enforce retry behavior
        self.call_count = 0

        def looping_generate(tokens, **kwargs):
            self.call_count += 1
            print(f"  Mocking generate (call {self.call_count})...")
            if self.call_count == 1:
                # First attempt: loop single token 11 (RepetitionTracker triggers at 10 repeats)
                for _ in range(12):
                    yield 11
            else:
                # Retry: return success tokens
                yield 20
                yield 21

        mock_llm.generate.side_effect = looping_generate
        mock_llm.token_eos.return_value = 2

        # Init engine
        print("Initializing NativeInferenceEngine...")
        engine = NativeInferenceEngine("dummy_path")

        # Run stream
        print("Starting generate_stream...")
        tokens = [1, 2, 3]
        output_tokens = []

        # Collect tokens from stream
        for t in engine.generate_stream(tokens, max_new_tokens=15):
            output_tokens.append(t)

        print(f"Output tokens: {output_tokens}")

        # Verify:
        # Final output should contain tokens from the retry (20, 21)
        self.assertIn(20, output_tokens)
        self.assertIn(21, output_tokens)

        # Should contain 5 of token 11 (the 6th one triggered the exception)
        self.assertEqual(output_tokens.count(11), 5)

        # Verify retry count
        self.assertEqual(self.call_count, 2)
        print("✓ Loop detection and retry successful")


if __name__ == "__main__":
    unittest.main()
