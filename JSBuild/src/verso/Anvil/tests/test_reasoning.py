import unittest
import numpy as np
from core.qsg.config import QSGConfig
from core.qsg.generator import QSGGenerator
from core.reasoning.coconut import ContinuousThoughtBlock


class TestCoCoNutReasoning(unittest.TestCase):
    def test_thought_block_shape(self):
        # 1. Verify output shape preserved
        block = ContinuousThoughtBlock(embedding_dim=64, num_paths=4, steps=2)

        context = np.random.randn(2, 64).astype(np.float32)  # Batch=2
        refined = block.explore(context)

        self.assertEqual(refined.shape, (2, 64))
        # Ensure it changed (not identity)
        self.assertFalse(np.allclose(context, refined))

    def test_generator_integration_enabled(self):
        # 2. Verify integration in Generator
        vocab_dim = 32
        vocab_size = 100
        vocab_emb = np.random.randn(vocab_size, vocab_dim).astype(np.float32)

        config = QSGConfig(use_coconut_reasoning=True)
        generator = QSGGenerator(config, vocab_emb)

        self.assertIsNotNone(generator.thought_block)

        context = np.random.randn(1, vocab_dim).astype(np.float32)

        # Should run without error
        tokens, probs = generator.generate_draft(context, seq_len=4)
        self.assertEqual(tokens.shape, (1, 4))

    def test_generator_integration_disabled(self):
        # 3. Verify it's None when disabled
        vocab_dim = 32
        vocab_size = 100
        vocab_emb = np.random.randn(vocab_size, vocab_dim).astype(np.float32)

        config = QSGConfig(use_coconut_reasoning=False)
        generator = QSGGenerator(config, vocab_emb)

        self.assertIsNone(generator.thought_block)


if __name__ == "__main__":
    unittest.main()
