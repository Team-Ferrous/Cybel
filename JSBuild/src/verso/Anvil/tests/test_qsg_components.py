import unittest
import numpy as np

from core.qsg.holographic_encoder import HolographicEncoder
from core.qsg.hopfield_vocab import HopfieldVocab
from core.qsg.jacobi_refiner import JacobiRefiner
from core.qsg.grover import GroverAmplifier


class TestQSGComponents(unittest.TestCase):
    def test_holographic_encoder(self):
        """
        Tests the HolographicEncoder's ability to create a consistent
        context-pooled embedding from a sequence of token embeddings.
        """
        dim = 32
        seq_len = 5

        encoder = HolographicEncoder(dim)

        # Create a dummy sequence of embeddings
        embeddings = np.random.randn(seq_len, dim).astype(np.float32)

        # Encode the sequence
        pooled_embedding = encoder.encode(embeddings)

        # 1. Check shape
        self.assertEqual(pooled_embedding.shape, (1, dim))

        # 2. Check consistency
        # Encoding the same sequence again should produce the exact same result
        pooled_embedding_2 = encoder.encode(embeddings)
        np.testing.assert_array_equal(pooled_embedding, pooled_embedding_2)

        # 3. Check that different sequences produce different results
        embeddings_2 = np.random.randn(seq_len, dim).astype(np.float32)
        pooled_embedding_3 = encoder.encode(embeddings_2)
        # It's technically possible for a collision, but highly improbable.
        self.assertFalse(np.allclose(pooled_embedding, pooled_embedding_3))

    def test_hopfield_vocab(self):
        """
        Tests the HopfieldVocab's ability to retrieve the correct token
        from a noisy input embedding.
        """
        dim = 16
        vocab_size = 4

        # Create a small, orthogonal vocabulary for clarity
        vocab_embeddings = np.eye(vocab_size, dim, dtype=np.float32)

        hopfield = HopfieldVocab(vocab_embeddings, beta=5.0)

        # Target token is index 2
        target_token_idx = 2
        target_embedding = vocab_embeddings[target_token_idx]

        # Create a noisy version of the target embedding
        noisy_embedding = target_embedding + np.random.normal(0, 0.1, dim)
        noisy_embedding = noisy_embedding.reshape(1, -1)  # Reshape for batch size 1

        # Get token probabilities
        probs = hopfield.get_token_probs(noisy_embedding)

        # 1. Check shape
        self.assertEqual(probs.shape, (1, vocab_size))

        # 2. Check that the highest probability is for the target token
        retrieved_token_idx = np.argmax(probs)
        self.assertEqual(retrieved_token_idx, target_token_idx)

        # 3. Check that the probability is high
        self.assertGreater(probs[0, target_token_idx], 0.8)

    def test_jacobi_refiner(self):
        """
        Tests the JacobiRefiner's ability to smooth a sequence of
        probability distributions, making them more coherent.
        """
        seq_len = 5
        vocab_size = 10

        refiner = JacobiRefiner(iterations=3)

        # Create a deterministic probability sequence
        probs = np.zeros((1, seq_len, vocab_size), dtype=np.float32)
        probs += 1e-6  # Add a small epsilon to avoid zero probabilities

        # Position 2 is confident about token 5
        # Position 3 is confident about token 5
        probs[0, 2, 5] = 0.9
        probs[0, 3, 5] = 0.8

        # Normalize to be valid probability distributions
        probs /= probs.sum(axis=-1, keepdims=True)

        # Refine the probabilities
        refined_probs = refiner.refine(probs)

        # 1. Check shape
        self.assertEqual(refined_probs.shape, (1, seq_len, vocab_size))

        # 2. Check that the refined probabilities are still valid
        self.assertTrue(np.allclose(refined_probs.sum(axis=-1), 1.0))

        # 3. Check that neighbors of confident positions are influenced
        # The probability of token 5 at position 1 and 4 should increase
        self.assertGreater(refined_probs[0, 1, 5], probs[0, 1, 5])
        self.assertGreater(refined_probs[0, 4, 5], probs[0, 4, 5])

    def test_grover_amplifier(self):
        """
        Tests the GroverAmplifier's ability to sharpen a probability
        distribution, making the most likely token even more probable.
        """
        vocab_size = 20

        amplifier = GroverAmplifier()

        # Create a logit distribution where one token is slightly more likely
        logits = np.random.randn(1, vocab_size).astype(np.float32) * 0.1
        target_token_idx = 7
        logits[0, target_token_idx] = 0.5  # Moderate advantage

        # Get initial probabilities (for comparison)
        initial_probs = np.exp(logits) / np.sum(np.exp(logits))

        # Amplify the logits
        amplified_probs = amplifier.amplify_logits(logits, iterations=1)

        print(f"Initial max prob: {np.max(initial_probs)}")
        print(f"Amplified max prob: {np.max(amplified_probs)}")
        print(f"Initial target prob: {initial_probs[0, target_token_idx]}")
        print(f"Amplified target prob: {amplified_probs[0, target_token_idx]}")

        # 1. Check shape
        self.assertEqual(amplified_probs.shape, (1, vocab_size))

        # 2. Check that the amplified probabilities are still valid
        self.assertTrue(np.allclose(amplified_probs.sum(), 1.0))

        # 3. Check that the target token's probability has increased
        self.assertGreater(
            amplified_probs[0, target_token_idx], initial_probs[0, target_token_idx]
        )

        # 4. Check that the amplified distribution is sharper
        # The new highest probability should be higher than the old one
        self.assertGreater(np.max(amplified_probs), np.max(initial_probs))


if __name__ == "__main__":
    unittest.main()
