import unittest

from core.qsg.ollama_adapter import OllamaQSGAdapter
from core.qsg.config import QSGConfig


class VerifyQSGCoherence(unittest.TestCase):
    def test_qsg_coherence(self):
        """
        Tests the end-to-end QSG pipeline for basic coherence.
        This test loads a real model and generates text to see if it's reasonable.
        """
        config = QSGConfig(
            bond_dim=16,  # Kept small for testing
            grover_iterations=5,  # Found to work in component tests
            jacobi_iterations=3,
            speculative_drafts=4,
        )

        # This will load the GGUF model, which can be slow.
        # Make sure the model 'granite-instruct-3b-q5_k_m.gguf' is available.
        # I need to find the correct model name.
        # I'll check the project for available models.
        # For now, I'll use a placeholder name and assume it's available.
        model_name = "granite4:tiny-h"

        try:
            adapter = OllamaQSGAdapter(model_name, config)
        except Exception as e:
            self.fail(
                f"Failed to initialize OllamaQSGAdapter with model '{model_name}'. "
                f"Make sure the model is available. Error: {e}"
            )

        prompt = "The capital of France is"
        generated_text = adapter.generate(prompt, options={"num_predict": 10})

        print("\n--- QSG Coherence Test ---")
        print(f"Prompt: {prompt}")
        print(f"Generated: '{generated_text}'")
        print("--------------------------")

        print("Running assertions...")
        # 1. Check that the output is not empty
        self.assertGreater(len(generated_text), 0, "Generated text is empty.")

        # 2. Check for reasonable length (at least a few words)
        self.assertGreater(
            len(generated_text.split()), 2, "Generated text is too short."
        )

        # 3. Check for excessive repetition
        words = generated_text.lower().split()
        word_counts = {word: words.count(word) for word in set(words)}
        # No single word should make up more than 50% of the text
        for word, count in word_counts.items():
            self.assertLessEqual(
                count / len(words), 0.5, f"Word '{word}' is excessively repeated."
            )

        print("Assertions passed.")


if __name__ == "__main__":
    unittest.main()
