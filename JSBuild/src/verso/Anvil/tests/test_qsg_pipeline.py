import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import numpy as np

from core.qsg.config import QSGConfig
from core.qsg.ollama_adapter import OllamaQSGAdapter
from core.qsg.generator import QSGGenerator


class TestQSGPipeline(unittest.TestCase):
    def setUp(self):
        self.config = QSGConfig(
            bond_dim=16, grover_iterations=1, jacobi_iterations=1, speculative_drafts=4
        )

    @patch("core.qsg.ollama_adapter.NativeInferenceEngine")
    @patch("core.qsg.ollama_adapter.get_loader")
    @patch("core.qsg.ollama_adapter.AnvilTokenizer")
    @patch("core.qsg.ollama_adapter.ModelProfile.from_loader")
    def test_pipeline_flow(
        self,
        mock_profile_from_loader,
        mock_tokenizer_cls,
        mock_get_loader,
        mock_native_engine_cls,
    ):
        # Mock Loader
        mock_loader = MagicMock()
        mock_loader.get_vocab_tokens.return_value = ["<pad>", "<s>", "</s>"] + [
            f"t{i}" for i in range(100)
        ]
        mock_loader.get_special_tokens.return_value = {"bos": 1, "eos": 2, "pad": 0}
        mock_loader.get_token_embeddings.return_value = np.random.randn(103, 16).astype(
            np.float32
        )
        mock_loader.extract_propagator.return_value = np.eye(16, dtype=np.float32)
        mock_loader.model_path = "/tmp/fake.gguf"
        mock_get_loader.return_value = mock_loader

        # Mock native engine path (unified QSG no longer has legacy fallback)
        mock_engine = mock_native_engine_cls.return_value
        mock_engine.tokenize.return_value = [1, 5, 10]
        mock_engine.generate.return_value = [1, 5, 10, 42, 43]
        mock_engine.detokenize.return_value = "Generated text"
        mock_engine.decode_generated_tokens.return_value = "Generated text"
        mock_engine.context_length = 200000
        mock_profile_from_loader.return_value = SimpleNamespace(
            gqa=False,
            speculative_acceptance_threshold=0.7,
            spec_num_candidates=4,
            spec_max_draft_length=4,
            spec_acceptance_threshold=0.7,
            logits_proxy_top_k=32,
            coconut_alpha=0.3,
            grover_top_k=2,
            grover_damping=0.8,
            chat_template="granite",
            propagator_strategy="mlp",
        )

        # Mock Tokenizer
        mock_tokenizer = mock_tokenizer_cls.return_value
        mock_tokenizer.encode.return_value = [1, 5, 10]  # Dummy tokens
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tokenizer.eos_id = 2

        # Initialize Adapter
        adapter = OllamaQSGAdapter("granite4:tiny-h", self.config)
        adapter.speculative_decoder = None

        # Test Generation
        prompt = "Hello world"
        result = adapter.generate(prompt)

        print(f"QSG Result: {result}")
        self.assertEqual(result, "Generated text")

    def test_generator_draft(self):
        # Test internal generator logic
        vocab_dim = 64
        vocab_size = 100
        vocab_emb = np.random.randn(vocab_size, vocab_dim).astype(np.float32)

        generator = QSGGenerator(self.config, vocab_emb)

        context = np.random.randn(1, vocab_dim).astype(np.float32)
        tokens, probs = generator.generate_draft(
            context,
            seq_len=4,
            oracle_context={
                "latent_prior": context[0].tolist(),
                "context_text": "code",
            },
        )

        self.assertEqual(tokens.shape, (1, 4))
        self.assertEqual(probs.shape, (1, 4, vocab_size))
        self.assertGreater(
            generator.last_generation_trace["jacobi_frontier"]["frontier_width"], 0
        )


if __name__ == "__main__":
    unittest.main()
