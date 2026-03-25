import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from core.qsg.config import QSGConfig
from core.qsg.ollama_adapter import OllamaQSGAdapter
from core.qsg.speculative import SpeculativeQSG


class TestSpeculativeQSG(unittest.TestCase):
    def setUp(self):
        self.config = QSGConfig(speculative_drafts=2)

    @patch("core.qsg.ollama_adapter.NativeInferenceEngine")
    @patch("core.qsg.ollama_adapter.get_loader")
    @patch("core.qsg.ollama_adapter.AnvilTokenizer")
    @patch("core.qsg.ollama_adapter.ModelProfile.from_loader")
    def test_speculative_flow(
        self,
        mock_profile_from_loader,
        mock_tokenizer_cls,
        mock_get_loader,
        mock_native_engine_cls,
    ):
        # Mock Loader
        mock_loader = mock_get_loader.return_value
        mock_loader.get_token_embeddings.return_value = np.random.randn(
            100, 4096
        ).astype(np.float32)
        mock_loader.get_lm_head_weight.return_value = np.random.randn(100, 4096).astype(
            np.float32
        )
        mock_loader.get_vocab_tokens.return_value = [f"token_{i}" for i in range(100)]
        mock_loader.get_special_tokens.return_value = {"eos_token_id": 0}
        mock_loader.extract_propagator.return_value = np.random.randn(
            4096, 4096
        ).astype(np.float32)
        mock_loader.get_metadata.return_value = {"general.architecture": "granite"}
        mock_loader.get_context_length.return_value = 2048
        mock_loader.model_path = "mock_path"
        mock_profile_from_loader.return_value = SimpleNamespace(
            gqa=False,
            grover_iterations=2,
            coconut_paths=8,
            speculative_acceptance_threshold=0.7,
            speculative_enabled=True,
            spec_num_candidates=4,
            spec_max_draft_length=4,
            spec_acceptance_threshold=0.7,
            logits_proxy_top_k=32,
            coconut_alpha=0.3,
            grover_top_k=2,
            grover_damping=0.8,
        )
        mock_engine = mock_native_engine_cls.return_value
        mock_engine.supports_token_api = True
        mock_tokenizer = mock_tokenizer_cls.return_value
        mock_tokenizer.batch_encode.return_value = (
            np.array([[1, 2, 3]], dtype=np.int32),
            np.array([3], dtype=np.int32),
        )

        # Init Pipeline
        adapter = OllamaQSGAdapter("granite4:tiny-h", self.config)
        pipeline = SpeculativeQSG(adapter)

        # Generate with mocked backend
        result = pipeline.generate("Start Prompt", max_tokens=10)

        print(f"Speculative Result: {result}")

        # Verify it appended content
        self.assertTrue(len(result) > len("Start Prompt"))
        self.assertIn("Block", result)

    def test_drafter_diversity(self):
        # Test that drafter produces K drafts
        adapter = OllamaQSGAdapter("granite4:tiny-h", self.config)
        pipeline = SpeculativeQSG(adapter)

        context = np.random.randn(1, adapter.vocab_embeddings.shape[1]).astype(
            np.float32
        )
        drafts = pipeline.drafter.generate_drafts(context, num_drafts=4, draft_length=5)

        self.assertEqual(len(drafts), 4)
        self.assertEqual(drafts[0].shape, (5,))


if __name__ == "__main__":
    unittest.main()
