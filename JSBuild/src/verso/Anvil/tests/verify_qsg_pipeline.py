import unittest
import numpy as np
import sys

# Add project root match current user's workspace
sys.path.append("/home/mike/Documents/granite-agent")


class TestQSGPipeline(unittest.TestCase):
    def test_rogue_imports_removed(self):
        """Ensure rogue classes cannot be imported."""
        print("Verifying rogue code removal...")
        try:
            from core.model.inference_engine import GraniteHybridEngine  # noqa: F401

            self.fail("GraniteHybridEngine should not exist!")
        except ImportError:
            print("✓ GraniteHybridEngine correctly removed.")

        try:
            from core.model.architecture import MambaBlock  # noqa: F401

            self.fail("MambaBlock should not exist!")
        except ImportError:
            print("✓ MambaBlock correctly removed.")

    def test_qsg_adapter_instantiation(self):
        """Ensure QSG Adapter can be instantiated with a real model."""
        print("Verifying QSG Adapter instantiation...")
        from core.qsg.ollama_adapter import OllamaQSGAdapter
        from core.qsg.config import QSGConfig

        config = QSGConfig()
        config.speculative_drafts = 2  # Low number for speed

        # Use a small model known to be present from 'ollama list'
        model_name = "granite4:tiny-h"

        try:
            adapter = OllamaQSGAdapter(model_name, config)
            print(f"✓ Adapter instantiated for {model_name}")

            # Simple context embedding check
            emb = adapter.get_embeddings("Hello world")
            self.assertIsInstance(emb, np.ndarray)
            self.assertEqual(len(emb.shape), 2)  # [1, Dim] or similar
            print(f"✓ Embeddings shape: {emb.shape}")

            # Simple generation check (mocking generator step if needed, but let's try real first)
            # This might be slow if it actually loads weights via GGUF, but it's the ultimate test.
            # Given we are in a test env, we just want to ensure no crash.
            # We'll mock the 'generate_draft' if GGUF loading is too heavy or fails permissions.

        except Exception as e:
            self.fail(f"Failed to instantiate adapter: {str(e)}")


if __name__ == "__main__":
    unittest.main()
