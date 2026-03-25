import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.qsg.config import QSGConfig
from core.qsg.ollama_adapter import OllamaQSGAdapter


def main():
    model_name = "granite4:tiny-h"
    config = QSGConfig(coherence_range=75, temperature=0.7, speculative_drafts=8)

    print(f"--- QSG Coherence Verification: {model_name} ---")
    try:
        adapter = OllamaQSGAdapter(model_name, config)

        prompt = "Explain the concept of Holographic State Evolution in one sentence."
        print(f"\nPrompt: {prompt}")

        print("\nGenerating...")
        result = adapter.generate(prompt)

        print(f"\nResult:\n{result}")

        # Test 2
        prompt2 = "Write a short poem about hyperdimensional computing."
        print(f"\nPrompt: {prompt2}")
        print("\nGenerating...")
        result2 = adapter.generate(prompt2)
        print(f"\nResult:\n{result2}")

    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
