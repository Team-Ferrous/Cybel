import sys

# Add project root match current user's workspace
sys.path.append("/home/mike/Documents/granite-agent")

from core.qsg.ollama_adapter import OllamaQSGAdapter
from core.qsg.config import QSGConfig


def test_generation():
    print("Initializing QSG Adapter...")
    config = QSGConfig()
    # Ensure some randomness but not too much
    config.temperature = 0.7
    config.speculative_drafts = 5

    model_name = "granite4:tiny-h"
    try:
        adapter = OllamaQSGAdapter(model_name, config)
    except Exception as e:
        print(f"Failed to init adapter: {e}")
        return

    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")
    print("Generating...")

    try:
        output = adapter.generate(prompt, options={"num_predict": 20})
        print(f"Output: '{output}'")

        if "Paris" in output:
            print("✓ Coherence Check: Passed (Found 'Paris')")
        else:
            print("? Coherence Check: Warning (Did not find 'Paris')")

    except Exception as e:
        print(f"Generation failed: {e}")


if __name__ == "__main__":
    test_generation()
