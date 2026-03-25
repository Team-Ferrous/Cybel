import sys

sys.path.append("/home/mike/Documents/granite-agent")
from core.model.gguf_loader import get_loader


def debug_vocab():
    model_name = "granite4:tiny-h"
    print(f"Inspecting {model_name}...")
    loader = get_loader(model_name)

    vocab_tokens = loader.get_vocab_tokens()
    embeddings = loader.get_token_embeddings()

    print(f"Vocab Token List Size: {len(vocab_tokens)}")
    print(f"Embedding Tensor Shape: {embeddings.shape}")

    if len(vocab_tokens) != embeddings.shape[0]:
        print("❗ MISMATCH DETECTED")
        diff = len(vocab_tokens) - embeddings.shape[0]
        print(f"Difference: {diff}")
    else:
        print("✓ Sizes match")


if __name__ == "__main__":
    debug_vocab()
