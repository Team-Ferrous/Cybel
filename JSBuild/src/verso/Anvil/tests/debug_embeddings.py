import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.qsg.config import QSGConfig
from core.qsg.ollama_adapter import OllamaQSGAdapter


def main():
    model_name = "granite4:tiny-h"
    config = QSGConfig()
    adapter = OllamaQSGAdapter(model_name, config)

    text = "Holographic"
    token_ids = adapter.tokenizer.encode(text, add_bos=False)
    print(f"Token IDs for '{text}': {token_ids}")

    # Original embedding
    orig_emb = adapter.vocab_embeddings[token_ids[0]]

    # Encoded state (Single token)
    # This will be Token * Pos0
    context_vec = adapter.get_embeddings(text)

    # Similarity to original token
    sim = np.dot(context_vec[0], orig_emb) / (
        np.linalg.norm(context_vec[0]) * np.linalg.norm(orig_emb)
    )
    print(f"Similarity of Encoded('{text}') to Original('{text}'): {sim:.4f}")

    # Similarity to other tokens (Top 5)
    scores = context_vec[0] @ adapter.vocab_embeddings.T
    top_indices = np.argsort(scores)[-5:][::-1]
    print("Top 5 matches in vocab for encoded vector:")
    for idx in top_indices:
        token = adapter.tokenizer.decode([idx])
        print(f"  ID {idx} ({token}): {scores[idx]:.4f}")


if __name__ == "__main__":
    main()
