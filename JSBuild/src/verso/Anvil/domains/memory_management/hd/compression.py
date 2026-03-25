import numpy as np
import zlib
import re
from typing import List, Dict
from domains.memory_management.hd.bundle import HolographicBundle


class HDCompressor:
    """
    Holographic Context Compressor.
    Encodes text into hyperdimensional vectors.
    """

    def __init__(self, bundle: HolographicBundle = None, dimension: int = 10000):
        self.dim = dimension
        self.bundle_op = bundle if bundle else HolographicBundle(dimension)
        # Cache for frequent tokens to avoid re-generating random vectors
        self.token_cache: Dict[str, np.ndarray] = {}

    def _get_token_vector(self, token: str) -> np.ndarray:
        """
        Get HD vector for a token deterministically via hashing.
        """
        if token in self.token_cache:
            return self.token_cache[token]

        # Deterministic seed from string
        seed = zlib.crc32(token.encode("utf-8"))
        rng = np.random.RandomState(seed)

        # Generator bipolar vector (0/1)
        vec = rng.randint(0, 2, self.dim, dtype=np.int8)

        # Cache simple LRU (simplified)
        if len(self.token_cache) > 10000:
            self.token_cache.clear()
        self.token_cache[token] = vec

        return vec

    def encode_sequence(
        self, tokens: List[str], preserve_order: bool = True
    ) -> np.ndarray:
        """
        Encode a sequence of tokens into a single HD vector.
        Formula: Sum(Permute(Vector(t_i), i))
        Position is encoded via cyclic shift (Permutation).
        """
        vectors = []
        for i, token in enumerate(tokens):
            vec = self._get_token_vector(token)
            if preserve_order:
                # Permute by position index
                permuted = self.bundle_op.permute(vec, shifts=i)
                vectors.append(permuted)
            else:
                vectors.append(vec)

        return self.bundle_op.bundle(vectors)

    def encode_context(self, text: str) -> np.ndarray:
        """Simple tokenizer and encoder with normalization. Uses BoW for retrieval robustness."""
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return self.encode_sequence(tokens, preserve_order=False)

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts via HD encoding."""
        v1 = self.encode_context(text1)
        v2 = self.encode_context(text2)
        return self.bundle_op.similarity(v1, v2)
