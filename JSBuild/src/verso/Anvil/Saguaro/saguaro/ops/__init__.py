try:
    from .quantum_ops import (
        fused_coconut_bfs,
        fused_qwt_tokenizer,
        fused_text_tokenize,
        holographic_bundle,
        quantum_embedding,
        time_crystal_step,
    )

    __all__ = [
        "quantum_embedding",
        "fused_qwt_tokenizer",
        "time_crystal_step",
        "fused_coconut_bfs",
        "holographic_bundle",
        "fused_text_tokenize",
    ]
except ImportError:
    # If tensorflow or other deps are missing, just export empty
    # This allows native_indexer to be used in isolation
    __all__ = []
