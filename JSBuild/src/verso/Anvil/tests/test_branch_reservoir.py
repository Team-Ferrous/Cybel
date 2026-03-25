from __future__ import annotations

import numpy as np

from core.memory.latent_memory import LatentMemory
from core.reasoning.coconut import ContinuousThoughtBlock


def test_branch_reservoir_records_coconut_session_and_persists_vector() -> None:
    coconut = ContinuousThoughtBlock(
        embedding_dim=8,
        num_paths=3,
        steps=1,
        use_gpu=False,
    )
    refined = coconut.explore(np.ones((1, 8), dtype=np.float32))
    reservoir = LatentMemory(max_size=4)
    reservoir.add_state(
        "coconut_branch",
        "stored branch session",
        vector=refined.reshape(-1).tolist(),
    )

    assert coconut.session_record is not None
    assert coconut.session_record["packet"]["hidden_dimension"] == 8
    assert reservoir.get_merged_vector(limit=1) is not None
