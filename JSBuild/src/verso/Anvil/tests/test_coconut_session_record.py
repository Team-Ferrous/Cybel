from __future__ import annotations

import numpy as np

from core.reasoning.coconut import ContinuousThoughtBlock


def test_coconut_exposes_additive_session_record() -> None:
    coconut = ContinuousThoughtBlock(
        embedding_dim=8,
        num_paths=2,
        steps=1,
        use_gpu=False,
    )

    refined = coconut.explore(np.ones((1, 8), dtype=np.float32))

    assert refined.shape == (1, 8)
    assert coconut.session_record is not None
    assert coconut.session_record["sequence_id"].startswith("coconut-")
    assert coconut.session_record["packet"]["hidden_dimension"] == 8
    assert coconut.session_record["packet"]["checksum"]
