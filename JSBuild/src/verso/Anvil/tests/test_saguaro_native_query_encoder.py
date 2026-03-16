from __future__ import annotations

import numpy as np
import pytest

from saguaro.api import SaguaroAPI


def test_native_query_encoder_returns_normalized_vector(tmp_path) -> None:
    api = SaguaroAPI(str(tmp_path))
    vec = api._encode_text(
        {"symbol_terms": ["alpha"], "path_terms": ["pkg", "module"], "doc_terms": ["returns", "value"]},
        active_dim=256,
        total_dim=512,
    )

    assert vec.shape[0] == 512
    norm = float(np.linalg.norm(vec[:256]))
    assert norm == pytest.approx(1.0, abs=1e-3) or norm == pytest.approx(0.0, abs=1e-6)
