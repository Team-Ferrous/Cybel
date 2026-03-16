from __future__ import annotations

import time

from saguaro.api import SaguaroAPI


def test_query_encoder_latency_smoke(tmp_path) -> None:
    api = SaguaroAPI(str(tmp_path))

    started = time.perf_counter()
    vector = api._encode_text("alpha beta gamma delta", active_dim=512, total_dim=1024)
    elapsed = time.perf_counter() - started

    assert vector.shape[0] == 1024
    # Smoke threshold to catch pathological regressions without flakiness.
    assert elapsed < 8.0
