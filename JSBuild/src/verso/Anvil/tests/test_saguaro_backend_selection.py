from __future__ import annotations

import pytest

import saguaro.indexing.backends as backends


def test_get_backend_requires_native_backend(monkeypatch) -> None:
    monkeypatch.setattr(backends, "_HAS_NATIVE", False)

    with pytest.raises(RuntimeError, match="requires the native backend"):
        backends.get_backend(prefer_tensorflow=True)
