import importlib

import config.settings as settings


def test_ownership_and_collaboration_are_disabled_by_default(monkeypatch):
    monkeypatch.delenv("OWNERSHIP_ENABLED", raising=False)
    monkeypatch.delenv("COLLABORATION_ENABLED", raising=False)

    reloaded = importlib.reload(settings)
    assert reloaded.OWNERSHIP_CONFIG["enabled"] is False
    assert reloaded.COLLABORATION_CONFIG["enabled"] is False
