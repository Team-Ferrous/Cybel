from __future__ import annotations

from pathlib import Path

from saguaro.indexing import auto_scaler


def test_repo_stats_cache_is_reused(tmp_path, monkeypatch) -> None:
    source = tmp_path / "demo.py"
    source.write_text("def demo():\n    return 1\n", encoding="utf-8")

    first = auto_scaler.get_repo_stats_and_config(str(tmp_path), ttl_seconds=3600)
    assert first["loc"] > 0

    def _boom(path: str):
        raise AssertionError("count_loc should not be called when cache is fresh")

    monkeypatch.setattr(auto_scaler, "count_loc", _boom)
    second = auto_scaler.get_repo_stats_and_config(str(tmp_path), ttl_seconds=3600)
    assert second == first


def test_repo_stats_force_refresh_bypasses_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "demo.py"
    source.write_text("def demo():\n    return 1\n", encoding="utf-8")
    auto_scaler.get_repo_stats_and_config(str(tmp_path), ttl_seconds=3600)

    monkeypatch.setattr(
        auto_scaler,
        "_scan_repo_stats",
        lambda path: (1, 123, {"Python": 123}),
    )
    refreshed = auto_scaler.get_repo_stats_and_config(
        str(tmp_path),
        force_refresh=True,
        ttl_seconds=3600,
    )
    assert refreshed["loc"] == 123
    assert refreshed["languages"] == {"Python": 123}


def test_repo_stats_caps_dimensions_for_large_corpus(tmp_path, monkeypatch) -> None:
    for index in range(1600):
        (tmp_path / f"module_{index:04d}.py").write_text(
            "def demo():\n    return 1\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        auto_scaler,
        "count_loc",
        lambda path: (600_000, {"Python": 600_000}),
    )

    stats = auto_scaler.get_repo_stats_and_config(
        str(tmp_path),
        force_refresh=True,
        ttl_seconds=0,
    )

    assert stats["candidate_files"] == 1600
    assert stats["active_dim"] == 2048
    assert stats["total_dim"] == 4096


def test_repo_stats_never_emit_active_dim_above_total_dim_for_huge_loc(
    tmp_path, monkeypatch
) -> None:
    for index in range(2000):
        (tmp_path / f"module_{index:04d}.py").write_text(
            "def demo():\n    return 1\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        auto_scaler,
        "count_loc",
        lambda path: (4_000_000, {"Python": 4_000_000}),
    )

    stats = auto_scaler.get_repo_stats_and_config(
        str(tmp_path),
        force_refresh=True,
        ttl_seconds=0,
    )

    assert stats["active_dim"] <= stats["total_dim"]
    assert stats["dark_space_ratio"] >= 0.0
