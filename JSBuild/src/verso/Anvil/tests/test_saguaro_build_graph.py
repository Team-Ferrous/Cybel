from __future__ import annotations

from pathlib import Path

from saguaro.build_system.ingestor import BuildGraphIngestor


def test_build_graph_skips_null_package_manifest(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text("null\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'demo'\ndependencies = ['requests']\n",
        encoding="utf-8",
    )

    report = BuildGraphIngestor(str(tmp_path)).ingest()

    assert report["target_count"] == 1
    assert "py:demo" in report["targets"]


def test_build_graph_respects_shared_manifest_exclusions(tmp_path: Path) -> None:
    root_package = tmp_path / "package.json"
    root_package.write_text('{"name":"root-app","dependencies":{"leftpad":"1.0.0"}}', encoding="utf-8")

    excluded_dir = tmp_path / "repo_analysis"
    excluded_dir.mkdir()
    (excluded_dir / "package.json").write_text(
        '{"name":"should-not-appear","dependencies":{"bad":"1.0.0"}}',
        encoding="utf-8",
    )

    report = BuildGraphIngestor(str(tmp_path)).ingest()
    names = set(report["targets"].keys())

    assert "npm:root-app" in names
    assert "npm:should-not-appear" not in names
