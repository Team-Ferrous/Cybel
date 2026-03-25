from __future__ import annotations

import subprocess
from pathlib import Path

from saguaro.query.corpus_rules import corpus_manifest


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=path,
        check=True,
        capture_output=True,
    )


def test_corpus_manifest_respects_gitignore_and_default_policy(tmp_path: Path) -> None:
    _git_init(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored/\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "live.py").write_text("def live():\n    return 1\n", encoding="utf-8")
    (tmp_path / "ignored").mkdir()
    (tmp_path / "ignored" / "skip.py").write_text("def skip():\n    return 0\n", encoding="utf-8")
    (tmp_path / "repo_analysis").mkdir()
    (tmp_path / "repo_analysis" / "analysis.py").write_text("def noise():\n    return 0\n", encoding="utf-8")

    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)

    manifest = corpus_manifest(str(tmp_path))
    rel_files = {str(Path(path).relative_to(tmp_path)).replace("\\", "/") for path in manifest.files}

    assert "src/live.py" in rel_files
    assert "ignored/skip.py" not in rel_files
    assert "repo_analysis/analysis.py" not in rel_files
    assert manifest.candidate_count >= len(rel_files)
    assert manifest.excluded_count >= 1
