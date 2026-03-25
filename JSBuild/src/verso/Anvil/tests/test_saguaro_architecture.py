from __future__ import annotations

from pathlib import Path

import saguaro.api as api_module
import saguaro.architecture.topology as topology_module
from saguaro.api import SaguaroAPI
from saguaro.architecture import ArchitectureAnalyzer
from saguaro.indexing.backends import NumPyBackend


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_architecture_analyzer_detects_misplaced_and_illegal_zone_dependency(
    tmp_path: Path,
) -> None:
    layout = Path("standards/REPO_LAYOUT.yaml").read_text(encoding="utf-8")
    _write(
        tmp_path / "saguaro" / "analysis" / "scan.py",
        "import tests.helpers\n",
    )
    _write(tmp_path / "tests" / "helpers.py", "def helper():\n    return 1\n")
    _write(tmp_path / "rogue.py", "def rogue():\n    return 0\n")
    _write(tmp_path / "standards" / "REPO_LAYOUT.yaml", layout)

    report = ArchitectureAnalyzer(str(tmp_path)).verify(".")
    rule_ids = {row["rule_id"] for row in report["findings"]}

    assert report["status"] == "fail"
    assert "AES-LAYOUT-001" in rule_ids
    assert "AES-ARCH-101" in rule_ids


def test_api_health_exposes_topology_and_coverage_vectors(
    monkeypatch, tmp_path: Path
) -> None:
    layout = Path("standards/REPO_LAYOUT.yaml").read_text(encoding="utf-8")
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write(tmp_path / "saguaro" / "__init__.py", "")
    _write(tmp_path / "saguaro" / "live.py", "def run():\n    return 1\n")
    _write(tmp_path / "standards" / "REPO_LAYOUT.yaml", layout)

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    health = api.health()

    assert "topology" in health
    assert "coverage_vector" in health
    assert "degraded_mode" in health
    assert "mapped_files_pct" in health["topology"]


def test_architecture_include_resolution_uses_cached_repo_index(
    monkeypatch, tmp_path: Path
) -> None:
    layout = Path("standards/REPO_LAYOUT.yaml").read_text(encoding="utf-8")
    _write(tmp_path / "saguaro" / "__init__.py", "")
    _write(
        tmp_path / "saguaro" / "native" / "public" / "api.h",
        '#include "internal/detail.h"\n',
    )
    _write(
        tmp_path / "saguaro" / "native" / "internal" / "detail.h",
        "int detail();\n",
    )
    _write(tmp_path / "standards" / "REPO_LAYOUT.yaml", layout)

    analyzer = ArchitectureAnalyzer(str(tmp_path))
    analyzer._prime_resolution_indexes(
        [
            "saguaro/native/public/api.h",
            "saguaro/native/internal/detail.h",
        ]
    )

    def _unexpected_walk(*args, **kwargs):
        raise AssertionError("os.walk should not be used during cached include resolution")

    monkeypatch.setattr(topology_module.os, "walk", _unexpected_walk)

    resolved = analyzer._resolve_include(
        current_dir="saguaro/native/public",
        include_target="internal/detail.h",
    )

    assert resolved == "saguaro/native/internal/detail.h"
