from __future__ import annotations

from pathlib import Path

from saguaro.api import SaguaroAPI


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_unwired_reports_isolated_multi_file_cluster(tmp_path: Path) -> None:
    _write(
        tmp_path / "main.py",
        "from app.core import run\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    )
    _write(tmp_path / "app" / "core.py", "def run():\n    return 'ok'\n")

    _write(tmp_path / "isolated" / "feature_a.py", "def alpha():\n    return 1\n")
    _write(
        tmp_path / "isolated" / "feature_b.py",
        "from isolated.feature_a import alpha\n"
        "\n"
        "def beta():\n"
        "    return alpha()\n",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    report = api.unwired(threshold=0.0, include_fragments=True, refresh_graph=False)

    assert report["status"] == "ok"
    cluster = next(
        (
            item
            for item in report["clusters"]
            if any(path.startswith("isolated/") for path in item.get("top_files", []))
        ),
        None,
    )
    assert cluster is not None
    assert cluster["classification"] == "unwired_feature"


def test_unwired_does_not_report_connected_graph_by_default(tmp_path: Path) -> None:
    _write(
        tmp_path / "main.py",
        "import worker\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    worker.run()\n",
    )
    _write(tmp_path / "worker.py", "def run():\n    return 1\n")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    report = api.unwired(threshold=0.0, refresh_graph=False)

    assert report["status"] == "ok"
    assert report["summary"]["cluster_count"] == 0
    assert report["clusters"] == []


def test_unwired_resolves_relative_imports_from_package_roots(tmp_path: Path) -> None:
    _write(
        tmp_path / "main.py",
        "from app import run\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    )
    _write(
        tmp_path / "app" / "__init__.py",
        "from .runner import run\n",
    )
    _write(
        tmp_path / "app" / "runner.py",
        "from .pipeline import build\n"
        "\n"
        "def run():\n"
        "    return build()\n",
    )
    _write(
        tmp_path / "app" / "pipeline.py",
        "def build():\n"
        "    return 1\n",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    report = api.unwired(threshold=0.0, include_fragments=True, refresh_graph=False)

    assert all(
        not any(path.startswith("app/") for path in item.get("top_files", []))
        for item in report["clusters"]
    )


def test_unwired_excludes_tests_by_default_and_includes_when_enabled(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "main.py",
        "from app.core import run\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    )
    _write(tmp_path / "app" / "core.py", "def run():\n    return 'ok'\n")

    _write(tmp_path / "tests" / "__init__.py", "")
    _write(
        tmp_path / "tests" / "island_a.py",
        "import tests.island_b\n"
        "\n"
        "def a():\n"
        "    return tests.island_b.b()\n",
    )
    _write(
        tmp_path / "tests" / "island_b.py",
        "def b():\n"
        "    return 2\n",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    without_tests = api.unwired(threshold=0.0, refresh_graph=False)
    with_tests = api.unwired(
        threshold=0.0,
        include_tests=True,
        include_fragments=True,
        refresh_graph=False,
    )

    assert all(
        not any(path.startswith("tests/") for path in item.get("top_files", []))
        for item in without_tests["clusters"]
    )
    assert any(
        any(path.startswith("tests/") for path in item.get("top_files", []))
        for item in with_tests["clusters"]
    )


def test_unwired_emits_duplicate_tree_warning_and_filters_legacy_nodes(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "main.py",
        "from saguaro.live import run\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    )
    _write(tmp_path / "saguaro" / "live.py", "def run():\n    return 3\n")

    _write(tmp_path / "Saguaro" / "legacy.py", "def legacy():\n    return 4\n")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    report = api.unwired(threshold=0.0, include_fragments=True, refresh_graph=False)

    assert any(
        "Both authoritative 'Saguaro/' and transitional top-level 'saguaro/' trees exist"
        in w
        for w in report["warnings"]
    )
    assert all(
        not any(path.startswith("saguaro/") for path in item.get("top_files", []))
        for item in report["clusters"]
    )


def test_unwired_excludes_mcp_tree_by_default(tmp_path: Path) -> None:
    _write(
        tmp_path / "main.py",
        "def run():\n"
        "    return 1\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    run()\n",
    )
    _write(
        tmp_path / "saguaro" / "mcp" / "island.py",
        "def orphaned():\n"
        "    return 42\n",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)

    report = api.unwired(threshold=0.0, include_fragments=True, refresh_graph=False)

    assert all(
        not any(path.startswith("saguaro/mcp/") for path in item.get("top_files", []))
        for item in report["clusters"]
    )
