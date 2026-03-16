from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import saguaro.api as api_module
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend


def _write_chain_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "a.py").write_text(
        "from pkg.b import b\n\n"
        "def a():\n"
        "    return b()\n",
        encoding="utf-8",
    )
    (pkg / "b.py").write_text(
        "from pkg.c import c\n\n"
        "def b():\n"
        "    return c()\n",
        encoding="utf-8",
    )
    (pkg / "c.py").write_text("def c():\n    return 1\n", encoding="utf-8")


def _write_symbol_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "hub.py").write_text("VALUE = 1\n", encoding="utf-8")
    a_module = (
        "from pkg.hub import VALUE\n\n"
        "class A:\n"
        "    def f(self):\n"
        "        return VALUE\n"
    )
    (pkg / "a.py").write_text(
        a_module,
        encoding="utf-8",
    )
    c_module = (
        "from pkg.hub import VALUE\n\n"
        "class C:\n"
        "    def h(self):\n"
        "        return VALUE\n"
    )
    (pkg / "c.py").write_text(
        c_module,
        encoding="utf-8",
    )


def test_graph_query_supports_path_and_reachability(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_chain_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    built = api.graph_build(path=".", incremental=False)
    assert built["status"] == "ok"

    path_result = api.graph_query(
        source="pkg/a.py",
        target="pkg/c.py",
        query_path=True,
        depth=4,
        max_depth=4,
    )
    assert path_result["path_query"]["status"] == "ok"
    assert path_result["path_query"]["found"] is True
    assert path_result["path_query"]["length"] >= 1

    reachable = api.graph_query(reachable_from="pkg/a.py", depth=4, max_depth=2)
    assert reachable["reachable"]["count"] >= 1
    reachable_files = {node.get("file") for node in reachable.get("nodes", [])}
    assert any(path and path.endswith("pkg/b.py") for path in reachable_files)

    symbol_path = api.graph_query(
        source="a",
        target="c",
        query_path=True,
        depth=8,
        max_depth=8,
    )
    assert symbol_path["path_query"]["status"] == "ok"
    assert symbol_path["path_query"]["found"] is True
    assert symbol_path["path_query"]["length"] >= 1


def test_graph_selector_prefers_non_test_symbols(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    nodes = {
        "prod": {
            "id": "prod",
            "name": "run",
            "qualified_name": "run",
            "type": "function",
            "file": "pkg/run.py",
        },
        "test": {
            "id": "test",
            "name": "run",
            "qualified_name": "run",
            "type": "function",
            "file": "tests/test_run.py",
        },
        "dfg": {
            "id": "dfg",
            "name": "run",
            "qualified_name": "run",
            "type": "dfg_node",
            "file": "pkg/run.py",
        },
    }
    selected = api._resolve_graph_selector_ids(  # noqa: SLF001
        selector="run",
        nodes=nodes,
        files={},
    )
    assert selected
    assert selected[0] == "prod"
    assert "dfg" not in selected[:1]


def test_graph_query_symbol_path_uses_file_anchors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_symbol_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    built = api.graph_build(path=".", incremental=False)
    assert built["status"] == "ok"

    path_result = api.graph_query(
        expression="path(A.f, C.h)",
        depth=4,
        max_depth=4,
    )
    assert path_result["query_path"]["status"] == "ok"
    assert path_result["query_path"]["found"] is True
    path_files = {
        str(node.get("file") or "") for node in path_result["query_path"]["path_nodes"]
    }
    assert any(path.endswith("pkg/a.py") for path in path_files)
    assert any(path.endswith("pkg/c.py") for path in path_files)


def test_graph_query_reports_no_match_for_unknown_symbol(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_symbol_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    built = api.graph_build(path=".", incremental=False)
    assert built["status"] == "ok"

    path_result = api.graph_query(
        expression="path(DoesNotExist.call, C.h)",
        depth=4,
        max_depth=4,
    )
    assert path_result["query_path"]["status"] == "no_match"
    assert path_result["query_path"]["found"] is False


def test_cli_help_exposes_graph_ffi_and_path_flags() -> None:
    query_help = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "saguaro.cli", "graph", "query", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert query_help.returncode == 0
    assert "--query-path" in query_help.stdout
    assert "--reachable-from" in query_help.stdout

    ffi_help = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "saguaro.cli", "graph", "ffi", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ffi_help.returncode == 0
    assert "--limit" in ffi_help.stdout
