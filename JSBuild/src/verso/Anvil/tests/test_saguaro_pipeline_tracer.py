from __future__ import annotations

from pathlib import Path

import saguaro.api as api_module
from saguaro.agents.perception import TracePerception
from saguaro.api import SaguaroAPI
from saguaro.indexing.backends import NumPyBackend


def _write_pipeline_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "from pkg.pipeline import run\n\n" "def main():\n" "    return run()\n",
        encoding="utf-8",
    )
    (pkg / "pipeline.py").write_text(
        "from pkg.stage import stage\n\n" "def run():\n" "    return stage()\n",
        encoding="utf-8",
    )
    (pkg / "stage.py").write_text(
        "def stage():\n" "    return 'ok'\n",
        encoding="utf-8",
    )


def test_api_trace_pipeline_uses_graph_and_returns_stages(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        api_module,
        "get_backend",
        lambda prefer_tensorflow=True: NumPyBackend(),
    )
    _write_pipeline_repo(tmp_path)

    api = SaguaroAPI(repo_path=str(tmp_path))
    build = api.graph_build(path=".", incremental=False)
    assert build["status"] == "ok"

    traced = api.trace(entry_point="main.py", depth=4)

    assert traced["status"] == "ok"
    assert traced["stage_count"] >= 2
    files = {stage.get("file") for stage in traced["stages"]}
    assert "main.py" in files
    assert any(
        file_name and file_name.endswith("pkg/pipeline.py") for file_name in files
    )


def test_seed_resolution_downranks_test_noise(tmp_path: Path) -> None:
    perception = TracePerception(repo_path=str(tmp_path))
    nodes = {
        "prod": {
            "id": "prod",
            "name": "forward",
            "qualified_name": "forward",
            "type": "function",
            "file": "pkg/model.py",
        },
        "test": {
            "id": "test",
            "name": "forward",
            "qualified_name": "forward",
            "type": "function",
            "file": "tests/test_model.py",
        },
        "dfg": {
            "id": "dfg",
            "name": "forward",
            "qualified_name": "forward",
            "type": "dfg_node",
            "file": "pkg/model.py",
        },
    }
    seeds = perception._resolve_seed_nodes(  # noqa: SLF001
        nodes,
        graph={},
        entry_point=None,
        query="inference pipeline",
    )
    assert seeds
    assert seeds[0] == "prod"
    assert "dfg" not in seeds[:1]
