from __future__ import annotations

import json

from saguaro.bootstrap import _hot_command_requested
from saguaro.fastpath import FastCommandAPI
from saguaro.services.platform import GraphService, ParseService


def test_fast_command_extract_terms_splits_underscored_identifiers() -> None:
    api = object.__new__(FastCommandAPI)

    terms = FastCommandAPI._extract_terms(api, "native_coordinator queue_limit")

    assert "native" in terms
    assert "coordinator" in terms
    assert "queue" in terms
    assert "limit" in terms


def test_bootstrap_recognizes_hot_command_paths() -> None:
    assert _hot_command_requested(["health"]) is True
    assert _hot_command_requested(["doctor"]) is True
    assert _hot_command_requested(["query", "native coordinator"]) is True
    assert _hot_command_requested(["abi", "verify"]) is True
    assert _hot_command_requested(["ffi", "audit"]) is True
    assert _hot_command_requested(["math", "parse"]) is True
    assert _hot_command_requested(["cpu", "scan"]) is True
    assert _hot_command_requested(["index", "--path", "."]) is False


def test_graph_service_load_graph_uses_mtime_cache(tmp_path, monkeypatch) -> None:
    saguaro_dir = tmp_path / ".saguaro"
    graph_dir = saguaro_dir / "graph"
    graph_dir.mkdir(parents=True)
    graph_path = graph_dir / "graph.json"
    payload = {
        "version": GraphService.GRAPH_SCHEMA_VERSION,
        "repo_path": str(tmp_path),
        "generated_at": 0,
        "generated_fmt": "now",
        "nodes": {},
        "edges": {},
        "files": {},
        "ffi_patterns": {},
        "symbol_index": {},
        "term_index": {},
        "entity_to_node": {},
        "stats": {},
    }
    graph_path.write_text(json.dumps(payload), encoding="utf-8")

    service = GraphService(str(tmp_path), ParseService(str(tmp_path)))
    first = service.load_graph()
    assert first["version"] == GraphService.GRAPH_SCHEMA_VERSION

    calls = {"count": 0}
    original_load = json.load

    def counting_load(handle):
        calls["count"] += 1
        return original_load(handle)

    monkeypatch.setattr(json, "load", counting_load)
    second = service.load_graph()

    assert second == first
    assert calls["count"] == 0


def test_graph_rebuild_indices_adds_split_search_terms(tmp_path) -> None:
    service = GraphService(str(tmp_path), ParseService(str(tmp_path)))
    graph = service._empty_graph()
    graph["nodes"] = {
        "node-1": {
            "id": "node-1",
            "name": "native_coordinator",
            "qualified_name": "saguaro.indexing.native_coordinator",
            "entity_id": "Saguaro/saguaro/indexing/native_coordinator.py:native_coordinator:1",
            "type": "file",
            "file": "Saguaro/saguaro/indexing/native_coordinator.py",
            "line": 1,
            "end_line": 1,
        }
    }

    service._rebuild_indices(graph)

    assert "native" in graph["term_index"]
    assert "coordinator" in graph["term_index"]
    assert "native_coordinator" in graph["term_index"]
