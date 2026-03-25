from __future__ import annotations

import json
from pathlib import Path

from saguaro.api import SaguaroAPI


def _write_repo(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "auth.py").write_text(
        "from pkg.util import check_password\n\n"
        "class AuthManager:\n"
        "    def login_with_password(self, username: str, password: str) -> bool:\n"
        "        return check_password(username, password)\n",
        encoding="utf-8",
    )
    (pkg / "util.py").write_text(
        "def check_password(username: str, password: str) -> bool:\n"
        "    return bool(username and password)\n",
        encoding="utf-8",
    )


def test_api_graph_build_and_query(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))

    index_result = api.index(path=".", force=True)
    assert index_result["graph"]["status"] == "ok"

    graph_query = api.graph_query(symbol="login", depth=1, limit=10)
    assert graph_query["status"] == "ok"
    assert graph_query["count"] >= 1
    assert any(node["file"].endswith("pkg/auth.py") for node in graph_query["nodes"])


def test_api_unwired_smoke(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    (tmp_path / "main.py").write_text(
        "from pkg.auth import AuthManager\n\n"
        "if __name__ == '__main__':\n"
        "    AuthManager()\n",
        encoding="utf-8",
    )
    (tmp_path / "legacy_feature.py").write_text(
        "def stale_flow():\n    return 1\n", encoding="utf-8"
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.graph_build(path=".", incremental=True)
    report = api.unwired(threshold=0.0, include_fragments=True, refresh_graph=False)

    assert {
        "status",
        "threshold",
        "summary",
        "clusters",
        "roots",
        "warnings",
        "graph",
    } <= set(report.keys())
    assert isinstance(report["clusters"], list)


def test_api_query_supports_strategy_and_explain(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.query(
        "authentication login password",
        k=5,
        strategy="lexical",
        explain=True,
    )

    assert result["strategy"] == "lexical"
    assert result["results"]
    top = result["results"][0]
    assert top["file"].endswith("pkg/auth.py")
    assert top["explanation"]["strategy"] == "lexical"
    assert top["explanation"]["matched_terms"]
    assert result["aes_envelope"]["action_type"] == "repo_query"


def test_api_query_supports_task_strategy_aliases(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.query(
        "AuthManager.login_with_password",
        k=5,
        strategy="symbol",
        explain=True,
    )

    assert result["strategy"] == "symbol"
    assert result["execution_strategy"] == "lexical"
    assert result["results"]
    assert result["results"][0]["qualified_name"] == "AuthManager.login_with_password"
    assert result["results"][0]["file"].endswith("pkg/auth.py")
    assert result["results"][0]["explanation"]["execution_strategy"] == "lexical"


def test_api_query_exposes_resolved_query_plan(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.query(
        "authentication login password",
        k=3,
        strategy="hybrid",
        explain=True,
        recall="high",
        breadth=48,
        score_threshold=0.0,
        stale_file_bias=0.5,
        cost_budget="generous",
    )

    assert result["results"]
    assert result["query_plan"]["recall"] == "high"
    assert result["query_plan"]["breadth"] == 48
    assert result["query_plan"]["cost_budget"] == "generous"
    assert result["query_plan"]["ann_search_k"] >= 48
    assert result["results"][0]["explanation"]["query_plan"]["breadth"] == 48


def test_api_query_recovers_from_stale_graph_indices(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    graph_path = tmp_path / ".saguaro" / "graph" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["version"] = 1
    graph.pop("symbol_index", None)
    graph.pop("term_index", None)
    graph.pop("entity_to_node", None)
    for node in graph.get("nodes", {}).values():
        node.pop("entity_id", None)
    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")

    result = api.query(
        "AuthManager.login_with_password",
        k=3,
        strategy="symbol",
        explain=True,
    )

    assert result["results"]
    assert result["results"][0]["qualified_name"] == "AuthManager.login_with_password"


def test_api_query_recovers_from_corrupt_graph_file(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    graph_path = tmp_path / ".saguaro" / "graph" / "graph.json"
    graph_path.write_text('{"nodes": {"broken": ', encoding="utf-8")

    result = api.query(
        "AuthManager.login_with_password",
        k=3,
        strategy="symbol",
        explain=True,
        auto_refresh=False,
    )

    assert result["results"]
    assert result["results"][0]["qualified_name"] == "AuthManager.login_with_password"
    backups = list((tmp_path / ".saguaro" / "graph").glob("graph.json.corrupt-*"))
    assert backups


def test_api_health_includes_graph_and_parser_coverage(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    report = api.health()

    assert "coverage" in report
    assert "graph" in report
    assert report["coverage"]["coverage_percent"] >= 0.0
    assert report["graph"]["status"] == "ready"


def test_api_graph_build_persists_advanced_edge_stats(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))

    build = api.graph_build(path=".", incremental=False)

    assert build["status"] == "ok"

    graph_path = tmp_path / ".saguaro" / "graph" / "graph.json"
    code_graph_path = tmp_path / ".saguaro" / "graph" / "code_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    stats = graph["stats"]

    assert code_graph_path.exists()
    assert stats["call_edges"] >= 1
    assert stats["cfg_edges"] >= 1
    assert stats["dfg_edges"] >= 1
    assert "ffi_patterns" in stats
    assert "bridge_edges" in stats


def test_api_verify_emits_evidence_bundle_and_coverage_gate(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.verify(
        path=".",
        engines="native",
        evidence_bundle=True,
        min_parser_coverage=101.0,
    )

    assert result["status"] == "fail"
    assert any(v["rule_id"] == "SAGUARO-PARSER-COVERAGE" for v in result["violations"])
    assert Path(result["evidence_bundle"]).exists()
    assert result["confidence_posture"]["parser_coverage_percent"] >= 0.0


def test_api_research_ingest_and_dashboard(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)
    ingest = api.research_ingest(
        source="arxiv",
        records=[{"title": "RepoGraph", "url": "https://arxiv.org/abs/2410.14684"}],
    )

    assert ingest["status"] == "ok"
    dashboard = api.app_dashboard()
    assert dashboard["research"]
    assert "health" in dashboard
    assert dashboard["global_code_map"]["hot_files"]
    assert "relation_counts" in dashboard["architecture_explorer"]
    assert "source_counts" in dashboard["research_center"]


def test_api_eval_run_persists_metrics(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.eval_run("cpu_perf", k=3, limit=3)

    assert result["status"] == "ok"
    assert result["suite"] == "cpu_perf"
    assert result["total_cases"] >= 1
    assert result["aes_envelope"]["action_type"] == "benchmark_run"
    runs = api.metrics_list(category="eval")
    assert runs
    assert runs[0]["payload"]["suite"] == "cpu_perf"
    assert runs[0]["aes_envelope"]["action_type"] == "benchmark_run"


def test_dashboard_includes_benchmark_status(tmp_path: Path) -> None:
    _write_repo(tmp_path)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)
    api.eval_run("cpu_perf", k=3, limit=2)

    dashboard = api.app_dashboard()

    assert dashboard["benchmarks"]
    assert dashboard["benchmark_status"] is not None
    assert dashboard["benchmark_status"]["payload"]["suite"] == "cpu_perf"
    assert dashboard["aes_posture"]["trusted"]
