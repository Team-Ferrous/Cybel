import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from domains.code_intelligence.saguaro_substrate import SaguaroSubstrate
import saguaro.api as api_module
import saguaro.cli as cli_module
from saguaro.api import SaguaroAPI
from saguaro.coverage import CoverageReporter
from saguaro.indexing import backends
from saguaro.indexing import native_coordinator as native_coordinator_module
from saguaro.indexing.coordinator import IndexCoordinator
from saguaro.indexing.native_coordinator import _process_pool_context, _worker_count
from saguaro.indexing.native_runtime import NativeIndexRuntime, NativeRuntimeError
from saguaro.parsing.parser import SAGUAROParser
from saguaro.sentinel.engines.base import BaseEngine
from saguaro.utils.file_utils import get_code_files
from tools.registry import ToolRegistry
from tools.saguaro_tools import SaguaroTools
from tools.schema import TOOL_SCHEMAS


@pytest.fixture
def substrate(tmp_path):
    d = tmp_path / "src"
    d.mkdir()
    f = d / "dummy.py"
    f.write_text(
        "class MyClass:\n"
        "    def my_method(self, x):\n"
        "        return x\n\n"
        "def my_func():\n"
        "    pass\n"
    )
    return SaguaroSubstrate(root_dir=str(tmp_path))


@pytest.fixture
def api(tmp_path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "__init__.py").write_text("__all__ = ['demo']\n")
    (d / "demo.py").write_text("def hello():\n    return 'world'\n")
    return SaguaroAPI(repo_path=str(tmp_path))


def test_saguaro_skeleton(substrate):
    res = substrate.execute_command("skeleton src/dummy.py")
    assert "class MyClass" in res
    assert "my_func" in res


def test_saguaro_tools_verify_defaults_include_aes():
    class _FakeSubstrate:
        def verify(self, *, path, engines, fix):
            return json.dumps({"path": path, "engines": engines, "fix": fix})

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(tools.verify())

    assert payload["engines"] == "native,ruff,semantic,aes"


def test_tool_schema_verify_exposes_engines_parameter():
    verify_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "verify"
    )

    assert "engines" in verify_schema["parameters"]["properties"]
    assert "preflight_only" in verify_schema["parameters"]["properties"]
    assert "timeout_seconds" in verify_schema["parameters"]["properties"]


def test_tool_schema_query_exposes_query_plan_parameters():
    query_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "saguaro_query"
    )

    props = query_schema["parameters"]["properties"]
    assert "recall" in props
    assert "breadth" in props
    assert "score_threshold" in props
    assert "stale_file_bias" in props
    assert "cost_budget" in props


def test_saguaro_tools_query_forwards_query_plan_controls():
    class _FakeSubstrate:
        def agent_query(
            self,
            query,
            *,
            k,
            scope,
            dedupe_by,
            recall,
            breadth,
            score_threshold,
            stale_file_bias,
            cost_budget,
        ):
            return json.dumps(
                {
                    "query": query,
                    "k": k,
                    "scope": scope,
                    "dedupe_by": dedupe_by,
                    "recall": recall,
                    "breadth": breadth,
                    "score_threshold": score_threshold,
                    "stale_file_bias": stale_file_bias,
                    "cost_budget": cost_budget,
                }
            )

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(
        tools.query(
            "runtime engine telemetry",
            k=7,
            scope="workspace",
            dedupe_by="path",
            recall="high",
            breadth=48,
            score_threshold=0.25,
            stale_file_bias=0.5,
            cost_budget="generous",
        )
    )

    assert payload["k"] == 7
    assert payload["scope"] == "workspace"
    assert payload["dedupe_by"] == "path"
    assert payload["recall"] == "high"
    assert payload["breadth"] == 48
    assert payload["score_threshold"] == 0.25
    assert payload["stale_file_bias"] == 0.5
    assert payload["cost_budget"] == "generous"


def test_tool_schema_exposes_cpu_scan():
    cpu_scan_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "cpu_scan"
    )

    assert "arch" in cpu_scan_schema["parameters"]["properties"]
    assert "limit" in cpu_scan_schema["parameters"]["properties"]


def test_saguaro_tools_deadcode_uses_json_default():
    class _FakeSubstrate:
        def deadcode(
            self,
            *,
            threshold,
            low_usage_max_refs,
            lang,
            evidence,
            runtime_observed,
            explain,
            output_format,
        ):
            return json.dumps(
                {
                    "threshold": threshold,
                    "low_usage_max_refs": low_usage_max_refs,
                    "lang": lang,
                    "evidence": evidence,
                    "runtime_observed": runtime_observed,
                    "explain": explain,
                    "output_format": output_format,
                    "count": 0,
                    "candidates": [],
                }
            )

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(tools.deadcode())
    assert payload["output_format"] == "json"
    assert payload["low_usage_max_refs"] == 1


def test_tool_schema_includes_deadcode():
    deadcode_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "deadcode"
    )
    props = deadcode_schema["parameters"]["properties"]

    assert "threshold" in props
    assert "low_usage_max_refs" in props
    assert "lang" in props
    assert "evidence" in props
    assert "runtime_observed" in props
    assert "explain" in props
    assert "output_format" in props


def test_saguaro_tools_low_usage_uses_json_default():
    class _FakeSubstrate:
        def low_usage(self, *, max_refs, include_tests, path, limit, output_format):
            return json.dumps(
                {
                    "max_refs": max_refs,
                    "include_tests": include_tests,
                    "path": path,
                    "limit": limit,
                    "output_format": output_format,
                    "count": 0,
                    "candidates": [],
                }
            )

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(tools.low_usage())
    assert payload["max_refs"] == 1
    assert payload["include_tests"] is False
    assert payload["path"] is None
    assert payload["limit"] is None
    assert payload["output_format"] == "json"


def test_tool_schema_includes_low_usage():
    low_usage_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "low_usage"
    )
    props = low_usage_schema["parameters"]["properties"]

    assert "max_refs" in props
    assert "include_tests" in props
    assert "path" in props
    assert "limit" in props
    assert "output_format" in props


def test_tool_registry_exposes_low_usage(tmp_path):
    registry = ToolRegistry(root_dir=str(tmp_path))
    assert "low_usage" in registry.tools


def test_tool_registry_exposes_cpu_scan(tmp_path):
    registry = ToolRegistry(root_dir=str(tmp_path))
    assert "cpu_scan" in registry.tools


def test_saguaro_tools_cpu_scan_forwards_parameters():
    class _FakeSubstrate:
        def cpu_scan(self, *, path, arch, limit):
            return json.dumps({"path": path, "arch": arch, "limit": limit})

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(
        tools.cpu_scan(
            path="core/simd/common/perf_utils.h",
            arch="arm64-neon",
            limit=4,
        )
    )

    assert payload["path"] == "core/simd/common/perf_utils.h"
    assert payload["arch"] == "arm64-neon"
    assert payload["limit"] == 4


def test_api_cpu_scan_reports_hotspots(tmp_path):
    target = tmp_path / "core" / "simd" / "common" / "perf_utils.h"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "inline void kernel(float* input, float* weights, float* output, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    output[i] = input[i] * weights[i];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    payload = SaguaroAPI(repo_path=str(tmp_path)).cpu_scan(
        path="core/simd/common/perf_utils.h",
        arch="x86_64-avx2",
    )

    assert payload["status"] == "ok"
    assert payload["hotspot_count"] >= 1


def test_api_low_usage_and_deadcode_include_low_usage(monkeypatch, tmp_path):
    class _FakeAnalyzer:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def analyze(self, **kwargs):
            live = {
                "symbol": "pkg.run",
                "name": "run",
                "file": "pkg/run.py",
                "line": 12,
                "type": "function",
                "classification": "live",
                "confidence": 0.1,
                "reason": "Symbol remains reachable from entrypoints or graph roots.",
                "evidence": {
                    "usage_count": 1,
                    "referencing_files": ["main.py"],
                },
            }
            dead = {
                "symbol": "pkg.dead",
                "name": "dead",
                "file": "pkg/dead.py",
                "line": 4,
                "type": "function",
                "classification": "dead_confident",
                "confidence": 0.91,
                "reason": "Symbol has no entrypoint reachability and no static reference evidence.",
                "evidence": {
                    "usage_count": 0,
                    "referencing_files": [],
                },
            }
            max_refs = int(kwargs.get("max_low_usage_refs", 1) or 0)
            return {
                "status": "ok",
                "graph_path": "synthetic",
                "count": 2,
                "candidates": [dead, live],
                "low_usage": {
                    "max_refs": max_refs,
                    "count": 1,
                    "returned_count": 1,
                    "candidates": [live],
                    "dry_count": 1,
                    "dry_candidates": [{**live, "reuse_candidate": True, "reuse_score": 2.3}],
                    "areas": [{"path": "pkg/run.py", "count": 1, "dry_count": 1, "examples": ["pkg.run"]}],
                    "path_filter": kwargs.get("path_prefix"),
                    "limit": kwargs.get("limit"),
                },
                "summary": {"dead_confident": 1},
            }

    monkeypatch.setattr(api_module, "LivenessAnalyzer", _FakeAnalyzer)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    low_usage = api.low_usage(max_refs=2, path="pkg", limit=5)
    assert low_usage["status"] == "ok"
    assert low_usage["max_refs"] == 2
    assert low_usage["count"] == 1
    assert low_usage["returned_count"] == 1
    assert low_usage["candidates"][0]["name"] == "run"
    assert low_usage["candidates"][0]["evidence"]["usage_count"] == 1
    assert low_usage["dry_count"] == 1
    assert low_usage["dry_candidates"][0]["reuse_candidate"] is True
    assert low_usage["path_filter"] == "pkg"
    assert low_usage["limit"] == 5

    deadcode = api.deadcode(threshold=0.5)
    assert deadcode["count"] == 1
    assert deadcode["candidates"][0]["symbol"] == "dead"
    assert deadcode["low_usage"]["count"] == 1
    assert deadcode["low_usage"]["candidates"][0]["name"] == "run"
    assert deadcode["lang"] is None
    assert deadcode["evidence"] is False
    assert deadcode["runtime_observed"] is False
    assert deadcode["explain"] is False


def test_api_deadcode_compatibility_flags(monkeypatch, tmp_path):
    class _FakeAnalyzer:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def analyze(self, **kwargs):
            dead = {
                "symbol": "pkg.dead",
                "name": "dead",
                "file": "pkg/dead.py",
                "line": 4,
                "type": "function",
                "classification": "dead_confident",
                "confidence": 0.91,
                "reason": "Synthetic dead candidate.",
                "evidence": {"usage_count": 0},
            }
            return {
                "status": "ok",
                "count": 1,
                "candidates": [dead],
                "low_usage": {"max_refs": 1, "count": 0, "candidates": []},
                "summary": {"dead_confident": 1},
            }

    monkeypatch.setattr(api_module, "LivenessAnalyzer", _FakeAnalyzer)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    report = api.deadcode(
        threshold=0.5,
        lang="python",
        evidence=True,
        runtime_observed=True,
        explain=True,
    )
    assert report["count"] == 1
    assert report["lang"] == "python"
    assert report["evidence"] is True
    assert report["runtime_observed"] is True
    assert report["explain"] is True
    assert report["candidates"][0]["language"] == "python"
    assert "explanation" in report["candidates"][0]


def test_api_chronicle_snapshot_uses_verifier_state_blob(monkeypatch, tmp_path):
    captured: dict[str, Any] = {}

    class _FakeStorage:
        def __init__(self, db_path):
            self.db_path = db_path

        def save_snapshot(self, *, hd_state_blob, description, metadata):
            captured["blob"] = hd_state_blob
            captured["description"] = description
            captured["metadata"] = metadata
            return 7

    monkeypatch.setattr(api_module, "ChronicleStorage", _FakeStorage)
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    tracked = tmp_path / "tracked.py"
    tracked.write_text("print('tracked')\n", encoding="utf-8")
    api._state_ledger.record_changes(changed_files=[str(tracked)], reason="test")
    monkeypatch.setattr(api, "trace", lambda **kwargs: {"status": "ok", "stage_count": 0})

    report = api.chronicle_snapshot(description="Regression snapshot")

    assert report["status"] == "ok"
    assert report["snapshot_id"] == 7
    assert captured["blob"] == api._current_state_blob()
    assert captured["blob"] == api._state_ledger.state_projection_lines()[0].encode("utf-8") + b"\n"
    assert captured["description"] == "Regression snapshot"


def test_api_compatibility_aliases_map_to_existing_endpoints(monkeypatch, tmp_path):
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    monkeypatch.setattr(
        api,
        "doctor",
        lambda: {
            "status": "ok",
            "native_abi": {"ok": True},
            "integrity": {"status": "ready"},
            "index": {"tracked_files": 1},
            "runtime": {"repo_path": str(tmp_path)},
        },
    )
    monkeypatch.setattr(
        api,
        "unwired",
        lambda **kwargs: {
            "status": "ok",
            "summary": {"cluster_count": 1, "unreachable_node_count": 3},
            "clusters": [{"id": "cluster-1"}],
            "warnings": [],
        },
    )
    monkeypatch.setattr(
        api,
        "ffi",
        lambda **kwargs: {"status": "ok", "count": 2, "boundaries": []},
    )
    monkeypatch.setattr(
        api,
        "duplicates",
        lambda **kwargs: {"status": "ok", "count": 3, "clusters": [{"id": "dup-1"}]},
    )
    monkeypatch.setattr(
        api,
        "liveness",
        lambda **kwargs: {
            "status": "ok",
            "count": 4,
            "summary": {"unreachable_feature_clusters": 1},
        },
    )
    monkeypatch.setattr(
        api,
        "architecture_verify",
        lambda **kwargs: {
            "status": "fail",
            "count": 1,
            "findings": [
                {
                    "rule_id": "AES-ARCH-101",
                    "file": "pkg/demo.py",
                    "line": 3,
                    "message": "Illegal dependency",
                }
            ],
            "summary": {},
            "policy": {},
        },
    )

    assert api.abi(action="verify")["action"] == "verify"
    orphaned = api.abi(action="orphaned")
    assert orphaned["action"] == "orphaned"
    assert orphaned["summary"]["cluster_count"] == 1

    assert api.ffi_audit(path=".", limit=9)["action"] == "audit"
    assert api.bridge(path=".", limit=9)["action"] == "bridge"
    assert api.redundancy(path=".")["count"] == 3
    assert api.clones(path=".")["count"] == 3
    assert api.duplicate_clusters(path=".")["count"] == 3
    assert api.reachability(threshold=0.2)["count"] == 4

    violations = api.architecture_violations(path=".")
    assert violations["count"] == 1
    assert violations["violations"][0]["rule_id"] == "AES-ARCH-101"


def test_saguaro_tools_unwired_uses_json_default():
    class _FakeSubstrate:
        def unwired(
            self,
            *,
            threshold,
            min_nodes,
            min_files,
            include_tests,
            include_fragments,
            max_clusters,
            refresh_graph,
            output_format,
        ):
            return json.dumps(
                {
                    "threshold": threshold,
                    "min_nodes": min_nodes,
                    "min_files": min_files,
                    "include_tests": include_tests,
                    "include_fragments": include_fragments,
                    "max_clusters": max_clusters,
                    "refresh_graph": refresh_graph,
                    "output_format": output_format,
                }
            )

    tools = SaguaroTools(_FakeSubstrate())
    payload = json.loads(tools.unwired())
    assert payload["output_format"] == "json"
    assert payload["threshold"] == 0.55


def test_tool_schema_includes_unwired():
    unwired_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "unwired"
    )
    props = unwired_schema["parameters"]["properties"]

    assert "threshold" in props
    assert "min_nodes" in props
    assert "min_files" in props
    assert "include_tests" in props
    assert "include_fragments" in props
    assert "max_clusters" in props
    assert "refresh_graph" in props
    assert "output_format" in props


def test_saguaro_slice_backward_compat(substrate):
    res = substrate.execute_command("slice src/dummy.py.MyClass")
    assert "class MyClass" in res
    assert "my_method" in res


def test_saguaro_impact(substrate):
    f2 = os.path.join(substrate.root_dir, "main.py")
    with open(f2, "w", encoding="utf-8") as f:
        f.write("import src.dummy\n")

    res = substrate.execute_command("impact src/dummy.py")
    assert "Impact Analysis" in res
    assert "main.py" in res


def test_api_read_file_and_line_range(api):
    out = api.read_file("pkg/demo.py", start_line=1, end_line=1)
    assert out["range"] == [1, 1]
    assert "def hello" in out["content"]


def test_api_directory_and_module_ops(api):
    listing = api.list_directory("pkg", recursive=False)
    assert any(e["path"].endswith("pkg/demo.py") for e in listing["entries"])

    mod = api.module_structure(".")
    assert any(m["module_path"] == "pkg" and m["is_package"] for m in mod["modules"])


def test_api_query_prefers_relevant_authentication_entity(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "auth.py").write_text(
        "class AuthManager:\n"
        "    def login_with_password(self, username: str, password: str):\n"
        "        return True\n"
    )
    (pkg / "payments.py").write_text(
        "class InvoicePrinter:\n"
        "    def render_invoice(self):\n"
        "        return 'invoice'\n"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))

    index_result = api.index(path=".", force=True)
    assert index_result["indexed_files"] >= 2

    query = api.query("authentication system login password", k=3)
    assert query["results"], query
    top = query["results"][0]
    assert top["file"].endswith("pkg/auth.py")
    assert top["name"] in {"AuthManager", "auth.py", "login_with_password"}


def test_api_query_reports_incompatible_index(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "demo.py").write_text("def hello():\n    return 'world'\n")
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()
    stats = api._load_stats()

    vectors_dir = tmp_path / ".saguaro" / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    (vectors_dir / "index_meta.json").write_text(
        json.dumps({"dim": int(stats["total_dim"]) * 2, "count": 1, "capacity": 1})
    )
    (tmp_path / ".saguaro" / "index_schema.json").write_text(
        json.dumps({"embedding_schema_version": 1, "repo_path": str(tmp_path)})
    )

    result = api.query("hello world", k=3)
    assert result["results"] == []
    assert "error" in result
    assert "compatibility" in result["error"].lower()


def test_api_index_rebuilds_incomplete_vector_store(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "auth.py").write_text(
        "class AuthManager:\n"
        "    def login(self, username: str, password: str):\n"
        "        return True\n"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.init()

    vectors_dir = tmp_path / ".saguaro" / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    (vectors_dir / "vectors.bin").write_bytes(b"stale-index")

    index_result = api.index(path=".", force=False)
    assert index_result["force"] is True
    assert "Incomplete vector store" in (index_result["rebuild_reason"] or "")
    assert (vectors_dir / "metadata.json").exists()
    assert (vectors_dir / "index_meta.json").exists()
    assert (tmp_path / ".saguaro" / "index_schema.json").exists()

    result = api.query("authentication login password", k=3)
    assert result["results"], result


def test_tensorflow_backend_reuses_prepared_projection():
    if not backends._HAS_TF:
        pytest.skip("TensorFlow backend unavailable")

    backend = backends.TensorFlowBackend()
    projection = backend.projection(vocab_size=32, dim=8, seed=7)

    prepared1 = backend.prepare_projection(projection)
    prepared2 = backend.prepare_projection(projection)

    assert prepared1 is prepared2

    embeddings = backend.embed(np.asarray([1, 2, 3], dtype=np.int32), prepared1)
    bundled = backend.bundle(embeddings)
    assert bundled.shape == (8,)


def test_api_projection_cache_stores_prepared_projection(tmp_path):
    if not backends._HAS_TF:
        pytest.skip("TensorFlow backend unavailable")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api._backend = backends.TensorFlowBackend()
    projection = api._projection(vocab_size=32, active_dim=8)

    assert not isinstance(projection, np.ndarray)


def test_api_verify_fix_and_rollback(monkeypatch, tmp_path):
    class _RuffEngine(BaseEngine):
        def run(self, path_arg: str = ".") -> list[dict]:
            return []

        def fix(self, violation: dict[str, object]) -> bool:
            file_path = Path(self.repo_path) / str(violation["file"])
            before = file_path.read_text(encoding="utf-8")
            after = before.replace("import os,sys\n", "import os\nimport sys\n")
            if before == after:
                return False
            file_path.write_text(after, encoding="utf-8")
            return True

    class _FakeVerifier:
        def __init__(self, repo_path: str, engines=None) -> None:
            self.repo_path = repo_path
            self.engines = [_RuffEngine(repo_path)]

        def verify_all(self, path_arg: str = ".", **_: object) -> list[dict]:
            file_path = Path(path_arg)
            text = file_path.read_text(encoding="utf-8")
            if "import os,sys" not in text:
                return []
            return [
                {
                    "file": str(file_path),
                    "line": 1,
                    "rule_id": "I001",
                    "message": "Imports are unsorted",
                    "severity": "P2",
                    "aal": "AAL-2",
                    "domain": ["universal"],
                    "closure_level": "guarded",
                }
            ]

    import saguaro.sentinel.verifier as sentinel_verifier

    monkeypatch.setattr(sentinel_verifier, "SentinelVerifier", _FakeVerifier)

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "module.py"
    target.write_text("import os,sys\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    result = api.verify(path="pkg/module.py", fix=True, dry_run=False, fix_mode="safe")

    assert result["status"] == "pass"
    assert result["count"] == 0
    assert result["fixed"] == 1
    assert result["fix_plan"]["batch_count"] == 1
    assert result["fix_receipts"][0]["status"] == "applied"
    assert target.read_text(encoding="utf-8") == "import os\nimport sys\n"

    rollback = api.rollback_fix_receipts(result["receipt_dir"])

    assert rollback["status"] == "ok"
    assert rollback["results"][0]["status"] == "restored"
    assert target.read_text(encoding="utf-8") == "import os,sys\n"


def test_api_skeleton_includes_module_constants_and_dependency_graph(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    target = pkg / "settings_like.py"
    target.write_text(
        "import os\n"
        "MY_CONST = 7\n"
        "APP_CONFIG = {'mode': 'test'}\n"
        "\n"
        "def use_constant() -> int:\n"
        "    return MY_CONST\n"
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    result = api.skeleton("pkg/settings_like.py")

    constant_names = {
        symbol.get("name")
        for symbol in result.get("symbols", [])
        if symbol.get("type") == "constant"
    }
    module_constant_names = {
        symbol.get("name") for symbol in result.get("module_constants", [])
    }
    graph = result.get("dependency_graph") or {}

    assert "MY_CONST" in constant_names
    assert "APP_CONFIG" in constant_names
    assert "MY_CONST" in module_constant_names
    assert "APP_CONFIG" in module_constant_names
    assert "os" in set(graph.get("imports", []))
    assert "use_constant" in set(graph.get("exports", []))
    assert any(
        edge.get("from") == "use_constant" and edge.get("to") == "MY_CONST"
        for edge in graph.get("internal_edges", [])
    )


def test_api_slice_returns_full_long_method_without_truncation(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    long_lines = [f"        total += {idx}" for idx in range(220)]
    file_contents = (
        "class LongWorker:\n"
        "    def long_method(self):\n"
        "        total = 0\n"
        + "\n".join(long_lines)
        + "\n        marker = 'SLICE_END_MARKER'\n"
        "        return marker\n"
    )
    (pkg / "long_worker.py").write_text(file_contents)

    api = SaguaroAPI(repo_path=str(tmp_path))
    index_result = api.index(path=".", force=True)
    assert index_result["indexed_files"] >= 1

    result = api.slice(
        "LongWorker.long_method", depth=1, file_path="pkg/long_worker.py"
    )
    focus = next(
        (item for item in result.get("content", []) if item.get("role") == "focus"),
        {},
    )
    focus_code = focus.get("code", "")

    assert "SLICE_END_MARKER" in focus_code
    assert len(focus_code.splitlines()) >= 220
    assert result["corpus_id"] == "primary"
    assert result["qualified_symbol_id"].startswith("primary:pkg/long_worker.py:")
    assert focus["qualified_symbol_id"] == result["qualified_symbol_id"]


def test_api_slice_returns_ambiguity_bundle_for_duplicate_symbols(tmp_path):
    (tmp_path / "a.py").write_text("def duplicate():\n    return 'a'\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("def duplicate():\n    return 'b'\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True)

    result = api.slice("duplicate", depth=1)

    assert result["type"] == "SYMBOL_AMBIGUOUS"
    assert result["corpus_id"] == "primary"
    assert len(result["matches"]) == 2
    assert {item["file"] for item in result["matches"]} == {"a.py", "b.py"}
    assert all(
        item["qualified_symbol_id"].startswith("primary:")
        for item in result["matches"]
    )


def test_skeleton_supports_shell_and_cmake_symbols(tmp_path):
    (tmp_path / "build_native.sh").write_text(
        "build_native() {\n" "  echo building\n" "}\n"
    )
    (tmp_path / "core.cc").write_text("int core_fn() { return 1; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(Demo)\n"
        "add_library(core STATIC core.cc)\n"
        "function(configure_target target)\n"
        "endfunction()\n"
    )

    substrate = SaguaroSubstrate(root_dir=str(tmp_path))
    shell = substrate.execute_command("skeleton build_native.sh")
    cmake = substrate.execute_command("skeleton CMakeLists.txt")

    assert "build_native" in shell
    assert "core" in cmake
    assert "configure_target" in cmake


def test_skeleton_supports_react_tsx_symbols(tmp_path):
    (tmp_path / "App.tsx").write_text(
        "import React from 'react';\n"
        "export function App() {\n"
        "  return <main>Hello</main>;\n"
        "}\n"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    result = api.skeleton("App.tsx")
    names = {item.get("name") for item in result.get("symbols", [])}
    imports = set(result.get("imports", []))

    assert "App" in names
    assert any("import React" in item for item in imports)


def test_skeleton_supports_makefile_targets(tmp_path):
    (tmp_path / "Makefile").write_text(
        "all: build test\n\n" "build:\n\t@echo build\n\n" "test:\n\t@echo test\n"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    result = api.skeleton("Makefile")
    names = {item.get("name") for item in result.get("symbols", [])}

    assert "all" in names
    assert "build" in names


def test_skeleton_uses_canonical_parser_for_template_and_iac_files(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    (templates / "page.jinja2").write_text(
        "{% extends 'base.html' %}\n"
        "{% block content %}Hello{% endblock %}\n",
        encoding="utf-8",
    )
    infra = tmp_path / "infra"
    infra.mkdir()
    (infra / "main.tf").write_text(
        'module "network" {\n'
        '  source = "./modules/network"\n'
        "}\n",
        encoding="utf-8",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    template_result = api.skeleton("templates/page.jinja2")
    infra_result = api.skeleton("infra/main.tf")

    template_names = {item.get("name") for item in template_result.get("symbols", [])}
    infra_names = {item.get("name") for item in infra_result.get("symbols", [])}

    assert "content" in template_names
    assert "network" in infra_names
    assert "base.html" in set(template_result.get("imports", []))
    assert any("modules/network" in item for item in infra_result.get("imports", []))


def test_substrate_sync_prunes_deleted_files(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    keep_file = pkg / "keep.py"
    drop_file = pkg / "drop.py"
    keep_file.write_text("def keep():\n    return 1\n")
    drop_file.write_text("def drop():\n    return 2\n")

    substrate = SaguaroSubstrate(root_dir=str(tmp_path))
    first = json.loads(substrate.sync(full=True, reason="bootstrap"))
    assert first["status"] == "ok"

    drop_file.unlink()
    second = json.loads(
        substrate.sync(deleted_files=["pkg/drop.py"], reason="delete_test")
    )
    assert second["status"] == "ok"
    assert int(second["sync"]["removed_files"]) >= 1

    tracking = json.loads((tmp_path / ".saguaro" / "tracking.json").read_text())
    assert str(drop_file) not in tracking
    assert str(keep_file) in tracking


def test_tool_schema_includes_sync_workspace_and_daemon():
    names = {item.get("name") for item in TOOL_SCHEMAS["tools"]}
    assert {
        "saguaro_sync",
        "saguaro_workspace",
        "saguaro_daemon",
        "saguaro_doctor",
    } <= names


def test_substrate_execute_command_supports_sync_workspace_and_daemon(tmp_path):
    (tmp_path / "demo.py").write_text("def ping():\n    return 'ok'\n")
    substrate = SaguaroSubstrate(root_dir=str(tmp_path))

    sync_payload = json.loads(substrate.execute_command("sync --full"))
    workspace_payload = json.loads(substrate.execute_command("workspace status"))
    daemon_payload = json.loads(substrate.execute_command("daemon status"))

    assert sync_payload["status"] == "ok"
    assert workspace_payload["status"] == "ok"
    assert "running" in daemon_payload


def test_api_index_supports_event_stream_and_prune_deleted(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    keep_file = pkg / "keep.py"
    drop_file = pkg / "drop.py"
    keep_file.write_text("def keep():\n    return 1\n")
    drop_file.write_text("def drop():\n    return 2\n")

    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"path": "pkg/keep.py", "op": "upsert"}) + "\n",
        encoding="utf-8",
    )

    api = SaguaroAPI(repo_path=str(tmp_path))
    first = api.index(path=".", force=True, incremental=False)
    assert first["indexed_files"] >= 2

    drop_file.unlink()
    second = api.index(
        path=".",
        force=False,
        incremental=True,
        events_path=str(events),
        prune_deleted=True,
    )
    assert second["updated_files"] >= 1
    assert int(second["removed_files"]) >= 1

    tracking = json.loads((tmp_path / ".saguaro" / "tracking.json").read_text())
    assert str(drop_file) not in tracking


def test_api_workspace_sync_and_snapshot(tmp_path):
    (tmp_path / "demo.py").write_text(
        "def ping():\n    return 'ok'\n", encoding="utf-8"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))

    created = api.workspace(action="create", name="feature-one", switch=True)
    assert created["status"] in {"ok", "exists"}
    assert created["active"] == "feature-one"

    synced = api.sync(
        action="index",
        changed_files=["demo.py"],
        deleted_files=[],
        reason="workspace_test",
    )
    assert synced["status"] == "ok"
    assert synced["events"]["events_written"] >= 1

    status = api.workspace(action="status")
    assert status["workspace_id"] == "feature-one"
    assert status["tracked_files"] >= 1

    snap = api.workspace(action="snapshot", label="checkpoint")
    assert snap["status"] == "ok"
    assert snap["snapshot"]["label"] == "checkpoint"


def test_api_sync_peer_push_pull_cycle(tmp_path):
    (tmp_path / "demo.py").write_text(
        "def ping():\n    return 'ok'\n", encoding="utf-8"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))

    peer_added = api.sync(
        action="peer-add",
        peer_name="node-b",
        peer_url="https://node-b.local",
    )
    assert peer_added["status"] == "ok"
    peer_id = peer_added["peer"]["peer_id"]

    pushed = api.sync(action="push", peer_id=peer_id, limit=10)
    assert pushed["status"] == "ok"
    assert os.path.exists(pushed["bundle_path"])

    incoming_bundle = tmp_path / "incoming_bundle.json"
    incoming_bundle.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "path": "peer_only.py",
                        "op": "upsert",
                        "content_hash_before": None,
                        "content_hash_after": "abc123",
                        "mtime_ns": 1,
                        "instance_id": "peer-instance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pulled = api.sync(
        action="pull",
        peer_id=peer_id,
        bundle_path=str(incoming_bundle),
    )
    assert pulled["status"] == "ok"
    assert pulled["applied_events"] >= 1

    ws = api.workspace(action="status")
    assert "peer_only.py" in ws["tracked_sample"] or ws["tracked_files"] >= 1


def test_api_query_scope_and_dedupe(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "auth.py").write_text(
        "def login(username: str, password: str):\n" "    return True\n",
        encoding="utf-8",
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)
    api.workspace(action="create", name="scopews", switch=True)
    api.sync(action="index", changed_files=["pkg/auth.py"], reason="scope")

    result = api.query(
        "login username password",
        k=5,
        scope="workspace",
        dedupe_by="path",
    )
    assert result["scope"] == "workspace"
    assert result["dedupe_by"] == "path"
    assert "freshness" in result
    files = [item.get("file", "") for item in result.get("results", [])]
    assert len(files) == len(set(files))


def test_substrate_supports_workspace_sync_peer_and_doctor(tmp_path):
    (tmp_path / "demo.py").write_text(
        "def ping():\n    return 'ok'\n", encoding="utf-8"
    )
    substrate = SaguaroSubstrate(root_dir=str(tmp_path))

    created = json.loads(substrate.execute_command("workspace create feat-a --switch"))
    assert created["status"] in {"ok", "exists"}
    assert created["active"] == "feat-a"

    peer_add = json.loads(
        substrate.execute_command(
            "sync peer add --name node-c --url https://node-c.local"
        )
    )
    assert peer_add["status"] == "ok"

    peer_list = json.loads(substrate.execute_command("sync peer list"))
    assert peer_list["status"] == "ok"
    assert peer_list["count"] >= 1

    doctor = json.loads(substrate.execute_command("doctor"))
    assert doctor["status"] in {"ok", "warning"}


def test_tool_schema_sync_workspace_daemon_contracts_are_extended():
    sync_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "saguaro_sync"
    )
    workspace_schema = next(
        item
        for item in TOOL_SCHEMAS["tools"]
        if item.get("name") == "saguaro_workspace"
    )
    daemon_schema = next(
        item for item in TOOL_SCHEMAS["tools"] if item.get("name") == "saguaro_daemon"
    )

    assert "action" in sync_schema["parameters"]["properties"]
    assert "peer_id" in sync_schema["parameters"]["properties"]
    assert "workspace_id" in sync_schema["parameters"]["properties"]

    workspace_actions = set(
        workspace_schema["parameters"]["properties"]["action"]["enum"]
    )
    assert {"create", "switch", "history", "diff", "snapshot"} <= workspace_actions

    daemon_actions = set(daemon_schema["parameters"]["properties"]["action"]["enum"])
    assert "logs" in daemon_actions


def test_cli_sync_subcommands_accept_workspace_scope():
    push_help = subprocess.run(
        [sys.executable, "-m", "saguaro.cli", "sync", "push", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    pull_help = subprocess.run(
        [sys.executable, "-m", "saguaro.cli", "sync", "pull", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert "--workspace" in (push_help.stdout + push_help.stderr)
    assert "--workspace" in (pull_help.stdout + pull_help.stderr)


def test_cli_unwired_help_exposes_expected_flags():
    help_result = subprocess.run(
        [sys.executable, "-m", "saguaro.cli", "unwired", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = help_result.stdout + help_result.stderr
    assert "--threshold" in output
    assert "--min-nodes" in output
    assert "--min-files" in output
    assert "--include-tests" in output
    assert "--include-fragments" in output
    assert "--max-clusters" in output
    assert "--no-refresh-graph" in output
    assert "--format" in output


def test_cli_low_usage_command_supports_json_and_text(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def low_usage(self, *, max_refs, include_tests, path, limit):
            return {
                "status": "ok",
                "max_refs": max_refs,
                "count": 1,
                "returned_count": 1,
                "candidates": [
                    {
                        "symbol": "pkg.run",
                        "name": "run",
                        "file": "pkg/run.py",
                        "evidence": {
                            "usage_count": 1,
                            "referencing_files": ["main.py"],
                        },
                    }
                ],
                "dry_count": 1,
                "dry_candidates": [
                    {
                        "symbol": "pkg.run",
                        "file": "pkg/run.py",
                        "reuse_score": 2.1,
                        "dry_signals": ["same_file_only"],
                    }
                ],
                "areas": [
                    {
                        "path": "pkg/run.py",
                        "count": 1,
                        "dry_count": 1,
                        "examples": ["pkg.run"],
                    }
                ],
                "path_filter": path,
                "limit": limit,
            }

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "low-usage",
            "--max-refs",
            "2",
            "--path",
            "pkg",
            "--limit",
            "5",
            "--format",
            "json",
        ],
    )
    cli_module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_refs"] == 2
    assert payload["count"] == 1
    assert payload["candidates"][0]["name"] == "run"
    assert payload["path_filter"] == "pkg"
    assert payload["limit"] == 5

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "low-usage",
            "--max-refs",
            "2",
            "--path",
            "pkg",
            "--format",
            "text",
        ],
    )
    cli_module.main()
    text_output = capsys.readouterr().out
    assert "Path filter: pkg" in text_output
    assert "Top DRY candidates:" in text_output
    assert "Top areas:" in text_output
    assert "low-usage live symbols" in text_output
    assert "pkg.run" in text_output
    assert "refs: main.py" in text_output


def test_cli_index_doctor_and_rebuild_compat_forms(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def doctor(self):
            return {"status": "ok", "mode": "doctor"}

        def index(self, **kwargs):
            return {"status": "ok", "mode": "index", "kwargs": kwargs}

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)

    monkeypatch.setattr(
        sys,
        "argv",
        ["saguaro", "--repo", str(tmp_path), "index", "doctor"],
    )
    cli_module.main()
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["mode"] == "doctor"

    monkeypatch.setattr(
        sys,
        "argv",
        ["saguaro", "--repo", str(tmp_path), "index", "rebuild", "--path", "pkg"],
    )
    cli_module.main()
    rebuild_payload = json.loads(capsys.readouterr().out)
    assert rebuild_payload["mode"] == "index"
    assert rebuild_payload["kwargs"]["force"] is True
    assert rebuild_payload["kwargs"]["path"] == "pkg"


def test_cli_liveness_explain_form_dispatches_symbol(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def liveness(self, **kwargs):
            return {"status": "ok", **kwargs}

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "liveness",
            "explain",
            "pkg.run",
            "--format",
            "json",
        ],
    )
    cli_module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol"] == "pkg.run"


def test_cli_roadmap_top_level_aliases_dispatch(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def abi(self, **kwargs):
            return {"status": "ok", "cmd": "abi", **kwargs}

        def ffi_audit(self, **kwargs):
            return {"status": "ok", "cmd": "ffi", **kwargs}

        def bridge(self, **kwargs):
            return {"status": "ok", "cmd": "bridge", **kwargs}

        def redundancy(self, **kwargs):
            return {"status": "ok", "cmd": "redundancy", **kwargs}

        def clones(self, **kwargs):
            return {"status": "ok", "cmd": "clones", **kwargs}

        def duplicate_clusters(self, **kwargs):
            return {"status": "ok", "cmd": "duplicate-clusters", **kwargs}

        def reachability(self, **kwargs):
            return {"status": "ok", "cmd": "reachability", **kwargs}

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)

    def _run(argv):
        monkeypatch.setattr(sys, "argv", argv)
        cli_module.main()
        return json.loads(capsys.readouterr().out)

    verify_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "abi", "verify", "--format", "json"]
    )
    orphaned_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "abi", "orphaned", "--format", "json"]
    )
    ffi_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "ffi",
            "audit",
            "--path",
            "pkg",
            "--limit",
            "9",
            "--format",
            "json",
        ]
    )
    redundancy_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "redundancy",
            "--path",
            "pkg",
            "--symbol",
            "pkg.run",
            "--format",
            "json",
        ]
    )
    clones_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "clones",
            "--path",
            "pkg",
            "--format",
            "json",
        ]
    )
    duplicate_clusters_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "duplicate-clusters",
            "--path",
            "pkg",
            "--format",
            "json",
        ]
    )
    bridge_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "bridge",
            "--path",
            "pkg",
            "--limit",
            "7",
            "--format",
            "json",
        ]
    )
    bridge_explain_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "bridge",
            "explain",
            "pkg.run",
            "--format",
            "json",
        ]
    )
    reachability_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "reachability",
            "pkg.run",
            "--threshold",
            "0.25",
            "--format",
            "json",
        ]
    )

    assert verify_payload["cmd"] == "abi"
    assert verify_payload["action"] == "verify"
    assert orphaned_payload["action"] == "orphaned"
    assert ffi_payload["cmd"] == "ffi"
    assert ffi_payload["path"] == "pkg"
    assert ffi_payload["limit"] == 9
    assert redundancy_payload["cmd"] == "redundancy"
    assert redundancy_payload["symbol"] == "pkg.run"
    assert clones_payload["cmd"] == "clones"
    assert duplicate_clusters_payload["cmd"] == "duplicate-clusters"
    assert bridge_payload["cmd"] == "bridge"
    assert bridge_payload["limit"] == 7
    assert bridge_explain_payload["symbol"] == "pkg.run"
    assert reachability_payload["cmd"] == "reachability"
    assert reachability_payload["symbol"] == "pkg.run"
    assert reachability_payload["threshold"] == 0.25


def test_cli_architecture_violations_subcommand(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def architecture_violations(self, **kwargs):
            return {
                "status": "fail",
                "count": 1,
                "violations": [{"rule_id": "AES-ARCH-101"}],
                **kwargs,
            }

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "architecture",
            "violations",
            "--path",
            "pkg",
            "--format",
            "json",
        ],
    )
    cli_module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 1
    assert payload["path"] == "pkg"


def test_cli_deadcode_compat_flags_forwarded(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def deadcode(self, **kwargs):
            return {"status": "ok", **kwargs}

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "deadcode",
            "--lang",
            "python",
            "--evidence",
            "--runtime-observed",
            "--explain",
            "--format",
            "json",
        ],
    )
    cli_module.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["lang"] == "python"
    assert payload["evidence"] is True
    assert payload["runtime_observed"] is True
    assert payload["explain"] is True


def test_cli_deadcode_text_shows_low_usage_even_when_no_dead(
    monkeypatch, capsys, tmp_path
):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def deadcode(self, **kwargs):
            return {
                "status": "ok",
                "count": 0,
                "candidates": [],
                "low_usage": {
                    "max_refs": kwargs.get("low_usage_max_refs", 1),
                    "count": 1,
                    "candidates": [
                        {
                            "symbol": "pkg.run",
                            "file": "pkg/run.py",
                            "evidence": {
                                "usage_count": 1,
                                "referencing_files": ["main.py"],
                            },
                        }
                    ],
                },
            }

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "deadcode",
            "--format",
            "text",
        ],
    )
    cli_module.main()
    output = capsys.readouterr().out

    assert "No dead code found" in output
    assert "Low-usage live symbols" in output
    assert "pkg.run" in output


def test_cli_agent_compatibility_aliases(monkeypatch, capsys, tmp_path):
    import saguaro.cli as cli_module

    class _FakeAPI:
        def __init__(self, repo_path):
            self.repo_path = repo_path

        def architecture_map(self, **kwargs):
            return {"status": "ok", "cmd": "architecture", **kwargs}

        def bridge(self, **kwargs):
            return {"status": "ok", "cmd": "bridge", **kwargs}

        def duplicates(self, **kwargs):
            return {"status": "ok", "cmd": "duplicates", **kwargs}

        def redundancy(self, **kwargs):
            return {"status": "ok", "cmd": "redundancy", **kwargs}

        def clones(self, **kwargs):
            return {"status": "ok", "cmd": "clones", **kwargs}

        def duplicate_clusters(self, **kwargs):
            return {"status": "ok", "cmd": "duplicate-clusters", **kwargs}

        def liveness(self, **kwargs):
            return {"status": "ok", "cmd": "liveness", **kwargs}

        def reachability(self, **kwargs):
            return {"status": "ok", "cmd": "reachability", **kwargs}

        def architecture_zones(self, **kwargs):
            return {"status": "ok", "cmd": "zones", **kwargs}

        def architecture_violations(self, **kwargs):
            return {"status": "ok", "cmd": "violations", **kwargs}

        def abi(self, **kwargs):
            return {"status": "ok", "cmd": "abi", **kwargs}

        def ffi_audit(self, **kwargs):
            return {"status": "ok", "cmd": "ffi", **kwargs}

    monkeypatch.setattr(api_module, "SaguaroAPI", _FakeAPI)

    def _run(argv):
        monkeypatch.setattr(sys, "argv", argv)
        cli_module.main()
        return json.loads(capsys.readouterr().out)

    architecture_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "architecture", "--path", "pkg"]
    )
    bridge_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "bridge",
            "--path",
            "pkg",
            "--limit",
            "5",
        ]
    )
    duplicates_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "duplicates", "--path", "pkg"]
    )
    redundancy_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "redundancy",
            "--path",
            "pkg",
            "--symbol",
            "pkg.run",
        ]
    )
    clones_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "clones", "--path", "pkg"]
    )
    duplicate_clusters_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "duplicate-clusters",
            "--path",
            "pkg",
        ]
    )
    liveness_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "liveness",
            "--symbol",
            "pkg.run",
        ]
    )
    reachability_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "reachability",
            "pkg.run",
        ]
    )
    zones_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "zones", "--path", "pkg"]
    )
    violations_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "violations", "--path", "pkg"]
    )
    abi_payload = _run(
        ["saguaro", "--repo", str(tmp_path), "agent", "abi", "orphaned"]
    )
    ffi_payload = _run(
        [
            "saguaro",
            "--repo",
            str(tmp_path),
            "agent",
            "ffi",
            "--path",
            "pkg",
            "--limit",
            "11",
        ]
    )

    assert architecture_payload["cmd"] == "architecture"
    assert architecture_payload["path"] == "pkg"
    assert bridge_payload["cmd"] == "bridge"
    assert bridge_payload["limit"] == 5
    assert duplicates_payload["cmd"] == "duplicates"
    assert redundancy_payload["cmd"] == "redundancy"
    assert redundancy_payload["symbol"] == "pkg.run"
    assert clones_payload["cmd"] == "clones"
    assert duplicate_clusters_payload["cmd"] == "duplicate-clusters"
    assert liveness_payload["cmd"] == "liveness"
    assert liveness_payload["symbol"] == "pkg.run"
    assert reachability_payload["cmd"] == "reachability"
    assert reachability_payload["symbol"] == "pkg.run"
    assert zones_payload["cmd"] == "zones"
    assert violations_payload["cmd"] == "violations"
    assert abi_payload["cmd"] == "abi"
    assert abi_payload["action"] == "orphaned"
    assert ffi_payload["cmd"] == "ffi"
    assert ffi_payload["limit"] == 11


def test_file_discovery_includes_authoritative_saguaro_tree(tmp_path):
    authoritative = tmp_path / "Saguaro" / "saguaro"
    authoritative.mkdir(parents=True)
    (authoritative / "live.py").write_text(
        "def live():\n    return 1\n", encoding="utf-8"
    )
    legacy = tmp_path / "_legacy_saguaro_to_remove"
    legacy.mkdir(parents=True)
    (legacy / "legacy.py").write_text("def old():\n    return 2\n", encoding="utf-8")

    files = get_code_files(str(tmp_path))
    rels = {str(Path(path).relative_to(tmp_path)).replace("\\", "/") for path in files}
    assert "Saguaro/saguaro/live.py" in rels
    assert "_legacy_saguaro_to_remove/legacy.py" not in rels


def test_native_index_coordinator_uses_bounded_parallel_defaults(monkeypatch):
    monkeypatch.delenv("SAGUARO_INDEX_WORKERS", raising=False)
    monkeypatch.delenv("SAGUARO_DISABLE_PARALLEL_INDEXING", raising=False)
    monkeypatch.delenv("SAGUARO_INDEX_MAX_WORKERS", raising=False)
    monkeypatch.setattr(
        "saguaro.indexing.native_coordinator._available_cpu_count", lambda: 16
    )
    assert _worker_count(batch_count=32, threads_per_worker=1) == 16

    monkeypatch.setenv("SAGUARO_INDEX_WORKERS", "3")
    assert _worker_count(batch_count=32, threads_per_worker=1) == 3

    monkeypatch.delenv("SAGUARO_INDEX_WORKERS", raising=False)
    monkeypatch.setenv("SAGUARO_DISABLE_PARALLEL_INDEXING", "1")
    assert _worker_count(batch_count=32, threads_per_worker=1) == 1


def test_native_index_coordinator_defaults_to_fork_context_on_linux(monkeypatch):
    monkeypatch.delenv("SAGUARO_INDEX_MP_CONTEXT", raising=False)
    monkeypatch.setattr(native_coordinator_module.sys, "platform", "linux")
    assert _process_pool_context() == "fork"

    monkeypatch.setenv("SAGUARO_INDEX_MP_CONTEXT", "fork")
    assert _process_pool_context() == "fork"


def test_native_index_coordinator_respects_bounded_executor_size(monkeypatch):
    class _DummyEngine:
        active_dim = 32
        total_dim = 64
        vocab_size = 128
        tracker = types.SimpleNamespace(update=lambda *_args, **_kwargs: None)

        def __init__(self) -> None:
            self.store = types.SimpleNamespace(remove_file=lambda *_args, **_kwargs: 0)
            self.created = False
            self.cleaned = False
            self.committed = False

        def create_shared_projection(self) -> None:
            self.created = True

        def cleanup_shared_projection(self) -> None:
            self.cleaned = True

        def ingest_worker_result(self, meta_list, vectors_np):
            return (len({item["file"] for item in meta_list}), len(meta_list))

        def commit(self) -> None:
            self.committed = True

    class _FakeFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    captured: dict[str, int] = {}

    class _FakeExecutor:
        def __init__(self, *, max_workers, mp_context):
            captured["max_workers"] = max_workers
            captured["mp_context"] = mp_context
            self._futures = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, batch, *args):
            future = _FakeFuture(fn(batch, *args))
            self._futures.append(future)
            return future

    def _fake_as_completed(futures):
        return list(futures)

    def _fake_worker(batch, *_args, **_kwargs):
        meta = [
            {
                "file": file_path,
                "name": Path(file_path).stem,
                "type": "file",
                "line": 1,
                "end_line": 1,
            }
            for file_path in batch
        ]
        return (
            meta,
            np.zeros((len(batch), 64), dtype=np.float32),
            list(batch),
            {
                "parse_seconds": float(len(batch)) * 0.1,
                "pipeline_seconds": float(len(batch)) * 0.2,
                "files_with_entities": len(batch),
            },
        )

    monkeypatch.setattr(native_coordinator_module, "_available_cpu_count", lambda: 16)
    monkeypatch.setattr(
        native_coordinator_module.concurrent.futures,
        "ProcessPoolExecutor",
        _FakeExecutor,
    )
    monkeypatch.setattr(
        native_coordinator_module.concurrent.futures,
        "wait",
        lambda futures, return_when=None: (set(_fake_as_completed(futures)), set()),
    )
    monkeypatch.setattr(
        native_coordinator_module,
        "process_batch_worker_native",
        _fake_worker,
    )

    engine = _DummyEngine()
    file_paths = [f"/tmp/file_{idx}.py" for idx in range(48)]

    result = native_coordinator_module.run_native_index_coordinator(
        engine=engine,
        file_paths=file_paths,
        batch_size=4,
    )

    assert captured["max_workers"] == 12
    assert result["workers"] == 12
    assert result["batches_processed"] == 48
    assert result["indexed_files"] == len(file_paths)
    assert result["parse_seconds"] == pytest.approx(4.8)
    assert result["pipeline_seconds"] == pytest.approx(9.6)
    assert result["files_with_entities"] == len(file_paths)
    assert engine.created is True
    assert engine.cleaned is True
    assert engine.committed is True


def test_native_runtime_degrades_when_capture_matcher_symbol_is_missing(monkeypatch):
    def _stub(*_args, **_kwargs):
        return 0

    class _FakeLib:
        saguaro_native_init_projection = staticmethod(_stub)
        saguaro_native_full_pipeline = staticmethod(_stub)
        saguaro_native_trie_create = staticmethod(lambda: 1)
        saguaro_native_trie_destroy = staticmethod(_stub)
        saguaro_native_trie_build_from_table = staticmethod(_stub)

    monkeypatch.setattr(NativeIndexRuntime, "_instance", None)
    monkeypatch.setattr(NativeIndexRuntime, "_lib", None)

    def _fake_load_library(self):
        NativeIndexRuntime._lib = _FakeLib()
        self._lib_path = "/tmp/fake_saguaro_native.so"

    monkeypatch.setattr(NativeIndexRuntime, "_load_library", _fake_load_library)
    monkeypatch.setattr(NativeIndexRuntime, "_load_manifest", lambda self: {})

    runtime = NativeIndexRuntime()
    assert runtime._match_capture_names is None

    with pytest.raises(NativeRuntimeError):
        runtime.match_capture_names(
            def_starts=[0],
            def_ends=[10],
            def_type_ids=[1],
            name_starts=[1],
            name_ends=[2],
            name_type_ids=[1],
        )

    parser = SAGUAROParser()
    parser._native_runtime = runtime
    matches = parser._match_tree_sitter_capture_names(
        def_starts=[0, 4],
        def_ends=[20, 10],
        def_type_ids=[1, 1],
        name_starts=[6],
        name_ends=[8],
        name_type_ids=[1],
    )
    assert matches == [-1, 0]

    monkeypatch.setattr(NativeIndexRuntime, "_instance", None)
    monkeypatch.setattr(NativeIndexRuntime, "_lib", None)


def test_file_discovery_includes_expanded_language_families(tmp_path):
    (tmp_path / "infra").mkdir()
    (tmp_path / "infra" / "main.tf").write_text('module "x" {}', encoding="utf-8")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "page.jinja2").write_text(
        "{% block body %}{% endblock %}",
        encoding="utf-8",
    )
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "Main.qml").write_text("Item {}", encoding="utf-8")

    files = get_code_files(str(tmp_path))
    rels = {str(Path(path).relative_to(tmp_path)).replace("\\", "/") for path in files}

    assert "infra/main.tf" in rels
    assert "templates/page.jinja2" in rels
    assert "ui/Main.qml" in rels


def test_index_coordinator_discovers_and_syncs_changes(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    keep_file = pkg / "keep.py"
    drop_file = pkg / "drop.py"
    keep_file.write_text("def keep():\n    return 1\n", encoding="utf-8")
    drop_file.write_text("def drop():\n    return 2\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)
    coordinator = IndexCoordinator(repo_path=str(tmp_path), api=api)

    keep_file.write_text("def keep():\n    return 3\n", encoding="utf-8")
    drop_file.unlink()
    discovered = coordinator.discover_changes(path=".")
    assert "pkg/keep.py" in set(discovered["changed_files"])
    assert "pkg/drop.py" in set(discovered["deleted_files"])

    synced = coordinator.sync(
        path=".",
        changed_files=discovered["changed_files"],
        deleted_files=discovered["deleted_files"],
        reason="coordinator_test",
    )
    assert synced["status"] == "ok"
    assert int(((synced.get("index") or {}).get("removed_files", 0) or 0)) >= 1


def test_eval_retrieval_quality_suite_reports_metrics(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (pkg / "beta.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    (pkg / "gamma.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")

    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)
    result = api.eval_run("retrieval_quality", k=5, limit=3)

    assert result["status"] == "ok"
    assert result["suite"] == "retrieval_quality"
    assert "hit_at_5" in result
    assert "mrr" in result
    assert "dedupe_ratio" in result
    assert "stale_hit_rate" in result
    assert "slo" in result


def test_doctor_reports_native_abi_payload(tmp_path):
    (tmp_path / "demo.py").write_text("def ping():\n    return 1\n", encoding="utf-8")
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)
    report = api.doctor()
    assert report["status"] in {"ok", "warning"}
    assert "native_abi" in report


def test_api_sandbox_commit_returns_sync_payload(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "demo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    api = SaguaroAPI(repo_path=str(tmp_path))
    api.index(path=".", force=True, incremental=False)

    patch = {
        "operations": [
            {
                "op": "replace",
                "content": "def hello():\n    return 'patched'\n",
            }
        ]
    }
    sandbox = api.sandbox_patch("pkg/demo.py", patch)
    result = api.sandbox_commit(sandbox["sandbox_id"])
    assert result["status"] == "ok"
    assert result["files_committed"] == 1
    assert "pkg/demo.py" in result["changed_files"]
    assert (result["sync"] or {}).get("status") == "ok"


def test_result_filter_rejects_legacy_entity_paths(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "demo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    api = SaguaroAPI(repo_path=str(tmp_path))
    assert (
        api._result_is_in_repo(  # noqa: SLF001
            {
                "file": "pkg/demo.py",
                "entity_id": "Saguaro/saguaro/pkg/demo.py:hello:function:1",
            }
        )
        is False
    )
    assert (
        api._result_is_in_repo(  # noqa: SLF001
            {
                "file": "pkg/demo.py",
                "entity_id": "pkg/demo.py:hello:function:1",
            }
        )
        is True
    )


def test_query_filters_legacy_entity_ids(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "demo.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    api = SaguaroAPI(repo_path=str(tmp_path))

    class _StubQueryService:
        def query(self, **kwargs):
            return {
                "results": [
                    {
                        "name": "hello",
                        "qualified_name": "hello",
                        "entity_id": "Saguaro/saguaro/pkg/demo.py:hello:function:1",
                        "file": "pkg/demo.py",
                        "line": 1,
                        "score": 0.9,
                    },
                    {
                        "name": "hello",
                        "qualified_name": "hello",
                        "entity_id": "pkg/demo.py:hello:function:1",
                        "file": "pkg/demo.py",
                        "line": 1,
                        "score": 0.8,
                    },
                ],
                "execution_strategy": "lexical",
                "aes_envelope": {},
                "semantic_candidates": 0,
                "lexical_candidates": 2,
            }

    api._query_service = _StubQueryService()  # noqa: SLF001
    result = api.query("hello", strategy="lexical", k=5)
    assert len(result["results"]) == 1
    assert result["results"][0]["entity_id"].startswith("pkg/demo.py:")


def test_parser_extracts_react_arrow_component_symbols(tmp_path):
    app_file = tmp_path / "App.tsx"
    app_file.write_text(
        "import React from 'react';\n"
        "const App = () => <main>Hello</main>;\n"
        "export default App;\n",
        encoding="utf-8",
    )
    parser = SAGUAROParser()
    entities = parser.parse_file(str(app_file))
    names = {item.name for item in entities if item.type in {"function", "class"}}
    assert "App" in names


def test_coverage_marks_shell_and_cmake_parser_support(tmp_path):
    (tmp_path / "build_native.sh").write_text(
        "build_native() { echo hi; }\n", encoding="utf-8"
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "add_library(core STATIC core.cc)\n", encoding="utf-8"
    )
    api = SaguaroAPI(repo_path=str(tmp_path))
    report = api.coverage(path=".", structural=True, by_language=True)
    shell_row = (report.get("language_breakdown") or {}).get("Shell") or {}
    cmake_row = (report.get("language_breakdown") or {}).get("CMake") or {}
    assert int(shell_row.get("structural_supported_files", 0)) >= 1
    assert int(cmake_row.get("structural_supported_files", 0)) >= 1
    assert int(shell_row.get("dependency_quality_supported_files", 0)) >= 1
    assert int(cmake_row.get("dependency_quality_supported_files", 0)) >= 1
    assert int(shell_row.get("ast_supported_files", 0)) >= 1
    assert int(cmake_row.get("ast_supported_files", 0)) == 0
    assert float(report.get("dependency_quality_coverage_percent", 0.0)) >= 100.0


def test_coverage_ast_support_matches_runtime_parser_surface(tmp_path):
    from saguaro.coverage import CoverageReporter

    reporter = CoverageReporter(str(tmp_path))
    parser = reporter.parser

    assert reporter._is_ast_supported("Python", tree_sitter_available=False) == parser.supports_ast_language("python")  # noqa: SLF001
    assert reporter._is_ast_supported("JavaScript", tree_sitter_available=True) == parser.supports_ast_language("javascript")  # noqa: SLF001
    assert reporter._is_ast_supported("Go", tree_sitter_available=True) == parser.supports_ast_language("go")  # noqa: SLF001
    assert reporter._is_ast_supported("Java", tree_sitter_available=True) == parser.supports_ast_language("java")  # noqa: SLF001
    assert reporter._is_ast_supported("Rust", tree_sitter_available=True) == parser.supports_ast_language("rust")  # noqa: SLF001
    assert reporter._is_dependency_quality_supported("Shell") is True  # noqa: SLF001
    assert reporter._is_dependency_quality_supported("CMake") is True  # noqa: SLF001


def test_coverage_percent_is_consistent_across_requested_views(tmp_path):
    (tmp_path / "build_native.sh").write_text(
        "build_native() { echo hi; }\n", encoding="utf-8"
    )
    reporter = CoverageReporter(str(tmp_path))

    dependency_view = reporter.generate_report(structural=False)
    structural_view = reporter.generate_report(structural=True)

    assert dependency_view["coverage_percent"] == structural_view["coverage_percent"]
    assert dependency_view["coverage_percent"] == dependency_view["dependency_quality_coverage_percent"]
    assert structural_view["requested_coverage_percent"] == structural_view["structural_coverage_percent"]


def test_api_defers_backend_for_roadmap_validation(monkeypatch, tmp_path):
    roadmap = tmp_path / "ROADMAP.md"
    cli_file = tmp_path / "saguaro" / "cli.py"
    test_file = tmp_path / "tests" / "test_saguaro_roadmap_validator.py"
    cli_file.parent.mkdir(parents=True)
    test_file.parent.mkdir(parents=True)
    cli_file.write_text("def main():\n    return 0\n", encoding="utf-8")
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    roadmap.write_text(
        "# Roadmap\n\n"
        "## 24. Recommended CLI Surface\n\n"
        "### 24.1 Docs and requirements\n\n"
        "- `saguaro docs parse --path .`\n",
        encoding="utf-8",
    )

    def _unexpected_backend(*_args, **_kwargs):
        raise AssertionError("backend initialization should be deferred")

    monkeypatch.setattr(api_module, "get_backend", _unexpected_backend)
    api = SaguaroAPI(repo_path=str(tmp_path))

    result = api.roadmap_validate(path="ROADMAP.md")

    assert result["summary"]["count"] == 1
    assert result["summary"]["completed"] == 1


def test_cli_agent_skeleton_bypasses_api_import(monkeypatch, tmp_path, capsys):
    target = tmp_path / "demo.py"
    target.write_text("def hello():\n    return 'world'\n", encoding="utf-8")

    import builtins

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "saguaro.api":
            raise AssertionError("agent skeleton should not import saguaro.api")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["saguaro", "agent", "skeleton", str(target)])
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["file_path"].endswith("demo.py")


def test_cli_health_bypasses_api_import(monkeypatch, tmp_path, capsys):
    import builtins

    (tmp_path / ".saguaro").mkdir()

    original_import = builtins.__import__

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "saguaro.api":
            raise AssertionError("health should not import saguaro.api")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["saguaro", "health"])
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert "integrity" in payload
    assert "locks" in payload


def test_cli_debuginfo_dispatches(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def _fake_debuginfo(self, *, output_path=None, event_limit=500):
        captured["output_path"] = output_path
        captured["event_limit"] = event_limit
        return {"status": "ok", "path": output_path or "bundle.tar.gz"}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SaguaroAPI, "debuginfo", _fake_debuginfo)
    monkeypatch.setattr(
        sys,
        "argv",
        ["saguaro", "debuginfo", "--output", "diag.tar.gz", "--event-limit", "7"],
    )

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert captured == {"output_path": "diag.tar.gz", "event_limit": 7}


def test_cli_state_restore_dispatches(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def _fake_state_restore(self, *, bundle_path, force=False):
        captured["bundle_path"] = bundle_path
        captured["force"] = force
        return {"status": "ok", "bundle_path": bundle_path}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SaguaroAPI, "state_restore", _fake_state_restore)
    monkeypatch.setattr(
        sys,
        "argv",
        ["saguaro", "state", "restore", "state.tar.gz", "--force"],
    )

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert captured == {"bundle_path": "state.tar.gz", "force": True}


def test_cli_admin_dispatches(monkeypatch, tmp_path, capsys):
    captured: dict[str, object] = {}

    def _fake_admin(
        self,
        *,
        action,
        bundle_path=None,
        output_path=None,
        force=False,
        include_reality=True,
        event_limit=500,
    ):
        captured.update(
            {
                "action": action,
                "bundle_path": bundle_path,
                "output_path": output_path,
                "force": force,
                "include_reality": include_reality,
                "event_limit": event_limit,
            }
        )
        return {"status": "ok", "admin_action": action}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(SaguaroAPI, "admin", _fake_admin)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "saguaro",
            "admin",
            "snapshot",
            "--output",
            "bundle.tar.gz",
            "--no-reality",
        ],
    )

    cli_module.main()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert captured == {
        "action": "snapshot",
        "bundle_path": None,
        "output_path": "bundle.tar.gz",
        "force": False,
        "include_reality": False,
        "event_limit": 500,
    }
