from __future__ import annotations

from pathlib import Path

from saguaro.analysis.bridge_synthesizer import BridgeSynthesizer
from saguaro.analysis.complexity_analyzer import ComplexityAnalyzer
from saguaro.analysis.ffi_scanner import FFIScanner, SharedObjectResolver
from saguaro.analysis.flop_counter import FLOPCounter
from saguaro.analysis.pipeline_diff import PipelineDiff
from saguaro.analysis.pipeline_tracer import (
    PipelineStage,
    PipelineTrace,
    PipelineTracer,
)
from saguaro.analysis.trace_output import TraceOutputFormatter


def test_ffi_scanner_typed_boundaries_and_multi_hop_chain() -> None:
    source = (
        "import ctypes\n"
        "lib = ctypes.CDLL('build/libnative_ops.so')\n"
        "from cffi import FFI\n"
        "ffi = FFI()\n"
        "ffi.dlopen('libnative_ops.so')\n"
    )
    scanner = FFIScanner(repo_path=".")
    findings = scanner.scan_file("bridge.py", source)

    assert len(findings) >= 2
    ctypes_item = next(
        item for item in findings if item["kind"] == "ctypes_load_library"
    )
    cffi_item = next(item for item in findings if item["kind"] == "cffi_dlopen")
    assert ctypes_item["boundary_type"].startswith("ctypes.")
    assert cffi_item["boundary_type"].startswith("cffi.")
    assert ctypes_item["shared_object"] == "libnative_ops.so"
    assert any(
        "src/native_ops." in item for item in ctypes_item["shared_object_candidates"]
    )

    chains = scanner.build_multi_hop_chains(findings, max_hops=4)
    assert chains
    assert chains[0]["hop_count"] >= 2
    chain = scanner.find_chain_for_finding(findings, finding_id=cffi_item["id"])
    assert chain is not None
    assert chain["token"] == "libnative_ops.so"  # noqa: S105 - test fixture token key


def test_shared_object_resolver_heuristics() -> None:
    candidates = SharedObjectResolver.resolve_candidates(
        "libkernels.so",
        rel_file="saguaro/analysis/ffi_scanner.py",
        repo_path=".",
    )
    assert "build/libkernels.so" in candidates
    assert any(item.endswith("src/kernels.cc") for item in candidates)


def test_bridge_synthesizer_uses_typed_fields() -> None:
    patterns = [
        {
            "id": "ffi::bridge.py::ctypes_load_library::2::1",
            "file": "bridge.py",
            "line": 2,
            "role": "consumer",
            "confidence": 0.95,
            "library_hint": "libnative_ops.so",
            "shared_object": "libnative_ops.so",
            "boundary_type": "ctypes.cdll",
        },
        {
            "id": "ffi::native.cc::extern_c_export::8::1",
            "file": "native.cc",
            "line": 8,
            "role": "provider",
            "confidence": 0.8,
            "library_hint": "libnative_ops.so",
            "shared_object": "libnative_ops.so",
            "boundary_type": "extern_c_export",
        },
    ]
    bridges = BridgeSynthesizer().synthesize(patterns)
    assert bridges
    assert bridges[0]["reason"] in {"shared_object_match", "library_hint_match"}
    assert bridges[0]["boundary_pair"] == ["ctypes.cdll", "extern_c_export"]


def test_pipeline_tracer_adds_conditional_labels_and_loop_annotations() -> None:
    graph = {
        "nodes": {
            "entry": {
                "name": "entry",
                "qualified_name": "entry",
                "file": "main.py",
                "line": 1,
            },
            "branch": {
                "name": "branch",
                "qualified_name": "branch",
                "file": "main.py",
                "line": 5,
            },
            "loop": {
                "name": "loop",
                "qualified_name": "loop",
                "file": "main.py",
                "line": 9,
            },
            "end": {
                "name": "end",
                "qualified_name": "end",
                "file": "main.py",
                "line": 12,
            },
        },
        "edges": {
            "e1": {
                "from": "entry",
                "to": "branch",
                "relation": "true",
                "data": "if x > 0",
            },
            "e2": {"from": "entry", "to": "end", "relation": "false", "data": "else"},
            "e3": {"from": "branch", "to": "loop", "relation": "calls"},
            "e4": {
                "from": "loop",
                "to": "branch",
                "relation": "back",
                "data": "loop back",
            },
        },
    }
    tracer = PipelineTracer(repo_path=".")
    trace = tracer.trace(
        "entry", graph, include_stdlib=True, include_complexity=False, max_depth=6
    )

    assert len(trace.stages) >= 3
    assert any(bool(edge.get("conditional")) for edge in trace.edges)
    assert any(bool(edge.get("loop")) for edge in trace.edges)
    assert any(edge.get("condition") == "x > 0" for edge in trace.edges)
    entry_stage = next(stage for stage in trace.stages if stage.id == "entry")
    assert "branch_stage" in entry_stage.annotations


def test_complexity_analyzer_emits_amortized_worst_and_parameterized(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "algo.py"
    file_path.write_text(
        "def tally(items):\n"
        "    table = {}\n"
        "    for item in items:\n"
        "        table.setdefault(item, 0)\n"
        "        table[item] = table.get(item, 0) + 1\n"
        "    return table\n",
        encoding="utf-8",
    )
    analyzer = ComplexityAnalyzer(repo_path=str(tmp_path))
    estimate = analyzer.analyze_function("tally", str(file_path), cfg={})

    assert estimate.time_complexity.startswith("O(")
    assert estimate.amortized_time_complexity is not None
    assert estimate.worst_case_time_complexity is not None
    assert "n" in estimate.parameterized_variables


def test_flop_counter_detects_norm_embedding_loss_and_attention_variants(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "model.py"
    file_path.write_text(
        "def model(x, q, k, v, tok, logits, target):\n"
        "    y = layer_norm(x)\n"
        "    y = rmsnorm(y)\n"
        "    y = flash_attn(q, k, v)\n"
        "    y = mqa_attention(q, k, v)\n"
        "    y = gqa_attention(q, k, v)\n"
        "    y = embedding(tok)\n"
        "    l1 = cross_entropy_loss(logits, target)\n"
        "    l2 = mse_loss(logits, target)\n"
        "    return y, l1, l2\n",
        encoding="utf-8",
    )
    counter = FLOPCounter(repo_path=str(tmp_path))
    estimates = counter.count_function(str(file_path), "model")
    ops = {item.operation for item in estimates}

    assert "layernorm" in ops
    assert "rmsnorm" in ops
    assert "flash_attention" in ops
    assert "mqa_attention" in ops
    assert "gqa_attention" in ops
    assert "embedding" in ops
    assert "cross_entropy_loss" in ops
    assert "mse_loss" in ops


def test_trace_output_html_export_and_pipeline_diff(tmp_path: Path) -> None:
    before = PipelineTrace(
        name="Pipeline<entry>",
        entry_point="entry",
        stages=[
            PipelineStage(
                id="entry",
                name="entry",
                file_path="main.py",
                start_line=1,
                end_line=3,
                language="python",
                function_signature="entry()",
                description="",
                stage_index=0,
                annotations=["branch_stage"],
            )
        ],
        edges=[],
    )
    after = PipelineTrace(
        name="Pipeline<entry>",
        entry_point="entry",
        stages=[
            before.stages[0],
            PipelineStage(
                id="next",
                name="next_stage",
                file_path="main.py",
                start_line=5,
                end_line=9,
                language="python",
                function_signature="next_stage()",
                description="",
                stage_index=1,
                annotations=["loop_stage"],
            ),
        ],
        edges=[
            {
                "from": "entry",
                "to": "next",
                "relation": "true",
                "data": "if ready",
                "label": "if ready",
                "condition": "ready",
                "conditional": True,
            }
        ],
    )

    formatter = TraceOutputFormatter()
    html = formatter.to_html(after, interactive=True)
    assert "<html" in html.lower()

    out_path = tmp_path / "trace.html"
    exported = formatter.export(
        after, fmt="html", output_path=str(out_path), interactive=True
    )
    assert exported == str(out_path)
    assert out_path.exists()

    mermaid = formatter.to_mermaid(after)
    assert "if ready" in mermaid

    diff = PipelineDiff().diff(before, after)
    assert diff["summary"]["added_stage_count"] == 1
    assert diff["summary"]["added_edge_count"] == 1


def test_pipeline_diff_accepts_serialized_trace_dicts() -> None:
    before = {
        "stages": [
            {"id": "entry", "name": "entry", "file": "main.py", "complexity": {}}
        ],
        "edges": [],
    }
    after = {
        "stages": [
            {"id": "entry", "name": "entry", "file": "main.py", "complexity": {}},
            {"id": "decode", "name": "decode", "file": "main.py", "complexity": {}},
        ],
        "edges": [{"from": "entry", "to": "decode", "relation": "calls"}],
    }

    diff = PipelineDiff().diff(before, after)
    assert diff["summary"]["added_stage_count"] == 1
    assert diff["summary"]["added_edge_count"] == 1
