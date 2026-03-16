from __future__ import annotations

from saguaro.analysis.dfg_builder import DFGBuilder, TaintTracker, TypeInferencer


def test_taint_tracker_detects_unsanitized_and_skips_sanitized_flow() -> None:
    source = """
import shlex
import subprocess

def run():
    user = input("cmd")
    safe = shlex.quote(user)
    subprocess.run(user, shell=True)
    subprocess.run(safe, shell=True)
"""
    tracker = TaintTracker()
    flows = tracker.analyze("pkg/sample.py", source)

    assert len(flows) == 1
    flow = flows[0]
    assert flow.source == "input"
    assert flow.sink == "subprocess.run"
    assert flow.variable == "user"
    assert flow.sink_line > flow.source_line


def test_type_inferencer_flow_sensitive_narrowing_and_tensor_hint() -> None:
    source = """
import torch

def infer(value):
    bucket = [value]
    if isinstance(value, int):
        narrowed = value + 1
    else:
        narrowed = str(value)
    tensor = torch.tensor(bucket)
    return narrowed, tensor
"""
    inferencer = TypeInferencer()
    types = inferencer.analyze("pkg/sample.py", source)

    function_scope = types["<module>.infer"]
    assert function_scope["bucket"] == ("list",)
    assert function_scope["narrowed"] == ("int", "str")
    assert function_scope["tensor"] == ("tensor",)


def test_dfg_builder_api_compat_and_interprocedural_fallback() -> None:
    source = """
import subprocess

def run():
    user = input("cmd")
    payload = [user]
    subprocess.run(user, shell=True)
    return payload
"""
    builder = DFGBuilder()

    payload = builder.build("pkg/sample.py", source)
    assert set(payload) == {"nodes", "edges"}
    assert [node["id"] for node in payload["nodes"]] == sorted(
        node["id"] for node in payload["nodes"]
    )

    assert any(
        node.get("type") == "dfg_def"
        and node.get("name") == "payload"
        and "list" in node.get("type_hints", [])
        for node in payload["nodes"]
    )
    assert any(edge.get("relation") == "dfg_taint_flow" for edge in payload["edges"])

    interprocedural = builder.build_interprocedural(
        "pkg/sample.py",
        source,
        call_resolver=lambda *_args, **_kwargs: {"resolved": []},
    )
    assert interprocedural == payload

    tracker = TaintTracker()
    assert tracker.analyze_interprocedural("pkg/sample.py", source) == tracker.analyze(
        "pkg/sample.py", source
    )

    inferencer = TypeInferencer()
    assert inferencer.analyze_interprocedural(
        "pkg/sample.py",
        source,
        call_resolver=lambda *_args, **_kwargs: None,
    ) == inferencer.analyze("pkg/sample.py", source)
