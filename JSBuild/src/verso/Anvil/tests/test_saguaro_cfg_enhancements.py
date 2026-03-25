from __future__ import annotations

import json
import textwrap

from saguaro.analysis.cfg_builder import CFGBuilder
from saguaro.analysis.icfg_builder import ICFGBuilder


def test_cfg_builder_build_api_remains_compatible_and_serializable() -> None:
    source = textwrap.dedent("""
        def compute(x):
            return x + 1
        """)
    payload = CFGBuilder().build("sample.py", source)

    assert set(payload) == {"nodes", "edges"}
    assert isinstance(payload["nodes"], list)
    assert isinstance(payload["edges"], list)
    assert any(node["type"] == "cfg_entry" for node in payload["nodes"])
    assert any(node["type"] == "cfg_exit" for node in payload["nodes"])
    assert any(edge["relation"] == "cfg_terminate" for edge in payload["edges"])

    encoded = json.dumps(payload)
    assert isinstance(encoded, str)


def test_cfg_builder_emits_exception_aware_and_with_finally_edges() -> None:
    source = textwrap.dedent("""
        def load(path):
            try:
                with open(path) as handle:
                    return handle.read()
            except OSError:
                return ""
            finally:
                path = path.strip()
        """)
    payload = CFGBuilder().build("flow.py", source)
    relations = {edge["relation"] for edge in payload["edges"]}

    assert "cfg_try" in relations
    assert "cfg_except" in relations
    assert "cfg_finally" in relations
    assert "cfg_with_finally" in relations
    assert "cfg_exception" in relations or "cfg_finally_exception" in relations


def test_cfg_builder_emits_async_concurrency_markers() -> None:
    source = textwrap.dedent("""
        import asyncio

        async def orchestrate(coro):
            task = asyncio.create_task(coro())
            await asyncio.gather(task)
            return await coro()
        """)
    payload = CFGBuilder().build("async_flow.py", source)
    roles = {node.get("concurrency_role") for node in payload["nodes"]}
    relations = {edge["relation"] for edge in payload["edges"]}

    assert {"fork", "join", "suspend", "resume"}.issubset(roles)
    assert "cfg_fork" in relations
    assert "cfg_join" in relations
    assert "cfg_suspend" in relations
    assert "cfg_resume" in relations


def test_icfg_builder_links_callsites_and_marks_recursive_backedges() -> None:
    source = textwrap.dedent("""
        def alpha(n):
            if n <= 0:
                return 0
            return beta(n - 1)

        def beta(n):
            if n <= 0:
                return 0
            return alpha(n - 1)
        """)
    payload = ICFGBuilder().build("recursive.py", source)
    call_edges = [edge for edge in payload["edges"] if edge["relation"] == "icfg_call"]
    return_edges = [
        edge for edge in payload["edges"] if edge["relation"] == "icfg_return"
    ]
    recursive_edges = [
        edge
        for edge in payload["edges"]
        if edge.get("recursive_backedge")
        or edge["relation"] == "icfg_recursive_backedge"
    ]

    assert call_edges
    assert return_edges
    assert any(edge["to"].endswith("::entry::icfg") for edge in call_edges)
    assert any(edge["from"].endswith("::exit::icfg") for edge in return_edges)
    assert recursive_edges
