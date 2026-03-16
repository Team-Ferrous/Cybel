from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any


def _load_lane_context_function():
    repl_path = Path("cli/repl.py")
    module = ast.parse(repl_path.read_text(encoding="utf-8"), filename=str(repl_path))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "AgentREPL":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "_deterministic_synthesis_lane_context"
                ):
                    item.decorator_list = []
                    isolated = ast.Module(body=[item], type_ignores=[])
                    ast.fix_missing_locations(isolated)
                    namespace: dict[str, Any] = {"os": os, "Any": Any}
                    exec(compile(isolated, filename=str(repl_path), mode="exec"), namespace)
                    return namespace[item.name]
    raise AssertionError("lane context helper not found in cli/repl.py")


def test_repl_detects_explicit_deterministic_synthesis_lane() -> None:
    lane_context = _load_lane_context_function()
    context = lane_context("synth: implement deterministic adapter generation")

    assert context["enabled"] is True
    assert context["label"] == "deterministic-synthesis"
    assert context["badge"] == "DET-SYNTH"


def test_repl_defaults_to_standard_lane_without_explicit_signal() -> None:
    lane_context = _load_lane_context_function()
    context = lane_context("explain the runtime capability ledger")

    assert context["enabled"] is False
    assert context["label"] == "standard-mission"
