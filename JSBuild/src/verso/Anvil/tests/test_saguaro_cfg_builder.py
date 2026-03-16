from __future__ import annotations

from pathlib import Path

from saguaro.agents.perception import TracePerception


def test_cfg_builder_detects_branches_and_loops(tmp_path: Path) -> None:
    target = tmp_path / "flow.py"
    target.write_text(
        "def compute(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        if item > 0:\n"
        "            total += item\n"
        "        else:\n"
        "            total -= item\n"
        "    return total\n",
        encoding="utf-8",
    )

    trace = TracePerception(repo_path=str(tmp_path))
    cfg = trace.build_cfg("flow.py", symbol="compute")

    assert cfg["status"] == "ok"
    assert cfg["node_count"] >= 5
    relations = {edge["relation"] for edge in cfg["edges"]}
    assert "next" in relations
    assert "back" in relations


def test_cfg_builder_rejects_non_python_files_without_fallback(tmp_path: Path) -> None:
    target = tmp_path / "module.cc"
    target.write_text(
        "int run(int x) {\n" "  if (x > 0) return x;\n" "  return 0;\n" "}\n",
        encoding="utf-8",
    )

    trace = TracePerception(repo_path=str(tmp_path))
    cfg = trace.build_cfg("module.cc")

    assert cfg["status"] == "unsupported_language"
    assert cfg["language"] == "cpp"
    assert cfg["node_count"] == 0
