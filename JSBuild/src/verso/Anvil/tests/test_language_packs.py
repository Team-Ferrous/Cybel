from __future__ import annotations

from saguaro.language_packs.cpp_pack import build_cpp_pack
from saguaro.language_packs.python_pack import build_python_pack


def test_language_packs_cover_function_level_constructs() -> None:
    python_pack = build_python_pack()
    cpp_pack = build_cpp_pack()

    python_report = python_pack.coverage_report(["function_def", "dict", "import"])
    cpp_report = cpp_pack.coverage_report(["function_definition", "std::vector", "include_directive"])

    assert python_report["language_pack_coverage_pct"] == 100.0
    assert cpp_report["language_pack_coverage_pct"] == 100.0
    assert "dynamic_exec" in python_pack.unsupported_constructs

