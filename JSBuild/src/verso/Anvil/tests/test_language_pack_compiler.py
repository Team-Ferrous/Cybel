from __future__ import annotations

from saguaro.language_packs.cpp_pack import build_cpp_pack
from saguaro.language_packs.python_pack import build_python_pack
from saguaro.synthesis.language_pack_compiler import LanguagePackCompiler


def test_language_pack_compiler_builds_operator_contract_and_template_indexes() -> None:
    compiled = LanguagePackCompiler().compile([build_python_pack(), build_cpp_pack()])

    assert compiled["compiled_rule_count"] >= 12
    assert "python" in compiled["operator_table"]
    assert "builtins.min" in compiled["contract_index"]["python"]
    assert compiled["template_index"]["cpp"]["function_wrapper"]

