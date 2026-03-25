from __future__ import annotations

from saguaro.synthesis.ast_builder import ASTBuilder, SagParameter


def test_ast_builder_emits_python_and_cpp_function_wrappers() -> None:
    builder = ASTBuilder()
    python_node = builder.build_function(
        language="python",
        name="clamp",
        return_type="float",
        parameters=[SagParameter("value", "float")],
        body_lines=["return value"],
    )
    cpp_node = builder.build_function(
        language="cpp",
        name="clamp",
        return_type="double",
        parameters=[SagParameter("value", "double")],
        body_lines=["return value;"],
        imports_or_includes=["algorithm"],
    )

    python_code = builder.emit(python_node)
    cpp_code = builder.emit(cpp_node)

    assert "def clamp" in python_code
    assert "double clamp" in cpp_code
    assert builder.roundtrip_report(python_node)["syntax_ok"] is True
    assert builder.roundtrip_report(cpp_node)["syntax_ok"] is True

