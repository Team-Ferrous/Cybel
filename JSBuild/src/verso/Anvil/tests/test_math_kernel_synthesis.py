from __future__ import annotations

from saguaro.synthesis.math_kernel_ir import MathKernelCompiler


def test_math_kernel_compiler_lowers_tiny_expression_dsl_to_python_and_cpp() -> None:
    compiler = MathKernelCompiler()
    ir = compiler.parse("max(lower, min(upper, value))")

    python_code = compiler.lower(ir, language="python", function_name="clamp")
    cpp_code = compiler.lower(ir, language="cpp", function_name="clamp")

    assert "def clamp" in python_code
    assert "double clamp" in cpp_code
    assert ir.variables == ["lower", "upper", "value"]

