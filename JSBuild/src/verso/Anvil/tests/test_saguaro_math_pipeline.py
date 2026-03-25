from __future__ import annotations

from saguaro.math import MathEngine


def test_math_pipeline_attaches_build_target_provenance(tmp_path) -> None:
    cmake = tmp_path / "core" / "native" / "CMakeLists.txt"
    source = tmp_path / "core" / "native" / "kernels" / "vec_kernel.cpp"
    cmake.parent.mkdir(parents=True, exist_ok=True)
    source.parent.mkdir(parents=True, exist_ok=True)
    cmake.write_text(
        "add_library(anvil_native_ops\n"
        "  kernels/vec_kernel.cpp\n"
        ")\n",
        encoding="utf-8",
    )
    source.write_text(
        "void kernel(float* x, float* y, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    y[i] = x[i] * 2.0f;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("core/native/kernels/vec_kernel.cpp")
    record = next(item for item in parsed["records"] if item["source_kind"] == "code_expression")

    assert "anvil_native_ops" in record["provenance"]["build_targets"]
    assert record["provenance"]["execution_domain"] == "native_kernel"
