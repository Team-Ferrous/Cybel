from __future__ import annotations

from saguaro.math import MathEngine


def test_math_engine_emits_loop_and_access_metadata(tmp_path) -> None:
    target = tmp_path / "core" / "simd" / "kernel.cpp"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "void kernel(float* input, float* weights, float& sum, int n) {\n"
        "  for (int i = 0; i < n; ++i) {\n"
        "    sum += input[i] * weights[i + 1];\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )

    parsed = MathEngine(str(tmp_path)).parse("core/simd/kernel.cpp")
    record = next(
        item
        for item in parsed["records"]
        if item["expression"] == "sum += input[i] * weights[i + 1]"
    )

    assert record["loop_context"]["loop_kind"] == "for"
    assert "i" in record["loop_context"]["loop_variables"]
    assert record["loop_context"]["reduction"] is True
    signatures = {item["base_symbol"]: item for item in record["access_signatures"]}
    assert signatures["input"]["stride_class"] == "contiguous"
    assert signatures["weights"]["stride_class"] == "contiguous_offset"
    layouts = {item["symbol"]: item for item in record["layout_states"]}
    assert layouts["input"]["layout"] == "packed_contiguous"
