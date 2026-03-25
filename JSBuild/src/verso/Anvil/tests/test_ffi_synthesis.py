from __future__ import annotations

from saguaro.synthesis.adapter_generator import AdapterGenerator


def test_adapter_generator_builds_python_and_cpp_boundary_adapters() -> None:
    generator = AdapterGenerator()
    python_plan = generator.generate_python_capability_adapter()
    cpp_plan = generator.generate_cpp_bridge_adapter()

    assert "build_runtime_capability_ledger" in python_plan.code
    assert "saguaro_native_version" in cpp_plan.code
    assert python_plan.bridge_validation["bridge_validation_pass_rate"] == 1.0
    assert cpp_plan.bridge_validation["bridge_validation_pass_rate"] == 1.0

