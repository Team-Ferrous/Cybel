from __future__ import annotations

from saguaro.synthesis.assembler import ComponentAssembler
from saguaro.synthesis.component_catalog import ComponentCatalog, ComponentDescriptor
from saguaro.synthesis.spec import SpecLowerer


def test_component_assembler_ranks_and_assembles_matching_components() -> None:
    spec = SpecLowerer().lower_objective(
        "Implement runtime adapter in generated/runtime_adapter.py"
    )
    catalog = ComponentCatalog(
        [
            ComponentDescriptor(
                qualified_name="build_runtime_capability_ledger",
                file_path="core/native/runtime_telemetry.py",
                component_type="function",
                language="python",
                terms=["runtime", "capability", "adapter"],
                contracts=["returns ledger"],
            ),
            ComponentDescriptor(
                qualified_name="unrelated_helper",
                file_path="misc/helper.py",
                component_type="function",
                language="python",
                terms=["misc"],
            ),
        ]
    )

    plan = ComponentAssembler().assemble(spec, catalog)

    assert plan.selected_components
    assert plan.selected_components[0].component.qualified_name == "build_runtime_capability_ledger"
    assert plan.reuse_ratio >= 1.0

