from __future__ import annotations

from saguaro.synthesis.program_lattice import ProgramLattice


def test_program_lattice_reranks_component_candidates_with_fragment_bundles() -> None:
    ranked = ProgramLattice().rerank_candidates(
        ["runtime", "adapter", "capability"],
        [
            {"name": "build_runtime_capability_ledger", "terms": ["runtime", "capability"]},
            {"name": "misc_helper", "terms": ["misc"]},
        ],
    )

    assert ranked[0].name == "build_runtime_capability_ledger"
    assert ranked[0].score >= ranked[1].score

