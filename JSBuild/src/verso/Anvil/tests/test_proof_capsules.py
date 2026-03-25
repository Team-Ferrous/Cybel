from __future__ import annotations

from saguaro.synthesis.proof_capsule import SynthesisProofCapsule
from saguaro.synthesis.replay_tape import SynthesisReplayTape
from saguaro.synthesis.spec import SpecLowerer


def test_proof_capsule_alias_exposes_capsule_builder() -> None:
    spec = SpecLowerer().lower_objective("Implement clamp helper in generated/clamp.py").to_dict()
    tape = SynthesisReplayTape.from_spec(spec, objective=spec["objective"])
    capsule = SynthesisProofCapsule.from_spec(spec, replay_tape=tape.to_dict())

    assert capsule.spec_digest
    assert capsule.metadata["witness_graph_size"] >= 1
