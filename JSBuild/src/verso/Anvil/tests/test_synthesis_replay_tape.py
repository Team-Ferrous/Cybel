from __future__ import annotations

from saguaro.synthesis.replay_tape import SynthesisProofCapsule, SynthesisReplayTape
from saguaro.synthesis.spec import SpecLowerer


def test_synthesis_replay_tape_and_proof_capsule_capture_spec_artifacts() -> None:
    spec = SpecLowerer().lower_objective("Implement clamp helper in generated/clamp.py").to_dict()
    lint = {"is_valid": True, "telemetry": {"constraint_count": 2}}

    tape = SynthesisReplayTape.from_spec(spec, lint_payload=lint, objective=spec["objective"])
    capsule = SynthesisProofCapsule.from_spec(spec, lint_payload=lint, replay_tape=tape.to_dict())

    assert tape.spec_digest
    assert capsule.capsule_id.startswith("proof:")
    assert capsule.metadata["proof_capsule_emission_rate"] == 1.0

