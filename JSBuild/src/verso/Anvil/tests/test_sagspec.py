from __future__ import annotations

from saguaro.synthesis.spec import SagSpec, SpecConstraint, SpecVerification


def test_sagspec_roundtrips_and_emits_stable_digest() -> None:
    spec = SagSpec(
        objective="Implement clamp helper",
        title="Clamp Helper",
        stage="bounded_function",
        language="python",
        target_files=["generated/clamp.py"],
        outputs={"clamp": "float"},
        inputs={"value": "float", "lower": "float", "upper": "float"},
        constraints=[SpecConstraint(kind="range_safety", expression="lower <= output <= upper")],
        verification=SpecVerification(commands=["pytest tests/test_clamp.py"]),
    )

    payload = spec.to_dict()

    assert payload["target_files"] == ["generated/clamp.py"]
    assert len(spec.stable_digest()) == 64
    assert spec.proposed_changes()[0]["file_path"] == "generated/clamp.py"

