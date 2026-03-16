from __future__ import annotations

from saguaro.synthesis.patch_inductor import PatchInductor


def test_patch_inductor_replays_structural_update() -> None:
    before = "return payload\n"
    after = "return dict(payload)\n"
    inductor = PatchInductor()
    rule = inductor.induce_rule(before, after, rule_name="wrap_payload")

    replay = inductor.replay(rule, before)

    assert replay["applied"] is True
    assert replay["result"] == after
    assert inductor.validate_replay(rule, before, after) is True

