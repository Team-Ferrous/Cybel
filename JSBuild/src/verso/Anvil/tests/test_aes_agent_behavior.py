from pathlib import Path

from core.aes.governance import GovernanceEngine
from core.aes.red_team_protocol import RedTeamProtocol
from core.prompts.aes_prompt_builder import AESPromptBuilder
from core.thinking import EnhancedThinkingSystem


def test_subagent_prompt_includes_high_aal_artifact_contract(tmp_path: Path) -> None:
    trainer = tmp_path / "trainer.py"
    trainer.write_text(
        "import torch\nloss.backward()\noptimizer.step()\n",
        encoding="utf-8",
    )

    prompt, contract = AESPromptBuilder().build_subagent_prompt(
        role="implementer",
        task_files=[str(trainer)],
        task_text="Implement a PyTorch training loop",
    )

    assert contract["AAL_CLASSIFICATION"] == "AAL-1"
    assert "AES-ML-1" in contract["APPLICABLE_RULE_IDS"]
    assert "traceability_record" in contract["REQUIRED_ARTIFACTS"]
    assert "High-AAL Checklist:" in prompt


def test_thinking_prompt_contains_compliance_block_and_aal() -> None:
    system = EnhancedThinkingSystem(coconut_enabled=False)
    system.set_compliance_context(trace_id="trace-7", evidence_bundle_id="bundle-7")

    prompt = system.generate_thinking_prompt(
        task="modify SIMD kernel",
        files=["core/simd_kernel.cc"],
    )

    assert '<thinking type="compliance">' in prompt
    assert "AAL Classification:" in prompt
    assert "trace_id=trace-7" in prompt


def test_governance_engine_escalates_high_aal_fallback_without_waiver() -> None:
    result = GovernanceEngine().check_action(
        action="fallback to warn-only mode",
        aal="AAL-1",
        action_type="code_modification",
        waiver_ids=[],
    )

    assert result.allowed is False
    assert result.escalation_required is True
    assert "without waiver" in result.reason.lower()


def test_red_team_protocol_requires_artifacts_for_high_aal() -> None:
    validation = RedTeamProtocol().validate(
        artifacts={},
        aal="AAL-1",
        red_team_required=False,
    )

    assert validation.required is True
    assert validation.passed is False
    assert "fmea.json" in validation.missing_artifacts
