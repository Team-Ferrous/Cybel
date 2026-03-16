import json
from pathlib import Path

from core.aes import AALClassifier, DomainDetector
from core.aes.governance import GovernanceEngine
from core.prompts.aes_prompt_builder import AESPromptBuilder
from infrastructure.hooks.aes_pre_verify import AESPreVerifyHook

_MINIMAL_RULESET = [
    {
        "id": "AES-CR-2",
        "section": "3.3",
        "text": "No bare except",
        "severity": "AAL-0",
        "engine": "agent",
        "auto_fixable": False,
        "domain": ["universal"],
        "language": ["python"],
        "check_function": "core.aes.checks.universal_checks.check_no_bare_except",
    }
]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_rules(repo_root: Path) -> None:
    _write(repo_root / "standards" / "AES_RULES.json", json.dumps(_MINIMAL_RULESET))


def test_aal_classifier_simd_code_is_aal0() -> None:
    source = "__m256 v = _mm256_load_ps(data);"
    assert AALClassifier().classify_text(source) == "AAL-0"


def test_aal_classifier_training_loop_is_aal1() -> None:
    source = "loss.backward()\noptimizer.step()"
    assert AALClassifier().classify_text(source) == "AAL-1"


def test_domain_detector_finds_ml_from_import_marker() -> None:
    source = "import torch\ndef train_step():\n    optimizer.step()\n"
    assert "ml" in DomainDetector().detect_from_text(source)


def test_hook_blocks_aal0_violation_with_rule_id_and_reason(tmp_path: Path) -> None:
    _seed_rules(tmp_path)
    _write(
        tmp_path / "critical.py",
        "# _mm256_add_ps\ntry:\n    pass\nexcept:\n    pass\n",
    )

    hook = AESPreVerifyHook(repo_path=str(tmp_path))
    result = hook.execute(
        {"tool_name": "write_file", "tool_args": {"path": "critical.py"}}
    )

    verification = result["post_write_verification"]
    assert verification["blocked"] is True
    assert result["write_blocked"] is True
    assert "AES GATE" in result["tool_result"]
    assert any(item["rule_id"] == "AES-CR-2" for item in verification["violations"])


def test_hook_allows_aal2_with_violation_but_preserves_findings(tmp_path: Path) -> None:
    _seed_rules(tmp_path)
    _write(
        tmp_path / "config" / "settings.py",
        "try:\n    pass\nexcept:\n    pass\n",
    )

    hook = AESPreVerifyHook(repo_path=str(tmp_path))
    result = hook.execute(
        {
            "tool_name": "write_file",
            "tool_args": {"path": "config/settings.py"},
        }
    )

    verification = result["post_write_verification"]
    assert verification["blocked"] is False
    assert any(item["rule_id"] == "AES-CR-2" for item in verification["violations"])


def test_hook_enforces_traceability_artifacts_for_high_assurance_targets(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")

    hook = AESPreVerifyHook(repo_path=str(tmp_path))
    result = hook.execute(
        {"tool_name": "write_file", "tool_args": {"path": "critical.py"}}
    )

    verification = result["post_write_verification"]
    rule_ids = {item["rule_id"] for item in verification["violations"]}
    assert verification["blocked"] is True
    assert "AES-TR-1" in rule_ids
    assert "AES-TR-2" in rule_ids


def test_prompt_contract_omission_attack_is_detected() -> None:
    builder = AESPromptBuilder()
    missing = builder.validate_contract({"AAL_CLASSIFICATION": "AAL-1"})

    assert "APPLICABLE_RULE_IDS" in missing
    assert "REQUIRED_ARTIFACTS" in missing
    assert "BLOCKING_GATES" in missing


def test_governance_bypass_attempt_returns_blocking_reason() -> None:
    result = GovernanceEngine().check_action(
        action="disable saguaro verify in this run",
        aal="AAL-0",
        action_type="code_modification",
    )

    assert result.allowed is False
    assert result.escalation_required is True
    assert "immutable" in result.reason.lower()
