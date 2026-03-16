import json
from pathlib import Path

from core.aes import AALClassifier, SecurityVerificationLevel
from core.aes.security_verification import check_svl_compliance, map_aal_to_svl
from saguaro.sentinel.engines.aes import AESEngine


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_rules(repo_root: Path, rules: list[dict]) -> None:
    _write(repo_root / "standards" / "AES_RULES.json", json.dumps(rules))


def test_aal_to_svl_mapping_is_deterministic() -> None:
    assert map_aal_to_svl("AAL-0") is SecurityVerificationLevel.SVL_3
    assert map_aal_to_svl("AAL-1") is SecurityVerificationLevel.SVL_2
    assert map_aal_to_svl("AAL-2") is SecurityVerificationLevel.SVL_1
    assert map_aal_to_svl("AAL-3") is SecurityVerificationLevel.SVL_0


def test_classifier_returns_aal_and_svl() -> None:
    aal, svl = AALClassifier().classify_text_with_security_level("loss.backward()\noptimizer.step()")
    assert aal == "AAL-1"
    assert svl is SecurityVerificationLevel.SVL_2


def test_svl_check_flags_cwe_and_threat_model_requirements() -> None:
    source = (
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "def endpoint(payload):\n"
        "    return eval(payload)\n"
    )
    violations = check_svl_compliance(source, "api.py")
    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-SEC-2" in rule_ids
    assert "AES-SEC-3" in rule_ids


def test_svl_check_flags_dependency_without_hash() -> None:
    source = "requests==2.32.0\n"
    violations = check_svl_compliance(source, "requirements.txt")
    assert any(item["rule_id"] == "AES-SEC-3" for item in violations)


def test_svl_check_ignores_regex_strings_without_runtime_calls() -> None:
    source = """
DANGEROUS = [
    r"\\beval\\(",
    r"\\bexec\\(",
]
"""
    assert check_svl_compliance(source, "security_verification.py") == []


def test_aes_engine_emits_svl_metadata_for_security_rules(tmp_path: Path) -> None:
    _seed_rules(
        tmp_path,
        [
            {
                "id": "AES-SEC-2",
                "section": "6",
                "text": "SVL composite check",
                "severity": "AAL-1",
                "engine": "aes",
                "auto_fixable": False,
                "domain": ["universal"],
                "language": ["python"],
                "check_function": "core.aes.security_verification.check_svl_compliance",
                "cwe": ["CWE-94"],
                "svl_min": "SVL-2",
            }
        ],
    )
    _write(
        tmp_path / "api.py",
        "from fastapi import FastAPI\napp = FastAPI()\ndef endpoint(payload):\n    return eval(payload)\n",
    )
    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path / "api.py"))

    assert violations
    assert all(item.get("svl") for item in violations)
    assert any(item.get("rule_id") == "AES-SEC-2" for item in violations)
