import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.hooks.aes_pre_verify import AESPreVerifyHook
from core.serialization import redact_secret_material, serialize_tool_provenance
from saguaro.sentinel.engines.aes import AESEngine
from saguaro.sentinel.policy import PolicyManager
from saguaro.sentinel.rules import RuleLoader
from saguaro.sentinel.verifier import SentinelVerifier
from tools.registry import ToolRegistry


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_rule_loader_prefers_structured_rule_definitions(tmp_path: Path) -> None:
    _write(
        tmp_path / "standards" / "AES_RULES.json",
        json.dumps(
            [
                {
                    "id": "AES-CR-2",
                    "section": "4",
                    "text": "structured message",
                    "severity": "AAL-0",
                    "engine": "native",
                    "auto_fixable": False,
                    "domain": ["universal"],
                    "language": ["python"],
                    "pattern": "except:\\s*pass",
                }
            ]
        ),
    )
    _write(
        tmp_path / ".saguaro.rules",
        """
rules:
  - id: AES-CR-2
    pattern: 'except:\\s*pass'
    severity: P0
    message: "legacy message"
  - id: AES-LEGACY-ONLY
    pattern: 'return\\s+None'
    severity: P2
    message: "legacy only"
""",
    )

    loaded = RuleLoader.load(str(tmp_path))
    by_id = {rule.id: rule for rule in loaded}

    assert by_id["AES-CR-2"].message == "structured message"
    assert "AES-LEGACY-ONLY" not in by_id


def test_rule_loader_warns_when_only_legacy_rules_exist(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write(
        tmp_path / ".saguaro.rules",
        """
rules:
  - id: AES-LEGACY
    pattern: 'except:\\s*pass'
    severity: P0
    message: "legacy"
""",
    )

    RuleLoader.load(str(tmp_path))
    assert any("ignored in strict mode" in msg for msg in caplog.messages)


def test_policy_normalization_enforces_blocking_for_strict_findings(tmp_path: Path) -> None:
    manager = PolicyManager(str(tmp_path))
    violations = manager.evaluate(
        [
            {
                "file": "core/high_risk.py",
                "line": 1,
                "rule_id": "AES-CR-2",
                "message": "bare except",
                "severity": "error",
                "aal": "AAL-0",
            }
        ]
    )

    assert violations[0]["severity"] == "P1"
    assert violations[0]["closure_level"] == "blocking"
    assert manager.should_fail(violations) is True


def test_verifier_orders_engines_deterministically() -> None:
    ordered = SentinelVerifier._order_engines(["aes", "semantic", "ruff", "native"])
    assert ordered[:4] == ["native", "ruff", "semantic", "aes"]


def test_aes_engine_blocks_missing_artifacts_for_aal0(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "module.py", "# _mm256_add_ps marker for AAL-0\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy({"block_on_missing_artifacts": True, "verify_context": {}})
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-TR-1" in rule_ids
    assert "AES-TR-2" in rule_ids


def test_aes_engine_runs_prompt_policy_subcheck(tmp_path: Path) -> None:
    _write(tmp_path / "standards" / "AES_RULES.json", "[]")
    _write(tmp_path / "module.py", "def noop():\n    return 1\n")
    _write(
        tmp_path / "scripts" / "validate_prompt_contracts.py",
        (
            "import json\n"
            "print(json.dumps({'ok': False, 'errors': "
            "[{'file': 'prompts/GEMINI.md', 'message': 'contract missing'}]}))\n"
            "raise SystemExit(1)\n"
        ),
    )

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    assert any(item["rule_id"] == "AES-PRM-1" for item in violations)


def test_write_tools_require_execute_tool_path(tmp_path: Path) -> None:
    agent = SimpleNamespace(audit_db=None, session_id="s1", _active_tool_execution=False)
    registry = ToolRegistry(root_dir=str(tmp_path), agent=agent)
    result = registry.dispatch("write_file", {"path": "blocked.txt", "content": "x"})
    assert "must execute via BaseAgent._execute_tool" in result


def test_aes_pre_verify_hook_blocks_aal0_write(tmp_path: Path) -> None:
    _write(
        tmp_path / "standards" / "AES_RULES.json",
        json.dumps(
            [
                {
                    "id": "AES-CR-2",
                    "section": "4",
                    "text": "No bare except",
                    "severity": "AAL-0",
                    "engine": "agent",
                    "auto_fixable": False,
                    "domain": ["universal"],
                    "language": ["python"],
                    "check_function": "core.aes.checks.universal_checks.check_no_bare_except",
                }
            ]
        ),
    )
    _write(tmp_path / "critical.py", "# _mm256_add_ps\ntry:\n    pass\nexcept:\n    pass\n")

    hook = AESPreVerifyHook(repo_path=str(tmp_path))
    context = {
        "tool_name": "write_file",
        "tool_args": {"path": "critical.py"},
    }
    result = hook.execute(context)

    assert result["post_write_verification"]["blocked"] is True
    assert result["write_blocked"] is True


def test_direct_write_without_agent_runs_post_write_gate(tmp_path: Path) -> None:
    _write(
        tmp_path / "standards" / "AES_RULES.json",
        json.dumps(
            [
                {
                    "id": "AES-CR-2",
                    "section": "4",
                    "text": "No bare except",
                    "severity": "AAL-0",
                    "engine": "agent",
                    "auto_fixable": False,
                    "domain": ["universal"],
                    "language": ["python"],
                    "check_function": "core.aes.checks.universal_checks.check_no_bare_except",
                }
            ]
        ),
    )

    registry = ToolRegistry(root_dir=str(tmp_path), agent=None)

    async def _execute_with_budget(_tool_name, func, params):
        return func(**params)

    registry.resource_manager.execute_with_budget = _execute_with_budget

    result = registry.dispatch(
        "write_file",
        {
            "path": "critical.py",
            "content": "# _mm256_add_ps\\ntry:\\n    pass\\nexcept:\\n    pass\\n",
        },
    )
    assert "AES GATE: direct write failed post-write verification" in result


def test_serialization_helpers_redact_and_tag_provenance() -> None:
    redacted = redact_secret_material("api_key=abc token=xyz")
    assert "abc" not in redacted
    assert "xyz" not in redacted

    provenance = serialize_tool_provenance("write_file", {"path": "a.py"}, "trace-1")
    assert provenance["tool_name"] == "write_file"
    assert provenance["trace_id"] == "trace-1"
    assert provenance["args_hash"]
