from types import SimpleNamespace

from core.aes import ActionEscalationEngine, GovernanceEngine
from core.chat_loop_enterprise import EnterpriseChatLoop
from core.error_recovery import ErrorRecoveryManager, RecoveryStrategy
from core.unified_chat_loop import UnifiedChatLoop
from saguaro.sentinel.policy import PolicyManager


def test_error_recovery_blocks_masking_strategy_for_high_aal_without_waiver():
    manager = ErrorRecoveryManager()
    result = manager._execute_recovery(
        RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
        RuntimeError("timeout"),
        {"aal": "AAL-0", "model": "granite-3.1-dense:8b"},
    )
    assert result.recovered is False
    assert result.context_modifications["escalate_supervised"] is True
    assert result.context_modifications["aes_compliant"] is False


def test_error_recovery_allows_masking_strategy_with_approved_waiver():
    manager = ErrorRecoveryManager()
    result = manager._execute_recovery(
        RecoveryStrategy.SWITCH_TO_FASTER_MODEL,
        RuntimeError("timeout"),
        {
            "aal": "AAL-1",
            "waiver_ids": ["fallback-waiver"],
            "model": "granite-3.1-dense:8b",
        },
    )
    assert result.recovered is True
    assert result.context_modifications["aes_compliant"] is True


def test_policy_runtime_decision_escalates_guarded_findings_for_high_aal(tmp_path):
    manager = PolicyManager(str(tmp_path))
    violations = manager.evaluate(
        [{"severity": "P2", "aal": "AAL-0", "rule_id": "AES-TEST-1"}]
    )
    decision = manager.runtime_decision(violations, aal="AAL-0")
    assert decision["should_fail"] is True
    assert decision["decision"] == "escalate"


def test_governance_engine_blocks_immutable_action():
    engine = GovernanceEngine()
    result = engine.check_action("disable saguaro verify", "AAL-2")
    assert result.allowed is False
    assert result.tier.value == "immutable"


def test_governance_engine_blocks_qsg_runtime_contract_mismatch():
    engine = GovernanceEngine()
    result = engine.check_action(
        "write runtime-sensitive code",
        "AAL-2",
        action_type="code_modification",
        qsg_runtime_status={
            "strict_native_qsg": True,
            "native_backend_abi_match": False,
            "repo_coupled_runtime": {"delta_authority": "tracker"},
        },
    )

    assert result.allowed is False
    assert result.escalation_required is True
    assert "qsg runtime contract violation" in result.reason.lower()


def test_action_escalation_blocks_high_aal_irreversible_without_signoff():
    engine = ActionEscalationEngine()
    result = engine.evaluate(
        tool_name="delete_file",
        arguments={"path": "critical.py"},
        aal="AAL-0",
        review_signoff_token=None,
        rollback_plan_artifact=None,
    )
    assert result.allowed is False
    assert result.requires_supervision is True


def test_enterprise_loop_blocks_high_aal_action_without_review_signoff(mocker):
    agent = SimpleNamespace(
        console=SimpleNamespace(print=lambda *args, **kwargs: None),
        brain=SimpleNamespace(stream_chat=lambda *args, **kwargs: iter(["ok"])),
        history=SimpleNamespace(add_message=lambda *args, **kwargs: None),
        registry=None,
        semantic_engine=None,
        approval_manager=SimpleNamespace(can_execute=lambda *args, **kwargs: True),
        name="test-agent",
    )
    loop = EnterpriseChatLoop(agent)
    mocker.patch.object(loop, "_gather_evidence", return_value={"codebase_files": []})
    mocker.patch.object(loop, "_classify_runtime_aal", return_value="AAL-0")
    mocker.patch.object(loop, "_generate_action_plan", return_value="plan")

    response = loop._execute_action("delete critical file")
    assert "AES ACTION BLOCK (AAL-0)" in response


def test_unified_loop_pre_action_checkpoint_blocks_high_aal_without_supervision():
    loop = UnifiedChatLoop.__new__(UnifiedChatLoop)
    loop.runtime_aal = "AAL-0"
    loop.current_compliance_context = {"waiver_ids": []}
    loop.governance_engine = GovernanceEngine()
    loop.action_escalation_engine = ActionEscalationEngine()
    result = UnifiedChatLoop._pre_action_tool_checkpoint(
        loop,
        [{"tool": "write_file", "args": {"path": "x.py", "content": "print(1)"}}],
        "task",
    )
    assert result["allowed"] is False
