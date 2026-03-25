from pathlib import Path
from unittest.mock import MagicMock

from core.aes import AESRuleRegistry
from core.multi_file_refactor import MultiFileRefactorer, RefactorPlan
from core.parallel_executor import ParallelToolExecutor
from core.prompts.system_prompt_builder import SystemPromptBuilder
from core.smart_editor import SmartFileEditor
from core.thinking import EnhancedThinkingSystem
from core.agents.subagent_quality_gate import SubagentQualityGate


class StubBrain:
    def embeddings(self, text):
        return [1.0, 0.0, 0.0]


def test_rule_registry_resolves_extended_domain_checks():
    registry = AESRuleRegistry()
    registry.load("standards/AES_RULES.json")

    for rule_id in ("AES-QC-2", "AES-HPC-2", "AES-PHYS-1", "AES-ML-2"):
        assert callable(registry.get_check_function(rule_id))


def test_prompt_builder_includes_rule_ids_and_required_artifacts(tmp_path: Path):
    target = tmp_path / "trainer.py"
    target.write_text("import torch\noptimizer.step()\n", encoding="utf-8")

    prompt = SystemPromptBuilder().build(files=[str(target)])
    assert "Applicable Rules:" in prompt
    assert "AES-ML-1" in prompt
    assert "Required Artifacts:" in prompt


def test_thinking_system_persists_compliance_context(tmp_path: Path):
    system = EnhancedThinkingSystem(coconut_enabled=False)
    system.set_compliance_context(
        trace_id="trace-1", evidence_bundle_id="bundle-1", waiver_id="waiver-1"
    )
    chain = system.start_chain(task_id="task-1")
    path = tmp_path / "thinking.json"
    chain.save(str(path))

    loaded = chain.load(str(path))
    assert loaded.compliance_context["trace_id"] == "trace-1"
    assert loaded.compliance_context["evidence_bundle_id"] == "bundle-1"
    assert loaded.compliance_context["waiver_id"] == "waiver-1"
    assert loaded.compliance_context["waiver_ids"] == ["waiver-1"]
    assert loaded.compliance_context["red_team_required"] is False


def test_thinking_system_autofills_missing_compliance_context():
    system = EnhancedThinkingSystem(coconut_enabled=False)
    chain = system.start_chain(task_id="task-1", compliance_context={})
    assert chain.compliance_context["trace_id"] == "trace::task-1"
    assert chain.compliance_context["evidence_bundle_id"] == "evidence::trace::task-1"
    assert chain.compliance_context["red_team_required"] is False
    assert chain.compliance_context["waiver_ids"] == []


def test_quality_gate_requires_compliance_ids(tmp_path: Path):
    target = tmp_path / "sample.py"
    target.write_text("def foo():\n    return 1\n", encoding="utf-8")

    gate = SubagentQualityGate(repo_root=str(tmp_path), brain=StubBrain())
    payload = {
        "subagent_analysis": "sample.py:L1\n```python\ndef foo():\n    return 1\n```",
        "codebase_files": ["sample.py"],
        "compliance": {"trace_id": "trace-only"},
    }
    result = gate.evaluate(payload, "query", complexity_score=3)
    assert result["accepted"] is False
    assert "evidence_bundle_id" in result["compliance"]["missing"]


def test_smart_editor_restore_uses_delegated_executor():
    calls = []
    editor = SmartFileEditor(
        registry=MagicMock(),
        console=MagicMock(),
        tool_executor=lambda name, arguments: calls.append((name, arguments)) or "ok",
    )
    editor.backup_cache["sample.py"] = "print('ok')\n"
    editor._restore_backup("sample.py")
    assert calls == [
        (
            "write_file",
            {"file_path": "sample.py", "content": "print('ok')\n"},
        )
    ]


def test_multi_file_refactor_create_uses_delegated_executor():
    calls = []
    refactorer = MultiFileRefactorer(
        registry=MagicMock(),
        saguaro_tools=MagicMock(),
        smart_editor=MagicMock(),
        console=MagicMock(),
        tool_executor=lambda name, arguments: calls.append((name, arguments)) or "ok",
    )
    plan = RefactorPlan(
        operation="extract_module",
        target="source.py",
        new_value="new_module.py",
        files_to_modify=["new_module.py"],
        modifications={"new_module.py": [{"action": "create", "content": "pass\n"}]},
        risk_level="medium",
        estimated_impact=1,
    )
    refactorer._apply_refactor_plan(plan)
    assert calls == [
        (
            "write_file",
            {"file_path": "new_module.py", "content": "pass\n"},
        )
    ]


def test_parallel_tool_executor_uses_delegated_executor():
    calls = []
    executor = ParallelToolExecutor(
        registry=MagicMock(),
        semantic_engine=MagicMock(),
        console=MagicMock(),
        tool_executor=lambda name, arguments: calls.append((name, arguments)) or "ok",
    )
    result = executor._execute_single_tool(
        {"tool": "read_file", "args": {"path": "sample.py"}}
    )
    assert calls
    assert calls[0][0] == "read_file"
    assert calls[0][1]["path"] == "sample.py"
    assert result["success"] is True
