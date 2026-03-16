import json
import subprocess
import sys
from pathlib import Path

from core.prompts import PromptManager
from core.prompts.aes_prompt_builder import AESPromptBuilder
from core.thinking import EnhancedThinkingSystem, ThinkingType


def test_prompt_manager_emits_required_contract_keys(tmp_path: Path) -> None:
    target = tmp_path / "trainer.py"
    target.write_text("import torch\noptimizer.step()\n", encoding="utf-8")

    manager = PromptManager()
    prompt = manager.get_master_prompt(
        agent_name="Validator",
        context_type="general",
        task_text="verify prompt contract",
        workset_files=[str(target)],
    )

    assert "<AES_PROMPT_CONTRACT>" in prompt
    assert "AAL_CLASSIFICATION=" in prompt
    assert "APPLICABLE_RULE_IDS=" in prompt
    assert "REQUIRED_ARTIFACTS=" in prompt
    assert "BLOCKING_GATES=" in prompt


def test_prompt_manager_includes_control_plane_contract_fields(tmp_path: Path) -> None:
    target = tmp_path / "trainer.py"
    target.write_text("import torch\noptimizer.step()\n", encoding="utf-8")

    manager = PromptManager()
    prompt = manager.get_master_prompt(
        agent_name="Validator",
        context_type="general",
        task_text="verify prompt contract",
        workset_files=[str(target)],
        prompt_context={
            "trace_id": "trace-77",
            "graph_snapshot_id": "graph::2026-03-11T07:00:00Z",
            "policy_posture": "continue",
            "runtime_posture": "native_ready",
            "changed_files": ["trainer.py"],
            "toolchain_state_vector": [
                {
                    "profile": "node-web",
                    "qualification_state": "ready",
                    "state": "system",
                }
            ],
        },
    )

    assert "TRACE_ID=trace-77" in prompt
    assert "GRAPH_SNAPSHOT_ID=graph::2026-03-11T07:00:00Z" in prompt
    assert "POLICY_POSTURE=continue" in prompt
    assert "RUNTIME_POSTURE=native_ready" in prompt
    assert "TOOLCHAIN_STATE_VECTOR=" in prompt


def test_aes_prompt_builder_subagent_scopes_domain_rules(tmp_path: Path) -> None:
    target = tmp_path / "trainer.py"
    target.write_text("import torch\noptimizer.step()\n", encoding="utf-8")

    builder = AESPromptBuilder(repo_root=Path.cwd())
    payload, contract = builder.build_subagent_prompt(
        role="Verifier",
        task_files=[str(target)],
        task_text="validate ml training loop",
    )

    assert contract["PROMPT_ROLE"] == "subagent:Verifier"
    assert contract["AAL_CLASSIFICATION"] in {"AAL-0", "AAL-1", "AAL-2", "AAL-3"}
    assert "gradient" in payload.lower()


def test_thinking_start_chain_inserts_compliance_block() -> None:
    system = EnhancedThinkingSystem(coconut_enabled=False)
    chain = system.start_chain(
        task_id="task-1",
        compliance_context={
            "trace_id": "trace-1",
            "evidence_bundle_id": "bundle-1",
            "red_team_required": True,
            "waiver_ids": ["w-1"],
        },
        files=[],
    )

    assert chain.blocks
    assert chain.blocks[0].type == ThinkingType.COMPLIANCE
    assert "AAL Classification:" in chain.blocks[0].content
    assert chain.compliance_context["trace_id"] == "trace-1"
    assert chain.compliance_context["evidence_bundle_id"] == "bundle-1"
    assert chain.compliance_context["red_team_required"] is True
    assert chain.compliance_context["waiver_ids"] == ["w-1"]


def test_thinking_repairs_invalid_compliance_context() -> None:
    system = EnhancedThinkingSystem(coconut_enabled=False)
    chain = system.start_chain(
        task_id="repair-task",
        compliance_context={
            "trace_id": "  ",
            "evidence_bundle_id": "",
            "red_team_required": "yes",
            "waiver_ids": "waiver-9",
        },
    )

    assert chain.compliance_context["trace_id"].startswith("trace::repair-task")
    assert chain.compliance_context["evidence_bundle_id"].startswith("evidence::")
    assert chain.compliance_context["red_team_required"] is True
    assert chain.compliance_context["waiver_ids"] == ["waiver-9"]
    assert any(block.type == ThinkingType.CORRECTION for block in chain.blocks)


def test_prompt_contract_validator_script_reports_ok() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/validate_prompt_contracts.py",
            "--repo",
            ".",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    payload = json.loads(proc.stdout)
    assert proc.returncode == 0
    assert payload["ok"] is True
    assert payload["errors"] == []


def test_prompt_contract_includes_visuals_rule_ids_for_visuals_task(tmp_path: Path) -> None:
    v1 = tmp_path / "aes_visuals" / "v1" / "directives.json"
    v2 = tmp_path / "aes_visuals" / "v2" / "directives.json"
    v1.parent.mkdir(parents=True, exist_ok=True)
    v2.parent.mkdir(parents=True, exist_ok=True)
    v1.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "anvil_aes_visual_directives",
                "profile": "v1",
                "generated_on": "2026-03-03",
                "owner": "Anvil",
                "upstream_context": {
                    "aes_baseline_file": "standards/AES_SOURCE_BASELINES.md",
                    "intent": "test",
                },
                "directives": [
                    {
                        "directive_id": "AV1-001",
                        "title": "Title",
                        "rationale": "Rationale",
                        "enforcement_targets": ["prompt_generation"],
                        "implementation_patterns": ["Pattern"],
                        "verification_checks": ["Check"],
                        "source_refs": ["S01"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    v2.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact": "anvil_aes_visual_directives",
                "profile": "v2",
                "generated_on": "2026-03-03",
                "owner": "Anvil",
                "upstream_context": {
                    "aes_baseline_file": "standards/AES_SOURCE_BASELINES.md",
                    "intent": "test",
                },
                "directives": [
                    {
                        "directive_id": "AV2-001",
                        "title": "Title",
                        "rationale": "Rationale",
                        "enforcement_targets": ["prompt_generation"],
                        "implementation_patterns": ["Pattern"],
                        "verification_checks": ["Check"],
                        "source_refs": ["S01"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    builder = AESPromptBuilder(repo_root=Path.cwd())
    _payload, contract = builder.build_subagent_prompt(
        role="Verifier",
        task_files=[str(v1), str(v2)],
        task_text="validate visuals governance packs",
    )

    rule_ids = set(contract["APPLICABLE_RULE_IDS"])
    assert "AES-VIS-1" in rule_ids
    assert "AES-VIS-2" in rule_ids
