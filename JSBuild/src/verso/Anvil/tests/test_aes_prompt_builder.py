import json
from pathlib import Path

from core.prompts.system_prompt_builder import SystemPromptBuilder


def test_prompt_builder_always_includes_condensed_rules():
    prompt = SystemPromptBuilder().build(task_text="implement config validation")
    assert "AES Condensed" in prompt
    assert "AAL Context" in prompt
    assert "Applicable Rules:" in prompt
    assert "AES-ARCH-1" in prompt


def test_prompt_builder_includes_domain_rules(tmp_path: Path):
    target = tmp_path / "trainer.py"
    target.write_text("import torch\noptimizer.step()\n", encoding="utf-8")

    prompt = SystemPromptBuilder().build(files=[str(target)])
    assert "Active Domains: ml" in prompt
    assert "check gradient finiteness before optimizer update" in prompt


def test_prompt_builder_includes_visuals_rules_for_visuals_files(tmp_path: Path):
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

    prompt = SystemPromptBuilder().build(files=[str(v1), str(v2)])

    assert "Applicable Rules:" in prompt
    assert "AES-VIS-1" in prompt
    assert "AES-VIS-2" in prompt
