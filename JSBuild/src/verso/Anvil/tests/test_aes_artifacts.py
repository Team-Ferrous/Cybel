import json
import re
from pathlib import Path

import yaml

REQUIRED_RULE_FIELDS = {
    "id",
    "section",
    "text",
    "severity",
    "engine",
    "auto_fixable",
    "domain",
    "language",
}


def test_rule_registry_artifact_shape():
    rules = json.loads(Path("standards/AES_RULES.json").read_text(encoding="utf-8"))
    assert rules
    for rule in rules:
        assert REQUIRED_RULE_FIELDS.issubset(rule)


def test_schema_required_fields_are_not_optional():
    traceability = json.loads(
        Path("standards/schemas/traceability.schema.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        Path("standards/schemas/evidence_bundle.schema.json").read_text(
            encoding="utf-8"
        )
    )
    waiver = json.loads(
        Path("standards/schemas/waiver.schema.json").read_text(encoding="utf-8")
    )

    assert "trace_id" in traceability["required"]
    assert "review_signoffs" in evidence["required"]
    assert "compensating_control" in waiver["required"]

    change_manifest = json.loads(
        Path("standards/schemas/change_manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    runtime_gate_report = json.loads(
        Path("standards/schemas/runtime_gate_report.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert "run_id" in change_manifest["required"]
    assert "required_runtime_gates" in change_manifest["required"]
    assert "results" in runtime_gate_report["required"]


def test_review_matrix_covers_all_aal_levels():
    matrix = yaml.safe_load(Path("standards/review_matrix.yaml").read_text())
    assert set(matrix["aal_levels"]) == {"AAL-0", "AAL-1", "AAL-2", "AAL-3"}


def test_prompt_to_rule_map_references_tests_and_files():
    mapping = yaml.safe_load(
        Path("standards/traceability/PROMPT_TO_RULE_MAP.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert mapping["prompt_to_rule_map"]
    for entry in mapping["prompt_to_rule_map"]:
        assert entry["implementation_files"]
        assert entry["test_artifacts"]


def test_prompt_to_rule_map_covers_condensed_and_domain_fragments():
    mapping = yaml.safe_load(
        Path("standards/traceability/PROMPT_TO_RULE_MAP.yaml").read_text(
            encoding="utf-8"
        )
    )
    sources = {entry["source"] for entry in mapping["prompt_to_rule_map"]}
    assert "standards/AES_CONDENSED.md" in sources
    assert "standards/domain_rules/ml.md" in sources
    assert "standards/domain_rules/quantum.md" in sources
    assert "standards/domain_rules/physics.md" in sources
    assert "standards/domain_rules/hpc.md" in sources


def test_aes_doc_to_rule_map_exists_and_marks_prompt_map_as_provenance():
    mapping = yaml.safe_load(
        Path("standards/traceability/AES_DOC_TO_RULE_MAP.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert mapping["aes_doc_to_rule_map"]
    prompt_entry = next(
        item
        for item in mapping["aes_doc_to_rule_map"]
        if item["source"] == "standards/traceability/PROMPT_TO_RULE_MAP.yaml"
    )
    assert prompt_entry["policy_role"] == "provenance_only"


def test_all_registry_rule_ids_are_documented_in_aes():
    rules = json.loads(Path("standards/AES_RULES.json").read_text(encoding="utf-8"))
    documented_ids = set(
        re.findall(r"AES-[A-Z]+-\d+", Path("standards/AES.md").read_text(encoding="utf-8"))
    )
    registry_ids = {rule["id"] for rule in rules}
    assert registry_ids.issubset(documented_ids)
