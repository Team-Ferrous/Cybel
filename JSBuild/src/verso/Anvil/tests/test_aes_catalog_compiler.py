import json
from pathlib import Path

from scripts.compile_aes_catalog import compile_catalog


def test_compiler_emits_extended_rule_fields() -> None:
    rules, obligations = compile_catalog(Path(".").resolve())

    trace_rule = next(rule for rule in rules if rule["id"] == "AES-TR-1")
    runtime_rule = next(rule for rule in rules if rule["id"] == "AES-AG-2")

    assert trace_rule["execution_mode"] == "artifact"
    assert trace_rule["source_refs"]
    assert trace_rule["title"]
    assert runtime_rule["execution_mode"] == "runtime_gate"
    assert obligations["authoritative_package_root"] == "saguaro"
    assert "Saguaro/" in obligations["excluded_reference_roots"]


def test_compiled_rules_written_to_repo_include_authoritative_fields() -> None:
    rules = json.loads(Path("standards/AES_RULES.json").read_text(encoding="utf-8"))
    assert rules
    sample = next(rule for rule in rules if rule["id"] == "AES-ARCH-4")
    dynamic_exec_rule = next(rule for rule in rules if rule["id"] == "AES-PY-4")
    for field in ("title", "source_version", "source_refs", "execution_mode", "status"):
        assert field in sample
    assert dynamic_exec_rule["status"] == "blocking"
    assert dynamic_exec_rule["check_function"].endswith("check_no_dynamic_execution")
