import json
from pathlib import Path

from saguaro.sentinel.engines.aes import AESEngine

UNIVERSAL_RULE = {
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

ML_RULE = {
    "id": "AES-ML-1",
    "section": "5a.1",
    "text": "Training loops must check finite values",
    "severity": "AAL-1",
    "engine": "semantic",
    "auto_fixable": False,
    "domain": ["ml"],
    "language": ["python"],
    "check_function": "core.aes.checks.ml_checks.check_gradient_health_gate",
}

VISUALS_PRESENCE_RULE = {
    "id": "AES-VIS-1",
    "section": "2",
    "text": "Visual governance packs must include v1 and v2 directives",
    "severity": "AAL-0",
    "engine": "agent",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["json"],
    "check_function": "core.aes.checks.universal_checks.check_aes_visuals_pack_presence",
}

VISUALS_SHAPE_RULE = {
    "id": "AES-VIS-2",
    "section": "2",
    "text": "Visual governance packs must remain model-readable",
    "severity": "AAL-0",
    "engine": "agent",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["json"],
    "check_function": "core.aes.checks.universal_checks.check_aes_visuals_pack_shape",
}

RUNTIME_RULE = {
    "id": "AES-OBS-1",
    "section": "10",
    "text": "Telemetry contract required",
    "severity": "AAL-2",
    "engine": "agent",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["python"],
    "execution_mode": "runtime_gate",
    "required_artifacts": ["runtime_gates.json", "telemetry_contract.json"],
    "source_version": "v2",
    "source_refs": ["aes_analysis/v2/deep-research-report v2.md#operations"],
    "precedence": 100,
    "selectors": {},
    "parameters": {},
    "waiverable": False,
    "rollout_stage": "ratchet",
    "status": "blocking",
    "title": "Telemetry Contract",
}

MANUAL_RULE = {
    "id": "AES-REV-9",
    "section": "11",
    "text": "Manual review required for irreversible deployment",
    "severity": "AAL-1",
    "engine": "human",
    "auto_fixable": False,
    "domain": ["universal"],
    "language": ["python", "md"],
    "execution_mode": "manual",
    "required_artifacts": [],
    "source_version": "v2",
    "source_refs": ["aes_analysis/v2/deep-research-report v2.md#independent-review"],
    "precedence": 100,
    "selectors": {},
    "parameters": {},
    "waiverable": True,
    "rollout_stage": "ratchet",
    "status": "advisory",
    "title": "Manual Review Record",
}


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_rules(repo_root: Path, rules: list[dict]) -> None:
    _write(repo_root / "standards" / "AES_RULES.json", json.dumps(rules))


def _valid_visual_pack(profile: str) -> dict:
    return {
        "schema_version": "1.0",
        "artifact": "anvil_aes_visual_directives",
        "profile": profile,
        "generated_on": "2026-03-03",
        "owner": "Anvil",
        "upstream_context": {
            "aes_baseline_file": "standards/AES_SOURCE_BASELINES.md",
            "intent": "deterministic test pack",
        },
        "directives": [
            {
                "directive_id": f"{profile.upper()}-001",
                "title": "Directive title",
                "rationale": "Directive rationale.",
                "enforcement_targets": ["prompt_generation"],
                "implementation_patterns": ["Pattern one"],
                "verification_checks": ["Check one"],
                "source_refs": ["S01"],
            }
        ],
    }


def _change_manifest() -> dict:
    return {
        "run_id": "run-1",
        "changed_files": ["critical.py"],
        "aal": "AAL-1",
        "domains": ["universal"],
        "hot_paths": [],
        "public_api_changes": [],
        "dependency_changes": [],
        "required_rule_ids": [],
        "required_runtime_gates": [],
    }


def _review_matrix() -> str:
    return """aal_levels:
  AAL-0:
    independent_reviews: 2
    iv_and_v_required: true
    human_approval_blocking: true
  AAL-1:
    independent_reviews: 1
    iv_and_v_required: true
    human_approval_blocking: true
  AAL-2:
    independent_reviews: 1
    iv_and_v_required: false
    human_approval_blocking: false
  AAL-3:
    independent_reviews: 0
    iv_and_v_required: false
    human_approval_blocking: false
"""


def test_aes_engine_loads_rules_from_registry(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE, ML_RULE])
    engine = AESEngine(str(tmp_path))

    assert len(engine.registry.rules) == 2
    assert engine.registry.get_rule("AES-ML-1") is not None


def test_aes_engine_runs_universal_check_functions(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE])
    _write(tmp_path / "module.py", "try:\n    pass\nexcept:\n    pass\n")

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    assert any(item["rule_id"] == "AES-CR-2" for item in violations)
    assert any(item["closure_level"] == "blocking" for item in violations)


def test_aes_engine_respects_domain_scoping_for_ml(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE, ML_RULE])
    _write(
        tmp_path / "trainer.py",
        (
            "try:\n    pass\nexcept:\n    pass\n"
            "def train(loss, optimizer):\n"
            "    loss.backward()\n"
            "    optimizer.step()\n"
        ),
    )

    engine = AESEngine(str(tmp_path))
    engine.set_policy({"verify_context": {"domain": "ml"}})
    violations = engine.run(path_arg=str(tmp_path / "trainer.py"))

    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-ML-1" in rule_ids
    assert "AES-CR-2" not in rule_ids


def test_aes_engine_fail_closed_on_missing_high_aal_artifacts(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [])
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": True,
            "verify_context": {},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    rule_ids = {item["rule_id"] for item in violations}
    assert "AES-AG-1" in rule_ids
    assert "AES-TR-1" in rule_ids
    assert "AES-TR-2" in rule_ids
    assert all(item["closure_level"] == "blocking" for item in violations)


def test_aes_engine_flags_missing_visuals_packs(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [VISUALS_PRESENCE_RULE, VISUALS_SHAPE_RULE])

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path))

    visuals_violations = [item for item in violations if item["rule_id"] == "AES-VIS-1"]
    assert len(visuals_violations) == 2
    assert all(item["closure_level"] == "blocking" for item in visuals_violations)
    messages = {item["message"] for item in visuals_violations}
    assert (
        "Missing model-readable visuals governance pack: aes_visuals/v1/directives.json"
        in messages
    )
    assert (
        "Missing model-readable visuals governance pack: aes_visuals/v2/directives.json"
        in messages
    )


def test_aes_engine_accepts_valid_visuals_packs(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [VISUALS_PRESENCE_RULE, VISUALS_SHAPE_RULE])
    _write(
        tmp_path / "aes_visuals" / "v1" / "directives.json",
        json.dumps(_valid_visual_pack("v1")),
    )
    _write(
        tmp_path / "aes_visuals" / "v2" / "directives.json",
        json.dumps(_valid_visual_pack("v2")),
    )

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path))

    assert not {item["rule_id"] for item in violations}.intersection({"AES-VIS-1", "AES-VIS-2"})


def test_aes_engine_skips_runtime_gate_rules_without_governed_context(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [RUNTIME_RULE])
    _write(tmp_path / "module.py", "def run(trace_id):\n    return trace_id\n")

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    assert not any(item["rule_id"] == "AES-OBS-1" for item in violations)


def test_aes_engine_blocks_missing_change_manifest_for_high_assurance_context(
    tmp_path: Path,
) -> None:
    _seed_rules(tmp_path, [])
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy({"block_on_missing_artifacts": True, "verify_context": {}})
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    ag1 = [item for item in violations if item["rule_id"] == "AES-AG-1"]
    assert ag1
    assert ag1[0]["closure_level"] == "blocking"


def test_aes_engine_flags_insufficient_review_signoffs_via_artifact_checks(
    tmp_path: Path,
) -> None:
    _seed_rules(tmp_path, [])
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    _write(
        tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl",
        '{"trace_id":"trace-1"}\n',
    )
    _write(
        tmp_path / "standards" / "evidence_bundle.json",
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "review_signoffs": [
                    {
                        "reviewer": "alice",
                        "decision": "approved",
                    }
                ],
            }
        ),
    )
    _write(tmp_path / "standards" / "review_matrix.yaml", _review_matrix())
    manifest_path = tmp_path / "standards" / "change_manifest.json"
    _write(manifest_path, json.dumps(_change_manifest()))

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": True,
            "verify_context": {"change_manifest_path": str(manifest_path)},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    rev = [item for item in violations if item["rule_id"] == "AES-REV-1"]
    assert rev
    assert "requires 2 approvals" in rev[0]["message"]


def test_aes_engine_satisfies_review_signoffs_when_threshold_is_met(
    tmp_path: Path,
) -> None:
    _seed_rules(tmp_path, [])
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    _write(
        tmp_path / "standards" / "traceability" / "TRACEABILITY.jsonl",
        '{"trace_id":"trace-1"}\n',
    )
    _write(
        tmp_path / "standards" / "evidence_bundle.json",
        json.dumps(
            {
                "bundle_id": "bundle-1",
                "review_signoffs": [
                    {"reviewer": "alice", "decision": "approved"},
                    {"reviewer": "bob", "decision": "approved"},
                ],
            }
        ),
    )
    _write(tmp_path / "standards" / "review_matrix.yaml", _review_matrix())
    manifest_path = tmp_path / "standards" / "change_manifest.json"
    _write(manifest_path, json.dumps(_change_manifest()))

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": True,
            "verify_context": {"change_manifest_path": str(manifest_path)},
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    assert not any(item["rule_id"] == "AES-REV-1" for item in violations)


def test_aes_engine_enforces_runtime_gate_rules_when_required_by_context(
    tmp_path: Path,
) -> None:
    _seed_rules(tmp_path, [RUNTIME_RULE])
    _write(tmp_path / "module.py", "def run(trace_id):\n    return trace_id\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "verify_context": {
                "compliance_context": {
                    "run_id": "run-7",
                    "aal": "AAL-1",
                    "required_rule_ids": ["AES-OBS-1"],
                }
            }
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    obs = [item for item in violations if item["rule_id"] == "AES-OBS-1"]
    assert obs
    assert "runtime_gates.json" in obs[0]["message"]
    assert obs[0]["closure_level"] == "blocking"


def test_aes_engine_manual_rules_require_waiver_or_record(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [MANUAL_RULE])
    _write(tmp_path / "module.py", "def run():\n    return 'ok'\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "verify_context": {
                "compliance_context": {
                    "run_id": "run-8",
                    "aal": "AAL-1",
                    "required_rule_ids": ["AES-REV-9"],
                }
            }
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    manual = [item for item in violations if item["rule_id"] == "AES-REV-9"]
    assert manual
    assert manual[0]["closure_level"] == "advisory"


def test_aes_engine_manual_rules_clear_when_waiver_is_present(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [MANUAL_RULE])
    _write(tmp_path / "module.py", "def run():\n    return 'ok'\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "verify_context": {
                "compliance_context": {
                    "run_id": "run-9",
                    "aal": "AAL-1",
                    "required_rule_ids": ["AES-REV-9"],
                    "waiver_ids": ["waiver-1"],
                }
            }
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    assert not any(item["rule_id"] == "AES-REV-9" for item in violations)


def test_aes_engine_excludes_generated_and_reference_roots(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE])
    _write(tmp_path / ".anvil" / "artifacts" / "generated.py", "try:\n    pass\nexcept:\n    pass\n")
    _write(tmp_path / "Saguaro" / "generated.py", "try:\n    pass\nexcept:\n    pass\n")

    engine = AESEngine(str(tmp_path))
    violations = engine.run(path_arg=str(tmp_path))

    assert violations == []


def test_aes_engine_blocks_invalid_change_manifest_for_high_aal(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [])
    _write(tmp_path / "critical.py", "# _mm256_add_ps\n")
    _write(
        tmp_path / "change_manifest.json",
        json.dumps({"run_id": "run-1", "aal": "AAL-0"}),
    )

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {
                "change_manifest_path": str(tmp_path / "change_manifest.json"),
            },
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "critical.py"))

    manifest_violations = [item for item in violations if item["rule_id"] == "AES-AG-1"]
    assert manifest_violations
    assert "Invalid change manifest" in manifest_violations[0]["message"]
    assert manifest_violations[0]["closure_level"] == "blocking"


def test_aes_engine_runs_runtime_gates_for_governed_context(tmp_path: Path) -> None:
    runtime_rule = {
        "id": "AES-OBS-1",
        "section": "10",
        "text": "Chronicle evidence required for hot-path work",
        "severity": "AAL-1",
        "engine": "agent",
        "auto_fixable": False,
        "domain": ["universal"],
        "language": ["python"],
        "execution_mode": "runtime_gate",
        "required_artifacts": ["chronicle.json"],
        "status": "blocking",
        "title": "Chronicle Evidence Required",
        "source_refs": ["aes_analysis/v2/deep-research-report v2.md#chronicle"],
    }
    _seed_rules(tmp_path, [runtime_rule])
    _write(tmp_path / "module.py", "def run(value):\n    return value\n")
    _write(
        tmp_path / "change_manifest.json",
        json.dumps(
            {
                "run_id": "run-rt",
                "changed_files": ["module.py"],
                "aal": "AAL-1",
                "domains": ["universal"],
                "hot_paths": ["module.py"],
                "public_api_changes": [],
                "dependency_changes": [],
                "required_rule_ids": ["AES-OBS-1"],
                "required_runtime_gates": ["chronicle_gate"],
            }
        ),
    )

    engine = AESEngine(str(tmp_path))
    engine.set_policy(
        {
            "block_on_missing_artifacts": False,
            "verify_context": {
                "change_manifest_path": str(tmp_path / "change_manifest.json"),
            },
        }
    )
    violations = engine.run(path_arg=str(tmp_path / "module.py"))

    assert any(
        item["rule_id"] == "AES-OBS-1"
        and item.get("context") == "runtime_gate:chronicle_gate"
        for item in violations
    )


def test_aes_engine_is_deterministic_for_repeated_repo_scans(tmp_path: Path) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE])
    _write(tmp_path / "saguaro" / "a.py", "try:\n    pass\nexcept:\n    pass\n")
    _write(tmp_path / "saguaro" / "b.py", "try:\n    pass\nexcept:\n    pass\n")

    engine = AESEngine(str(tmp_path))

    first = engine.run(path_arg=str(tmp_path))
    second = engine.run(path_arg=str(tmp_path))

    assert first == second


def test_aes_engine_scopes_repo_root_scan_to_authoritative_package_root(
    tmp_path: Path,
) -> None:
    _seed_rules(tmp_path, [UNIVERSAL_RULE])
    _write(tmp_path / "saguaro" / "owned.py", "try:\n    pass\nexcept:\n    pass\n")
    _write(tmp_path / "external" / "ignored.py", "try:\n    pass\nexcept:\n    pass\n")

    engine = AESEngine(str(tmp_path))
    engine.set_policy({"aes_authoritative_package_root": "saguaro"})
    violations = engine.run(path_arg=str(tmp_path))

    files = {item["file"] for item in violations}
    assert "saguaro/owned.py" in files
    assert "external/ignored.py" not in files
