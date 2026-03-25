from core.agents.prompt_loader import SpecialistPromptLoader


def test_compose_includes_base_and_inline_addendum(tmp_path):
    (tmp_path / "base_specialist.md").write_text(
        "BASE SPECIALIST PROMPT", encoding="utf-8"
    )
    loader = SpecialistPromptLoader(prompts_root=tmp_path)

    composed = loader.compose(role_addendum="INLINE ROLE ADDENDUM")

    assert "BASE SPECIALIST PROMPT" in composed
    assert "INLINE ROLE ADDENDUM" in composed


def test_compose_supports_sovereign_and_keyed_addendum(tmp_path):
    (tmp_path / "base_specialist.md").write_text("BASE", encoding="utf-8")
    (tmp_path / "sovereign_build.md").write_text("SOVEREIGN", encoding="utf-8")
    (tmp_path / "research.md").write_text("KEYED ROLE", encoding="utf-8")
    loader = SpecialistPromptLoader(prompts_root=tmp_path)

    composed = loader.compose(
        role_addendum="INLINE",
        prompt_profile="sovereign_build",
        specialist_prompt_key="research",
        sovereign_policy_block="BLOCK OVERRIDE",
    )

    assert "BASE" in composed
    assert "SOVEREIGN" in composed
    assert "KEYED ROLE" in composed
    assert "BLOCK OVERRIDE" in composed
    assert "INLINE" in composed


def test_compose_supports_nested_prompt_keys(tmp_path):
    (tmp_path / "base_specialist.md").write_text("BASE", encoding="utf-8")
    nested = tmp_path / "architecture"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "software_architecture.md").write_text("ARCH ROLE", encoding="utf-8")
    loader = SpecialistPromptLoader(prompts_root=tmp_path)

    composed = loader.compose(
        role_addendum="INLINE",
        specialist_prompt_key="architecture/software_architecture",
    )

    assert "BASE" in composed
    assert "ARCH ROLE" in composed
    assert "INLINE" in composed


def test_foundation_prompt_keys_resolve_to_content():
    loader = SpecialistPromptLoader()
    keys = [
        "foundation/campaign_director",
        "foundation/repo_ingestion",
        "foundation/research_crawler",
        "foundation/architecture_adjudicator",
        "foundation/feature_cartographer",
        "foundation/hypothesis_lab",
        "foundation/market_analysis",
        "foundation/math_analysis",
        "foundation/hardware_optimization",
        "foundation/quantum_algorithms",
        "foundation/physics_simulation",
        "foundation/implementation_engineer",
        "foundation/telemetry_systems",
        "foundation/test_audit",
        "foundation/codebase_cartographer",
        "foundation/release_packaging",
        "foundation/documentation_whitepaper",
        "foundation/aes_sentinel",
        "foundation/determinism_compliance",
    ]
    missing = [key for key in keys if not loader.load_role_addendum(key).strip()]
    assert not missing, f"Missing prompt addendum content for keys: {missing}"
