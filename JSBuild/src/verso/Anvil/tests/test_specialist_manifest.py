from core.agents.specialists import SpecialistRegistry, route_specialist
from core.agents.specialists.manifest import (
    manifest_generic_role_aliases,
    manifest_prompt_key_overrides,
)


def test_specialist_manifest_exposes_authoritative_aliases():
    aliases = manifest_generic_role_aliases()

    assert aliases["implementer"] == "ImplementationEngineerSubagent"
    assert aliases["deadcode"] == "DeadCodeTriageSubagent"


def test_specialist_registry_applies_manifest_prompt_key_overrides():
    registry = SpecialistRegistry()
    prompt_keys = manifest_prompt_key_overrides()

    assert (
        registry.prompt_key_for_role("DeadCodeTriageSubagent")
        == prompt_keys["DeadCodeTriageSubagent"]
    )


def test_route_specialist_uses_manifest_backed_deadcode_alias():
    registry = SpecialistRegistry()
    decision = route_specialist(
        registry=registry,
        objective="Classify dead code candidates from the latest Saguaro report",
        requested_role="deadcode",
        aal="AAL-2",
        domains=[],
        question_type="investigation",
        repo_roles=["analysis_local"],
    )

    assert decision.primary_role == "DeadCodeTriageSubagent"
    assert "requested_role_explicit" in decision.reasons
