from core.agents.specialists import SpecialistRegistry, route_specialist
from core.agents.specialists.runtime import resolve_specialist_class


def test_route_specialist_aliases_implementer_to_foundation_role():
    registry = SpecialistRegistry()
    decision = route_specialist(
        registry=registry,
        objective="Implement logging for the auth workflow and add tests",
        requested_role="implementer",
        aal="AAL-2",
        domains=["code"],
        question_type="",
        repo_roles=["target"],
    )

    assert decision.primary_role == "ImplementationEngineerSubagent"
    assert "requested_role_explicit" in decision.reasons


def test_route_specialist_aliases_validator_to_test_audit():
    registry = SpecialistRegistry()
    decision = route_specialist(
        registry=registry,
        objective="Validate test coverage and check for flaky regressions",
        requested_role="validator",
        aal="AAL-2",
        domains=["test"],
        question_type="investigation",
        repo_roles=["analysis_local"],
    )

    assert decision.primary_role == "TestAuditSubagent"
    assert "requested_role_explicit" in decision.reasons


def test_registry_has_prompt_keys_for_foundational_roles():
    registry = SpecialistRegistry()
    assert registry.prompt_key_for_role("CampaignDirectorSubagent") == "foundation/campaign_director"
    assert registry.prompt_key_for_role("AESSentinelSubagent") == "foundation/aes_sentinel"


def test_resolve_specialist_class_for_expansion_roles():
    assert resolve_specialist_class("AeronauticsSubagent").__name__ == "AeronauticsSubagent"
    assert resolve_specialist_class("CFDSurrogateSubagent").__name__ == "CFDSurrogateSubagent"
