from core.agents.specialists import SpecialistRegistry, default_specialist_catalog


def test_architecture_routes_to_software_architecture_first():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Design service boundaries for attached repo intake",
        domains=["architecture", "repo"],
        repo_roles=["analysis_local"],
        question_type="architecture",
    )

    assert decision.primary_role == "SoftwareArchitectureSubagent"
    assert "architecture_first" in decision.reasons


def test_catalog_includes_modern_architecture_roles():
    roles = {profile.role for profile in default_specialist_catalog()}
    assert "SoftwareArchitectureSubagent" in roles
    assert "RepoCampaignAnalysisSubagent" in roles
