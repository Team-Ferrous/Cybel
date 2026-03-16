from core.agents.specialists import SpecialistRegistry, default_specialist_catalog


def test_routes_cfd_surrogate_requests():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Create a CFD surrogate for expensive turbulence simulations",
        domains=[],
    )

    assert decision.primary_role == "CFDSurrogateSubagent"


def test_routes_industrial_automation_requests():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Plan industrial automation for a multi-station assembly line",
        domains=[],
    )

    assert decision.primary_role == "ManufacturingAutomationSubagent"


def test_routes_process_control_requests():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Tune process control loop with MPC fallback strategy",
        domains=[],
    )

    assert decision.primary_role == "ProcessControlSubagent"


def test_catalog_includes_expansion_roles():
    roles = {profile.role for profile in default_specialist_catalog()}
    assert "ElectromagnetismSubagent" in roles
    assert "CFDSurrogateSubagent" in roles
    assert "DroneAutonomySubagent" in roles
    assert "ManufacturingAutomationSubagent" in roles
    assert "DeadCodeTriageSubagent" in roles
    assert "SupplyChainProvenanceAuditorSubagent" in roles


def test_task_packet_uses_new_prompt_key_for_electric_propulsion():
    registry = SpecialistRegistry()
    packet = registry.build_task_packet(
        objective="Perform Hall thruster sizing with plume constraints",
        domains=[],
    )

    assert packet.specialist_role == "ElectricPropulsionSubagent"
    assert packet.specialist_prompt_key == "em/electric_propulsion"


def test_routes_supply_chain_provenance_requests():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Audit supply chain provenance and generate SBOM integrity findings",
        domains=[],
    )

    assert decision.primary_role == "SupplyChainProvenanceAuditorSubagent"


def test_task_packet_uses_governance_prompt_key_for_supply_chain():
    registry = SpecialistRegistry()
    packet = registry.build_task_packet(
        objective="Run supply chain provenance attestation and dependency trust checks",
        domains=[],
    )

    assert packet.specialist_role == "SupplyChainProvenanceAuditorSubagent"
    assert (
        packet.specialist_prompt_key
        == "governance/supply_chain_provenance_auditor"
    )
