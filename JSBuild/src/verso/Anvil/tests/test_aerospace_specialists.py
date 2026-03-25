from core.agents.specialists import SpecialistRegistry


def test_aeronautics_route_by_domain():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Analyze aircraft stability margins",
        domains=["aeronautics"],
    )

    assert decision.primary_role == "AeronauticsSubagent"


def test_thruster_route_by_objective_keyword():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Compare electric thruster options for RCS",
        domains=[],
    )

    assert decision.primary_role == "ThrusterSubagent"


def test_simulation_vnv_route_by_phrase():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Create simulation V&V plan for flight control loop",
        domains=[],
    )

    assert decision.primary_role == "SimulationVnVSubagent"
