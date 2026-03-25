from core.agents.specialists import SpecialistRegistry


def test_route_hardware_high_aal_task():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Implement AVX2 optimized kernel with benchmark telemetry",
        domains=["performance", "benchmark", "native"],
        hardware_targets=["avx2"],
        repo_roles=["target"],
        aal="AAL-1",
    )

    assert decision.primary_role == "CPUOptimizerSubagent"
    assert "AESSentinelSubagent" in decision.reviewer_roles
    assert "DeterminismComplianceSubagent" in decision.reviewer_roles
    assert "BenchmarkEngineerSubagent" in decision.reviewer_roles


def test_build_task_packet_captures_constraints():
    registry = SpecialistRegistry()
    packet = registry.build_task_packet(
        objective="Resolve blocking architecture question",
        aal="AAL-2",
        domains=["architecture"],
        repo_roles=["artifact_store"],
        allowed_repos=["artifact_store"],
        required_artifacts=["architecture"],
        produced_artifacts=["roadmap_draft"],
        question_type="architecture",
        allowed_tools=["saguaro_query", "read_file"],
    )

    assert packet.specialist_role == "SoftwareArchitectureSubagent"
    assert packet.repo_constraint.allowed_repos == ["artifact_store"]
    assert packet.artifact_expectation.required_artifacts == ["architecture"]
    assert "DeterminismComplianceSubagent" in packet.aes_constraint.required_reviewers
    assert packet.prompt_profile == "sovereign_build"
    assert packet.sovereign_policy.enabled is True


def test_routes_coil_design_to_magnetics():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Design a coil winding strategy for magnetic actuator prototype",
        domains=[],
    )

    assert decision.primary_role == "MagneticsSubagent"


def test_routes_hall_thruster_sizing_to_electric_propulsion():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Perform Hall thruster sizing with plume constraints",
        domains=[],
    )

    assert decision.primary_role == "ElectricPropulsionSubagent"


def test_routes_drone_racing_to_drone_autonomy_with_sensor_fusion_reviewer():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Build a drone racing autonomy stack for sim-to-real transfer",
        domains=[],
    )

    assert decision.primary_role == "DroneAutonomySubagent"
    assert "SensorFusionSubagent" in decision.reviewer_roles


def test_routes_deadcode_triage_requests():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Triage Saguaro deadcode output and classify obsolete symbols",
        domains=[],
    )

    assert decision.primary_role == "DeadCodeTriageSubagent"
