from core.agents.specialists import SpecialistRegistry


def test_abi_drift_routes_to_toolchain_specialist():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Investigate ABI drift across compiler upgrades",
        domains=["abi"],
    )

    assert decision.primary_role == "ABICartographerSubagent"
    assert "DeterminismComplianceSubagent" in decision.reviewer_roles


def test_determinism_routes_to_sheriff():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Hunt nondeterminism in release build pipeline",
        domains=["determinism"],
    )

    assert decision.primary_role == "DeterminismSheriffSubagent"


def test_cpu_optimizer_route_requires_benchmark_reviewer():
    registry = SpecialistRegistry()
    decision = registry.route(
        objective="Optimize SIMD kernel latency",
        domains=["performance"],
        hardware_targets=["avx2"],
    )

    assert decision.primary_role == "CPUOptimizerSubagent"
    assert "BenchmarkEngineerSubagent" in decision.reviewer_roles
