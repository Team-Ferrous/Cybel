"""Deterministic specialist catalog and routing rules."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from core.agents.specialists.manifest import manifest_prompt_key_overrides
from core.agents.specialists.task_packet import (
    AESConstraint,
    ArtifactExpectation,
    FailureEscalation,
    RepoConstraint,
    SovereignPolicy,
    TaskPacket,
    TelemetryContract,
)


@dataclass
class SpecialistProfile:
    role: str
    domains: List[str] = field(default_factory=list)
    hardware_tags: List[str] = field(default_factory=list)
    question_types: List[str] = field(default_factory=list)
    repo_roles: List[str] = field(default_factory=list)
    required_for_aal: List[str] = field(default_factory=list)


@dataclass
class RoutingDecision:
    primary_role: str
    reviewer_roles: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def default_specialist_catalog() -> List[SpecialistProfile]:
    return [
        SpecialistProfile(
            role="CampaignDirectorSubagent", domains=["planning", "campaign"]
        ),
        SpecialistProfile(
            role="SoftwareArchitectureSubagent",
            domains=["architecture", "design", "boundaries", "interfaces"],
            question_types=["architecture", "objective_unknown"],
        ),
        SpecialistProfile(
            role="RepoCampaignAnalysisSubagent",
            domains=["repo", "repo_intake", "attached_repo", "competitor"],
            repo_roles=[
                "target",
                "analysis_local",
                "analysis_external",
                "attached_repo",
            ],
        ),
        SpecialistProfile(
            role="RepoIngestionSubagent",
            domains=["repo"],
            repo_roles=["target", "analysis_local", "analysis_external"],
        ),
        SpecialistProfile(role="ResearchCrawlerSubagent", domains=["research", "web"]),
        SpecialistProfile(
            role="ArchitectureAdjudicatorSubagent",
            question_types=["architecture", "objective_unknown"],
        ),
        SpecialistProfile(
            role="FeatureCartographerSubagent", domains=["feature_map", "market"]
        ),
        SpecialistProfile(
            role="HypothesisLabSubagent", domains=["hypothesis", "experiments"]
        ),
        SpecialistProfile(
            role="MarketAnalysisSubagent", domains=["market", "competitor"]
        ),
        SpecialistProfile(role="MathAnalysisSubagent", domains=["math", "algorithms"]),
        SpecialistProfile(
            role="CPUOptimizerSubagent",
            domains=["native", "performance", "cpu_optimization"],
            hardware_tags=["simd", "avx2", "openmp", "neon"],
        ),
        SpecialistProfile(
            role="HardwareOptimizationSubagent",
            domains=["native", "performance"],
            hardware_tags=["simd", "avx2", "openmp"],
        ),
        SpecialistProfile(
            role="QuantumAlgorithmsSubagent", domains=["quantum_algorithms"]
        ),
        SpecialistProfile(
            role="QuantumComputingSubagent",
            domains=["quantum_computing", "quantum_runtime", "quantum_circuits"],
        ),
        SpecialistProfile(
            role="CADGeometrySubagent", domains=["cad", "geometry", "mechanical"]
        ),
        SpecialistProfile(
            role="AeronauticsSubagent",
            domains=["aerospace", "aeronautics", "aircraft", "flight_dynamics"],
        ),
        SpecialistProfile(
            role="ThrusterSubagent",
            domains=["thruster", "propulsion", "electric_propulsion", "rcs"],
        ),
        SpecialistProfile(
            role="ElectromagnetismSubagent",
            domains=["electromagnetism", "electromagnetic", "em", "emi", "emc", "rf"],
        ),
        SpecialistProfile(
            role="MagneticsSubagent",
            domains=["magnetics", "magnetic", "coil", "inductor", "transformer"],
        ),
        SpecialistProfile(
            role="MagnetohydrodynamicsSubagent",
            domains=["mhd", "magnetohydrodynamics", "conductive_fluid"],
        ),
        SpecialistProfile(
            role="PlasmaPhysicsSubagent",
            domains=["plasma", "plasma_physics", "plasma_diagnostics"],
        ),
        SpecialistProfile(
            role="ElectricPropulsionSubagent",
            domains=["electric_propulsion", "hall_thruster", "ion_thruster", "vasimr"],
        ),
        SpecialistProfile(
            role="MagneticsManufacturingSubagent",
            domains=["magnetics_manufacturing", "coil_winding", "magnet_fabrication"],
        ),
        SpecialistProfile(
            role="CFDSurrogateSubagent",
            domains=["cfd_surrogate", "surrogate", "deepcfd", "rom"],
        ),
        SpecialistProfile(
            role="MHDSurrogateSubagent",
            domains=["mhd_surrogate", "surrogate", "mhd"],
        ),
        SpecialistProfile(
            role="EMSurrogateSubagent",
            domains=["em_surrogate", "surrogate", "electromagnetic_surrogate"],
        ),
        SpecialistProfile(
            role="HeatTransferSurrogateSubagent",
            domains=["heat_transfer_surrogate", "thermal_surrogate", "surrogate"],
        ),
        SpecialistProfile(
            role="DigitalTwinArchitectSubagent",
            domains=["digital_twin", "digital_twins", "digital_twin_architecture"],
        ),
        SpecialistProfile(
            role="SimulationAccelerationSubagent",
            domains=["simulation_acceleration", "solver_acceleration"],
        ),
        SpecialistProfile(
            role="DroneAutonomySubagent",
            domains=["drone", "uav", "drone_autonomy", "drone_racing"],
        ),
        SpecialistProfile(
            role="SensorFusionSubagent",
            domains=["sensor_fusion", "state_estimation", "perception"],
        ),
        SpecialistProfile(
            role="RoboticsKinematicsSubagent",
            domains=["robotics", "kinematics", "robot_arm"],
        ),
        SpecialistProfile(
            role="UAVSwarmSubagent",
            domains=["uav_swarm", "swarm", "multi_agent_uav"],
        ),
        SpecialistProfile(
            role="AutonomousSystemsVnVSubagent",
            domains=["autonomy_vnv", "autonomous_vnv", "autonomous_verification"],
        ),
        SpecialistProfile(
            role="ManufacturingAutomationSubagent",
            domains=[
                "manufacturing_automation",
                "industrial_automation",
                "factory_automation",
            ],
        ),
        SpecialistProfile(
            role="AdditiveManufacturingSubagent",
            domains=["additive_manufacturing", "3d_printing"],
        ),
        SpecialistProfile(
            role="IndustrialIoTSubagent",
            domains=["industrial_iot", "iiot", "edge_telemetry"],
        ),
        SpecialistProfile(
            role="ProcessControlSubagent",
            domains=["process_control", "pid", "mpc", "control_loops"],
        ),
        SpecialistProfile(
            role="IndustrialEconomicsSubagent",
            domains=["industrial_economics", "capex", "opex"],
        ),
        SpecialistProfile(
            role="ElectromechanicalSystemsSubagent",
            domains=["electromechanical", "actuator_integration"],
        ),
        SpecialistProfile(
            role="MagnetoresponsiveMaterialsSubagent",
            domains=["magnetoresponsive", "smart_materials", "magnetic_materials"],
        ),
        SpecialistProfile(
            role="PiezoelectricAndSmartMaterialsSubagent",
            domains=["piezoelectric", "shape_memory", "smart_materials"],
        ),
        SpecialistProfile(
            role="DeadCodeTriageSubagent",
            domains=["deadcode", "dead_code", "obsolete_code", "code_cleanup"],
        ),
        SpecialistProfile(
            role="SimulationVnVSubagent",
            domains=["simulation", "simulation_vv", "verification_validation", "vnv"],
        ),
        SpecialistProfile(
            role="ABICartographerSubagent",
            domains=["abi", "abi_drift", "compatibility"],
        ),
        SpecialistProfile(
            role="DeterminismSheriffSubagent",
            domains=["determinism", "reproducibility", "nondeterminism"],
        ),
        SpecialistProfile(
            role="PlatformDXArchitectSubagent",
            domains=["platform_dx", "developer_experience", "ergonomics"],
        ),
        SpecialistProfile(
            role="ObservabilityContractEngineerSubagent",
            domains=["observability_contract", "telemetry_contract", "slo"],
        ),
        SpecialistProfile(
            role="SupplyChainProvenanceAuditorSubagent",
            domains=["supply_chain", "provenance", "dependency_integrity", "sbom"],
        ),
        SpecialistProfile(
            role="ThreatBoundaryAnalystSubagent",
            domains=["threat_modeling", "trust_boundary", "attack_surface", "security"],
        ),
        SpecialistProfile(
            role="CompatibilityMigrationArchitectSubagent",
            domains=["compatibility_migration", "compatibility", "migration", "deprecation"],
        ),
        SpecialistProfile(
            role="FailureEconomistSubagent",
            domains=["failure_economics", "blast_radius", "resilience", "reliability"],
        ),
        SpecialistProfile(
            role="FinOpsModelEconomistSubagent",
            domains=["finops", "cost_optimization", "model_economics", "budget"],
        ),
        SpecialistProfile(
            role="FitnessFunctionEngineerSubagent",
            domains=["fitness_function", "objective_function", "quality_gates", "optimization"],
        ),
        SpecialistProfile(
            role="PhysicsSimulationSubagent", domains=["physics", "simulation"]
        ),
        SpecialistProfile(
            role="ImplementationEngineerSubagent", domains=["implementation", "code"]
        ),
        SpecialistProfile(
            role="BenchmarkEngineerSubagent", domains=["benchmark", "performance"]
        ),
        SpecialistProfile(
            role="TelemetrySystemsSubagent", domains=["telemetry", "observability"]
        ),
        SpecialistProfile(role="TestAuditSubagent", domains=["test", "audit"]),
        SpecialistProfile(
            role="CodebaseCartographerSubagent", domains=["architecture", "codebase"]
        ),
        SpecialistProfile(
            role="ReleasePackagingSubagent", domains=["release", "packaging"]
        ),
        SpecialistProfile(
            role="DocumentationWhitepaperSubagent", domains=["docs", "whitepaper"]
        ),
        SpecialistProfile(
            role="AESSentinelSubagent", required_for_aal=["AAL-0", "AAL-1"]
        ),
        SpecialistProfile(
            role="DeterminismComplianceSubagent",
            required_for_aal=["AAL-0", "AAL-1", "AAL-2"],
        ),
    ]


class SpecialistRegistry:
    """Route work to specialist roles and build structured task packets."""

    def __init__(self, catalog: Iterable[SpecialistProfile] | None = None) -> None:
        self.catalog = {
            profile.role: profile
            for profile in (catalog or default_specialist_catalog())
        }
        self._prompt_key_by_role = {
            "SoftwareArchitectureSubagent": "architecture/software_architecture",
            "SystemCartographerSubagent": "architecture/system_cartographer",
            "ADRStewardSubagent": "architecture/adr_steward",
            "CounterfactualStrategistSubagent": "architecture/counterfactual_strategist",
            "RepoCampaignAnalysisSubagent": "repo/repo_campaign_analysis",
            "FormalMethodsSubagent": "science/formal_methods",
            "ScientificComputingSubagent": "science/scientific_computing",
            "ResearchLibrarianSubagent": "science/research_librarian",
            "QuantumMechanicsSubagent": "science/quantum_mechanics",
            "QuantumComputingSubagent": "science/quantum_computing",
            "PlatformDXArchitectSubagent": "governance/platform_dx_architect",
            "ObservabilityContractEngineerSubagent": "governance/observability_contract_engineer",
            "SupplyChainProvenanceAuditorSubagent": "governance/supply_chain_provenance_auditor",
            "ThreatBoundaryAnalystSubagent": "governance/threat_boundary_analyst",
            "CompatibilityMigrationArchitectSubagent": "governance/compatibility_migration_architect",
            "FailureEconomistSubagent": "governance/failure_economist",
            "FinOpsModelEconomistSubagent": "governance/finops_model_economist",
            "FitnessFunctionEngineerSubagent": "governance/fitness_function_engineer",
            "MechanicalEngineeringSubagent": "engineering/mechanical_engineering",
            "CADGeometrySubagent": "engineering/cad_geometry",
            "ElectricalEngineeringSubagent": "engineering/electrical_engineering",
            "EmbeddedSystemsSubagent": "engineering/embedded_systems",
            "MaterialsManufacturingSubagent": "engineering/materials_manufacturing",
            "DFMNegotiatorSubagent": "engineering/dfm_negotiator",
            "GCodeSafetyMarshalSubagent": "engineering/gcode_safety_marshal",
            "AerospaceSystemsSubagent": "aerospace/aerospace_systems",
            "AeronauticsSubagent": "aerospace/aeronautics",
            "PropulsionSubagent": "aerospace/propulsion",
            "ThrusterSubagent": "aerospace/thruster",
            "FluidsSubagent": "aerospace/fluids",
            "ControlsGNCSubagent": "aerospace/controls_gnc",
            "SimulationVnVSubagent": "aerospace/simulation_vnv",
            "MDOTradeStudySubagent": "aerospace/mdo_trade_study",
            "ThermalSystemsSubagent": "aerospace/thermal_systems",
            "ElectromagnetismSubagent": "em/electromagnetism",
            "MagneticsSubagent": "em/magnetics",
            "MagnetohydrodynamicsSubagent": "em/magnetohydrodynamics",
            "PlasmaPhysicsSubagent": "em/plasma_physics",
            "ElectricPropulsionSubagent": "em/electric_propulsion",
            "MagneticsManufacturingSubagent": "em/magnetics_manufacturing",
            "ElectromechanicalSystemsSubagent": "em/electromechanical_systems",
            "MagnetoresponsiveMaterialsSubagent": "em/magnetoresponsive_materials",
            "PiezoelectricAndSmartMaterialsSubagent": "em/piezoelectric_smart_materials",
            "CFDSurrogateSubagent": "surrogates/cfd_surrogate",
            "MHDSurrogateSubagent": "surrogates/mhd_surrogate",
            "EMSurrogateSubagent": "surrogates/em_surrogate",
            "HeatTransferSurrogateSubagent": "surrogates/heat_transfer_surrogate",
            "DigitalTwinArchitectSubagent": "surrogates/digital_twin_architect",
            "SimulationAccelerationSubagent": "surrogates/simulation_acceleration",
            "DroneAutonomySubagent": "robotics/drone_autonomy",
            "SensorFusionSubagent": "robotics/sensor_fusion",
            "RoboticsKinematicsSubagent": "robotics/robotics_kinematics",
            "UAVSwarmSubagent": "robotics/uav_swarm",
            "AutonomousSystemsVnVSubagent": "robotics/autonomous_systems_vnv",
            "ManufacturingAutomationSubagent": "industrial/manufacturing_automation",
            "AdditiveManufacturingSubagent": "industrial/additive_manufacturing",
            "IndustrialIoTSubagent": "industrial/industrial_iot",
            "ProcessControlSubagent": "industrial/process_control",
            "IndustrialEconomicsSubagent": "industrial/industrial_economics",
            "BenchmarkEngineerSubagent": "toolchain/benchmark_engineer",
            "PMUPathologistSubagent": "toolchain/pmu_pathologist",
            "BinaryLayoutSurgeonSubagent": "toolchain/binary_layout_surgeon",
            "LoweringBlacksmithSubagent": "toolchain/lowering_blacksmith",
            "TranslationValidationProsecutorSubagent": "toolchain/translation_validation_prosecutor",
            "ABICartographerSubagent": "toolchain/abi_cartographer",
            "DeterminismSheriffSubagent": "toolchain/determinism_sheriff",
            "IntegrationTreatyBrokerSubagent": "toolchain/integration_treaty_broker",
            "CounterexampleCoronerSubagent": "toolchain/counterexample_coroner",
            "DeadCodeTriageSubagent": "toolchain/dead_code_triage",
            "CPUOptimizerSubagent": "toolchain/benchmark_engineer",
            "CampaignDirectorSubagent": "foundation/campaign_director",
            "RepoIngestionSubagent": "foundation/repo_ingestion",
            "ResearchCrawlerSubagent": "foundation/research_crawler",
            "ArchitectureAdjudicatorSubagent": "foundation/architecture_adjudicator",
            "FeatureCartographerSubagent": "foundation/feature_cartographer",
            "HypothesisLabSubagent": "foundation/hypothesis_lab",
            "MarketAnalysisSubagent": "foundation/market_analysis",
            "MathAnalysisSubagent": "foundation/math_analysis",
            "HardwareOptimizationSubagent": "foundation/hardware_optimization",
            "QuantumAlgorithmsSubagent": "foundation/quantum_algorithms",
            "PhysicsSimulationSubagent": "foundation/physics_simulation",
            "ImplementationEngineerSubagent": "foundation/implementation_engineer",
            "TelemetrySystemsSubagent": "foundation/telemetry_systems",
            "TestAuditSubagent": "foundation/test_audit",
            "CodebaseCartographerSubagent": "foundation/codebase_cartographer",
            "ReleasePackagingSubagent": "foundation/release_packaging",
            "DocumentationWhitepaperSubagent": "foundation/documentation_whitepaper",
            "AESSentinelSubagent": "foundation/aes_sentinel",
            "DeterminismComplianceSubagent": "foundation/determinism_compliance",
        }
        self._prompt_key_by_role.update(manifest_prompt_key_overrides())

    def prompt_key_for_role(self, role: str) -> str:
        """Return optional prompt key mapped to a role."""
        return str(self._prompt_key_by_role.get(str(role or "").strip(), "")).strip()

    def route(
        self,
        *,
        objective: str,
        domains: Iterable[str] | None = None,
        hardware_targets: Iterable[str] | None = None,
        repo_roles: Iterable[str] | None = None,
        question_type: str = "",
        aal: str = "AAL-3",
        required_artifacts: Iterable[str] | None = None,
    ) -> RoutingDecision:
        objective_lower = str(objective or "").lower()
        domain_set = {str(item).strip().lower() for item in (domains or [])}
        hardware_set = {str(item).lower() for item in (hardware_targets or [])}
        repo_role_set = {str(item).strip().lower() for item in (repo_roles or [])}
        required_artifact_set = {str(item) for item in (required_artifacts or [])}
        deadcode_keywords = {
            "deadcode",
            "dead code",
            "obsolete code",
            "unused code",
            "code cleanup",
        }

        reasons: List[str] = []
        primary = "ImplementationEngineerSubagent"

        if (
            question_type in {"architecture", "objective_unknown"}
            or {"architecture", "design"} & domain_set
        ):
            primary = "SoftwareArchitectureSubagent"
            reasons.append("architecture_first")
        elif (
            {"repo", "repo_intake", "attached_repo", "competitor"} & domain_set
            or repo_role_set & {"analysis_local", "analysis_external", "attached_repo"}
            or "attached repo" in objective_lower
            or "repo intake" in objective_lower
        ):
            primary = "RepoCampaignAnalysisSubagent"
            reasons.append("repo_analysis")
        elif {
            "deadcode",
            "dead_code",
            "obsolete_code",
            "code_cleanup",
        } & domain_set or (
            any(keyword in objective_lower for keyword in deadcode_keywords)
            and (
                "triage" in objective_lower
                or "classify" in objective_lower
                or "obsolete" in objective_lower
            )
        ):
            primary = "DeadCodeTriageSubagent"
            reasons.append("deadcode_triage")
        elif {
            "quantum_runtime",
            "quantum_computing",
            "quantum_circuits",
        } & domain_set or "quantum runtime" in objective_lower:
            primary = "QuantumComputingSubagent"
            reasons.append("quantum_runtime")
        elif (
            {"electromagnetism", "electromagnetic", "em", "emc", "emi", "rf"}
            & domain_set
            or "electromagnetic" in objective_lower
            or "shielding" in objective_lower
            or "em interference" in objective_lower
        ):
            primary = "ElectromagnetismSubagent"
            reasons.append("electromagnetism")
        elif (
            {"magnetics", "magnetic", "coil", "inductor", "transformer"} & domain_set
            or "coil design" in objective_lower
            or "coil winding" in objective_lower
            or "magnetic circuit" in objective_lower
            or "magnetic actuator" in objective_lower
            or "inductor" in objective_lower
        ):
            primary = "MagneticsSubagent"
            reasons.append("magnetics")
        elif (
            {"mhd", "magnetohydrodynamics", "conductive_fluid"} & domain_set
            or "magnetohydrodynamics" in objective_lower
            or "mhd" in objective_lower
        ):
            primary = "MagnetohydrodynamicsSubagent"
            reasons.append("mhd")
        elif (
            {"plasma", "plasma_physics", "plasma_diagnostics"} & domain_set
            or "plasma thruster" in objective_lower
            or "plasma confinement" in objective_lower
        ):
            primary = "PlasmaPhysicsSubagent"
            reasons.append("plasma")
        elif (
            {"hall_thruster", "ion_thruster", "electric_propulsion"} & domain_set
            or "hall thruster" in objective_lower
            or "ion thruster" in objective_lower
            or "electric propulsion" in objective_lower
        ):
            primary = "ElectricPropulsionSubagent"
            reasons.append("electric_propulsion")
        elif (
            {"cfd_surrogate", "deepcfd", "rom"} & domain_set
            or "cfd surrogate" in objective_lower
            or "deepcfd" in objective_lower
        ):
            primary = "CFDSurrogateSubagent"
            reasons.append("cfd_surrogate")
        elif {"mhd_surrogate"} & domain_set or "mhd surrogate" in objective_lower:
            primary = "MHDSurrogateSubagent"
            reasons.append("mhd_surrogate")
        elif (
            {"em_surrogate", "electromagnetic_surrogate"} & domain_set
            or "em surrogate" in objective_lower
            or "electromagnetic surrogate" in objective_lower
        ):
            primary = "EMSurrogateSubagent"
            reasons.append("em_surrogate")
        elif (
            {"heat_transfer_surrogate", "thermal_surrogate"} & domain_set
            or "thermal surrogate" in objective_lower
            or "heat transfer surrogate" in objective_lower
        ):
            primary = "HeatTransferSurrogateSubagent"
            reasons.append("heat_transfer_surrogate")
        elif {
            "digital_twin",
            "digital_twins",
            "digital_twin_architecture",
        } & domain_set or "digital twin" in objective_lower:
            primary = "DigitalTwinArchitectSubagent"
            reasons.append("digital_twin")
        elif {
            "simulation_acceleration",
            "solver_acceleration",
        } & domain_set or "simulation acceleration" in objective_lower:
            primary = "SimulationAccelerationSubagent"
            reasons.append("simulation_acceleration")
        elif (
            {"drone", "uav", "drone_autonomy", "drone_racing"} & domain_set
            or "drone racing" in objective_lower
            or "uav autonomy" in objective_lower
        ):
            primary = "DroneAutonomySubagent"
            reasons.append("drone_autonomy")
        elif {
            "sensor_fusion",
            "state_estimation",
            "perception",
        } & domain_set or "sensor fusion" in objective_lower:
            primary = "SensorFusionSubagent"
            reasons.append("sensor_fusion")
        elif {
            "robotics",
            "kinematics",
            "robot_arm",
        } & domain_set or "robot kinematics" in objective_lower:
            primary = "RoboticsKinematicsSubagent"
            reasons.append("robotics_kinematics")
        elif (
            {"uav_swarm", "swarm", "multi_agent_uav"} & domain_set
            or "uav swarm" in objective_lower
            or "drone swarm" in objective_lower
        ):
            primary = "UAVSwarmSubagent"
            reasons.append("uav_swarm")
        elif {
            "autonomy_vnv",
            "autonomous_vnv",
            "autonomous_verification",
        } & domain_set or "autonomous verification" in objective_lower:
            primary = "AutonomousSystemsVnVSubagent"
            reasons.append("autonomy_vnv")
        elif (
            {"manufacturing_automation", "industrial_automation", "factory_automation"}
            & domain_set
            or "manufacturing automation" in objective_lower
            or "industrial automation" in objective_lower
        ):
            primary = "ManufacturingAutomationSubagent"
            reasons.append("manufacturing_automation")
        elif (
            {"process_control", "pid", "mpc", "control_loops"} & domain_set
            or "process control" in objective_lower
            or "pid tuning" in objective_lower
            or "model predictive control" in objective_lower
        ):
            primary = "ProcessControlSubagent"
            reasons.append("process_control")
        elif (
            {"additive_manufacturing", "3d_printing"} & domain_set
            or "additive manufacturing" in objective_lower
            or "3d printing process" in objective_lower
        ):
            primary = "AdditiveManufacturingSubagent"
            reasons.append("additive_manufacturing")
        elif (
            {"industrial_iot", "iiot", "edge_telemetry"} & domain_set
            or "industrial iot" in objective_lower
            or "iiot" in objective_lower
        ):
            primary = "IndustrialIoTSubagent"
            reasons.append("industrial_iot")
        elif (
            {"industrial_economics", "capex", "opex"} & domain_set
            or "industrial economics" in objective_lower
            or "capex" in objective_lower
            or "opex" in objective_lower
        ):
            primary = "IndustrialEconomicsSubagent"
            reasons.append("industrial_economics")
        elif {"cad", "geometry", "mechanical"} & domain_set or "cad" in objective_lower:
            primary = "CADGeometrySubagent"
            reasons.append("cad_geometry")
        elif (
            {"aerospace", "aeronautics", "aircraft", "flight_dynamics"} & domain_set
            or "aeronautics" in objective_lower
            or "aircraft" in objective_lower
        ):
            primary = "AeronauticsSubagent"
            reasons.append("aerospace")
        elif (
            {"thruster", "propulsion", "electric_propulsion", "rcs"} & domain_set
            or "thruster" in objective_lower
            or "electric propulsion" in objective_lower
        ):
            primary = "ThrusterSubagent"
            reasons.append("thruster")
        elif (
            {"simulation", "simulation_vv", "verification_validation", "vnv"}
            & domain_set
            or "simulation v&v" in objective_lower
            or "simulation vnv" in objective_lower
        ):
            primary = "SimulationVnVSubagent"
            reasons.append("simulation_vv")
        elif (
            {"abi", "abi_drift", "compatibility"} & domain_set
            or "abi drift" in objective_lower
            or "abi compatibility" in objective_lower
        ):
            primary = "ABICartographerSubagent"
            reasons.append("abi_drift")
        elif (
            {"determinism", "reproducibility", "nondeterminism"} & domain_set
            or "determinism" in objective_lower
            or "nondeterminism" in objective_lower
            or "reproducible build" in objective_lower
        ):
            primary = "DeterminismSheriffSubagent"
            reasons.append("determinism")
        elif (
            {"platform_dx", "developer_experience", "ergonomics"} & domain_set
            or "developer experience" in objective_lower
            or "dx" in objective_lower
        ):
            primary = "PlatformDXArchitectSubagent"
            reasons.append("platform_dx")
        elif (
            {"observability_contract", "telemetry_contract", "slo"} & domain_set
            or "telemetry contract" in objective_lower
            or "observability contract" in objective_lower
        ):
            primary = "ObservabilityContractEngineerSubagent"
            reasons.append("observability_contract")
        elif (
            {"supply_chain", "provenance", "dependency_integrity", "sbom"} & domain_set
            or "supply chain" in objective_lower
            or "artifact provenance" in objective_lower
            or "sbom" in objective_lower
        ):
            primary = "SupplyChainProvenanceAuditorSubagent"
            reasons.append("supply_chain_provenance")
        elif (
            {"threat_modeling", "trust_boundary", "attack_surface", "security"} & domain_set
            or "threat model" in objective_lower
            or "trust boundary" in objective_lower
            or "attack surface" in objective_lower
        ):
            primary = "ThreatBoundaryAnalystSubagent"
            reasons.append("threat_boundary")
        elif (
            {"compatibility_migration", "compatibility", "migration", "deprecation"}
            & domain_set
            or "backward compatible" in objective_lower
            or "deprecation strategy" in objective_lower
            or "migration plan" in objective_lower
        ):
            primary = "CompatibilityMigrationArchitectSubagent"
            reasons.append("compatibility_migration")
        elif (
            {"failure_economics", "blast_radius", "resilience", "reliability"} & domain_set
            or "blast radius" in objective_lower
            or "failure economics" in objective_lower
        ):
            primary = "FailureEconomistSubagent"
            reasons.append("failure_economics")
        elif (
            {"finops", "cost_optimization", "model_economics", "budget"} & domain_set
            or "finops" in objective_lower
            or "cost optimization" in objective_lower
        ):
            primary = "FinOpsModelEconomistSubagent"
            reasons.append("finops")
        elif (
            {"fitness_function", "objective_function", "quality_gates", "optimization"}
            & domain_set
            or "fitness function" in objective_lower
            or "quality gate" in objective_lower
        ):
            primary = "FitnessFunctionEngineerSubagent"
            reasons.append("fitness_function")
        elif {"feature_map", "market"} & domain_set:
            primary = "FeatureCartographerSubagent"
            reasons.append("feature_mapping")
        elif {
            "telemetry",
            "observability",
        } & domain_set or "telemetry" in required_artifact_set:
            primary = "TelemetrySystemsSubagent"
            reasons.append("telemetry")
        elif {"benchmark", "performance"} & domain_set or hardware_set & {
            "simd",
            "avx2",
            "openmp",
        }:
            primary = "CPUOptimizerSubagent"
            reasons.append("hardware")
        elif {"hypothesis", "experiments"} & domain_set:
            primary = "HypothesisLabSubagent"
            reasons.append("hypothesis")
        elif {"research", "web"} & domain_set:
            primary = "ResearchCrawlerSubagent"
            reasons.append("research")
        elif {"math", "algorithms"} & domain_set:
            primary = "MathAnalysisSubagent"
            reasons.append("math")
        elif {"audit", "test"} & domain_set:
            primary = "TestAuditSubagent"
            reasons.append("audit")

        reviewers: List[str] = []
        if aal in {"AAL-0", "AAL-1"}:
            reviewers.extend(["AESSentinelSubagent", "DeterminismComplianceSubagent"])
            reasons.append("high_aal")
        elif aal == "AAL-2":
            reviewers.append("DeterminismComplianceSubagent")
            reasons.append("aal2")

        if {"telemetry", "audit"} & domain_set:
            reviewers.append("TelemetrySystemsSubagent")
        if primary in {"CPUOptimizerSubagent", "HardwareOptimizationSubagent"}:
            reviewers.append("BenchmarkEngineerSubagent")
        if primary in {"ABICartographerSubagent", "DeterminismSheriffSubagent"}:
            reviewers.append("DeterminismComplianceSubagent")
        if primary == "DroneAutonomySubagent":
            reviewers.append("SensorFusionSubagent")
        if primary == "DeadCodeTriageSubagent":
            reviewers.append("TestAuditSubagent")

        deduped_reviewers: List[str] = []
        seen = {primary}
        for role in reviewers:
            if role not in seen:
                seen.add(role)
                deduped_reviewers.append(role)
        return RoutingDecision(
            primary_role=primary, reviewer_roles=deduped_reviewers, reasons=reasons
        )

    def build_task_packet(
        self,
        *,
        objective: str,
        aal: str = "AAL-3",
        domains: Iterable[str] | None = None,
        hardware_targets: Iterable[str] | None = None,
        repo_roles: Iterable[str] | None = None,
        allowed_repos: Iterable[str] | None = None,
        forbidden_repos: Iterable[str] | None = None,
        required_artifacts: Iterable[str] | None = None,
        produced_artifacts: Iterable[str] | None = None,
        question_type: str = "",
        allowed_tools: Iterable[str] | None = None,
    ) -> TaskPacket:
        decision = self.route(
            objective=objective,
            domains=domains,
            hardware_targets=hardware_targets,
            repo_roles=repo_roles,
            question_type=question_type,
            aal=aal,
            required_artifacts=required_artifacts,
        )
        specialist_prompt_key = self._prompt_key_by_role.get(decision.primary_role, "")
        return TaskPacket(
            task_packet_id=f"specialist_{int(time.time() * 1000)}",
            objective=objective,
            specialist_role=decision.primary_role,
            repo_constraint=RepoConstraint(
                allowed_repos=list(allowed_repos or []),
                forbidden_repos=list(forbidden_repos or []),
                required_repo_roles=list(repo_roles or []),
            ),
            artifact_expectation=ArtifactExpectation(
                required_artifacts=list(required_artifacts or []),
                produced_artifacts=list(produced_artifacts or []),
            ),
            aes_constraint=AESConstraint(
                aal=aal,
                required_reviewers=decision.reviewer_roles,
            ),
            telemetry_contract=TelemetryContract(
                required_metrics=["wall_time", "artifact_emission_status"],
                logs_required=True,
            ),
            failure_escalation=FailureEscalation(
                escalate_to=decision.reviewer_roles or ["CampaignDirectorSubagent"],
                retry_policy={"max_attempts": 2},
                stop_conditions=["unsafe_action", "unresolved_blocker"],
            ),
            allowed_tools=list(allowed_tools or []),
            prompt_profile="sovereign_build",
            specialist_prompt_key=specialist_prompt_key,
            sovereign_policy=SovereignPolicy(
                enabled=True,
                policy_block=(
                    "Default assumption: build owned internal algorithms, IRs, "
                    "simulators, geometry/netlist pipelines, and benchmarks "
                    "unless the user explicitly requests external frameworks."
                ),
            ),
            evidence_bundle={"routing_reasons": decision.reasons},
        )
