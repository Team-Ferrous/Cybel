"""Static import anchor for catalog-declared AES checks."""

from __future__ import annotations

from core.aes.checks.api_contract_checks import check_public_api_contract_markers
from core.aes.checks.architecture_checks import check_header_namespace_pollution
from core.aes.checks.hpc_checks import (
    check_alignment_contracts,
    check_c_style_casts,
    check_explicit_omp_clauses,
    check_nodiscard_returns,
    check_raii_enforcement,
    check_scalar_reference_impl,
)
from core.aes.checks.metadata_checks import (
    check_catalog_authority,
    check_no_silent_fallback_markers,
    check_no_verification_bypass_markers,
)
from core.aes.checks.ml_checks import (
    check_data_validation,
    check_gradient_health_gate,
    check_reproducibility_manifest,
    check_stable_numerics,
)
from core.aes.checks.physics_checks import (
    check_conservation_monitors,
    check_symplectic_integrator,
)
from core.aes.checks.quantum_checks import (
    check_no_magic_angles,
    check_noise_model_present,
    check_shot_sufficiency,
    check_transpilation_required,
)
from core.aes.checks.ruff_checks import check_ruff_import_order
from core.aes.checks.supply_chain_checks import check_dependency_locking
from core.aes.checks.telemetry_checks import (
    check_operational_logging_markers,
    check_traceability_markers,
)
from core.aes.checks.universal_checks import (
    check_aes_visuals_pack_presence,
    check_aes_visuals_pack_shape,
    check_complexity_bounds,
    check_context_managed_open,
    check_mutable_defaults,
    check_no_bare_except,
    check_no_dynamic_execution,
    check_no_wildcard_imports,
    check_suspicious_exception_none_returns,
    check_type_annotations,
)
from core.aes.security_verification import (
    check_hardcoded_secrets,
    check_svl_compliance,
)
from core.aes.supply_chain import check_provenance

REGISTERED_AES_CHECKS = {
    "core.aes.checks.api_contract_checks.check_public_api_contract_markers": check_public_api_contract_markers,
    "core.aes.checks.architecture_checks.check_header_namespace_pollution": check_header_namespace_pollution,
    "core.aes.checks.hpc_checks.check_alignment_contracts": check_alignment_contracts,
    "core.aes.checks.hpc_checks.check_c_style_casts": check_c_style_casts,
    "core.aes.checks.hpc_checks.check_explicit_omp_clauses": check_explicit_omp_clauses,
    "core.aes.checks.hpc_checks.check_nodiscard_returns": check_nodiscard_returns,
    "core.aes.checks.hpc_checks.check_raii_enforcement": check_raii_enforcement,
    "core.aes.checks.hpc_checks.check_scalar_reference_impl": check_scalar_reference_impl,
    "core.aes.checks.metadata_checks.check_catalog_authority": check_catalog_authority,
    "core.aes.checks.metadata_checks.check_no_silent_fallback_markers": check_no_silent_fallback_markers,
    "core.aes.checks.metadata_checks.check_no_verification_bypass_markers": check_no_verification_bypass_markers,
    "core.aes.checks.ml_checks.check_data_validation": check_data_validation,
    "core.aes.checks.ml_checks.check_gradient_health_gate": check_gradient_health_gate,
    "core.aes.checks.ml_checks.check_reproducibility_manifest": check_reproducibility_manifest,
    "core.aes.checks.ml_checks.check_stable_numerics": check_stable_numerics,
    "core.aes.checks.physics_checks.check_conservation_monitors": check_conservation_monitors,
    "core.aes.checks.physics_checks.check_symplectic_integrator": check_symplectic_integrator,
    "core.aes.checks.quantum_checks.check_no_magic_angles": check_no_magic_angles,
    "core.aes.checks.quantum_checks.check_noise_model_present": check_noise_model_present,
    "core.aes.checks.quantum_checks.check_shot_sufficiency": check_shot_sufficiency,
    "core.aes.checks.quantum_checks.check_transpilation_required": check_transpilation_required,
    "core.aes.checks.ruff_checks.check_ruff_import_order": check_ruff_import_order,
    "core.aes.checks.supply_chain_checks.check_dependency_locking": check_dependency_locking,
    "core.aes.checks.telemetry_checks.check_operational_logging_markers": check_operational_logging_markers,
    "core.aes.checks.telemetry_checks.check_traceability_markers": check_traceability_markers,
    "core.aes.checks.universal_checks.check_aes_visuals_pack_presence": check_aes_visuals_pack_presence,
    "core.aes.checks.universal_checks.check_aes_visuals_pack_shape": check_aes_visuals_pack_shape,
    "core.aes.checks.universal_checks.check_complexity_bounds": check_complexity_bounds,
    "core.aes.checks.universal_checks.check_context_managed_open": check_context_managed_open,
    "core.aes.checks.universal_checks.check_mutable_defaults": check_mutable_defaults,
    "core.aes.checks.universal_checks.check_no_bare_except": check_no_bare_except,
    "core.aes.checks.universal_checks.check_no_dynamic_execution": check_no_dynamic_execution,
    "core.aes.checks.universal_checks.check_no_wildcard_imports": check_no_wildcard_imports,
    "core.aes.checks.universal_checks.check_suspicious_exception_none_returns": check_suspicious_exception_none_returns,
    "core.aes.checks.universal_checks.check_type_annotations": check_type_annotations,
    "core.aes.security_verification.check_hardcoded_secrets": check_hardcoded_secrets,
    "core.aes.security_verification.check_svl_compliance": check_svl_compliance,
    "core.aes.supply_chain.check_provenance": check_provenance,
}

__all__ = ["REGISTERED_AES_CHECKS"]
