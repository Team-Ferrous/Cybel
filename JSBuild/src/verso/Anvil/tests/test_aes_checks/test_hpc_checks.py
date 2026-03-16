from core.aes.checks.hpc_checks import (
    check_alignment_contracts,
    check_explicit_omp_clauses,
    check_scalar_reference_impl,
)


def test_alignment_contracts_flag_simd_without_alignment_markers() -> None:
    source = """
void kernel(float* data) {
    __m256 v = _mm256_load_ps(data);
}
"""
    violations = check_alignment_contracts(source, "kernel.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-HPC-2"


def test_alignment_contracts_pass_with_alignas_marker() -> None:
    source = """
alignas(32) float data[8];
__m256 v = _mm256_load_ps(data);
"""
    assert check_alignment_contracts(source, "kernel.cc") == []


def test_alignment_contracts_ignore_unaligned_intrinsics_without_marker() -> None:
    source = """
void kernel(float* data) {
    __m256 v = _mm256_loadu_ps(data);
}
"""
    assert check_alignment_contracts(source, "kernel.cc") == []


def test_omp_clauses_flag_parallel_without_explicit_clause() -> None:
    source = """
#pragma omp parallel
for (int i = 0; i < n; ++i) {
    out[i] = in[i];
}
"""
    violations = check_explicit_omp_clauses(source, "omp.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-HPC-3"


def test_omp_clauses_pass_with_private_and_schedule() -> None:
    source = """
#pragma omp parallel private(i) schedule(static)
for (int i = 0; i < n; ++i) {
    out[i] = in[i];
}
"""
    assert check_explicit_omp_clauses(source, "omp.cc") == []


def test_scalar_reference_impl_flags_simd_without_oracle_marker() -> None:
    source = """
void kernel(float* data) {
    __m256 v = _mm256_load_ps(data);
}
"""
    violations = check_scalar_reference_impl(source, "kernel.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-HPC-4"


def test_scalar_reference_impl_pass_with_scalar_reference_marker() -> None:
    source = """
// scalar reference implementation available in oracle path
void kernel(float* data) {
    __m256 v = _mm256_load_ps(data);
}
"""
    assert check_scalar_reference_impl(source, "kernel.cc") == []
