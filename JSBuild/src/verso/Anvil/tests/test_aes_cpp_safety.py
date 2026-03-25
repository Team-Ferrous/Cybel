from core.aes.checks.hpc_checks import (
    check_c_style_casts,
    check_nodiscard_returns,
    check_raii_enforcement,
)


def test_raii_enforcement_flags_raw_new_without_smart_ptr_markers() -> None:
    source = """
Status allocate_buffer() {
    auto ptr = new float[64];
    return Status::Ok();
}
"""
    violations = check_raii_enforcement(source, "kernel.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CPP-3"


def test_raii_enforcement_allows_smart_pointer_usage() -> None:
    source = """
Status allocate_buffer() {
    auto ptr = std::make_unique<float[]>(64);
    return Status::Ok();
}
"""
    assert check_raii_enforcement(source, "kernel.cc") == []


def test_nodiscard_returns_flags_error_bearing_signature_without_attribute() -> None:
    source = """
Status run_kernel(int n) {
    return Status::Ok();
}
"""
    violations = check_nodiscard_returns(source, "kernel.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CPP-4"


def test_nodiscard_returns_accepts_annotated_signature() -> None:
    source = """
[[nodiscard]] Status run_kernel(int n) {
    return Status::Ok();
}
"""
    assert check_nodiscard_returns(source, "kernel.cc") == []


def test_c_style_casts_flags_c_style_cast() -> None:
    source = """
int cast_it(float value) {
    return (int)value;
}
"""
    violations = check_c_style_casts(source, "kernel.cc")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CPP-5"


def test_c_style_casts_accepts_static_cast() -> None:
    source = """
int cast_it(float value) {
    return static_cast<int>(value);
}
"""
    assert check_c_style_casts(source, "kernel.cc") == []
