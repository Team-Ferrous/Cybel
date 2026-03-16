from core.aes.checks.api_contract_checks import check_public_api_contract_markers
from core.aes.policy_surfaces import get_surface_policy, is_error_contract_surface
from core.aes.checks.universal_checks import (
    check_complexity_bounds,
    check_error_contracts,
    check_mutable_defaults,
    check_no_bare_except,
    check_no_eval_exec,
    check_suspicious_exception_none_returns,
    check_type_annotations,
)


def test_no_bare_except_flags_bare_handler() -> None:
    source = """
def guarded() -> None:
    try:
        risky()
    except:
        pass
"""
    violations = check_no_bare_except(source, "sample.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-2"


def test_no_bare_except_ignores_typed_handler() -> None:
    source = """
def guarded() -> None:
    try:
        risky()
    except ValueError:
        pass
"""
    assert check_no_bare_except(source, "sample.py") == []


def test_type_annotations_flag_public_function_missing_hints() -> None:
    source = """
def public_api(value):
    return value
"""
    violations = check_type_annotations(source, "api.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-PY-1"


def test_type_annotations_pass_with_full_hints() -> None:
    source = """
def public_api(value: int) -> int:
    return value
"""
    assert check_type_annotations(source, "api.py") == []


def test_type_annotations_skip_native_binding_surfaces() -> None:
    source = """
def load_native(value):
    return value
"""
    assert check_type_annotations(source, "saguaro/native/__init__.py") == []


def test_no_eval_exec_flags_eval_and_exec_calls() -> None:
    source = """
def unsafe(payload: str) -> None:
    eval(payload)
    exec(payload)
"""
    violations = check_no_eval_exec(source, "unsafe.py")
    assert {item["rule_id"] for item in violations} == {"AES-SEC-2"}
    assert len(violations) == 2


def test_complexity_bounds_flags_high_complexity_function() -> None:
    source = """
def complicated(x: int) -> int:
    if x == 0:
        return 0
    if x == 1:
        return 1
    if x == 2:
        return 2
    if x == 3:
        return 3
    if x == 4:
        return 4
    if x == 5:
        return 5
    if x == 6:
        return 6
    if x == 7:
        return 7
    if x == 8:
        return 8
    if x == 9:
        return 9
    if x == 10:
        return 10
    return x
"""
    violations = check_complexity_bounds(source, "logic.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CPLX-1"


def test_error_contracts_flag_missing_raises_docstring() -> None:
    source = """
def parse_config(path: str) -> dict[str, str]:
    \"\"\"Parse config into a dictionary.\"\"\"
    return {"path": path}
"""
    violations = check_error_contracts(source, "config.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-ERR-1"


def test_error_contracts_pass_with_raises_section() -> None:
    source = '''
def parse_config(path: str) -> dict[str, str]:
    """Parse config into a dictionary.

    Raises:
        ValueError: If the input path is invalid.
    """
    return {"path": path}
'''
    assert check_error_contracts(source, "config.py") == []


def test_error_contracts_skip_native_binding_surfaces() -> None:
    source = """
def load_native(path: str) -> dict[str, str]:
    \"\"\"Load a native binding wrapper.\"\"\"
    return {"path": path}
"""
    assert check_error_contracts(source, "saguaro/native/__init__.py") == []


def test_public_api_contract_markers_skip_native_binding_surfaces() -> None:
    source = """
__all__ = ["load_native"]

def load_native(path):
    return path
"""
    assert check_public_api_contract_markers(source, "/tmp/repo/saguaro/native/__init__.py") == []


def test_aes_surface_policy_defaults_model_annotation_and_error_contract_scope() -> None:
    annotation_policy = get_surface_policy("AES-PY-1", "/tmp/repo/saguaro/native/__init__.py")
    error_policy = get_surface_policy("AES-ERR-1", "/tmp/repo/pkg/service.py")

    assert "saguaro/native/" in annotation_policy.exclude_path_prefixes
    assert "api" in error_policy.include_path_tokens
    assert error_policy.public_export_marker == "__all__"


def test_error_contract_surface_uses_aes_policy_markers() -> None:
    assert is_error_contract_surface("pkg/service.py", "def run():\n    return 1\n") is True
    assert is_error_contract_surface("pkg/module.py", "__all__ = ['run']\n") is True
    assert is_error_contract_surface("saguaro/native/__init__.py", "__all__ = ['run']\n") is False


def test_mutable_defaults_flag_list_default_argument() -> None:
    source = """
def append_item(value: int, items: list[int] = []):
    items.append(value)
    return items
"""
    violations = check_mutable_defaults(source, "mutable.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-PY-3"


def test_suspicious_none_return_flags_silent_exception_handler() -> None:
    source = """
def load_value(path: str) -> str:
    try:
        return read_value(path)
    except OSError:
        return None
"""
    violations = check_suspicious_exception_none_returns(source, "loader.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-ERR-2"


def test_suspicious_none_return_ignores_logged_exception_handler() -> None:
    source = """
import logging

def load_value(path: str) -> str:
    try:
        return read_value(path)
    except OSError:
        logging.exception("failed to load value")
        return None
"""
    assert check_suspicious_exception_none_returns(source, "loader.py") == []


def test_suspicious_none_return_ignores_optional_contract() -> None:
    source = """
from typing import Optional

def maybe_load(path: str) -> Optional[str]:
    try:
        return read_value(path)
    except OSError:
        return None
"""
    assert check_suspicious_exception_none_returns(source, "loader.py") == []


def test_suspicious_none_return_ignores_non_exception_none_return() -> None:
    source = """
def helper(flag: bool) -> None:
    if not flag:
        return None
    return None
"""
    assert check_suspicious_exception_none_returns(source, "helper.py") == []
