from core.aes.checks.metadata_checks import (
    check_no_silent_fallback_markers,
    check_no_verification_bypass_markers,
)


def test_silent_fallback_markers_flag_swallowed_exception() -> None:
    source = """
def parse_payload(payload: str):
    try:
        return decode(payload)
    except Exception:
        return None
"""
    violations = check_no_silent_fallback_markers(source, "parser.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-1"


def test_silent_fallback_markers_ignore_logged_exception() -> None:
    source = """
import logging

def parse_payload(payload: str):
    try:
        return decode(payload)
    except Exception:
        logging.exception("decode failed")
        return None
"""
    assert check_no_silent_fallback_markers(source, "parser.py") == []


def test_verification_bypass_markers_ignore_strings_and_comments() -> None:
    source = '''
IMMUTABLE_RULES = [
    "Cannot disable saguaro verify for AAL-0/1 code",
]

def message() -> str:
    # skip verification would be bad here, but this is just commentary
    return "Skipping drift check."
'''
    assert check_no_verification_bypass_markers(source, "governance.py") == []


def test_verification_bypass_markers_flag_truthy_assignment() -> None:
    source = """
def configure():
    skip_verification = True
    return skip_verification
"""
    violations = check_no_verification_bypass_markers(source, "config.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-3"


def test_verification_bypass_markers_flag_truthy_keyword() -> None:
    source = """
def run():
    return verifier.execute(disable_validation_check=True)
"""
    violations = check_no_verification_bypass_markers(source, "runner.py")
    assert len(violations) == 1
    assert violations[0]["rule_id"] == "AES-CR-3"
