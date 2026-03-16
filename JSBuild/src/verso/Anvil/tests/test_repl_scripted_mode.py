from cli.repl import (
    EXIT_POLICY_DENIED,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    EXIT_TOOL_FAILURE,
    EXIT_VERIFICATION_FAILED,
    _classify_exit,
    _coerce_policy_profile,
)


def test_coerce_policy_profile_accepts_known_values():
    assert _coerce_policy_profile("trusted") == "trusted"
    assert _coerce_policy_profile("STRICT") == "strict"


def test_classify_exit_detects_policy_denial():
    result = _classify_exit(
        response="done",
        messages=[{"content": "Tool 'run_command' execution was denied by the user."}],
        exit_on_warning=False,
    )
    assert result["status"] == "policy_denied"
    assert result["exit_code"] == EXIT_POLICY_DENIED


def test_classify_exit_detects_tool_failure():
    result = _classify_exit(
        response="done",
        messages=[{"content": "Tool Execution Error: bad arg"}],
        exit_on_warning=False,
    )
    assert result["status"] == "tool_failure"
    assert result["exit_code"] == EXIT_TOOL_FAILURE


def test_classify_exit_detects_timeout():
    result = _classify_exit(
        response="Command timed out",
        messages=[],
        exit_on_warning=False,
    )
    assert result["status"] == "timeout"
    assert result["exit_code"] == EXIT_TIMEOUT


def test_classify_exit_warning_gate():
    result = _classify_exit(
        response="ok",
        messages=[{"content": "WARNING: lint warning"}],
        exit_on_warning=True,
    )
    assert result["status"] == "verification_failed"
    assert result["exit_code"] == EXIT_VERIFICATION_FAILED

    success = _classify_exit(
        response="ok",
        messages=[{"content": "all good"}],
        exit_on_warning=True,
    )
    assert success["status"] == "success"
    assert success["exit_code"] == EXIT_SUCCESS
