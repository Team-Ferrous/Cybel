from tools.verify import run_all_verifications, verify_qsg_runtime


def test_verify_qsg_runtime_passes_smoke_check() -> None:
    result = verify_qsg_runtime(".")

    assert result.passed is True
    assert result.tool == "qsg_runtime"


def test_run_all_verifications_includes_qsg_runtime() -> None:
    result = run_all_verifications(".")
    tools = {item.tool for item in result.results}

    assert "qsg_runtime" in tools
