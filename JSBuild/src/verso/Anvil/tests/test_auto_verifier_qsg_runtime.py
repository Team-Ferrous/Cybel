from types import SimpleNamespace

from domains.verification.auto_verifier import AutoVerifier


class _Console:
    def print(self, *args, **kwargs) -> None:
        del args, kwargs


class _Registry:
    def dispatch(self, name, payload):
        del name, payload
        return ""


def test_auto_verifier_requires_qsg_runtime_pass(monkeypatch) -> None:
    verifier = AutoVerifier(_Registry(), _Console())

    monkeypatch.setattr(verifier, "_check_syntax", lambda files: {"passed": True, "errors": []})
    monkeypatch.setattr(verifier, "_run_tests", lambda files: {"passed": True, "output": "", "skipped": True})
    monkeypatch.setattr(verifier, "_check_lint", lambda files: {"passed": True, "warnings": [], "skipped": True})
    monkeypatch.setattr(
        verifier,
        "_check_qsg_runtime",
        lambda files: {"passed": False, "violations": ["missing capability digest"], "message": "failed"},
    )
    monkeypatch.setattr(
        verifier,
        "_check_sentinel",
        lambda files: {
            "passed": True,
            "violations": [],
            "runtime_symbols": [],
            "counterexamples": [],
        },
    )
    monkeypatch.setattr(verifier, "_display_results", lambda results: None)

    results = verifier.verify_changes(["core/native/parallel_generation.py"])

    assert results["qsg_runtime"]["passed"] is False
    assert results["all_passed"] is False
