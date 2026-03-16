from __future__ import annotations

from core.aes.domain_detector import DomainDetector

try:
    from hypothesis import given, strategies as st
except ModuleNotFoundError:
    given = None
    st = None


def _assert_domain_subset(domains: set[str]) -> None:
    assert domains.issubset({"ml", "quantum", "physics", "hpc"})


if given is not None and st is not None:

    @given(st.text())
    def test_detector_returns_known_domain_subset(content: str) -> None:
        detector = DomainDetector()
        domains = detector.detect_from_text(content)
        _assert_domain_subset(domains)


    @given(st.text())
    def test_detector_description_alias_matches_text(description: str) -> None:
        detector = DomainDetector()
        assert detector.detect_from_description(description) == detector.detect_from_text(
            description
        )

else:

    def test_detector_returns_known_domain_subset() -> None:
        detector = DomainDetector()
        for content in ["", "import torch", "qiskit QuantumCircuit", "plain text"]:
            _assert_domain_subset(detector.detect_from_text(content))


    def test_detector_description_alias_matches_text() -> None:
        detector = DomainDetector()
        for description in ["ml optimizer", "quantum qubit", ""]:
            assert detector.detect_from_description(description) == detector.detect_from_text(
                description
            )
