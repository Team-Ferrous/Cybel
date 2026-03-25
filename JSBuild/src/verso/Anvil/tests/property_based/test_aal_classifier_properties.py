from __future__ import annotations

from core.aes.aal_classifier import AALClassifier

try:
    from hypothesis import given, strategies as st
except ModuleNotFoundError:
    given = None
    st = None


def _assert_known_level(level: str) -> None:
    assert level in {"AAL-0", "AAL-1", "AAL-2", "AAL-3"}


if given is not None and st is not None:

    @given(st.text())
    def test_classifier_always_returns_known_level(content: str) -> None:
        classifier = AALClassifier()
        _assert_known_level(classifier.classify_text(content))


    @given(st.text())
    def test_description_classification_returns_known_level(description: str) -> None:
        classifier = AALClassifier()
        _assert_known_level(classifier.classify_from_description(description))

else:

    def test_classifier_always_returns_known_level() -> None:
        classifier = AALClassifier()
        samples = ["", "loss.backward()", "_mm256_load_ps(x)", "logging.info('x')"]
        for content in samples:
            _assert_known_level(classifier.classify_text(content))


    def test_description_classification_returns_known_level() -> None:
        classifier = AALClassifier()
        samples = ["optimize gradient", "quantum circuit", "cli config", "misc text"]
        for description in samples:
            _assert_known_level(classifier.classify_from_description(description))
