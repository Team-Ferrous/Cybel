import numpy as np

from core.reasoning.tool_intent_classifier import ToolIntentClassifier


def test_tool_intent_detects_saguaro_query_from_context_bias():
    classifier = ToolIntentClassifier(
        embedding_dim=32,
        tool_names=["saguaro_query", "read_file", "slice"],
        threshold=0.40,
    )
    hidden = np.ones((1, 32), dtype=np.float32)
    signal = classifier.detect(
        hidden_state=hidden,
        context_text="Analyze architecture and search where this module is implemented.",
    )
    assert signal is not None
    assert signal.tool_name in {"saguaro_query", "slice"}
    assert signal.confidence >= 0.40


def test_tool_intent_returns_none_when_below_threshold():
    classifier = ToolIntentClassifier(
        embedding_dim=16,
        tool_names=["saguaro_query", "read_file"],
        threshold=0.99,
    )
    hidden = np.zeros((1, 16), dtype=np.float32)
    signal = classifier.detect(hidden_state=hidden, context_text="hello")
    assert signal is None
