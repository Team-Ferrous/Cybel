from unittest.mock import MagicMock

from core.unified_chat_loop import UnifiedChatLoop


def _bare_loop():
    loop = UnifiedChatLoop.__new__(UnifiedChatLoop)
    loop.brain = MagicMock()
    loop._last_classification_meta = {}
    return loop


def test_keyword_match_uses_fast_path_without_llm():
    loop = _bare_loop()
    loop._semantic_intent_similarity = MagicMock(return_value=("investigation", 0.7))
    loop._llm_classify_intent = MagicMock(return_value="investigation")

    intent = UnifiedChatLoop._classify_request(loop, "Create a new file for metrics")

    assert intent == "creation"
    loop._semantic_intent_similarity.assert_not_called()
    loop._llm_classify_intent.assert_not_called()


def test_ambiguous_input_uses_semantic_similarity():
    loop = _bare_loop()
    loop._semantic_intent_similarity = MagicMock(return_value=("investigation", 0.83))
    loop._llm_classify_intent = MagicMock(return_value=None)

    intent = UnifiedChatLoop._classify_request(loop, "Need help with this part")

    assert intent == "investigation"
    loop._semantic_intent_similarity.assert_called_once()
    loop._llm_classify_intent.assert_not_called()
