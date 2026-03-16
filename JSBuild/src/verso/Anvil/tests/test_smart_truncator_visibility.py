from core.utils.smart_truncator import SmartTruncator


def test_slice_output_not_truncated_by_default():
    truncator = SmartTruncator(char_threshold=100)
    text = "x" * 500
    assert truncator.truncate("slice", {}, text) == text


def test_slice_output_can_be_limited_with_max_chars():
    truncator = SmartTruncator(char_threshold=100)
    text = "x" * 500
    result = truncator.truncate("slice", {"max_chars": 100}, text)
    assert len(result) < len(text)
