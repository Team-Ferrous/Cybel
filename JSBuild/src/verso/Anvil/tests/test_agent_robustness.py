from core.response_utils import ResponseStreamParser
from core.agent import BaseAgent


def test_parser_robustness():
    parser = ResponseStreamParser()

    test_cases = [
        # Case 1: Standard XML
        (
            ["I will list. ", "<tool_call>", '{"name": "ls"}', "</tool_call>"],
            ["tool_start", "tool_end"],
        ),
        # Case 2: Granite Native
        (["tool|>", '{"name": "ls"}', "<|"], ["tool_start", "tool_end"]),
        # Case 3: XML Attributes (User Reported)
        (
            [
                'Analyze. <|start_of_role|>tool<tool_name="skeleton"><arguments><target>"core/agent.py"</target></arguments></tool>'
            ],
            ["tool_start", "tool_end"],
        ),
    ]

    for chunks, expected in test_cases:
        parser = ResponseStreamParser()
        events = []
        for chunk in chunks:
            events.extend(list(parser.process_chunk(chunk)))
        events.extend(list(parser.finalize()))

        event_types = [e.type for e in events]
        print(f"Test case {chunks} -> {event_types}")
        for exp in expected:
            assert exp in event_types

    print("✅ Parser Robustness Test Passed!")


def test_extraction_robustness():
    agent = BaseAgent(name="Test")

    # Test cases for different formats
    formats = [
        '<tool_call> {"name": "test_xml", "arguments": {}} </tool_call>',
        'tool|> {"name": "test_native", "arguments": {}} <|',
        # User reported case
        'tool<tool_name="skeleton"><arguments><target>"core/agent.py"</target></arguments></tool>',
        # Malformed but common
        'tool<tool_name="slice"> target: "file.py"',
    ]

    for i, text in enumerate(formats):
        calls = agent._extract_tool_calls(text)
        print(f"Extraction Case {i}: {text} -> {calls}")
        assert len(calls) == 1

    print("✅ Extraction Robustness Test Passed!")


if __name__ == "__main__":
    test_parser_robustness()
    test_extraction_robustness()
