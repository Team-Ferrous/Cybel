import pytest
import os
import sys
from unittest.mock import MagicMock

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.response_utils import finalize_synthesis
from core.thinking import EnhancedThinkingSystem, ThinkingType


@pytest.fixture
def mock_dependencies():
    """Provides mock objects for dependencies of finalize_synthesis."""
    mock_thinking_system = EnhancedThinkingSystem()
    mock_brain = MagicMock()
    mock_console = MagicMock()
    # Start a chain so it's not None
    mock_thinking_system.start_chain("test_task")
    return mock_thinking_system, mock_brain, mock_console


def test_finalize_synthesis_with_answer_and_thoughts(mock_dependencies):
    """
    Tests that the function correctly extracts the answer and removes thinking blocks.
    """
    thinking_system, brain, console = mock_dependencies

    # Use unindented string to ensure regex matches correctly
    # IMPORTANT: Ensure no leading spaces on lines with tags for safety, although regex uses finditer
    raw_response = """<thinking type="understanding">
This is the user's request.
</thinking>
This is the final answer that the user should see.
<thinking type="reflection">
I think I did a good job.
</thinking>
And here is some more of the answer."""

    print(f"DEBUG: raw_response length={len(raw_response)}")
    parsed = thinking_system.parser.parse(raw_response)
    print(f"DEBUG: parsed blocks count={len(parsed)}")
    for b in parsed:
        print(f"DEBUG: block type={b.type}, content='{b.content}'")

    final_answer = finalize_synthesis(raw_response, thinking_system, brain, [], console)

    # 1. Check that the final answer is clean
    expected_answer = "This is the final answer that the user should see.\n\nAnd here is some more of the answer."
    assert final_answer.strip() == expected_answer.strip()

    # 2. Check that the thinking blocks were parsed and added to the chain
    non_compliance_blocks = [
        block
        for block in thinking_system.current_chain.blocks
        if block.type != ThinkingType.COMPLIANCE
    ]
    assert len(non_compliance_blocks) == 2
    assert non_compliance_blocks[0].type == ThinkingType.UNDERSTANDING
    assert "user's request" in non_compliance_blocks[0].content
    assert non_compliance_blocks[1].type == ThinkingType.REFLECTION

    # 3. Check that the fallback brain call was NOT made
    brain.stream_chat.assert_not_called()


def test_finalize_synthesis_with_only_thoughts(mock_dependencies):
    """Tests the fallback mechanism: when the response only contains thoughts,
    it should make a new call to the brain to synthesize them.
    """
    thinking_system, brain, console = mock_dependencies

    raw_response = """<thinking type="reasoning">
Okay, the first step is to analyze the evidence. The evidence shows A and B.
</thinking>
<thinking type="reasoning">
From A and B, I can conclude C.
Therefore, the final conclusion is C.
</thinking>"""

    parsed = thinking_system.parser.parse(raw_response)
    print(f"DEBUG: only_thoughts parsed count={len(parsed)}")

    # Mock the brain's response for the fallback call
    fallback_response_stream = ["The final answer is C, based on the evidence."]
    brain.stream_chat.return_value = iter(fallback_response_stream)

    final_answer = finalize_synthesis(raw_response, thinking_system, brain, [], console)

    # 1. Check that the final answer is the one from the fallback call
    assert final_answer.strip() == "The final answer is C, based on the evidence."

    # 2. Check that the thinking blocks were parsed
    non_compliance_blocks = [
        block
        for block in thinking_system.current_chain.blocks
        if block.type != ThinkingType.COMPLIANCE
    ]
    assert len(non_compliance_blocks) == 2

    # 3. Check that the fallback brain call WAS made
    brain.stream_chat.assert_called_once()
    call_args = brain.stream_chat.call_args[0][0]
    # Check the content of the prompt sent to the fallback call
    assert "Internal Monologue:" in call_args[1]["content"]
    assert "From A and B, I can conclude C" in call_args[1]["content"]


def test_finalize_synthesis_with_empty_response(mock_dependencies):
    """Tests that it returns a default message if the response is empty and has no thoughts."""
    thinking_system, brain, console = mock_dependencies

    raw_response = ""

    final_answer = finalize_synthesis(raw_response, thinking_system, brain, [], console)

    assert "could not formulate a final answer" in final_answer
    brain.stream_chat.assert_not_called()


def test_finalize_synthesis_with_whitespace_answer(mock_dependencies):
    """Tests that it triggers fallback if the answer is only whitespace."""
    thinking_system, brain, console = mock_dependencies

    raw_response = """<thinking type="planning">
I will do this.
</thinking>
  
    """

    fallback_response_stream = ["Based on my plan, I will do this."]
    brain.stream_chat.return_value = iter(fallback_response_stream)

    final_answer = finalize_synthesis(raw_response, thinking_system, brain, [], console)

    assert final_answer.strip() == "Based on my plan, I will do this."
    brain.stream_chat.assert_called_once()
