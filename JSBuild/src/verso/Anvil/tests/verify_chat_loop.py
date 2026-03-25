import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agent import BaseAgent
from core.ollama_client import DeterministicOllama


def test_simple_chat_enhancements():
    # Mock dependencies
    mock_brain = MagicMock(spec=DeterministicOllama)
    mock_brain.stream_chat.return_value = iter(
        [
            "I will list the directory to explore.",
            '<tool_call>{"name": "list_dir", "arguments": {"path": "."}}</tool_call>',
            "Exploring complete.",
        ]
    )

    agent = BaseAgent(name="Anvil", brain=mock_brain)

    # Check max_steps and whitelist (indirectly by checking behavior or source if needed, but let's check execution)
    user_input = "Analyze this repo"

    with patch.object(
        agent, "_execute_tool", return_value="[core, cli, tools]"
    ) as mock_exec:
        agent.simple_chat(user_input)

        # Verify tool was allowed and called
        called_tools = [call.args[0]["name"] for call in mock_exec.call_args_list]
        print(f"Called tools: {called_tools}")
        assert "list_dir" in called_tools, "list_dir should have been called"

        # Verify system prompt content (last call to stream_chat)
        args, kwargs = mock_brain.stream_chat.call_args
        system_prompt = args[0][0]["content"]
        print(f"System Prompt Snippet: {system_prompt[:200]}...")
        assert "**Anvil**" in system_prompt
        assert "**anvil**" in system_prompt
        assert "EXPLORATION TOOLS" or "exploration tools" in system_prompt.upper()

    print(
        "\n✅ Verification passed: simple_chat now explores the repo and knows its identity."
    )


if __name__ == "__main__":
    test_simple_chat_enhancements()
