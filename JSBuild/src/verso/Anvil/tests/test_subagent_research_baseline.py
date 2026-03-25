from unittest.mock import MagicMock, patch

from core.agents.subagent import (
    SubAgent,
    UNIVERSAL_SPECIALIST_RESEARCH_TOOLS,
)


class MinimalRestrictedSubagent(SubAgent):
    tools = ["saguaro_query"]
    system_prompt = "Restricted specialist for testing."


def _make_agent(tool_names):
    mock_brain = MagicMock()
    mock_brain.model_name = "tiny"
    mock_console = MagicMock()
    mock_console.quiet = False
    mock_registry = MagicMock()
    mock_registry.get_schemas.return_value = {
        "tools": [{"name": name, "parameters": {}} for name in tool_names]
    }
    with patch("core.agent.ToolRegistry", return_value=mock_registry):
        return MinimalRestrictedSubagent(
            "Investigate specialist fallback behavior",
            "Parent",
            mock_brain,
            mock_console,
        )


def test_restricted_specialists_include_universal_research_baseline():
    tool_names = sorted(
        set(UNIVERSAL_SPECIALIST_RESEARCH_TOOLS)
        | {"saguaro_query", "grep_search", "find_by_name"}
    )
    agent = _make_agent(tool_names)
    exposed = {schema["name"] for schema in agent.tool_schemas}

    for required_tool in UNIVERSAL_SPECIALIST_RESEARCH_TOOLS:
        assert required_tool in exposed


def test_saguaro_failures_are_tagged_with_fallback_static_scan():
    tool_names = sorted(
        set(UNIVERSAL_SPECIALIST_RESEARCH_TOOLS)
        | {"saguaro_query", "grep_search", "find_by_name"}
    )
    agent = _make_agent(tool_names)
    agent.max_autonomous_steps = 1

    agent._build_specialized_system_prompt = MagicMock(return_value="SYSTEM")
    agent._build_oneshot_messages = MagicMock(return_value=[])
    agent._stream_response = MagicMock(
        return_value=(
            '<tool_call>\n{"name": "saguaro_query", "arguments": {"query": "auth", "k": 2}}\n</tool_call>'
        )
    )
    agent._execute_tool = MagicMock(
        return_value="Error executing saguaro_query: index missing"
    )
    agent._record_tool_result_message = MagicMock()
    agent._publish_progress = MagicMock()
    agent._post_shared_finding = MagicMock()
    agent._consume_master_guidance = MagicMock(return_value="")

    result = agent._isolated_inference("Find auth flow")

    trace = result["stats"]["tool_trace"][0]
    assert trace["failure_tag"] == "saguaro_failure"
    assert trace["fallback_mode"] == "fallback_static_scan"
    assert result["stats"]["fallback_mode"] == "fallback_static_scan"
    assert result["evidence_envelope"]["fallback_mode"] == "fallback_static_scan"
    assert result["evidence_envelope"]["saguaro_failures"]


def test_run_payload_includes_evidence_envelope_defaults_and_prompt_metadata():
    tool_names = sorted(
        set(UNIVERSAL_SPECIALIST_RESEARCH_TOOLS)
        | {"saguaro_query", "grep_search", "find_by_name"}
    )
    agent = _make_agent(tool_names)
    agent._isolated_inference = MagicMock(
        return_value={
            "response": "Done.",
            "stats": {"steps": 1},
            "files_read": [],
            "latent": {},
            "latent_state": None,
            "latent_tool_signals": [],
            "latent_reinjections": 0,
        }
    )

    payload = agent.run(
        prompt_profile="sovereign_build",
        specialist_prompt_key="research",
    )

    assert payload["prompt_profile"] == "sovereign_build"
    assert payload["specialist_prompt_key"] == "research"
    assert payload["evidence_envelope"]["schema_version"] == "phase1"
