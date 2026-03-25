import json
from unittest.mock import MagicMock, patch

from core.agents.specialists.registry import RoutingDecision
from tools.subagent_tool import ExecuteSubagentTaskTool


@patch("core.agents.specialists.build_specialist_subagent")
@patch("core.agents.specialists.route_specialist")
def test_execute_subagent_task_uses_routing_and_returns_payload(
    mock_route_specialist, mock_build_specialist
):
    fake_agent = MagicMock()
    fake_agent.name = "Master"
    fake_agent.brain = MagicMock()
    fake_agent.console = MagicMock()
    fake_agent.message_bus = MagicMock()
    fake_agent.ownership_registry = MagicMock()

    fake_subagent = MagicMock()
    fake_subagent.run.return_value = {
        "summary": "Implemented feature and tests.",
        "full_response": "Detailed response body",
        "files_read": ["core/feature.py", "tests/test_feature.py"],
        "latent": {"state_dim": 4},
        "evidence_envelope": {"schema_version": "phase1"},
    }
    mock_build_specialist.return_value = fake_subagent
    mock_route_specialist.return_value = RoutingDecision(
        primary_role="ImplementationEngineerSubagent",
        reviewer_roles=["TestAuditSubagent"],
        reasons=["requested_role_explicit"],
    )

    tool = ExecuteSubagentTaskTool(fake_agent)
    output = tool.execute(
        role="implementer",
        task="Implement feature X and cover with tests",
        aal="AAL-2",
        domains=["code"],
        compliance={"trace_id": "trace-1"},
    )

    payload = json.loads(output)
    assert payload["requested_role"] == "implementer"
    assert payload["role"] == "ImplementationEngineerSubagent"
    assert payload["reviewer_roles"] == ["TestAuditSubagent"]
    assert payload["subagent_analysis"] == "Implemented feature and tests."
    assert "core/feature.py" in payload["codebase_files"]
