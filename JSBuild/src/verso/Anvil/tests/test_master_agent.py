from unittest.mock import patch

from agents.master import MasterAgent
from agents.recovery import RecoveryManager


def test_recovery_manager_sanitize_json():
    rm = RecoveryManager(None)
    raw = 'Here is the list: ```json\n[{"id": 1}]\n```'
    data = rm.sanitize_json(raw)
    assert isinstance(data, list)
    assert data[0]["id"] == 1


def test_master_agent_init():
    with patch("agents.master.DeterministicOllama"), patch(
        "agents.master.SaguaroSubstrate"
    ):
        agent = MasterAgent(use_unified_adapter=False)
    assert agent.brain is not None
    assert agent.recovery is not None


def test_master_agent_decomposition(mocker):
    mocker.patch("agents.master.DeterministicOllama")
    mocker.patch("agents.master.SaguaroSubstrate")
    agent = MasterAgent(use_unified_adapter=False)
    mock_gen = mocker.patch.object(agent.brain, "generate")
    mock_gen.return_value = (
        '```json\n[{"id": 1, "role": "researcher", "task": "test"}]\n```'
    )

    # Mock sub-agent dispatch to avoid full execution
    mocker.patch.object(agent, "dispatch_subagent", return_value="done")
    mocker.patch.object(agent.quality_gate, "evaluate", return_value={"accepted": True})

    # Mock review to pass immediately
    mock_gen.side_effect = [mock_gen.return_value, "YES"]  # Decompression  # Review

    agent.run_mission("test objective")
    assert len(agent.context_memory) == 1
    assert agent.context_memory[0]["task"] == "test"
    assert agent.context_memory[0]["result"] == "done"
