from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli.repl import AgentREPL


def test_repl_startup_bootstraps_runtime_tuning_on_first_run() -> None:
    with patch("core.agent.DeterministicOllama"), patch(
        "core.agent.ConversationHistory"
    ), patch("cli.repl.SaguaroSubstrate"), patch("cli.repl.PromptSession"):
        repl = AgentREPL()

    repl.console = MagicMock()
    repl.console.status.return_value.__enter__.return_value = None
    repl.console.status.return_value.__exit__.return_value = None
    repl.registry = MagicMock()
    repl.semantic_engine = SimpleNamespace(_indexed=False)
    repl.brain = SimpleNamespace(model_name="granite4:tiny-h")

    with patch(
        "cli.repl.ensure_runtime_affinity", return_value={"expanded": False}
    ), patch(
        "cli.repl.assess_runtime_tuning",
        return_value={
            "status": "calibration_required",
            "models": ["granite4:tiny-h"],
            "refresh_models": ["granite4:tiny-h"],
            "host_fingerprint": "host123",
            "required_visible_threads": 16,
            "ready": False,
            "tuning_state": "stale",
            "admission_decision": "probe",
            "invocation_source": "repl_startup",
        },
    ), patch(
        "cli.repl.bootstrap_runtime_tuning"
    ) as bootstrap, patch(
        "cli.repl.has_benchmark_evidence",
        return_value=False,
    ), patch(
        "cli.repl.resolve_runtime_tuning_bootstrap_policy",
        return_value="on_first_run",
    ), patch(
        "cli.repl.EnvironmentManager"
    ) as env_manager_cls:
        env_manager_cls.return_value.ensure_ready.return_value = {"mode": "venv"}
        repl.ensure_environment_ready()

    bootstrap.assert_called_once()
    assert bootstrap.call_args.kwargs["invocation_source"] == "repl_startup"
    assert repl.semantic_engine._indexed is True
