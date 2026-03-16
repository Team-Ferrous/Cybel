from pathlib import Path
from types import SimpleNamespace

from core.prompts.system_prompt_builder import SystemPromptBuilder
from core.unified_chat_loop import UnifiedChatLoop


class _PolicyManager:
    def runtime_decision(self, files, aal=None):
        _ = (files, aal)
        return {"decision": "continue", "reason": "ok"}


class _RepoPresence:
    def build_prompt_context(self):
        return {
            "local_campaign_id": "cmp-1",
            "local_phase_id": "development",
            "local_claim_count": 2,
            "peer_count": 2,
            "promotable_peer_count": 1,
            "transport_provider": "filesystem",
            "trust_zone": "internal",
        }


class _DummyLoop:
    def __init__(self):
        self.saguaro = SimpleNamespace(root_dir=".")
        self.current_compliance_context = {"trace_id": "trace-1", "toolchain_state_vector": []}
        self.runtime_aal = "AAL-2"
        self.policy_manager = _PolicyManager()
        self.current_runtime_control = {}
        self.agent = SimpleNamespace(repo_presence=_RepoPresence())


def test_unified_chat_loop_injects_connectivity_context():
    loop = _DummyLoop()
    context = UnifiedChatLoop._current_prompt_contract_context(loop, ["core/a.py"])

    assert context["connectivity_context"]["peer_count"] == 2
    assert context["connectivity_context"]["transport_provider"] == "filesystem"


def test_system_prompt_builder_renders_connectivity_context():
    builder = SystemPromptBuilder(repo_root=Path("."))
    prompt = builder.build(
        task_text="Coordinate a change",
        connectivity_context={
            "local_campaign_id": "cmp-1",
            "local_phase_id": "verification",
            "local_claim_count": 3,
            "peer_count": 2,
            "promotable_peer_count": 1,
            "transport_provider": "filesystem",
            "trust_zone": "internal",
        },
    )

    assert "Connectivity Context:" in prompt
    assert "promotable_peers=1" in prompt
