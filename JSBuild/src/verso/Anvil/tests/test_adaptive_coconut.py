from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

from config.settings import ORCHESTRATION_CONFIG
from core.multi_agent_gatherer import MultiAgentEvidenceGatherer


class _BrainStub:
    def stream_chat(self, _messages, max_tokens=0, temperature=0.0):
        _ = (max_tokens, temperature)
        yield "analysis"

    def embeddings(self, text: str):
        seed = max(1, len(text) % 17)
        return np.arange(16, dtype=np.float32) * float(seed)


class _RegistryStub:
    def dispatch(self, _name: str, _args: dict):
        return ""


def _make_gatherer() -> MultiAgentEvidenceGatherer:
    return MultiAgentEvidenceGatherer(
        brain=_BrainStub(),
        console=Console(record=True),
        registry=_RegistryStub(),
    )


def test_resolve_gathering_guidance_from_profile():
    gatherer = _make_gatherer()
    profile = SimpleNamespace(subagent_count=3, recommended_context_budget=120000)
    guidance = gatherer._resolve_gathering_guidance(profile)

    assert guidance["max_agents"] == 3
    assert guidance["agent_context_limit"] == 40000


def test_gather_evidence_uses_resolved_guidance(monkeypatch):
    gatherer = _make_gatherer()
    captured = {"chunk_limit": None, "analyze_limits": []}

    monkeypatch.setitem(ORCHESTRATION_CONFIG, "execution_mode", "sequential")
    monkeypatch.setattr(
        gatherer,
        "_resolve_gathering_guidance",
        lambda _profile: {"max_agents": 1, "agent_context_limit": 32100},
    )

    def _fake_chunk(_files, context_limit=None):
        captured["chunk_limit"] = context_limit
        return [["a.py"], ["b.py"]]

    def _fake_analyze(_query, _group, _idx, context_limit=None, latent_depth=1):
        captured["analyze_limits"].append(context_limit)
        _ = latent_depth
        return {
            "summary": "ok",
            "files_loaded": 0,
            "file_contents": {},
            "tokens_used": 0,
        }

    def _fake_aggregate(_query, agent_results, complexity_profile=None):
        _ = complexity_profile
        return {
            "agent_summaries": [r["summary"] for r in agent_results],
            "files_analyzed": 0,
            "file_contents": {},
            "all_results": agent_results,
        }

    monkeypatch.setattr(gatherer, "_chunk_files_by_tokens", _fake_chunk)
    monkeypatch.setattr(gatherer, "_analyze_file_group", _fake_analyze)
    monkeypatch.setattr(gatherer, "_coconut_aggregate", _fake_aggregate)

    result = gatherer.gather_evidence(
        query="inspect flow",
        candidate_files=["a.py", "b.py"],
        complexity_profile=SimpleNamespace(),
    )

    assert captured["chunk_limit"] == 32100
    assert captured["analyze_limits"] == [32100]
    assert result["adaptive_allocation"]["max_agents"] == 1


def test_coconut_aggregate_requires_native_bridge(monkeypatch):
    gatherer = _make_gatherer()

    class _BrokenBridge:
        def __init__(self):
            raise RuntimeError("native unavailable")

    native_module = types.ModuleType("core.native.coconut_bridge")
    native_module.CoconutNativeBridge = _BrokenBridge

    monkeypatch.setitem(sys.modules, "core.native.coconut_bridge", native_module)

    agent_results = [
        {
            "summary": "auth manager in core/auth.py",
            "file_contents": {"core/auth.py": "class AuthManager: pass"},
            "files_loaded": 1,
            "tokens_used": 50,
        },
        {
            "summary": "router in api/routes.py",
            "file_contents": {"api/routes.py": "def route(): pass"},
            "files_loaded": 1,
            "tokens_used": 40,
        },
    ]

    with pytest.raises(
        RuntimeError,
        match="Native COCONUT aggregation failed; Python fallback path is disabled",
    ):
        gatherer._coconut_aggregate(
            query="find authentication flow",
            agent_results=agent_results,
            complexity_profile=SimpleNamespace(coconut_paths=4, coconut_steps=3),
        )


def test_analyze_file_group_emits_latent_payload():
    gatherer = _make_gatherer()
    result = gatherer._analyze_file_group(
        query="inspect auth flow",
        files=[],
        agent_id=0,
        context_limit=1000,
        latent_depth=5,
    )

    assert isinstance(result.get("latent"), dict)
    assert result["latent"]["depth_used"] == 5
    assert result["latent"]["state_dim"] > 0
    assert isinstance(result.get("latent_state"), list)
    assert result.get("latent_reinjections") == 0
