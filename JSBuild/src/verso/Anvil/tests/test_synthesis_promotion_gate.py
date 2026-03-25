from __future__ import annotations

import sys
import types

safetensors = types.ModuleType("safetensors")
safetensors_numpy = types.ModuleType("safetensors.numpy")
safetensors_numpy.load_file = lambda *args, **kwargs: {}
safetensors_numpy.save_file = lambda *args, **kwargs: None
sys.modules.setdefault("safetensors", safetensors)
sys.modules.setdefault("safetensors.numpy", safetensors_numpy)

from core.unified_chat_loop import UnifiedChatLoop


def test_deterministic_synthesis_promotion_requires_full_bundle() -> None:
    loop = UnifiedChatLoop.__new__(UnifiedChatLoop)

    result = loop._evaluate_deterministic_synthesis_promotion(
        evidence={"artifacts": {"spec_path": "spec.json"}},
        execution_result={
            "deterministic_synthesis": {"spec_path": "spec.json"},
            "verification": {"passed": True},
        },
    )

    assert result["allowed"] is False
    assert "missing_replay_tape_path" in result["errors"]
    assert "missing_proof_capsule_path" in result["errors"]
    assert "missing_benchmark_summary" in result["errors"]
    assert "missing_roadmap_validation" in result["errors"]


def test_pre_finalize_checkpoint_blocks_failed_deterministic_synthesis_promotion() -> None:
    loop = UnifiedChatLoop.__new__(UnifiedChatLoop)
    recorded: list[tuple[str, str, str, dict]] = []

    loop.runtime_aal = None
    loop._classify_runtime_aal = lambda user_input, evidence: "AAL-2"
    loop._record_reality_event = (
        lambda event_type, phase, status, metadata, artifacts=None: recorded.append(
            (event_type, phase, status, metadata)
        )
    )

    result = loop._pre_finalize_governance_checkpoint(
        "synth: add bounded eqsat",
        evidence={"artifacts": {"spec_path": "spec.json"}},
        execution_result={
            "deterministic_synthesis": {"spec_path": "spec.json"},
            "verification": {"passed": True},
        },
    )

    assert result["allowed"] is False
    assert result["reason"] == "deterministic_synthesis_promotion_gate_failed"
    assert recorded[-1][2] == "blocked"
