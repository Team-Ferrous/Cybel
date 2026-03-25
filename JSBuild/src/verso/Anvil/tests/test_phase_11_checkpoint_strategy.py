from pathlib import Path


def test_phase_11_strategy_artifact_covers_cpu_first_checkpoint_requirements():
    strategy = Path("docs/cpu_first_checkpoint_strategy.md").read_text(encoding="utf-8")

    assert "# CPU-First Checkpoint Strategy" in strategy
    assert "Family C: Primary CPU-First Checkpoint Line" in strategy
    assert "CPU-first inference properties" in strategy
    assert "latent recurrence support" in strategy
    assert "evidence projection compatibility" in strategy
    assert "latent resume robustness" in strategy
    assert "core/native/model_graph.cpp" in strategy
    assert "core/native/model_graph_wrapper.py" in strategy
    assert "core/native/native_qsg_engine.py" in strategy
    assert "core/model/model_profile.py" in strategy
    assert "core/native/runtime_telemetry.py" in strategy
    assert "Required checkpoint metadata and export contract" in strategy
    assert "`n_kv_heads`" in strategy
    assert "`head_dim`" in strategy
    assert "`draft head manifest`" in strategy
    assert "`draft_head_kind`" in strategy
    assert "`latent_packet_schema_version`" in strategy
    assert "`checkpoint_hash`" in strategy
    assert "no Python or NumPy hot path dependency" in strategy

    lowered = strategy.lower()
    for term in (
        "gqa",
        "mqa",
        "mtp",
        "early-exit",
        "latent-intercept",
        "quantization-aware",
    ):
        assert term in lowered


def test_phase_11_roadmap_addendum_references_strategy_and_blockers():
    roadmap = Path("Latent_Space_QSG_Roadmap.md").read_text(encoding="utf-8")

    assert "### 26.10 Implementation addendum - 2026-03-10" in roadmap
    assert "docs/cpu_first_checkpoint_strategy.md" in roadmap
    assert "specs/phase_11_model_level_future_checkpoints_implementation.md" in roadmap
    assert "runs/roadmap/phase_11/verification_summary.md" in roadmap
    assert "trained/exported checkpoint families do not exist yet" in roadmap
    assert "Phases 6, 7, 9, and 12" in roadmap
    assert "latent recurrence" in roadmap
    assert "latent resume robustness" in roadmap
    assert "checkpoint-schema and export requirements" in roadmap


def test_phase_11_repo_surfaces_still_back_strategy_requirements():
    model_profile = Path("core/model/model_profile.py").read_text(encoding="utf-8")
    model_graph_wrapper = Path("core/native/model_graph_wrapper.py").read_text(
        encoding="utf-8"
    )
    native_engine = Path("core/native/native_qsg_engine.py").read_text(
        encoding="utf-8"
    )
    telemetry = Path("core/native/runtime_telemetry.py").read_text(encoding="utf-8")

    assert "n_kv_heads" in model_profile
    assert "gqa = bool" in model_profile

    for term in (
        "_split_fused_qkv_weight",
        "supports_execution_checkpoint",
        "supports_exit_continuation",
    ):
        assert term in model_graph_wrapper

    assert "_discover_draft_head_config" in native_engine

    for term in ("hot_path_proof", "native_backend_abi_match", "strict_native_qsg"):
        assert term in telemetry
