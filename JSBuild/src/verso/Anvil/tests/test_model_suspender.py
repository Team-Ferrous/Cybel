"""Unit tests for core.native.model_suspender."""

from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch

import pytest

from core.native.model_suspender import (
    ModelSuspender,
    _EngineSnapshot,
    _lock_for,
    available_memory_gb,
    rss_mb,
    should_suspend_model,
    suspend_model,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeLlm:
    """Simulates the llama-cpp Llama object."""
    pass


class FakeLoader:
    """Simulates GGUFModelLoader."""
    def close(self):
        pass


class FakeEngine:
    """Simulates LlamaCppInferenceEngine with realistic attributes."""

    def __init__(self, model_path="/tmp/fake_model.gguf"):
        self.model_path = model_path
        self.context_length = 4096
        self.use_mmap = True
        self.embedding_enabled = False
        self.llm = FakeLlm()
        self.kv_cache_manager = MagicMock()
        self.loader = FakeLoader()


class FakeWeightStore:
    def __init__(self):
        self._cache = {"layer_0": "tensor_data", "layer_1": "tensor_data"}


class FakeAdapter:
    """Simulates OllamaQSGAdapter."""

    def __init__(self):
        self.native_engine = FakeEngine()
        self._weight_store = FakeWeightStore()
        self._propagator = MagicMock()
        self._encoder = MagicMock()


class FakeBrain:
    """Simulates DeterministicOllama."""

    _loader_cache: dict = {}

    def __init__(self, model_name="test-model"):
        self.model_name = model_name
        self.loader = FakeAdapter()
        self.qsg_loader = None

    def _get_loader(self):
        return FakeAdapter()


# ── Memory probe tests ──────────────────────────────────────────────────────

class TestMemoryProbes:
    def test_available_memory_returns_float(self):
        result = available_memory_gb()
        assert isinstance(result, float)
        assert result >= 0

    def test_rss_mb_returns_float(self):
        result = rss_mb()
        assert isinstance(result, float)
        assert result >= 0

    def test_should_suspend_with_plenty_of_memory(self):
        with patch("core.native.model_suspender.available_memory_gb", return_value=32.0):
            assert should_suspend_model(estimated_need_gb=2.0) is False

    def test_should_suspend_with_tight_memory(self):
        with patch("core.native.model_suspender.available_memory_gb", return_value=2.0):
            assert should_suspend_model(estimated_need_gb=2.0) is True


# ── ModelSuspender core tests ────────────────────────────────────────────────

class TestModelSuspender:
    def _make_suspender(self, force=True, **kwargs) -> ModelSuspender:
        brain = FakeBrain()
        return ModelSuspender(brain, force=force, reason="test", **kwargs)

    def test_suspend_evicts_llm(self):
        """The Llama C object should be removed during suspend."""
        brain = FakeBrain()
        engine = brain.loader.native_engine
        assert engine.llm is not None

        # We need to patch the reload to avoid real model loading
        with patch.object(ModelSuspender, "_reload_engine"):
            with ModelSuspender(brain, force=True, reason="test"):
                assert engine.llm is None

    def test_suspend_clears_weight_cache(self):
        brain = FakeBrain()
        ws = brain.loader._weight_store
        assert len(ws._cache) > 0

        with patch.object(ModelSuspender, "_reload_engine"):
            with ModelSuspender(brain, force=True, reason="test"):
                assert len(ws._cache) == 0

    def test_skips_when_force_false_and_memory_plentiful(self):
        brain = FakeBrain()
        engine = brain.loader.native_engine
        original_llm = engine.llm

        with patch("core.native.model_suspender.should_suspend_model", return_value=False):
            suspender = ModelSuspender(
                brain, force=False, reason="test"
            )
            with suspender:
                # LLM should NOT have been evicted
                assert engine.llm is original_llm
            assert suspender.was_skipped

    def test_is_suspended_flag(self):
        brain = FakeBrain()
        with patch.object(ModelSuspender, "_reload_engine"):
            suspender = ModelSuspender(brain, force=True, reason="test")
            assert not suspender.is_suspended
            suspender._suspend()
            assert suspender.is_suspended
            suspender._resume()
            assert not suspender.is_suspended

    def test_memory_freed_mb_positive_when_suspended(self):
        brain = FakeBrain()
        with patch.object(ModelSuspender, "_reload_engine"):
            # rss_mb is called: (1) before eviction, (2) after eviction, (3) during resume log
            with patch("core.native.model_suspender.rss_mb", side_effect=[100.0, 50.0, 55.0]):
                suspender = ModelSuspender(brain, force=True, reason="test")
                suspender._suspend()
                assert suspender.memory_freed_mb == 50.0
                suspender._resume()

    def test_memory_freed_mb_zero_when_skipped(self):
        brain = FakeBrain()
        with patch("core.native.model_suspender.should_suspend_model", return_value=False):
            suspender = ModelSuspender(brain, force=False, reason="test")
            with suspender:
                pass
            assert suspender.memory_freed_mb == 0.0


# ── Locking tests ────────────────────────────────────────────────────────────

class TestSuspendLocking:
    def test_lock_per_model_path(self):
        lock_a = _lock_for("/path/model_a.gguf")
        lock_b = _lock_for("/path/model_b.gguf")
        assert lock_a is not lock_b
        assert _lock_for("/path/model_a.gguf") is lock_a  # Same path → same lock

    def test_concurrent_suspend_skips_second(self):
        """If one suspend is active (from another thread), a second should skip."""
        brain = FakeBrain()
        model_path = brain.loader.native_engine.model_path

        lock = _lock_for(str(model_path))
        # Acquire from a different thread to simulate concurrent suspension
        acquired = threading.Event()
        release = threading.Event()

        def hold_lock():
            lock.acquire()
            acquired.set()
            release.wait()  # Hold until told to release
            lock.release()

        t = threading.Thread(target=hold_lock, daemon=True)
        t.start()
        acquired.wait()  # Wait for the other thread to have the lock

        try:
            suspender = ModelSuspender(brain, force=True, reason="test")
            suspender._suspend()
            assert suspender.was_skipped
        finally:
            release.set()
            t.join(timeout=2)


# ── Message bus tests ────────────────────────────────────────────────────────

class TestBusEvents:
    def test_broadcasts_suspend_and_resume(self):
        brain = FakeBrain()
        bus = MagicMock()

        with patch.object(ModelSuspender, "_reload_engine"):
            with ModelSuspender(brain, force=True, reason="test", message_bus=bus):
                pass

        # Should have called publish for suspending and resumed
        calls = bus.publish.call_args_list
        topics = [call.kwargs.get("topic") or call.args[0] for call in calls]
        assert "model.suspending" in topics
        assert "model.resumed" in topics


# ── suspend_model convenience function ───────────────────────────────────────

class TestSuspendModelFunction:
    def test_context_manager_protocol(self):
        brain = FakeBrain()
        with patch.object(ModelSuspender, "_reload_engine"):
            with suspend_model(brain, reason="test", force=True) as s:
                assert isinstance(s, ModelSuspender)
                assert s.is_suspended


# ── Campaign memory_policy integration (smoke test) ─────────────────────────

_YAML_AVAILABLE = True
try:
    import yaml
except ImportError:
    _YAML_AVAILABLE = False


@pytest.mark.skipif(not _YAML_AVAILABLE, reason="yaml package not installed")
class TestCampaignMemoryPolicy:
    def test_phase_decorator_stores_memory_policy(self):
        from core.campaign.base_campaign import phase

        @phase(order=1, name="Test Phase", memory_policy="suspend_model")
        def test_phase(self):
            return {"status": "ok"}

        assert test_phase._phase_memory_policy == "suspend_model"

    def test_phase_decorator_default_keep_model(self):
        from core.campaign.base_campaign import phase

        @phase(order=1, name="Impl Phase")
        def impl_phase(self):
            return {"status": "ok"}

        assert impl_phase._phase_memory_policy == "keep_model"


# ── Manifest memory_policy integration ──────────────────────────────────────

@pytest.mark.skipif(not _YAML_AVAILABLE, reason="yaml package not installed")
class TestManifestMemoryPolicy:
    def test_phase_spec_has_memory_policy(self):
        from core.campaign.manifest import CampaignPhaseSpec

        spec = CampaignPhaseSpec(
            id="test", name="Test", memory_policy="suspend_model"
        )
        assert spec.memory_policy == "suspend_model"

    def test_phase_spec_default(self):
        from core.campaign.manifest import CampaignPhaseSpec

        spec = CampaignPhaseSpec(id="impl", name="Impl")
        assert spec.memory_policy == "keep_model"
