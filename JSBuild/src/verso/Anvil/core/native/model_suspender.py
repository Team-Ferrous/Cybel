"""
Model Memory Suspender — Pause/Resume for CPU-constrained environments.

On CPU-only systems the inference model (1-5+ GB) holds RAM that tests,
builds, and other heavy subprocesses need.  ``ModelSuspender`` is a
context-manager that:

1. Snapshots the current engine configuration.
2. Releases all model-weight memory (Llama C object, weight caches, QSG
   adapter).
3. Forces a GC sweep so the OS can reclaim pages.
4. On exit, reloads the model from the same checkpoint — restoring the
   agent's ability to generate text.

Usage
-----
::

    from core.native.model_suspender import ModelSuspender

    with ModelSuspender(brain):
        subprocess.run(["pytest", "tests/", "-v"])

Multi-agent safety
------------------
A threading lock prevents two concurrent suspends on the same engine.
Bus events (``model.suspending`` / ``model.resumed``) let other agents
coordinate.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Global lock per model path to prevent concurrent suspend/resume ──────────
_SUSPEND_LOCKS: Dict[str, threading.RLock] = {}
_GLOBAL_LOCK = threading.Lock()


def _lock_for(model_path: str) -> threading.RLock:
    """Return (or create) a per-model reentrant lock."""
    with _GLOBAL_LOCK:
        if model_path not in _SUSPEND_LOCKS:
            _SUSPEND_LOCKS[model_path] = threading.RLock()
        return _SUSPEND_LOCKS[model_path]


# ── Memory helpers ───────────────────────────────────────────────────────────

def available_memory_gb() -> float:
    """Return available system RAM in GB (best-effort)."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        # Fallback: read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
        except Exception:
            pass
    return float("inf")  # if unknown, assume plenty


def rss_mb() -> float:
    """Return current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
    return 0.0


def should_suspend_model(
    estimated_need_gb: float = 2.0,
    safety_margin_gb: float = 1.0,
) -> bool:
    """Heuristic: suspend if available RAM minus safety margin is too tight."""
    avail = available_memory_gb()
    return avail < (estimated_need_gb + safety_margin_gb)


# ── Engine config snapshot ───────────────────────────────────────────────────

@dataclass
class _EngineSnapshot:
    """Everything needed to reconstruct the engine after eviction."""
    model_path: str
    context_length: int = 400_000
    use_mmap: bool = True
    embedding: bool = False


# ── ModelSuspender ───────────────────────────────────────────────────────────

class ModelSuspender:
    """Context manager that evicts model weights and reloads on exit.

    Parameters
    ----------
    brain : DeterministicOllama
        The shared brain instance whose underlying engine will be
        suspended.
    reason : str
        Logged reason for the suspend (e.g. ``"pytest_run"``).
    gc_collect : bool
        Whether to call ``gc.collect()`` after eviction.
    message_bus : optional
        If provided, ``model.suspending`` and ``model.resumed`` events
        are broadcast so other agents can pause inference attempts.
    force : bool
        If *False* (default), the suspender checks available memory
        and skips the eviction if there's plenty of headroom.
    estimated_need_gb : float
        How much RAM the caller expects to need during the suspend
        window.  Only used when *force* is False.
    """

    def __init__(
        self,
        brain,
        *,
        reason: str = "test_execution",
        gc_collect: bool = True,
        message_bus=None,
        force: bool = True,
        estimated_need_gb: float = 2.0,
    ):
        self.brain = brain
        self.reason = reason
        self.gc_collect = gc_collect
        self.message_bus = message_bus
        self.force = force
        self.estimated_need_gb = estimated_need_gb

        self._snapshot: Optional[_EngineSnapshot] = None
        self._suspended = False
        self._rss_before: float = 0.0
        self._rss_after_evict: float = 0.0
        self._reload_duration_s: float = 0.0
        self._skipped = False

    # ── Context manager protocol ─────────────────────────────────────────

    def __enter__(self) -> "ModelSuspender":
        self._suspend()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._resume()
        return False  # never suppress exceptions

    # ── Public queries ───────────────────────────────────────────────────

    @property
    def is_suspended(self) -> bool:
        return self._suspended

    @property
    def was_skipped(self) -> bool:
        """True if the suspend was skipped because memory was plentiful."""
        return self._skipped

    @property
    def memory_freed_mb(self) -> float:
        """Approximate MB freed by eviction (0 if skipped)."""
        if self._skipped:
            return 0.0
        return max(0.0, self._rss_before - self._rss_after_evict)

    @property
    def reload_seconds(self) -> float:
        return self._reload_duration_s

    # ── Internal: suspend ────────────────────────────────────────────────

    def _suspend(self) -> None:
        if not self.force and not should_suspend_model(self.estimated_need_gb):
            logger.info(
                "[ModelSuspender] Skipping eviction — %.1f GB available, "
                "need %.1f GB",
                available_memory_gb(),
                self.estimated_need_gb,
            )
            self._skipped = True
            return

        engine = self._resolve_engine()
        if engine is None:
            logger.warning("[ModelSuspender] No engine found on brain — nothing to suspend")
            self._skipped = True
            return

        model_path = getattr(engine, "model_path", None)
        if not model_path:
            logger.warning("[ModelSuspender] Engine has no model_path — skipping")
            self._skipped = True
            return

        lock = _lock_for(str(model_path))
        if not lock.acquire(blocking=False):
            logger.warning(
                "[ModelSuspender] Another suspend is active for %s — skipping",
                model_path,
            )
            self._skipped = True
            return

        try:
            self._rss_before = rss_mb()

            # 1. Snapshot configuration
            self._snapshot = _EngineSnapshot(
                model_path=str(model_path),
                context_length=getattr(engine, "context_length", 400_000),
                use_mmap=getattr(engine, "use_mmap", True) if hasattr(engine, "use_mmap") else True,
                embedding=getattr(engine, "embedding_enabled", False),
            )

            # 2. Broadcast suspend intent
            self._broadcast("model.suspending", {
                "reason": self.reason,
                "model_path": str(model_path),
            })

            logger.info(
                "[ModelSuspender] Evicting model weights (%s) — reason: %s, "
                "RSS before: %.0f MB",
                model_path,
                self.reason,
                self._rss_before,
            )

            # 3. Evict the Llama C object (main memory consumer)
            if hasattr(engine, "llm") and engine.llm is not None:
                try:
                    # llama-cpp-python's Llama object releases mmap on __del__
                    del engine.llm
                except Exception:
                    pass
                engine.llm = None

            # 4. Release KV cache manager
            if hasattr(engine, "kv_cache_manager") and engine.kv_cache_manager is not None:
                try:
                    del engine.kv_cache_manager
                except Exception:
                    pass
                engine.kv_cache_manager = None

            # 5. Release GGUF loader on engine
            if hasattr(engine, "loader") and engine.loader is not None:
                try:
                    engine.loader.close()
                except Exception:
                    pass
                engine.loader = None

            # 6. Release QSG adapter chain (OllamaQSGAdapter → weight caches)
            self._evict_qsg_adapter()

            # 7. Release _loader_cache entries on DeterministicOllama
            self._evict_loader_cache()

            # 8. Force GC
            if self.gc_collect:
                gc.collect()

            self._rss_after_evict = rss_mb()
            freed = self._rss_before - self._rss_after_evict
            logger.info(
                "[ModelSuspender] Eviction complete — RSS after: %.0f MB "
                "(freed ≈ %.0f MB)",
                self._rss_after_evict,
                max(0, freed),
            )

            self._suspended = True
        except Exception:
            # If something went wrong during suspend, release the lock
            lock.release()
            raise

    # ── Internal: resume ─────────────────────────────────────────────────

    def _resume(self) -> None:
        if self._skipped:
            return

        if not self._suspended:
            return

        snapshot = self._snapshot
        if snapshot is None:
            return

        lock = _lock_for(snapshot.model_path)
        try:
            t0 = time.monotonic()
            logger.info(
                "[ModelSuspender] Reloading model from %s …",
                snapshot.model_path,
            )

            # Rebuild the engine from scratch
            self._reload_engine(snapshot)

            self._reload_duration_s = time.monotonic() - t0
            self._suspended = False

            # Broadcast resume
            self._broadcast("model.resumed", {
                "reason": self.reason,
                "model_path": snapshot.model_path,
                "reload_seconds": round(self._reload_duration_s, 2),
            })

            logger.info(
                "[ModelSuspender] Model reloaded in %.1f s — RSS: %.0f MB",
                self._reload_duration_s,
                rss_mb(),
            )
        finally:
            try:
                lock.release()
            except RuntimeError:
                pass  # Lock not held — shouldn't happen

    # ── Helpers ──────────────────────────────────────────────────────────

    def _resolve_engine(self):
        """Walk the brain → adapter → engine chain to find the LlamaCpp engine."""
        brain = self.brain
        if brain is None:
            return None

        # DeterministicOllama.loader is the OllamaQSGAdapter
        adapter = getattr(brain, "loader", None)
        if adapter is None:
            return None

        # OllamaQSGAdapter.native_engine is the LlamaCppInferenceEngine
        engine = getattr(adapter, "native_engine", None)
        if engine is not None:
            return engine

        # Fallback: directly on adapter
        engine = getattr(adapter, "_native_engine", None)
        return engine

    def _evict_qsg_adapter(self) -> None:
        """Release heavy objects inside the QSG adapter (weight store, etc.)."""
        adapter = getattr(self.brain, "loader", None)
        if adapter is None:
            return

        # Weight store caches
        ws = getattr(adapter, "_weight_store", None)
        if ws is not None:
            cache = getattr(ws, "_cache", None)
            if cache is not None:
                cache.clear()

        # Propagator (phase controller)
        if hasattr(adapter, "_propagator"):
            adapter._propagator = None

        # Encoder (holographic)
        if hasattr(adapter, "_encoder"):
            adapter._encoder = None

        # Lazy QSG loader
        qsg_loader = getattr(self.brain, "qsg_loader", None)
        if qsg_loader is not None:
            qsg_loader._qsg_adapter = None
            qsg_loader._initialization_attempted = False

    def _evict_loader_cache(self) -> None:
        """Clear the class-level _loader_cache on DeterministicOllama."""
        try:
            from core.ollama_client import DeterministicOllama
            model_name = getattr(self.brain, "model_name", None)
            if model_name and model_name in DeterministicOllama._loader_cache:
                del DeterministicOllama._loader_cache[model_name]
        except Exception:
            pass

    def _reload_engine(self, snapshot: _EngineSnapshot) -> None:
        """Reconstruct the full engine + adapter chain from a snapshot."""
        from core.ollama_client import DeterministicOllama

        model_name = getattr(self.brain, "model_name", None)
        if model_name is None:
            logger.error("[ModelSuspender] Cannot reload — brain has no model_name")
            return

        # Rebuild by clearing the cache and re-initializing
        # DeterministicOllama.__init__ checks _loader_cache and creates
        # a new OllamaQSGAdapter → NativeInferenceEngine chain.
        if model_name in DeterministicOllama._loader_cache:
            del DeterministicOllama._loader_cache[model_name]

        new_loader = self.brain._get_loader()
        DeterministicOllama._loader_cache[model_name] = new_loader
        self.brain.loader = new_loader

    def _broadcast(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Send an event on the message bus if available."""
        bus = self.message_bus
        if bus is None:
            return
        try:
            bus.publish(
                topic=event_type,
                message={
                    "type": event_type,
                    "payload": payload,
                    "timestamp": time.time(),
                },
                sender="ModelSuspender",
            )
        except Exception as exc:
            logger.debug("[ModelSuspender] Bus broadcast failed: %s", exc)


# ── Convenience function-based context manager ──────────────────────────────

@contextmanager
def suspend_model(
    brain,
    *,
    reason: str = "test_execution",
    gc_collect: bool = True,
    message_bus=None,
    force: bool = True,
    estimated_need_gb: float = 2.0,
):
    """Functional context manager wrapping :class:`ModelSuspender`.

    ::

        with suspend_model(brain, reason="pytest"):
            subprocess.run(["pytest", "tests/"])
    """
    suspender = ModelSuspender(
        brain,
        reason=reason,
        gc_collect=gc_collect,
        message_bus=message_bus,
        force=force,
        estimated_need_gb=estimated_need_gb,
    )
    with suspender:
        yield suspender
