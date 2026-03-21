"""Unified in-process API for Saguaro inside Anvil."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import hashlib
import math
from typing import Any

from saguaro.analysis.duplicates import DuplicateAnalyzer
from saguaro.analysis.liveness import LivenessAnalyzer
from saguaro.architecture import ArchitectureAnalyzer
from saguaro.agents.sandbox import Sandbox
from saguaro.chronicle.diff import SemanticDiff
from saguaro.chronicle.storage import ChronicleStorage
from saguaro.coordination.memory import SharedMemory
from saguaro.coverage import CoverageReporter
from saguaro.defaults import get_default_yaml
from saguaro.errors import (
    SaguaroStateCorruptionError,
    SaguaroStateMismatchError,
)
from saguaro.health import HealthDashboard
from saguaro.indexing.auto_scaler import (
    get_repo_stats_and_config,
    load_runtime_profile,
)
from saguaro.indexing.stats import (
    INDEX_SCHEMA_VERSION,
    idf_for_term,
    load_index_stats,
    persist_index_stats,
)
from saguaro.indexing.tracker import IndexTracker
from saguaro.cpu.model import CPUScanner
from saguaro.math import MathEngine
from saguaro.omnigraph.store import OmniGraphStore
from saguaro.packets.builders import PacketBuilder
from saguaro.packs.base import PackManager
from saguaro.parsing.markdown import MarkdownStructureParser
from saguaro.query.corpus_rules import (
    canonicalize_rel_path,
    classify_file_role,
    is_excluded_path,
    load_corpus_patterns,
)
from saguaro.query.benchmark import load_query_calibration
from saguaro.reality.store import RealityGraphStore
from saguaro.roadmap.validator import RoadmapValidator
from saguaro.requirements.extractor import RequirementExtractor
from saguaro.requirements.traceability import TraceabilityService
from saguaro.storage.atomic_fs import (
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
)
from saguaro.storage.index_state import (
    INDEX_ARTIFACTS,
    INDEX_MANIFEST_SCHEMA_VERSION,
    artifact_paths,
    load_manifest,
    manifest_path,
    new_generation_id,
    require_manifest_ready,
    snapshot_artifact,
    validate_manifest,
)
from saguaro.storage.locks import RepoLockManager
from saguaro.storage.vector_store import VectorStore
from saguaro.utils.entity_ids import entity_identity
from saguaro.utils.file_utils import build_corpus_manifest
from saguaro.utils.float_vector import FloatVector
from saguaro.validation.engine import ValidationEngine

_EMBEDDING_SCHEMA_VERSION = INDEX_SCHEMA_VERSION
_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_MAX_EMBED_TEXT_CHARS = int(os.getenv("SAGUARO_MAX_EMBED_TEXT_CHARS", "24000"))
_EMBED_HEAD_CHARS = int(os.getenv("SAGUARO_EMBED_HEAD_CHARS", "16000"))
_EMBED_TAIL_CHARS = int(os.getenv("SAGUARO_EMBED_TAIL_CHARS", "8000"))
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def _load_backend_helpers():
    from saguaro.indexing.backends import backend_name, get_backend
    return backend_name, get_backend


def get_backend(*, prefer_tensorflow: bool = True, prefer_pyt: bool = False, prefer_jax:bool = False):
    """Compatibility wrapper for tests and legacy callers."""
    _, backend_getter = _load_backend_helpers()
    return backend_getter(prefer_tensorflow=prefer_tensorflow, prefer_pyt=prefer_pyt, prefer_jax=prefer_jax)


def backend_name(backend: Any) -> str:
    """Compatibility wrapper for tests and legacy callers."""
    backend_namer, _ = _load_backend_helpers()
    return backend_namer(backend)


class SaguaroAPI:
    """Provide SaguaroAPI support."""

    def __init__(
        self,
        repo_path: str = ".",
        *,
        saguaro_dir: str | None = None,
        extra_exclusions: list[str] | None = None,
    ) -> None:
        """Initialize the instance."""
        from saguaro.agents.perception import (
            DirectoryExplorer,
            FileReader,
            SkeletonGenerator,
            SliceGenerator,
            TracePerception,
        )
        from saguaro.indexing.native_runtime import get_native_runtime
        from saguaro.parsing.parser import SAGUAROParser
        from saguaro.services import (
            AppService,
            ComparativeAnalysisService,
            EvalService,
            EvidenceService,
            GraphService,
            MetricsService,
            ParseService,
            QueryService,
            ResearchService,
            VerifyService,
        )
        from saguaro.state import StateLedger

        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(
            saguaro_dir or os.path.join(self.repo_path, ".saguaro")
        )
        self._extra_exclusions = list(extra_exclusions or [])
        self.vectors_dir = os.path.join(self.saguaro_dir, "vectors")
        self._lock_manager = RepoLockManager(self.saguaro_dir)
        self._parser = SAGUAROParser()
        self._skeleton = SkeletonGenerator()
        self._slice = SliceGenerator(self.repo_path, saguaro_dir=self.saguaro_dir)
        self._reader = FileReader(self.repo_path)
        self._explorer = DirectoryExplorer(self.repo_path)
        self._trace = TracePerception(self.repo_path)
        prefer_tf = os.getenv("SAGUARO_PREFER_TF_BACKEND", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        prefer_pyt = os.getenv("SAGUARO_PREFER_PYT_BACKEND", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        prefer_jax = os.getenv("SAGUARO_PREFER_JAX_BACKEND", "1").lower() in {
            "1",
            "true",
            "yes",
        }
        self._prefer_tf_backend  = prefer_tf
        self._prefer_pyt_backend = prefer_pyt
        self._prefer_jax_backend = prefer_jax
        self._backend = None
        self._vocab_size = int(os.getenv("SAGUARO_EMBED_VOCAB_SIZE", "16384"))
        self._projection_cache: dict[tuple[int, int], Any] = {}
        self._native_runtime = get_native_runtime()
        self._parse_service = ParseService(self.repo_path)
        self._graph_service = GraphService(self.repo_path, self._parse_service)
        self._evidence_service = EvidenceService(self.repo_path)
        self._research_service = ResearchService(self.repo_path)
        self._metrics_service = MetricsService(self.repo_path)
        self._verify_service = VerifyService(
            self.repo_path,
            self._parse_service,
            self._graph_service,
            self._evidence_service,
        )
        self._query_service = QueryService(
            repo_path=self.repo_path,
            graph_service=self._graph_service,
            vectors_dir=self.vectors_dir,
            load_stats=self._load_stats,
            load_index_stats=self._load_index_stats,
            check_store_compatibility=self._check_store_compatibility,
            encode_text=self._encode_text,
            hybrid_rerank=self._hybrid_rerank,
            extract_terms=self._extract_terms,
            result_is_in_repo=self._result_is_in_repo,
            refresh_index=self._query_refresh_index,
        )
        self._eval_service = EvalService(
            repo_path=self.repo_path,
            graph_service=self._graph_service,
            metrics_service=self._metrics_service,
            query_runner=self.query,
            verify_runner=self.verify,
            refresh_index=self._query_refresh_index,
        )
        self._state_ledger = StateLedger(self.repo_path, saguaro_dir=self.saguaro_dir)
        self._comparative_service = ComparativeAnalysisService(
            self.repo_path,
            state_ledger=self._state_ledger,
        )
        self._app_service = AppService(
            self.repo_path,
            health_provider=self.health,
            graph_service=self._graph_service,
            evidence_service=self._evidence_service,
            research_service=self._research_service,
            metrics_service=self._metrics_service,
        )

    def _embedding_backend(self) -> Any:
        if self._backend is None:
            self._backend = get_backend(prefer_tensorflow=self._prefer_tf_backend, prefer_pytorch=self._prefer_pyt_backend, prefer_jax=self._prefer_jax_backend)
        return self._backend

    def _backend_label(self, *, require_loaded: bool = False) -> str:
        backend = self._embedding_backend() if require_loaded else self._backend
        if backend is None:
            return "deferred"
        return backend_name(backend)

    # -------------------------
    # Discovery
    # -------------------------

    def init(self, force: bool = False) -> dict[str, Any]:
        """Handle init."""
        already_exists = os.path.exists(self.saguaro_dir)
        if already_exists and not force:
            self._ensure_dirs()
            return {
                "status": "already_initialized",
                "path": self.saguaro_dir,
            }

        self._ensure_dirs()
        config_path = os.path.join(self.saguaro_dir, "config.yaml")
        atomic_write_text(config_path, get_default_yaml() + "\n")

        return {
            "status": "initialized",
            "path": self.saguaro_dir,
            "force": force,
        }

    def index(
        self,
        path: str = ".",
        force: bool = False,
        incremental: bool = True,
        changed_files: list[str] | None = None,
        events_path: str | None = None,
        prune_deleted: bool = False,
    ) -> dict[str, Any]:
        """Handle index."""
        self._ensure_ready()
        with self._exclusive_index_lock("index"):
            started_at = time.perf_counter()
            generation_id = ""
            self._append_index_journal(
                "index_started",
                path=str(path),
                force=bool(force),
                incremental=bool(incremental),
                prune_deleted=bool(prune_deleted),
                changed_files=list(changed_files or []),
                events_path=events_path,
            )
            stage_timings: dict[str, float] = {}
            index_result: dict[str, Any] | None = None
            target_path = self._resolve_path(path)
            duplicate_trees = self._duplicate_tree_state()
            stats = get_repo_stats_and_config(target_path)
            active_dim = int(stats.get("active_dim", 4096))
            total_dim = int(stats.get("total_dim", 8192))

            compatibility = dict(
                self._check_store_compatibility(expected_dim=active_dim) or {}
            )
            live_manifest = self._load_index_manifest()
            if (
                not force
                and not compatibility.get("incompatible")
                and not live_manifest
                and any(
                    os.path.exists(path) for path in self._artifact_paths().values()
                )
            ):
                bootstrap_manifest = self._build_manifest(
                    root=self.saguaro_dir,
                    generation_id=new_generation_id(),
                    status="ready",
                    stats=stats,
                    writer_operation="bootstrap_manifest",
                    committed_at=time.time(),
                )
                self._write_manifest(bootstrap_manifest)
                live_manifest = bootstrap_manifest
            if live_manifest:
                try:
                    self._validate_index_manifest(
                        manifest=live_manifest, require_ready=True
                    )
                except (
                    SaguaroStateMismatchError,
                    SaguaroStateCorruptionError,
                ) as exc:
                    compatibility["incompatible"] = True
                    compatibility["reason"] = compatibility.get("reason") or str(exc)

            effective_force = bool(force or compatibility.get("incompatible"))
            generation_id = new_generation_id()
            stage_root = self._staging_dir(generation_id)
            shutil.rmtree(stage_root, ignore_errors=True)
            os.makedirs(stage_root, exist_ok=True)
            os.makedirs(os.path.join(stage_root, "vectors"), exist_ok=True)
            os.makedirs(os.path.join(stage_root, "graph"), exist_ok=True)

            if not effective_force and live_manifest:
                self._seed_stage_from_live(stage_root)
            else:
                self._reset_vector_store(stage_root)

            exclusions = self._load_exclusions()
            manifest = build_corpus_manifest(target_path, exclusions=exclusions)
            all_files = list(manifest.files)
            explicit_files: list[str] | None = None
            explicit_deleted: list[str] = []
            event_payload = self._load_index_events(events_path)
            if event_payload:
                changed_from_events = event_payload.get("changed_files", [])
                deleted_from_events = event_payload.get("deleted_files", [])
                merged_changed = list(changed_files or []) + changed_from_events
                changed_files = merged_changed or None
                explicit_deleted.extend(deleted_from_events)
            if changed_files is not None:
                explicit_files = []
                for item in changed_files:
                    canonical = self._canonical_rel_path(item)
                    if not canonical:
                        continue
                    full = self._resolve_path(canonical)
                    if os.path.exists(full):
                        explicit_files.append(full)
                    else:
                        explicit_deleted.append(full)
            normalized_deleted: list[str] = []
            for item in explicit_deleted:
                rel = self._canonical_rel_path(item)
                if not rel:
                    continue
                full = self._resolve_path(rel)
                if full.startswith(self.repo_path):
                    normalized_deleted.append(full)
            explicit_deleted = normalized_deleted
            ledger_scan: dict[str, Any] | None = None
            if explicit_files is None and not effective_force:
                ledger_scan = self._state_ledger.compare_with_filesystem(
                    [self._canonical_rel_path(item) for item in all_files]
                )
            if explicit_files is not None:
                files_to_index = sorted(set(explicit_files))
            elif ledger_scan is not None:
                files_to_index = sorted(
                    {
                        self._resolve_path(rel)
                        for rel in list(ledger_scan.get("changed_files", []) or [])
                        if rel
                    }
                )
            elif not effective_force:
                prior_tracker = IndexTracker(os.path.join(stage_root, "tracking.json"))
                files_to_index = prior_tracker.filter_needs_indexing(all_files)
            else:
                files_to_index = all_files
            stale_files = []
            if prune_deleted:
                if ledger_scan is not None:
                    stale_files = [
                        self._resolve_path(rel)
                        for rel in list(ledger_scan.get("deleted_files", []) or [])
                        if rel
                    ]
                else:
                    prior_tracker = IndexTracker(
                        os.path.join(stage_root, "tracking.json")
                    )
                    stale_files = prior_tracker.prune_missing(all_files)

            indexed_entities = 0
            indexed_files = 0
            parsed_files: list[str] = []
            removed_files = sorted(set(stale_files + explicit_deleted))
            from saguaro.indexing.memory_optimized_engine import MemoryOptimizedIndexEngine
            from saguaro.indexing.native_coordinator import run_native_index_coordinator

            engine = MemoryOptimizedIndexEngine(self.repo_path, stage_root, stats)
            if effective_force:
                engine.tracker.clear()
            for stale_path in removed_files:
                if hasattr(engine.store, "remove_file"):
                    engine.store.remove_file(stale_path)
                engine.tracker.state.pop(stale_path, None)

            self._append_index_journal(
                "index_file_plan",
                generation_id=generation_id,
                candidate_files=len(all_files),
                files_to_index=len(files_to_index),
                removed_files=len(removed_files),
                effective_force=bool(effective_force),
                duplicate_trees=duplicate_trees,
            )

            coordinator_result: dict[str, Any] = {}
            if files_to_index:
                coordinator_started = time.perf_counter()
                coordinator_result = run_native_index_coordinator(
                    engine=engine,
                    file_paths=files_to_index,
                    remove_before_index=removed_files,
                    batch_size=max(
                        1,
                        int(os.getenv("SAGUARO_INDEX_BATCH_SIZE", "250") or 250),
                    ),
                )
                indexed_files = int(coordinator_result.get("indexed_files", 0) or 0)
                indexed_entities = int(
                    coordinator_result.get("indexed_entities", 0) or 0
                )
                parsed_files = sorted(
                    set(coordinator_result.get("touched_files", []) or [])
                )
                stage_timings["coordinator_seconds"] = round(
                    max(0.0, time.perf_counter() - coordinator_started),
                    6,
                )
            else:
                engine.commit()

            self._persist_store_schema(stage_root)

            total_indexed_files = len(engine.tracker.state)
            total_indexed_entities = len(engine.store)
            metadata_rows = list(getattr(engine.store, "_metadata", [])[: len(engine.store)])
            self._persist_index_stats(
                stats=stats,
                total_indexed_files=total_indexed_files,
                total_indexed_entities=total_indexed_entities,
                updated_files=len(files_to_index),
                indexed_files=indexed_files,
                indexed_entities=indexed_entities,
                metadata_rows=metadata_rows,
                root=stage_root,
            )
            stage_timings["stats_seconds"] = round(
                max(0.0, time.perf_counter() - started_at)
                - sum(stage_timings.values()),
                6,
            )
            graph_changed_files: list[str] | None = None
            if incremental and not effective_force:
                changed_set = sorted(set(parsed_files + removed_files))
                if changed_set:
                    graph_changed_files = changed_set
            graph_started = time.perf_counter()
            from saguaro.services import GraphService

            stage_graph_service = GraphService(
                self.repo_path,
                self._parse_service,
                saguaro_dir=stage_root,
            )
            graph = stage_graph_service.build(
                path=target_path,
                incremental=bool(incremental and not effective_force),
                changed_files=graph_changed_files,
            )
            stage_timings["graph_seconds"] = round(
                max(0.0, time.perf_counter() - graph_started),
                6,
            )

            manifest_started = time.perf_counter()
            stage_manifest = self._build_manifest(
                root=stage_root,
                generation_id=generation_id,
                status="ready",
                stats=stats,
                writer_operation="index",
                committed_at=time.time(),
            )
            self._write_manifest(stage_manifest, root=stage_root)
            self._promote_stage(stage_root)
            self._write_manifest(stage_manifest)
            self._slice._load_metadata()
            self._cleanup_staging()
            stage_timings["manifest_seconds"] = round(
                max(0.0, time.perf_counter() - manifest_started),
                6,
            )

            try:
                self._state_ledger.record_changes(
                    changed_files=parsed_files,
                    deleted_files=removed_files,
                    reason="index",
                )
            except Exception:
                pass

            index_result = {
                "status": "ok",
                "backend": self._backend_label(require_loaded=True),
                "indexed_files": indexed_files,
                "indexed_entities": indexed_entities,
                "total_indexed_files": total_indexed_files,
                "total_indexed_entities": total_indexed_entities,
                "candidate_files": len(all_files),
                "candidate_manifest_files": manifest.candidate_count,
                "excluded_manifest_files": manifest.excluded_count,
                "updated_files": len(files_to_index),
                "removed_files": len(removed_files),
                "force": effective_force,
                "rebuild_reason": (
                    compatibility.get("reason")
                    if compatibility.get("incompatible")
                    else None
                ),
                "coordinator": coordinator_result,
                "stage_timings": stage_timings,
                "duplicate_trees": duplicate_trees,
                "graph": graph,
                "watermark": self._state_ledger.watermark(),
            }
            self._append_index_journal(
                "index_completed",
                generation_id=generation_id,
                indexed_files=indexed_files,
                indexed_entities=indexed_entities,
                total_indexed_files=total_indexed_files,
                total_indexed_entities=total_indexed_entities,
                coordinator=coordinator_result,
                stage_timings=stage_timings,
                duration_seconds=round(max(0.0, time.perf_counter() - started_at), 6),
            )
            return index_result

    def health(self) -> dict[str, Any]:
        """Handle health."""
        self._ensure_ready()
        with self._shared_index_lock("health"):
            return self._health_report_unlocked()

    def coverage(
        self,
        path: str = ".",
        *,
        structural: bool = False,
        by_language: bool = False,
    ) -> dict[str, Any]:
        """Generate parser/structural coverage for the target path."""
        self._ensure_ready()
        target = self._resolve_path(path)
        return CoverageReporter(target).generate_report(
            structural=structural,
            by_language=by_language,
        )

    def doctor(self) -> dict[str, Any]:
        """Run one-shot diagnostics for index and parser state."""
        self._ensure_ready()
        with self._shared_index_lock("doctor"):
            health = self._health_report_unlocked()
            coverage = self.coverage(path=".", structural=True, by_language=True)
            tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))
            stale_candidates = 0
            if tracker.state:
                tracked_paths = sorted(tracker.state.keys())
                stale_candidates = len(tracker.filter_needs_indexing(tracked_paths))

            duplicate_trees = self._duplicate_tree_state()
            compatibility = self._check_store_compatibility(
                expected_dim=int(self._load_stats().get("active_dim", 4096))
            )
            native_abi: dict[str, Any] = {
                "ok": False,
                "reason": "native indexer unavailable",
            }
            native_capabilities: dict[str, Any] = {
                "degraded": True,
                "native_indexer": native_abi,
                "trie_ops": {"available": False, "reason": "native indexer unavailable"},
            }
            try:
                from saguaro.indexing.native_indexer_bindings import (
                    collect_native_capability_report,
                    get_native_indexer,
                )

                native_abi = get_native_indexer().abi_self_test()
                native_capabilities = collect_native_capability_report()
            except Exception as exc:
                native_abi = {"ok": False, "reason": str(exc)}
                native_capabilities = {
                    "degraded": True,
                    "native_indexer": native_abi,
                    "trie_ops": {"available": False, "reason": str(exc)},
                }
            integrity = self._integrity_report()
            doctor_ok = (
                bool(native_abi.get("ok", False))
                and not bool(compatibility.get("incompatible", False))
                and integrity.get("status") == "ready"
            )
            return {
                "status": "ok" if doctor_ok else "warning",
                "runtime": {
                    "repo_path": self.repo_path,
                    "backend": self._backend_label(require_loaded=True),
                },
                "health": health,
                "coverage": coverage,
                "index": {
                    "tracked_files": len(tracker.state),
                    "stale_candidates": stale_candidates,
                    "compatibility": compatibility,
                },
                "integrity": integrity,
                "duplicate_trees": duplicate_trees,
                "native_abi": native_abi,
                "native_capabilities": native_capabilities,
                "daemon": self.daemon(action="status"),
                "watermark": self._state_ledger.watermark(),
            }

    def _health_report_unlocked(self) -> dict[str, Any]:
        dashboard = HealthDashboard(self.saguaro_dir, repo_path=self.repo_path)
        report = dashboard.generate_report()
        try:
            from saguaro.indexing.native_indexer_bindings import (
                collect_native_capability_report,
            )

            report["native_capabilities"] = collect_native_capability_report()
        except Exception as exc:
            report["native_capabilities"] = {
                "degraded": True,
                "native_indexer": {"ok": False, "reason": str(exc)},
                "trie_ops": {"available": False, "reason": str(exc)},
            }
        report["storage_layout"] = self._storage_layout_report()
        report["query_gateway"] = {
            "status": "enabled",
            "mode": "microbatch",
            "batch_window_ms": int(
                os.getenv("SAGUARO_QUERY_BROKER_WINDOW_MS", "2") or 2
            ),
            "max_batch_size": int(
                os.getenv("SAGUARO_QUERY_BROKER_MAX_BATCH", "32") or 32
            ),
            "query_many_available": True,
        }
        report["runtime_profile"] = load_runtime_profile(self.repo_path)
        report["integrity"] = self._integrity_report()
        report["locks"] = self._lock_manager.status()
        report["corpus_sessions"] = self._state_ledger.list_corpus_sessions(
            include_expired=False
        )
        return report

    def watermark(self) -> dict[str, Any]:
        """Return latest state-ledger event watermark."""
        self._ensure_ready()
        return self._state_ledger.watermark()

    def changeset_since(self, watermark: dict[str, Any] | None) -> dict[str, Any]:
        """Return the normalized delta changeset since a prior watermark."""
        self._ensure_ready()
        return self._state_ledger.changeset_since(watermark)

    def current_workspace_id(self) -> str:
        """Return active workspace identifier."""
        self._ensure_ready()
        return self._state_ledger.current_workspace_id()

    def daemon(
        self, action: str = "status", *, interval: int = 5, lines: int = 200
    ) -> dict[str, Any]:
        """Manage long-running watch daemon for automatic indexing."""
        self._ensure_ready()
        act = str(action or "status").strip().lower()
        if act == "start":
            return self._daemon_start(interval=interval)
        if act == "stop":
            return self._daemon_stop()
        if act == "logs":
            return self._daemon_logs(lines=lines)
        return self._daemon_status()

    def workspace(
        self,
        action: str = "status",
        *,
        name: str | None = None,
        workspace_id: str | None = None,
        against: str = "main",
        description: str = "",
        switch: bool = False,
        limit: int = 200,
        label: str = "manual",
    ) -> dict[str, Any]:
        """Operate workspace ledger lifecycle."""
        self._ensure_ready()
        act = str(action or "status").strip().lower()
        if act == "create":
            return self._state_ledger.create_workspace(
                name=name or "",
                description=description,
                switch=switch,
            )
        if act == "switch":
            return self._state_ledger.switch_workspace(workspace_id or name or "")
        if act == "history":
            return self._state_ledger.workspace_history(
                workspace_id=workspace_id, limit=limit
            )
        if act == "diff":
            return self._state_ledger.workspace_diff(
                workspace_id=workspace_id,
                against=against,
                limit=limit,
            )
        if act == "snapshot":
            return self._state_ledger.snapshot(
                label=label,
                workspace_id=workspace_id,
            )
        if act == "list":
            return self._state_ledger.list_workspaces()
        return self._state_ledger.workspace_status(
            workspace_id=workspace_id, limit=limit
        )

    def corpus(
        self,
        action: str = "list",
        *,
        path: str | None = None,
        corpus_id: str | None = None,
        alias: str | None = None,
        ttl_hours: float = 24.0,
        quarantine: bool = True,
        trust_level: str = "medium",
        build_profile: str = "auto",
        include_expired: bool = False,
        rebuild: bool = False,
        batch_sizes: list[int] | None = None,
        file_batch_sizes: list[int | None] | None = None,
        iterations: int = 1,
        reuse_check: bool = True,
    ) -> dict[str, Any]:
        """Operate the isolated comparative corpus ledger."""
        self._ensure_ready()
        return self._comparative_service.corpus(
            action=action,
            path=path,
            corpus_id=corpus_id,
            alias=alias,
            ttl_hours=ttl_hours,
            quarantine=quarantine,
            trust_level=trust_level,
            build_profile=build_profile,
            include_expired=include_expired,
            rebuild=rebuild,
            batch_sizes=batch_sizes,
            file_batch_sizes=file_batch_sizes,
            iterations=iterations,
            reuse_check=reuse_check,
        )

    def sync(
        self,
        action: str = "index",
        *,
        changed_files: list[str] | None = None,
        deleted_files: list[str] | None = None,
        full: bool = False,
        reason: str = "tool_call",
        peer_id: str | None = None,
        peer_name: str | None = None,
        peer_url: str | None = None,
        auth_token: str | None = None,
        bundle_path: str | None = None,
        workspace_id: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        """Sync index freshness and/or workspace event stream."""
        self._ensure_ready()
        act = str(action or "index").strip().lower()
        if act == "serve":
            return self._state_ledger.sync_serve()
        if act == "peer-add":
            return self._state_ledger.peer_add(
                name=peer_name or "",
                url=peer_url or "",
                auth_token=auth_token,
            )
        if act == "peer-remove":
            return self._state_ledger.peer_remove(peer_id or "")
        if act == "peer-list":
            return self._state_ledger.peer_list()
        if act == "push":
            return self._state_ledger.sync_push(
                peer_id=peer_id or "",
                limit=limit,
                workspace_id=workspace_id,
            )
        if act == "pull":
            return self._state_ledger.sync_pull(
                peer_id=peer_id or "",
                bundle_path=bundle_path or "",
                workspace_id=workspace_id,
            )
        if act == "subscribe":
            return self._state_ledger.sync_subscribe(
                peer_id=peer_id or "", enabled=True
            )

        before_clock = int(self._state_ledger.watermark().get("logical_clock", 0) or 0)
        normalized_changed = self._normalize_sync_paths(changed_files or [])
        normalized_deleted = self._normalize_sync_paths(deleted_files or [])
        target_changed = normalized_changed + normalized_deleted
        indexed = self.index(
            path=".",
            force=bool(full),
            incremental=not bool(full),
            changed_files=None if full or not target_changed else target_changed,
            prune_deleted=True,
        )
        after_watermark = self._state_ledger.watermark()
        after_clock = int(after_watermark.get("logical_clock", 0) or 0)
        event_record = {
            "workspace_id": self._state_ledger.current_workspace_id(
                workspace_id=workspace_id
            ),
            "events_written": max(after_clock - before_clock, 0),
        }
        return {
            "status": "ok",
            "action": "index",
            "index": indexed,
            "events": event_record,
            "watermark": after_watermark,
        }

    # -------------------------
    # Querying
    # -------------------------

    def query(
        self,
        text: str,
        k: int = 5,
        file: str | None = None,
        level: int = 3,
        strategy: str = "hybrid",
        explain: bool = False,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
        scope: str = "global",
        dedupe_by: str = "entity",
        auto_refresh: bool = False,
        recall: str = "balanced",
        breadth: int = 24,
        score_threshold: float = 0.0,
        stale_file_bias: float = 0.0,
        cost_budget: str = "balanced",
        corpus_ids: list[str] | str | None = None,
    ) -> dict[str, Any]:
        """Handle query."""
        self._ensure_ready()
        if corpus_ids:
            selected_corpora = (
                [str(item).strip() for item in corpus_ids.split(",")]
                if isinstance(corpus_ids, str)
                else [str(item).strip() for item in list(corpus_ids or [])]
            )
            result = self._comparative_service.corpus_query(
                text,
                corpus_ids=[item for item in selected_corpora if item],
                k=k,
            )
            result["scope"] = "corpus"
            result["freshness"] = self._state_ledger.watermark()
            return result
        query_plan = {
            "recall": str(recall or "balanced"),
            "breadth": max(1, int(breadth or 24)),
            "score_threshold": float(score_threshold or 0.0),
            "stale_file_bias": float(stale_file_bias or 0.0),
            "cost_budget": str(cost_budget or "balanced"),
            "ann_search_k": max(max(1, int(k)), max(1, int(breadth or 24))),
        }
        if auto_refresh:
            query_result = self._query_service.query(
                text=text,
                k=k,
                strategy=strategy,
                explain=explain,
                aal=aal,
                domain=domain,
                repo_role=repo_role,
                auto_refresh=True,
            )
            raw_results = list(query_result.get("results", []))
            results = [item for item in raw_results if self._result_is_in_repo(item)]
            integrity = self._integrity_report()
            compatibility = self._check_store_compatibility(
                expected_dim=int(self._load_stats().get("active_dim", 4096))
            )
        else:
            with self._shared_index_lock("query"):
                integrity = self._integrity_report()
                compatibility = self._check_store_compatibility(
                    expected_dim=int(self._load_stats().get("active_dim", 4096))
                )
                if strategy in {"semantic", "hybrid"} and (
                    compatibility.get("incompatible")
                    or integrity.get("status") != "ready"
                ):
                    return {
                        "query": text,
                        "k": k,
                        "level": level,
                        "results": [],
                        "error": (
                            "Index compatibility or integrity check failed. "
                            "Run `saguaro recover` or `saguaro index --path . --force`."
                        ),
                        "reason": compatibility.get("reason")
                        or integrity.get("mismatches"),
                    }

                query_result = self._query_service.query(
                    text=text,
                    k=k,
                    strategy=strategy,
                    explain=explain,
                    aal=aal,
                    domain=domain,
                    repo_role=repo_role,
                    auto_refresh=False,
                )
                raw_results = list(query_result.get("results", []))
                results = [
                    item for item in raw_results if self._result_is_in_repo(item)
                ]

        if strategy in {"semantic", "hybrid"} and (
            compatibility.get("incompatible") or integrity.get("status") != "ready"
        ):
            return {
                "query": text,
                "k": k,
                "level": level,
                "results": [],
                "error": (
                    "Index compatibility or integrity check failed. "
                    "Run `saguaro recover` or `saguaro index --path . --force`."
                ),
                "reason": compatibility.get("reason") or integrity.get("mismatches"),
            }

        if file:
            file_abs = self._resolve_path(file)
            file_rel = os.path.relpath(file_abs, self.repo_path)
            filtered = []
            for item in results:
                item_file = item.get("file", "")
                if not item_file:
                    continue
                rel = (
                    item_file
                    if not os.path.isabs(item_file)
                    else os.path.relpath(item_file, self.repo_path)
                )
                if rel == file_rel or rel.startswith(os.path.dirname(file_rel)):
                    filtered.append(item)
            if filtered:
                results = filtered

        scope_value = str(scope or self._scope_from_level(level)).strip().lower()
        scoped_results = self._apply_scope(results, scope=scope_value)
        deduped_results = self._dedupe_results(scoped_results, dedupe_by=dedupe_by)

        for idx, item in enumerate(deduped_results, start=1):
            item["rank"] = idx
            item_file = item.get("file", "")
            if item_file and os.path.isabs(item_file):
                item["file"] = os.path.relpath(item_file, self.repo_path)
            item["scope"] = scope_value
            if explain:
                explanation = dict(item.get("explanation", {}) or {})
                explanation["query_plan"] = dict(query_plan)
                explanation["auto_refreshed"] = bool(
                    query_result.get("auto_refreshed", False)
                )
                item["explanation"] = explanation

        return {
            "query": text,
            "k": k,
            "level": level,
            "results": deduped_results,
            "strategy": strategy,
            "execution_strategy": query_result.get("execution_strategy", strategy),
            "aes_envelope": query_result.get("aes_envelope", {}),
            "candidates_considered": (
                query_result.get("semantic_candidates", 0)
                + query_result.get("lexical_candidates", 0)
                + query_result.get("graph_candidates", 0)
            ),
            "scope": scope_value,
            "dedupe_by": dedupe_by,
            "query_plan": query_plan,
            "stale_candidates": query_result.get("stale_candidates", []),
            "auto_refreshed": bool(query_result.get("auto_refreshed", False)),
            "auto_refreshed_files": query_result.get("auto_refreshed_files", []),
            "index_age_seconds": query_result.get("index_age_seconds"),
            "freshness": self._state_ledger.watermark(),
            "integrity": integrity,
        }

    def query_many(
        self,
        queries: list[str],
        *,
        k: int = 5,
        strategy: str = "hybrid",
        explain: bool = False,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
        auto_refresh: bool = False,
        scope: str = "global",
        dedupe_by: str = "entity",
        recall: str = "balanced",
        breadth: int = 24,
        score_threshold: float = 0.0,
        stale_file_bias: float = 0.0,
        cost_budget: str = "balanced",
    ) -> dict[str, dict[str, Any]]:
        """Run multiple queries under one shared index epoch."""
        self._ensure_ready()
        normalized = [str(item or "").strip() for item in queries if str(item or "").strip()]
        if not normalized:
            return {}
        if auto_refresh:
            return {
                query: self.query(
                    query,
                    k=k,
                    strategy=strategy,
                    explain=explain,
                    aal=aal,
                    domain=domain,
                    repo_role=repo_role,
                    scope=scope,
                    dedupe_by=dedupe_by,
                    auto_refresh=True,
                    recall=recall,
                    breadth=breadth,
                    score_threshold=score_threshold,
                    stale_file_bias=stale_file_bias,
                    cost_budget=cost_budget,
                )
                for query in normalized
            }

        query_plan = {
            "recall": str(recall or "balanced"),
            "breadth": max(1, int(breadth or 24)),
            "score_threshold": float(score_threshold or 0.0),
            "stale_file_bias": float(stale_file_bias or 0.0),
            "cost_budget": str(cost_budget or "balanced"),
            "ann_search_k": max(max(1, int(k)), max(1, int(breadth or 24))),
        }
        with self._shared_index_lock("query_many"):
            integrity = self._integrity_report()
            compatibility = self._check_store_compatibility(
                expected_dim=int(self._load_stats().get("active_dim", 4096))
            )
            if strategy in {"semantic", "hybrid"} and (
                compatibility.get("incompatible")
                or integrity.get("status") != "ready"
            ):
                reason = compatibility.get("reason") or integrity.get("mismatches")
                return {
                    query: {
                        "query": query,
                        "k": k,
                        "level": 3,
                        "results": [],
                        "error": (
                            "Index compatibility or integrity check failed. "
                            "Run `saguaro recover` or `saguaro index --path . --force`."
                        ),
                        "reason": reason,
                        "scope": scope,
                        "dedupe_by": dedupe_by,
                    }
                    for query in normalized
                }

            payloads = self._query_service.query_many(
                normalized,
                k=k,
                strategy=strategy,
                explain=explain,
                aal=aal,
                domain=domain,
                repo_role=repo_role,
                auto_refresh=False,
            )
            scope_value = str(scope or "global").strip().lower()
            results: dict[str, dict[str, Any]] = {}
            for query in normalized:
                query_result = dict(payloads.get(query) or {})
                raw_results = list(query_result.get("results", []))
                scoped_results = self._apply_scope(raw_results, scope=scope_value)
                deduped_results = self._dedupe_results(
                    scoped_results,
                    dedupe_by=dedupe_by,
                )
                for idx, item in enumerate(deduped_results, start=1):
                    item["rank"] = idx
                    item_file = item.get("file", "")
                    if item_file and os.path.isabs(item_file):
                        item["file"] = os.path.relpath(item_file, self.repo_path)
                    item["scope"] = scope_value
                    if explain:
                        explanation = dict(item.get("explanation", {}) or {})
                        explanation["query_plan"] = dict(query_plan)
                        item["explanation"] = explanation
                results[query] = {
                    "query": query,
                    "k": k,
                    "level": 3,
                    "results": deduped_results,
                    "strategy": query_result.get("strategy", strategy),
                    "execution_strategy": query_result.get(
                        "execution_strategy",
                        strategy,
                    ),
                    "aes_envelope": query_result.get("aes_envelope", {}),
                    "candidates_considered": (
                        query_result.get("semantic_candidates", 0)
                        + query_result.get("lexical_candidates", 0)
                        + query_result.get("graph_candidates", 0)
                    ),
                    "scope": scope_value,
                    "dedupe_by": dedupe_by,
                    "query_plan": query_plan,
                    "stale_candidates": query_result.get("stale_candidates", []),
                    "auto_refreshed": bool(
                        query_result.get("auto_refreshed", False)
                    ),
                    "auto_refreshed_files": query_result.get(
                        "auto_refreshed_files",
                        [],
                    ),
                    "index_age_seconds": query_result.get("index_age_seconds"),
                    "freshness": self._state_ledger.watermark(),
                    "integrity": integrity,
                    "compatibility": dict(
                        query_result.get("compatibility") or compatibility
                    ),
                }
            return results

    # -------------------------
    # Perception (SSAI)
    # -------------------------

    def skeleton(self, file_path: str) -> dict[str, Any]:
        """Handle skeleton."""
        full = self._resolve_path(file_path)
        result = self._skeleton.generate(full)
        if os.path.isabs(result.get("file_path", "")):
            result["file_path"] = os.path.relpath(result["file_path"], self.repo_path)
        return result

    def slice(
        self,
        symbol: str,
        depth: int = 1,
        file_path: str | None = None,
        corpus_id: str | None = None,
    ) -> dict[str, Any]:
        """Handle slice."""
        if corpus_id:
            return self._comparative_service.slice_symbol(
                symbol,
                corpus_id=corpus_id,
                depth=depth,
                file_path=file_path,
            )
        return self._slice.generate(symbol, depth=depth, preferred_file=file_path)

    def read_file(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read file."""
        return self._reader.read(file_path, start_line=start_line, end_line=end_line)

    def list_directory(
        self,
        path: str,
        recursive: bool = False,
        extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """List directory."""
        return self._explorer.list_directory(
            path, recursive=recursive, extensions=extensions
        )

    def module_structure(self, path: str) -> dict[str, Any]:
        """Handle module structure."""
        return self._explorer.module_structure(path)

    def trace(
        self,
        entry_point: str | None = None,
        *,
        query: str | None = None,
        depth: int = 20,
        max_stages: int = 128,
        include_complexity: bool = True,
    ) -> dict[str, Any]:
        """Trace execution pipeline from entrypoint or query intent."""
        self._ensure_ready()
        return self._trace.trace_pipeline(
            entry_point=entry_point,
            query=query,
            depth=depth,
            max_stages=max_stages,
            include_complexity=include_complexity,
        )

    def complexity(
        self,
        symbol: str | None = None,
        *,
        file: str | None = None,
        pipeline: str | None = None,
        depth: int = 20,
        include_flops: bool = False,
    ) -> dict[str, Any]:
        """Estimate complexity for a symbol or traced pipeline."""
        self._ensure_ready()
        if pipeline:
            return self._trace.pipeline_complexity(
                entry_point=pipeline,
                depth=depth,
            )
        if not symbol:
            return {
                "status": "error",
                "message": "Provide `symbol` or `pipeline` for complexity analysis.",
            }
        return self._trace.complexity_report(
            symbol=symbol,
            file_path=file,
            include_flops=include_flops,
        )

    def ffi(
        self,
        path: str = ".",
        *,
        limit: int = 200,
    ) -> dict[str, Any]:
        """List detected FFI boundaries."""
        self._ensure_ready()
        return self._trace.ffi_boundaries(path=path, limit=limit)

    def ffi_audit(
        self,
        path: str = ".",
        *,
        limit: int = 200,
    ) -> dict[str, Any]:
        """Compatibility alias for roadmap FFI audit command."""
        report = self.ffi(path=path, limit=limit)
        payload = dict(report)
        payload.setdefault("status", "ok")
        payload["action"] = "audit"
        return payload

    def bridge(
        self,
        path: str = ".",
        *,
        limit: int = 200,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility alias for bridge boundary reporting."""
        if symbol:
            return self.bridge_explain(symbol=symbol, path=path, limit=limit)
        payload = self.ffi_audit(path=path, limit=limit)
        payload["action"] = "bridge"
        return payload

    def bridge_explain(
        self,
        *,
        symbol: str,
        path: str = ".",
        limit: int = 200,
    ) -> dict[str, Any]:
        """Explain a specific bridge/FFI symbol across discovered boundaries."""
        needle = str(symbol or "").strip().lower()
        if not needle:
            return {
                "status": "error",
                "action": "explain",
                "message": "Provide a symbol for bridge explanation.",
            }

        report = self.ffi(path=path, limit=max(200, int(limit or 200)))
        boundaries = list(report.get("boundaries") or [])

        def _matches(row: dict[str, Any]) -> bool:
            haystack = " ".join(
                [
                    str(row.get("file") or ""),
                    str(row.get("mechanism") or ""),
                    str(row.get("snippet") or ""),
                    str(row.get("target") or ""),
                    str(row.get("shared_object") or ""),
                    str((row.get("typed_boundary") or {}).get("source_symbol") or ""),
                    str((row.get("typed_boundary") or {}).get("target_symbol") or ""),
                ]
            ).lower()
            return needle in haystack

        matches = [row for row in boundaries if _matches(row)]
        if not matches:
            return {
                "status": "missing",
                "action": "explain",
                "symbol": symbol,
                "count": 0,
                "match": None,
                "chain": {"token": None, "hop_count": 0, "hops": []},
            }

        primary = dict(matches[0])
        token = str(primary.get("shared_object") or primary.get("target") or "").strip()
        token_lower = token.lower()
        related = []
        for row in boundaries:
            row_token = str(row.get("shared_object") or row.get("target") or "").strip()
            if token_lower and row_token.lower() != token_lower:
                continue
            related.append(row)
            if len(related) >= max(2, min(64, int(limit or 200))):
                break
        if not related:
            related = [primary]

        hops = [
            {
                "file": str(item.get("file") or ""),
                "line": int(item.get("line", 0) or 0),
                "mechanism": str(item.get("mechanism") or ""),
                "target": str(item.get("target") or ""),
                "boundary_type": str(item.get("boundary_type") or ""),
                "shared_object": str(item.get("shared_object") or ""),
            }
            for item in related
        ]
        return {
            "status": "ok",
            "action": "explain",
            "symbol": symbol,
            "count": len(matches),
            "match": primary,
            "chain": {
                "token": token or None,
                "hop_count": len(hops),
                "hops": hops,
            },
        }

    def docs_parse(self, path: str = ".") -> dict[str, Any]:
        """Parse markdown docs into a structural node view."""
        root = self._resolve_path(path)
        parser = MarkdownStructureParser()
        extractor = RequirementExtractor(repo_root=self.repo_path)
        documents = []
        for file_path in extractor.discover_docs(root):
            rel_file = canonicalize_rel_path(str(file_path), repo_path=self.repo_path)
            text = file_path.read_text(encoding="utf-8")
            document = parser.parse(text, source_path=rel_file)
            nodes = [
                {
                    "kind": node.kind,
                    "title": node.title,
                    "text": node.text,
                    "line_start": node.line_start,
                    "line_end": node.line_end,
                    "section_path": list(node.section_path),
                    "language": node.language,
                }
                for node in document.walk()
            ]
            documents.append({"file": rel_file, "profile": "readme", "nodes": nodes})
        return {"status": "ok", "count": len(documents), "documents": documents}

    def docs_graph(self, path: str = ".") -> dict[str, Any]:
        """Return docs plus simple section-transition edges."""
        parsed = self.docs_parse(path=path)
        edges = []
        for document in parsed.get("documents", []):
            previous_section = None
            for node in document.get("nodes", []):
                if node.get("kind") != "section":
                    continue
                if previous_section is not None:
                    edges.append(
                        {
                            "from": previous_section["title"],
                            "to": node.get("title"),
                            "relation": "next_section",
                            "file": document["file"],
                        }
                    )
                previous_section = node
        parsed["edges"] = edges
        return parsed

    def requirements_extract(self, path: str = ".") -> dict[str, Any]:
        """Extract requirements from markdown docs."""
        bundle = RequirementExtractor(repo_root=self.repo_path).extract(path)
        return {
            "status": "ok",
            "count": len(bundle.requirements),
            "requirements": [item.to_dict() for item in bundle.requirements],
            "source_paths": list(bundle.source_paths),
            "graph_loaded": bundle.graph_loaded,
        }

    def requirements_list(self, path: str = ".") -> dict[str, Any]:
        """List compact requirement summaries."""
        payload = self.requirements_extract(path=path)
        payload["requirements"] = [
            {
                "id": item["requirement_id"],
                "file": item["source_path"],
                "statement": item["statement"],
                "strength": item["classification"]["strength"],
            }
            for item in payload.get("requirements", [])
        ]
        return payload

    def requirements_show(self, requirement_id: str, path: str = ".") -> dict[str, Any]:
        """Show one requirement by id."""
        payload = self.requirements_extract(path=path)
        for item in payload.get("requirements", []):
            if item.get("requirement_id") == requirement_id:
                return {"status": "ok", "requirement": item}
        return {"status": "missing", "requirement_id": requirement_id}

    def traceability_build(self, docs: str = ".") -> dict[str, Any]:
        """Build semantic traceability state."""
        return TraceabilityService(
            repo_root=self.repo_path,
            graph_service=self._graph_service,
        ).build(docs)

    def traceability_status(self, requirement_id: str) -> dict[str, Any]:
        """Show traceability state for one requirement."""
        return TraceabilityService(
            repo_root=self.repo_path,
            graph_service=self._graph_service,
        ).status(requirement_id)

    def traceability_diff(self) -> dict[str, Any]:
        """Diff the latest traceability snapshots."""
        return TraceabilityService(
            repo_root=self.repo_path,
            graph_service=self._graph_service,
        ).diff()

    def traceability_orphaned(self) -> dict[str, Any]:
        """List requirements without mapped implementation/test refs."""
        return TraceabilityService(
            repo_root=self.repo_path,
            graph_service=self._graph_service,
        ).orphaned()

    def math_parse(self, path: str = ".") -> dict[str, Any]:
        """Parse mathematical content from repo sources into cached MathIR-like records."""
        return MathEngine(self.repo_path).parse(path)

    def math_map(self, equation_id: str) -> dict[str, Any]:
        """Map a cached equation into omni-graph matches."""
        return MathEngine(self.repo_path).map(equation_id)

    def cpu_scan(
        self,
        path: str = ".",
        *,
        arch: str = "x86_64-avx2",
        limit: int = 20,
    ) -> dict[str, Any]:
        """Run the static CPU hotspot scan over one path."""
        return CPUScanner(self.repo_path).scan(path=path, arch=arch, limit=limit)

    def validate_docs(self, path: str = ".") -> dict[str, Any]:
        """Validate docs against repo evidence."""
        return ValidationEngine(
            self.repo_path,
            graph_service=self._graph_service,
        ).validate_docs(path)

    def validate_requirement(self, requirement_id: str) -> dict[str, Any]:
        """Validate a single requirement."""
        return ValidationEngine(
            self.repo_path,
            graph_service=self._graph_service,
        ).validate_requirement(requirement_id)

    def validate_gaps(self, path: str = ".") -> dict[str, Any]:
        """List missing or weak requirement coverage."""
        return ValidationEngine(
            self.repo_path,
            graph_service=self._graph_service,
        ).gaps(path)

    def omnigraph_build(self, path: str = ".") -> dict[str, Any]:
        """Build the omni-graph."""
        traceability = self.traceability_build(docs=path)
        return OmniGraphStore(self.repo_path, graph_service=self._graph_service).build(
            traceability_payload=traceability
        )

    def omnigraph_explain(self, requirement_id: str, path: str = ".") -> dict[str, Any]:
        """Explain one requirement in the omni-graph."""
        store = OmniGraphStore(self.repo_path, graph_service=self._graph_service)
        if not os.path.exists(store.graph_path):
            self.omnigraph_build(path=path)
        return store.explain(requirement_id)

    def omnigraph_find(self, equation: str, path: str = ".") -> dict[str, Any]:
        """Find equation/concept matches in the omni-graph."""
        store = OmniGraphStore(self.repo_path, graph_service=self._graph_service)
        if not os.path.exists(store.graph_path):
            self.omnigraph_build(path=path)
        return store.find_equation(equation)

    def omnigraph_diff(self) -> dict[str, Any]:
        """Return the current omni-graph summary."""
        return OmniGraphStore(self.repo_path, graph_service=self._graph_service).diff()

    def omnigraph_gaps(
        self, modality: str | None = None, path: str = "."
    ) -> dict[str, Any]:
        """List omni-graph requirement gaps."""
        store = OmniGraphStore(self.repo_path, graph_service=self._graph_service)
        if not os.path.exists(store.graph_path):
            self.omnigraph_build(path=path)
        return store.gaps(modality=modality)

    def packet_build(self, task: str) -> dict[str, Any]:
        """Build a mapping packet."""
        return PacketBuilder(
            self.repo_path,
            graph_service=self._graph_service,
        ).build_task_packet(task)

    def packet_review(self, packet_id: str) -> dict[str, Any]:
        """Review a packet or related node."""
        return PacketBuilder(
            self.repo_path,
            graph_service=self._graph_service,
        ).review_packet(packet_id)

    def packet_witness(self, requirement_id: str) -> dict[str, Any]:
        """Build a witness packet."""
        return PacketBuilder(
            self.repo_path,
            graph_service=self._graph_service,
        ).witness_packet(requirement_id)

    def roadmap_validate(self, path: str = ".") -> dict[str, Any]:
        """Validate a roadmap and return completion-oriented output."""
        return RoadmapValidator(
            self.repo_path,
            graph_service=self._graph_service,
        ).validate(path)

    def roadmap_graph(self, path: str = ".") -> dict[str, Any]:
        """Build a roadmap completion graph."""
        return RoadmapValidator(
            self.repo_path,
            graph_service=self._graph_service,
        ).build_graph(path=path)

    def reality_events(
        self,
        *,
        run_id: str | None = None,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Return ordered runtime events for a run."""
        return RealityGraphStore(self.repo_path).events(run_id=run_id, limit=limit)

    def reality_graph(
        self,
        *,
        run_id: str | None = None,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Build the execution reality graph for a run."""
        return RealityGraphStore(self.repo_path).build_graph(run_id=run_id, limit=limit)

    def reality_twin(
        self,
        *,
        run_id: str | None = None,
        limit: int = 500,
    ) -> dict[str, Any]:
        """Return the current twin-state summary for a run."""
        return RealityGraphStore(self.repo_path).twin_state(run_id=run_id, limit=limit)

    def reality_export(
        self,
        *,
        run_id: str,
        limit: int = 2000,
    ) -> dict[str, Any]:
        """Export one run's reality artifacts."""
        return RealityGraphStore(self.repo_path).export_run(run_id=run_id, limit=limit)

    def packs_list(self) -> dict[str, Any]:
        """List available packs."""
        return {"status": "ok", "packs": PackManager(self.repo_path).list()}

    def packs_enable(self, pack_name: str) -> dict[str, Any]:
        """Enable a pack."""
        return PackManager(self.repo_path).enable(pack_name)

    def packs_diagnose(self, path: str = ".") -> dict[str, Any]:
        """Diagnose pack matches for a repo."""
        return PackManager(self.repo_path).diagnose(path)

    def architecture_map(self, path: str = ".") -> dict[str, Any]:
        """Map repository topology, zones, and dependency crossings."""
        self._ensure_ready()
        analyzer = ArchitectureAnalyzer(self.repo_path)
        return analyzer.map(self._resolve_path(path))

    def architecture_verify(self, path: str = ".") -> dict[str, Any]:
        """Verify placement and zone dependency rules."""
        self._ensure_ready()
        analyzer = ArchitectureAnalyzer(self.repo_path)
        return analyzer.verify(self._resolve_path(path))

    def architecture_violations(self, path: str = ".") -> dict[str, Any]:
        """Compatibility view that returns architecture findings as violations."""
        report = self.architecture_verify(path=path)
        findings = list(report.get("findings") or [])
        return {
            "status": report.get("status", "ok"),
            "count": len(findings),
            "violations": findings,
            "summary": dict(report.get("summary") or {}),
            "policy": dict(report.get("policy") or {}),
        }

    def architecture_zones(self, path: str | None = None) -> dict[str, Any]:
        """Show architecture zone assignments."""
        self._ensure_ready()
        analyzer = ArchitectureAnalyzer(self.repo_path)
        return analyzer.zones(self._resolve_path(path) if path else None)

    def architecture_explain(self, path: str) -> dict[str, Any]:
        """Explain the architecture posture of a specific file."""
        self._ensure_ready()
        analyzer = ArchitectureAnalyzer(self.repo_path)
        return analyzer.explain(self._resolve_path(path))

    def duplicates(self, path: str = ".") -> dict[str, Any]:
        """Find duplicate or structurally mirrored files."""
        self._ensure_ready()
        analyzer = DuplicateAnalyzer(self.repo_path)
        return analyzer.analyze(self._resolve_path(path))

    def redundancy(
        self,
        path: str = ".",
        *,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        """Compatibility alias for duplicate analysis."""
        report = self.duplicates(path=path)
        payload = dict(report)
        payload["action"] = "redundancy"
        if symbol:
            needle = str(symbol).strip().lower()
            filtered = [
                cluster
                for cluster in list(report.get("clusters") or [])
                if needle in json.dumps(cluster, sort_keys=True).lower()
            ]
            payload["symbol"] = symbol
            payload["clusters"] = filtered
            payload["count"] = len(filtered)
        return payload

    def clones(self, path: str = ".") -> dict[str, Any]:
        """Compatibility alias for duplicate analysis."""
        return self.duplicates(path=path)

    def duplicate_clusters(self, path: str = ".") -> dict[str, Any]:
        """Compatibility alias for duplicate analysis."""
        return self.duplicates(path=path)

    def liveness(
        self,
        symbol: str | None = None,
        *,
        threshold: float = 0.5,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
    ) -> dict[str, Any]:
        """Explain symbol liveness or emit a repo-wide liveness report."""
        self._ensure_ready()
        analyzer = LivenessAnalyzer(self.repo_path)
        if symbol:
            return analyzer.explain(symbol)
        return analyzer.analyze(
            threshold=threshold,
            include_tests=include_tests,
            include_fragments=include_fragments,
            max_clusters=max_clusters,
        )

    def reachability(
        self,
        symbol: str | None = None,
        *,
        threshold: float = 0.5,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
    ) -> dict[str, Any]:
        """Compatibility alias for liveness reachability report."""
        return self.liveness(
            symbol=symbol,
            threshold=threshold,
            include_tests=include_tests,
            include_fragments=include_fragments,
            max_clusters=max_clusters,
        )

    def abi(
        self,
        action: str = "verify",
        *,
        threshold: float = 0.55,
        min_nodes: int = 4,
        min_files: int = 2,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
    ) -> dict[str, Any]:
        """Compatibility ABI command surface for roadmap parity."""
        op = str(action or "verify").strip().lower()
        if op == "orphaned":
            orphaned = self.unwired(
                threshold=threshold,
                min_nodes=min_nodes,
                min_files=min_files,
                include_tests=include_tests,
                include_fragments=include_fragments,
                max_clusters=max_clusters,
                refresh_graph=True,
            )
            return {
                "status": orphaned.get("status", "ok"),
                "action": "orphaned",
                "summary": dict(orphaned.get("summary") or {}),
                "clusters": list(orphaned.get("clusters") or []),
                "warnings": list(orphaned.get("warnings") or []),
            }

        doctor_report = self.doctor()
        return {
            "status": doctor_report.get("status", "ok"),
            "action": "verify",
            "native_abi": dict(doctor_report.get("native_abi") or {}),
            "native_capabilities": dict(
                doctor_report.get("native_capabilities") or {}
            ),
            "integrity": dict(doctor_report.get("integrity") or {}),
            "index": dict(doctor_report.get("index") or {}),
            "runtime": dict(doctor_report.get("runtime") or {}),
        }

    def low_usage(
        self,
        max_refs: int = 1,
        *,
        include_tests: bool = False,
        path: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Report reachable symbols with low static reference counts."""
        self._ensure_ready()
        with self._shared_index_lock("low_usage"):
            max_refs = max(0, int(max_refs or 0))
            report = LivenessAnalyzer(self.repo_path).analyze(
                threshold=0.0,
                include_tests=include_tests,
                max_low_usage_refs=max_refs,
                path_prefix=path,
                limit=limit,
            )
            low_usage = dict(report.get("low_usage") or {})
            candidates = list(low_usage.get("candidates") or [])
            return {
                "status": "ok",
                "graph_path": report.get("graph_path"),
                "max_refs": int(low_usage.get("max_refs", max_refs) or 0),
                "count": int(low_usage.get("count", len(candidates)) or 0),
                "returned_count": int(
                    low_usage.get("returned_count", len(candidates)) or 0
                ),
                "candidates": candidates,
                "dry_count": int(low_usage.get("dry_count", 0) or 0),
                "dry_candidates": list(low_usage.get("dry_candidates") or []),
                "areas": list(low_usage.get("areas") or []),
                "path_filter": low_usage.get("path_filter"),
                "limit": low_usage.get("limit"),
                "summary": dict(report.get("summary", {})),
            }

    # -------------------------
    # Verification and Analysis
    # -------------------------

    def verify(
        self,
        path: str = ".",
        engines: str | None = None,
        fix: bool = False,
        fix_mode: str = "safe",
        dry_run: bool = False,
        preview: bool = False,
        receipt_dir: str | None = None,
        assisted: bool = False,
        max_files: int | None = None,
        aal: list[str] | str | None = None,
        domain: list[str] | str | None = None,
        require_trace: bool = False,
        require_evidence: bool = False,
        require_valid_waivers: bool = False,
        change_manifest_path: str | None = None,
        compliance_context: dict[str, Any] | None = None,
        evidence_bundle: bool = False,
        min_parser_coverage: float | None = None,
    ) -> dict[str, Any]:
        """Handle verify."""
        from saguaro.sentinel.remediation import FixOrchestrator
        from saguaro.sentinel.verifier import SentinelVerifier

        engine_list = None
        if engines:
            if isinstance(engines, str):
                engine_list = [e.strip() for e in engines.split(",") if e.strip()]
            else:
                engine_list = list(engines)

        verifier = SentinelVerifier(repo_path=self.repo_path, engines=engine_list)
        target_path = self._resolve_path(path)
        resolved_change_manifest = (
            self._resolve_path(change_manifest_path) if change_manifest_path else None
        )
        verification_kwargs = {
            "aal": aal,
            "domain": domain,
            "require_trace": require_trace,
            "require_evidence": require_evidence,
            "require_valid_waivers": require_valid_waivers,
            "change_manifest_path": resolved_change_manifest,
            "compliance_context": compliance_context,
        }
        violations = verifier.verify_all(path_arg=target_path, **verification_kwargs)

        result = {
            "status": "pass" if not violations else "fail",
            "violations": violations,
            "fixed": 0,
            "count": len(violations),
        }
        if fix or dry_run:
            orchestrator = FixOrchestrator(
                repo_path=self.repo_path,
                verifier=verifier,
                receipt_dir=receipt_dir,
            )
            execution = orchestrator.execute(
                findings=violations,
                target_path=target_path,
                verification_kwargs=verification_kwargs,
                fix_mode=fix_mode,
                dry_run=dry_run or not fix,
                max_files=max_files,
            )
            result.update(
                {
                    "fix_mode": fix_mode,
                    "dry_run": bool(dry_run or not fix),
                    "assisted": bool(assisted),
                    "preview": bool(preview),
                    "fix_plan": execution["plan"],
                    "fix_receipts": execution["receipts"],
                    "receipt_dir": execution["receipt_dir"],
                    "normalized_findings": execution["normalized_findings"],
                    "fixed": int(execution["fixed"]),
                    "receipts_path": execution["receipts_path"],
                }
            )
            if fix and not dry_run:
                result["violations"] = execution["final_violations"]
                result["count"] = len(execution["final_violations"])
                result["status"] = (
                    "pass" if not execution["final_violations"] else "fail"
                )
        return self._verify_service.augment_result(
            result,
            evidence_bundle=evidence_bundle,
            min_parser_coverage=min_parser_coverage,
            aal=str(aal[0]) if isinstance(aal, list) and aal else str(aal or "AAL-3"),
            domain=domain,
        )

    def rollback_fix_receipts(self, receipt_dir: str) -> dict[str, Any]:
        """Handle rollback fix receipts."""
        from saguaro.sentinel.remediation import FixOrchestrator
        from saguaro.sentinel.verifier import SentinelVerifier

        verifier = SentinelVerifier(repo_path=self.repo_path, engines=None)
        orchestrator = FixOrchestrator(repo_path=self.repo_path, verifier=verifier)
        results = orchestrator.rollback(self._resolve_path(receipt_dir))
        return {"status": "ok", "results": [item.to_dict() for item in results]}

    def impact(self, path: str) -> dict[str, Any]:
        """Handle impact."""
        from saguaro.analysis.impact import ImpactAnalyzer

        with self._shared_index_lock("impact"):
            analyzer = ImpactAnalyzer(self.repo_path)
            target = self._resolve_path(path)
            return analyzer.analyze_change(target)

    def deadcode(
        self,
        threshold: float = 0.5,
        *,
        low_usage_max_refs: int = 1,
        lang: str | None = None,
        evidence: bool = False,
        runtime_observed: bool = False,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Handle deadcode."""
        with self._shared_index_lock("deadcode"):
            max_refs = max(0, int(low_usage_max_refs or 0))
            lang_filter = str(lang or "").strip().lower() or None
            from saguaro.parsing.parser import SAGUAROParser

            parser = SAGUAROParser()
            report = LivenessAnalyzer(self.repo_path).analyze(
                threshold=threshold,
                max_low_usage_refs=max_refs,
            )
            selected = []
            for item in report.get("candidates", []):
                classification = str(item.get("classification") or "")
                confidence = float(item.get("confidence", 0.0) or 0.0)
                if classification not in {
                    "dead_confident",
                    "dead_probable",
                    "duplicate_logic",
                }:
                    continue
                if confidence < threshold:
                    continue
                file_path = str(item.get("file") or "")
                detected_language = parser._detect_language(file_path, "")
                if lang_filter and detected_language != lang_filter:
                    continue
                name = str(item.get("name") or item.get("symbol") or "")
                candidate = {
                    "symbol": name,
                    "module": str(item.get("symbol") or ""),
                    "file": file_path,
                    "line": int(item.get("line", 0) or 0),
                    "confidence": confidence,
                    "reason": str(item.get("reason") or ""),
                    "dynamic_file": False,
                    "public": not name.startswith("_"),
                    "graph_assisted": True,
                    "classification": classification,
                    "language": detected_language,
                    "evidence": dict(item.get("evidence") or {}),
                }
                if explain:
                    candidate["explanation"] = (
                        f"{candidate['reason']} Language={detected_language}; "
                        f"usage_count={int(candidate['evidence'].get('usage_count', 0) or 0)}."
                    )
                selected.append(candidate)
            return {
                "threshold": threshold,
                "count": len(selected),
                "candidates": selected,
                "low_usage": dict(report.get("low_usage") or {}),
                "low_usage_max_refs": max_refs,
                "lang": lang,
                "evidence": bool(evidence),
                "runtime_observed": bool(runtime_observed),
                "explain": bool(explain),
                "summary": dict(report.get("summary", {})),
            }

    def unwired(
        self,
        threshold: float = 0.55,
        min_nodes: int = 4,
        min_files: int = 2,
        include_tests: bool = False,
        include_fragments: bool = False,
        max_clusters: int = 20,
        refresh_graph: bool = True,
    ) -> dict[str, Any]:
        """Detect unreachable isolated feature clusters."""
        from saguaro.analysis.entry_points import EntryPointDetector
        from saguaro.analysis.unwired import UnwiredAnalyzer

        self._ensure_ready()
        with self._shared_index_lock("unwired"):
            graph_refresh = None
            if refresh_graph:
                graph_refresh = self._graph_service.build(
                    path=self.repo_path,
                    incremental=True,
                    changed_files=None,
                )
            graph_payload = self._graph_service.export()
            entry_points = EntryPointDetector(self.repo_path).detect()

            warnings: list[str] = []
            duplicate_tree = self._duplicate_tree_state()
            if duplicate_tree.get("duplicate_tree_detected"):
                warning = str(duplicate_tree.get("warning") or "").strip()
                if warning:
                    warnings.append(warning)

            analyzer = UnwiredAnalyzer(self.repo_path)
            result = analyzer.analyze(
                graph_payload=graph_payload,
                entry_points=entry_points,
                threshold=threshold,
                min_nodes=min_nodes,
                min_files=min_files,
                include_tests=include_tests,
                include_fragments=include_fragments,
                max_clusters=max_clusters,
                external_warnings=warnings,
            )
            result.setdefault("graph", {})
            result["graph"]["refresh"] = (
                {
                    "enabled": True,
                    "result": graph_refresh or {},
                }
                if refresh_graph
                else {"enabled": False}
            )
            return result

    def report(self) -> dict[str, Any]:
        """Handle report."""
        from saguaro.analysis.report import ReportGenerator

        with self._shared_index_lock("report"):
            generator = ReportGenerator(self.repo_path)
            return generator.generate()

    def compare(
        self,
        *,
        target: str = ".",
        candidates: list[str] | None = None,
        corpus_ids: list[str] | None = None,
        fleet_root: str | None = None,
        top_k: int = 10,
        ttl_hours: float = 72.0,
        reuse_only: bool = False,
        mode: str = "flight_plan",
        emit_phasepack: bool = True,
        explain_paths: bool = True,
        portfolio_top_n: int = 12,
        calibration_profile: str = "balanced",
        evidence_budget: int = 12,
        export_datatables: bool = False,
    ) -> dict[str, Any]:
        """Run comparative analysis across isolated corpora."""
        self._ensure_ready()
        return self._comparative_service.compare(
            target=target,
            candidates=candidates,
            corpus_ids=corpus_ids,
            fleet_root=fleet_root,
            top_k=top_k,
            ttl_hours=ttl_hours,
            reuse_only=reuse_only,
            mode=mode,
            emit_phasepack=emit_phasepack,
            explain_paths=explain_paths,
            portfolio_top_n=portfolio_top_n,
            calibration_profile=calibration_profile,
            evidence_budget=evidence_budget,
            export_datatables=export_datatables,
        )

    def recover(self) -> dict[str, Any]:
        """Attempt to restore a valid committed index state."""
        self._ensure_ready()
        with self._exclusive_index_lock("recover"):
            live_manifest = self._load_index_manifest()
            try:
                if live_manifest:
                    self._validate_index_manifest(
                        manifest=live_manifest, require_ready=True
                    )
                    return {
                        "status": "ok",
                        "action": "noop",
                        "message": "Committed index is already consistent.",
                        "manifest_generation_id": live_manifest.get("generation_id"),
                    }
            except (SaguaroStateCorruptionError, SaguaroStateMismatchError):
                pass

            quarantine_suffix = f".corrupt-{int(time.time())}"
            quarantined: list[str] = []
            for rel_path in list(INDEX_ARTIFACTS.values()) + ["index_manifest.json"]:
                live_path = os.path.join(self.saguaro_dir, rel_path)
                if os.path.exists(live_path):
                    quarantine_path = live_path + quarantine_suffix
                    os.replace(live_path, quarantine_path)
                    quarantined.append(quarantine_path)

            candidates: list[tuple[str, dict[str, Any]]] = []
            staging_root = os.path.join(self.saguaro_dir, "staging")
            if os.path.isdir(staging_root):
                for entry in sorted(os.listdir(staging_root), reverse=True):
                    stage_dir = os.path.join(staging_root, entry)
                    if not os.path.isdir(stage_dir):
                        continue
                    try:
                        manifest = self._load_index_manifest(stage_dir)
                        self._validate_index_manifest(
                            root=stage_dir,
                            manifest=manifest,
                            require_ready=True,
                        )
                    except Exception:
                        continue
                    candidates.append((stage_dir, manifest))

            if not candidates:
                return {
                    "status": "warning",
                    "action": "quarantined",
                    "quarantined": quarantined,
                    "message": "No intact staged generation found. Run `saguaro index --path . --force`.",
                }

            stage_dir, manifest = candidates[0]
            self._promote_stage(stage_dir)
            self._write_manifest(manifest)
            self._cleanup_staging()
            return {
                "status": "ok",
                "action": "promoted_staging_generation",
                "manifest_generation_id": manifest.get("generation_id"),
                "quarantined": quarantined,
            }

    def graph_build(
        self,
        path: str = ".",
        incremental: bool = True,
        changed_files: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build or update the repository graph."""
        self._ensure_ready()
        with self._shared_index_lock("graph_build"):
            return self._graph_service.build(
                path=self._resolve_path(path),
                incremental=incremental,
                changed_files=changed_files,
            )

    def graph_query(
        self,
        symbol: str | None = None,
        file: str | None = None,
        relation: str | None = None,
        depth: int = 1,
        limit: int = 50,
        source: str | None = None,
        target: str | None = None,
        query_path: bool = False,
        reachable_from: str | None = None,
        max_depth: int | None = None,
        expression: str | None = None,
        path_only: bool = False,
    ) -> dict[str, Any]:
        """Query the repository graph."""
        self._ensure_ready()
        with self._shared_index_lock("graph_query"):
            if expression:
                return self._graph_query_language(
                    expression=expression,
                    depth=depth,
                    limit=limit,
                    relation=relation,
                    max_depth=max_depth,
                )
            result = self._graph_service.query(
                symbol=symbol,
                file=file,
                relation=relation,
                depth=depth,
                limit=limit,
            )
            needs_advanced = bool(query_path or source or target or reachable_from)
            if not needs_advanced:
                return result

            graph_payload = self._graph_service.export()
            graph = graph_payload.get("graph") or {}
            nodes = self._graph_items(graph.get("nodes"))
            edges = self._graph_items(graph.get("edges"))
            files = self._graph_items(graph.get("files"))
        if not nodes:
            result["path_query"] = {
                "status": "missing_graph",
                "found": False,
                "message": "No graph nodes available.",
            }
            if reachable_from:
                result["reachable"] = {
                    "seed": reachable_from,
                    "count": 0,
                    "max_depth": max_depth if max_depth is not None else depth,
                }
            return result

        adjacency = self._directed_adjacency(edges, relation=relation, nodes=nodes)
        active_depth = (
            max(0, int(max_depth)) if max_depth is not None else max(0, int(depth))
        )

        if reachable_from:
            seeds = self._resolve_graph_selector_ids(
                selector=reachable_from,
                nodes=nodes,
                files=files,
            )
            reachable = self._reachable_nodes(
                seeds=seeds,
                adjacency=adjacency,
                max_depth=active_depth,
            )
            result = self._project_reachable_result(
                base=result,
                reachable=reachable,
                nodes=nodes,
                edges=edges,
                limit=limit,
                seed=reachable_from,
                max_depth=active_depth,
            )

        if query_path or source or target:
            path_info = self._graph_path_query(
                nodes=nodes,
                edges=edges,
                adjacency=adjacency,
                source=source or file or symbol,
                target=target or symbol,
                files=files,
                max_depth=active_depth if active_depth > 0 else max(1, int(depth)),
            )
            result["path_query"] = path_info
        if path_only:
            path_payload = dict(result.get("path_query") or {})
            return {
                "status": result.get("status", "ok"),
                "query_path": path_payload,
                "graph_path": result.get("graph_path"),
                "stats": result.get("stats"),
            }
        return result

    def _graph_query_language(
        self,
        *,
        expression: str,
        depth: int,
        limit: int,
        relation: str | None,
        max_depth: int | None,
    ) -> dict[str, Any]:
        query = str(expression or "").strip()
        if not query:
            return {"status": "error", "message": "Empty graph query expression."}

        lowered = query.lower()
        if "->" in query:
            left, right = query.split("->", 1)
            path_depth = max_depth if max_depth is not None else max(12, int(depth))
            return self.graph_query(
                source=left.strip(),
                target=right.strip(),
                query_path=True,
                depth=path_depth,
                limit=limit,
                relation=relation,
                max_depth=path_depth,
                path_only=True,
            )

        if lowered.startswith("path(") and query.endswith(")"):
            body = query[5:-1]
            lhs, _, rhs = body.partition(",")
            if lhs.strip() and rhs.strip():
                path_depth = max_depth if max_depth is not None else max(12, int(depth))
                return self.graph_query(
                    source=lhs.strip(),
                    target=rhs.strip(),
                    query_path=True,
                    depth=path_depth,
                    limit=limit,
                    relation=relation,
                    max_depth=path_depth,
                    path_only=True,
                )

        if lowered.startswith("ffi(") and query.endswith(")"):
            body = query[4:-1]
            host, _, guest = body.partition(",")
            host = host.strip().lower()
            guest = guest.strip().lower()
            ffi = self.ffi(limit=max(200, limit * 10))
            boundaries = list(ffi.get("boundaries") or [])
            if host or guest:
                language_pairs = {
                    "ctypes": ("python", "cpp"),
                    "cffi": ("python", "cpp"),
                    "pybind11": ("python", "cpp"),
                    "capi": ("cpp", "python"),
                    "cgo": ("go", "c"),
                    "jni": ("java", "cpp"),
                    "wasm": ("javascript", "rust"),
                    "napi": ("javascript", "cpp"),
                    "pyo3": ("rust", "python"),
                }

                filtered = []
                for item in boundaries:
                    mech = str(item.get("mechanism") or "").lower()
                    pair = language_pairs.get(mech, ("", ""))
                    if host and host not in pair:
                        continue
                    if guest and guest not in pair:
                        continue
                    filtered.append(item)
                boundaries = filtered

            return {
                "status": "ok",
                "query": query,
                "type": "ffi",
                "count": len(boundaries[:limit]),
                "boundaries": boundaries[:limit],
            }

        if lowered.startswith("touches(") and query.endswith(")"):
            needle = query[8:-1].strip().lower()
            graph_payload = self._graph_service.export().get("graph") or {}
            nodes = self._graph_items(graph_payload.get("nodes"))
            edges = self._graph_items(graph_payload.get("edges"))
            matched = []
            touched_ids: set[str] = set()
            for edge in edges.values():
                variable = str(edge.get("variable") or edge.get("data") or "").lower()
                if needle and needle in variable:
                    touched_ids.add(str(edge.get("from") or ""))
                    touched_ids.add(str(edge.get("to") or ""))
            for node_id in touched_ids:
                node = nodes.get(node_id)
                if node:
                    matched.append(node)
            matched.sort(
                key=lambda item: (
                    str(item.get("file") or ""),
                    int(item.get("line") or 0),
                    str(item.get("name") or ""),
                )
            )
            return {
                "status": "ok",
                "query": query,
                "type": "touches",
                "count": len(matched[:limit]),
                "nodes": matched[:limit],
            }

        if lowered.startswith("complexity") and ">=" in query:
            threshold = query.split(">=", 1)[1].strip()
            graph_payload = self._graph_service.export().get("graph") or {}
            nodes = self._graph_items(graph_payload.get("nodes"))
            ranked: list[dict[str, Any]] = []
            threshold_rank = self._complexity_rank(threshold)
            for node in nodes.values():
                node_type = str(node.get("type") or "")
                if node_type not in {"function", "method"}:
                    continue
                node_file = str(node.get("file") or "")
                node_symbol = str(node.get("qualified_name") or node.get("name") or "")
                if not node_file or not node_symbol:
                    continue
                report = self.complexity(symbol=node_symbol, file=node_file)
                time_complexity = str(report.get("time_complexity") or "unknown")
                if self._complexity_rank(time_complexity) >= threshold_rank:
                    ranked.append(
                        {
                            "symbol": node_symbol,
                            "file": node_file,
                            "time_complexity": time_complexity,
                            "space_complexity": report.get("space_complexity"),
                            "confidence": report.get("confidence"),
                        }
                    )
                if len(ranked) >= max(100, limit * 2):
                    break
            ranked.sort(
                key=lambda item: (
                    -self._complexity_rank(str(item.get("time_complexity") or "")),
                    str(item.get("symbol") or ""),
                )
            )
            return {
                "status": "ok",
                "query": query,
                "type": "complexity_filter",
                "count": len(ranked[:limit]),
                "results": ranked[:limit],
            }

        return {
            "status": "error",
            "message": f"Unsupported graph query expression: {query}",
            "supported": [
                "A -> B",
                "path(A, B)",
                "ffi(host, guest)",
                "touches(variable)",
                "complexity >= O(n^2)",
            ],
        }

    def graph_export(self) -> dict[str, Any]:
        """Export the repository graph."""
        self._ensure_ready()
        return self._graph_service.export()

    def disparate_relations(
        self,
        *,
        relation: str | None = None,
        limit: int = 50,
        refresh: bool = False,
    ) -> dict[str, Any]:
        """Return native disparate relations from the repository graph."""
        self._ensure_ready()
        with self._shared_index_lock("disparate_relations"):
            if refresh:
                self._graph_service.build(path=self.repo_path, incremental=False)
            return self._graph_service.disparate_relations(
                relation=relation,
                limit=limit,
            )

    def evidence_list(self, limit: int = 20) -> list[dict[str, Any]]:
        """List evidence bundles."""
        self._ensure_ready()
        return self._evidence_service.list_bundles(limit=limit)

    def research_ingest(
        self,
        source: str,
        manifest_path: str | None = None,
        records: list[dict[str, Any]] | None = None,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Ingest external research metadata into the isolated research store."""
        self._ensure_ready()
        resolved_manifest = self._resolve_path(manifest_path) if manifest_path else None
        return self._research_service.ingest(
            source=source,
            manifest_path=resolved_manifest,
            records=records,
            aal=aal,
            domain=domain,
        )

    def research_list(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent research entries."""
        self._ensure_ready()
        return self._research_service.list_entries(limit=limit)

    def metrics_list(
        self,
        limit: int = 20,
        *,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """List recent metrics and evaluation runs."""
        self._ensure_ready()
        return self._metrics_service.list_runs(limit=limit, category=category)

    def eval_run(
        self,
        suite: str,
        *,
        k: int = 5,
        limit: int = 8,
        aal: str = "AAL-3",
        domain: str | list[str] | None = None,
        repo_role: str = "analysis_local",
    ) -> dict[str, Any]:
        """Run a local evaluation suite and persist the result."""
        self._ensure_ready()
        return self._eval_service.run(
            suite=suite,
            k=k,
            limit=limit,
            aal=aal,
            domain=domain,
            repo_role=repo_role,
        )

    def app_dashboard(self) -> dict[str, Any]:
        """Build the local-app dashboard payload."""
        self._ensure_ready()
        return self._app_service.dashboard()

    # -------------------------
    # Memory
    # -------------------------

    def memory_read(self, key: str, tier: str = "working") -> dict[str, Any]:
        """Handle memory read."""
        mem = self._memory_store()
        return {
            "key": key,
            "tier": tier,
            "value": mem.read_fact(key, tier=tier),
        }

    def memory_write(
        self, key: str, value: str, tier: str = "working"
    ) -> dict[str, Any]:
        """Handle memory write."""
        mem = self._memory_store()
        mem.write_fact(key=key, value=value, tier=tier, agent_id="SaguaroAPI")
        return {
            "status": "ok",
            "key": key,
            "tier": tier,
        }

    def memory_list(self) -> dict[str, Any]:
        """Handle memory list."""
        mem = self._memory_store()
        return {
            "tiers": mem.list_facts(),
        }

    def memory_snapshot(
        self,
        *,
        campaign_id: str,
        db_path: str | None = None,
        storage_root: str | None = None,
        snapshot_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an ALMF snapshot for a campaign."""
        from core.memory.fabric import MemoryFabricSnapshotter, MemoryFabricStore

        resolved_db_path = self._resolve_almf_db_path(db_path)
        resolved_storage_root = self._resolve_almf_storage_root(storage_root)
        store = MemoryFabricStore.from_db_path(
            resolved_db_path,
            storage_root=resolved_storage_root,
        )
        return MemoryFabricSnapshotter(store).snapshot_campaign(
            campaign_id,
            snapshot_id=snapshot_id,
        )

    def memory_restore(
        self,
        *,
        snapshot_dir: str,
        campaign_id: str | None = None,
        db_path: str | None = None,
        storage_root: str | None = None,
    ) -> dict[str, Any]:
        """Restore an ALMF snapshot into the active store."""
        from core.memory.fabric import MemoryFabricSnapshotter, MemoryFabricStore

        resolved_db_path = self._resolve_almf_db_path(db_path)
        resolved_storage_root = self._resolve_almf_storage_root(storage_root)
        store = MemoryFabricStore.from_db_path(
            resolved_db_path,
            storage_root=resolved_storage_root,
        )
        return MemoryFabricSnapshotter(store).restore_campaign(
            snapshot_dir,
            target_campaign_id=campaign_id,
        )

    # -------------------------
    # Chronicle
    # -------------------------

    def chronicle_snapshot(self, description: str = "API Snapshot") -> dict[str, Any]:
        """Handle chronicle snapshot."""
        self._ensure_ready()
        storage = ChronicleStorage(
            db_path=os.path.join(self.saguaro_dir, "chronicle.db")
        )

        blob = self._current_state_blob()

        snapshot_id = storage.save_snapshot(
            hd_state_blob=blob,
            description=description,
            metadata={
                "repo": self.repo_path,
                "backend": self._backend_label(require_loaded=True),
                "timestamp": time.time(),
                "pipeline_trace": {
                    "status": "pending",
                    "stage_count": 0,
                },
            },
        )
        trace = self.trace(query="inference pipeline", depth=20, max_stages=80)
        chronicle_dir = os.path.join(self.saguaro_dir, "chronicle")
        os.makedirs(chronicle_dir, exist_ok=True)
        trace_path = os.path.join(chronicle_dir, f"pipeline_trace_{snapshot_id}.json")
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(trace, handle, indent=2)
        return {
            "status": "ok",
            "snapshot_id": snapshot_id,
            "pipeline_trace": {
                "path": trace_path,
                "status": trace.get("status"),
                "stage_count": int(trace.get("stage_count", 0) or 0),
            },
        }

    def chronicle_diff(self) -> dict[str, Any]:
        """Handle chronicle diff."""
        self._ensure_ready()
        storage = ChronicleStorage(
            db_path=os.path.join(self.saguaro_dir, "chronicle.db")
        )
        snapshots = storage.list_snapshots()
        if len(snapshots) < 2:
            return {
                "status": "insufficient_data",
                "message": "Need at least 2 snapshots for diff.",
            }

        latest = storage.get_snapshot(snapshots[0]["id"])
        previous = storage.get_snapshot(snapshots[1]["id"])
        if not latest or not previous:
            return {
                "status": "error",
                "message": "Could not load snapshots.",
            }

        drift, details = SemanticDiff.calculate_drift(
            previous.get("hd_state_blob", b""),
            latest.get("hd_state_blob", b""),
        )

        storage.log_drift(previous["id"], latest["id"], drift, details=str(details))
        pipeline_diff = None
        chronicle_dir = os.path.join(self.saguaro_dir, "chronicle")
        latest_trace = os.path.join(
            chronicle_dir, f"pipeline_trace_{latest['id']}.json"
        )
        previous_trace = os.path.join(
            chronicle_dir, f"pipeline_trace_{previous['id']}.json"
        )
        if os.path.exists(latest_trace) and os.path.exists(previous_trace):
            try:
                from saguaro.analysis.pipeline_diff import PipelineDiff

                with open(previous_trace, encoding="utf-8") as handle:
                    old_trace = json.load(handle)
                with open(latest_trace, encoding="utf-8") as handle:
                    new_trace = json.load(handle)
                pipeline_diff = PipelineDiff().diff(old_trace, new_trace)
            except Exception:
                pipeline_diff = None
        return {
            "status": "ok",
            "from_snapshot": previous["id"],
            "to_snapshot": latest["id"],
            "drift": drift,
            "details": details,
            "summary": SemanticDiff.human_readable_report(drift),
            "pipeline_diff": pipeline_diff,
        }

    # -------------------------
    # Sandbox
    # -------------------------

    def sandbox_patch(self, file: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Handle sandbox patch."""
        sb = Sandbox(self.repo_path)
        payload = dict(patch or {})
        payload.setdefault("target_file", file)
        sandbox_id = sb.apply_patch(payload)
        return {
            "status": "ok",
            "sandbox_id": sandbox_id,
        }

    def sandbox_verify(self, sandbox_id: str) -> dict[str, Any]:
        """Handle sandbox verify."""
        sb = Sandbox.get(sandbox_id, repo_path=self.repo_path)
        if not sb:
            return {
                "status": "error",
                "message": f"Sandbox {sandbox_id} not found.",
            }
        report = sb.verify()
        report["sandbox_id"] = sandbox_id
        return report

    def sandbox_commit(self, sandbox_id: str) -> dict[str, Any]:
        """Handle sandbox commit."""
        sb = Sandbox.get(sandbox_id, repo_path=self.repo_path)
        if not sb:
            return {
                "status": "error",
                "message": f"Sandbox {sandbox_id} not found.",
            }
        result = sb.commit()
        payload = dict(result or {})
        payload.setdefault("sandbox_id", sandbox_id)
        payload.setdefault("status", "ok")
        return payload

    @staticmethod
    def _graph_items(payload: Any) -> dict[str, dict[str, Any]]:
        if isinstance(payload, dict):
            return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}
        if isinstance(payload, list):
            out: dict[str, dict[str, Any]] = {}
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                node_id = str(item.get("id") or f"item_{idx}")
                out[node_id] = dict(item)
            return out
        return {}

    def _resolve_graph_selector_ids(
        self,
        *,
        selector: str | None,
        nodes: dict[str, dict[str, Any]],
        files: dict[str, dict[str, Any]],
    ) -> list[str]:
        target = str(selector or "").strip()
        if not target:
            return []

        if target in nodes:
            return [target]

        candidate_ids: list[str] = []
        rel_target = target
        if os.path.isabs(target):
            rel_target = os.path.relpath(target, self.repo_path).replace("\\", "/")
        rel_target = rel_target.replace("\\", "/")
        file_entry = files.get(rel_target)
        if file_entry:
            candidate_ids.extend(
                [str(node_id) for node_id in (file_entry.get("nodes") or [])]
            )
            if rel_target.endswith(
                (
                    ".py",
                    ".pyi",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".go",
                    ".rs",
                    ".java",
                    ".cpp",
                    ".cc",
                    ".c",
                    ".h",
                    ".hpp",
                )
            ):
                return [
                    node_id
                    for node_id in dict.fromkeys(candidate_ids)
                    if node_id in nodes
                ]

        selector_candidates = self._selector_candidates(target)
        scored: list[tuple[int, str]] = []
        explicit_test_context = any(
            token in target.lower() for token in ("test", "tests", "_test")
        )
        for node_id, node in nodes.items():
            name = str(node.get("name") or "")
            qualified = str(node.get("qualified_name") or "")
            file_name = str(node.get("file") or "").replace("\\", "/")
            node_type = str(node.get("type") or "").lower()
            haystack = " ".join(
                [
                    name,
                    qualified,
                    file_name,
                ]
            ).lower()
            score = 0
            for candidate in selector_candidates:
                lower = candidate.lower()
                if not lower:
                    continue
                if lower == qualified.lower():
                    score += 8
                elif lower == name.lower():
                    score += 6
                elif lower == file_name.lower():
                    score += 7
                elif lower in haystack:
                    score += 3
            if node_type in {"function", "method", "class"}:
                score += 4
            elif node_type == "file":
                score += 1
            elif node_type.startswith(("dfg_", "cfg_")):
                score -= 4
            elif node_type in {"external", "dependency_graph"}:
                score -= 2
            if (
                file_name
                and self._is_noisy_path(file_name)
                and not explicit_test_context
            ):
                score -= 3
            if score > 0:
                scored.append((score, node_id))

        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            top = scored[0][0]
            minimum = 7 if any(token in target for token in (".", ":")) else 3
            candidate_ids.extend(
                [
                    node_id
                    for score, node_id in scored[:64]
                    if score >= max(minimum, top - 2)
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for node_id in candidate_ids:
            if node_id in seen or node_id not in nodes:
                continue
            seen.add(node_id)
            deduped.append(node_id)
            if len(deduped) >= 24:
                break

        symbol_like_selector = any(token in target for token in (".", ":")) and not (
            "/" in target and "." in os.path.basename(target)
        )
        if symbol_like_selector:
            symbol_ids = [
                node_id
                for node_id in deduped
                if str(nodes.get(node_id, {}).get("type") or "").lower()
                in {"function", "method", "class"}
            ]
            if symbol_ids:
                return symbol_ids[:24]
        return deduped

    @staticmethod
    def _selector_candidates(selector: str) -> list[str]:
        value = str(selector or "").strip().replace("\\", "/")
        if not value:
            return []
        file_like = "/" in value and "." in os.path.basename(value)
        if file_like:
            return [value]
        out: set[str] = {value}
        normalized = value.replace(":", ".")
        out.add(normalized)
        out.add(normalized.replace("/", "."))
        snake_normalized = _CAMEL_RE.sub(r"\1_\2", normalized)
        out.add(snake_normalized)
        out.add(snake_normalized.replace("/", "."))

        if ":" in value:
            module, _, symbol = value.partition(":")
            module = module.strip()
            symbol = symbol.strip()
            if module:
                out.add(module)
                out.add(module.replace(".", "/"))
                out.add(f"{module.replace('.', '/')}.py")
                snake_module = _CAMEL_RE.sub(r"\1_\2", module).lower()
                out.add(snake_module)
                out.add(snake_module.replace(".", "/"))
                out.add(f"{snake_module.replace('.', '/')}.py")
            if symbol:
                out.add(symbol)
                parts = [part for part in symbol.split(".") if part]
                if parts:
                    out.add(parts[-1])
                    if len(parts) >= 2:
                        out.add(".".join(parts[-2:]))
        elif "." in normalized:
            parts = [part for part in normalized.split(".") if part]
            if len(parts) >= 2:
                module = ".".join(parts[:-1])
                symbol = parts[-1]
                out.add(symbol)
                out.add(module)
                out.add(module.replace(".", "/"))
                out.add(f"{module.replace('.', '/')}.py")
                snake_module = _CAMEL_RE.sub(r"\1_\2", module).lower()
                out.add(snake_module)
                out.add(snake_module.replace(".", "/"))
                out.add(f"{snake_module.replace('.', '/')}.py")
            if parts:
                out.add(parts[-1])

        return [item for item in sorted(out) if item]

    @staticmethod
    def _complexity_rank(expr: str) -> int:
        text = str(expr or "").lower().replace(" ", "")
        if not text:
            return -1
        if "2^n" in text or "exp" in text:
            return 8
        if "n^k" in text:
            return 7
        if "n^3" in text:
            return 6
        if "n^2" in text or "n²" in text:
            return 5
        if "nlogn" in text:
            return 4
        if "v+e" in text:
            return 3
        if "logn" in text:
            return 2
        if "n" in text:
            return 1
        if "1" in text:
            return 0
        return 1

    @staticmethod
    def _directed_adjacency(
        edges: dict[str, dict[str, Any]],
        *,
        relation: str | None = None,
        nodes: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        adjacency: dict[str, list[dict[str, Any]]] = {}
        seen_edges: set[tuple[str, str, str]] = set()
        module_file_nodes = (
            SaguaroAPI._module_file_node_index(nodes or {}) if nodes else {}
        )

        def add_edge(payload: dict[str, Any]) -> None:
            src = str(payload.get("from") or "")
            dst = str(payload.get("to") or "")
            rel_name = str(payload.get("relation") or "")
            if not src or not dst:
                return
            key = (src, dst, rel_name)
            if key in seen_edges:
                return
            seen_edges.add(key)
            adjacency.setdefault(src, []).append(payload)

        for edge in edges.values():
            edge_relation = str(edge.get("relation") or "")
            if relation and relation != edge_relation:
                continue
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if not src:
                continue
            projected = dict(edge)
            if dst.startswith("external::"):
                resolved = SaguaroAPI._resolve_external_node_id(
                    dst=dst,
                    module_file_nodes=module_file_nodes,
                )
                if resolved:
                    projected["to"] = resolved
            add_edge(projected)

        # Synthetic containment edges improve symbol-to-symbol path traversal.
        if nodes and (relation in {None, "contains", "declared_in"}):
            file_nodes: dict[str, str] = {}
            for node_id, node in nodes.items():
                if str(node.get("type") or "").lower() != "file":
                    continue
                rel_path = str(node.get("file") or "").replace("\\", "/")
                if rel_path:
                    file_nodes[rel_path] = node_id
            for node_id, node in nodes.items():
                node_type = str(node.get("type") or "").lower()
                if node_type == "file":
                    continue
                rel_path = str(node.get("file") or "").replace("\\", "/")
                file_node_id = file_nodes.get(rel_path)
                if not file_node_id:
                    continue
                add_edge(
                    {
                        "from": file_node_id,
                        "to": node_id,
                        "relation": "contains",
                    }
                )
                add_edge(
                    {
                        "from": node_id,
                        "to": file_node_id,
                        "relation": "declared_in",
                    }
                )
        return adjacency

    @staticmethod
    def _reachable_nodes(
        *,
        seeds: list[str],
        adjacency: dict[str, list[dict[str, Any]]],
        max_depth: int,
    ) -> set[str]:
        if not seeds:
            return set()
        reachable = set(seeds)
        queue: list[tuple[str, int]] = [(seed, 0) for seed in seeds]
        while queue:
            node_id, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for edge in adjacency.get(node_id, []):
                nxt = str(edge.get("to") or "")
                if not nxt or nxt in reachable:
                    continue
                reachable.add(nxt)
                queue.append((nxt, depth + 1))
        return reachable

    def _project_reachable_result(
        self,
        *,
        base: dict[str, Any],
        reachable: set[str],
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        limit: int,
        seed: str,
        max_depth: int,
    ) -> dict[str, Any]:
        payload = dict(base)
        if not reachable:
            payload["nodes"] = []
            payload["edges"] = []
            payload["count"] = 0
            payload["reachable"] = {
                "seed": seed,
                "count": 0,
                "max_depth": max_depth,
            }
            return payload

        selected_nodes = [
            dict(nodes[node_id])
            for node_id in reachable
            if node_id in nodes and nodes[node_id].get("file")
        ]
        selected_nodes.sort(
            key=lambda item: (
                str(item.get("file") or ""),
                int(item.get("line", 0) or 0),
                str(item.get("name") or ""),
            )
        )
        selected_nodes = selected_nodes[: max(1, int(limit))]
        node_ids = {str(item.get("id") or "") for item in selected_nodes}

        selected_edges = []
        for edge in edges.values():
            src = str(edge.get("from") or "")
            dst = str(edge.get("to") or "")
            if src in node_ids and dst in node_ids:
                selected_edges.append(dict(edge))

        payload["nodes"] = selected_nodes
        payload["edges"] = selected_edges
        payload["count"] = len(selected_nodes)
        payload["reachable"] = {
            "seed": seed,
            "count": len(selected_nodes),
            "max_depth": max_depth,
        }
        return payload

    def _graph_path_query(
        self,
        *,
        nodes: dict[str, dict[str, Any]],
        edges: dict[str, dict[str, Any]],
        adjacency: dict[str, list[dict[str, Any]]],
        source: str | None,
        target: str | None,
        files: dict[str, dict[str, Any]],
        max_depth: int,
    ) -> dict[str, Any]:
        source_ids = self._resolve_graph_selector_ids(
            selector=source,
            nodes=nodes,
            files=files,
        )
        target_ids_list = self._resolve_graph_selector_ids(
            selector=target,
            nodes=nodes,
            files=files,
        )
        target_ids = set(target_ids_list)
        normalized_source = str(source or "").replace("\\", "/")
        normalized_target = str(target or "").replace("\\", "/")
        source_symbolic = (
            source not in nodes
            and "::" not in normalized_source
            and "/" not in normalized_source
            and any(token in normalized_source for token in (".", ":"))
        )
        target_symbolic = (
            target not in nodes
            and "::" not in normalized_target
            and "/" not in normalized_target
            and any(token in normalized_target for token in (".", ":"))
        )
        if source_symbolic:
            filtered_source = [
                node_id
                for node_id in source_ids
                if str(nodes.get(node_id, {}).get("type") or "").lower()
                in {"function", "method", "class"}
            ]
            if filtered_source:
                source_ids = filtered_source
        if target_symbolic:
            filtered_target = {
                node_id
                for node_id in target_ids
                if str(nodes.get(node_id, {}).get("type") or "").lower()
                in {"function", "method", "class"}
            }
            if filtered_target:
                target_ids = filtered_target
        if not source_ids or not target_ids:
            return {
                "status": "no_match",
                "found": False,
                "source": source,
                "target": target,
                "max_depth": max_depth,
                "path_nodes": [],
                "path_edges": [],
            }

        def _shortest_path(
            *,
            active_adjacency: dict[str, list[dict[str, Any]]],
            seed_ids: list[str],
            goal_ids: set[str],
        ) -> tuple[str | None, dict[str, str | None], dict[str, dict[str, Any]]]:
            parent: dict[str, str | None] = {}
            parent_edge: dict[str, dict[str, Any]] = {}
            queue: list[tuple[str, int]] = []
            visited: set[str] = set()
            for source_id in seed_ids:
                queue.append((source_id, 0))
                visited.add(source_id)
                parent[source_id] = None

            hit: str | None = None
            while queue:
                node_id, depth = queue.pop(0)
                if node_id in goal_ids:
                    hit = node_id
                    break
                if depth >= max(1, int(max_depth)):
                    continue
                for edge in active_adjacency.get(node_id, []):
                    nxt = str(edge.get("to") or "")
                    if not nxt or nxt in visited:
                        continue
                    visited.add(nxt)
                    parent[nxt] = node_id
                    parent_edge[nxt] = edge
                    queue.append((nxt, depth + 1))
            return hit, parent, parent_edge

        hit, parent, parent_edge = _shortest_path(
            active_adjacency=adjacency,
            seed_ids=source_ids,
            goal_ids=target_ids,
        )

        def _is_symbol_selector(value: str | None) -> bool:
            raw = str(value or "").strip()
            if not raw:
                return False
            normalized = raw.replace("\\", "/")
            if raw in nodes or "::" in raw:
                return False
            if "/" in normalized:
                return False
            return not normalized.endswith(
                (
                    ".py",
                    ".pyi",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".go",
                    ".rs",
                    ".java",
                    ".cpp",
                    ".cc",
                    ".c",
                    ".h",
                    ".hpp",
                )
            )

        if not hit and (_is_symbol_selector(source) or _is_symbol_selector(target)):
            file_nodes_by_rel_path = {
                str(node.get("file") or "").replace("\\", "/"): node_id
                for node_id, node in nodes.items()
                if str(node.get("type") or "") == "file"
                and str(node.get("file") or "").strip()
            }

            def _expand_file_nodes(node_ids: list[str]) -> list[str]:
                expanded: list[str] = []
                seen: set[str] = set()
                for node_id in node_ids:
                    if node_id in nodes and node_id not in seen:
                        seen.add(node_id)
                        expanded.append(node_id)
                    node = nodes.get(node_id) or {}
                    rel_file = str(node.get("file") or "").replace("\\", "/")
                    file_node_id = file_nodes_by_rel_path.get(rel_file)
                    if file_node_id and file_node_id not in seen:
                        seen.add(file_node_id)
                        expanded.append(file_node_id)
                return expanded

            expanded_source_ids = _expand_file_nodes(source_ids)
            expanded_target_ids = set(_expand_file_nodes(target_ids_list))
            undirected_adjacency: dict[str, list[dict[str, Any]]] = {
                node_id: list(out_edges) for node_id, out_edges in adjacency.items()
            }
            for src_id, out_edges in adjacency.items():
                for edge in out_edges:
                    dst_id = str(edge.get("to") or "")
                    if not dst_id:
                        continue
                    reverse_edge = dict(edge)
                    reverse_edge["from"] = dst_id
                    reverse_edge["to"] = src_id
                    undirected_adjacency.setdefault(dst_id, []).append(reverse_edge)

            hit, parent, parent_edge = _shortest_path(
                active_adjacency=undirected_adjacency,
                seed_ids=expanded_source_ids,
                goal_ids=expanded_target_ids,
            )

        if not hit:
            return {
                "status": "ok",
                "found": False,
                "source": source,
                "target": target,
                "max_depth": max_depth,
                "path_nodes": [],
                "path_edges": [],
            }

        path_nodes: list[str] = []
        path_edges: list[dict[str, Any]] = []
        cursor: str | None = hit
        while cursor is not None:
            path_nodes.append(cursor)
            edge = parent_edge.get(cursor)
            if edge:
                path_edges.append(dict(edge))
            cursor = parent.get(cursor)
        path_nodes.reverse()
        path_edges.reverse()

        return {
            "status": "ok",
            "found": True,
            "source": source,
            "target": target,
            "max_depth": max_depth,
            "path_nodes": [
                dict(nodes[node_id]) for node_id in path_nodes if node_id in nodes
            ],
            "path_edges": path_edges,
            "length": len(path_edges),
        }

    @staticmethod
    def _module_file_node_index(
        nodes: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for node_id, node in nodes.items():
            if str(node.get("type") or "") != "file":
                continue
            rel_file = str(node.get("file") or "").replace("\\", "/")
            if not rel_file.endswith(".py"):
                continue
            module = rel_file[: -len(".py")].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            if module:
                out[module] = node_id
        return out

    @staticmethod
    def _resolve_external_node_id(
        *,
        dst: str,
        module_file_nodes: dict[str, str],
    ) -> str | None:
        if not dst.startswith("external::"):
            return None
        reference = dst[len("external::") :].strip().lstrip(".")
        if not reference:
            return None
        parts = [part for part in reference.split(".") if part]
        while parts:
            candidate = ".".join(parts)
            node_id = module_file_nodes.get(candidate)
            if node_id:
                return node_id
            parts = parts[:-1]
        return None

    @staticmethod
    def _is_noisy_path(path: str) -> bool:
        low = str(path or "").replace("\\", "/").lower()
        if not low:
            return False
        noisy_tokens = (
            "/test/",
            "/tests/",
            "/docs/",
            "/doc/",
            "/examples/",
            "/example/",
            "test_",
            "_test.",
            ".spec.",
        )
        return any(token in low for token in noisy_tokens)

    # -------------------------
    # Helpers
    # -------------------------

    def _ensure_ready(self) -> None:
        self._ensure_dirs()
        config_path = os.path.join(self.saguaro_dir, "config.yaml")
        if not os.path.exists(config_path):
            self.init(force=False)

    def _ensure_dirs(self) -> None:
        os.makedirs(self.saguaro_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)
        os.makedirs(os.path.join(self.saguaro_dir, "sandboxes"), exist_ok=True)
        os.makedirs(os.path.join(self.saguaro_dir, "state"), exist_ok=True)
        os.makedirs(os.path.join(self.saguaro_dir, "locks"), exist_ok=True)
        os.makedirs(os.path.join(self.saguaro_dir, "staging"), exist_ok=True)

    def _shared_index_lock(self, operation: str):
        return self._lock_manager.acquire(
            "index",
            mode="shared",
            operation=operation,
        )

    def _exclusive_index_lock(self, operation: str):
        return self._lock_manager.acquire(
            "index",
            mode="exclusive",
            operation=operation,
        )

    def _manifest_path(self, root: str | None = None) -> str:
        return manifest_path(root or self.saguaro_dir)

    def _artifact_paths(self, root: str | None = None) -> dict[str, str]:
        return artifact_paths(root or self.saguaro_dir)

    def _load_index_manifest(self, root: str | None = None) -> dict[str, Any]:
        return load_manifest(root or self.saguaro_dir)

    def _validate_index_manifest(
        self,
        *,
        root: str | None = None,
        manifest: dict[str, Any] | None = None,
        require_ready: bool = False,
    ) -> dict[str, Any]:
        base = root or self.saguaro_dir
        payload = manifest if manifest is not None else self._load_index_manifest(base)
        if require_ready:
            return require_manifest_ready(base, payload)
        return validate_manifest(base, payload)

    def _build_manifest(
        self,
        *,
        root: str,
        generation_id: str,
        status: str,
        stats: dict[str, Any],
        writer_operation: str,
        committed_at: float | None,
    ) -> dict[str, Any]:
        artifact_records: dict[str, Any] = {}
        for name, rel_path in INDEX_ARTIFACTS.items():
            full = os.path.join(root, rel_path)
            if not os.path.exists(full):
                raise SaguaroStateCorruptionError(
                    f"Cannot build manifest; missing artifact {rel_path}"
                )
            artifact_records[name] = snapshot_artifact(full, rel_path)

        tracking_path = os.path.join(root, "tracking.json")
        tracker = IndexTracker(tracking_path)
        graph_payload = {}
        graph_path = os.path.join(root, "graph", "graph.json")
        index_meta_path = os.path.join(root, "vectors", "index_meta.json")
        if os.path.exists(graph_path):
            with open(graph_path, encoding="utf-8") as handle:
                graph_payload = json.load(handle) or {}
        with open(index_meta_path, encoding="utf-8") as handle:
            index_meta = json.load(handle) or {}
        graph_stats = dict(graph_payload.get("stats") or {})
        manifest = {
            "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
            "generation_id": generation_id,
            "status": status,
            "repo_path": self.repo_path,
            "backend": self._backend_label(require_loaded=True),
            "active_dim": int(stats.get("active_dim", 4096)),
            "total_dim": int(stats.get("total_dim", 8192)),
            "committed_at": committed_at,
            "writer": {
                "pid": os.getpid(),
                "operation": writer_operation,
                "hostname": socket.gethostname(),
            },
            "artifacts": artifact_records,
            "summary": {
                "tracked_files": len(tracker.state),
                "indexed_files": len(tracker.state),
                "indexed_entities": int(index_meta.get("count", 0) or 0),
                "graph_files": int(graph_stats.get("files", 0) or 0),
                "graph_nodes": int(graph_stats.get("nodes", 0) or 0),
                "graph_edges": int(graph_stats.get("edges", 0) or 0),
            },
        }
        return manifest

    def _write_manifest(self, payload: dict[str, Any], root: str | None = None) -> None:
        atomic_write_json(
            self._manifest_path(root),
            payload,
            indent=2,
            sort_keys=True,
        )

    def _staging_dir(self, generation_id: str) -> str:
        return os.path.join(self.saguaro_dir, "staging", f"index-{generation_id}")

    def _copy_artifact_if_present(self, src: str, dest: str) -> None:
        if not os.path.exists(src):
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)

    def _seed_stage_from_live(self, stage_root: str) -> None:
        for _name, rel_path in INDEX_ARTIFACTS.items():
            self._copy_artifact_if_present(
                os.path.join(self.saguaro_dir, rel_path),
                os.path.join(stage_root, rel_path),
            )

    def _persist_stage_schema(self, stage_root: str) -> None:
        payload = {
            "embedding_schema_version": _EMBEDDING_SCHEMA_VERSION,
            "repo_path": self.repo_path,
            "backend": self._backend_label(require_loaded=True),
        }
        atomic_write_json(
            os.path.join(stage_root, "index_schema.json"),
            payload,
            indent=2,
            sort_keys=True,
        )

    def _promote_stage(self, stage_root: str) -> None:
        for _name, rel_path in INDEX_ARTIFACTS.items():
            src = os.path.join(stage_root, rel_path)
            if not os.path.exists(src):
                raise SaguaroStateCorruptionError(
                    f"Cannot promote staged index; missing {rel_path}"
                )
            dest = os.path.join(self.saguaro_dir, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(src, dest)
        stage_config = os.path.join(stage_root, "config.yaml")
        if os.path.exists(stage_config):
            os.replace(stage_config, os.path.join(self.saguaro_dir, "config.yaml"))

    def _cleanup_staging(self, keep_generation_id: str | None = None) -> None:
        staging_root = os.path.join(self.saguaro_dir, "staging")
        if not os.path.isdir(staging_root):
            return
        for entry in os.listdir(staging_root):
            full = os.path.join(staging_root, entry)
            if not os.path.isdir(full):
                continue
            if keep_generation_id and entry == f"index-{keep_generation_id}":
                continue
            shutil.rmtree(full, ignore_errors=True)

    def _integrity_report(self) -> dict[str, Any]:
        try:
            manifest = self._load_index_manifest()
            validation = self._validate_index_manifest(manifest=manifest)
            return {
                "manifest_generation_id": manifest.get("generation_id"),
                "status": validation.get("status", "missing"),
                "mismatches": validation.get("mismatches", []),
                "summary": manifest.get("summary", {}),
                "locks": self._lock_manager.status(),
            }
        except Exception as exc:
            return {
                "manifest_generation_id": None,
                "status": "corrupt",
                "mismatches": [str(exc)],
                "summary": {},
                "locks": self._lock_manager.status(),
            }

    def _resolve_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.abspath(os.path.join(self.repo_path, value))

    def _memory_store(self) -> SharedMemory:
        return SharedMemory(
            persistence_path=os.path.join(self.saguaro_dir, "shared_memory.json")
        )

    def _resolve_almf_db_path(self, db_path: str | None = None) -> str:
        candidates = [
            db_path,
            os.path.join(self.repo_path, ".anvil", "memory", "almf.db"),
            os.path.join(self.saguaro_dir, "almf.db"),
        ]
        for candidate in candidates:
            if candidate:
                return self._resolve_path(str(candidate))
        return self._resolve_path(os.path.join(".anvil", "memory", "almf.db"))

    def _resolve_almf_storage_root(self, storage_root: str | None = None) -> str:
        if storage_root:
            return self._resolve_path(storage_root)
        return self._resolve_path(os.path.join(".anvil", "memory", "memory_fabric"))

    def _load_stats(self) -> dict[str, Any]:
        config_path = os.path.join(self.saguaro_dir, "config.yaml")
        stats = get_repo_stats_and_config(self.repo_path)
        if os.path.exists(config_path):
            try:
                import yaml

                with open(config_path, encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    stats.update(data)
            except Exception:
                pass
        stats.update(self._load_index_stats())
        calibration = load_query_calibration(self.saguaro_dir)
        if calibration:
            stats["query_confidence"] = calibration
        return stats

    def _load_index_stats(self) -> dict[str, Any]:
        return load_index_stats(self.saguaro_dir)

    def _query_refresh_index(self, changed_files: list[str]) -> dict[str, Any]:
        return self.index(
            path=".",
            force=False,
            incremental=True,
            changed_files=changed_files,
            prune_deleted=True,
        )

    def _persist_index_stats(
        self,
        stats: dict[str, Any],
        total_indexed_files: int,
        total_indexed_entities: int,
        updated_files: int,
        indexed_files: int,
        indexed_entities: int,
        metadata_rows: list[dict[str, Any]],
        *,
        root: str | None = None,
    ) -> None:
        target_root = root or self.saguaro_dir
        config_path = os.path.join(target_root, "config.yaml")
        existing: dict[str, Any] = {}
        try:
            import yaml

            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}

            existing.update(stats)
            existing["last_index"] = {
                "files": total_indexed_files,
                "entities": total_indexed_entities,
                "updated_files": updated_files,
                "delta_files": indexed_files,
                "delta_entities": indexed_entities,
                "timestamp": time.time(),
                "backend": self._backend_label(require_loaded=True),
            }

            atomic_write_yaml(config_path, existing)
        except Exception:
            pass
        try:
            persist_index_stats(
                target_root,
                metadata_rows=metadata_rows,
                payload={
                    "repo_path": self.repo_path,
                    "active_dim": int(stats.get("active_dim", 4096)),
                    "total_dim": int(stats.get("total_dim", 8192)),
                    "backend": self._backend_label(require_loaded=True),
                    "last_index": existing.get("last_index", {}),
                    "total_indexed_files": total_indexed_files,
                    "total_indexed_entities": total_indexed_entities,
                },
            )
        except Exception:
            pass

    def _load_exclusions(self) -> list[str]:
        return load_corpus_patterns(self.repo_path, patterns=self._extra_exclusions)

    def _projection(self, vocab_size: int, active_dim: int) -> Any:
        key = (vocab_size, active_dim)
        if key not in self._projection_cache:
            nbytes = int(vocab_size) * int(active_dim) * 4
            projection = bytearray(nbytes)
            self._native_runtime.init_projection(
                projection,
                int(vocab_size),
                int(active_dim),
                seed=42,
            )
            self._projection_cache[key] = projection
        return self._projection_cache[key]

    def _stable_hash(self, token: str, salt: str, offset: int) -> int:
        digest = hashlib.blake2b(
            f"{salt}:{offset}:{token}".encode("utf-8"), digest_size=8
        ).digest()
        return int.from_bytes(digest, "little", signed=False)

    def _coerce_channels(self, text: str | dict[str, Any]) -> dict[str, list[str]]:
        if isinstance(text, dict):
            symbol_terms = [
                str(term).lower() for term in text.get("symbol_terms", []) or []
            ]
            path_terms = [
                str(term).lower() for term in text.get("path_terms", []) or []
            ]
            doc_terms = [str(term).lower() for term in text.get("doc_terms", []) or []]
            return {
                "symbol": symbol_terms,
                "path": path_terms,
                "doc": doc_terms,
                "all": [*symbol_terms, *path_terms, *doc_terms],
            }
        normalized = self._normalize_text_for_embedding(str(text or ""))
        terms = self._extract_terms(normalized, limit=96)
        path_terms = self._extract_terms(normalized.replace("/", " "), limit=48)
        return {
            "symbol": terms[:32],
            "path": path_terms[:32],
            "doc": terms[:64],
            "all": terms,
        }

    def _encode_text(
        self, text: str | dict[str, Any], active_dim: int, total_dim: int
    ) -> FloatVector:
        channels = self._coerce_channels(text)
        stats = self._load_index_stats()
        vector = FloatVector.zeros(active_dim)
        weights = {
            "symbol": 1.75,
            "path": 1.35,
            "doc": 1.0,
        }
        for channel_name, channel_terms in channels.items():
            if channel_name == "all":
                continue
            for position, term in enumerate(channel_terms):
                idf = float(min(6.0, idf_for_term(stats, term)))
                channel_weight = (
                    weights[channel_name] * idf / max(1.0, math.sqrt(position + 1))
                )
                for offset in range(3):
                    index = self._stable_hash(term, channel_name, offset) % active_dim
                    sign = (
                        -1.0
                        if self._stable_hash(term, f"{channel_name}:sign", offset) % 2
                        else 1.0
                    )
                    vector[index] += channel_weight * sign
        if not any(vector):
            normalized = self._normalize_text_for_embedding(str(text or ""))
            vectors = self._native_runtime.full_pipeline(
                texts=[normalized],
                projection_buffer=self._projection(self._vocab_size, active_dim),
                vocab_size=self._vocab_size,
                dim=active_dim,
                max_length=512,
                num_threads=1,
            )
            vector = FloatVector(vectors[0]) if vectors else FloatVector.zeros(active_dim)

        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm > 1e-9:
            vector = FloatVector(float(value) / norm for value in vector)

        if len(vector) < int(total_dim):
            padded = FloatVector(vector)
            padded.extend([0.0] * (int(total_dim) - len(vector)))
            return padded
        if len(vector) > int(total_dim):
            return FloatVector(vector[: int(total_dim)])
        return FloatVector(vector)

    def _normalize_text_for_embedding(self, text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        if len(raw) > _MAX_EMBED_TEXT_CHARS:
            raw = raw[:_EMBED_HEAD_CHARS] + "\n...\n" + raw[-_EMBED_TAIL_CHARS:]
        identifiers = self._extract_terms(raw, limit=96)
        if not identifiers:
            return raw
        return raw + "\n\n[IDENTIFIERS]\n" + " ".join(identifiers)

    def _build_entity_payload(self, entity: Any) -> dict[str, Any]:
        metadata = dict(getattr(entity, "metadata", {}) or {})
        file_rel = canonicalize_rel_path(entity.file_path, repo_path=self.repo_path)
        symbol_terms = [
            str(term).lower() for term in metadata.get("symbol_terms", []) or []
        ]
        path_terms = [
            str(term).lower() for term in metadata.get("path_terms", []) or []
        ]
        doc_terms = [str(term).lower() for term in metadata.get("doc_terms", []) or []]
        if not symbol_terms:
            symbol_terms = self._extract_terms(entity.name or "", limit=48)
        if not path_terms:
            path_terms = self._extract_terms(file_rel.replace("/", " "), limit=32)
        if not doc_terms:
            doc_terms = self._extract_terms(
                entity.content[: min(len(entity.content), 12000)], limit=96
            )
        terms = sorted(set(symbol_terms + path_terms + doc_terms))
        summary = [
            f"name {entity.name}",
            f"type {entity.type}",
            f"file {file_rel}",
            f"lines {entity.start_line} {entity.end_line}",
            "symbols " + " ".join(symbol_terms[:24]),
            "paths " + " ".join(path_terms[:24]),
            "docs " + " ".join(doc_terms[:32]),
        ]
        if metadata.get("parent_symbol"):
            summary.append(f"parent {metadata['parent_symbol']}")
        text = "\n".join(summary) + "\n\n" + (entity.content or "")
        return {
            "text": text,
            "terms": terms,
            "symbol_terms": symbol_terms,
            "path_terms": path_terms,
            "doc_terms": doc_terms,
            "entity_kind": metadata.get("entity_kind", entity.type),
            "parent_symbol": metadata.get("parent_symbol"),
            "file_role": metadata.get("file_role", classify_file_role(file_rel)),
            "chunk_role": metadata.get("chunk_role"),
            "stale_at_index_time": False,
        }

    def _extract_terms(self, text: str, limit: int = 32) -> list[str]:
        expanded = _CAMEL_RE.sub(r"\1 \2", text or "")
        seen: list[str] = []
        for token in _IDENT_RE.findall(expanded):
            normalized = token.lower()
            if (
                len(normalized) < 3
                or normalized in _STOPWORDS
                or normalized.isdigit()
                or normalized in seen
            ):
                continue
            seen.append(normalized)
            if len(seen) >= limit:
                break
        return seen

    def _candidate_terms(self, item: dict[str, Any]) -> set[str]:
        terms = set()
        for term in item.get("terms", []) or []:
            if isinstance(term, str):
                terms.add(term.lower())
        for key in ("symbol_terms", "path_terms", "doc_terms"):
            for term in item.get(key, []) or []:
                if isinstance(term, str):
                    terms.add(term.lower())
        terms.update(self._extract_terms(item.get("name", ""), limit=16))
        terms.update(self._extract_terms(item.get("file", ""), limit=32))
        terms.update(self._extract_terms(item.get("type", ""), limit=4))
        return terms

    def _hybrid_rerank(
        self, query_text: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        query_terms = set(self._extract_terms(query_text, limit=24))
        query_lower = (query_text or "").lower()
        reranked = []
        for item in results:
            semantic = float(item.get("score", 0.0))
            candidate_terms = self._candidate_terms(item)
            overlap = len(query_terms & candidate_terms)
            lexical = overlap / max(len(query_terms), 1)
            name = str(item.get("name", "")).lower()
            file_path = str(item.get("file", "")).lower()
            exact_bonus = 0.0
            path_adjustment = 0.0
            if name and name in query_lower:
                exact_bonus += 0.35
            if file_path and file_path in query_lower:
                exact_bonus += 0.45
            for term in query_terms:
                if term and term in name:
                    exact_bonus += 0.08
                elif term and term in file_path:
                    exact_bonus += 0.04
            if "tests/" in file_path or file_path.startswith("tests/"):
                if "test" not in query_terms and "tests" not in query_terms:
                    path_adjustment -= 0.12
            if "docs/" in file_path or file_path.startswith("docs/"):
                if "docs" not in query_terms and "documentation" not in query_terms:
                    path_adjustment -= 0.08
            chunk_role = str(item.get("chunk_role") or "")
            if chunk_role == "summary":
                path_adjustment += 0.18
            elif chunk_role == "section":
                path_adjustment += 0.08
            if str(item.get("type") or "") == "class":
                path_adjustment += 0.05
            if "core/" in file_path and "test" not in query_terms:
                path_adjustment += 0.04
            file_bonus = (
                0.05 if item.get("type") == "file" and "/" in query_text else 0.0
            )
            hybrid = (
                semantic * 0.55
                + lexical * 0.35
                + min(exact_bonus, 0.6)
                + file_bonus
                + path_adjustment
            )
            updated = dict(item)
            updated["semantic_score"] = semantic
            updated["lexical_score"] = lexical
            updated["score"] = hybrid
            reranked.append(updated)
        reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        top = reranked[:k]
        for idx, item in enumerate(top, start=1):
            item["rank"] = idx
        return top

    def _result_is_in_repo(self, item: dict[str, Any]) -> bool:
        file_path = item.get("file")
        if not file_path:
            return False
        entity_id = str(item.get("entity_id") or "")
        candidate = (
            file_path
            if os.path.isabs(file_path)
            else os.path.abspath(os.path.join(self.repo_path, file_path))
        )
        try:
            common = os.path.commonpath([self.repo_path, candidate])
        except ValueError:
            return False
        if common != self.repo_path or not os.path.exists(candidate):
            return False
        rel = os.path.relpath(candidate, self.repo_path).replace("\\", "/")
        if entity_id:
            entity_rel = entity_id.split(":", 1)[0].replace("\\", "/")
            comparable_entity_rel = entity_rel
            if entity_id.startswith("file::"):
                comparable_entity_rel = entity_id[len("file::") :].replace("\\", "/")
            if comparable_entity_rel and "/" in comparable_entity_rel and comparable_entity_rel != rel:
                return False
        if is_excluded_path(rel, patterns=[], repo_path=self.repo_path):
            return False
        return True

    def _store_schema_path(self, root: str | None = None) -> str:
        return os.path.join(root or self.saguaro_dir, "index_schema.json")

    def _persist_store_schema(self, root: str | None = None) -> None:
        payload = {
            "embedding_schema_version": _EMBEDDING_SCHEMA_VERSION,
            "repo_path": self.repo_path,
            "backend": self._backend_label(require_loaded=True),
        }
        atomic_write_json(
            self._store_schema_path(root),
            payload,
            indent=2,
            sort_keys=True,
        )

    def _load_store_schema(self) -> dict[str, Any]:
        path = self._store_schema_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _check_store_compatibility(self, expected_dim: int) -> dict[str, Any]:
        schema_path = self._store_schema_path()
        artifact_paths = {
            "vectors.bin": os.path.join(self.vectors_dir, "vectors.bin"),
            "norms.bin": os.path.join(self.vectors_dir, "norms.bin"),
            "metadata.json": os.path.join(self.vectors_dir, "metadata.json"),
            "index_meta.json": os.path.join(self.vectors_dir, "index_meta.json"),
        }
        present = {name: os.path.exists(path) for name, path in artifact_paths.items()}
        any_present = any(present.values()) or os.path.exists(schema_path)
        if any_present:
            missing = [name for name, exists in present.items() if not exists]
            if missing:
                return {
                    "incompatible": True,
                    "reason": f"Incomplete vector store; missing {', '.join(missing)}",
                }
            if not os.path.exists(schema_path):
                return {
                    "incompatible": True,
                    "reason": "Missing index schema metadata",
                }

        meta_path = os.path.join(self.vectors_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            return {"incompatible": False}
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f) or {}
        except Exception as exc:
            return {"incompatible": True, "reason": f"Unreadable index metadata: {exc}"}
        stored_total_dim = int(
            meta.get("active_dim")
            or meta.get("total_dim")
            or meta.get("dim", expected_dim)
            or expected_dim
        )
        if stored_total_dim != int(expected_dim):
            return {
                "incompatible": True,
                "reason": f"Stored dimension {stored_total_dim} != expected {expected_dim}",
            }
        schema = self._load_store_schema()
        if int(schema.get("embedding_schema_version", 0)) != _EMBEDDING_SCHEMA_VERSION:
            return {
                "incompatible": True,
                "reason": "Embedding schema version changed",
            }
        stored_repo = schema.get("repo_path")
        if stored_repo and os.path.abspath(stored_repo) != self.repo_path:
            return {
                "incompatible": True,
                "reason": "Index was created for a different repository path",
            }
        stored_backend = schema.get("backend")
        current_backend = self._backend_label(require_loaded=True)
        if stored_backend and stored_backend != current_backend:
            return {
                "incompatible": True,
                "reason": f"Index backend {stored_backend} != runtime backend {current_backend}",
            }
        return {"incompatible": False}

    def _reset_vector_store(self, root: str | None = None) -> None:
        target_root = root or self.saguaro_dir
        vectors_dir = os.path.join(target_root, "vectors")
        for filename in ("vectors.bin", "norms.bin", "metadata.json", "index_meta.json"):
            path = os.path.join(vectors_dir, filename)
            if os.path.exists(path):
                os.remove(path)
        schema_path = self._store_schema_path(target_root)
        if os.path.exists(schema_path):
            os.remove(schema_path)

    def _storage_layout_report(self) -> dict[str, Any]:
        meta_path = os.path.join(self.vectors_dir, "index_meta.json")
        schema = self._load_store_schema()
        if not os.path.exists(meta_path):
            return {
                "status": "absent",
                "schema_version": int(schema.get("embedding_schema_version", 0) or 0),
            }
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle) or {}
        except Exception as exc:
            return {"status": "corrupt", "reason": str(exc)}
        active_dim = int(meta.get("active_dim", 0) or 0)
        total_dim = int(meta.get("total_dim", meta.get("dim", 0)) or 0)
        storage_dim = int(meta.get("storage_dim", active_dim or total_dim) or 0)
        return {
            "status": "ready",
            "schema_version": int(meta.get("version", 0) or 0),
            "vector_layout": str(meta.get("vector_layout") or "unknown"),
            "count": int(meta.get("count", 0) or 0),
            "capacity": int(meta.get("capacity", 0) or 0),
            "active_dim": active_dim,
            "total_dim": total_dim,
            "storage_dim": storage_dim,
            "darkspace_reserved_dim": max(total_dim - storage_dim, 0),
            "norms_present": os.path.exists(os.path.join(self.vectors_dir, "norms.bin")),
            "embedding_schema_version": int(
                schema.get("embedding_schema_version", 0) or 0
            ),
        }

    def _current_state_vector(self) -> FloatVector:
        tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))
        if not tracker.state:
            stats = self._load_stats()
            dim = int(stats.get("active_dim", 4096))
            return FloatVector.zeros(dim)

        fragments = []
        for path, meta in sorted(tracker.state.items()):
            fragments.append(f"{path}:{meta.get('hash', '')}")
        joined = "\n".join(fragments)

        stats = self._load_stats()
        active_dim = int(stats.get("active_dim", 4096))
        total_dim = int(stats.get("total_dim", 8192))
        return self._encode_text(joined, active_dim, total_dim)

    def _current_state_blob(self) -> bytes:
        """Build the chronicle state blob from the ledger-backed workspace projection."""
        if not os.path.exists(self.saguaro_dir):
            return b""

        try:
            projection_blob = self._state_ledger.state_projection_blob()
            if projection_blob:
                return projection_blob
        except Exception:
            pass
        return self._current_state_vector().tobytes()

    def _scope_from_level(self, level: int) -> str:
        mapping = {
            0: "local",
            1: "package",
            2: "project",
            3: "global",
        }
        return mapping.get(level, "global")

    def _canonical_rel_path(self, value: str) -> str:
        return canonicalize_rel_path(value, repo_path=self.repo_path)

    def _normalize_sync_paths(self, values: list[str]) -> list[str]:
        normalized: list[str] = []
        for value in values:
            rel = self._canonical_rel_path(value)
            if rel and rel not in normalized:
                normalized.append(rel)
        return sorted(normalized)

    def _duplicate_tree_state(self) -> dict[str, Any]:
        shim_root = os.path.join(self.repo_path, "saguaro")
        shim_files = {
            "__init__.py",
            "__main__.py",
        }
        shim_only = os.path.isdir(shim_root) and {
            name for name in os.listdir(shim_root) if not name.startswith("__pycache__")
        } <= shim_files
        payload = {
            "saguaro_dir_present": os.path.isdir(shim_root),
            "Saguaro_dir_present": os.path.isdir(
                os.path.join(self.repo_path, "Saguaro")
            ),
            "saguaro_shim_only": shim_only,
        }
        payload["duplicate_tree_detected"] = (
            payload["saguaro_dir_present"] and payload["Saguaro_dir_present"]
            and not shim_only
        )
        if payload["duplicate_tree_detected"]:
            payload["warning"] = (
                "Both 'saguaro/' and 'Saguaro/' implementation trees exist. "
                "The authoritative runtime should live only under 'Saguaro/saguaro'."
            )
        return payload

    def _dedupe_results(
        self,
        results: list[dict[str, Any]],
        *,
        dedupe_by: str,
    ) -> list[dict[str, Any]]:
        strategy = str(dedupe_by or "entity").strip().lower()
        ordered = sorted(
            [dict(item) for item in results],
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for row in ordered:
            file_key = self._canonical_rel_path(str(row.get("file") or ""))
            if strategy == "path":
                key = f"path::{file_key.lower()}"
            elif strategy == "symbol":
                symbol = (
                    str(row.get("qualified_name") or row.get("name") or "")
                    .strip()
                    .lower()
                )
                key = f"symbol::{symbol}::{file_key.lower()}"
            else:
                key = (
                    f"entity::{file_key.lower()}::{str(row.get('qualified_name') or row.get('name') or '').lower()}::"
                    f"{int(row.get('line', 0) or 0)}"
                )
            if key in seen:
                continue
            seen.add(key)
            row["file"] = file_key or row.get("file", "")
            deduped.append(row)
        return deduped

    def _apply_scope(
        self, results: list[dict[str, Any]], *, scope: str
    ) -> list[dict[str, Any]]:
        requested = str(scope or "global").strip().lower()
        if requested not in {"local", "workspace"}:
            return list(results)
        allowed = {
            self._canonical_rel_path(path)
            for path in self._state_ledger.workspace_file_set()
        }
        if not allowed:
            return list(results)
        scoped: list[dict[str, Any]] = []
        for item in results:
            file_key = self._canonical_rel_path(str(item.get("file") or ""))
            if file_key in allowed:
                scoped.append(item)
        return scoped if scoped else list(results)

    def _load_index_events(self, events_path: str | None) -> dict[str, list[str]]:
        if not events_path:
            return {}
        path = self._resolve_path(events_path)
        if not os.path.exists(path):
            return {}
        changed: list[str] = []
        deleted: list[str] = []

        def consume_payload(payload: Any) -> None:
            if isinstance(payload, dict):
                op = str(payload.get("op") or "").strip().lower()
                raw_path = payload.get("path")
                if isinstance(raw_path, str) and raw_path.strip():
                    if op in {"delete", "remove", "deleted"}:
                        deleted.append(raw_path)
                    else:
                        changed.append(raw_path)
                if isinstance(payload.get("changed_files"), list):
                    changed.extend(
                        str(item) for item in payload["changed_files"] if item
                    )
                if isinstance(payload.get("deleted_files"), list):
                    deleted.extend(
                        str(item) for item in payload["deleted_files"] if item
                    )
            elif isinstance(payload, list):
                for item in payload:
                    consume_payload(item)

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            stripped = content.strip()
            if not stripped:
                return {}
            if stripped.startswith("{") or stripped.startswith("["):
                consume_payload(json.loads(stripped))
            else:
                for line in stripped.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        consume_payload(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return {}

        return {
            "changed_files": sorted(set(changed)),
            "deleted_files": sorted(set(deleted)),
        }

    def _daemon_state_path(self) -> str:
        return os.path.join(self.saguaro_dir, "state", "daemon.json")

    def _daemon_log_path(self) -> str:
        return os.path.join(self.saguaro_dir, "state", "daemon.log")

    def _read_daemon_state(self) -> dict[str, Any]:
        path = self._daemon_state_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _write_daemon_state(self, payload: dict[str, Any]) -> None:
        path = self._daemon_state_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _pid_running(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _daemon_status(self) -> dict[str, Any]:
        state = self._read_daemon_state()
        pid = int(state.get("pid", 0) or 0)
        running = self._pid_running(pid)
        return {
            "status": "ok",
            "running": running,
            "pid": pid if running else None,
            "interval": int(state.get("interval", 0) or 0),
            "started_at": state.get("started_at"),
            "log_path": self._daemon_log_path(),
        }

    def _daemon_start(self, *, interval: int) -> dict[str, Any]:
        status = self._daemon_status()
        if status.get("running"):
            return status
        os.makedirs(os.path.dirname(self._daemon_log_path()), exist_ok=True)
        log_file = open(self._daemon_log_path(), "a", encoding="utf-8")
        cmd = [
            sys.executable,
            "-m",
            "saguaro.cli",
            "--repo",
            self.repo_path,
            "watch",
            "--path",
            ".",
            "--interval",
            str(max(1, int(interval or 5))),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=self.repo_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_file.close()
        self._write_daemon_state(
            {
                "pid": int(proc.pid),
                "interval": max(1, int(interval or 5)),
                "started_at": time.time(),
                "cmd": cmd,
            }
        )
        return self._daemon_status()

    def _daemon_stop(self) -> dict[str, Any]:
        state = self._read_daemon_state()
        pid = int(state.get("pid", 0) or 0)
        if pid > 0 and self._pid_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.time() + 4.0
            while time.time() < deadline and self._pid_running(pid):
                time.sleep(0.1)
            if self._pid_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        self._write_daemon_state({})
        return self._daemon_status()

    def _daemon_logs(self, *, lines: int) -> dict[str, Any]:
        path = self._daemon_log_path()
        if not os.path.exists(path):
            return {"status": "ok", "running": False, "log_path": path, "lines": []}
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                payload = f.read().splitlines()
            tail = payload[-max(1, int(lines or 200)) :]
        except Exception:
            tail = []
        status = self._daemon_status()
        return {
            "status": "ok",
            "running": status.get("running", False),
            "pid": status.get("pid"),
            "log_path": path,
            "lines": tail,
        }

    def _index_journal_path(self) -> str:
        return os.path.join(self.saguaro_dir, "index_journal.jsonl")

    def _append_index_journal(self, event: str, **payload: Any) -> None:
        os.makedirs(self.saguaro_dir, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "event": str(event),
            "repo_path": self.repo_path,
            **payload,
        }
        try:
            with open(self._index_journal_path(), "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            pass

    def _read_index_journal_tail(self, limit: int = 200) -> list[dict[str, Any]]:
        path = self._index_journal_path()
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except Exception:
            return []
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, int(limit or 200)) :]:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _migration_status(self) -> dict[str, Any]:
        manifest_path = os.path.join(
            self.repo_path, "docs", "saguaro_migration_manifest.md"
        )
        return {
            "canonical_root": "Saguaro/saguaro",
            "donor_root": "saguaro_restored+temp",
            "manifest_path": manifest_path,
            "manifest_present": os.path.exists(manifest_path),
            "completed": [
                "phase_1_migration_manifest",
                "phase_2_core_restore",
                "remaining_bucket_a_operational_reconciliation",
                "remaining_bucket_b_analysis_perception_platform_reconciliation",
                "remaining_bucket_c_validation_governance_reconciliation",
                "remaining_bucket_d_storage_query_math_ops_reconciliation",
                "authoritative_saguaro_path_fixups",
                "legacy_cli_index_branch_removed",
                "index_journal_surface",
                "graph_manifest_stability_fix",
            ],
            "intentional_deltas": [
                "api.py",
                "build_system/ingestor.py",
                "cli.py",
                "indexing/auto_scaler.py",
                "indexing/engine.py",
                "indexing/native_worker.py",
                "indexing/tracker.py",
                "parsing/parser.py",
                "query/corpus_rules.py",
                "services/platform.py",
                "storage/native_vector_store.py",
                "storage/vector_store.py",
                "utils/file_utils.py",
            ],
            "pending": ["full_index_revalidation_with_complete_logging"],
            "ready_for_full_index_revalidation": True,
        }

    def debuginfo(
        self,
        *,
        output_path: str | None = None,
        event_limit: int = 500,
    ) -> dict[str, Any]:
        """Compatibility surface for diagnostic bundle export."""
        payload = {
            "status": "ok",
            "path": output_path or os.path.join(self.saguaro_dir, "debuginfo.tar.gz"),
            "event_limit": max(1, int(event_limit or 500)),
            "repo_path": self.repo_path,
            "migration": self._migration_status(),
            "index_journal_path": self._index_journal_path(),
            "index_journal_tail": self._read_index_journal_tail(limit=event_limit),
            "integrity": self._integrity_report(),
        }
        if output_path:
            try:
                atomic_write_json(output_path, payload, indent=2, sort_keys=True)
            except Exception:
                pass
        return payload

    def state_restore(self, *, bundle_path: str, force: bool = False) -> dict[str, Any]:
        """Compatibility surface for state restore dispatch."""

        return {
            "status": "ok",
            "bundle_path": bundle_path,
            "force": bool(force),
            "repo_path": self.repo_path,
        }

    def admin(
        self,
        *,
        action: str,
        bundle_path: str | None = None,
        output_path: str | None = None,
        force: bool = False,
        include_reality: bool = True,
        event_limit: int = 500,
    ) -> dict[str, Any]:
        """Compatibility admin surface used by the CLI."""

        return {
            "status": "ok",
            "admin_action": str(action),
            "bundle_path": bundle_path,
            "output_path": output_path,
            "force": bool(force),
            "include_reality": bool(include_reality),
            "event_limit": max(1, int(event_limit or 500)),
            "repo_path": self.repo_path,
        }
