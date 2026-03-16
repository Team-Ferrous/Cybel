"""Lean native-first indexing runtime for comparative corpus sessions."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from typing import Any

from saguaro.errors import SaguaroStateCorruptionError, SaguaroStateMismatchError
from saguaro.indexing.auto_scaler import get_repo_stats_and_config
from saguaro.indexing.memory_optimized_engine import MemoryOptimizedIndexEngine
from saguaro.indexing.native_coordinator import run_native_index_coordinator
from saguaro.indexing.stats import INDEX_SCHEMA_VERSION, load_index_stats, persist_index_stats
from saguaro.indexing.tracker import IndexTracker
from saguaro.query.corpus_rules import canonicalize_rel_path, load_corpus_patterns
from saguaro.services.platform import GraphService, ParseService
from saguaro.state.ledger import StateLedger
from saguaro.storage.atomic_fs import atomic_write_json, atomic_write_yaml
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
from saguaro.utils.file_utils import build_corpus_manifest


class ComparativeIndexRuntime:
    """Minimal index/runtime facade used by comparative corpus ingestion."""

    _BACKEND_LABEL = "NativeIndexerBackend"

    def __init__(
        self,
        repo_path: str,
        *,
        saguaro_dir: str,
        extra_exclusions: list[str] | None = None,
        state_ledger: StateLedger | None = None,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.abspath(saguaro_dir)
        self.vectors_dir = os.path.join(self.saguaro_dir, "vectors")
        self._extra_exclusions = list(extra_exclusions or [])
        self._lock_manager = RepoLockManager(self.saguaro_dir)
        self._state_ledger = state_ledger or StateLedger(
            self.repo_path,
            saguaro_dir=self.saguaro_dir,
        )

    def index(
        self,
        path: str = ".",
        force: bool = False,
        incremental: bool = True,
        changed_files: list[str] | None = None,
        events_path: str | None = None,
        prune_deleted: bool = False,
    ) -> dict[str, Any]:
        with self._lock_manager.acquire("index", mode="exclusive", operation="index"):
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
            target_path = self._resolve_path(path)
            stats = get_repo_stats_and_config(target_path)

            compatibility = dict(
                self._check_store_compatibility(expected_dim=int(stats.get("active_dim", 4096)))
            )
            live_manifest = self._load_index_manifest()
            if (
                not force
                and not compatibility.get("incompatible")
                and not live_manifest
                and any(os.path.exists(path) for path in self._artifact_paths().values())
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
                        manifest=live_manifest,
                        require_ready=True,
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
            stale_files: list[str] = []
            if prune_deleted:
                if ledger_scan is not None:
                    stale_files = [
                        self._resolve_path(rel)
                        for rel in list(ledger_scan.get("deleted_files", []) or [])
                        if rel
                    ]
                else:
                    prior_tracker = IndexTracker(os.path.join(stage_root, "tracking.json"))
                    stale_files = prior_tracker.prune_missing(all_files)

            indexed_entities = 0
            indexed_files = 0
            parsed_files: list[str] = []
            removed_files = sorted(set(stale_files + explicit_deleted))

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
                indexed_entities = int(coordinator_result.get("indexed_entities", 0) or 0)
                parsed_files = sorted(set(coordinator_result.get("touched_files", []) or []))
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
                max(0.0, time.perf_counter() - started_at) - sum(stage_timings.values()),
                6,
            )
            graph_changed_files: list[str] | None = None
            if incremental and not effective_force:
                changed_set = sorted(set(parsed_files + removed_files))
                if changed_set:
                    graph_changed_files = changed_set
            graph_started = time.perf_counter()
            stage_graph_service = GraphService(
                self.repo_path,
                ParseService(self.repo_path),
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

            result = {
                "status": "ok",
                "backend": self._BACKEND_LABEL,
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
                    compatibility.get("reason") if compatibility.get("incompatible") else None
                ),
                "coordinator": coordinator_result,
                "stage_timings": stage_timings,
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
            return result

    def _store_schema_path(self, root: str | None = None) -> str:
        return os.path.join(root or self.saguaro_dir, "index_schema.json")

    def _load_store_schema(self, root: str | None = None) -> dict[str, Any]:
        path = self._store_schema_path(root)
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle) or {}
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
        return {}

    def _check_store_compatibility(self, expected_dim: int) -> dict[str, Any]:
        schema_path = self._store_schema_path()
        current_artifacts = {
            "vectors.bin": os.path.join(self.vectors_dir, "vectors.bin"),
            "norms.bin": os.path.join(self.vectors_dir, "norms.bin"),
            "metadata.json": os.path.join(self.vectors_dir, "metadata.json"),
            "index_meta.json": os.path.join(self.vectors_dir, "index_meta.json"),
        }
        present = {name: os.path.exists(path) for name, path in current_artifacts.items()}
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
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle) or {}
        except Exception as exc:
            return {"incompatible": True, "reason": f"Unreadable index metadata: {exc}"}
        stored_dim = int(
            meta.get("active_dim")
            or meta.get("total_dim")
            or meta.get("dim", expected_dim)
            or expected_dim
        )
        if stored_dim != int(expected_dim):
            return {
                "incompatible": True,
                "reason": f"Stored dimension {stored_dim} != expected {expected_dim}",
            }
        schema = self._load_store_schema()
        if int(schema.get("embedding_schema_version", 0) or 0) != INDEX_SCHEMA_VERSION:
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
        if stored_backend and stored_backend != self._BACKEND_LABEL:
            return {
                "incompatible": True,
                "reason": f"Index backend {stored_backend} != runtime backend {self._BACKEND_LABEL}",
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

        tracker = IndexTracker(os.path.join(root, "tracking.json"))
        graph_payload = {}
        graph_path = os.path.join(root, "graph", "graph.json")
        index_meta_path = os.path.join(root, "vectors", "index_meta.json")
        if os.path.exists(graph_path):
            with open(graph_path, encoding="utf-8") as handle:
                graph_payload = json.load(handle) or {}
        with open(index_meta_path, encoding="utf-8") as handle:
            index_meta = json.load(handle) or {}
        graph_stats = dict(graph_payload.get("stats") or {})
        return {
            "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
            "generation_id": generation_id,
            "status": status,
            "repo_path": self.repo_path,
            "backend": self._BACKEND_LABEL,
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

    def _write_manifest(self, payload: dict[str, Any], root: str | None = None) -> None:
        atomic_write_json(manifest_path(root or self.saguaro_dir), payload, indent=2, sort_keys=True)

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

    def _persist_store_schema(self, root: str | None = None) -> None:
        payload = {
            "embedding_schema_version": INDEX_SCHEMA_VERSION,
            "repo_path": self.repo_path,
            "backend": self._BACKEND_LABEL,
        }
        atomic_write_json(self._store_schema_path(root), payload, indent=2, sort_keys=True)

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
                with open(config_path, encoding="utf-8") as handle:
                    existing = yaml.safe_load(handle) or {}

            existing.update(stats)
            existing["last_index"] = {
                "files": total_indexed_files,
                "entities": total_indexed_entities,
                "updated_files": updated_files,
                "delta_files": indexed_files,
                "delta_entities": indexed_entities,
                "timestamp": time.time(),
                "backend": self._BACKEND_LABEL,
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
                    "backend": self._BACKEND_LABEL,
                    "last_index": existing.get("last_index", {}),
                    "total_indexed_files": total_indexed_files,
                    "total_indexed_entities": total_indexed_entities,
                },
            )
        except Exception:
            pass

    def _load_exclusions(self) -> list[str]:
        return load_corpus_patterns(self.repo_path, patterns=self._extra_exclusions)

    def _resolve_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.abspath(os.path.join(self.repo_path, value))

    def _canonical_rel_path(self, value: str) -> str:
        return canonicalize_rel_path(value, repo_path=self.repo_path)

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
                    changed.extend(str(item) for item in payload["changed_files"] if item)
                if isinstance(payload.get("deleted_files"), list):
                    deleted.extend(str(item) for item in payload["deleted_files"] if item)
            elif isinstance(payload, list):
                for item in payload:
                    consume_payload(item)

        try:
            with open(path, encoding="utf-8") as handle:
                stripped = handle.read().strip()
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

    def load_index_stats(self) -> dict[str, Any]:
        return load_index_stats(self.saguaro_dir)
