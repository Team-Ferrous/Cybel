"""Lightweight command implementations for latency-sensitive Saguaro CLI paths."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import time
from typing import Any

from saguaro.health import collect_native_compute_report
from saguaro.indexing.auto_scaler import (
    calibrate_runtime_profile,
    get_repo_stats_and_config,
    load_runtime_profile,
)
from saguaro.indexing.native_indexer_bindings import (
    collect_native_capability_report,
    get_native_indexer,
)
from saguaro.indexing.native_runtime import get_native_runtime
from saguaro.indexing.stats import INDEX_SCHEMA_VERSION, idf_for_term, load_index_stats
from saguaro.indexing.tracker import IndexTracker
from saguaro.query.benchmark import load_query_calibration
from saguaro.query.corpus_rules import canonicalize_rel_path, is_excluded_path
from saguaro.storage.atomic_fs import atomic_write_json
from saguaro.storage.index_state import load_manifest, manifest_path, validate_manifest
from saguaro.storage.index_state import INDEX_ARTIFACTS, snapshot_artifact
from saguaro.storage.locks import RepoLockManager
from saguaro.utils.float_vector import FloatVector

_IDENT_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_MAX_EMBED_TEXT_CHARS = int(os.getenv("SAGUARO_MAX_EMBED_TEXT_CHARS", "24000"))
_EMBED_HEAD_CHARS = int(os.getenv("SAGUARO_EMBED_HEAD_CHARS", "16000"))
_EMBED_TAIL_CHARS = int(os.getenv("SAGUARO_EMBED_TAIL_CHARS", "8000"))
_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


class FastCommandAPI:
    """Reduced-overhead subset of the Saguaro API for hot commands."""

    def __init__(self, repo_path: str = ".", *, use_gateway: bool = True) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.join(self.repo_path, ".saguaro")
        self.vectors_dir = os.path.join(self.saguaro_dir, "vectors")
        self._lock_manager = RepoLockManager(self.saguaro_dir)
        self._native_runtime = get_native_runtime()
        self._use_gateway = bool(use_gateway)
        self._projection_cache: dict[tuple[int, int], bytearray] = {}
        self._vocab_size = int(os.getenv("SAGUARO_EMBED_VOCAB_SIZE", "16384"))
        self._trace = None
        self._parse_service = None
        self._graph_service = None
        self._query_service = None

    def _ensure_ready(self) -> None:
        os.makedirs(self.saguaro_dir, exist_ok=True)
        os.makedirs(self.vectors_dir, exist_ok=True)

    def _ensure_trace(self):
        if self._trace is None:
            from saguaro.agents.perception import TracePerception

            self._trace = TracePerception(self.repo_path)
        return self._trace

    def _ensure_query_runtime(self):
        if self._query_service is None:
            from saguaro.services.platform import GraphService, ParseService, QueryService

            self._parse_service = ParseService(self.repo_path)
            self._graph_service = GraphService(self.repo_path, self._parse_service)
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
                refresh_index=None,
            )
        return self._query_service

    def _shared_index_lock(self, operation: str):
        return self._lock_manager.acquire("index", mode="shared", operation=operation)

    def _load_stats(self) -> dict[str, Any]:
        stats = get_repo_stats_and_config(self.repo_path)
        stats.update(self._load_index_stats())
        calibration = load_query_calibration(self.saguaro_dir)
        if calibration:
            stats["query_confidence"] = calibration
        return stats

    def _load_index_stats(self) -> dict[str, Any]:
        return load_index_stats(self.saguaro_dir)

    def _default_native_threads(self) -> int:
        profile = load_runtime_profile(self.repo_path)
        layout = dict(profile.get("selected_runtime_layout") or {})
        query_threads = int(layout.get("query_threads", 0) or 0)
        if query_threads > 0:
            return query_threads
        with contextlib.suppress(Exception):
            return int(self._native_runtime.default_threads())
        return max(1, int(os.cpu_count() or 1))

    def _projection(self, vocab_size: int, active_dim: int) -> bytearray:
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

    def _extract_terms(self, text: str, limit: int = 32) -> list[str]:
        expanded = _CAMEL_RE.sub(r"\1 \2", text or "")
        expanded = re.sub(r"[/_.:-]+", " ", expanded)
        seen: list[str] = []
        for token in _IDENT_RE.findall(expanded):
            for normalized in dict.fromkeys(
                part.lower()
                for part in re.split(r"[_\-.]+", token)
                if part and not part.isdigit()
            ):
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
            if len(seen) >= limit:
                break
        return seen

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
        weights = {"symbol": 1.75, "path": 1.35, "doc": 1.0}
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
                num_threads=self._default_native_threads(),
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

    def _candidate_terms(self, item: dict[str, Any]) -> set[str]:
        terms = set()
        for key in ("terms", "symbol_terms", "path_terms", "doc_terms"):
            for term in item.get(key, []) or []:
                if isinstance(term, str):
                    terms.add(term.lower())
        terms.update(self._extract_terms(item.get("name", ""), limit=16))
        terms.update(self._extract_terms(item.get("file", ""), limit=32))
        return terms

    def _hybrid_rerank(
        self, query_text: str, results: list[dict[str, Any]], k: int
    ) -> list[dict[str, Any]]:
        if not results:
            return []
        query_terms = set(self._extract_terms(query_text, limit=24))
        query_lower = (query_text or "").lower()
        reranked: list[dict[str, Any]] = []
        for item in results:
            semantic = float(item.get("score", 0.0) or 0.0)
            candidate_terms = self._candidate_terms(item)
            lexical = len(query_terms & candidate_terms) / max(len(query_terms), 1)
            name = str(item.get("name", "")).lower()
            file_path = str(item.get("file", "")).lower()
            exact_bonus = 0.0
            if name and name in query_lower:
                exact_bonus += 0.35
            if file_path and file_path in query_lower:
                exact_bonus += 0.45
            for term in query_terms:
                if term and term in name:
                    exact_bonus += 0.08
                elif term and term in file_path:
                    exact_bonus += 0.04
            updated = dict(item)
            updated["semantic_score"] = semantic
            updated["lexical_score"] = lexical
            updated["score"] = semantic * 0.55 + lexical * 0.35 + min(exact_bonus, 0.6)
            reranked.append(updated)
        reranked.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return reranked[:k]

    def _canonical_rel_path(self, value: str) -> str:
        return canonicalize_rel_path(value, repo_path=self.repo_path)

    def _result_is_in_repo(self, item: dict[str, Any]) -> bool:
        file_path = item.get("file")
        if not file_path:
            return False
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
        if is_excluded_path(rel, patterns=[], repo_path=self.repo_path):
            return False
        return True

    def _store_schema_path(self) -> str:
        return os.path.join(self.saguaro_dir, "index_schema.json")

    def _load_store_schema(self) -> dict[str, Any]:
        path = self._store_schema_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle) or {}
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _check_store_compatibility(self, expected_dim: int) -> dict[str, Any]:
        schema_path = self._store_schema_path()
        meta_path = os.path.join(self.vectors_dir, "index_meta.json")
        artifact_paths = {
            "vectors.bin": os.path.join(self.vectors_dir, "vectors.bin"),
            "norms.bin": os.path.join(self.vectors_dir, "norms.bin"),
            "metadata.json": os.path.join(self.vectors_dir, "metadata.json"),
            "index_meta.json": meta_path,
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
                return {"incompatible": True, "reason": "Missing index schema metadata"}
        if not os.path.exists(meta_path):
            return {"incompatible": False}
        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle) or {}
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
        if int(schema.get("embedding_schema_version", 0) or 0) not in {
            0,
            INDEX_SCHEMA_VERSION,
        }:
            return {"incompatible": True, "reason": "Embedding schema version changed"}
        stored_repo = schema.get("repo_path")
        if stored_repo and os.path.abspath(stored_repo) != self.repo_path:
            return {
                "incompatible": True,
                "reason": "Index was created for a different repository path",
            }
        return {"incompatible": False}

    def _integrity_report(self) -> dict[str, Any]:
        try:
            payload = load_manifest(self.saguaro_dir)
            validation = validate_manifest(self.saguaro_dir, payload)
            if (
                validation.get("status") == "mismatch"
                and self._graph_manifest_mismatch_only(validation.get("mismatches", []))
                and self._refresh_graph_manifest_record()
            ):
                payload = load_manifest(self.saguaro_dir)
                validation = validate_manifest(self.saguaro_dir, payload)
            return {
                "manifest_generation_id": payload.get("generation_id"),
                "status": validation.get("status", "missing"),
                "mismatches": validation.get("mismatches", []),
                "summary": payload.get("summary", {}),
                "locks": self._lock_manager.status(),
                "manifest_path": manifest_path(self.saguaro_dir),
            }
        except Exception as exc:
            return {
                "manifest_generation_id": None,
                "status": "corrupt",
                "mismatches": [str(exc)],
                "summary": {},
                "locks": self._lock_manager.status(),
                "manifest_path": manifest_path(self.saguaro_dir),
            }

    @staticmethod
    def _graph_manifest_mismatch_only(mismatches: list[str]) -> bool:
        if not mismatches:
            return False
        allowed_prefixes = {
            "size:graph/graph.json",
            "mtime:graph/graph.json",
            "manifest_missing:graph/graph.json",
        }
        return set(str(item) for item in mismatches).issubset(allowed_prefixes)

    def _refresh_graph_manifest_record(self) -> bool:
        payload = load_manifest(self.saguaro_dir)
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        rel_path = INDEX_ARTIFACTS.get("graph/graph.json")
        if not rel_path:
            return False
        full_path = os.path.join(self.saguaro_dir, rel_path)
        if not os.path.exists(full_path):
            return False
        artifacts["graph/graph.json"] = snapshot_artifact(full_path, rel_path)
        summary = dict(payload.get("summary") or {})
        graph_stats = self._graph_stats()
        summary.update(
            {
                "graph_files": int(graph_stats.get("files", 0) or 0),
                "graph_nodes": int(graph_stats.get("nodes", 0) or 0),
                "graph_edges": int(graph_stats.get("edges", 0) or 0),
            }
        )
        payload["summary"] = summary
        atomic_write_json(
            manifest_path(self.saguaro_dir),
            payload,
            indent=2,
            sort_keys=True,
        )
        return True

    def _duplicate_tree_state(self) -> dict[str, Any]:
        shim_root = os.path.join(self.repo_path, "saguaro")
        shim_files = {"__init__.py", "__main__.py"}
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
            payload["saguaro_dir_present"] and payload["Saguaro_dir_present"] and not shim_only
        )
        return payload

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
        }

    def _graph_stats(self) -> dict[str, Any]:
        graph_path = os.path.join(self.saguaro_dir, "graph", "graph.json")
        stats_path = os.path.join(self.saguaro_dir, "graph", "graph_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, encoding="utf-8") as handle:
                    payload = json.load(handle) or {}
            except Exception as exc:
                return {"status": "corrupt", "graph_path": graph_path, "reason": str(exc)}
            return {
                "status": "ready",
                "graph_path": graph_path,
                "files": int(payload.get("files", 0) or 0),
                "nodes": int(payload.get("nodes", 0) or 0),
                "edges": int(payload.get("edges", 0) or 0),
                "graph_coverage_percent": float(payload.get("graph_coverage_percent", 0.0) or 0.0),
                "incremental": bool(payload.get("incremental", False)),
            }
        if not os.path.exists(graph_path):
            manifest_summary = dict(load_manifest(self.saguaro_dir).get("summary") or {})
            if manifest_summary:
                return {
                    "status": "ready",
                    "graph_path": graph_path,
                    "files": int(manifest_summary.get("graph_files", 0) or 0),
                    "nodes": int(manifest_summary.get("graph_nodes", 0) or 0),
                    "edges": int(manifest_summary.get("graph_edges", 0) or 0),
                    "graph_coverage_percent": 0.0,
                    "incremental": True,
                }
            return {"status": "missing", "graph_path": graph_path}
        try:
            manifest_summary = dict(load_manifest(self.saguaro_dir).get("summary") or {})
        except Exception as exc:
            return {"status": "corrupt", "graph_path": graph_path, "reason": str(exc)}
        if manifest_summary:
            return {
                "status": "ready",
                "graph_path": graph_path,
                "files": int(manifest_summary.get("graph_files", 0) or 0),
                "nodes": int(manifest_summary.get("graph_nodes", 0) or 0),
                "edges": int(manifest_summary.get("graph_edges", 0) or 0),
                "graph_coverage_percent": 0.0,
                "incremental": True,
            }
        return {
            "status": "missing",
            "graph_path": graph_path,
        }

    def _freshness_report(self) -> dict[str, Any]:
        meta_path = os.path.join(self.vectors_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            return {"status": "not_indexed"}
        mtime = os.path.getmtime(meta_path)
        return {
            "status": "ready",
            "last_update_ts": mtime,
            "last_update_fmt": time.ctime(mtime),
            "age_seconds": round(max(0.0, time.time() - mtime), 3),
        }

    def _storage_report(self) -> dict[str, Any]:
        total = 0.0
        parts: dict[str, float] = {}
        for name in ("vectors.bin", "norms.bin", "metadata.json", "index_meta.json"):
            path = os.path.join(self.vectors_dir, name)
            size_mb = (os.path.getsize(path) / (1024.0 * 1024.0)) if os.path.exists(path) else 0.0
            parts[name] = round(size_mb, 3)
            total += size_mb
        parts["total_mb"] = round(total, 3)
        return parts

    def _query_gateway_status(self) -> dict[str, Any]:
        from saguaro.query.gateway import read_gateway_state

        state = read_gateway_state(self.repo_path)
        status = "enabled"
        if state.get("status") != "running":
            status = "enabled"
        return {
            "status": status,
            "mode": "resident_queue",
            "running": bool(state.get("status") == "running"),
            "socket_path": state.get("socket_path"),
            "limits": dict(state.get("limits") or {}),
            "metrics": dict(state.get("metrics") or {}),
            "query_many_available": True,
        }

    def _scope_from_level(self, level: int) -> str:
        return {0: "local", 1: "package", 2: "project", 3: "global"}.get(
            int(level), "global"
        )

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
                symbol = str(row.get("qualified_name") or row.get("name") or "").lower()
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

    def health(self) -> dict[str, Any]:
        self._ensure_ready()
        repo_stats = get_repo_stats_and_config(self.repo_path)
        profile = load_runtime_profile(self.repo_path)
        if not dict(profile.get("selected_runtime_layout") or {}):
            profile = calibrate_runtime_profile(self.repo_path)
        tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))
        return {
            "freshness": self._freshness_report(),
            "storage": self._storage_report(),
            "runtime": {
                "repo_path": self.repo_path,
                "cpu_count": os.cpu_count() or 1,
                "execution_target": "cpu",
                "index_mode": "cpu-first-local",
                "backend": "fastpath",
            },
            "corpus": {
                "candidate_files": int(repo_stats.get("candidate_files", 0) or 0),
                "loc": int(repo_stats.get("loc", 0) or 0),
                "languages": dict(repo_stats.get("languages") or {}),
                "active_dim": int(repo_stats.get("active_dim", 4096) or 4096),
                "total_dim": int(repo_stats.get("total_dim", 8192) or 8192),
            },
            "graph": self._graph_stats(),
            "governance": {
                "status": "ready",
                "total_tracked_files": len(tracker.state),
                "verified_files": sum(
                    1 for entry in tracker.state.values() if entry.get("verified", False)
                ),
            },
            "native_compute": collect_native_compute_report(),
            "native_capabilities": collect_native_capability_report(),
            "storage_layout": self._storage_layout_report(),
            "runtime_profile": profile,
            "integrity": self._integrity_report(),
            "locks": self._lock_manager.status(),
            "duplicate_trees": self._duplicate_tree_state(),
            "query_gateway": self._query_gateway_status(),
        }

    def doctor(self) -> dict[str, Any]:
        self._ensure_ready()
        with self._shared_index_lock("doctor"):
            health = self.health()
            tracker = IndexTracker(os.path.join(self.saguaro_dir, "tracking.json"))
            stale_candidates = 0
            if tracker.state:
                tracked_paths = sorted(tracker.state.keys())
                stale_candidates = len(tracker.filter_needs_indexing(tracked_paths))
            native_abi = {"ok": False, "reason": "native indexer unavailable"}
            with contextlib.suppress(Exception):
                native_abi = get_native_indexer().abi_self_test()
            compatibility = self._check_store_compatibility(
                expected_dim=int(self._load_stats().get("active_dim", 4096) or 4096)
            )
            integrity = self._integrity_report()
            doctor_ok = (
                bool(native_abi.get("ok", False))
                and not bool(compatibility.get("incompatible", False))
                and integrity.get("status") == "ready"
            )
            return {
                "status": "ok" if doctor_ok else "warning",
                "runtime": {"repo_path": self.repo_path, "backend": "fastpath"},
                "health": health,
                "index": {
                    "tracked_files": len(tracker.state),
                    "stale_candidates": stale_candidates,
                    "compatibility": compatibility,
                },
                "integrity": integrity,
                "duplicate_trees": self._duplicate_tree_state(),
                "native_abi": native_abi,
                "native_capabilities": health.get("native_capabilities", {}),
            }

    def abi(self, action: str = "verify") -> dict[str, Any]:
        op = str(action or "verify").strip().lower()
        if op != "verify":
            return {"status": "error", "action": op, "message": "Fast path supports verify only."}
        report = self.doctor()
        return {
            "status": report.get("status", "ok"),
            "action": "verify",
            "native_abi": dict(report.get("native_abi") or {}),
            "native_capabilities": dict(report.get("native_capabilities") or {}),
            "integrity": dict(report.get("integrity") or {}),
            "index": dict(report.get("index") or {}),
            "runtime": dict(report.get("runtime") or {}),
        }

    def ffi_audit(self, path: str = ".", *, limit: int = 200) -> dict[str, Any]:
        payload = self._ensure_trace().ffi_boundaries(path=path, limit=limit)
        payload.setdefault("status", "ok")
        payload["action"] = "audit"
        return payload

    def math_parse(self, path: str = ".") -> dict[str, Any]:
        from saguaro.math import MathEngine

        return MathEngine(self.repo_path).parse(path)

    def cpu_scan(
        self,
        path: str = ".",
        *,
        arch: str = "x86_64-avx2",
        limit: int = 20,
    ) -> dict[str, Any]:
        from saguaro.cpu import CPUScanner

        return CPUScanner(self.repo_path).scan(path=path, arch=arch, limit=limit)

    def prime_query_runtime(self, *, strategy: str = "hybrid") -> dict[str, Any]:
        self._ensure_ready()
        query_service = self._ensure_query_runtime()
        requested = str(strategy or "hybrid").strip().lower()
        resolved = query_service._TASK_STRATEGIES.get(requested, "hybrid")
        started = time.perf_counter()
        context = query_service._prepare_query_context(strategies={resolved})
        return {
            "status": "ok",
            "strategy": requested,
            "execution_strategy": resolved,
            "prepared_in_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "graph_nodes": len(dict(context.get("graph") or {}).get("nodes", {})),
            "store_ready": context.get("store") is not None,
        }

    def query(
        self,
        text: str,
        *,
        k: int = 5,
        file: str | None = None,
        level: int = 3,
        strategy: str = "hybrid",
        explain: bool = False,
        scope: str = "global",
        dedupe_by: str = "entity",
    ) -> dict[str, Any]:
        if self._use_gateway and os.getenv("SAGUARO_QUERY_GATEWAY_DISABLE", "").lower() not in {
            "1",
            "true",
            "yes",
        }:
            from saguaro.query.gateway import request_gateway

            payload = request_gateway(
                self.repo_path,
                {
                    "action": "query",
                    "text": text,
                    "k": k,
                    "file": file,
                    "level": level,
                    "strategy": strategy,
                    "explain": explain,
                    "scope": scope,
                    "dedupe_by": dedupe_by,
                    "timeout_seconds": float(os.getenv("SAGUARO_QUERY_GATEWAY_TIMEOUT", "30") or 30.0),
                },
            ) 
            if payload.get("status") not in {"error", "busy"}:
                return payload
        self._ensure_ready()
        query_service = self._ensure_query_runtime()
        with self._shared_index_lock("query"):
            integrity = self._integrity_report()
            compatibility = self._check_store_compatibility(
                expected_dim=int(self._load_stats().get("active_dim", 4096) or 4096)
            )
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
            query_result = query_service.query(
                text=text,
                k=k,
                strategy=strategy,
                explain=explain,
                auto_refresh=False,
            )
            results = [
                item
                for item in list(query_result.get("results", []))
                if self._result_is_in_repo(item)
            ]

        if file:
            file_abs = os.path.abspath(
                file if os.path.isabs(file) else os.path.join(self.repo_path, file)
            )
            file_rel = os.path.relpath(file_abs, self.repo_path)
            filtered = []
            for item in results:
                item_file = item.get("file", "")
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
        deduped_results = self._dedupe_results(results, dedupe_by=dedupe_by)
        for idx, item in enumerate(deduped_results, start=1):
            item["rank"] = idx
            item["scope"] = scope_value
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
            "integrity": integrity,
        }
