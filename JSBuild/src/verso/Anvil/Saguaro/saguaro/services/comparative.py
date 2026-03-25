"""Comparative multi-corpus analysis services for Saguaro."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from typing import Any

import numpy as np

from saguaro.analysis.comparative_ranker import ComparativeRankFusion, compress_program_groups
from saguaro.analysis.disparate_relations import DisparateRelationSynthesizer
from saguaro.analysis.path_proofs import PathProofBuilder
from saguaro.analysis.program_ir import ProgramIRBuilder
from saguaro.analysis.report import ReportGenerator
from saguaro.build_system.ingestor import BuildGraphIngestor
from saguaro.indexing.comparative_runtime import ComparativeIndexRuntime
from saguaro.indexing.native_indexer_bindings import NativeIndexerError, get_native_indexer
from saguaro.parsing.parser import SAGUAROParser
from saguaro.query.corpus_rules import canonicalize_rel_path, classify_file_role
from saguaro.reality.store import RealityGraphStore
from saguaro.state.ledger import StateLedger
from saguaro.storage.atomic_fs import atomic_write_json
from saguaro.storage.locks import RepoLockManager


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
_STRUCTURAL_LANGUAGES = {"python", "c", "cpp", "c_header", "cpp_header", "rust", "go"}
_SHALLOW_LANGUAGES = {
    "markdown",
    "json",
    "yaml",
    "toml",
    "ql",
    "qll",
    "shell",
    "ini",
    "xml",
    "html",
    "css",
}
_EXTERNAL_TEMP_EXCLUDE_PATTERNS = (
    ".git/**",
    ".venv/**",
    "venv/**",
    "env/**",
    "ENV/**",
    "node_modules/**",
    "bower_components/**",
    "vendor/**",
    "third_party/**",
    "third-party/**",
    "deps/**",
    "external/**",
    "build/**",
    "dist/**",
    "out/**",
    "target/**",
    "tmp/**",
    "temp/**",
    ".cache/**",
    ".pytest_cache/**",
    ".mypy_cache/**",
    ".ruff_cache/**",
    ".tox/**",
    ".nox/**",
    "__pycache__/**",
    "coverage/**",
    ".next/**",
    ".nuxt/**",
    ".terraform/**",
)


def _slug(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip()).strip("-").lower()
    return text or "corpus"


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(str(text or ""))}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


class ComparativeAnalysisService:
    """Create isolated corpus sessions and compare them against a target corpus."""

    def __init__(
        self,
        repo_path: str,
        *,
        state_ledger: StateLedger | None = None,
    ) -> None:
        self.repo_path = os.path.abspath(repo_path)
        self.saguaro_dir = os.path.join(self.repo_path, ".saguaro")
        self.corpora_dir = os.path.join(self.saguaro_dir, "corpora")
        self.reports_dir = os.path.join(self.saguaro_dir, "comparative", "reports")
        self.phasepacks_dir = os.path.join(self.saguaro_dir, "comparative", "phasepacks")
        self.leaderboards_dir = os.path.join(self.saguaro_dir, "comparative", "leaderboards")
        self.benchmarks_dir = os.path.join(self.saguaro_dir, "comparative", "benchmarks")
        self.proof_graphs_dir = os.path.join(self.saguaro_dir, "comparative", "proof_graphs")
        self.markdown_reports_dir = os.path.join(self.repo_path, "comparative_reports")
        self.state_ledger = state_ledger or StateLedger(self.repo_path)
        self._lock_manager = RepoLockManager(self.saguaro_dir)
        self._api_cache: dict[tuple[str, str], Any] = {}
        self._runtime_cache: dict[tuple[str, str], ComparativeIndexRuntime] = {}
        self._native_indexer: Any | None = None
        self._parser: SAGUAROParser | None = None
        self._disparate_synthesizer = DisparateRelationSynthesizer()
        self._rank_fusion = ComparativeRankFusion()
        self._path_proof_builder = PathProofBuilder()
        self._program_ir_builder = ProgramIRBuilder()
        self._target_twin_cache: dict[str, dict[str, Any]] = {}
        self._signature_cache: dict[str, dict[str, Any]] = {}
        os.makedirs(self.corpora_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.phasepacks_dir, exist_ok=True)
        os.makedirs(self.leaderboards_dir, exist_ok=True)
        os.makedirs(self.benchmarks_dir, exist_ok=True)
        os.makedirs(self.proof_graphs_dir, exist_ok=True)
        os.makedirs(self.markdown_reports_dir, exist_ok=True)

    @staticmethod
    def _load_api_cls() -> type[Any]:
        from saguaro.api import SaguaroAPI

        return SaguaroAPI

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
        act = str(action or "list").strip().lower()
        if act == "create":
            if not path:
                raise ValueError("path is required for corpus create")
            return self.create_session(
                path=path,
                corpus_id=corpus_id,
                alias=alias,
                ttl_hours=ttl_hours,
                quarantine=quarantine,
                trust_level=trust_level,
                build_profile=build_profile,
                rebuild=rebuild,
            )
        if act == "benchmark":
            if not path:
                raise ValueError("path is required for corpus benchmark")
            return self.benchmark_session(
                path=path,
                alias=alias,
                ttl_hours=ttl_hours,
                quarantine=quarantine,
                trust_level=trust_level,
                build_profile=build_profile,
                batch_sizes=batch_sizes,
                file_batch_sizes=file_batch_sizes,
                iterations=iterations,
                reuse_check=reuse_check,
            )
        if act == "show":
            if not corpus_id:
                raise ValueError("corpus_id is required for corpus show")
            session = self.state_ledger.get_corpus_session(corpus_id, touch=True)
            return {
                "status": "ok" if session else "missing",
                "session": session,
                "analysis_pack": self._load_session_pack(corpus_id) if session else None,
            }
        if act == "gc":
            return self.state_ledger.gc_corpus_sessions()
        return self.state_ledger.list_corpus_sessions(include_expired=include_expired)

    def create_session(
        self,
        *,
        path: str,
        corpus_id: str | None = None,
        alias: str | None = None,
        ttl_hours: float = 24.0,
        quarantine: bool = True,
        trust_level: str = "medium",
        build_profile: str = "auto",
        rebuild: bool = False,
    ) -> dict[str, Any]:
        root_path = self._resolve_input_path(path)
        kind = self._infer_corpus_kind(root_path)
        candidate_id = corpus_id or self._corpus_id_for_path(root_path, alias=alias)
        session_dir = os.path.join(self.corpora_dir, candidate_id)
        build_scope = (
            self._lock_manager.external_corpus_build(
                operation="corpus_create",
                corpus_id=candidate_id,
            )
            if kind != "primary"
            else nullcontext()
        )
        with build_scope:
            if not rebuild:
                existing_by_path = self._session_for_root(root_path)
                if existing_by_path:
                    existing_pack = self._load_session_pack(
                        str(existing_by_path.get("corpus_id") or "")
                    )
                    if existing_pack and str(existing_pack.get("producer") or "").endswith(
                        "native_index"
                    ):
                        self.state_ledger.get_corpus_session(
                            str(existing_by_path.get("corpus_id") or ""),
                            touch=True,
                        )
                        return {
                            "status": "exists",
                            "session": existing_by_path,
                            "analysis_pack": self._session_pack_summary(
                                str(existing_by_path.get("corpus_id") or "")
                            ),
                        }
                existing = self.state_ledger.get_corpus_session(candidate_id, touch=True)
                if existing:
                    existing_pack = self._load_session_pack(candidate_id)
                    if existing_pack and str(existing_pack.get("producer") or "").endswith(
                        "native_index"
                    ):
                        return {
                            "status": "exists",
                            "session": existing,
                            "analysis_pack": self._session_pack_summary(candidate_id),
                        }
            if rebuild and kind != "primary":
                self._api_cache.pop((root_path, session_dir), None)
                self._runtime_cache.pop((root_path, session_dir), None)
            os.makedirs(session_dir, exist_ok=True)

            index_root = self.saguaro_dir if kind == "primary" else session_dir
            manifest_path = os.path.join(index_root, "index_manifest.json")
            index_result: dict[str, Any] | None = None
            if rebuild or not os.path.exists(manifest_path):
                runtime = self._session_runtime_for_root(root_path, index_root=index_root)
                refresh_scope = (
                    self._lock_manager.target_refresh(
                        operation="corpus_create",
                        target_id=candidate_id,
                    )
                    if kind == "primary"
                    else nullcontext()
                )
                with refresh_scope:
                    index_result = runtime.index(
                        path=".",
                        force=bool(rebuild),
                        incremental=not rebuild,
                        prune_deleted=True,
                    )

            pack = self._build_native_pack(
                index_root=index_root,
                corpus_id=candidate_id,
                root_path=root_path,
                kind=kind,
            )
            build_fingerprint = self._build_fingerprint(root_path, pack, build_profile=build_profile)
            language_truth_matrix = self._language_truth_matrix(
                pack,
                build_fingerprint=build_fingerprint,
            )
            capability_matrix = self._capability_matrix(
                pack,
                build_fingerprint=build_fingerprint,
                language_truth_matrix=language_truth_matrix,
            )
            parser_environment = dict(language_truth_matrix.get("parser_environment") or {})
            pack["build_fingerprint"] = build_fingerprint
            pack["language_truth_matrix"] = language_truth_matrix
            pack["capability_matrix"] = capability_matrix
            pack["semantic_inventory"] = self._semantic_inventory(pack, candidate_id)
            pack["schema_version"] = "comparative_native_pack.v2"
            pack["parser_failures"] = list(
                (language_truth_matrix.get("parser_environment") or {}).get(
                    "grammar_probe_failures",
                    [],
                )
            )
            pack["corpus_quality"] = self._corpus_quality_metrics(
                pack=pack,
                build_fingerprint=build_fingerprint,
                language_truth_matrix=language_truth_matrix,
            )
            pack["compare_pack"] = self._compare_ready_pack(pack)

            created_at = time.time()
            ttl_seconds = max(300, int(float(ttl_hours or 24.0) * 3600))
            session = {
                "schema_version": "corpus_session.v1",
                "corpus_id": candidate_id,
                "alias": alias or os.path.basename(root_path.rstrip(os.sep)) or candidate_id,
                "kind": kind,
                "status": "ready",
                "root_path": root_path,
                "relative_root": self._relative_root(root_path),
                "quarantine": bool(quarantine),
                "trust_level": trust_level,
                "build_profile": build_profile,
                "snapshot_digest": self._snapshot_digest(pack),
                "build_fingerprint": build_fingerprint,
                "language_truth_matrix": language_truth_matrix,
                "capability_matrix": capability_matrix,
                "parser_environment": parser_environment,
                "focus_cone": self._focus_cone(root_path, pack),
                "index_root": index_root,
                "artifact_paths": {
                    "manifest": os.path.join(session_dir, "manifest.json"),
                    "analysis_pack": os.path.join(session_dir, "analysis_pack.json"),
                    "compare_pack": self._compare_pack_path(session_dir),
                    "signature_cache": self._signature_cache_path(session_dir, pack),
                    "index_manifest": os.path.join(index_root, "index_manifest.json"),
                    "vectors_metadata": os.path.join(index_root, "vectors", "metadata.json"),
                    "graph": os.path.join(index_root, "graph", "graph.json"),
                },
                "created_at": created_at,
                "updated_at": created_at,
                "last_accessed_at": created_at,
                "ttl_seconds": ttl_seconds,
                "expires_at": created_at + ttl_seconds,
                "telemetry": {
                    "corpus_session_boot_ms": 0.0,
                    "quarantine_breach_count": 0,
                    "focus_corpus_file_count": int(pack.get("file_count", 0) or 0),
                    "native_indexed": True,
                    "parser_environment_ready": bool(
                        parser_environment.get("parser_environment_ready")
                    ),
                    "language_truth_rows": int(
                        len((language_truth_matrix.get("per_language") or {}))
                    ),
                    "external_corpus_build_lock_ms": float(
                        getattr(build_scope, "wait_ms", 0.0) or 0.0
                    ),
                    "ready_manifest_reused": bool(
                        kind == "primary" and not rebuild and index_result is None
                    ),
                },
            }
            if index_result:
                session["telemetry"]["native_index_result"] = {
                    "indexed_files": int(index_result.get("indexed_files", 0) or 0),
                    "indexed_entities": int(index_result.get("indexed_entities", 0) or 0),
                    "backend": str(index_result.get("backend") or ""),
                }
            started = time.perf_counter()
            atomic_write_json(session["artifact_paths"]["analysis_pack"], pack, indent=2, sort_keys=True)
            atomic_write_json(
                session["artifact_paths"]["compare_pack"],
                dict(pack.get("compare_pack") or {}),
                indent=2,
                sort_keys=True,
            )
            atomic_write_json(session["artifact_paths"]["manifest"], session, indent=2, sort_keys=True)
            self._persist_signature_cache(session, pack)
            session["telemetry"]["corpus_session_boot_ms"] = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            self.state_ledger.create_corpus_session(session)
            return {
                "status": "ok",
                "session": session,
                "analysis_pack": self._pack_summary(pack),
                "manifest_reused": bool(
                    kind == "primary" and not rebuild and index_result is None
                ),
            }

    def corpus_query(
        self,
        text: str,
        *,
        corpus_ids: list[str] | None = None,
        k: int = 5,
        merge: bool = True,
    ) -> dict[str, Any]:
        sessions = self._selected_sessions(corpus_ids)
        results: list[dict[str, Any]] = []
        for session in sessions:
            api = self._session_api(session)
            query_result = api.query(text, k=max(1, int(k or 5)))
            native_results = list(query_result.get("results") or [])
            if not native_results:
                continue
            for item in native_results:
                path_value = str(item.get("file") or "")
                name = str(item.get("name") or path_value or session["corpus_id"])
                kind = str(item.get("type") or "symbol")
                results.append(
                    {
                        **item,
                        "corpus_id": session["corpus_id"],
                        "qualified_symbol_id": self._qualified_symbol_id(
                            session["corpus_id"],
                            path_value,
                            name,
                            kind,
                        ),
                        "scope": "corpus",
                    }
                )
        results.sort(
            key=lambda item: (-float(item.get("score") or 0.0), item.get("corpus_id", ""), item.get("file", ""))
        )
        for index, item in enumerate(results, start=1):
            item["rank"] = index
        if not merge:
            return {
                "query": text,
                "results_by_corpus": {
                    session["corpus_id"]: [item for item in results if item["corpus_id"] == session["corpus_id"]][:k]
                    for session in sessions
                },
                "corpora": [session["corpus_id"] for session in sessions],
            }
        return {
            "query": text,
            "results": results[:k],
            "corpora": [session["corpus_id"] for session in sessions],
            "mode": "federated",
        }

    def resolve_symbol(
        self,
        symbol: str,
        *,
        corpus_ids: list[str] | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        target_name = str(symbol or "").strip()
        if not target_name:
            return {"status": "error", "message": "symbol is required", "matches": []}
        for session in self._selected_sessions(corpus_ids):
            pack = self._load_session_pack(session["corpus_id"])
            if not pack:
                continue
            for file_record in pack.get("files", []):
                for symbol_record in file_record.get("symbols", []):
                    name = str(symbol_record.get("name") or "")
                    if name != target_name and not name.endswith(f".{target_name}"):
                        continue
                    matches.append(
                        {
                            "corpus_id": session["corpus_id"],
                            "file": str(file_record.get("path") or ""),
                            "line": int(symbol_record.get("line") or 1),
                            "name": name,
                            "kind": str(symbol_record.get("kind") or "symbol"),
                            "qualified_symbol_id": self._qualified_symbol_id(
                                session["corpus_id"],
                                str(file_record.get("path") or ""),
                                name,
                                symbol_record.get("kind", "symbol"),
                            ),
                        }
                    )
        matches.sort(key=lambda item: (item["name"], item["corpus_id"], item["file"]))
        status = "resolved" if len(matches) == 1 else ("ambiguous" if matches else "missing")
        return {"status": status, "matches": matches[:limit]}

    def slice_symbol(
        self,
        symbol: str,
        *,
        corpus_id: str,
        depth: int = 1,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        session = self.state_ledger.get_corpus_session(corpus_id, touch=True)
        if not session:
            return {
                "error": f"Unknown corpus_id: {corpus_id}",
                "type": "CORPUS_MISS",
            }
        preferred_file = file_path
        if not preferred_file:
            resolved = self.resolve_symbol(symbol, corpus_ids=[corpus_id], limit=5)
            if resolved.get("status") == "resolved":
                preferred_file = resolved["matches"][0]["file"]
            elif resolved.get("status") == "ambiguous":
                return {
                    "error": "Symbol is ambiguous across corpus candidates.",
                    "type": "SYMBOL_AMBIGUOUS",
                    "symbol": symbol,
                    "matches": resolved.get("matches", []),
                }
        api = self._session_api(session)
        result = api.slice(symbol, depth=depth, file_path=preferred_file)
        result["corpus_id"] = corpus_id
        self._requalify_slice_result(result, corpus_id=corpus_id)
        return result

    def compare(
        self,
        *,
        target: str = ".",
        candidates: list[str] | None = None,
        corpus_ids: list[str] | None = None,
        fleet_root: str | None = None,
        top_k: int = 10,
        ttl_hours: float = 72.0,
        sequential_fleet: bool | None = None,
        reuse_only: bool = False,
        mode: str = "flight_plan",
        emit_phasepack: bool = True,
        explain_paths: bool = True,
        portfolio_top_n: int = 12,
        calibration_profile: str = "balanced",
        evidence_budget: int = 12,
        export_datatables: bool = False,
    ) -> dict[str, Any]:
        compare_started = time.perf_counter()
        with self._lock_manager.compare_read(operation="compare") as compare_lock:
            target_result = self._ensure_session(
                target,
                alias="target",
                ttl_hours=ttl_hours,
            )
            target_session = target_result["session"]
            target_pack = self._load_session_pack(target_session["corpus_id"])
        if not target_pack:
            raise ValueError("Target corpus pack is unavailable.")
        target_twin = self._target_twin(target_session, target_pack)
        target_manifest_reused = bool(
            target_result.get("manifest_reused")
            or str(target_result.get("status") or "") == "exists"
        )

        seen_ids: set[str] = set()
        pending_sessions: list[dict[str, Any]] = []
        fleet_processing_order: list[str] = []
        skipped_candidates: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        aggregate_ledger: list[dict[str, Any]] = []
        frontier_packets: list[dict[str, Any]] = []
        creation_ledger: list[dict[str, Any]] = []
        per_candidate_compare_ms: list[float] = []

        def queue_session(session: dict[str, Any]) -> None:
            corpus_id = str(session.get("corpus_id") or "")
            if not corpus_id or corpus_id in seen_ids:
                return
            seen_ids.add(corpus_id)
            pending_sessions.append(session)
            fleet_processing_order.append(corpus_id)

        for corpus_id in list(corpus_ids or []):
            session = self.state_ledger.get_corpus_session(corpus_id, touch=True)
            if session:
                queue_session(session)

        def compare_session(session: dict[str, Any]) -> dict[str, Any] | None:
            corpus_id = str(session.get("corpus_id") or "")
            if not corpus_id:
                return None
            candidate_pack = self._load_session_pack(corpus_id)
            if not candidate_pack:
                return None
            started = time.perf_counter()
            comparison = self._compare_packs(
                target_session=target_session,
                target_pack=target_pack,
                target_twin=target_twin,
                candidate_session=session,
                candidate_pack=candidate_pack,
                top_k=top_k,
            )
            comparison.setdefault("telemetry", {})
            comparison["telemetry"]["per_candidate_compare_ms"] = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            return comparison

        def load_candidate_session(candidate_path: str) -> dict[str, Any] | None:
            root_path = self._resolve_input_path(candidate_path)
            if reuse_only:
                existing = self._session_for_root(root_path)
                if existing:
                    corpus_id = str(existing.get("corpus_id") or "")
                    existing_pack = self._load_session_pack(corpus_id)
                    if existing_pack and str(existing_pack.get("producer") or "").endswith(
                        "native_index"
                    ):
                        self.state_ledger.get_corpus_session(corpus_id, touch=True)
                        return existing
                skipped_candidates.append(
                    {
                        "path": root_path,
                        "reason": "reuse_only_missing_corpus",
                    }
                )
                return None
            return self._ensure_session(root_path, ttl_hours=ttl_hours)["session"]

        for item in list(candidates or []):
            session = load_candidate_session(item)
            if session:
                queue_session(session)

        fleet_mode = bool(fleet_root)
        fleet_sequential = bool(fleet_mode) if sequential_fleet is None else bool(sequential_fleet)
        if fleet_mode:
            fleet_paths = self._discover_fleet_repos(fleet_root)
            if fleet_sequential:
                for path in fleet_paths:
                    if os.path.abspath(path) == os.path.abspath(target_session["root_path"]):
                        continue
                    session = load_candidate_session(path)
                    if not session:
                        continue
                    queue_session(session)
            else:
                for path in fleet_paths:
                    if os.path.abspath(path) == os.path.abspath(target_session["root_path"]):
                        continue
                    session = load_candidate_session(path)
                    if session:
                        queue_session(session)

        fleet_router_pruned = 0
        if fleet_mode and pending_sessions:
            pending_sessions, fleet_router_pruned = self._fleet_router(
                target_pack=target_pack,
                pending_sessions=pending_sessions,
                portfolio_top_n=portfolio_top_n,
            )

        max_parallel = 1 if fleet_sequential else min(
            max(1, (os.cpu_count() or 4) // 2),
            max(1, len(pending_sessions)),
            8,
        )
        if max_parallel > 1 and len(pending_sessions) > 1:
            future_map = {}
            ordered_results: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                for index, session in enumerate(pending_sessions):
                    future = executor.submit(compare_session, session)
                    future_map[future] = (index, session)
                for future in as_completed(future_map):
                    index, _session = future_map[future]
                    result = future.result()
                    if result:
                        ordered_results[index] = result
            for index in sorted(ordered_results):
                comparison = ordered_results[index]
                comparisons.append(comparison)
                aggregate_ledger.extend(comparison.get("port_ledger", []))
                frontier_packets.extend(comparison.get("frontier_packets", []))
                creation_ledger.extend(comparison.get("creation_ledger", []))
                per_candidate_compare_ms.append(
                    float((comparison.get("telemetry") or {}).get("per_candidate_compare_ms") or 0.0)
                )
        else:
            for session in pending_sessions:
                comparison = compare_session(session)
                if not comparison:
                    continue
                comparisons.append(comparison)
                aggregate_ledger.extend(comparison.get("port_ledger", []))
                frontier_packets.extend(comparison.get("frontier_packets", []))
                creation_ledger.extend(comparison.get("creation_ledger", []))
                per_candidate_compare_ms.append(
                    float((comparison.get("telemetry") or {}).get("per_candidate_compare_ms") or 0.0)
                )

        if fleet_mode:
            for session in pending_sessions:
                self._release_session_resources(session)

        candidate_count = len(comparisons)
        report_id = f"comparative_{int(time.time())}_{hashlib.sha1(json.dumps([item.get('candidate', {}).get('corpus_id') for item in comparisons], sort_keys=True).encode('utf-8')).hexdigest()[:8]}"
        best_of_breed = self._best_of_breed_synthesis(
            comparisons=comparisons,
            target_session=target_session,
            target_pack=target_pack,
        )
        native_programs = self._native_migration_programs(
            target_session=target_session,
            target_pack=target_pack,
            best_of_breed=best_of_breed,
            creation_ledger=creation_ledger,
        )
        for program in native_programs:
            program["flight_twin"] = self._simulate_program(program, target_pack)
        subsystem_upgrade_summary = self._subsystem_upgrade_summary(
            [
                relation
                for comparison in comparisons
                for relation in list(comparison.get("primary_recommendations") or [])
                + list(comparison.get("secondary_recommendations") or [])
            ],
            creation_entries=creation_ledger,
        )
        detailed_candidate_limit = min(
            candidate_count,
            max(3, min(8, top_k)),
        )
        report = {
            "schema_version": "comparative_report.v3",
            "report_id": report_id,
            "generated_at": time.time(),
            "target": self._session_summary(target_session, target_pack),
            "candidate_count": candidate_count,
            "comparisons": comparisons,
            "port_ledger": aggregate_ledger[: max(top_k * 3, 10)],
            "frontier_packets": frontier_packets[: max(top_k, 5)],
            "creation_ledger": creation_ledger[: max(top_k * 4, 12)],
            "best_of_breed_synthesis": best_of_breed,
            "native_migration_programs": native_programs,
            "subsystem_upgrade_summary": subsystem_upgrade_summary,
            "negative_evidence": self._negative_evidence_ledger(comparisons),
            "portfolio_leaderboard": self._portfolio_leaderboard(
                comparisons=comparisons,
                native_programs=native_programs,
            ),
            "mode": mode,
            "phase_packets": [],
            "render_policy": {
                "detailed_candidate_limit": detailed_candidate_limit,
                "summary_only_candidate_count": max(0, candidate_count - detailed_candidate_limit),
                "evidence_budget": int(evidence_budget),
                "explain_paths": bool(explain_paths),
            },
            "telemetry": {
                "corpus_session_count": candidate_count + 1,
                "aggregate_candidate_count": len(aggregate_ledger),
                "fleet_repo_count": candidate_count,
                "fleet_processing_mode": "sequential" if fleet_sequential else "batched",
                "fleet_processing_order": fleet_processing_order,
                "fleet_parallel_workers": max_parallel,
                "fleet_queue_depth": len(pending_sessions),
                "fleet_triage_prune_rate": round(
                    float(fleet_router_pruned) / max(1, len(pending_sessions) + fleet_router_pruned),
                    4,
                ),
                "fleet_repo_scanned": len(pending_sessions) + fleet_router_pruned,
                "fleet_repo_deep_compared": len(pending_sessions),
                "reuse_only": bool(reuse_only),
                "skipped_candidate_count": len(skipped_candidates),
                "skipped_candidates": skipped_candidates,
                "report_compile_ms": 0.0,
                "report_evidence_density": 0.0,
                "comparative_boot_ms": round((time.perf_counter() - compare_started) * 1000.0, 3),
                "compare_lock_wait_ms": float(compare_lock.wait_ms or 0.0),
                "compare_target_manifest_reuse_hits": int(target_manifest_reused),
                "target_twin_reuse_hits": int(target_twin.get("cache_hit", False)),
                "target_signature_build_ms": float(target_twin.get("signature_build_ms") or 0.0),
                "compare_target_cache_bytes": int(target_twin.get("cache_bytes") or 0),
                "pair_candidates_before_filter": int(
                    sum(
                        ((item.get("summary") or {}).get("pair_screening") or {}).get("pair_candidates_before_filter", 0)
                        for item in comparisons
                    )
                ),
                "pair_candidates_after_filter": int(
                    sum(
                        ((item.get("summary") or {}).get("pair_screening") or {}).get("pair_candidates_after_filter", 0)
                        for item in comparisons
                    )
                ),
                "per_candidate_compare_ms": per_candidate_compare_ms,
                "feature_winner_count": len(best_of_breed.get("feature_winners") or []),
                "creation_candidate_count": len(creation_ledger),
                "comparative_program_count": len(native_programs),
                "subsystem_upgrade_count": len(
                    [
                        row
                        for row in subsystem_upgrade_summary
                        if int(row.get("recommendation_count") or 0)
                        or int(row.get("creation_candidate_count") or 0)
                    ]
                ),
                "primary_recommendation_count": sum(
                    len(item.get("primary_recommendations") or [])
                    for item in comparisons
                ),
                "secondary_recommendation_count": sum(
                    len(item.get("secondary_recommendations") or [])
                    for item in comparisons
                ),
                "comparative_frontier_acceptance_rate": round(
                    len(native_programs) / max(1, len(frontier_packets)),
                    3,
                )
                if frontier_packets
                else 0.0,
                "winner_confidence_distribution": dict(
                    best_of_breed.get("winner_confidence_distribution") or {}
                ),
                "native_rewrite_promotion_count": sum(
                    1
                    for program in native_programs
                    if str(program.get("posture") or "") == "native_rewrite"
                ),
                "wrapper_budget_count": sum(
                    1
                    for program in native_programs
                    if str((program.get("recipe_ir") or {}).get("wrapper_policy") or "")
                    == "thin_python_wrapper_only"
                ),
                "migration_recipe_lowering_count": sum(
                    1 for program in native_programs if program.get("recipe_ir")
                ),
                "phasepack_export_requested": bool(emit_phasepack),
                "calibration_profile": calibration_profile,
                "proof_graph_count": sum(
                    1
                    for comparison in comparisons
                    for relation in list(comparison.get("primary_recommendations") or [])
                    if relation.get("proof_graph")
                ),
            },
        }
        if emit_phasepack:
            report["phase_packets"] = self._phase_packets_from_report(
                report=report,
                evidence_budget=evidence_budget,
            )
        aggregate_ledger.sort(
            key=lambda item: (-float(item.get("relation_score") or 0.0), item.get("corpus_id", ""), item.get("source_path", ""))
        )
        frontier_packets.sort(
            key=lambda item: (-float(item.get("priority") or 0.0), item.get("corpus_id", ""), item.get("title", ""))
        )
        creation_ledger.sort(
            key=lambda item: (
                -float(item.get("priority") or item.get("confidence_score") or 0.0),
                str(item.get("feature_family") or ""),
                str(item.get("candidate_corpus_id") or ""),
            )
        )
        for comparison in comparisons:
            comparison["report_id"] = report_id
            for entry in list(comparison.get("port_ledger") or []):
                entry["report_id"] = report_id
        for entry in aggregate_ledger:
            entry["report_id"] = report_id
        started = time.perf_counter()
        report["telemetry"]["report_evidence_density"] = round(
            (
                len(report.get("port_ledger") or [])
                + len(report.get("frontier_packets") or [])
                + len(report.get("creation_ledger") or [])
                + len(report.get("phase_packets") or [])
                + sum(
                    len(item.get("migration_recipes") or [])
                    for item in comparisons
                )
            )
            / max(1, len(comparisons)),
            3,
        )
        report["artifacts"] = self._persist_report_artifacts(report)
        report["telemetry"]["report_compile_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        atomic_write_json(
            str((report.get("artifacts") or {}).get("json_path") or ""),
            report,
            indent=2,
            sort_keys=True,
        )
        self._record_report_event(report)
        return report

    def benchmark_session(
        self,
        *,
        path: str,
        alias: str | None = None,
        ttl_hours: float = 24.0,
        quarantine: bool = True,
        trust_level: str = "medium",
        build_profile: str = "auto",
        batch_sizes: list[int] | None = None,
        file_batch_sizes: list[int | None] | None = None,
        iterations: int = 1,
        reuse_check: bool = True,
    ) -> dict[str, Any]:
        root_path = self._resolve_input_path(path)
        alias_base = alias or f"bench-{_slug(os.path.basename(root_path) or 'corpus')}"
        resolved_batch_sizes = sorted(
            {max(1, int(value)) for value in list(batch_sizes or [250])}
        )
        resolved_file_batch_sizes: list[int | None] = []
        for raw in list(file_batch_sizes or [None]):
            normalized = None if raw is None else max(1, int(raw))
            if normalized not in resolved_file_batch_sizes:
                resolved_file_batch_sizes.append(normalized)
        resolved_iterations = max(1, int(iterations or 1))

        results: list[dict[str, Any]] = []
        for batch_size in resolved_batch_sizes:
            for file_batch_size in resolved_file_batch_sizes:
                cold_trials: list[dict[str, Any]] = []
                warm_trials: list[dict[str, Any]] = []
                for iteration in range(1, resolved_iterations + 1):
                    cold_trials.append(
                        self._run_corpus_benchmark_trial(
                            path=root_path,
                            alias=alias_base,
                            ttl_hours=ttl_hours,
                            quarantine=quarantine,
                            trust_level=trust_level,
                            build_profile=build_profile,
                            batch_size=batch_size,
                            file_batch_size=file_batch_size,
                            rebuild=True,
                            iteration=iteration,
                        )
                    )
                    if reuse_check:
                        warm_trials.append(
                            self._run_corpus_benchmark_trial(
                                path=root_path,
                                alias=alias_base,
                                ttl_hours=ttl_hours,
                                quarantine=quarantine,
                                trust_level=trust_level,
                                build_profile=build_profile,
                                batch_size=batch_size,
                                file_batch_size=file_batch_size,
                                rebuild=False,
                                iteration=iteration,
                            )
                        )
                results.append(
                    {
                        "batch_size": batch_size,
                        "file_batch_size": file_batch_size,
                        "cold_runs": cold_trials,
                        "warm_runs": warm_trials,
                        "cold_summary": self._benchmark_trial_summary(cold_trials),
                        "warm_summary": self._benchmark_trial_summary(warm_trials),
                    }
                )

        return {
            "status": "ok",
            "benchmark_kind": "corpus_index",
            "path": root_path,
            "alias": alias_base,
            "iterations": resolved_iterations,
            "reuse_check": bool(reuse_check),
            "results": results,
            "summary": self._benchmark_result_summary(results),
        }

    def _persist_report_artifacts(self, report: dict[str, Any]) -> dict[str, str]:
        report_id = str(report.get("report_id") or f"comparative_{int(time.time())}")
        report_path = os.path.join(self.reports_dir, f"{report_id}.json")
        markdown_path = os.path.join(self.markdown_reports_dir, f"{report_id}.md")
        port_ledger_path = os.path.join(self.reports_dir, f"{report_id}.port_ledger.json")
        frontier_path = os.path.join(self.reports_dir, f"{report_id}.frontier_packets.json")
        creation_path = os.path.join(self.reports_dir, f"{report_id}.creation_ledger.json")
        synthesis_path = os.path.join(self.reports_dir, f"{report_id}.best_of_breed.json")
        programs_path = os.path.join(self.reports_dir, f"{report_id}.native_programs.json")
        phasepack_path = os.path.join(self.phasepacks_dir, f"{report_id}.phasepack.json")
        leaderboard_path = os.path.join(self.leaderboards_dir, f"{report_id}.leaderboard.json")
        proof_graphs_path = os.path.join(self.proof_graphs_dir, f"{report_id}.proof_graphs.json")
        artifacts = {
            "json_path": report_path,
            "markdown_path": markdown_path,
            "port_ledger_path": port_ledger_path,
            "frontier_packets_path": frontier_path,
            "creation_ledger_path": creation_path,
            "best_of_breed_path": synthesis_path,
            "native_programs_path": programs_path,
            "phasepack_path": phasepack_path,
            "leaderboard_path": leaderboard_path,
            "proof_graphs_path": proof_graphs_path,
        }
        report["artifacts"] = artifacts
        atomic_write_json(report_path, report, indent=2, sort_keys=True)
        atomic_write_json(
            port_ledger_path,
            {"port_ledger": list(report.get("port_ledger") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            frontier_path,
            {"frontier_packets": list(report.get("frontier_packets") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            creation_path,
            {"creation_ledger": list(report.get("creation_ledger") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            synthesis_path,
            dict(report.get("best_of_breed_synthesis") or {}),
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            programs_path,
            {"native_migration_programs": list(report.get("native_migration_programs") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            phasepack_path,
            {"phase_packets": list(report.get("phase_packets") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            leaderboard_path,
            {"portfolio_leaderboard": list(report.get("portfolio_leaderboard") or [])},
            indent=2,
            sort_keys=True,
        )
        atomic_write_json(
            proof_graphs_path,
            {
                "proof_graphs": [
                    relation.get("proof_graph")
                    for comparison in list(report.get("comparisons") or [])
                    for relation in list(comparison.get("primary_recommendations") or [])
                    + list(comparison.get("secondary_recommendations") or [])
                    if relation.get("proof_graph")
                ]
            },
            indent=2,
            sort_keys=True,
        )
        ReportGenerator(self.repo_path).save_comparative_markdown(report, markdown_path)
        return artifacts

    def _record_report_event(self, report: dict[str, Any]) -> None:
        report_id = str(report.get("report_id") or "comparative")
        comparisons = list(report.get("comparisons") or [])
        files = []
        symbols = []
        for comparison in comparisons:
            for relation in list(comparison.get("analogous_mechanisms") or []):
                source_path = str(relation.get("source_path") or "")
                target_path = str(relation.get("target_path") or "")
                if source_path:
                    files.append(source_path)
                if target_path:
                    files.append(target_path)
                source_symbol = str(relation.get("source_symbol_id") or "")
                target_symbol = str(relation.get("target_symbol_id") or "")
                if source_symbol:
                    symbols.append(source_symbol)
                if target_symbol:
                    symbols.append(target_symbol)
        RealityGraphStore(self.repo_path).record_event(
            "comparative_report",
            run_id=report_id,
            phase="comparative",
            status="ready",
            files=files,
            symbols=symbols,
            artifacts=dict(report.get("artifacts") or {}),
            metadata={
                "target_corpus_id": (report.get("target") or {}).get("corpus_id"),
                "candidate_count": int(report.get("candidate_count", 0) or 0),
                "port_candidate_count": len(report.get("port_ledger") or []),
                "frontier_packet_count": len(report.get("frontier_packets") or []),
                "phase_packet_count": len(report.get("phase_packets") or []),
                "comparison_backends": sorted(
                    {
                        str((comparison.get("summary") or {}).get("comparison_backend") or "")
                        for comparison in comparisons
                        if (comparison.get("summary") or {}).get("comparison_backend")
                    }
                ),
            },
            source="ComparativeAnalysisService.compare",
        )

    def _ensure_session(
        self,
        path: str,
        *,
        alias: str | None = None,
        ttl_hours: float,
    ) -> dict[str, Any]:
        return self.create_session(
            path=path,
            alias=alias,
            ttl_hours=ttl_hours,
            quarantine=self._infer_corpus_kind(self._resolve_input_path(path)) != "primary",
        )

    def _compare_packs(
        self,
        *,
        target_session: dict[str, Any],
        target_pack: dict[str, Any],
        target_twin: dict[str, Any],
        candidate_session: dict[str, Any],
        candidate_pack: dict[str, Any],
        top_k: int,
    ) -> dict[str, Any]:
        target_records = list(target_twin.get("records") or [])
        target_token_rows = list(target_twin.get("token_rows") or [])
        candidate_records = self._prioritize_records_for_comparison(
            [
                item for item in candidate_pack.get("files", []) if item.get("classification") == "source"
            ]
        )
        candidate_token_rows = self._token_rows_for_records(
            candidate_session,
            candidate_pack,
            candidate_records,
        )
        relations: list[dict[str, Any]] = []
        channel_rows: dict[str, list[tuple[int, int, float, list[str]]]] = {}
        ranked_pairs, native_backend, pair_screening = self._rank_record_pairs(
            target_records=target_records[: min(256, len(target_records))],
            candidate_records=candidate_records[: min(256, len(candidate_records))],
            target_token_rows=target_token_rows[: min(256, len(target_token_rows))],
            candidate_token_rows=candidate_token_rows[: min(256, len(candidate_token_rows))],
            top_k=max(4, min(16, top_k)),
        )
        channel_rows["token"] = list(ranked_pairs)
        pair_map: dict[tuple[int, int], dict[str, Any]] = {}
        for target_index, candidate_index, score, shared_tokens in ranked_pairs:
            pair_map[(target_index, candidate_index)] = {
                "base_score": float(score),
                "shared_evidence": list(shared_tokens),
                "evidence_kind": "token",
                "evidence_kinds": {"token"},
            }
        native_relation_pairs = self._disparate_synthesizer.synthesize_cross_scope(
            source_records=[
                self._relation_record_view(item)
                for item in candidate_records[: min(128, len(candidate_records))]
            ],
            target_records=[
                self._relation_record_view(item)
                for item in target_records[: min(128, len(target_records))]
            ],
            generation_id=f"{target_session['corpus_id']}::{candidate_session['corpus_id']}",
            top_k=max(8, min(64, top_k * 4)),
        )
        native_disparate_rows: list[tuple[int, int, float, list[str]]] = []
        for native_relation in native_relation_pairs:
            target_index = int(native_relation.get("target_index") or -1)
            candidate_index = int(native_relation.get("source_index") or -1)
            if target_index < 0 or candidate_index < 0:
                continue
            native_disparate_rows.append(
                (
                    target_index,
                    candidate_index,
                    float(native_relation.get("confidence") or 0.0),
                    list(native_relation.get("shared_feature_families") or []),
                )
            )
            key = (target_index, candidate_index)
            existing = pair_map.setdefault(
                key,
                {
                    "base_score": 0.0,
                    "shared_evidence": [],
                    "evidence_kind": "native_disparate",
                    "evidence_kinds": set(),
                },
            )
            native_score = float(native_relation.get("confidence") or 0.0)
            if native_score > float(existing.get("base_score") or 0.0):
                existing["base_score"] = native_score
                existing["shared_evidence"] = list(native_relation.get("shared_feature_families") or [])
                existing["evidence_kind"] = "native_disparate"
                existing["native_relation_payload"] = native_relation
            existing["evidence_kinds"].add("native_disparate")
        channel_rows["native_disparate"] = native_disparate_rows
        feature_rows = list(
            self._feature_pair_candidates(
                target_records=target_records[: min(96, len(target_records))],
                candidate_records=candidate_records[: min(96, len(candidate_records))],
                top_k=max(6, min(24, top_k * 2)),
            )
        )
        for (
            target_index,
            candidate_index,
            score,
            shared_tags,
        ) in feature_rows:
            key = (target_index, candidate_index)
            existing = pair_map.setdefault(
                key,
                {
                    "base_score": 0.0,
                    "shared_evidence": [],
                    "evidence_kind": "feature",
                    "evidence_kinds": set(),
                },
            )
            if float(score) > float(existing.get("base_score") or 0.0):
                existing["base_score"] = float(score)
                existing["shared_evidence"] = list(shared_tags)
                existing["evidence_kind"] = "feature"
            existing["evidence_kinds"].add("feature")
        channel_rows["feature"] = feature_rows
        disparate_rows = list(
            self._disparate_pair_candidates(
                target_records=target_records[: min(128, len(target_records))],
                candidate_records=candidate_records[: min(128, len(candidate_records))],
                top_k=max(8, min(48, top_k * 3)),
            )
        )
        for (
            target_index,
            candidate_index,
            score,
            shared_roles,
        ) in disparate_rows:
            key = (target_index, candidate_index)
            existing = pair_map.setdefault(
                key,
                {
                    "base_score": 0.0,
                    "shared_evidence": [],
                    "evidence_kind": "disparate",
                    "evidence_kinds": set(),
                },
            )
            if float(score) > float(existing.get("base_score") or 0.0):
                existing["base_score"] = float(score)
                existing["shared_evidence"] = list(shared_roles)
                existing["evidence_kind"] = "disparate"
            existing["evidence_kinds"].add("disparate")
        channel_rows["disparate"] = disparate_rows
        fused_pairs, fusion_telemetry = self._rank_fusion.fuse(
            channel_rows=channel_rows,
            top_k=max(top_k * 4, 24),
        )

        for fused_pair in fused_pairs:
            target_index = int(fused_pair.get("target_index") or -1)
            candidate_index = int(fused_pair.get("candidate_index") or -1)
            pair = dict(pair_map.get((target_index, candidate_index)) or {})
            score = max(
                float(pair.get("base_score") or 0.0),
                float(fused_pair.get("fused_score") or 0.0),
            )
            shared_evidence = list(pair.get("shared_evidence") or [])
            evidence_kind = str(pair.get("evidence_kind") or "token")
            evidence_kinds = sorted(pair.get("evidence_kinds") or [])
            if score < 0.18:
                continue
            target_record = target_records[target_index]
            candidate_record = candidate_records[candidate_index]
            candidate_feature_families = self._record_feature_families(candidate_record)
            target_feature_families = self._record_feature_families(target_record)
            initial_shared_role_tags = self._shared_role_tags(target_record, candidate_record)
            feature_families = (
                self._shared_feature_tags(target_record, candidate_record)
                or sorted((set(candidate_feature_families) & set(target_feature_families)))
                or candidate_feature_families[:3]
            )
            target_record = self._recommended_target_record(
                target_pack=target_pack,
                initial_target_record=target_record,
                feature_families=feature_families,
                shared_role_tags=initial_shared_role_tags,
            )
            shared_feature_tags = self._shared_feature_tags(target_record, candidate_record)
            shared_role_tags = self._shared_role_tags(target_record, candidate_record)
            if not shared_feature_tags and feature_families:
                shared_feature_tags = feature_families
            native_relation_payload = dict(pair.get("native_relation_payload") or {})
            relation_type = self._relation_type(
                score=score,
                target_record=target_record,
                candidate_record=candidate_record,
                shared_feature_tags=shared_feature_tags,
                shared_role_tags=shared_role_tags,
            )
            if native_relation_payload:
                relation_type = self._comparative_relation_family(
                    str(native_relation_payload.get("relation") or relation_type)
                )
            arbitration = self._arbitrate_relation(
                relation_type=relation_type,
                score=score,
                target_record=target_record,
                candidate_record=candidate_record,
                target_pack=target_pack,
                candidate_pack=candidate_pack,
                shared_feature_tags=shared_feature_tags,
                shared_role_tags=shared_role_tags,
            )
            noise_flags = sorted(
                set(self._noise_flags(target_record) + self._noise_flags(candidate_record))
            )
            actionability = self._actionability_assessment(
                score=score,
                relation_type=relation_type,
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
                noise_flags=noise_flags,
                target_record=target_record,
                candidate_record=candidate_record,
            )
            source_symbol = self._best_symbol(candidate_record)
            target_symbol = self._best_symbol(target_record)
            source_path = str(candidate_record.get("path") or "")
            target_path = str(target_record.get("path") or "")
            relation_score = round(
                max(
                    score,
                    float(actionability.get("actionability_score") or 0.0),
                    float(native_relation_payload.get("confidence") or 0.0),
                ),
                4,
            )
            subsystem_routing = self._subsystem_routing(
                target_path=target_path,
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
            )
            calibration = self._rank_fusion.calibrate_relation(
                relation_score=relation_score,
                actionability_score=float(actionability.get("actionability_score") or 0.0),
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
                evidence_channels=evidence_kinds,
                noise_flags=noise_flags,
                build_alignment=self._build_alignment(target_pack, candidate_pack),
                target_pack=target_pack,
                candidate_pack=candidate_pack,
                top_margin=float(fusion_telemetry.get("top1_top2_margin") or 0.0),
            )
            recommendation_tier = self._recommendation_tier(
                relation_type=relation_type,
                posture=str(arbitration.get("posture") or ""),
                relation_score=relation_score,
                actionability_score=float(actionability.get("actionability_score") or 0.0),
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
                noise_flags=noise_flags,
            )
            if calibration.get("no_port"):
                recommendation_tier = "low_signal"
            relation = {
                "relation_type": relation_type,
                "relation_score": relation_score,
                "actionability_score": float(actionability.get("actionability_score") or 0.0),
                "fused_rank_score": float(fused_pair.get("fused_score") or 0.0),
                "source_path": source_path,
                "target_path": target_path,
                "source_symbol_id": self._qualified_symbol_id(
                    candidate_session["corpus_id"],
                    source_path,
                    source_symbol.get("name", source_path),
                    source_symbol.get("kind", "file"),
                ),
                "target_symbol_id": self._qualified_symbol_id(
                    target_session["corpus_id"],
                    target_path,
                    target_symbol.get("name", target_path),
                    target_symbol.get("kind", "file"),
                ),
                "source_language": str(candidate_record.get("language") or "unknown"),
                "target_language": str(target_record.get("language") or "unknown"),
                "posture": arbitration["posture"],
                "rationale": arbitration["rationale"],
                "backend": native_backend,
                "feature_families": feature_families,
                "expected_value": list(actionability.get("expected_value") or []),
                "expected_value_summary": str(actionability.get("expected_value_summary") or ""),
                "why_port": str(actionability.get("why_port") or ""),
                "why_target_here": (
                    f"{str(actionability.get('why_target_here') or '')} "
                    f"This routes into {subsystem_routing['recommended_subsystem_label']} because "
                    f"{subsystem_routing['subsystem_rationale']}."
                ).strip(),
                "why_not_obvious": str(actionability.get("why_not_obvious") or ""),
                "recommendation_tier": recommendation_tier,
                "calibrated_confidence": float(
                    calibration.get("calibrated_confidence") or relation_score
                ),
                "confidence_label": str(calibration.get("confidence_label") or "low"),
                "abstain_reason": calibration.get("abstain_reason"),
                "negative_evidence": list(calibration.get("negative_evidence") or []),
                **subsystem_routing,
                "evidence": {
                    "evidence_kind": evidence_kind,
                    "evidence_kinds": evidence_kinds,
                    "shared_tokens": shared_evidence[:12] if evidence_kind == "token" else [],
                    "shared_feature_tags": shared_feature_tags,
                    "shared_role_tags": shared_role_tags,
                    "native_relation_family": str(native_relation_payload.get("relation") or ""),
                    "evidence_spans": list(native_relation_payload.get("evidence_spans") or []),
                    "evidence_mix": list(native_relation_payload.get("evidence_mix") or []),
                    "confidence_components": dict(
                        native_relation_payload.get("confidence_components") or {}
                    ),
                    "parser_uncertainty": str(
                        native_relation_payload.get("parser_uncertainty") or "unknown"
                    ),
                    "counterevidence": list(native_relation_payload.get("counterevidence") or []),
                    "target_tags": list(target_record.get("tags") or []),
                    "source_tags": list(candidate_record.get("tags") or []),
                    "symbol_kind_alignment": self._symbol_kind_alignment(
                        target_record,
                        candidate_record,
                    ),
                    "subsystem_rationale": subsystem_routing["subsystem_rationale"],
                    "build_alignment": bool(
                        set((target_pack.get("build_files") or []))
                        & set((candidate_pack.get("build_files") or []))
                    ),
                    "noise_flags": noise_flags,
                    "source_summary": (
                        f"{source_path} [{candidate_record.get('language')}] "
                        f"symbols={len(candidate_record.get('symbols') or [])}"
                    ),
                    "target_summary": (
                        f"{target_path} [{target_record.get('language')}] "
                        f"symbols={len(target_record.get('symbols') or [])}"
                    ),
                },
            }
            proof_graph = self._path_proof_builder.build(
                relation=relation,
                target_pack=target_pack,
                candidate_pack=candidate_pack,
            )
            relation["proof_graph"] = proof_graph
            relation["proof_graph_id"] = proof_graph.get("proof_graph_id")
            relation["landing_zone_confidence"] = float(
                proof_graph.get("landing_zone_confidence") or 0.0
            )
            relations.append(relation)
        deduped_relations: dict[tuple[str, str, str, tuple[str, ...]], dict[str, Any]] = {}
        for relation in relations:
            key = (
                str(relation.get("source_path") or ""),
                str(relation.get("target_path") or ""),
                str(relation.get("posture") or ""),
                tuple(sorted(relation.get("feature_families") or [])),
            )
            existing = deduped_relations.get(key)
            if existing is None or float(relation.get("relation_score") or 0.0) > float(
                existing.get("relation_score") or 0.0
            ):
                deduped_relations[key] = relation
        relations = list(deduped_relations.values())
        relations.sort(
            key=lambda item: (
                {"primary": 0, "secondary": 1, "low_signal": 2}.get(
                    str(item.get("recommendation_tier") or "low_signal"),
                    2,
                ),
                -float(item.get("actionability_score") or 0.0),
                -float(item.get("relation_score") or 0.0),
                item.get("source_path", ""),
                item.get("target_path", ""),
            )
        )
        top_relations = relations[:top_k]
        for index, relation in enumerate(top_relations, start=1):
            relation["recommendation_id"] = f"{candidate_session['corpus_id']}:recommendation:{index}"
        primary_recommendations = [
            relation for relation in top_relations if str(relation.get("recommendation_tier")) == "primary"
        ]
        secondary_recommendations = [
            relation for relation in top_relations if str(relation.get("recommendation_tier")) == "secondary"
        ]
        low_signal_relations = [
            relation for relation in top_relations if str(relation.get("recommendation_tier")) == "low_signal"
        ]
        recipe_relations = primary_recommendations + secondary_recommendations
        if not recipe_relations:
            recipe_relations = top_relations[: min(len(top_relations), max(3, top_k))]
        recipes = [
            self._recipe_from_relation(candidate_session, target_session, relation, rank=index + 1)
            for index, relation in enumerate(recipe_relations)
        ]
        subsystem_summary = self._subsystem_upgrade_summary(
            primary_recommendations + secondary_recommendations,
            creation_entries=[],
        )
        cluster_input = primary_recommendations + secondary_recommendations or top_relations[:8]
        upgrade_clusters = self._recommendation_clusters(cluster_input)
        program_groups, compression_telemetry = compress_program_groups(
            cluster_input,
            max_groups=max(8, min(16, top_k)),
        )
        recipe_map = {str(recipe.get("source_path") or ""): recipe for recipe in recipes}
        disparate_opportunities = [
            relation
            for relation in secondary_recommendations
            if relation.get("relation_type")
            in {
                "disparate_mechanism_candidate",
                "registry_analogue",
                "orchestration_analogue",
                "reporting_analogue",
                "evaluation_analogue",
                "port_program_candidate",
            }
            and (
                relation.get("feature_families")
                or float(relation.get("actionability_score") or 0.0) >= 0.58
            )
        ][: max(5, min(12, top_k))]
        frontier_packets = [
            self._frontier_packet(candidate_session, relation, rank=index + 1)
            for index, relation in enumerate(
                (primary_recommendations or secondary_recommendations or top_relations)[: max(3, min(5, top_k))]
            )
        ]
        creation_entries = self._creation_ledger(
            target_session=target_session,
            target_pack=target_pack,
            candidate_session=candidate_session,
            candidate_pack=candidate_pack,
            relations=top_relations,
        )
        subsystem_summary = self._subsystem_upgrade_summary(
            primary_recommendations + secondary_recommendations,
            creation_entries=creation_entries,
        )
        language_overlap = sorted(
            set((candidate_pack.get("languages") or {}).keys()) & set((target_pack.get("languages") or {}).keys())
        )
        overlay_graph = self._overlay_graph(
            target_session=target_session,
            candidate_session=candidate_session,
            relations=top_relations,
        )
        ledger_entries = []
        for index, relation in enumerate(top_relations, start=1):
            matching_recipe = dict(recipe_map.get(str(relation.get("source_path") or "")) or {})
            ledger_entries.append(
                {
                    "ledger_entry_id": f"{candidate_session['corpus_id']}:port:{index}",
                    "status": "candidate",
                    "lifecycle_state": "candidate",
                    "comparison_id": f"{target_session['corpus_id']}::{candidate_session['corpus_id']}",
                    "report_id": None,
                    "verification_refs": list(
                        matching_recipe.get("verification_requirements") or []
                    ),
                    "linked_recipe_id": matching_recipe.get("recipe_id"),
                    "provenance": {
                        "candidate_corpus_id": candidate_session["corpus_id"],
                        "target_corpus_id": target_session["corpus_id"],
                        "comparison_backend": native_backend,
                    },
                    "evidence_closure": {
                        "report_generated": True,
                        "verification_ready": bool(
                            matching_recipe.get("verification_requirements")
                        ),
                        "promoted": False,
                    },
                    "status_reason": relation.get("rationale"),
                    "target_insertion_path": relation.get("target_path"),
                    "source_path": relation.get("source_path"),
                    **relation,
                }
            )
        summary = {
            "feature_overlap": self._feature_overlap(top_relations),
            "candidate_feature_gaps": self._candidate_feature_gaps(
                target_pack,
                candidate_pack,
            ),
            "feature_matrix": self._comparison_feature_matrix(
                primary_recommendations + secondary_recommendations,
                target_pack=target_pack,
                candidate_pack=candidate_pack,
            ),
            "common_tech_stack": sorted(
                set(target_pack.get("tech_stack") or []) & set(candidate_pack.get("tech_stack") or [])
            ),
            "language_overlap": language_overlap,
            "relation_count": len(top_relations),
            "build_alignment": self._build_alignment(target_pack, candidate_pack),
            "capability_delta": self._capability_delta(target_pack, candidate_pack),
            "comparison_backend": native_backend,
            "pair_screening": pair_screening,
            "rank_fusion": fusion_telemetry,
            "preferred_implementation_language": self._preferred_implementation_language(
                target_pack
            ),
            "recommendation_counts": {
                "primary": len(primary_recommendations),
                "secondary": len(secondary_recommendations),
                "low_signal": len(low_signal_relations),
            },
            "compression": compression_telemetry,
        }
        return {
            "comparison_id": f"{target_session['corpus_id']}::{candidate_session['corpus_id']}",
            "target": self._session_summary(target_session, target_pack),
            "candidate": self._session_summary(candidate_session, candidate_pack),
            "summary": summary,
            "analogous_mechanisms": top_relations,
            "primary_recommendations": primary_recommendations,
            "secondary_recommendations": secondary_recommendations,
            "low_signal_relations": low_signal_relations,
            "disparate_opportunities": disparate_opportunities,
            "migration_recipes": recipes,
            "port_ledger": ledger_entries,
            "frontier_packets": frontier_packets,
            "creation_ledger": creation_entries,
            "overlay_graph": overlay_graph,
            "candidate_scorecard": self._candidate_scorecard(
                candidate_pack=candidate_pack,
                summary=summary,
                primary_recommendations=primary_recommendations,
                secondary_recommendations=secondary_recommendations,
                low_signal_relations=low_signal_relations,
            ),
            "upgrade_clusters": upgrade_clusters,
            "program_groups": program_groups,
            "subsystem_routing_summary": subsystem_summary,
            "value_realization": self._comparison_value_realization(
                primary_recommendations + secondary_recommendations
            ),
            "manual_validation_seed": self._manual_validation_seed(
                primary_recommendations + secondary_recommendations or top_relations[:8]
            ),
            "evidence_quality_summary": self._evidence_quality_summary(
                primary_recommendations=primary_recommendations,
                secondary_recommendations=secondary_recommendations,
                low_signal_relations=low_signal_relations,
                pair_screening=pair_screening,
            ),
            "report_text": self._comparison_summary_text(
                candidate_session=candidate_session,
                target_session=target_session,
                primary_recommendations=primary_recommendations,
                secondary_recommendations=secondary_recommendations,
                feature_matrix=summary["feature_matrix"],
            ),
        }

    def _comparison_summary_text(
        self,
        *,
        candidate_session: dict[str, Any],
        target_session: dict[str, Any],
        primary_recommendations: list[dict[str, Any]],
        secondary_recommendations: list[dict[str, Any]],
        feature_matrix: list[dict[str, Any]],
    ) -> str:
        if not primary_recommendations and not secondary_recommendations:
            return (
                f"No high-confidence mechanism-level matches were found between "
                f"{target_session['corpus_id']} and {candidate_session['corpus_id']}. "
                f"The next pass should focus on feature deltas, insertion points, and native rewrite targets."
            )
        lead = (primary_recommendations or secondary_recommendations)[0]
        feature_tags = ", ".join(lead.get("feature_families") or []) or "mechanism"
        follow_on = ", ".join(
            row.get("feature_family", "")
            for row in feature_matrix[:3]
            if row.get("feature_family")
        )
        subsystem_label = str(lead.get("recommended_subsystem_label") or "Anvil")
        return (
            f"{candidate_session['corpus_id']} best aligns with {target_session['corpus_id']} "
            f"through {feature_tags} on {lead['source_path']} -> {lead['target_path']} "
            f"with posture {lead['posture']}. Current best landing zone is in {subsystem_label}. "
            f"Leading migration themes: {follow_on or feature_tags}."
        )

    def _recipe_from_relation(
        self,
        candidate_session: dict[str, Any],
        target_session: dict[str, Any],
        relation: dict[str, Any],
        *,
        rank: int,
    ) -> dict[str, Any]:
        preferred_language = self._preferred_implementation_language(
            {"languages": {relation.get("target_language", "unknown"): 1}}
        )
        relation["preferred_implementation_language"] = preferred_language
        impact_assessment = self._integration_impact(
            target_language=str(relation.get("target_language") or "unknown"),
            candidate_language=str(relation.get("source_language") or "unknown"),
            posture=str(relation.get("posture") or "pattern_only_adoption"),
            target_path=str(relation.get("target_path") or ""),
        )
        feature_label = ", ".join(list(relation.get("feature_families") or [])[:3]) or "mechanism"
        target_pack = self._load_session_pack(str(target_session.get("corpus_id") or "")) or {}
        candidate_pack = self._load_session_pack(str(candidate_session.get("corpus_id") or "")) or {}
        program_ir = self._program_ir_builder.compile(
            relation=relation,
            target_pack=target_pack,
            candidate_pack=candidate_pack,
            impact_assessment=impact_assessment,
        )
        return {
            "recipe_id": f"{candidate_session['corpus_id']}:recipe:{rank}",
            "title": (
                f"{relation['posture'].replace('_', ' ').title()} "
                f"{feature_label} from {relation['source_path']} into {relation['target_path']}"
            ),
            "source_corpus_id": candidate_session["corpus_id"],
            "target_corpus_id": target_session["corpus_id"],
            "source_path": relation["source_path"],
            "target_insertion_path": relation["target_path"],
            "posture": relation["posture"],
            "relation_type": relation["relation_type"],
            "feature_families": list(relation.get("feature_families") or []),
            "recommended_subsystem": relation.get("recommended_subsystem"),
            "recommended_subsystem_label": relation.get("recommended_subsystem_label"),
            "preferred_implementation_language": preferred_language,
            "invariants": [
                "preserve public API contracts",
                "keep target build system authoritative",
                f"implement the substantive path in {preferred_language}, with Python limited to thin wrappers or orchestration",
                "verify behavior with focused tests before promotion",
            ],
            "verification_requirements": [
                step.get("command")
                for step in list(program_ir.get("verification_plan") or [])
            ],
            "evidence": relation["evidence"],
            "why_port": relation.get("why_port"),
            "why_target_here": relation.get("why_target_here"),
            "expected_value_summary": relation.get("expected_value_summary"),
            "impact_assessment": impact_assessment,
            "lowering_ir": program_ir,
        }

    def _frontier_packet(
        self,
        candidate_session: dict[str, Any],
        relation: dict[str, Any],
        *,
        rank: int,
    ) -> dict[str, Any]:
        impact_assessment = self._integration_impact(
            target_language=str(relation.get("target_language") or "unknown"),
            candidate_language=str(relation.get("source_language") or "unknown"),
            posture=str(relation.get("posture") or "pattern_only_adoption"),
            target_path=str(relation.get("target_path") or ""),
        )
        return {
            "packet_id": f"{candidate_session['corpus_id']}:frontier:{rank}",
            "corpus_id": candidate_session["corpus_id"],
            "title": (
                f"Port {', '.join(list(relation.get('feature_families') or [])[:2]) or relation['posture']} "
                f"from {relation['source_path']}"
            ),
            "priority": round(float(relation.get("relation_score") or 0.0), 3),
            "posture": relation["posture"],
            "source_path": relation["source_path"],
            "target_path": relation["target_path"],
            "recommended_subsystem": relation.get("recommended_subsystem"),
            "recommended_subsystem_label": relation.get("recommended_subsystem_label"),
            "recommended_tracks": [
                "comparative_spike",
                "verification_lane",
            ],
            "rationale": relation["rationale"],
            "why_port": relation.get("why_port"),
            "impact_assessment": impact_assessment,
            "phase_id": "eid.comparative_frontier_scheduler",
        }

    def _overlay_graph(
        self,
        *,
        target_session: dict[str, Any],
        candidate_session: dict[str, Any],
        relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        seen_nodes: set[str] = set()
        for relation in relations:
            for node_id, corpus_id, path_key in (
                (relation["target_symbol_id"], target_session["corpus_id"], relation["target_path"]),
                (relation["source_symbol_id"], candidate_session["corpus_id"], relation["source_path"]),
            ):
                if node_id in seen_nodes:
                    continue
                nodes.append(
                    {
                        "id": node_id,
                        "corpus_id": corpus_id,
                        "path": path_key,
                    }
                )
                seen_nodes.add(node_id)
            edges.append(
                {
                    "from": relation["source_symbol_id"],
                    "to": relation["target_symbol_id"],
                    "relation": relation["relation_type"],
                    "score": relation["relation_score"],
                }
            )
        return {"nodes": nodes, "edges": edges}

    def _build_alignment(
        self,
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
    ) -> dict[str, Any]:
        target_fp = dict(target_pack.get("build_fingerprint") or {})
        candidate_fp = dict(candidate_pack.get("build_fingerprint") or {})
        shared = sorted(set(target_fp.get("build_files", [])) & set(candidate_fp.get("build_files", [])))
        return {
            "shared_build_files": shared,
            "compatible": bool(shared) or target_fp.get("primary_build_system") == candidate_fp.get("primary_build_system"),
            "target": target_fp,
            "candidate": candidate_fp,
        }

    def _capability_delta(
        self,
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
    ) -> dict[str, Any]:
        target_cap = dict(target_pack.get("capability_matrix") or {})
        candidate_cap = dict(candidate_pack.get("capability_matrix") or {})
        target_deep = set(target_cap.get("deep_languages") or [])
        candidate_deep = set(candidate_cap.get("deep_languages") or [])
        return {
            "shared_deep_languages": sorted(target_deep & candidate_deep),
            "candidate_only_deep_languages": sorted(candidate_deep - target_deep),
            "target_only_deep_languages": sorted(target_deep - candidate_deep),
        }

    def _preferred_implementation_language(self, pack: dict[str, Any]) -> str:
        languages = dict(pack.get("languages") or {})
        native_total = sum(
            int(languages.get(language, 0) or 0)
            for language in ("cpp", "cpp_header", "c", "c_header", "rust", "go")
        )
        if native_total > 0:
            return "cpp"
        if int(languages.get("python", 0) or 0) > 0:
            return "python"
        return "cpp"

    def _feature_overlap(self, relations: list[dict[str, Any]]) -> list[str]:
        tags: set[str] = set()
        for relation in relations:
            tags.update(relation.get("feature_families") or [])
        return sorted(tags)

    def _candidate_feature_gaps(
        self,
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
    ) -> list[str]:
        target_features = self._pack_feature_families(target_pack)
        candidate_features = self._pack_feature_families(candidate_pack)
        return sorted(candidate_features - target_features)

    def _comparison_feature_matrix(
        self,
        relations: list[dict[str, Any]],
        *,
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
    ) -> list[dict[str, Any]]:
        matrix: list[dict[str, Any]] = []
        feature_gaps = set(self._candidate_feature_gaps(target_pack, candidate_pack))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for relation in relations:
            for feature_family in list(relation.get("feature_families") or []):
                grouped[feature_family].append(relation)
        for feature_family in sorted(set(grouped) | feature_gaps):
            relations_for_feature = grouped.get(feature_family, [])
            top_relation = relations_for_feature[0] if relations_for_feature else {}
            matrix.append(
                {
                    "feature_family": feature_family,
                    "status": "winner_candidate" if top_relation else "create_candidate",
                    "top_relation_score": float(top_relation.get("relation_score") or 0.0),
                    "actionability_score": float(top_relation.get("actionability_score") or 0.0),
                    "posture": str(
                        top_relation.get("posture") or ("native_rewrite" if feature_family in feature_gaps else "pattern_only_adoption")
                    ),
                    "source_path": str(top_relation.get("source_path") or ""),
                    "target_path": str(top_relation.get("target_path") or ""),
                }
            )
        return matrix

    def _comparison_value_realization(
        self,
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        counts: Counter[str] = Counter()
        for relation in recommendations:
            counts.update(list(relation.get("expected_value") or []))
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return {
            "top_value_categories": [name for name, _count in ranked[:5]],
            "value_category_counts": dict(counts),
        }

    def _evidence_quality_summary(
        self,
        *,
        primary_recommendations: list[dict[str, Any]],
        secondary_recommendations: list[dict[str, Any]],
        low_signal_relations: list[dict[str, Any]],
        pair_screening: dict[str, Any],
    ) -> dict[str, Any]:
        ranked = primary_recommendations + secondary_recommendations + low_signal_relations
        return {
            "primary_count": len(primary_recommendations),
            "secondary_count": len(secondary_recommendations),
            "low_signal_count": len(low_signal_relations),
            "pair_screen_backend": str(pair_screening.get("pair_screen_backend") or "unknown"),
            "pair_candidates_before_filter": int(pair_screening.get("pair_candidates_before_filter") or 0),
            "pair_candidates_after_filter": int(pair_screening.get("pair_candidates_after_filter") or 0),
            "no_port_count": len([item for item in ranked if item.get("abstain_reason")]),
            "average_confidence": round(
                sum(
                    float(item.get("calibrated_confidence") or item.get("relation_score") or 0.0)
                    for item in ranked
                )
                / max(1, len(ranked)),
                4,
            ),
        }

    def _candidate_scorecard(
        self,
        *,
        candidate_pack: dict[str, Any],
        summary: dict[str, Any],
        primary_recommendations: list[dict[str, Any]],
        secondary_recommendations: list[dict[str, Any]],
        low_signal_relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        families = Counter()
        for relation in primary_recommendations + secondary_recommendations:
            families.update(list(relation.get("feature_families") or []))
        ranked_families = [name for name, _count in families.most_common(6)]
        return {
            "overall_fit": "high"
            if len(primary_recommendations) >= 4
            else ("medium" if primary_recommendations or secondary_recommendations else "low"),
            "top_feature_families": ranked_families,
            "recommended_subsystems": [
                row.get("subsystem_label")
                for row in self._subsystem_upgrade_summary(
                    primary_recommendations + secondary_recommendations,
                    creation_entries=[],
                )[:3]
                if int(row.get("recommendation_count") or 0) > 0
            ],
            "primary_recommendation_count": len(primary_recommendations),
            "secondary_recommendation_count": len(secondary_recommendations),
            "low_signal_count": len(low_signal_relations),
            "build_alignment": dict(summary.get("build_alignment") or {}),
            "language_overlap": list(summary.get("language_overlap") or []),
            "candidate_feature_families": sorted(self._pack_feature_families(candidate_pack)),
        }

    def _manual_validation_seed(
        self,
        recommendations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        seed: list[dict[str, Any]] = []
        for relation in recommendations[:50]:
            seed.append(
                {
                    "recommendation_id": str(relation.get("recommendation_id") or ""),
                    "source_path": relation.get("source_path"),
                    "target_path": relation.get("target_path"),
                    "feature_families": list(relation.get("feature_families") or []),
                    "review_status": "pending",
                    "accepted": None,
                    "review_notes": "",
                    "missed_supporting_files": [],
                    "corrected_target_path": None,
                }
            )
        return seed

    def _creation_ledger(
        self,
        *,
        target_session: dict[str, Any],
        target_pack: dict[str, Any],
        candidate_session: dict[str, Any],
        candidate_pack: dict[str, Any],
        relations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        existing_features = set(self._feature_overlap(relations))
        target_languages = set((target_pack.get("languages") or {}).keys())
        candidate_languages = set((candidate_pack.get("languages") or {}).keys())
        for feature_family in self._candidate_feature_gaps(target_pack, candidate_pack):
            priority = round(
                0.65
                + (0.08 if feature_family not in existing_features else 0.0)
                + (0.05 if bool((candidate_pack.get("build_fingerprint") or {}).get("build_backed")) else 0.0),
                3,
            )
            impact_assessment = self._integration_impact(
                target_language=self._preferred_implementation_language(target_pack),
                candidate_language=self._preferred_implementation_language(candidate_pack),
                posture="native_rewrite",
                target_path=f"feature/{feature_family}",
            )
            entries.append(
                {
                    "entry_id": f"{candidate_session['corpus_id']}:create:{feature_family}",
                    "candidate_corpus_id": candidate_session["corpus_id"],
                    "target_corpus_id": target_session["corpus_id"],
                    "feature_family": feature_family,
                    "kind": "feature_creation",
                    "suggested_posture": "native_rewrite",
                    "priority": priority,
                    "confidence_score": priority,
                    "recommended_subsystem": self._preferred_subsystems(
                        feature_families=[feature_family],
                        shared_role_tags=[],
                    )[0],
                    "recommended_subsystem_label": self._subsystem_catalog()[
                        self._preferred_subsystems(
                            feature_families=[feature_family],
                            shared_role_tags=[],
                        )[0]
                    ]["label"],
                    "rationale": (
                        f"{candidate_session['corpus_id']} exposes {feature_family} capability that is absent from "
                        f"{target_session['corpus_id']} and should be created as a native primitive."
                    ),
                    "impact_assessment": impact_assessment,
                    "lowering_ir": self._recipe_lowering_ir(
                        source_path=f"feature::{feature_family}",
                        target_path=f"feature/{feature_family}",
                        posture="native_rewrite",
                        preferred_language=self._preferred_implementation_language(target_pack),
                        impact_assessment=impact_assessment,
                    ),
                }
            )
        for feature_family in sorted(
            self._pack_feature_families(candidate_pack)
            - self._pack_feature_families(target_pack)
            - existing_features
        ):
            if feature_family in {
                entry.get("feature_family")
                for entry in entries
            }:
                continue
            impact_assessment = self._integration_impact(
                target_language=self._preferred_implementation_language(target_pack),
                candidate_language=self._preferred_implementation_language(candidate_pack),
                posture="native_rewrite",
                target_path=f"feature/{feature_family}",
            )
            entries.append(
                {
                    "entry_id": f"{candidate_session['corpus_id']}:mechanism:{feature_family}",
                    "candidate_corpus_id": candidate_session["corpus_id"],
                    "target_corpus_id": target_session["corpus_id"],
                    "feature_family": feature_family,
                    "kind": "mechanism_creation",
                    "suggested_posture": "native_rewrite",
                    "priority": round(0.58 + float((candidate_pack.get("build_fingerprint") or {}).get("build_backed", False)) * 0.07, 3),
                    "confidence_score": 0.58,
                    "recommended_subsystem": self._preferred_subsystems(
                        feature_families=[feature_family],
                        shared_role_tags=[],
                    )[0],
                    "recommended_subsystem_label": self._subsystem_catalog()[
                        self._preferred_subsystems(
                            feature_families=[feature_family],
                            shared_role_tags=[],
                        )[0]
                    ]["label"],
                    "rationale": (
                        f"{candidate_session['corpus_id']} carries {feature_family} mechanisms that are not represented "
                        f"in {target_session['corpus_id']} and should be evaluated as a new native program."
                    ),
                    "impact_assessment": impact_assessment,
                    "lowering_ir": self._recipe_lowering_ir(
                        source_path=f"feature::{feature_family}",
                        target_path=f"feature/{feature_family}",
                        posture="native_rewrite",
                        preferred_language=self._preferred_implementation_language(target_pack),
                        impact_assessment=impact_assessment,
                    ),
                }
            )
        for language in sorted(candidate_languages - target_languages):
            candidate_row = (
                (candidate_pack.get("language_truth_matrix") or {}).get("per_language") or {}
            ).get(language, {})
            if not candidate_row or float(candidate_row.get("confidence_score") or 0.0) < 0.45:
                continue
            entries.append(
                {
                    "entry_id": f"{candidate_session['corpus_id']}:language:{language}",
                    "candidate_corpus_id": candidate_session["corpus_id"],
                    "target_corpus_id": target_session["corpus_id"],
                    "feature_family": language,
                    "kind": "language_enablement",
                    "suggested_posture": "native_rewrite",
                    "priority": round(0.55 + float(candidate_row.get("confidence_score") or 0.0) * 0.35, 3),
                    "confidence_score": float(candidate_row.get("confidence_score") or 0.0),
                    "recommended_subsystem": "saguaro",
                    "recommended_subsystem_label": self._subsystem_catalog()["saguaro"]["label"],
                    "rationale": (
                        f"{candidate_session['corpus_id']} shows credible {language} support beyond the target's current truth matrix."
                    ),
                }
            )
        return entries

    def _best_of_breed_synthesis(
        self,
        *,
        comparisons: list[dict[str, Any]],
        target_session: dict[str, Any],
        target_pack: dict[str, Any],
    ) -> dict[str, Any]:
        winners: dict[str, dict[str, Any]] = {}
        for comparison in comparisons:
            candidate = dict(comparison.get("candidate") or {})
            ranked_relations = list(comparison.get("primary_recommendations") or []) + list(
                comparison.get("secondary_recommendations") or []
            )
            for relation in ranked_relations:
                for feature_family in list(relation.get("feature_families") or []):
                    current = winners.get(feature_family)
                    candidate_winner = {
                        "feature_family": feature_family,
                        "winner_corpus_id": candidate.get("corpus_id"),
                        "score": float(
                            relation.get("actionability_score")
                            or relation.get("relation_score")
                            or 0.0
                        ),
                        "posture": relation.get("posture"),
                        "source_path": relation.get("source_path"),
                        "target_path": relation.get("target_path"),
                        "rationale": relation.get("rationale"),
                        "recommended_subsystem": relation.get("recommended_subsystem"),
                        "recommended_subsystem_label": relation.get("recommended_subsystem_label"),
                        "comparison_id": comparison.get("comparison_id"),
                        "recipe_id": next(
                            (
                                recipe.get("recipe_id")
                                for recipe in list(comparison.get("migration_recipes") or [])
                                if str(recipe.get("source_path") or "") == str(relation.get("source_path") or "")
                            ),
                            None,
                        ),
                    }
                    if current is None or float(candidate_winner["score"]) > float(current.get("score") or 0.0):
                        winners[feature_family] = candidate_winner
            for entry in list(comparison.get("creation_ledger") or []):
                feature_family = str(entry.get("feature_family") or "")
                if not feature_family or feature_family in winners:
                    continue
                winners[feature_family] = {
                    "feature_family": feature_family,
                    "winner_corpus_id": entry.get("candidate_corpus_id"),
                    "score": float(entry.get("priority") or 0.0),
                    "posture": entry.get("suggested_posture"),
                    "source_path": entry.get("feature_family"),
                    "target_path": f"feature/{feature_family}",
                    "rationale": entry.get("rationale"),
                    "recommended_subsystem": entry.get("recommended_subsystem"),
                    "recommended_subsystem_label": entry.get("recommended_subsystem_label"),
                    "comparison_id": comparison.get("comparison_id"),
                    "recipe_id": None,
                    "creation_entry_id": entry.get("entry_id"),
                }
        confidence_distribution: Counter[str] = Counter()
        for winner in winners.values():
            score = float(winner.get("score") or 0.0)
            label = "high" if score >= 0.8 else ("medium" if score >= 0.5 else "low")
            confidence_distribution.update([label])
        return {
            "schema_version": "best_of_breed_synthesis.v1",
            "target_corpus_id": target_session.get("corpus_id"),
            "preferred_implementation_language": self._preferred_implementation_language(target_pack),
            "feature_winners": sorted(
                winners.values(),
                key=lambda item: (-float(item.get("score") or 0.0), str(item.get("feature_family") or "")),
            ),
            "synthesis_rows": len(winners),
            "winner_confidence_distribution": dict(confidence_distribution),
        }

    def _native_migration_programs(
        self,
        *,
        target_session: dict[str, Any],
        target_pack: dict[str, Any],
        best_of_breed: dict[str, Any],
        creation_ledger: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        creation_by_feature = {
            str(entry.get("feature_family") or ""): entry
            for entry in creation_ledger
        }
        programs: list[dict[str, Any]] = []
        for winner in list(best_of_breed.get("feature_winners") or []):
            feature_family = str(winner.get("feature_family") or "")
            creation_entry = creation_by_feature.get(feature_family, {})
            impact_assessment = dict(
                creation_entry.get("impact_assessment")
                or self._integration_impact(
                    target_language=self._preferred_implementation_language(target_pack),
                    candidate_language=self._preferred_implementation_language(target_pack),
                    posture=str(winner.get("posture") or "native_rewrite"),
                    target_path=str(winner.get("target_path") or ""),
                )
            )
            programs.append(
                {
                    "program_id": f"{target_session['corpus_id']}:comparative_program:{feature_family}",
                    "feature_family": feature_family,
                    "winner_corpus_id": winner.get("winner_corpus_id"),
                    "posture": winner.get("posture") or "native_rewrite",
                    "priority": round(float(winner.get("score") or 0.0), 3),
                    "target_path": winner.get("target_path"),
                    "source_path": winner.get("source_path"),
                    "recommended_subsystem": winner.get("recommended_subsystem"),
                    "recommended_subsystem_label": winner.get("recommended_subsystem_label"),
                    "impact_assessment": impact_assessment,
                    "recipe_ir": creation_entry.get("lowering_ir")
                    or self._recipe_lowering_ir(
                        source_path=str(winner.get("source_path") or ""),
                        target_path=str(winner.get("target_path") or ""),
                        posture=str(winner.get("posture") or "native_rewrite"),
                        preferred_language=self._preferred_implementation_language(target_pack),
                        impact_assessment=impact_assessment,
                    ),
                    "promotion_gates": [
                        "comparative evidence reviewed",
                        "saguaro verify . --engines native,ruff,semantic",
                        "targeted tests pass",
                        "impact risk accepted",
                    ],
                }
            )
        return programs

    @staticmethod
    def _recipe_lowering_ir(
        *,
        source_path: str,
        target_path: str,
        posture: str,
        preferred_language: str,
        impact_assessment: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ir_version": "migration_program_ir.v2",
            "implementation_tier": preferred_language,
            "source_scope": source_path,
            "target_scope": target_path,
            "posture": posture,
            "wrapper_policy": "thin_python_wrapper_only" if preferred_language == "cpp" else "native_primary",
            "dependencies": [target_path],
            "verification_plan": [
                {
                    "kind": "governance",
                    "command": "./venv/bin/saguaro verify . --engines native,ruff,semantic --format json",
                },
                {"kind": "targeted_test", "command": "pytest tests"},
            ],
            "rollback_criteria": [
                "targeted tests fail",
                "build graph consistency regresses",
            ],
            "impact_assessment": impact_assessment,
        }

    @staticmethod
    def _integration_impact(
        *,
        target_language: str,
        candidate_language: str,
        posture: str,
        target_path: str,
    ) -> dict[str, Any]:
        score = 0.2
        if target_language != candidate_language:
            score += 0.2
        if posture == "native_rewrite":
            score += 0.25
        if "test" in target_path:
            score += 0.1
        if any(part in target_path for part in ("core", "runtime", "native")):
            score += 0.15
        score = round(min(score, 0.95), 3)
        return {
            "impact_score": score,
            "risk_level": "high" if score >= 0.7 else ("medium" if score >= 0.4 else "low"),
            "likely_fallout": [
                "build graph",
                "targeted tests",
                "api contracts" if posture == "native_rewrite" else "integration shims",
            ],
        }

    def _feature_tag_universe(self) -> set[str]:
        return {
            "alerting",
            "analysis",
            "artifact",
            "attack",
            "dataflow",
            "diagnostics",
            "extractor",
            "evaluation",
            "framework",
            "model",
            "optimizer",
            "orchestration",
            "pipeline",
            "query_engine",
            "registry",
            "reporting",
            "security",
            "taint",
            "target",
            "vulnerability",
        }

    @staticmethod
    def _role_tag_universe() -> set[str]:
        return {
            "adapter",
            "artifact",
            "cli",
            "config_surface",
            "core_runtime",
            "entrypoint",
            "example_surface",
            "formatter",
            "grammar",
            "module_init",
            "orchestration",
            "plugin",
            "registry",
            "report_surface",
            "secondary_surface",
            "service",
            "session",
            "state",
            "test_harness",
        }

    @staticmethod
    def _feature_family_aliases() -> dict[str, str]:
        return {
            "alerting": "alerting",
            "analysis": "security_analysis",
            "artifact": "artifact_output",
            "attack": "attack_orchestration",
            "dataflow": "dataflow",
            "diagnostics": "diagnostics",
            "evaluation": "evaluation_pipeline",
            "extractor": "extractor",
            "framework": "framework_adapter",
            "model": "evaluation_pipeline",
            "optimizer": "optimization",
            "orchestration": "attack_orchestration",
            "pipeline": "evaluation_pipeline",
            "query_engine": "query_engine",
            "registry": "target_registry",
            "reporting": "reporting",
            "security": "security_analysis",
            "taint": "dataflow",
            "target": "target_registry",
            "vulnerability": "security_analysis",
        }

    def _record_feature_families(self, record: dict[str, Any]) -> list[str]:
        aliases = self._feature_family_aliases()
        families = {
            aliases[tag]
            for tag in list(record.get("tags") or [])
            if tag in aliases
        }
        return sorted(families)

    def _pack_feature_families(self, pack: dict[str, Any]) -> set[str]:
        families: set[str] = set()
        for record in list(pack.get("files") or []):
            if str(record.get("classification") or "") != "source":
                continue
            families.update(self._record_feature_families(record))
        return families

    @staticmethod
    def _subsystem_catalog() -> dict[str, dict[str, Any]]:
        return {
            "saguaro": {
                "label": "Saguaro",
                "layer": "semantic",
                "path_prefixes": [
                    "Saguaro/saguaro/",
                    "Saguaro/src/",
                    "Saguaro/include/",
                ],
                "rationale": "semantic operating layer for code understanding, indexing, query, analysis, and verification",
            },
            "qsg": {
                "label": "QSG",
                "layer": "native_runtime",
                "path_prefixes": [
                    "core/qsg/",
                    "core/native/",
                    "core/model/",
                ],
                "rationale": "native inference/runtime subsystem for the QSG pipeline, model contracts, and low-level acceleration",
            },
            "anvil": {
                "label": "Anvil",
                "layer": "control_plane",
                "path_prefixes": [
                    "core/",
                    "cli/",
                    "config/",
                    "domains/",
                ],
                "rationale": "control-plane, operator, and orchestration surfaces outside the semantic and native runtime layers",
            },
        }

    def _feature_family_subsystem_preferences(self, feature_family: str) -> list[str]:
        return {
            "artifact_output": ["anvil", "saguaro"],
            "attack_orchestration": ["anvil"],
            "target_registry": ["anvil"],
            "framework_adapter": ["anvil", "qsg"],
            "optimization": ["qsg", "anvil"],
            "evaluation_pipeline": ["qsg", "anvil"],
            "security_analysis": ["saguaro", "anvil"],
            "query_engine": ["saguaro"],
            "dataflow": ["saguaro"],
            "diagnostics": ["saguaro", "anvil"],
            "extractor": ["saguaro"],
            "reporting": ["saguaro", "anvil"],
            "alerting": ["anvil", "saguaro"],
        }.get(feature_family, ["anvil", "saguaro", "qsg"])

    def _subsystem_for_path(self, path: str) -> str:
        normalized = str(path or "").lstrip("./")
        for subsystem_id, entry in self._subsystem_catalog().items():
            for prefix in list(entry.get("path_prefixes") or []):
                if normalized.startswith(prefix):
                    return subsystem_id
        return "anvil"

    def _preferred_subsystems(
        self,
        *,
        feature_families: list[str],
        shared_role_tags: list[str],
        target_path: str = "",
    ) -> list[str]:
        preferred: list[str] = []
        path_subsystem = self._subsystem_for_path(target_path) if target_path else ""
        if path_subsystem:
            preferred.append(path_subsystem)
        for feature_family in feature_families:
            preferred.extend(self._feature_family_subsystem_preferences(feature_family))
        role_tags = set(shared_role_tags)
        if {"orchestration", "cli", "entrypoint", "registry"} & role_tags:
            preferred.append("anvil")
        if {"artifact", "formatter", "report_surface"} & role_tags:
            preferred.extend(["saguaro", "anvil"])
        if {"service", "plugin", "session", "state"} & role_tags:
            preferred.extend(["saguaro", "anvil"])
        if {"core_runtime"} & role_tags:
            preferred.extend(["qsg", "anvil"])
        ordered: list[str] = []
        for subsystem_id in preferred:
            if subsystem_id and subsystem_id not in ordered:
                ordered.append(subsystem_id)
        return ordered or ["anvil", "saguaro", "qsg"]

    def _subsystem_routing(
        self,
        *,
        target_path: str,
        feature_families: list[str],
        shared_role_tags: list[str],
    ) -> dict[str, Any]:
        catalog = self._subsystem_catalog()
        current_subsystem = self._subsystem_for_path(target_path)
        preferred_subsystems = self._preferred_subsystems(
            feature_families=feature_families,
            shared_role_tags=shared_role_tags,
            target_path=target_path,
        )
        recommended_subsystem = preferred_subsystems[0] if preferred_subsystems else current_subsystem
        confidence = 0.96 if current_subsystem == recommended_subsystem else 0.72
        if not feature_families and len(shared_role_tags) < 2:
            confidence = min(confidence, 0.61)
        rationale_bits = []
        if feature_families:
            rationale_bits.append(
                f"feature families {', '.join(feature_families[:3])} favor {catalog[recommended_subsystem]['label']}"
            )
        if shared_role_tags:
            rationale_bits.append(
                f"shared roles {', '.join(shared_role_tags[:3])} align with {catalog[recommended_subsystem]['layer']}"
            )
        rationale_bits.append(
            f"target path {target_path} sits in {catalog[current_subsystem]['label']}"
        )
        return {
            "current_target_subsystem": current_subsystem,
            "current_target_subsystem_label": catalog[current_subsystem]["label"],
            "recommended_subsystem": recommended_subsystem,
            "recommended_subsystem_label": catalog[recommended_subsystem]["label"],
            "preferred_subsystems": preferred_subsystems,
            "subsystem_confidence": round(confidence, 3),
            "subsystem_rationale": "; ".join(rationale_bits),
        }

    def _recommendation_clusters(
        self,
        recommendations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, tuple[str, ...], str], dict[str, Any]] = {}
        for relation in recommendations:
            feature_families = tuple(
                sorted(list(relation.get("feature_families") or ["mechanism"]))
            )
            subsystem_id = str(relation.get("recommended_subsystem") or "anvil")
            key = (
                subsystem_id,
                str(relation.get("target_path") or ""),
                feature_families,
                str(relation.get("posture") or ""),
            )
            entry = grouped.setdefault(
                key,
                {
                    "recommended_subsystem": subsystem_id,
                    "recommended_subsystem_label": relation.get("recommended_subsystem_label")
                    or self._subsystem_catalog()[subsystem_id]["label"],
                    "target_path": str(relation.get("target_path") or ""),
                    "feature_families": list(feature_families),
                    "posture": str(relation.get("posture") or ""),
                    "relation_types": Counter(),
                    "source_paths": [],
                    "source_count": 0,
                    "top_relation_score": 0.0,
                    "top_actionability_score": 0.0,
                    "expected_value_counts": Counter(),
                    "rationales": [],
                },
            )
            source_path = str(relation.get("source_path") or "")
            if source_path and source_path not in entry["source_paths"]:
                entry["source_paths"].append(source_path)
            entry["source_count"] = len(entry["source_paths"])
            entry["relation_types"].update([str(relation.get("relation_type") or "unknown")])
            entry["expected_value_counts"].update(list(relation.get("expected_value") or []))
            entry["top_relation_score"] = max(
                float(entry.get("top_relation_score") or 0.0),
                float(relation.get("relation_score") or 0.0),
            )
            entry["top_actionability_score"] = max(
                float(entry.get("top_actionability_score") or 0.0),
                float(relation.get("actionability_score") or 0.0),
            )
            rationale = str(relation.get("why_port") or relation.get("rationale") or "")
            if rationale and rationale not in entry["rationales"]:
                entry["rationales"].append(rationale)
        clusters: list[dict[str, Any]] = []
        for cluster in grouped.values():
            clusters.append(
                {
                    "recommended_subsystem": cluster["recommended_subsystem"],
                    "recommended_subsystem_label": cluster["recommended_subsystem_label"],
                    "target_path": cluster["target_path"],
                    "feature_families": list(cluster["feature_families"]),
                    "posture": cluster["posture"],
                    "source_paths": list(cluster["source_paths"])[:8],
                    "source_count": int(cluster["source_count"]),
                    "top_relation_score": round(float(cluster["top_relation_score"] or 0.0), 3),
                    "top_actionability_score": round(
                        float(cluster["top_actionability_score"] or 0.0),
                        3,
                    ),
                    "dominant_relation_type": next(
                        iter(
                            name
                            for name, _count in cluster["relation_types"].most_common(1)
                        ),
                        "unknown",
                    ),
                    "expected_value": [
                        name
                        for name, _count in cluster["expected_value_counts"].most_common(4)
                    ],
                    "program_summary": next(iter(cluster["rationales"]), ""),
                }
            )
        clusters.sort(
            key=lambda item: (
                -float(item.get("top_actionability_score") or 0.0),
                -float(item.get("top_relation_score") or 0.0),
                -int(item.get("source_count") or 0),
                str(item.get("target_path") or ""),
            )
        )
        return clusters

    def _subsystem_upgrade_summary(
        self,
        recommendations: list[dict[str, Any]],
        *,
        creation_entries: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        catalog = self._subsystem_catalog()
        grouped: dict[str, dict[str, Any]] = {
            subsystem_id: {
                "subsystem_id": subsystem_id,
                "subsystem_label": entry["label"],
                "primary_count": 0,
                "secondary_count": 0,
                "recommendation_count": 0,
                "creation_candidate_count": 0,
                "feature_families": Counter(),
                "target_paths": Counter(),
            }
            for subsystem_id, entry in catalog.items()
        }
        for relation in recommendations:
            subsystem_id = str(relation.get("recommended_subsystem") or "anvil")
            row = grouped.setdefault(
                subsystem_id,
                {
                    "subsystem_id": subsystem_id,
                    "subsystem_label": subsystem_id.title(),
                    "primary_count": 0,
                    "secondary_count": 0,
                    "recommendation_count": 0,
                    "creation_candidate_count": 0,
                    "feature_families": Counter(),
                    "target_paths": Counter(),
                },
            )
            row["recommendation_count"] += 1
            tier = str(relation.get("recommendation_tier") or "")
            if tier == "primary":
                row["primary_count"] += 1
            elif tier == "secondary":
                row["secondary_count"] += 1
            row["feature_families"].update(list(relation.get("feature_families") or []))
            row["target_paths"].update([str(relation.get("target_path") or "")])
        for entry in list(creation_entries or []):
            subsystem_id = str(entry.get("recommended_subsystem") or "anvil")
            row = grouped.setdefault(
                subsystem_id,
                {
                    "subsystem_id": subsystem_id,
                    "subsystem_label": subsystem_id.title(),
                    "primary_count": 0,
                    "secondary_count": 0,
                    "recommendation_count": 0,
                    "creation_candidate_count": 0,
                    "feature_families": Counter(),
                    "target_paths": Counter(),
                },
            )
            row["creation_candidate_count"] += 1
            if entry.get("feature_family"):
                row["feature_families"].update([str(entry.get("feature_family"))])
        rows: list[dict[str, Any]] = []
        for subsystem_id, row in grouped.items():
            rows.append(
                {
                    "subsystem_id": subsystem_id,
                    "subsystem_label": row["subsystem_label"],
                    "primary_count": int(row["primary_count"]),
                    "secondary_count": int(row["secondary_count"]),
                    "recommendation_count": int(row["recommendation_count"]),
                    "creation_candidate_count": int(row["creation_candidate_count"]),
                    "top_feature_families": [
                        name for name, _count in row["feature_families"].most_common(5)
                    ],
                    "top_target_paths": [
                        name
                        for name, _count in row["target_paths"].most_common(4)
                        if name
                    ],
                }
            )
        rows.sort(
            key=lambda item: (
                -int(item.get("primary_count") or 0),
                -int(item.get("secondary_count") or 0),
                -int(item.get("creation_candidate_count") or 0),
                str(item.get("subsystem_id") or ""),
            )
        )
        return rows

    def _record_role_tags(self, record: dict[str, Any]) -> list[str]:
        return sorted(
            {
                tag
                for tag in list(record.get("tags") or [])
                if tag in self._role_tag_universe()
            }
        )

    def _shared_role_tags(
        self,
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
    ) -> list[str]:
        return sorted(
            set(self._record_role_tags(target_record))
            & set(self._record_role_tags(candidate_record))
        )

    @staticmethod
    def _symbol_kind_alignment(
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
    ) -> float:
        left = set(target_record.get("symbol_kinds") or [])
        right = set(candidate_record.get("symbol_kinds") or [])
        if not left or not right:
            return 0.0
        return round(len(left & right) / max(1, len(left | right)), 3)

    def _noise_flags(self, record: dict[str, Any]) -> list[str]:
        path = str(record.get("path") or "")
        tags = set(record.get("tags") or [])
        flags: list[str] = []
        if "module_init" in tags:
            flags.append("module_init")
        if "grammar" in tags:
            flags.append("grammar")
        if "example_surface" in tags:
            flags.append("example_surface")
        if "test_harness" in tags:
            flags.append("test_harness")
        if "config_surface" in tags and str(record.get("language") or "") in {
            "json",
            "yaml",
            "toml",
            "ini",
        }:
            flags.append("config_surface")
        if "doc" in tags or path.endswith(".md"):
            flags.append("doc")
        return flags

    def _recommendation_tier(
        self,
        *,
        relation_type: str,
        posture: str,
        relation_score: float,
        actionability_score: float,
        feature_families: list[str],
        shared_role_tags: list[str],
        noise_flags: list[str],
    ) -> str:
        evidence_ready = bool(feature_families) or len(shared_role_tags) >= 2
        if (
            evidence_ready
            and (actionability_score >= 0.58 or relation_score >= 0.72)
            and posture != "pattern_only_adoption"
            and not noise_flags
        ):
            return "primary"
        if (
            evidence_ready
            and (actionability_score >= 0.38 or relation_score >= 0.45)
            and (relation_type != "analogous_mechanism" or relation_score >= 0.62)
        ):
            return "secondary"
        return "low_signal"

    def _expected_value_categories(
        self,
        *,
        feature_families: list[str],
        shared_role_tags: list[str],
        candidate_record: dict[str, Any],
        target_record: dict[str, Any],
    ) -> list[str]:
        categories: list[str] = []
        family_set = set(feature_families)
        if family_set & {"security_analysis", "dataflow", "target_registry", "framework_adapter"}:
            categories.append("capability_gain")
        if family_set & {"reporting", "artifact_output"}:
            categories.append("reporting_gain")
        if family_set & {"attack_orchestration", "optimization"} or set(shared_role_tags) & {
            "cli",
            "orchestration",
            "session",
            "state",
        }:
            categories.append("operator_experience_gain")
        if family_set & {"evaluation_pipeline", "framework_adapter", "target_registry"}:
            categories.append("coverage_gain")
        candidate_tags = set(candidate_record.get("tags") or [])
        target_tags = set(target_record.get("tags") or [])
        if "native" in candidate_tags and "native" not in target_tags:
            categories.append("performance_gain")
        return categories or ["capability_gain"]

    def _actionability_assessment(
        self,
        *,
        score: float,
        relation_type: str,
        feature_families: list[str],
        shared_role_tags: list[str],
        noise_flags: list[str],
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
    ) -> dict[str, Any]:
        target_path = str(target_record.get("path") or "")
        candidate_path = str(candidate_record.get("path") or "")
        symbol_alignment = self._symbol_kind_alignment(target_record, candidate_record)
        expected_value = self._expected_value_categories(
            feature_families=feature_families,
            shared_role_tags=shared_role_tags,
            candidate_record=candidate_record,
            target_record=target_record,
        )
        score_value = min(
            0.99,
            max(
                0.0,
                score
                + (0.05 * len(feature_families))
                + (0.04 * len(shared_role_tags))
                + (0.06 * symbol_alignment)
                - (0.08 * len(noise_flags)),
            ),
        )
        if not feature_families:
            score_value = max(0.0, score_value - 0.1)
        if relation_type in {
            "disparate_mechanism_candidate",
            "port_program_candidate",
            "orchestration_analogue",
            "reporting_analogue",
            "registry_analogue",
            "evaluation_analogue",
        } and len(shared_role_tags) < 2:
            score_value = max(0.0, score_value - 0.06)
        actionable = bool(feature_families or shared_role_tags or score >= 0.48) and len(noise_flags) < 2
        if relation_type in {"analogous_mechanism", "portable_pattern"} and not feature_families:
            actionable = actionable and score_value >= 0.46
        why_port = (
            f"Porting {candidate_path} would strengthen {', '.join(feature_families or ['mechanism'])} "
            f"and deliver {', '.join(expected_value)}."
        )
        why_target_here = (
            f"{target_path} is the nearest target surface because it shares "
            f"{', '.join(shared_role_tags or feature_families or ['operational role'])}."
        )
        return {
            "actionable": actionable,
            "actionability_score": round(score_value, 3),
            "expected_value": expected_value,
            "expected_value_summary": ", ".join(expected_value),
            "why_port": why_port,
            "why_target_here": why_target_here,
            "why_not_obvious": (
                "Role alignment is stronger than lexical overlap."
                if relation_type
                in {
                    "disparate_mechanism_candidate",
                    "registry_analogue",
                    "orchestration_analogue",
                    "reporting_analogue",
                    "evaluation_analogue",
                }
                else ""
            ),
        }

    @staticmethod
    def _feature_target_hints(feature_family: str) -> list[str]:
        return {
            "artifact_output": [
                "core/artifacts/manager.py",
                "core/artifacts/renderers.py",
                "Saguaro/saguaro/cpu/report.py",
                "core/campaign/artifact_registry.py",
            ],
            "reporting": [
                "Saguaro/saguaro/analysis/report.py",
                "Saguaro/saguaro/analysis/trace_output.py",
                "core/artifacts/renderers.py",
                "core/artifacts/manager.py",
                "core/campaign/artifact_registry.py",
            ],
            "target_registry": [
                "core/campaign/repo_registry.py",
                "core/hooks/registry.py",
                "cli/command_registry.py",
            ],
            "framework_adapter": [
                "core/campaign/tooling_factory.py",
                "core/qsg/ollama_adapter.py",
                "core/campaign/repo_registry.py",
                "core/hooks/registry.py",
            ],
            "attack_orchestration": [
                "core/campaign/control_plane.py",
                "core/campaign/runner.py",
                "cli/commands/thinking.py",
                "cli/commands/features.py",
            ],
            "optimization": [
                "core/qsg/continuous_engine.py",
                "core/native/native_qsg_engine.py",
                "core/qsg/phase_controller.py",
                "core/campaign/control_plane.py",
                "core/campaign/completion_engine.py",
                "core/campaign/risk_radar.py",
            ],
            "evaluation_pipeline": [
                "core/qsg/continuous_engine.py",
                "core/qsg/generator.py",
                "core/model/model_contract.py",
                "core/campaign/models.py",
                "core/model/model_contract.py",
                "core/campaign/tooling_factory.py",
            ],
            "security_analysis": [
                "Saguaro/saguaro/sentinel/verifier.py",
                "Saguaro/saguaro/analysis/report.py",
                "Saguaro/saguaro/services/platform.py",
                "core/campaign/control_plane.py",
                "core/campaign/risk_radar.py",
                "core/aes/runtime_gates/domain_report_gate.py",
            ],
            "dataflow": [
                "Saguaro/saguaro/analysis/dfg_builder.py",
                "Saguaro/saguaro/analysis/code_graph.py",
                "Saguaro/saguaro/analysis/impact.py",
                "core/campaign/coverage_engine.py",
                "core/campaign/control_plane.py",
            ],
            "query_engine": [
                "Saguaro/saguaro/query/pipeline.py",
                "Saguaro/saguaro/query/gateway.py",
                "Saguaro/saguaro/services/platform.py",
            ],
            "diagnostics": [
                "Saguaro/saguaro/analysis/health_card.py",
                "Saguaro/saguaro/analysis/report.py",
                "core/telemetry/black_box.py",
            ],
            "extractor": [
                "Saguaro/saguaro/parsing/parser.py",
                "Saguaro/saguaro/indexing/engine.py",
                "Saguaro/saguaro/indexing/native_worker.py",
            ],
        }.get(feature_family, [])

    def _target_record_match_score(
        self,
        record: dict[str, Any],
        *,
        feature_families: list[str],
        shared_role_tags: list[str],
    ) -> float:
        path = str(record.get("path") or "")
        record_roles = set(self._record_role_tags(record))
        record_families = set(self._record_feature_families(record))
        score = 0.0
        score += 0.22 * len(record_families & set(feature_families))
        score += 0.18 * len(record_roles & set(shared_role_tags))
        score -= 0.16 * len(self._noise_flags(record))
        for feature_family in feature_families:
            for index, hint in enumerate(self._feature_target_hints(feature_family)):
                if path.endswith(hint):
                    score += max(0.72 - (index * 0.12), 0.28)
                elif hint.rsplit("/", 1)[-1] in path:
                    score += max(0.3 - (index * 0.04), 0.12)
        record_subsystem = self._subsystem_for_path(path)
        for index, subsystem_id in enumerate(
            self._preferred_subsystems(
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
                target_path="",
            )[:3]
        ):
            if record_subsystem == subsystem_id:
                score += max(0.42 - (index * 0.14), 0.12)
        if "artifact_registry" in path and "artifact_output" not in feature_families and "reporting" not in feature_families:
            score -= 0.14
        if "cli/commands/features.py" in path and "attack_orchestration" not in feature_families:
            score -= 0.12
        if not feature_families and not shared_role_tags and record_subsystem in {"saguaro", "qsg"}:
            score -= 0.22
        return score

    def _recommended_target_record(
        self,
        *,
        target_pack: dict[str, Any],
        initial_target_record: dict[str, Any],
        feature_families: list[str],
        shared_role_tags: list[str],
    ) -> dict[str, Any]:
        best_record = initial_target_record
        best_score = self._target_record_match_score(
            initial_target_record,
            feature_families=feature_families,
            shared_role_tags=shared_role_tags,
        )
        for record in list(target_pack.get("files") or []):
            if str(record.get("classification") or "") != "source":
                continue
            score = self._target_record_match_score(
                record,
                feature_families=feature_families,
                shared_role_tags=shared_role_tags,
            )
            if score > best_score:
                best_score = score
                best_record = record
        return best_record

    def _shared_feature_tags(
        self,
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
    ) -> list[str]:
        return sorted(
            set(self._record_feature_families(target_record))
            & set(self._record_feature_families(candidate_record))
        )

    def _relation_record_view(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(record.get("path") or ""),
            "line": int(
                ((record.get("symbols") or [{}])[0] or {}).get("line")
                or record.get("line")
                or 1
            ),
            "language": str(record.get("language") or "unknown"),
            "tags": list(record.get("tags") or []),
            "role_tags": self._record_role_tags(record),
            "feature_families": self._record_feature_families(record),
            "boundary_markers": sorted(
                {
                    str(tag)
                    for tag in list(record.get("tags") or [])
                    if str(tag) in {"ffi", "native", "framework", "adapter", "pipeline", "model"}
                }
            ),
            "symbol_kinds": list(record.get("symbol_kinds") or []),
            "terms": sorted(
                self._record_tokens(record)
                | set(self._record_role_tags(record))
                | set(self._record_feature_families(record))
            ),
        }

    @staticmethod
    def _comparative_relation_family(native_relation: str) -> str:
        mapping = {
            "adaptation_candidate": "native_feature_adaptation",
            "analogous_to": "analogous_mechanism",
            "evaluation_analogue": "evaluation_analogue",
            "native_upgrade_path": "performance_upgrade_path",
            "port_program_candidate": "port_program_candidate",
            "subsystem_analogue": "disparate_mechanism_candidate",
        }
        return mapping.get(str(native_relation or ""), "analogous_mechanism")

    def _record_priority(self, record: dict[str, Any]) -> tuple[float, int, str]:
        tags = set(record.get("tags") or [])
        path = str(record.get("path") or "")
        feature_weight = sum(4 for tag in self._record_feature_families(record))
        role_weight = sum(
            2
            for tag in self._record_role_tags(record)
            if tag not in {"example_surface", "test_harness", "config_surface", "module_init"}
        )
        native_bonus = 6 if "native" in tags else 0
        source_bonus = 4 if record.get("classification") == "source" else 0
        doc_penalty = -6 if "doc" in tags else 0
        noise_penalty = -4 * len(self._noise_flags(record))
        symbol_count = len(record.get("symbols") or [])
        return (
            float(
                symbol_count * 2
                + feature_weight
                + role_weight
                + native_bonus
                + source_bonus
                + doc_penalty
                + noise_penalty
            ),
            symbol_count,
            path,
        )

    def _prioritize_records_for_comparison(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return sorted(
            records,
            key=lambda item: self._record_priority(item),
            reverse=True,
        )

    def _feature_pair_candidates(
        self,
        *,
        target_records: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
        top_k: int,
    ) -> list[tuple[int, int, float, list[str]]]:
        ranked: list[tuple[int, int, float, list[str]]] = []
        for target_index, target_record in enumerate(target_records):
            for candidate_index, candidate_record in enumerate(candidate_records):
                shared_tags = self._shared_feature_tags(target_record, candidate_record)
                if not shared_tags:
                    continue
                noise_penalty = 0.05 * len(self._noise_flags(target_record) + self._noise_flags(candidate_record))
                score = min(
                    0.92,
                    0.18
                    + (0.1 * len(shared_tags))
                    + (0.04 * min(len(target_record.get("symbols") or []), 4))
                    + (0.04 * min(len(candidate_record.get("symbols") or []), 4))
                    + (0.03 * self._symbol_kind_alignment(target_record, candidate_record))
                    - noise_penalty,
                )
                if score >= 0.18:
                    ranked.append((target_index, candidate_index, round(score, 4), shared_tags))
        ranked.sort(key=lambda item: (-float(item[2]), item[1], item[0]))
        return ranked[:top_k]

    def _disparate_pair_candidates(
        self,
        *,
        target_records: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
        top_k: int,
    ) -> list[tuple[int, int, float, list[str]]]:
        ranked: list[tuple[int, int, float, list[str]]] = []
        for target_index, target_record in enumerate(target_records):
            target_roles = self._record_role_tags(target_record)
            if not target_roles:
                continue
            for candidate_index, candidate_record in enumerate(candidate_records):
                shared_roles = self._shared_role_tags(target_record, candidate_record)
                if not shared_roles:
                    continue
                feature_families = sorted(
                    set(self._record_feature_families(target_record))
                    | set(self._record_feature_families(candidate_record))
                )
                noise_penalty = 0.04 * len(self._noise_flags(target_record) + self._noise_flags(candidate_record))
                score = min(
                    0.88,
                    0.2
                    + (0.08 * len(shared_roles))
                    + (0.03 * len(feature_families))
                    + (0.05 * self._symbol_kind_alignment(target_record, candidate_record))
                    - noise_penalty,
                )
                if score < 0.22:
                    continue
                ranked.append((target_index, candidate_index, round(score, 4), shared_roles))
        ranked.sort(key=lambda item: (-float(item[2]), item[1], item[0]))
        return ranked[:top_k]

    def _relation_type(
        self,
        *,
        score: float,
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
        shared_feature_tags: list[str],
        shared_role_tags: list[str],
    ) -> str:
        target_language = str(target_record.get("language") or "unknown")
        candidate_language = str(candidate_record.get("language") or "unknown")
        target_tags = set(target_record.get("tags") or [])
        candidate_tags = set(candidate_record.get("tags") or [])
        if "registry" in shared_role_tags and score >= 0.22:
            return "registry_analogue"
        if {"orchestration", "cli", "entrypoint"} & set(shared_role_tags) and score >= 0.22:
            return "orchestration_analogue"
        if {"artifact", "formatter", "report_surface"} & set(shared_role_tags) and score >= 0.22:
            return "reporting_analogue"
        if {"adapter", "service", "state", "session"} & set(shared_role_tags) and score >= 0.22:
            return "disparate_mechanism_candidate"
        if {"pipeline", "model"} & set(candidate_tags | set(shared_role_tags)) and score >= 0.22:
            return "evaluation_analogue"
        if shared_feature_tags and score >= 0.24 and target_language != candidate_language:
            return "native_feature_adaptation"
        if shared_feature_tags and score >= 0.22:
            return "feature_gap_candidate"
        if shared_role_tags and score >= 0.26:
            return "port_program_candidate"
        if "native" in candidate_tags and "native" not in target_tags and score >= 0.24:
            return "performance_upgrade_path"
        if target_language != candidate_language and score >= 0.28:
            return "ffi_analogue" if "native" in candidate_tags or "native" in target_tags else "rewrite_candidate"
        if score >= 0.58:
            return "copy_candidate"
        if score >= 0.34:
            return "portable_pattern"
        return "analogous_mechanism"

    def _arbitrate_relation(
        self,
        *,
        relation_type: str,
        score: float,
        target_record: dict[str, Any],
        candidate_record: dict[str, Any],
        target_pack: dict[str, Any],
        candidate_pack: dict[str, Any],
        shared_feature_tags: list[str],
        shared_role_tags: list[str],
    ) -> dict[str, str]:
        same_language = str(target_record.get("language") or "") == str(candidate_record.get("language") or "")
        candidate_tags = set(candidate_record.get("tags") or [])
        preferred_language = self._preferred_implementation_language(target_pack)
        if relation_type == "copy_candidate" and same_language and score >= 0.65:
            return {
                "posture": "direct_copy",
                "rationale": "High structural overlap and language parity favor a direct transplant with verification.",
            }
        if relation_type in {
            "disparate_mechanism_candidate",
            "port_program_candidate",
            "performance_upgrade_path",
            "registry_analogue",
            "orchestration_analogue",
            "reporting_analogue",
            "evaluation_analogue",
            "rewrite_candidate",
            "native_feature_adaptation",
            "feature_gap_candidate",
        } or ("native" in candidate_tags and not same_language):
            return {
                "posture": "native_rewrite",
                "rationale": (
                    "Mechanism is relevant and should be recreated natively in "
                    f"{preferred_language}, with Python reserved for thin wrapper layers."
                ),
            }
        if relation_type == "ffi_analogue" and "ffi" in candidate_tags:
            return {
                "posture": "wrapper_integration",
                "rationale": "Cross-language alignment is real, but the safest first step is wrapper or FFI-style integration.",
            }
        if shared_feature_tags or shared_role_tags or score >= 0.35:
            return {
                "posture": "native_rewrite",
                "rationale": (
                    "Shared mechanism exists, but the target should absorb it through a native rewrite "
                    f"in {preferred_language} rather than a foreign-language transplant."
                ),
            }
        return {
            "posture": "pattern_only_adoption",
            "rationale": "Evidence supports borrowing the pattern rather than copying implementation.",
        }

    def _best_symbol(self, file_record: dict[str, Any]) -> dict[str, Any]:
        symbols = list(file_record.get("symbols") or [])
        if not symbols:
            return {"name": str(file_record.get("path") or ""), "kind": "file"}
        return symbols[0]

    def _record_tokens(self, file_record: dict[str, Any]) -> set[str]:
        tokens = _tokenize(file_record.get("path", ""))
        tokens |= {str(tag).lower() for tag in file_record.get("tags", [])}
        for note in file_record.get("analysis_notes", []) or []:
            tokens |= _tokenize(note)
        for symbol in file_record.get("symbols", []) or []:
            tokens |= _tokenize(symbol.get("name", ""))
        return tokens

    def _record_token_ids(self, file_record: dict[str, Any]) -> tuple[list[int], list[str]]:
        tokens = sorted(self._record_tokens(file_record))
        token_id_map = {
            token: int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF
            for token in tokens
        }
        token_ids = sorted(set(token_id_map.values()))
        return token_ids, tokens

    def _parser_or_none(self) -> SAGUAROParser | None:
        if self._parser is not None:
            return self._parser
        try:
            self._parser = SAGUAROParser()
        except Exception:
            self._parser = None
        return self._parser

    @staticmethod
    def _compare_pack_path(session_dir: str) -> str:
        return os.path.join(session_dir, "compare_pack.json")

    def _signature_cache_path(self, session_dir: str, pack: dict[str, Any]) -> str:
        digest = str(pack.get("snapshot_digest") or self._snapshot_digest(pack))
        return os.path.join(session_dir, f"signature_cache.{digest}.json")

    @staticmethod
    def _compare_ready_pack(pack: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "comparative_compare_pack.v1",
            "producer": str(pack.get("producer") or "ComparativeAnalysisService.native_index"),
            "repo_id": pack.get("repo_id"),
            "repo_path": pack.get("repo_path"),
            "file_count": int(pack.get("file_count", 0) or 0),
            "files": list(pack.get("files") or []),
            "languages": dict(pack.get("languages") or {}),
            "tech_stack": list(pack.get("tech_stack") or []),
            "build_files": list(pack.get("build_files") or []),
            "index_stats": dict(pack.get("index_stats") or {}),
            "graph_stats": dict(pack.get("graph_stats") or {}),
            "build_fingerprint": dict(pack.get("build_fingerprint") or {}),
            "language_truth_matrix": dict(pack.get("language_truth_matrix") or {}),
            "capability_matrix": dict(pack.get("capability_matrix") or {}),
            "semantic_inventory": dict(pack.get("semantic_inventory") or {}),
            "repo_dossier": dict(pack.get("repo_dossier") or {}),
            "snapshot_digest": str(pack.get("snapshot_digest") or ""),
        }

    def _persist_signature_cache(self, session: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
        cache = self._load_signature_cache(session, pack, build_if_missing=True)
        path = str((session.get("artifact_paths") or {}).get("signature_cache") or "")
        if path:
            atomic_write_json(path, cache, indent=2, sort_keys=True)
        return cache

    def _load_signature_cache(
        self,
        session: dict[str, Any],
        pack: dict[str, Any],
        *,
        build_if_missing: bool = False,
    ) -> dict[str, Any]:
        cache_key = f"{session.get('corpus_id')}::{session.get('snapshot_digest') or self._snapshot_digest(pack)}"
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]
        path = str((session.get("artifact_paths") or {}).get("signature_cache") or "")
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    self._signature_cache[cache_key] = payload
                    return payload
            except Exception:
                pass
        if not build_if_missing:
            return {}
        rows = {}
        for record in pack.get("files", []):
            if record.get("classification") != "source":
                continue
            token_ids, tokens = self._record_token_ids(record)
            rows[str(record.get("path") or "")] = {
                "token_ids": token_ids,
                "tokens": tokens,
            }
        payload = {
            "schema_version": "comparative_signature_cache.v1",
            "corpus_id": session.get("corpus_id"),
            "snapshot_digest": session.get("snapshot_digest") or self._snapshot_digest(pack),
            "rows": rows,
        }
        self._signature_cache[cache_key] = payload
        return payload

    def _token_rows_for_records(
        self,
        session: dict[str, Any],
        pack: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> list[tuple[list[int], list[str]]]:
        cache = self._load_signature_cache(session, pack, build_if_missing=True)
        rows = dict(cache.get("rows") or {})
        token_rows: list[tuple[list[int], list[str]]] = []
        for record in records:
            cached = dict(rows.get(str(record.get("path") or "")) or {})
            if cached:
                token_rows.append(
                    (
                        [int(item) for item in list(cached.get("token_ids") or [])],
                        [str(item) for item in list(cached.get("tokens") or [])],
                    )
                )
            else:
                token_rows.append(self._record_token_ids(record))
        return token_rows

    def _target_twin(
        self,
        session: dict[str, Any],
        pack: dict[str, Any],
    ) -> dict[str, Any]:
        cache_key = f"{session.get('corpus_id')}::{session.get('snapshot_digest') or self._snapshot_digest(pack)}"
        cached = self._target_twin_cache.get(cache_key)
        if cached:
            return {**cached, "cache_hit": True}
        started = time.perf_counter()
        records = self._prioritize_records_for_comparison(
            [
                item
                for item in pack.get("files", [])
                if item.get("classification") == "source"
            ]
        )
        token_rows = self._token_rows_for_records(session, pack, records)
        cache_bytes = len(json.dumps({"rows": token_rows}, sort_keys=True, default=str).encode("utf-8"))
        twin = {
            "records": records,
            "token_rows": token_rows,
            "signature_build_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "cache_bytes": cache_bytes,
            "cache_hit": False,
        }
        self._target_twin_cache[cache_key] = twin
        return twin

    def _native_indexer_or_none(self) -> Any | None:
        if self._native_indexer is not None:
            return self._native_indexer
        try:
            self._native_indexer = get_native_indexer()
        except Exception:
            self._native_indexer = None
        return self._native_indexer

    def _rank_record_pairs(
        self,
        *,
        target_records: list[dict[str, Any]],
        candidate_records: list[dict[str, Any]],
        target_token_rows: list[tuple[list[int], list[str]]] | None = None,
        candidate_token_rows: list[tuple[list[int], list[str]]] | None = None,
        top_k: int,
    ) -> tuple[list[tuple[int, int, float, list[str]]], str, dict[str, Any]]:
        if not target_records or not candidate_records:
            return [], "empty", {
                "pair_candidates_before_filter": 0,
                "pair_candidates_after_filter": 0,
                "pair_screen_ms": 0.0,
                "pair_filter_recall": 1.0,
            }

        target_token_rows = list(target_token_rows or [self._record_token_ids(item) for item in target_records])
        candidate_token_rows = list(
            candidate_token_rows or [self._record_token_ids(item) for item in candidate_records]
        )
        screen_started = time.perf_counter()
        screened_candidate_map, screen_backend = self._screen_candidate_indices(
            target_token_rows=target_token_rows,
            candidate_token_rows=candidate_token_rows,
            top_k=top_k,
        )
        pair_candidates_before_filter = len(target_token_rows) * len(candidate_token_rows)
        pair_candidates_after_filter = sum(len(item) for item in screened_candidate_map.values())
        backend = "python"
        native_indexer = self._native_indexer_or_none()
        ranked: list[tuple[int, int, float, list[str]]] = []
        if native_indexer is not None:
            try:
                for target_index, pool in screened_candidate_map.items():
                    if not pool:
                        continue
                    subset_rows = [candidate_token_rows[candidate_index] for candidate_index in pool]
                    indices, scores = self._rank_record_pairs_native(
                        native_indexer=native_indexer,
                        target_token_rows=[target_token_rows[target_index]],
                        candidate_token_rows=subset_rows,
                        top_k=min(top_k, len(subset_rows)),
                    )
                    left_tokens = set(target_token_rows[target_index][1])
                    for slot in range(indices.shape[1]):
                        pool_index = int(indices[0, slot])
                        score = float(scores[0, slot])
                        if pool_index < 0 or pool_index >= len(pool) or score <= 0.0:
                            continue
                        candidate_index = pool[pool_index]
                        right_tokens = set(candidate_token_rows[candidate_index][1])
                        ranked.append(
                            (
                                target_index,
                                candidate_index,
                                score,
                                sorted(left_tokens & right_tokens),
                            )
                        )
                backend = "native_cpp"
            except NativeIndexerError:
                backend = "python_fallback"

        if backend != "native_cpp":
            ranked = []
            for target_index, (_, left_tokens) in enumerate(target_token_rows):
                left_set = set(left_tokens)
                if not left_set:
                    continue
                scored: list[tuple[int, float, list[str]]] = []
                for candidate_index in screened_candidate_map.get(target_index, []):
                    right_set = set(candidate_token_rows[candidate_index][1])
                    score = _jaccard(left_set, right_set)
                    if score <= 0.0:
                        continue
                    scored.append((candidate_index, score, sorted(left_set & right_set)))
                scored.sort(key=lambda item: (-item[1], item[0]))
                for candidate_index, score, shared in scored[:top_k]:
                    ranked.append((target_index, candidate_index, score, shared))
        return ranked, backend, {
            "pair_candidates_before_filter": pair_candidates_before_filter,
            "pair_candidates_after_filter": pair_candidates_after_filter,
            "pair_screen_ms": round((time.perf_counter() - screen_started) * 1000.0, 3),
            "pair_filter_recall": 1.0 if ranked else 0.0,
            "pair_screen_backend": screen_backend,
        }

    def _screen_candidate_indices(
        self,
        *,
        target_token_rows: list[tuple[list[int], list[str]]],
        candidate_token_rows: list[tuple[list[int], list[str]]],
        top_k: int,
    ) -> tuple[dict[int, list[int]], str]:
        native_indexer = self._native_indexer_or_none()
        if native_indexer is not None:
            try:
                screened = self._screen_candidate_indices_native(
                    native_indexer=native_indexer,
                    target_token_rows=target_token_rows,
                    candidate_token_rows=candidate_token_rows,
                    top_k=top_k,
                )
                return screened, "native_cpp"
            except NativeIndexerError:
                pass

        inverted: dict[int, list[int]] = defaultdict(list)
        for candidate_index, (token_ids, _) in enumerate(candidate_token_rows):
            for token_id in token_ids:
                inverted[int(token_id)].append(candidate_index)

        screened: dict[int, list[int]] = {}
        pool_limit = max(16, min(96, top_k * 8))
        default_pool = list(range(min(len(candidate_token_rows), pool_limit)))
        for target_index, (token_ids, _) in enumerate(target_token_rows):
            counts: Counter[int] = Counter()
            for token_id in token_ids:
                for candidate_index in inverted.get(int(token_id), []):
                    counts[candidate_index] += 1
            if counts:
                pool = [
                    candidate_index
                    for candidate_index, _count in sorted(
                        counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:pool_limit]
                ]
                screened[target_index] = pool
            else:
                screened[target_index] = list(default_pool)
        return screened, "python"

    @staticmethod
    def _screen_candidate_indices_native(
        *,
        native_indexer: Any,
        target_token_rows: list[tuple[list[int], list[str]]],
        candidate_token_rows: list[tuple[list[int], list[str]]],
        top_k: int,
    ) -> dict[int, list[int]]:
        token_stride = max(
            max((len(ids) for ids, _ in target_token_rows), default=0),
            max((len(ids) for ids, _ in candidate_token_rows), default=0),
            1,
        )
        left_tokens = np.zeros((len(target_token_rows), token_stride), dtype=np.int32)
        left_lengths = np.zeros(len(target_token_rows), dtype=np.int32)
        right_tokens = np.zeros((len(candidate_token_rows), token_stride), dtype=np.int32)
        right_lengths = np.zeros(len(candidate_token_rows), dtype=np.int32)

        for row_index, (token_ids, _) in enumerate(target_token_rows):
            left_lengths[row_index] = len(token_ids)
            if token_ids:
                left_tokens[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
        for row_index, (token_ids, _) in enumerate(candidate_token_rows):
            right_lengths[row_index] = len(token_ids)
            if token_ids:
                right_tokens[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)

        pool_limit = max(16, min(96, top_k * 8))
        indices, _scores = native_indexer.screen_overlap_pairs(
            left_tokens,
            left_lengths,
            right_tokens,
            right_lengths,
            top_k=min(pool_limit, max(1, len(candidate_token_rows))),
        )
        default_pool = list(range(min(len(candidate_token_rows), pool_limit)))
        screened: dict[int, list[int]] = {}
        for row_index in range(indices.shape[0]):
            pool = [
                int(candidate_index)
                for candidate_index in indices[row_index].tolist()
                if 0 <= int(candidate_index) < len(candidate_token_rows)
            ]
            screened[row_index] = pool or list(default_pool)
        return screened

    def _rank_record_pairs_native(
        self,
        *,
        native_indexer: Any,
        target_token_rows: list[tuple[list[int], list[str]]],
        candidate_token_rows: list[tuple[list[int], list[str]]],
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        token_stride = max(
            max((len(ids) for ids, _ in target_token_rows), default=0),
            max((len(ids) for ids, _ in candidate_token_rows), default=0),
            1,
        )
        left_tokens = np.zeros((len(target_token_rows), token_stride), dtype=np.int32)
        left_lengths = np.zeros(len(target_token_rows), dtype=np.int32)
        right_tokens = np.zeros((len(candidate_token_rows), token_stride), dtype=np.int32)
        right_lengths = np.zeros(len(candidate_token_rows), dtype=np.int32)

        for row_index, (token_ids, _) in enumerate(target_token_rows):
            left_lengths[row_index] = len(token_ids)
            if token_ids:
                left_tokens[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)
        for row_index, (token_ids, _) in enumerate(candidate_token_rows):
            right_lengths[row_index] = len(token_ids)
            if token_ids:
                right_tokens[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int32)

        return native_indexer.rank_jaccard_pairs(
            left_tokens,
            left_lengths,
            right_tokens,
            right_lengths,
            top_k=top_k,
        )

    def _requalify_slice_result(self, result: dict[str, Any], *, corpus_id: str) -> None:
        focus_file = ""
        focus_name = str(result.get("focus_symbol") or "")
        focus_type = "symbol"
        for item in list(result.get("content") or []):
            item["corpus_id"] = corpus_id
            item_file = str(item.get("file") or "")
            item_name = str(item.get("name") or focus_name or item_file)
            item_type = str(item.get("type") or "symbol")
            if item_file:
                item["qualified_symbol_id"] = self._qualified_symbol_id(
                    corpus_id,
                    item_file,
                    item_name,
                    item_type,
                )
            if item.get("role") == "focus":
                focus_file = item_file
                focus_name = item_name
                focus_type = item_type
        if focus_file:
            result["qualified_symbol_id"] = self._qualified_symbol_id(
                corpus_id,
                focus_file,
                focus_name,
                focus_type,
            )

    def _build_native_pack(
        self,
        *,
        index_root: str,
        corpus_id: str,
        root_path: str,
        kind: str,
    ) -> dict[str, Any]:
        metadata = self._load_json_file(
            os.path.join(index_root, "vectors", "metadata.json"),
            default=[],
        )
        tracking = self._load_json_file(
            os.path.join(index_root, "tracking.json"),
            default={},
        )
        graph = self._load_json_file(
            os.path.join(index_root, "graph", "graph.json"),
            default={},
        )
        index_stats = self._load_json_file(
            os.path.join(index_root, "index_stats.json"),
            default={},
        )

        file_records: dict[str, dict[str, Any]] = {}
        languages: Counter[str] = Counter()
        tech_stack: set[str] = set()
        build_files: list[str] = []
        test_files: list[str] = []
        entry_points: list[str] = []
        operator_surfaces: list[str] = []
        file_graph_evidence: dict[str, dict[str, Any]] = {}

        for row in list(metadata or []):
            rel_path = canonicalize_rel_path(
                str(row.get("file") or ""),
                repo_path=root_path,
            )
            if not rel_path:
                continue
            record = file_records.setdefault(
                rel_path,
                self._new_file_record(rel_path),
            )
            symbol = {
                "name": str(row.get("name") or ""),
                "kind": str(row.get("entity_kind") or row.get("type") or "symbol"),
                "line": int(row.get("line") or 1),
                "end_line": int(row.get("end_line") or row.get("line") or 1),
            }
            if symbol["name"] and all(
                existing.get("name") != symbol["name"] or existing.get("line") != symbol["line"]
                for existing in record["symbols"]
            ):
                record["symbols"].append(symbol)
            record["symbol_kinds"] = sorted(
                set(record.get("symbol_kinds") or []) | {symbol["kind"]}
            )
            record["classification"] = str(row.get("file_role") or record["classification"])
            language = self._language_for_path(rel_path)
            record["language"] = language
            record["digest"] = str(
                (tracking.get(os.path.join(root_path, rel_path)) or {}).get("hash")
                or record.get("digest")
                or ""
            )
            record["analysis_notes"] = [
                "native_indexed",
                f"entity_count={len(record['symbols'])}",
            ]
            record["tags"] = sorted(
                set(record.get("tags") or [])
                | self._tags_for_record(
                    rel_path,
                    classification=record["classification"],
                    language=language,
                )
            )
            record["imports"] = sorted(
                set(record.get("imports") or [])
                | {
                    str(item)
                    for item in list(row.get("imports") or [])
                    if str(item).strip()
                }
            )

        for abs_path, meta in dict(tracking or {}).items():
            rel_path = canonicalize_rel_path(abs_path, repo_path=root_path)
            if not rel_path:
                continue
            record = file_records.setdefault(
                rel_path,
                self._new_file_record(rel_path),
            )
            language = record.get("language") or self._language_for_path(rel_path)
            record["language"] = language
            record["classification"] = classify_file_role(rel_path)
            record["digest"] = str((meta or {}).get("hash") or record.get("digest") or "")
            record["tags"] = sorted(
                set(record.get("tags") or [])
                | self._tags_for_record(
                    rel_path,
                    classification=record["classification"],
                    language=language,
                )
            )
            record["loc"] = self._count_loc(root_path, rel_path)

        for rel_path, graph_row in dict(graph.get("files") or {}).items():
            canonical_path = canonicalize_rel_path(str(rel_path or ""), repo_path=root_path)
            if not canonical_path:
                continue
            record = file_records.setdefault(
                canonical_path,
                self._new_file_record(canonical_path),
            )
            edges = list(graph_row.get("edges") or [])
            imports = sorted(
                {
                    str(
                        edge.get("target") or edge.get("dst") or ""
                    )
                    if isinstance(edge, dict)
                    else str(edge or "")
                    for edge in edges
                    if (
                        str(edge.get("target") or edge.get("dst") or "").strip()
                        if isinstance(edge, dict)
                        else str(edge or "").strip()
                    )
                }
            )
            record["imports"] = sorted(set(record.get("imports") or []) | set(imports))
            record["graph_evidence"] = {
                "edge_count": len(edges),
                "node_count": len(list(graph_row.get("nodes") or [])),
                "module_hint": str(graph_row.get("module_hint") or ""),
                "ffi_patterns": list(graph_row.get("ffi_patterns") or []),
            }
            file_graph_evidence[canonical_path] = dict(record["graph_evidence"])

        files = sorted(file_records.values(), key=lambda item: str(item.get("path") or ""))
        loc_total = 0
        for record in files:
            languages.update([str(record.get("language") or "unknown")])
            tech_stack.update(record.get("tags") or [])
            loc_total += int(record.get("loc") or 0)
            if self._is_build_file(str(record.get("path") or "")):
                build_files.append(str(record.get("path") or ""))
            if (
                record.get("classification") == "test"
                or "/tests/" in f"/{record.get('path')}"
                or str(record.get("path") or "").startswith("tests/")
            ):
                test_files.append(str(record.get("path") or ""))
            if self._is_entry_surface(str(record.get("path") or "")):
                entry_points.append(str(record.get("path") or ""))
            if self._is_operator_surface(record):
                operator_surfaces.append(str(record.get("path") or ""))

        top_reuse = sorted(
            (
                {
                    "path": str(record.get("path") or ""),
                    "language": str(record.get("language") or "unknown"),
                    "symbol_count": len(record.get("symbols") or []),
                    "reason": "native comparative candidate",
                }
                for record in files
                if record.get("classification") == "source"
            ),
            key=lambda item: (-int(item.get("symbol_count") or 0), item.get("path", "")),
        )[:16]
        graph_stats = dict(graph.get("stats") or {})
        repo_dossier = {
            "schema_version": "repo_dossier.v1",
            "dossier_id": f"{corpus_id}:repo_dossier",
            "repo_id": corpus_id,
            "repo_path": root_path,
            "summary": {
                "kind": kind,
                "file_count": len(files),
                "source_file_count": sum(
                    1 for item in files if item.get("classification") == "source"
                ),
                "language_count": len(languages),
                "indexed_entities": int(index_stats.get("document_count", len(metadata)) or len(metadata)),
                "graph_nodes": int(graph_stats.get("nodes", 0) or 0),
                "graph_edges": int(graph_stats.get("edges", 0) or 0),
            },
            "brief_summary": (
                f"{corpus_id} has {len(files)} tracked files across "
                f"{len(languages)} languages with "
                f"{int(index_stats.get('document_count', len(metadata)) or len(metadata))} indexed entities."
            ),
            "languages": dict(languages),
            "tech_stack": sorted(tech_stack),
            "reuse_candidates": top_reuse,
        }
        return {
            "schema_version": "comparative_native_pack.v2",
            "producer": "ComparativeAnalysisService.native_index",
            "repo_id": corpus_id,
            "repo_path": root_path,
            "file_count": len(files),
            "loc": loc_total,
            "files": files,
            "languages": dict(languages),
            "tech_stack": sorted(tech_stack),
            "build_files": sorted(set(build_files)),
            "test_files": sorted(set(test_files)),
            "entry_points": sorted(set(entry_points)),
            "operator_surfaces": sorted(set(operator_surfaces)),
            "index_stats": dict(index_stats),
            "graph_stats": graph_stats,
            "file_graph_evidence": file_graph_evidence,
            "repo_dossier": repo_dossier,
        }

    @staticmethod
    def _load_json_file(path: str, *, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return default

    def _new_file_record(self, rel_path: str) -> dict[str, Any]:
        return {
            "path": rel_path,
            "classification": classify_file_role(rel_path),
            "language": self._language_for_path(rel_path),
            "digest": "",
            "symbols": [],
            "symbol_kinds": [],
            "tags": [],
            "imports": [],
            "loc": 0,
            "graph_evidence": {},
            "analysis_notes": ["native_indexed"],
        }

    def _session_api_for_root(self, root_path: str, *, index_root: str) -> Any:
        key = (os.path.abspath(root_path), os.path.abspath(index_root))
        cached = self._api_cache.get(key)
        if cached is not None:
            return cached
        api_cls = self._load_api_cls()
        extra_exclusions = (
            list(_EXTERNAL_TEMP_EXCLUDE_PATTERNS)
            if self._infer_corpus_kind(root_path) != "primary"
            else []
        )
        api = api_cls(
            repo_path=root_path,
            saguaro_dir=index_root,
            extra_exclusions=extra_exclusions,
        )
        self._api_cache[key] = api
        return api

    def _session_runtime_for_root(
        self,
        root_path: str,
        *,
        index_root: str,
    ) -> ComparativeIndexRuntime:
        key = (os.path.abspath(root_path), os.path.abspath(index_root))
        cached = self._runtime_cache.get(key)
        if cached is not None:
            return cached
        extra_exclusions = (
            list(_EXTERNAL_TEMP_EXCLUDE_PATTERNS)
            if self._infer_corpus_kind(root_path) != "primary"
            else []
        )
        runtime = ComparativeIndexRuntime(
            root_path,
            saguaro_dir=index_root,
            extra_exclusions=extra_exclusions,
        )
        self._runtime_cache[key] = runtime
        return runtime

    def _session_api(self, session: dict[str, Any]) -> Any:
        return self._session_api_for_root(
            str(session.get("root_path") or self.repo_path),
            index_root=str(session.get("index_root") or self.saguaro_dir),
        )

    def _run_corpus_benchmark_trial(
        self,
        *,
        path: str,
        alias: str,
        ttl_hours: float,
        quarantine: bool,
        trust_level: str,
        build_profile: str,
        batch_size: int,
        file_batch_size: int | None,
        rebuild: bool,
        iteration: int,
    ) -> dict[str, Any]:
        child_env = os.environ.copy()
        child_env["SAGUARO_INDEX_BATCH_SIZE"] = str(max(1, int(batch_size)))
        if file_batch_size is None:
            child_env.pop("SAGUARO_INDEX_FILE_BATCH_SIZE", None)
        else:
            child_env["SAGUARO_INDEX_FILE_BATCH_SIZE"] = str(max(1, int(file_batch_size)))
        child_env["SAGUARO_BENCH_REPO_ROOT"] = self.repo_path
        child_env["SAGUARO_BENCH_TARGET_PATH"] = path
        child_env["SAGUARO_BENCH_ALIAS"] = alias
        child_env["SAGUARO_BENCH_TTL_HOURS"] = str(float(ttl_hours or 24.0))
        child_env["SAGUARO_BENCH_QUARANTINE"] = "1" if quarantine else "0"
        child_env["SAGUARO_BENCH_TRUST_LEVEL"] = trust_level
        child_env["SAGUARO_BENCH_BUILD_PROFILE"] = build_profile
        child_env["SAGUARO_BENCH_REBUILD"] = "1" if rebuild else "0"
        script = """
import json
import os
import resource
import sys
import time

from saguaro.services.comparative import ComparativeAnalysisService

started = time.perf_counter()
try:
    comparative = ComparativeAnalysisService(os.environ["SAGUARO_BENCH_REPO_ROOT"])
    result = comparative.create_session(
        path=os.environ["SAGUARO_BENCH_TARGET_PATH"],
        alias=os.environ["SAGUARO_BENCH_ALIAS"],
        ttl_hours=float(os.environ.get("SAGUARO_BENCH_TTL_HOURS", "24.0") or 24.0),
        quarantine=os.environ.get("SAGUARO_BENCH_QUARANTINE", "1") == "1",
        trust_level=os.environ.get("SAGUARO_BENCH_TRUST_LEVEL", "medium"),
        build_profile=os.environ.get("SAGUARO_BENCH_BUILD_PROFILE", "auto"),
        rebuild=os.environ.get("SAGUARO_BENCH_REBUILD", "0") == "1",
    )
    session = dict(result.get("session") or {})
    pack = dict(result.get("analysis_pack") or {})
    telemetry = dict(session.get("telemetry") or {})
    native_index_result = dict(telemetry.get("native_index_result") or {})
    rss_value = float(getattr(resource.getrusage(resource.RUSAGE_SELF), "ru_maxrss", 0.0) or 0.0)
    rss_mb = rss_value / 1024.0
    if sys.platform == "darwin":
        rss_mb = rss_value / (1024.0 * 1024.0)
    payload = {
        "status": "ok",
        "session_status": str(result.get("status") or "ok"),
        "corpus_id": str(session.get("corpus_id") or ""),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "peak_rss_mb": round(rss_mb, 3),
        "indexed_files": int(native_index_result.get("indexed_files", 0) or 0),
        "indexed_entities": int(native_index_result.get("indexed_entities", 0) or 0),
        "focus_corpus_file_count": int(telemetry.get("focus_corpus_file_count", 0) or 0),
        "backend": str(native_index_result.get("backend") or ""),
        "producer": str(pack.get("producer") or ""),
    }
except Exception as exc:  # pragma: no cover - subprocess guardrail
    payload = {
        "status": "error",
        "message": str(exc),
        "wall_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
print(json.dumps(payload))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=self.repo_path,
            env=child_env,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        try:
            payload = json.loads(stdout) if stdout else {}
        except json.JSONDecodeError:
            payload = {
                "status": "error",
                "message": stdout or "benchmark subprocess emitted invalid JSON",
            }
        payload["batch_size"] = int(batch_size)
        payload["file_batch_size"] = file_batch_size
        payload["rebuild"] = bool(rebuild)
        payload["iteration"] = int(iteration)
        payload["returncode"] = int(completed.returncode)
        if stderr:
            payload["stderr"] = stderr
        if completed.returncode != 0 and payload.get("status") == "ok":
            payload["status"] = "error"
            payload["message"] = stderr or f"benchmark subprocess exited with {completed.returncode}"
        return payload

    def _benchmark_trial_summary(self, trials: list[dict[str, Any]]) -> dict[str, Any]:
        successful = [trial for trial in trials if str(trial.get("status") or "") == "ok"]
        if not successful:
            return {
                "trial_count": len(trials),
                "success_count": 0,
                "avg_wall_ms": None,
                "avg_peak_rss_mb": None,
            }
        avg_wall_ms = sum(float(item.get("wall_ms") or 0.0) for item in successful) / len(successful)
        avg_peak_rss_mb = sum(
            float(item.get("peak_rss_mb") or 0.0) for item in successful
        ) / len(successful)
        return {
            "trial_count": len(trials),
            "success_count": len(successful),
            "avg_wall_ms": round(avg_wall_ms, 3),
            "avg_peak_rss_mb": round(avg_peak_rss_mb, 3),
            "best_wall_ms": round(
                min(float(item.get("wall_ms") or 0.0) for item in successful), 3
            ),
            "max_peak_rss_mb": round(
                max(float(item.get("peak_rss_mb") or 0.0) for item in successful), 3
            ),
        }

    def _benchmark_result_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        cold_candidates = [
            entry
            for entry in results
            if (entry.get("cold_summary") or {}).get("avg_wall_ms") is not None
        ]
        warm_candidates = [
            entry
            for entry in results
            if (entry.get("warm_summary") or {}).get("avg_wall_ms") is not None
        ]

        def _pick_best(
            items: list[dict[str, Any]],
            *,
            key_name: str,
            summary_key: str,
        ) -> dict[str, Any] | None:
            if not items:
                return None
            best = min(
                items,
                key=lambda item: float((item.get(summary_key) or {}).get(key_name) or 0.0),
            )
            return {
                "batch_size": best.get("batch_size"),
                "file_batch_size": best.get("file_batch_size"),
                key_name: (best.get(summary_key) or {}).get(key_name),
            }

        return {
            "fastest_cold": _pick_best(
                cold_candidates,
                key_name="avg_wall_ms",
                summary_key="cold_summary",
            ),
            "lowest_cold_peak_rss": _pick_best(
                cold_candidates,
                key_name="avg_peak_rss_mb",
                summary_key="cold_summary",
            ),
            "fastest_warm": _pick_best(
                warm_candidates,
                key_name="avg_wall_ms",
                summary_key="warm_summary",
            ),
        }

    def _tags_for_record(
        self,
        rel_path: str,
        *,
        classification: str,
        language: str,
    ) -> set[str]:
        tags = {classification, language}
        lowered = rel_path.lower()
        path_parts = [part for part in lowered.split("/") if part]
        if language in {"c", "cpp", "c_header", "cpp_header", "rust", "go"}:
            tags.add("native")
        if self._is_build_file(rel_path):
            tags.add("build")
            if lowered.endswith("cmakelists.txt") or lowered.endswith(".cmake"):
                tags.add("cmake")
            elif lowered.endswith("cargo.toml"):
                tags.add("cargo")
            elif lowered.endswith("package.json"):
                tags.add("npm")
            elif lowered.endswith(("pyproject.toml", "requirements.txt", "setup.py")):
                tags.add("python")
        if "ffi" in lowered:
            tags.add("ffi")
        if os.path.basename(lowered) == "__init__.py":
            tags.add("module_init")
        if "/examples/" in f"/{lowered}/" or "/demo" in f"/{lowered}/":
            tags.update({"example_surface", "secondary_surface"})
        if "/tests/" in f"/{lowered}/" or lowered.startswith("tests/") or lowered.endswith("_test.py"):
            tags.update({"test_harness", "secondary_surface"})
        if any(part in {"core", "runtime", "engine", "service"} for part in path_parts):
            tags.update({"core_runtime", "service"})
        if any(part in {"cli", "terminal", "commands", "command", "shell"} for part in path_parts):
            tags.update({"cli", "orchestration", "entrypoint"})
        if any(part in {"registry", "registries", "plugins", "plugin"} for part in path_parts):
            tags.update({"registry", "plugin"})
        if any(part in {"state", "session", "context"} for part in path_parts):
            tags.update({"state", "session"})
        if any(part in {"output", "outputs", "report", "reporting", "artifact", "artifacts"} for part in path_parts):
            tags.update({"artifact", "report_surface"})
        if any(part in {"framework", "frameworks", "adapter", "adapters"} for part in path_parts):
            tags.update({"framework", "adapter"})
        if any(part in {"attack", "attacks", "exploit", "mutation"} for part in path_parts):
            tags.add("attack")
        if any(part in {"target", "targets"} for part in path_parts):
            tags.add("target")
        if any(part in {"optimize", "optimizer", "optimizers"} for part in path_parts):
            tags.add("optimizer")
        if any(part in {"model", "models", "inference", "pipeline", "evaluation"} for part in path_parts):
            tags.update({"model", "pipeline"})
        if lowered.endswith((".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf")):
            tags.add("config_surface")
        if lowered.endswith(".md"):
            tags.add("doc")
        if "grammar" in lowered:
            tags.add("grammar")
        feature_keywords = {
            "attack": "attack",
            "security": "security",
            "vuln": "vulnerability",
            "vulnerability": "vulnerability",
            "report": "reporting",
            "alert": "alerting",
            "taint": "taint",
            "dataflow": "dataflow",
            "query": "query_engine",
            "extractor": "extractor",
            "diagnostic": "diagnostics",
            "analy": "analysis",
            "target": "target",
            "framework": "framework",
            "adapter": "adapter",
            "orchestrat": "orchestration",
            "pipeline": "pipeline",
            "model": "model",
            "evaluat": "evaluation",
            "optim": "optimizer",
            "artifact": "artifact",
            "format": "formatter",
            "output": "artifact",
            "state": "state",
            "session": "session",
            "registry": "registry",
            "plugin": "plugin",
            "cli": "cli",
        }
        for needle, tag in feature_keywords.items():
            if needle in lowered:
                tags.add(tag)
        return {item for item in tags if item and item != "unknown"}

    @staticmethod
    def _is_build_file(rel_path: str) -> bool:
        lowered = rel_path.lower()
        base = os.path.basename(lowered)
        return base in {
            "cmakelists.txt",
            "cargo.toml",
            "package.json",
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "makefile",
            "meson.build",
        }

    @staticmethod
    def _language_for_path(rel_path: str) -> str:
        lowered = rel_path.lower()
        base = os.path.basename(lowered)
        if base == "cmakelists.txt" or lowered.endswith(".cmake"):
            return "cmake"
        if base == "cargo.toml":
            return "toml"
        if base == "package.json":
            return "json"
        if base in {"pyproject.toml", "requirements.txt", "setup.py"}:
            return "python"
        ext = os.path.splitext(lowered)[1]
        return {
            ".py": "python",
            ".cc": "cpp",
            ".cpp": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c_header",
            ".hh": "cpp_header",
            ".hpp": "cpp_header",
            ".rs": "rust",
            ".go": "go",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".ql": "ql",
            ".qll": "qll",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".sh": "shell",
            ".ini": "ini",
            ".cfg": "ini",
            ".conf": "ini",
            ".dbscheme": "dbscheme",
            ".bzl": "starlark",
            ".txt": "text",
        }.get(ext, "unknown")

    def _semantic_inventory(self, pack: dict[str, Any], corpus_id: str) -> dict[str, Any]:
        top_symbols: list[dict[str, Any]] = []
        function_body_graph_count = 0
        for file_record in pack.get("files", []):
            path_value = str(file_record.get("path") or "")
            for symbol in file_record.get("symbols", [])[:4]:
                top_symbols.append(
                    {
                        "qualified_symbol_id": self._qualified_symbol_id(
                            corpus_id,
                            path_value,
                            symbol.get("name", ""),
                            symbol.get("kind", "symbol"),
                        ),
                        "name": symbol.get("name", ""),
                        "kind": symbol.get("kind", "symbol"),
                        "path": path_value,
                        "line": int(symbol.get("line") or 1),
                    }
                )
                if symbol.get("kind") in {"function", "method"}:
                    function_body_graph_count += 1
        return {
            "top_symbols": top_symbols[:32],
            "function_body_graph_count": function_body_graph_count,
        }

    def _build_fingerprint(
        self,
        root_path: str,
        pack: dict[str, Any],
        *,
        build_profile: str,
    ) -> dict[str, Any]:
        build_files = list(pack.get("build_files") or [])
        primary = "none"
        if "CMakeLists.txt" in build_files:
            primary = "cmake"
        elif "Cargo.toml" in build_files:
            primary = "cargo"
        elif "package.json" in build_files:
            primary = "npm"
        elif "pyproject.toml" in build_files or "requirements.txt" in build_files or "setup.py" in build_files:
            primary = "python"
        include_roots = sorted(
            {
                os.path.dirname(str(item.get("path") or ""))
                for item in pack.get("files", [])
                if str(item.get("language") or "").startswith(("c", "cpp"))
            }
        )
        structured_build: dict[str, Any] = {}
        try:
            structured_build = BuildGraphIngestor(root_path).ingest()
        except Exception:
            structured_build = {}
        structured_inputs = dict(structured_build.get("structured_inputs") or {})
        targets = dict(structured_build.get("targets") or {})
        cargo_targets = self._cargo_targets(root_path)
        gradle_modules = self._gradle_modules(root_path)
        build_hint_coverage = sum(
            1
            for value in (
                structured_inputs.get("compile_databases"),
                structured_inputs.get("cmake_file_api_replies"),
                len(cargo_targets),
                len(gradle_modules),
            )
            if int(value or 0) > 0
        )
        return {
            "build_profile": build_profile,
            "primary_build_system": primary,
            "build_files": build_files,
            "include_roots": include_roots[:16],
            "repo_root": root_path,
            "build_backed": primary in {"cmake", "cargo", "npm", "python"},
            "build_fingerprint_depth": (
                "deep" if build_hint_coverage >= 2 else ("structured" if build_hint_coverage == 1 else "shallow")
            ),
            "compile_commands_found": int(structured_inputs.get("compile_databases", 0) or 0),
            "cmake_file_api_replies": int(structured_inputs.get("cmake_file_api_replies", 0) or 0),
            "structured_target_count": len(targets),
            "cargo_targets": cargo_targets,
            "gradle_modules": gradle_modules,
            "build_hint_coverage": build_hint_coverage,
            "build_hints": {
                "structured_inputs": structured_inputs,
                "targets": sorted(targets.keys())[:24],
            },
        }

    def _capability_matrix(
        self,
        pack: dict[str, Any],
        *,
        build_fingerprint: dict[str, Any],
        language_truth_matrix: dict[str, Any],
    ) -> dict[str, Any]:
        per_language = {}
        deep_languages: list[str] = []
        for language, row in sorted((language_truth_matrix.get("per_language") or {}).items()):
            lang = str(language or "unknown")
            if row.get("dependency_quality"):
                deep_languages.append(lang)
            per_language[lang] = {
                "count": int(row.get("count", 0) or 0),
                "lexical": bool(row.get("lexical")),
                "ast": bool(row.get("ast")),
                "cfg": bool(row.get("structural")),
                "dfg": bool(row.get("dependency_graph")),
                "symbol": bool(row.get("structural")),
                "build_backed": bool(row.get("build_backed")),
                "confidence": str(row.get("confidence") or "low"),
                "confidence_score": float(row.get("confidence_score") or 0.0),
                "support_tier": str(row.get("support_tier") or "unknown"),
            }
        return {
            "per_language": per_language,
            "deep_languages": sorted(set(deep_languages)),
            "deep_parse_coverage_percent": float(
                language_truth_matrix.get("deep_parse_coverage_percent") or 0.0
            ),
            "parser_environment": dict(language_truth_matrix.get("parser_environment") or {}),
        }

    def _language_truth_matrix(
        self,
        pack: dict[str, Any],
        *,
        build_fingerprint: dict[str, Any],
    ) -> dict[str, Any]:
        parser = self._parser_or_none()
        languages = dict(pack.get("languages") or {})
        parser_environment = (
            parser.environment_probe()
            if parser is not None
            else {
                "parser_environment_ready": False,
                "modules": {},
                "parser_plugin_count": 0,
                "grammar_probe_failures": ["parser_unavailable"],
                "language_golden_pass_rate": 0.0,
            }
        )
        per_language: dict[str, dict[str, Any]] = {}
        deep_count = 0
        total = max(1, sum(int(v) for v in languages.values()))
        for language, count in sorted(languages.items()):
            lang = str(language or "unknown")
            descriptor = (
                parser.language_support_descriptor(lang)
                if parser is not None
                else {
                    "lexical": True,
                    "ast": lang in _STRUCTURAL_LANGUAGES,
                    "structural": lang in _STRUCTURAL_LANGUAGES or lang in _SHALLOW_LANGUAGES,
                    "dependency_graph": lang in _STRUCTURAL_LANGUAGES,
                    "dependency_quality": lang in _STRUCTURAL_LANGUAGES,
                    "support_tier": "deep" if lang in _STRUCTURAL_LANGUAGES else "lexical",
                    "backend_kind": "none",
                }
            )
            build_backed = bool(build_fingerprint.get("build_backed")) and lang in {
                "c",
                "cpp",
                "c_header",
                "cpp_header",
                "rust",
                "go",
                "java",
            }
            confidence_score = self._capability_confidence(
                descriptor=descriptor,
                parser_environment=parser_environment,
                build_backed=build_backed,
            )
            confidence = (
                "high"
                if confidence_score >= 0.8
                else ("medium" if confidence_score >= 0.45 else "low")
            )
            if descriptor.get("dependency_quality"):
                deep_count += int(count or 0)
            per_language[lang] = {
                "count": int(count or 0),
                "lexical": bool(descriptor.get("lexical")),
                "declared_ast": bool(descriptor.get("declared_ast")),
                "ast": bool(descriptor.get("ast")),
                "structural": bool(descriptor.get("structural")),
                "dependency_graph": bool(descriptor.get("dependency_graph")),
                "dependency_quality": bool(descriptor.get("dependency_quality")),
                "build_backed": build_backed,
                "support_tier": str(descriptor.get("support_tier") or "unknown"),
                "backend_kind": str(descriptor.get("backend_kind") or "none"),
                "tree_sitter_language": str(descriptor.get("tree_sitter_language") or ""),
                "confidence": confidence,
                "confidence_score": confidence_score,
            }
        return {
            "schema_version": "language_truth_matrix.v1",
            "per_language": per_language,
            "parser_environment": parser_environment,
            "deep_parse_coverage_percent": round((deep_count / total) * 100.0, 1),
            "unsupported_language_regions": sorted(
                language
                for language, row in per_language.items()
                if not row.get("structural")
            ),
        }

    @staticmethod
    def _capability_confidence(
        *,
        descriptor: dict[str, Any],
        parser_environment: dict[str, Any],
        build_backed: bool,
    ) -> float:
        score = 0.1 if descriptor.get("lexical") else 0.0
        if descriptor.get("ast"):
            score += 0.35
        if descriptor.get("dependency_graph"):
            score += 0.2
        if descriptor.get("dependency_quality"):
            score += 0.15
        if build_backed:
            score += 0.15
        if parser_environment.get("parser_environment_ready"):
            score += 0.05
        return round(min(score, 1.0), 3)

    @staticmethod
    def _cargo_targets(root_path: str) -> list[str]:
        cargo_path = os.path.join(root_path, "Cargo.toml")
        if not os.path.exists(cargo_path):
            return []
        try:
            with open(cargo_path, "rb") as handle:
                payload = tomllib.load(handle)
        except Exception:
            return []
        targets: list[str] = []
        package = dict(payload.get("package") or {})
        if package.get("name"):
            targets.append(str(package["name"]))
        for key in ("bin", "example", "test", "bench"):
            for item in list(payload.get(key) or []):
                name = str((item or {}).get("name") or "").strip()
                if name:
                    targets.append(name)
        lib = dict(payload.get("lib") or {})
        if lib.get("name"):
            targets.append(str(lib["name"]))
        return sorted(set(targets))

    @staticmethod
    def _gradle_modules(root_path: str) -> list[str]:
        modules: set[str] = set()
        settings_paths = [
            os.path.join(root_path, "settings.gradle"),
            os.path.join(root_path, "settings.gradle.kts"),
        ]
        include_re = re.compile(r"include\s*\((?P<body>[^)]*)\)|include\s+(?P<flat>.+)")
        for settings_path in settings_paths:
            if not os.path.exists(settings_path):
                continue
            try:
                with open(settings_path, encoding="utf-8") as handle:
                    content = handle.read()
            except Exception:
                continue
            for match in include_re.finditer(content):
                body = str(match.group("body") or match.group("flat") or "")
                for item in re.findall(r"[\"']([^\"']+)[\"']", body):
                    modules.add(item.lstrip(":"))
        return sorted(item for item in modules if item)

    @staticmethod
    def _count_loc(root_path: str, rel_path: str) -> int:
        abs_path = os.path.join(root_path, rel_path)
        try:
            with open(abs_path, encoding="utf-8", errors="ignore") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError:
            return 0

    @staticmethod
    def _is_entry_surface(path: str) -> bool:
        normalized = path.replace("\\", "/").lower()
        return normalized.endswith(("__main__.py", "main.py", "cli.py", ".sh")) or normalized.startswith(
            ("bin/", "scripts/")
        )

    @staticmethod
    def _is_operator_surface(record: dict[str, Any]) -> bool:
        tags = {str(item) for item in record.get("tags") or []}
        return bool(
            tags
            & {
                "cli",
                "artifact",
                "report_surface",
                "orchestration",
                "session",
                "state",
            }
        )

    def _corpus_quality_metrics(
        self,
        *,
        pack: dict[str, Any],
        build_fingerprint: dict[str, Any],
        language_truth_matrix: dict[str, Any],
    ) -> dict[str, Any]:
        files = list(pack.get("files") or [])
        source_files = [
            item for item in files if str(item.get("classification") or "") == "source"
        ]
        import_coverage = (
            len([item for item in source_files if list(item.get("imports") or [])])
            / max(1, len(source_files))
        )
        return {
            "import_coverage": round(import_coverage, 4),
            "entry_surface_count": len(pack.get("entry_points") or []),
            "test_surface_count": len(pack.get("test_files") or []),
            "operator_surface_count": len(pack.get("operator_surfaces") or []),
            "build_truth_depth": str(
                build_fingerprint.get("build_fingerprint_depth") or "shallow"
            ),
            "parser_failure_count": len(pack.get("parser_failures") or []),
            "deep_parse_coverage_percent": float(
                language_truth_matrix.get("deep_parse_coverage_percent") or 0.0
            ),
        }

    def _fleet_router(
        self,
        *,
        target_pack: dict[str, Any],
        pending_sessions: list[dict[str, Any]],
        portfolio_top_n: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if len(pending_sessions) <= max(1, int(portfolio_top_n)):
            return pending_sessions, 0
        ranked: list[tuple[float, dict[str, Any]]] = []
        target_features = self._pack_feature_families(target_pack)
        target_languages = set((target_pack.get("languages") or {}).keys())
        target_build = str(
            ((target_pack.get("build_fingerprint") or {}).get("primary_build_system") or "")
        )
        for session in pending_sessions:
            pack = self._load_session_pack(str(session.get("corpus_id") or "")) or {}
            feature_overlap = len(target_features & self._pack_feature_families(pack))
            language_overlap = len(target_languages & set((pack.get("languages") or {}).keys()))
            build_match = float(
                str(((pack.get("build_fingerprint") or {}).get("primary_build_system") or ""))
                == target_build
            )
            quality = float(((pack.get("corpus_quality") or {}).get("import_coverage") or 0.0))
            score = (0.42 * feature_overlap) + (0.26 * language_overlap) + (0.18 * build_match) + (0.14 * quality)
            ranked.append((score, session))
        ranked.sort(
            key=lambda item: (-float(item[0]), str(item[1].get("corpus_id") or "")),
        )
        kept = [session for _score, session in ranked[: max(1, int(portfolio_top_n))]]
        return kept, max(0, len(pending_sessions) - len(kept))

    def _portfolio_leaderboard(
        self,
        *,
        comparisons: list[dict[str, Any]],
        native_programs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        programs_by_corpus: Counter[str] = Counter()
        for program in native_programs:
            programs_by_corpus.update([str(program.get("winner_corpus_id") or "")])
        rows = []
        for comparison in comparisons:
            candidate = dict(comparison.get("candidate") or {})
            candidate_id = str(candidate.get("corpus_id") or "")
            primary = list(comparison.get("primary_recommendations") or [])
            secondary = list(comparison.get("secondary_recommendations") or [])
            ranked = primary + secondary
            average_confidence = (
                sum(
                    float(item.get("calibrated_confidence") or item.get("relation_score") or 0.0)
                    for item in ranked
                )
                / max(1, len(ranked))
            )
            rows.append(
                {
                    "corpus_id": candidate_id,
                    "repo_path": candidate.get("root_path"),
                    "portfolio_rank_score": round(
                        average_confidence
                        + (0.08 * len(primary))
                        + (0.03 * programs_by_corpus.get(candidate_id, 0)),
                        4,
                    ),
                    "primary_count": len(primary),
                    "secondary_count": len(secondary),
                    "no_port_count": len([item for item in ranked if item.get("abstain_reason")]),
                    "build_truth_depth": str(
                        (((comparison.get("summary") or {}).get("build_alignment") or {}).get("candidate") or {}).get(
                            "build_fingerprint_depth",
                            "shallow",
                        )
                    ),
                }
            )
        rows.sort(
            key=lambda item: (-float(item.get("portfolio_rank_score") or 0.0), str(item.get("corpus_id") or "")),
        )
        return rows

    @staticmethod
    def _negative_evidence_ledger(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for comparison in comparisons:
            ranked = (
                list(comparison.get("primary_recommendations") or [])
                + list(comparison.get("secondary_recommendations") or [])
                + list(comparison.get("low_signal_relations") or [])
            )
            for relation in ranked:
                for item in list(relation.get("negative_evidence") or []):
                    rows.append(
                        {
                            "comparison_id": comparison.get("comparison_id"),
                            "recommendation_id": relation.get("recommendation_id"),
                            "source_path": relation.get("source_path"),
                            "target_path": relation.get("target_path"),
                            **item,
                        }
                    )
        return rows

    def _phase_packets_from_report(
        self,
        *,
        report: dict[str, Any],
        evidence_budget: int,
    ) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        target = dict(report.get("target") or {})
        promoted = sorted(
            list(report.get("native_migration_programs") or []),
            key=lambda item: (-float(item.get("priority") or 0.0), str(item.get("feature_family") or "")),
        )[: max(3, min(8, int(evidence_budget)))]
        if not promoted:
            promoted = []
            for comparison in list(report.get("comparisons") or [])[: max(1, int(evidence_budget))]:
                relation = (
                    list(comparison.get("primary_recommendations") or [])
                    or list(comparison.get("secondary_recommendations") or [])
                    or list(comparison.get("analogous_mechanisms") or [])
                )
                if not relation:
                    continue
                top = dict(relation[0])
                promoted.append(
                    {
                        "program_id": top.get("recommendation_id")
                        or f"fallback:{top.get('source_path')}:{top.get('target_path')}",
                        "feature_family": ", ".join(top.get("feature_families") or ["mechanism"]),
                        "priority": float(
                            top.get("calibrated_confidence") or top.get("relation_score") or 0.0
                        ),
                        "target_path": top.get("target_path"),
                        "source_path": top.get("source_path"),
                        "recipe_ir": {"dependencies": [top.get("target_path")], "rollback_criteria": ["targeted tests fail"]},
                        "flight_twin": {
                            "predicted_breakage_risk": float(
                                top.get("calibrated_confidence") or 0.0
                            )
                        },
                    }
                )
        for index, program in enumerate(promoted, start=1):
            packets.append(
                {
                    "phase_id": "development" if index <= 2 else "analysis_upgrade",
                    "objective": program.get("feature_family") or "comparative_migration",
                    "repo_scope": sorted(
                        {
                            str(program.get("target_path") or ""),
                            str(program.get("source_path") or ""),
                        }
                    ),
                    "owning_specialist_type": "migration_compiler_specialist",
                    "allowed_writes": sorted(
                        {
                            str(program.get("target_path") or ""),
                            "Saguaro/saguaro/services/comparative.py",
                        }
                    ),
                    "telemetry_contract": {
                        "predicted_breakage_risk": float(
                            ((program.get("flight_twin") or {}).get("predicted_breakage_risk") or 0.0)
                        ),
                        "portfolio_rank_score": float(program.get("priority") or 0.0),
                    },
                    "required_evidence": [
                        str(program.get("source_path") or ""),
                        str(program.get("target_path") or ""),
                        str(program.get("program_id") or ""),
                    ],
                    "rollback_criteria": list(
                        ((program.get("recipe_ir") or {}).get("rollback_criteria") or [])
                    )
                    or ["targeted tests fail"],
                    "promotion_gate": [
                        "landing zone proof reviewed",
                        "./venv/bin/saguaro verify . --engines native,ruff,semantic --format json",
                    ],
                    "success_criteria": [
                        f"Promote {program.get('feature_family')} into {target.get('corpus_id')}",
                    ],
                    "dependencies": list(
                        ((program.get("recipe_ir") or {}).get("dependencies") or [])[:6]
                    ),
                    "program_id": program.get("program_id"),
                }
            )
        return packets

    def _simulate_program(
        self,
        program: dict[str, Any],
        target_pack: dict[str, Any],
    ) -> dict[str, Any]:
        from core.research.comparative_flight_twin import ComparativeFlightTwin

        return ComparativeFlightTwin().simulate(program=program, target_pack=target_pack)

    def _focus_cone(self, root_path: str, pack: dict[str, Any]) -> dict[str, Any]:
        spillover: list[str] = []
        for file_record in pack.get("files", [])[:20]:
            imports = list(file_record.get("imports") or [])
            spillover.extend(imports[:4])
        return {
            "include_roots": [self._relative_root(root_path)],
            "spillover_edges": sorted({item for item in spillover if item})[:32],
        }

    def _session_summary(self, session: dict[str, Any], pack: dict[str, Any]) -> dict[str, Any]:
        return {
            "corpus_id": session["corpus_id"],
            "kind": session.get("kind"),
            "root_path": session.get("root_path"),
            "summary": dict((pack or {}).get("repo_dossier", {}).get("summary") or {}),
            "build_fingerprint": dict((pack or {}).get("build_fingerprint") or {}),
            "language_truth_matrix": dict((pack or {}).get("language_truth_matrix") or {}),
            "capability_matrix": dict((pack or {}).get("capability_matrix") or {}),
        }

    def _pack_summary(self, pack: dict[str, Any]) -> dict[str, Any]:
        dossier = dict(pack.get("repo_dossier") or {})
        return {
            "repo_id": pack.get("repo_id"),
            "repo_path": pack.get("repo_path"),
            "producer": str(pack.get("producer") or ""),
            "file_count": int(pack.get("file_count", 0) or 0),
            "loc": int(pack.get("loc", 0) or 0),
            "languages": dict(pack.get("languages") or {}),
            "entry_points": list(pack.get("entry_points") or []),
            "test_files": list(pack.get("test_files") or []),
            "corpus_quality": dict(pack.get("corpus_quality") or {}),
            "build_fingerprint": dict(pack.get("build_fingerprint") or {}),
            "language_truth_matrix": dict(pack.get("language_truth_matrix") or {}),
            "capability_matrix": dict(pack.get("capability_matrix") or {}),
            "summary": dict(dossier.get("summary") or {}),
        }

    def _session_pack_summary(self, corpus_id: str) -> dict[str, Any] | None:
        pack = self._load_session_pack(corpus_id)
        if not pack:
            return None
        return self._pack_summary(pack)

    def _load_session_pack(self, corpus_id: str) -> dict[str, Any] | None:
        session = self.state_ledger.get_corpus_session(corpus_id)
        if not session:
            return None
        artifact_paths = dict(session.get("artifact_paths") or {})
        candidate_paths = [
            str(artifact_paths.get("compare_pack") or ""),
            str(artifact_paths.get("analysis_pack") or ""),
        ]
        path = next(
            (item for item in candidate_paths if item and os.path.exists(item)),
            "",
        )
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return None

    def _selected_sessions(self, corpus_ids: list[str] | None = None) -> list[dict[str, Any]]:
        payload = self.state_ledger.list_corpus_sessions(include_expired=False)
        sessions = list(payload.get("sessions") or [])
        if not corpus_ids:
            return sessions
        allowed = {str(item) for item in corpus_ids}
        return [session for session in sessions if str(session.get("corpus_id") or "") in allowed]

    def _relative_root(self, root_path: str) -> str:
        if os.path.abspath(root_path) == self.repo_path:
            return "."
        if os.path.abspath(root_path).startswith(self.repo_path + os.sep):
            return os.path.relpath(root_path, self.repo_path).replace("\\", "/")
        return root_path

    def _snapshot_digest(self, pack: dict[str, Any]) -> str:
        file_digest = [
            (item.get("path"), item.get("digest"))
            for item in pack.get("files", [])
        ]
        return hashlib.sha1(
            json.dumps(file_digest, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _qualified_symbol_id(
        self,
        corpus_id: str,
        path: str,
        symbol_name: str,
        kind: str,
    ) -> str:
        return f"{corpus_id}:{path}:{kind}:{symbol_name}"

    def _resolve_input_path(self, path: str) -> str:
        candidate = str(path or ".").strip() or "."
        if os.path.isabs(candidate):
            return os.path.abspath(candidate)
        return os.path.abspath(os.path.join(self.repo_path, candidate))

    def _infer_corpus_kind(self, root_path: str) -> str:
        normalized = os.path.abspath(root_path)
        if normalized == self.repo_path:
            return "primary"
        if normalized.startswith(self.repo_path + os.sep):
            return "subtree"
        return "external"

    def _corpus_id_for_path(self, root_path: str, *, alias: str | None = None) -> str:
        base = alias or self._relative_root(root_path) or os.path.basename(root_path)
        digest = hashlib.sha1(os.path.abspath(root_path).encode("utf-8")).hexdigest()[:8]
        return f"{_slug(base)}-{digest}"

    def _discover_fleet_repos(self, fleet_root: str) -> list[str]:
        base = self._resolve_input_path(fleet_root)
        if not os.path.isdir(base):
            return []
        repos: list[str] = []
        for name in sorted(os.listdir(base)):
            candidate = os.path.join(base, name)
            if not os.path.isdir(candidate) or name.startswith(".") or name == "__pycache__":
                continue
            # repo_analysis fleet scans treat each first-level directory as a distinct repo.
            repos.append(candidate)
        return repos

    def _release_session_resources(self, session: dict[str, Any]) -> None:
        root_path = str(session.get("root_path") or "")
        index_root = str(session.get("index_root") or "")
        self._api_cache.pop((root_path, index_root), None)
        self._runtime_cache.pop((root_path, index_root), None)
        gc.collect()

    def _session_for_root(self, root_path: str) -> dict[str, Any] | None:
        target = os.path.abspath(root_path)
        payload = self.state_ledger.list_corpus_sessions(include_expired=True)
        for session in payload.get("sessions", []):
            if bool(session.get("expired", False)):
                continue
            if os.path.abspath(str(session.get("root_path") or "")) == target:
                return session
        return None
