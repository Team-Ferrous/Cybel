"""Utilities for health."""

import json
import logging
import os
import time
from typing import Any

import yaml

from saguaro.analysis.entry_points import EntryPointDetector
from saguaro.architecture import ArchitectureAnalyzer
from saguaro.build_system.ingestor import BuildGraphIngestor
from saguaro.coverage import CoverageReporter
from saguaro.errors import SaguaroStateCorruptionError
from saguaro.parsing import RuntimeSymbolResolver
from saguaro.storage.index_state import load_manifest, validate_manifest
from saguaro.storage.locks import RepoLockManager
from saguaro.utils.file_utils import build_corpus_manifest

logger = logging.getLogger(__name__)


def collect_native_compute_report() -> dict[str, Any]:
    """Collect native compute/runtime capability evidence for health reporting."""
    try:
        from saguaro.indexing.native_indexer_bindings import get_native_indexer
        from saguaro.storage.native_vector_store import native_vector_store_perf_counters

        runtime = get_native_indexer()
        capability_report = runtime.capability_report()
        abi = dict(capability_report.get("native_indexer") or {})
        trie_ops = dict(capability_report.get("trie_ops") or {})
        parallel_runtime = dict(capability_report.get("parallel_runtime") or {})
        return {
            "status": "ready" if abi.get("ok") else "degraded",
            "required_backend": "native_cpp",
            "required_parallelism": "openmp",
            "required_simd": "avx2",
            "requirements": {
                "openmp_required": True,
                "avx2_required": True,
                "satisfied": bool(abi.get("ok", False)),
            },
            "parallel_runtime": parallel_runtime,
            "simd": {
                "baseline": str(abi.get("isa_baseline") or "scalar"),
                "avx2_compiled": bool(abi.get("avx2_enabled", False)),
                "fma_compiled": bool(abi.get("fma_enabled", False)),
            },
            "vector_store": native_vector_store_perf_counters(),
            "abi": abi,
            "trie_ops": trie_ops,
            "ops": dict(capability_report.get("ops") or {}),
            "manifest": dict(capability_report.get("manifest") or {}),
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "required_backend": "native_cpp",
            "required_parallelism": "openmp",
            "required_simd": "avx2",
            "requirements": {
                "openmp_required": True,
                "avx2_required": True,
                "satisfied": False,
            },
            "parallel_runtime": {
                "compiled": False,
                "default_threads": 0,
                "max_threads": 0,
            },
            "simd": {
                "baseline": "scalar",
                "avx2_compiled": False,
                "fma_compiled": False,
            },
            "vector_store": {"available": 0, "error": str(exc)},
            "abi": {"ok": False, "reason": str(exc)},
        }


class HealthDashboard:
    """Provide HealthDashboard support."""
    def __init__(self, saguaro_dir: str, repo_path: str | None = None) -> None:
        """Initialize the instance."""
        self.saguaro_dir = saguaro_dir
        self.repo_path = os.path.abspath(repo_path or os.path.dirname(saguaro_dir))
        self.config_path = os.path.join(saguaro_dir, "config.yaml")
        # We assume vectors stored in 'vectors' subdir for now or check config

    def generate_report(self) -> dict[str, Any]:
        """Generates health metrics for the SAGUARO Index."""
        report = {}

        # 1. Freshness & Storage
        vectors_dir = os.path.join(self.saguaro_dir, "vectors")
        # Support both legacy index.pkl and new vectors.bin
        store_path = os.path.join(vectors_dir, "vectors.bin")
        if not os.path.exists(store_path):
            store_path = os.path.join(vectors_dir, "index.pkl")

        if os.path.exists(store_path):
            mtime = os.path.getmtime(store_path)
            report["freshness"] = {
                "last_update_ts": mtime,
                "last_update_fmt": time.ctime(mtime),
                "age_seconds": time.time() - mtime,
            }

            idx_size_mb = os.path.getsize(store_path) / (1024 * 1024)
            meta_path = os.path.join(vectors_dir, "metadata.json")
            meta_size_mb = (
                os.path.getsize(meta_path) / (1024 * 1024)
                if os.path.exists(meta_path)
                else 0
            )

            report["storage"] = {
                "vector_index_mb": round(idx_size_mb, 2),
                "metadata_mb": round(meta_size_mb, 2),
                "total_mb": round(idx_size_mb + meta_size_mb, 2),
            }
        else:
            report["freshness"] = {"status": "not_indexed"}

        # 2. Config & Performance
        if os.path.exists(self.config_path):
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
                report["config"] = {
                    "active_dim": config.get("active_dim", "Unknown"),
                    "total_dim": config.get("total_dim", "Unknown"),
                    "dark_space_ratio": config.get("dark_space_ratio", "Unknown"),
                }

                # Memory metrics from last index
                last_idx = config.get("last_index", {})
                if last_idx:
                    report["performance"] = {
                        "peak_memory_mb": last_idx.get("peak_memory_mb"),
                        "indexed_files": last_idx.get("files"),
                        "indexed_entities": last_idx.get("entities"),
                    }

        # 3. Governance (Verification Coverage)
        tracking_file = os.path.join(self.saguaro_dir, "tracking.json")
        if os.path.exists(tracking_file):
            from saguaro.indexing.tracker import IndexTracker

            try:
                tracker = IndexTracker(tracking_file)
                total_files = len(tracker.state)
                verified_files = sum(
                    1 for entry in tracker.state.values() if entry.get("verified", False)
                )

                report["governance"] = {
                    "status": "ready",
                    "total_tracked_files": total_files,
                    "verified_files": verified_files,
                    "coverage_percent": (
                        round((verified_files / total_files * 100), 1)
                        if total_files > 0
                        else 0
                    ),
                }
            except SaguaroStateCorruptionError as exc:
                report["governance"] = {
                    "status": "corrupt",
                    "error": str(exc),
                    "total_tracked_files": 0,
                    "verified_files": 0,
                    "coverage_percent": 0.0,
                }

        report["runtime"] = {
            "repo_path": self.repo_path,
            "cpu_count": os.cpu_count() or 1,
            "execution_target": "cpu",
            "index_mode": "cpu-first-local",
            "embedding_backend": self._stored_backend(),
        }
        report["corpus"] = self._timed_step("corpus_manifest_report", self._corpus_report)
        report["coverage"] = self._timed_step("coverage_report", self._coverage_report)
        report["graph"] = self._timed_step("graph_report", self._graph_report)
        report["graph_confidence"] = self._timed_step(
            "graph_confidence_report",
            self._graph_confidence_report,
        )
        report["symbol_truth"] = self._timed_step(
            "symbol_truth_report",
            self._symbol_truth_report,
        )
        report["runtime_symbols"] = self._timed_step(
            "runtime_symbol_report",
            self._runtime_symbol_report,
        )
        report["build_topology"] = self._timed_step(
            "build_topology_report",
            self._build_topology_report,
        )
        report["coverage_vector"] = self._timed_step(
            "coverage_vector_report",
            lambda: self._coverage_vector_report(
                coverage=report["coverage"],
                graph=report["graph"],
                runtime_symbols=report["runtime_symbols"],
                build_topology=report["build_topology"],
            ),
        )
        report["topology"] = self._timed_step(
            "topology_health",
            lambda: ArchitectureAnalyzer(self.repo_path).health(),
        )
        report["layout"] = {
            "policy_path": os.path.join(self.repo_path, "standards", "REPO_LAYOUT.yaml"),
            "misplaced_files": int(report["topology"].get("misplaced_files", 0) or 0),
            "illegal_zone_crossings": int(
                report["topology"].get("illegal_zone_crossings", 0) or 0
            ),
        }
        report["degraded_mode"] = self._timed_step(
            "degraded_mode_report",
            self._degraded_mode_report,
        )
        report["native_compute"] = self._timed_step(
            "native_compute_report",
            collect_native_compute_report,
        )
        report["native_capabilities"] = report["native_compute"]
        report["state_journal"] = self._timed_step(
            "state_journal_report",
            self._state_journal_report,
        )
        report["snapshots"] = self._timed_step(
            "snapshots_report",
            self._snapshots_report,
        )
        report["task_arbiter"] = self._timed_step(
            "task_arbiter_report",
            self._task_arbiter_report,
        )
        report["integrity"] = self._timed_step(
            "integrity_report",
            self._integrity_report,
        )
        report["locks"] = self._timed_step(
            "lock_status",
            lambda: RepoLockManager(self.saguaro_dir).status(),
        )

        return report

    def _timed_step(self, stage: str, fn: Any) -> Any:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        threshold = float(os.getenv("SAGUARO_HEALTH_SLOW_STAGE_SECONDS", "2.0") or 2.0)
        if logger.isEnabledFor(logging.INFO) or elapsed >= threshold:
            message = f"HealthDashboard stage={stage} elapsed={elapsed:.3f}s"
            if elapsed >= threshold:
                logger.warning(message)
            else:
                logger.info(message)
        return result

    def print_dashboard(self) -> None:
        """Handle print dashboard."""
        report = self.generate_report()

        print("\n" + "═" * 50)
        print(" SAGUARO Enterprise Q-COS Health Dashboard ".center(50, "═"))
        print("═" * 50)

        # 1. System Health Status
        fresh = report.get("freshness", {})
        if "last_update_fmt" in fresh:
            age = fresh["age_seconds"]
            if age < 3600:
                status, icon = "Synchronized", "🟢"
            elif age < 86400:
                status, icon = "Stale (>1h)", "🟡"
            else:
                status, icon = "Diverged (>24h)", "🔴"

            print(f"\n{icon} System Status: {status}")
            print(f"  Last Pulse: {fresh['last_update_fmt']}")
        else:
            print("\n⚪ System Status: PENDING (Initial Index Required)")

        # 2. Governance & Compliance
        gov = report.get("governance", {})
        if gov:
            print("\n🛡️ Governance Audit")
            icon = "✅" if gov["coverage_percent"] > 90 else "⚠️"
            print(f"  Verification Coverage: {gov['coverage_percent']}% {icon}")
            print(
                f"  Verified Files:        {gov['verified_files']} / {gov['total_tracked_files']}"
            )

        # 3. Memory & Performance
        perf = report.get("performance", {})
        if perf:
            print("\n⚡ Resource Intelligence")
            print(f"  Peak Indexing RAM: {perf['peak_memory_mb']:.1f} MB")
            print(
                f"  Indexed Assets:    {perf['indexed_entities']} entities across {perf['indexed_files']} files"
            )

        # 4. Storage Footprint
        store = report.get("storage", {})
        if store:
            print("\n💾 Neural Storage")
            print(f"  Holographic Bundle: {store['vector_index_mb']} MB")
            print(f"  Semantic Metadata:  {store['metadata_mb']} MB")
            print(f"  Total Capacity:     {store['total_mb']} MB")

        # 5. Holographic Config
        conf = report.get("config", {})
        if conf:
            print("\n🧬 Quantum Configuration")
            print(
                f"  Manifold Dimension: {conf.get('active_dim')} (Active) / {conf.get('total_dim')} (Total)"
            )
            print(f"  Dark Space Buffer:  {conf.get('dark_space_ratio') * 100:.0f}%")

        coverage = report.get("coverage", {})
        if coverage:
            print("\n🧠 Parser Coverage")
            print(
                f"  AST Coverage:       {coverage.get('coverage_percent', 0.0):.1f}%"
            )
            print(
                f"  Supported AST Files:{coverage.get('ast_supported_files', 0)} / {coverage.get('total_files', 0)}"
            )

        native_compute = report.get("native_compute", {})
        if native_compute:
            print("\n🛠️ Native Compute Fabric")
            print(f"  Status:             {native_compute.get('status', 'unknown')}")
            print(
                f"  Parallel Runtime:   {native_compute.get('required_parallelism', 'unknown')}"
            )
            print(
                f"  SIMD Baseline:      {((native_compute.get('simd') or {}).get('baseline')) or native_compute.get('required_simd', 'unknown')}"
            )

        graph = report.get("graph", {})
        if graph.get("status") == "ready":
            print("\n🕸️ Repository Graph")
            print(
                f"  Graph Coverage:     {graph.get('graph_coverage_percent', 0.0):.1f}%"
            )
            print(
                f"  Nodes / Edges:      {graph.get('nodes', 0)} / {graph.get('edges', 0)}"
            )
            confidence = report.get("graph_confidence", {})
            deficits = ", ".join(list(confidence.get("deficits") or [])[:4]) or "none"
            print(
                f"  Confidence Score:   {confidence.get('score', 0.0):.2f} (deficits: {deficits})"
            )

        symbol_truth = report.get("symbol_truth", {})
        if symbol_truth:
            print("\n🧭 Symbol Truth")
            print(
                f"  Canonical Modules:  {symbol_truth.get('canonical_module_count', 0)}"
            )
            print(
                f"  Shadowed Modules:   {symbol_truth.get('shadowed_module_count', 0)}"
            )
        runtime_symbols = report.get("runtime_symbols", {})
        if runtime_symbols:
            print("\n🔌 Runtime Symbols")
            print(
                "  Coverage:           "
                f"{runtime_symbols.get('coverage_percent', 0.0):.1f}%"
            )
            print(
                "  Matched / Referenced:"
                f" {runtime_symbols.get('matched_symbol_count', 0)} / "
                f"{runtime_symbols.get('referenced_symbol_count', 0)}"
            )

        build = report.get("build_topology", {})
        coverage = build.get("source_coverage", {})
        if build:
            print("\n🏗️ Build Topology")
            print(f"  Targets:            {build.get('target_count', 0)}")
            print(
                "  Structured Inputs:  "
                f"{(build.get('structured_inputs') or {}).get('compile_databases', 0)} compile DB, "
                f"{(build.get('structured_inputs') or {}).get('cmake_file_api_replies', 0)} file API"
            )
            print(
                "  Source Coverage:    "
                f"{coverage.get('owned_sources', 0)} / {coverage.get('compiled_sources', 0)} "
                f"({coverage.get('coverage_percent', 0.0)}%)"
            )

        print("\n" + "═" * 50)

    def _stored_backend(self) -> str:
        schema_path = os.path.join(self.saguaro_dir, "index_schema.json")
        if not os.path.exists(schema_path):
            return "unknown"
        try:
            with open(schema_path, encoding="utf-8") as f:
                return str((json.load(f) or {}).get("backend", "unknown"))
        except Exception:
            return "unknown"

    def _coverage_report(self) -> dict[str, Any]:
        try:
            stats = CoverageReporter(self.repo_path).generate_report()
            total = int(stats.get("total_files", 0) or 0)
            dependency_quality = int(
                stats.get("dependency_quality_supported_files", 0) or 0
            )
            stats["coverage_percent"] = (
                round((dependency_quality / total) * 100, 1) if total else 0.0
            )
            return stats
        except Exception:
            return {"coverage_percent": 0.0, "total_files": 0, "ast_supported_files": 0}

    def _corpus_report(self) -> dict[str, Any]:
        manifest = build_corpus_manifest(self.repo_path)
        return {
            "source": manifest.source,
            "candidate_files": manifest.candidate_count,
            "indexed_candidates": len(manifest.files),
            "excluded_files": manifest.excluded_count,
            "excluded_roots": manifest.excluded_roots,
        }

    def _coverage_vector_report(
        self,
        *,
        coverage: dict[str, Any],
        graph: dict[str, Any],
        runtime_symbols: dict[str, Any],
        build_topology: dict[str, Any],
    ) -> dict[str, Any]:
        total = max(int(coverage.get("total_files", 0) or 0), 1)
        structural = int(coverage.get("structural_supported_files", 0) or 0)
        ast_supported = int(coverage.get("ast_supported_files", 0) or 0)
        dependency_quality = int(
            coverage.get("dependency_quality_supported_files", 0) or 0
        )
        header_total = int((coverage.get("languages", {}) or {}).get("C/C++ Header", 0) or 0)
        graph_files = int(graph.get("files", 0) or 0)
        header_graph_files = self._header_files_in_graph() if graph.get("status") == "ready" else 0
        entrypoints = EntryPointDetector(self.repo_path).detect()
        resolved_entrypoints = self._resolved_entrypoints()
        ffi_patterns, bridge_edges = self._graph_bridge_stats()
        return {
            "parser_eligibility_coverage_pct": round((structural / total) * 100.0, 1),
            "parser_success_coverage_pct": round((dependency_quality / total) * 100.0, 1),
            "parser_ast_coverage_pct": round((ast_supported / total) * 100.0, 1),
            "dependency_quality_coverage_pct": round((dependency_quality / total) * 100.0, 1),
            "header_declaration_coverage_pct": (
                round((header_graph_files / max(header_total, 1)) * 100.0, 1)
                if header_total
                else 100.0
            ),
            "graph_node_coverage_pct": round((graph_files / total) * 100.0, 1),
            "graph_edge_coverage_pct": float(graph.get("graph_coverage_percent", 0.0) or 0.0),
            "entrypoint_resolution_coverage_pct": (
                round((resolved_entrypoints / max(len(entrypoints), 1)) * 100.0, 1)
                if entrypoints
                else 100.0
            ),
            "ffi_boundary_coverage_pct": (
                round((bridge_edges / max(ffi_patterns, 1)) * 100.0, 1)
                if ffi_patterns
                else 100.0
            ),
            "build_target_coverage_pct": float(
                (build_topology.get("source_coverage") or {}).get("coverage_percent", 0.0)
                or 0.0
            ),
            "runtime_symbol_coverage_pct": float(
                runtime_symbols.get("coverage_percent", 0.0) or 0.0
            ),
        }

    def _runtime_symbol_report(self) -> dict[str, Any]:
        try:
            return RuntimeSymbolResolver(self.repo_path).build_symbol_manifest(
                persist=True
            )
        except Exception:
            return {
                "referenced_symbol_count": 0,
                "exported_symbol_count": 0,
                "matched_symbol_count": 0,
                "coverage_percent": 0.0,
                "referenced_symbols": [],
                "exported_symbols": [],
                "matched_symbols": [],
                "unresolved_symbols": [],
            }

    def _degraded_mode_report(self) -> dict[str, Any]:
        backend = self._stored_backend()
        active = backend in {"NumPyBackend", "unknown"}
        return {
            "active": active,
            "backend": backend,
            "reason": (
                "non_native_backend"
                if backend == "NumPyBackend"
                else ("missing_backend_metadata" if backend == "unknown" else "")
            ),
            "fallback_count": 1 if active else 0,
        }

    def _native_capability_report(self) -> dict[str, Any]:
        return collect_native_compute_report()

    def _state_journal_report(self) -> dict[str, Any]:
        events_path = os.path.join(self.saguaro_dir, "state", "events.jsonl")
        count = 0
        if os.path.exists(events_path):
            try:
                with open(events_path, encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            except Exception:
                count = 0
        return {
            "path": events_path,
            "count": count,
        }

    def _snapshots_report(self) -> dict[str, Any]:
        snapshots_path = os.path.join(self.saguaro_dir, "state", "snapshots.jsonl")
        count = 0
        if os.path.exists(snapshots_path):
            try:
                with open(snapshots_path, encoding="utf-8") as handle:
                    count = sum(1 for line in handle if line.strip())
            except Exception:
                count = 0
        return {
            "path": snapshots_path,
            "count": count,
        }

    def _task_arbiter_report(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "active_tasks": 0,
            "queued_tasks": 0,
        }

    def _header_files_in_graph(self) -> int:
        data = self._load_graph_payload()
        if not data:
            return 0
        files = data.get("files", {}) if isinstance(data, dict) else {}
        return sum(
            1
            for rel_path in files.keys()
            if str(rel_path).endswith((".h", ".hpp", ".hh", ".hxx"))
        )

    def _resolved_entrypoints(self) -> int:
        graph = self._load_graph_payload()
        if not graph:
            return 0
        files = graph.get("files", {}) if isinstance(graph, dict) else {}
        count = 0
        for entry in EntryPointDetector(self.repo_path).detect():
            rel_file = os.path.relpath(
                os.path.abspath(str(entry.get("file") or "")),
                self.repo_path,
            ).replace("\\", "/")
            if rel_file in files:
                count += 1
        return count

    def _graph_bridge_stats(self) -> tuple[int, int]:
        graph = self._load_graph_payload()
        if not graph:
            return 0, 0
        stats = self._graph_stats(graph)
        return (
            int(stats.get("ffi_patterns", 0) or 0),
            int(stats.get("bridge_edges", 0) or 0),
        )

    def _graph_report(self) -> dict[str, Any]:
        path = self._graph_artifact_path()
        if not path:
            return {"status": "missing"}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"status": "unreadable"}
        payload = data.get("graph") if isinstance(data, dict) and "graph" in data else data
        stats = self._graph_stats(payload)
        return {
            "status": "ready",
            "path": path,
            "generated_at": (
                payload.get("generated_at") if isinstance(payload, dict) else None
            ),
            "files": stats.get("files", 0),
            "nodes": stats.get("nodes", 0),
            "edges": stats.get("edges", 0),
            "graph_coverage_percent": stats.get("graph_coverage_percent", 0.0),
            "missing_edge_classes": self._missing_edge_classes(stats),
        }

    def _graph_confidence_report(self) -> dict[str, Any]:
        graph = self._load_graph_payload()
        stats = self._graph_stats(graph)
        coverage_pct = float(stats.get("graph_coverage_percent", 0.0) or 0.0)
        edge_classes = {
            "cfg": int(stats.get("cfg_edges", 0) or 0),
            "dfg": int(stats.get("dfg_edges", 0) or 0),
            "call": int(stats.get("call_edges", 0) or 0),
            "ffi_bridge": int(stats.get("bridge_edges", 0) or 0),
            "disparate_relation": int(stats.get("disparate_relation_edges", 0) or 0),
        }
        present_classes = sum(1 for count in edge_classes.values() if count > 0)
        deficits = self._missing_edge_classes(stats)
        score = 0.0
        if edge_classes:
            score = round(
                min(1.0, (coverage_pct / 100.0) * 0.6 + (present_classes / len(edge_classes)) * 0.4),
                3,
            )
        return {
            "score": score,
            "coverage_percent": coverage_pct,
            "edge_classes": edge_classes,
            "deficits": deficits,
        }

    def _build_topology_report(self) -> dict[str, Any]:
        try:
            return BuildGraphIngestor(self.repo_path).ingest()
        except Exception as exc:
            return {
                "root": self.repo_path,
                "target_count": 0,
                "targets": {},
                "structured_inputs": {},
                "source_coverage": {"compiled_sources": 0, "owned_sources": 0, "coverage_percent": 0.0},
                "status": "error",
                "error": str(exc),
            }

    def _graph_artifact_candidates(self) -> tuple[str, ...]:
        return (
            os.path.join(self.saguaro_dir, "graph", "graph.json"),
            os.path.join(self.saguaro_dir, "graph", "code_graph.json"),
            os.path.join(self.saguaro_dir, "code_graph.json"),
        )

    def _graph_artifact_path(self) -> str | None:
        for candidate in self._graph_artifact_candidates():
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_graph_payload(self) -> dict[str, Any]:
        path = self._graph_artifact_path()
        if not path:
            return {}
        try:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle) or {}
        except Exception:
            return {}
        graph = raw.get("graph") if isinstance(raw, dict) and "graph" in raw else raw
        return graph if isinstance(graph, dict) else {}

    @staticmethod
    def _graph_stats(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        stats = dict(payload.get("stats") or {})
        nodes = payload.get("nodes") if isinstance(payload.get("nodes"), dict) else {}
        files = payload.get("files") if isinstance(payload.get("files"), dict) else {}
        ffi_patterns = (
            payload.get("ffi_patterns")
            if isinstance(payload.get("ffi_patterns"), dict)
            else {}
        )
        raw_edges = payload.get("edges")
        if isinstance(raw_edges, dict):
            edge_values = list(raw_edges.values())
            edge_count = len(raw_edges)
        elif isinstance(raw_edges, list):
            edge_values = [edge for edge in raw_edges if isinstance(edge, dict)]
            edge_count = len(edge_values)
        else:
            edge_values = []
            edge_count = 0

        stats["files"] = len(files)
        stats["nodes"] = len(nodes)
        stats["edges"] = edge_count
        stats["cfg_edges"] = sum(
            1
            for edge in edge_values
            if str(edge.get("relation") or "").startswith("cfg_")
        )
        stats["dfg_edges"] = sum(
            1
            for edge in edge_values
            if str(edge.get("relation") or "").startswith("dfg_")
        )
        stats["call_edges"] = sum(
            1 for edge in edge_values if str(edge.get("relation") or "") == "calls"
        )
        stats["ffi_patterns"] = len(ffi_patterns)
        stats["bridge_edges"] = sum(
            1 for edge in edge_values if str(edge.get("relation") or "") == "ffi_bridge"
        )
        stats["disparate_relation_edges"] = sum(
            1
            for edge in edge_values
            if str(edge.get("relation") or "")
            in {
                "analogous_to",
                "subsystem_analogue",
                "evaluation_analogue",
                "adaptation_candidate",
                "native_upgrade_path",
                "port_program_candidate",
            }
        )
        stats.setdefault("graph_coverage_percent", 0.0)
        return stats

    def _integrity_report(self) -> dict[str, Any]:
        try:
            manifest = load_manifest(self.saguaro_dir)
            validation = validate_manifest(self.saguaro_dir, manifest)
            return {
                "manifest_generation_id": manifest.get("generation_id"),
                "status": validation.get("status", "missing"),
                "mismatches": validation.get("mismatches", []),
                "summary": manifest.get("summary", {}),
            }
        except Exception as exc:
            return {
                "manifest_generation_id": None,
                "status": "corrupt",
                "mismatches": [str(exc)],
                "summary": {},
            }

    @staticmethod
    def _missing_edge_classes(stats: dict[str, Any]) -> list[str]:
        mapping = {
            "cfg_edges": "cfg",
            "dfg_edges": "dfg",
            "call_edges": "call",
            "bridge_edges": "ffi_bridge",
            "disparate_relation_edges": "disparate_relation",
        }
        return [
            label
            for key, label in mapping.items()
            if int(stats.get(key, 0) or 0) <= 0
        ]

    def _symbol_truth_report(self) -> dict[str, Any]:
        owners: dict[str, list[str]] = {}
        for root, dirs, files in os.walk(self.repo_path):
            rel_root = os.path.relpath(root, self.repo_path).replace("\\", "/")
            if any(
                rel_root == prefix.rstrip("/")
                or rel_root.startswith(prefix)
                for prefix in (
                    ".git/",
                    ".saguaro/",
                    "repo_analysis/",
                    "venv/",
                    "__pycache__/",
                    ".venv/",
                )
            ):
                dirs[:] = []
                continue
            dirs[:] = [
                directory
                for directory in dirs
                if directory
                not in {
                    ".git",
                    ".saguaro",
                    "repo_analysis",
                    "venv",
                    ".venv",
                    "__pycache__",
                }
            ]
            for file_name in files:
                if not file_name.endswith(".py"):
                    continue
                rel_path = os.path.relpath(
                    os.path.join(root, file_name),
                    self.repo_path,
                ).replace("\\", "/")
                module_name = self._symbol_truth_module_name(rel_path)
                if not module_name:
                    continue
                owners.setdefault(module_name, []).append(rel_path)

        registry: dict[str, dict[str, Any]] = {}
        shadowed = 0
        for module_name, paths in sorted(owners.items()):
            ranked = sorted(set(paths), key=self._symbol_truth_rank)
            canonical = ranked[0]
            duplicates = ranked[1:]
            if duplicates:
                shadowed += 1
            registry[module_name] = {
                "canonical_path": canonical,
                "shadowed_duplicates": duplicates,
                "active_vs_vendored_ranking": ranked,
            }

        artifact_path = os.path.join(self.saguaro_dir, "symbol_truth.json")
        with open(artifact_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "generated_at": time.time(),
                    "repo_path": self.repo_path,
                    "modules": registry,
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        return {
            "artifact_path": artifact_path,
            "canonical_module_count": len(registry),
            "shadowed_module_count": shadowed,
        }

    @staticmethod
    def _symbol_truth_rank(rel_path: str) -> tuple[int, str]:
        if rel_path.startswith("saguaro/"):
            return (0, rel_path)
        if rel_path.startswith("Saguaro/saguaro/"):
            return (2, rel_path)
        if rel_path.startswith(".anvil/toolchains/"):
            return (3, rel_path)
        return (1, rel_path)

    @staticmethod
    def _symbol_truth_module_name(rel_path: str) -> str:
        normalized = rel_path.replace("\\", "/")
        if normalized.endswith("/__init__.py"):
            normalized = normalized[: -len("/__init__.py")]
        elif normalized.endswith(".py"):
            normalized = normalized[:-3]
        if normalized.startswith("Saguaro/saguaro/"):
            normalized = normalized[len("Saguaro/") :]
        return normalized.replace("/", ".").strip(".")
