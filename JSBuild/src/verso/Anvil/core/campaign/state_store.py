"""SQLite-backed state store for long-running autonomy campaigns."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence


class CampaignStateStore:
    """Persistent campaign database with typed helpers and generic row access."""

    _TABLES: Dict[str, str] = {
        "campaigns": """
            CREATE TABLE IF NOT EXISTS campaigns (
                campaign_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                objective TEXT,
                runtime_state TEXT NOT NULL,
                campaign_path TEXT,
                root_dir TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "campaign_state_transitions": """
            CREATE TABLE IF NOT EXISTS campaign_state_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                from_state TEXT,
                to_state TEXT NOT NULL,
                reason TEXT,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "campaign_repos": """
            CREATE TABLE IF NOT EXISTS campaign_repos (
                repo_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                name TEXT NOT NULL,
                origin TEXT,
                revision TEXT,
                local_path TEXT NOT NULL,
                role TEXT NOT NULL,
                write_policy TEXT NOT NULL,
                topic_tags_json TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                ingestion_status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "repo_ingestion_reports": """
            CREATE TABLE IF NOT EXISTS repo_ingestion_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                report_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "artifacts": """
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                family TEXT NOT NULL,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                canonical_path TEXT NOT NULL,
                rendered_path TEXT,
                approval_state TEXT NOT NULL,
                blocking INTEGER NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "artifact_versions": """
            CREATE TABLE IF NOT EXISTS artifact_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                canonical_path TEXT NOT NULL,
                rendered_path TEXT,
                content_hash TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "artifact_approvals": """
            CREATE TABLE IF NOT EXISTS artifact_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artifact_id TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                decision TEXT NOT NULL,
                notes TEXT,
                created_at REAL NOT NULL
            )
        """,
        "research_sources": """
            CREATE TABLE IF NOT EXISTS research_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_context TEXT NOT NULL DEFAULT '',
                source_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                origin_url TEXT,
                trust_level TEXT NOT NULL,
                digest TEXT,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "research_documents": """
            CREATE TABLE IF NOT EXISTS research_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_context TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL,
                source_id TEXT,
                title TEXT NOT NULL,
                path TEXT,
                digest TEXT,
                normalized_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "research_claims": """
            CREATE TABLE IF NOT EXISTS research_claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_context TEXT NOT NULL DEFAULT '',
                claim_id TEXT NOT NULL,
                document_id TEXT,
                topic TEXT,
                summary TEXT NOT NULL,
                confidence REAL NOT NULL,
                complexity_score REAL NOT NULL DEFAULT 0.0,
                topic_hierarchy TEXT NOT NULL DEFAULT '',
                applicability_score REAL NOT NULL DEFAULT 0.0,
                evidence_type TEXT NOT NULL DEFAULT 'implementation',
                provenance_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "crawl_queue": """
            CREATE TABLE IF NOT EXISTS crawl_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                frontier_id TEXT NOT NULL,
                url TEXT NOT NULL,
                topic TEXT,
                status TEXT NOT NULL,
                priority REAL NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "crawl_edges": """
            CREATE TABLE IF NOT EXISTS crawl_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                from_frontier_id TEXT NOT NULL,
                to_frontier_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "topic_clusters": """
            CREATE TABLE IF NOT EXISTS topic_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                label TEXT NOT NULL,
                members_json TEXT NOT NULL,
                score REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "repo_findings": """
            CREATE TABLE IF NOT EXISTS repo_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                summary TEXT NOT NULL,
                severity TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "hypotheses": """
            CREATE TABLE IF NOT EXISTS hypotheses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                statement TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "experiments": """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                name TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "experiment_runs": """
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                command_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "experiment_results": """
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                result_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                verdict TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "architecture_questions": """
            CREATE TABLE IF NOT EXISTS architecture_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "architecture_answers": """
            CREATE TABLE IF NOT EXISTS architecture_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                question_id TEXT NOT NULL,
                answer_mode TEXT NOT NULL,
                answer_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "feature_catalog": """
            CREATE TABLE IF NOT EXISTS feature_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                feature_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "feature_selections": """
            CREATE TABLE IF NOT EXISTS feature_selections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                feature_id TEXT NOT NULL,
                selection_state TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "roadmap_items": """
            CREATE TABLE IF NOT EXISTS roadmap_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                title TEXT NOT NULL,
                item_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "roadmap_dependencies": """
            CREATE TABLE IF NOT EXISTS roadmap_dependencies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                depends_on_item_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "task_packets": """
            CREATE TABLE IF NOT EXISTS task_packets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                task_packet_id TEXT NOT NULL,
                phase_id TEXT,
                role TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "task_runs": """
            CREATE TABLE IF NOT EXISTS task_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                task_packet_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "task_telemetry": """
            CREATE TABLE IF NOT EXISTS task_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                task_packet_id TEXT NOT NULL,
                telemetry_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "benchmark_runs": """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                benchmark_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "audit_runs": """
            CREATE TABLE IF NOT EXISTS audit_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                audit_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "audit_findings": """
            CREATE TABLE IF NOT EXISTS audit_findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                audit_id TEXT NOT NULL,
                finding_id TEXT NOT NULL,
                severity TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "whitepapers": """
            CREATE TABLE IF NOT EXISTS whitepapers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                whitepaper_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "completion_checks": """
            CREATE TABLE IF NOT EXISTS completion_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                check_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "knowledge_exports": """
            CREATE TABLE IF NOT EXISTS knowledge_exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                export_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "repo_snapshots": """
            CREATE TABLE IF NOT EXISTS repo_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                snapshot_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "analysis_packs": """
            CREATE TABLE IF NOT EXISTS analysis_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                pack_id TEXT NOT NULL,
                repo_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "source_chunks": """
            CREATE TABLE IF NOT EXISTS source_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_context TEXT NOT NULL DEFAULT '',
                chunk_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "decision_records": """
            CREATE TABLE IF NOT EXISTS decision_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                decision_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "unknown_resolution_queue": """
            CREATE TABLE IF NOT EXISTS unknown_resolution_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                unknown_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "specialist_assignments": """
            CREATE TABLE IF NOT EXISTS specialist_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                assignment_id TEXT NOT NULL,
                specialist_role TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "phase_packets": """
            CREATE TABLE IF NOT EXISTS phase_packets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "phase_artifact_links": """
            CREATE TABLE IF NOT EXISTS phase_artifact_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "phase_artifacts": """
            CREATE TABLE IF NOT EXISTS phase_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                phase_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                path TEXT NOT NULL,
                status TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """,
        "convergence_checkpoints": """
            CREATE TABLE IF NOT EXISTS convergence_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                loop_name TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                converged INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "experiment_telemetry": """
            CREATE TABLE IF NOT EXISTS experiment_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                unit TEXT,
                created_at REAL NOT NULL
            )
        """,
        "usage_traces": """
            CREATE TABLE IF NOT EXISTS usage_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                repo_context TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "tooling_tasks": """
            CREATE TABLE IF NOT EXISTS tooling_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                tooling_task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "tooling_runs": """
            CREATE TABLE IF NOT EXISTS tooling_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                tooling_task_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "telemetry_spans": """
            CREATE TABLE IF NOT EXISTS telemetry_spans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                span_name TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "resource_profiles": """
            CREATE TABLE IF NOT EXISTS resource_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "hardware_profiles": """
            CREATE TABLE IF NOT EXISTS hardware_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "market_feature_candidates": """
            CREATE TABLE IF NOT EXISTS market_feature_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "soak_runs": """
            CREATE TABLE IF NOT EXISTS soak_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                soak_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "reliability_events": """
            CREATE TABLE IF NOT EXISTS reliability_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "loop_runs": """
            CREATE TABLE IF NOT EXISTS loop_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                loop_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_objects": """
            CREATE TABLE IF NOT EXISTS memory_objects (
                memory_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                memory_kind TEXT NOT NULL,
                workspace_id TEXT NOT NULL DEFAULT '',
                repo_context TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                source_system TEXT NOT NULL DEFAULT '',
                summary_text TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                canonical_hash TEXT NOT NULL,
                importance_score REAL NOT NULL DEFAULT 0.5,
                confidence_score REAL NOT NULL DEFAULT 0.5,
                retention_class TEXT NOT NULL DEFAULT 'durable',
                sensitivity_class TEXT NOT NULL DEFAULT 'internal',
                lifecycle_state TEXT NOT NULL DEFAULT 'active',
                schema_version TEXT NOT NULL DEFAULT 'almf.v1',
                created_at REAL NOT NULL,
                observed_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                hypothesis_id TEXT NOT NULL DEFAULT '',
                document_id TEXT NOT NULL DEFAULT '',
                claim_id TEXT NOT NULL DEFAULT '',
                experiment_id TEXT NOT NULL DEFAULT '',
                artifact_id TEXT NOT NULL DEFAULT '',
                task_packet_id TEXT NOT NULL DEFAULT '',
                lane_id TEXT NOT NULL DEFAULT '',
                operator_id TEXT NOT NULL DEFAULT '',
                thread_id TEXT NOT NULL DEFAULT '',
                model_family TEXT NOT NULL DEFAULT '',
                model_revision TEXT NOT NULL DEFAULT ''
            )
        """,
        "memory_aliases": """
            CREATE TABLE IF NOT EXISTS memory_aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                source_table TEXT NOT NULL,
                source_id TEXT NOT NULL,
                campaign_id TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_embeddings": """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                embedding_family TEXT NOT NULL,
                embedding_version TEXT NOT NULL,
                dim INTEGER NOT NULL,
                vector_uri TEXT NOT NULL,
                norm REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_multivectors": """
            CREATE TABLE IF NOT EXISTS memory_multivectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                embedding_family TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                vector_uri TEXT NOT NULL,
                indexing_mode TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_hd_bundles": """
            CREATE TABLE IF NOT EXISTS memory_hd_bundles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                bundle_family TEXT NOT NULL,
                bundle_version TEXT NOT NULL,
                bundle_uri TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_edges": """
            CREATE TABLE IF NOT EXISTS memory_edges (
                edge_id TEXT PRIMARY KEY,
                src_memory_id TEXT NOT NULL,
                dst_memory_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL NOT NULL,
                valid_from REAL NOT NULL,
                valid_to REAL,
                recorded_at REAL NOT NULL,
                evidence_json TEXT NOT NULL
            )
        """,
        "latent_packages": """
            CREATE TABLE IF NOT EXISTS latent_packages (
                latent_package_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                branch_id TEXT NOT NULL DEFAULT '',
                model_family TEXT NOT NULL DEFAULT '',
                model_revision TEXT NOT NULL DEFAULT '',
                tokenizer_hash TEXT NOT NULL DEFAULT '',
                adapter_hash TEXT NOT NULL DEFAULT '',
                prompt_protocol_hash TEXT NOT NULL DEFAULT '',
                hidden_dim INTEGER NOT NULL DEFAULT 0,
                qsg_runtime_version TEXT NOT NULL DEFAULT '',
                rope_config_hash TEXT NOT NULL DEFAULT '',
                quantization_profile TEXT NOT NULL DEFAULT '',
                capture_stage TEXT NOT NULL DEFAULT '',
                tensor_format TEXT NOT NULL DEFAULT 'npy',
                tensor_uri TEXT NOT NULL DEFAULT '',
                summary_text TEXT NOT NULL DEFAULT '',
                compatibility_json TEXT NOT NULL DEFAULT '{}',
                supporting_memory_ids_json TEXT NOT NULL DEFAULT '[]',
                creation_reason TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL
            )
        """,
        "memory_reads": """
            CREATE TABLE IF NOT EXISTS memory_reads (
                read_id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                query_kind TEXT NOT NULL,
                query_text TEXT NOT NULL,
                planner_mode TEXT NOT NULL,
                result_memory_ids_json TEXT NOT NULL,
                latency_ms REAL NOT NULL,
                created_at REAL NOT NULL
            )
        """,
        "memory_feedback": """
            CREATE TABLE IF NOT EXISTS memory_feedback (
                feedback_id TEXT PRIMARY KEY,
                read_id TEXT NOT NULL,
                consumer_system TEXT NOT NULL,
                usefulness_score REAL NOT NULL,
                grounding_score REAL NOT NULL,
                citation_score REAL NOT NULL,
                token_savings_estimate REAL NOT NULL,
                outcome_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """,
    }
    _INDEXES: Sequence[str] = (
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_research_sources_dedupe
        ON research_sources (campaign_id, repo_context, source_type, origin_url, digest)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_research_documents_dedupe
        ON research_documents (campaign_id, repo_context, source_id, title, digest)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_research_claims_dedupe
        ON research_claims (campaign_id, repo_context, document_id, topic, summary)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_crawl_queue_dedupe
        ON crawl_queue (campaign_id, url, topic)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_analysis_packs_campaign_repo
        ON analysis_packs (campaign_id, repo_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_repo_snapshots_campaign_repo
        ON repo_snapshots (campaign_id, repo_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_task_telemetry_campaign
        ON task_telemetry (campaign_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_source_chunks_campaign_document
        ON source_chunks (campaign_id, repo_context, document_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_topic_clusters_campaign
        ON topic_clusters (campaign_id, topic, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_phase_artifacts_campaign_phase
        ON phase_artifacts (campaign_id, phase_id, created_at)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_phase_artifacts_dedupe
        ON phase_artifacts (campaign_id, phase_id, artifact_id, path)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_convergence_campaign_loop
        ON convergence_checkpoints (campaign_id, loop_name, iteration, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_experiment_telemetry_campaign
        ON experiment_telemetry (campaign_id, experiment_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_usage_traces_campaign_repo
        ON usage_traces (campaign_id, repo_context, document_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_objects_campaign_kind_created
        ON memory_objects (campaign_id, memory_kind, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_objects_campaign_repo_kind
        ON memory_objects (campaign_id, repo_context, memory_kind)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_objects_campaign_session_created
        ON memory_objects (campaign_id, session_id, created_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_objects_canonical_hash
        ON memory_objects (canonical_hash)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_aliases_campaign_source
        ON memory_aliases (campaign_id, source_table, source_id)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_embeddings_identity
        ON memory_embeddings (memory_id, embedding_family, embedding_version)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_multivectors_identity
        ON memory_multivectors (memory_id, embedding_family)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_hd_identity
        ON memory_hd_bundles (memory_id, bundle_family, bundle_version)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_src
        ON memory_edges (src_memory_id, edge_type, recorded_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_edges_dst
        ON memory_edges (dst_memory_id, edge_type, recorded_at)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_latent_packages_memory_created
        ON latent_packages (memory_id, created_at)
        """,
    )

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA temp_store=MEMORY")
            for ddl in self._TABLES.values():
                cursor.execute(ddl)
            self._ensure_column(
                cursor, "research_sources", "repo_context", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                cursor, "research_documents", "repo_context", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                cursor, "research_claims", "repo_context", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                cursor,
                "research_claims",
                "complexity_score",
                "REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                cursor, "research_claims", "topic_hierarchy", "TEXT NOT NULL DEFAULT ''"
            )
            self._ensure_column(
                cursor,
                "research_claims",
                "applicability_score",
                "REAL NOT NULL DEFAULT 0.0",
            )
            self._ensure_column(
                cursor,
                "research_claims",
                "evidence_type",
                "TEXT NOT NULL DEFAULT 'implementation'",
            )
            self._ensure_column(
                cursor, "source_chunks", "repo_context", "TEXT NOT NULL DEFAULT ''"
            )
            cursor.execute("DROP INDEX IF EXISTS idx_research_sources_dedupe")
            cursor.execute("DROP INDEX IF EXISTS idx_research_documents_dedupe")
            cursor.execute("DROP INDEX IF EXISTS idx_research_claims_dedupe")
            cursor.execute("DROP INDEX IF EXISTS idx_source_chunks_campaign_document")
            for ddl in self._INDEXES:
                cursor.execute(ddl)
            self._conn.commit()

    @staticmethod
    def _ensure_column(
        cursor: sqlite3.Cursor,
        table: str,
        column: str,
        column_sql: str,
    ) -> None:
        existing = {
            row[1] for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")

    @staticmethod
    def _json(payload: Any) -> str:
        return json.dumps(payload or {}, default=str, sort_keys=True)

    def execute(self, query: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(query, params)
            self._conn.commit()
            return cursor

    def fetchall(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def fetchone(
        self, query: str, params: Sequence[Any] = ()
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return dict(row) if row is not None else None

    def insert_json_row(
        self,
        table: str,
        campaign_id: str,
        payload: Dict[str, Any],
        id_field: Optional[str] = None,
        id_value: Optional[str] = None,
        status: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        data = dict(extra or {})
        data["campaign_id"] = campaign_id
        if id_field and id_value is not None:
            data[id_field] = id_value
        if status is not None:
            data["status"] = status
        payload_json_field = "payload_json"
        if table in {"campaigns", "artifacts", "campaign_repos"}:
            raise ValueError(f"Use dedicated helper for table {table}")
        data[payload_json_field] = self._json(payload)
        data["created_at"] = now
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        self.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            tuple(data.values()),
        )

    def upsert_campaign(
        self,
        campaign_id: str,
        name: str,
        version: str,
        runtime_state: str,
        root_dir: str,
        objective: str = "",
        campaign_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        self.execute(
            """
            INSERT INTO campaigns (
                campaign_id, name, version, objective, runtime_state, campaign_path,
                root_dir, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id) DO UPDATE SET
                name=excluded.name,
                version=excluded.version,
                objective=excluded.objective,
                runtime_state=excluded.runtime_state,
                campaign_path=excluded.campaign_path,
                root_dir=excluded.root_dir,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                campaign_id,
                name,
                version,
                objective,
                runtime_state,
                campaign_path,
                root_dir,
                self._json(metadata),
                now,
                now,
            ),
        )

    def get_campaign(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        payload = self.fetchone(
            "SELECT * FROM campaigns WHERE campaign_id = ?",
            (campaign_id,),
        )
        if payload is None:
            return None
        payload.setdefault("current_state", payload.get("runtime_state"))
        return payload

    def initialize_campaign(
        self,
        campaign_id: str,
        *,
        name: str,
        objective: str = "",
        current_state: str = "INTAKE",
        status: str = "active",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        del status
        payload = dict(metadata or {})
        self.upsert_campaign(
            campaign_id=campaign_id,
            name=name,
            version=str(payload.get("campaign_version", "1.0")),
            runtime_state=current_state,
            root_dir=str(payload.get("root_dir", ".")),
            objective=objective,
            campaign_path=payload.get("campaign_path"),
            metadata=payload,
        )

    def transition_state(
        self,
        campaign_id: str,
        to_state: str,
        reason: str = "",
        payload: Optional[Dict[str, Any]] = None,
        cause: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reason = cause or reason
        payload = metadata if metadata is not None else payload
        current = self.get_campaign(campaign_id)
        from_state = current.get("runtime_state") if current else None
        self.insert_json_row(
            "campaign_state_transitions",
            campaign_id=campaign_id,
            payload=payload or {},
            extra={
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
            },
        )
        if current is not None:
            self.execute(
                """
                UPDATE campaigns
                SET runtime_state = ?, updated_at = ?
                WHERE campaign_id = ?
                """,
                (to_state, time.time(), campaign_id),
            )
        return {
            "campaign_id": campaign_id,
            "from_state": from_state,
            "to_state": to_state,
            "cause": reason,
            "metadata": payload or {},
        }

    def record_transition(
        self,
        campaign_id: str,
        from_state: Optional[str],
        to_state: str,
        reason: str,
        *,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        del from_state
        self.transition_state(campaign_id, to_state, reason=reason, payload=payload)

    def upsert_repo(
        self, repo_id: str, campaign_id: str, payload: Dict[str, Any]
    ) -> None:
        self.execute(
            """
            INSERT INTO campaign_repos (
                repo_id, campaign_id, name, origin, revision, local_path, role,
                write_policy, topic_tags_json, trust_level, ingestion_status,
                metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_id) DO UPDATE SET
                name=excluded.name,
                origin=excluded.origin,
                revision=excluded.revision,
                local_path=excluded.local_path,
                role=excluded.role,
                write_policy=excluded.write_policy,
                topic_tags_json=excluded.topic_tags_json,
                trust_level=excluded.trust_level,
                ingestion_status=excluded.ingestion_status,
                metadata_json=excluded.metadata_json
            """,
            (
                repo_id,
                campaign_id,
                payload["name"],
                payload.get("origin"),
                payload.get("revision"),
                payload["local_path"],
                payload["role"],
                payload["write_policy"],
                self._json(payload.get("topic_tags") or []),
                payload.get("trust_level", "local"),
                payload.get("ingestion_status", "registered"),
                self._json(payload.get("metadata")),
                time.time(),
            ),
        )

    def list_repos(self, campaign_id: str) -> List[Dict[str, Any]]:
        return self.fetchall(
            "SELECT * FROM campaign_repos WHERE campaign_id = ? ORDER BY created_at ASC",
            (campaign_id,),
        )

    def register_repo(self, payload: Dict[str, Any]) -> None:
        self.upsert_repo(payload["repo_id"], payload["campaign_id"], payload)

    def upsert_artifact(
        self,
        artifact_id: str | Dict[str, Any],
        campaign_id: Optional[str] = None,
        family: Optional[str] = None,
        name: Optional[str] = None,
        version: int = 1,
        canonical_path: str = "",
        rendered_path: Optional[str] = None,
        approval_state: str = "pending",
        blocking: bool = False,
        status: str = "draft",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(artifact_id, dict):
            payload = artifact_id
            return self.upsert_artifact(
                artifact_id=payload["artifact_id"],
                campaign_id=payload["campaign_id"],
                family=payload["family"],
                name=payload["name"],
                version=int(payload.get("version", 1)),
                canonical_path=payload.get("canonical_path", ""),
                rendered_path=payload.get("rendered_path"),
                approval_state=payload.get("approval_state", "pending"),
                blocking=bool(payload.get("blocking", False)),
                status=payload.get("status", "draft"),
                metadata=payload.get("metadata") or payload.get("provenance") or {},
            )
        now = time.time()
        self.execute(
            """
            INSERT INTO artifacts (
                artifact_id, campaign_id, family, name, version, canonical_path,
                rendered_path, approval_state, blocking, status, metadata_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(artifact_id) DO UPDATE SET
                version=excluded.version,
                canonical_path=excluded.canonical_path,
                rendered_path=excluded.rendered_path,
                approval_state=excluded.approval_state,
                blocking=excluded.blocking,
                status=excluded.status,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                artifact_id,
                campaign_id,
                family,
                name,
                version,
                canonical_path,
                rendered_path,
                approval_state,
                int(blocking),
                status,
                self._json(metadata),
                now,
                now,
            ),
        )

    def list_artifacts(
        self, campaign_id: str, family: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if family:
            return self.fetchall(
                """
                SELECT * FROM artifacts
                WHERE campaign_id = ? AND family = ?
                ORDER BY family, name, version
                """,
                (campaign_id, family),
            )
        return self.fetchall(
            "SELECT * FROM artifacts WHERE campaign_id = ? ORDER BY family, name, version",
            (campaign_id,),
        )

    def record_phase_artifact(
        self,
        campaign_id: str,
        phase_id: str,
        artifact_id: str,
        artifact_type: str,
        path: str,
        *,
        status: str = "published",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        self.execute(
            """
            INSERT INTO phase_artifacts (
                campaign_id, phase_id, artifact_id, artifact_type, path, status,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id, phase_id, artifact_id, path) DO UPDATE SET
                artifact_type = excluded.artifact_type,
                status = excluded.status,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                campaign_id,
                phase_id,
                artifact_id,
                artifact_type,
                path,
                status,
                self._json(metadata),
                now,
                now,
            ),
        )

    def list_phase_artifacts(
        self,
        campaign_id: str,
        phase_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM phase_artifacts
            WHERE campaign_id = ?
        """
        params: list[Any] = [campaign_id]
        if phase_id is None:
            query += " ORDER BY phase_id, created_at ASC"
        else:
            query += " AND phase_id = ? ORDER BY created_at ASC"
            params.append(phase_id)
        rows = self.fetchall(query, tuple(params))
        return [
            {
                **row,
                "metadata": json.loads(row.get("metadata_json") or "{}"),
            }
            for row in rows
        ]

    def record_convergence_checkpoint(
        self,
        campaign_id: str,
        loop_name: str,
        iteration: int,
        metrics: Dict[str, Any],
        *,
        converged: bool,
    ) -> None:
        self.execute(
            """
            INSERT INTO convergence_checkpoints (
                campaign_id, loop_name, iteration, metrics_json, converged, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                loop_name,
                iteration,
                self._json(metrics),
                int(converged),
                time.time(),
            ),
        )

    def list_convergence_checkpoints(
        self,
        campaign_id: str,
        loop_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM convergence_checkpoints
            WHERE campaign_id = ?
        """
        params: list[Any] = [campaign_id]
        if loop_name is not None:
            query += " AND loop_name = ?"
            params.append(loop_name)
        query += " ORDER BY created_at ASC"
        rows = self.fetchall(query, tuple(params))
        return [
            {
                **row,
                "metrics": json.loads(row.get("metrics_json") or "{}"),
            }
            for row in rows
        ]

    def record_experiment_telemetry(
        self,
        campaign_id: str,
        experiment_id: str,
        metric_name: str,
        metric_value: float,
        *,
        unit: str = "",
    ) -> None:
        self.execute(
            """
            INSERT INTO experiment_telemetry (
                campaign_id, experiment_id, metric_name, metric_value, unit, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                experiment_id,
                metric_name,
                float(metric_value),
                unit,
                time.time(),
            ),
        )

    def record_usage_trace(
        self,
        campaign_id: str,
        repo_context: str,
        document_id: str,
        symbol_name: str,
        payload: Dict[str, Any],
    ) -> None:
        self.insert_json_row(
            "usage_traces",
            campaign_id=campaign_id,
            payload=payload,
            extra={
                "repo_context": repo_context,
                "document_id": document_id,
                "symbol_name": symbol_name,
            },
        )

    def list_usage_traces(
        self,
        campaign_id: str,
        repo_context: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
            SELECT repo_context, document_id, symbol_name, payload_json, created_at
            FROM usage_traces
            WHERE campaign_id = ?
        """
        params: list[Any] = [campaign_id]
        if repo_context is not None:
            query += " AND repo_context = ?"
            params.append(repo_context)
        query += " ORDER BY created_at ASC"
        rows = self.fetchall(query, tuple(params))
        return [
            {
                **row,
                "payload": json.loads(row["payload_json"] or "{}"),
            }
            for row in rows
        ]

    def record_artifact_approval(
        self,
        artifact_id: str,
        *,
        state: str,
        approved_by: str,
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        del metadata
        self.execute(
            """
            INSERT INTO artifact_approvals (
                artifact_id, approved_by, decision, notes, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (artifact_id, approved_by, state, notes, time.time()),
        )
        self.execute(
            """
            UPDATE artifacts
            SET approval_state = ?, updated_at = ?
            WHERE artifact_id = ?
            """,
            (state, time.time(), artifact_id),
        )

    def record_question(self, payload: Dict[str, Any]) -> None:
        question_id = payload["question_id"]
        now = time.time()
        self.execute(
            """
            DELETE FROM architecture_questions
            WHERE campaign_id = ? AND question_id = ?
            """,
            (payload["campaign_id"], question_id),
        )
        self.insert_json_row(
            "architecture_questions",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="question_id",
            id_value=question_id,
            status=payload.get("current_status", "open"),
            extra={"updated_at": now},
        )

    def list_questions(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT question_id, payload_json, status
            FROM architecture_questions
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        output: List[Dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["current_status"] = row["status"]
            output.append(payload)
        return output

    def record_feature(self, payload: Dict[str, Any]) -> None:
        feature_id = payload["feature_id"]
        self.insert_json_row(
            "feature_catalog",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="feature_id",
            id_value=feature_id,
        )
        self.insert_json_row(
            "feature_selections",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="feature_id",
            id_value=feature_id,
            extra={
                "selection_state": payload.get(
                    "selection_state", payload.get("default_state", "defer")
                )
            },
        )

    def list_features(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT feature_id, payload_json
            FROM feature_catalog
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def record_roadmap_item(self, payload: Dict[str, Any]) -> None:
        self.insert_json_row(
            "roadmap_items",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="item_id",
            id_value=payload["item_id"],
            extra={
                "phase_id": payload["phase_id"],
                "title": payload["title"],
                "item_type": payload["type"],
            },
        )
        for dependency in payload.get("depends_on", []):
            self.execute(
                """
                INSERT INTO roadmap_dependencies (
                    campaign_id, item_id, depends_on_item_id, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (payload["campaign_id"], payload["item_id"], dependency, time.time()),
            )

    def list_roadmap_items(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT item_id, payload_json
            FROM roadmap_items
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def record_task_packet(self, payload: Dict[str, Any]) -> None:
        self.insert_json_row(
            "task_packets",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="task_packet_id",
            id_value=payload["task_packet_id"],
            extra={
                "phase_id": payload.get("phase_id"),
                "role": payload.get(
                    "specialist_role", payload.get("role", "specialist")
                ),
            },
        )

    def list_task_packets(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT task_packet_id, payload_json
            FROM task_packets
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def record_task_run(
        self,
        task_packet_id: str,
        *,
        status: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        campaign_id = ""
        if isinstance(result, dict) and result.get("campaign_id"):
            campaign_id = str(result["campaign_id"])
        else:
            row = self.fetchone(
                """
                SELECT campaign_id
                FROM task_packets
                WHERE task_packet_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (task_packet_id,),
            )
            if row is not None:
                campaign_id = str(row.get("campaign_id") or "")
        self.insert_json_row(
            "task_runs",
            campaign_id=campaign_id,
            payload=result or {},
            id_field="task_packet_id",
            id_value=task_packet_id,
            status=status,
        )

    def record_telemetry(
        self,
        campaign_id: str,
        *,
        telemetry_kind: str,
        payload: Dict[str, Any],
        task_packet_id: Optional[str] = None,
    ) -> None:
        self.execute(
            """
            INSERT INTO task_telemetry (
                campaign_id, task_packet_id, telemetry_json, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                campaign_id,
                task_packet_id or "campaign",
                self._json({**payload, "telemetry_kind": telemetry_kind}),
                time.time(),
            ),
        )

    def list_telemetry(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT telemetry_json
            FROM task_telemetry
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["telemetry_json"]) for row in rows]

    def record_audit_run(self, payload: Dict[str, Any]) -> None:
        self.insert_json_row(
            "audit_runs",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="audit_id",
            id_value=payload.get("audit_run_id") or payload.get("audit_id", "audit"),
        )

    def list_audit_runs(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT payload_json
            FROM audit_runs
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def record_audit_findings(self, findings: Iterable[Dict[str, Any]]) -> None:
        for finding in findings:
            self.insert_json_row(
                "audit_findings",
                campaign_id=finding["campaign_id"],
                payload=finding,
                id_field="finding_id",
                id_value=finding["finding_id"],
                extra={
                    "audit_id": finding.get("audit_run_id", "audit"),
                    "severity": finding["severity"],
                },
            )

    def list_audit_findings(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT payload_json
            FROM audit_findings
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def record_completion_check(self, payload: Dict[str, Any]) -> None:
        self.insert_json_row(
            "completion_checks",
            campaign_id=payload["campaign_id"],
            payload=payload,
            id_field="check_id",
            id_value=payload["check_id"],
            status="passed" if payload.get("passed") else "failed",
        )

    def list_completion_checks(self, campaign_id: str) -> List[Dict[str, Any]]:
        rows = self.fetchall(
            """
            SELECT payload_json
            FROM completion_checks
            WHERE campaign_id = ?
            ORDER BY created_at ASC
            """,
            (campaign_id,),
        )
        return [json.loads(row["payload_json"]) for row in rows]

    def insert_record(
        self,
        table: str,
        campaign_id: str,
        payload: Dict[str, Any],
        *,
        record_key: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> None:
        del topic
        if table == "task_packets":
            data = dict(payload)
            data.setdefault("campaign_id", campaign_id)
            self.record_task_packet(data)
            return
        if table in {"tooling_tasks", "whitepapers"}:
            key = record_key or str(
                payload.get("tooling_task_id") or payload.get("whitepaper_id")
            )
            id_field = (
                "tooling_task_id" if table == "tooling_tasks" else "whitepaper_id"
            )
            self.insert_json_row(
                table,
                campaign_id=campaign_id,
                payload=payload,
                id_field=id_field,
                id_value=key,
            )
            return
        raise ValueError(f"Unsupported insert_record table: {table}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()
