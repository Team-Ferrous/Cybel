"""External repository acquisition and snapshot recording."""

from __future__ import annotations

from typing import Any, Dict

from core.campaign.repo_cache import RepoCache
from core.campaign.repo_registry import RepoRegistry
from saguaro.services.comparative import ComparativeAnalysisService


class RepoAcquisitionService:
    """Clones or snapshots repos into immutable campaign analysis storage."""

    def __init__(self, repo_cache: RepoCache, repo_registry: RepoRegistry):
        self.repo_cache = repo_cache
        self.repo_registry = repo_registry

    @staticmethod
    def _native_pack(
        *,
        repo_id: str,
        repo_path: str,
        name: str,
        role: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        comparative = ComparativeAnalysisService(repo_path)
        created = comparative.create_session(
            path=repo_path,
            corpus_id=repo_id,
            alias=name,
            quarantine=role != "target",
            trust_level="high" if role == "target" else "medium",
            build_profile="auto",
            rebuild=False,
        )
        session = dict(created.get("session") or {})
        shown = comparative.corpus(action="show", corpus_id=str(session.get("corpus_id") or repo_id))
        pack = dict(shown.get("analysis_pack") or {})
        dossier = dict(pack.get("repo_dossier") or {})
        return session, pack, dossier

    def acquire_local(
        self, name: str, local_path: str, role: str = "analysis_local"
    ) -> Dict[str, Any]:
        snapshot = self.repo_cache.snapshot_local_repo(local_path)
        record = self.repo_registry.register_repo(
            name=name,
            local_path=snapshot["source_path"],
            role=role,
            write_policy="immutable",
            origin=snapshot["origin"],
            revision=snapshot["revision"],
            metadata={"cache_metadata_path": snapshot["metadata_path"]},
        )
        session, pack, dossier = self._native_pack(
            repo_id=record.repo_id,
            repo_path=record.local_path,
            name=name,
            role=record.role,
        )
        self.repo_registry.record_ingestion_report(
            record.repo_id,
            {
                "schema_version": "repo_ingestion_report.v3",
                "analysis_backend": "native_comparative",
                "corpus_session": session,
                "analysis_pack": pack,
                "repo_dossier": dossier,
            },
        )
        return {
            "repo": record,
            "snapshot": snapshot,
            "corpus_session": session,
            "analysis_pack": pack,
            "repo_dossier": dossier,
        }

    def acquire_remote(
        self, name: str, origin_url: str, revision: str = "HEAD"
    ) -> Dict[str, Any]:
        snapshot = self.repo_cache.clone_remote_repo(origin_url, revision=revision)
        record = self.repo_registry.register_repo(
            name=name,
            local_path=snapshot["source_path"],
            role="analysis_external",
            write_policy="immutable",
            origin=origin_url,
            revision=revision,
            metadata={"cache_metadata_path": snapshot["metadata_path"]},
        )
        session, pack, dossier = self._native_pack(
            repo_id=record.repo_id,
            repo_path=record.local_path,
            name=name,
            role=record.role,
        )
        self.repo_registry.record_ingestion_report(
            record.repo_id,
            {
                "schema_version": "repo_ingestion_report.v3",
                "analysis_backend": "native_comparative",
                "corpus_session": session,
                "analysis_pack": pack,
                "repo_dossier": dossier,
            },
        )
        return {
            "repo": record,
            "snapshot": snapshot,
            "corpus_session": session,
            "analysis_pack": pack,
            "repo_dossier": dossier,
        }
