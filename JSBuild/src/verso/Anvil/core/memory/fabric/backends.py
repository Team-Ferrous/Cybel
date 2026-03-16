"""Backend profile helpers for ALMF local and enterprise modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib.util
import os
from typing import Any, Dict


@dataclass(slots=True)
class MemoryBackendProfile:
    """Resolved storage profile for ALMF state."""

    requested_backend: str
    effective_backend: str
    canonical_dsn: str
    storage_root: str
    blob_root: str
    index_root: str
    snapshot_root: str
    tenant_key: str = "default"
    tenant_isolated: bool = True
    driver_name: str = "sqlite3"
    fallback_reason: str = ""
    object_store_uri: str = ""
    distributed_index_uri: str = ""
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return bool(self.fallback_reason)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def resolve_memory_backend_profile(
    *,
    db_path: str,
    storage_root: str | None = None,
    requested_backend: str = "auto",
    postgres_dsn: str | None = None,
    tenant_key: str = "default",
    options: Dict[str, Any] | None = None,
) -> MemoryBackendProfile:
    """Resolve the active ALMF storage profile.

    The enterprise profile shares the same contract, but falls back to SQLite when
    no Postgres driver is available or no DSN has been configured.
    """

    requested = str(requested_backend or "auto").strip().lower()
    db_abs = os.path.abspath(db_path)
    base_root = os.path.abspath(
        storage_root
        or os.path.join(os.path.dirname(db_abs), "memory_fabric")
    )
    blob_root = os.path.join(base_root, "blobs")
    index_root = os.path.join(base_root, "indexes")
    snapshot_root = os.path.join(base_root, "snapshots")
    for path in (base_root, blob_root, index_root, snapshot_root):
        os.makedirs(path, exist_ok=True)

    psycopg_available = bool(importlib.util.find_spec("psycopg")) or bool(
        importlib.util.find_spec("psycopg2")
    )
    resolved_dsn = str(
        postgres_dsn
        or os.environ.get("ALMF_POSTGRES_DSN")
        or ""
    ).strip()

    if requested in {"postgres", "enterprise"} or (
        requested == "auto" and resolved_dsn
    ):
        if psycopg_available and resolved_dsn:
            return MemoryBackendProfile(
                requested_backend=requested,
                effective_backend="postgres",
                canonical_dsn=resolved_dsn,
                storage_root=base_root,
                blob_root=blob_root,
                index_root=index_root,
                snapshot_root=snapshot_root,
                tenant_key=tenant_key,
                tenant_isolated=True,
                driver_name="psycopg"
                if importlib.util.find_spec("psycopg")
                else "psycopg2",
                object_store_uri=os.environ.get("ALMF_OBJECT_STORE_URI", ""),
                distributed_index_uri=os.environ.get("ALMF_INDEX_URI", ""),
                options=dict(options or {}),
            )
        fallback_reason = (
            "postgres requested but psycopg driver is unavailable"
            if not psycopg_available
            else "postgres requested but no ALMF_POSTGRES_DSN was configured"
        )
        return MemoryBackendProfile(
            requested_backend=requested,
            effective_backend="sqlite",
            canonical_dsn=db_abs,
            storage_root=base_root,
            blob_root=blob_root,
            index_root=index_root,
            snapshot_root=snapshot_root,
            tenant_key=tenant_key,
            tenant_isolated=True,
            driver_name="sqlite3",
            fallback_reason=fallback_reason,
            options=dict(options or {}),
        )

    return MemoryBackendProfile(
        requested_backend=requested,
        effective_backend="sqlite",
        canonical_dsn=db_abs,
        storage_root=base_root,
        blob_root=blob_root,
        index_root=index_root,
        snapshot_root=snapshot_root,
        tenant_key=tenant_key,
        tenant_isolated=True,
        driver_name="sqlite3",
        options=dict(options or {}),
    )
