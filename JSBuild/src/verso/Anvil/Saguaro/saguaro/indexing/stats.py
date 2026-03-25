"""Persistent index statistics used by retrieval scoring and schema management."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any

from saguaro.storage.atomic_fs import atomic_write_json

INDEX_SCHEMA_VERSION = 3
INDEX_STATS_FILENAME = "index_stats.json"


def stats_path(saguaro_dir: str) -> str:
    """Return the path to the persisted index stats artifact."""
    return os.path.join(saguaro_dir, INDEX_STATS_FILENAME)


def load_index_stats(saguaro_dir: str) -> dict[str, Any]:
    """Load persisted index stats if they exist."""
    path = stats_path(saguaro_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def compute_term_statistics(metadata_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute corpus statistics from indexed metadata rows."""
    df: Counter[str] = Counter()
    document_count = 0
    total_terms = 0
    for row in metadata_rows:
        terms = set()
        for key in (
            "terms",
            "symbol_terms",
            "path_terms",
            "doc_terms",
            "matched_terms",
        ):
            for term in row.get(key, []) or []:
                if isinstance(term, str) and term:
                    terms.add(term.lower())
        if not terms:
            continue
        document_count += 1
        total_terms += len(terms)
        df.update(terms)

    return {
        "document_count": document_count,
        "avg_terms_per_doc": round(total_terms / document_count, 3)
        if document_count
        else 0.0,
        "term_doc_freq": dict(df.most_common(20000)),
    }


def persist_index_stats(
    saguaro_dir: str,
    *,
    metadata_rows: list[dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persist index statistics and return the merged payload."""
    stats = compute_term_statistics(metadata_rows)
    merged = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "generated_at": time.time(),
        **payload,
        **stats,
    }
    os.makedirs(saguaro_dir, exist_ok=True)
    atomic_write_json(
        stats_path(saguaro_dir),
        merged,
        indent=2,
        sort_keys=True,
    )
    return merged


def idf_for_term(stats: dict[str, Any], term: str) -> float:
    """Return a simple IDF-style weighting for a term."""
    df = int((stats.get("term_doc_freq") or {}).get(str(term).lower(), 0))
    doc_count = max(1, int(stats.get("document_count", 0) or 0))
    return 1.0 + (doc_count / (1.0 + df))
